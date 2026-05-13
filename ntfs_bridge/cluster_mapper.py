"""Cluster mapper for NTFS-ext4 bridge.

Scans the MFT of an NTFS image to build a cluster-to-ext4-file mapping.
Handles read/write routing: metadata from image, data from ext4.
Tracks MFT writes to detect new files, renames, deletions, and reallocations.

Handles both non-resident files (data in clusters → mapped to ext4) and
resident files (data in MFT record → injected from ext4 on reads).

Supports lazy allocation: large files can start as sparse (no clusters),
get allocated on first read, and deallocated after a timeout.
"""
import bisect
import mmap
import os
import queue
import shutil
import struct
import threading
import time
import traceback
from typing import Dict, Iterable, List, Tuple, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .lazy_allocator import LazyAllocator
    from .virtual_files import VirtualFileManager

MFT_RECORD_SIZE = 1024
CLUSTER_SIZE = 4096  # Standard NTFS cluster size

# All files use run-based mapping (_direct_run_map) to avoid per-cluster dict
# entries that consume hundreds of MB for large file collections.
# Set to 0 so every file — small or large — uses the O(log n) run lookup.
RUN_MAP_THRESHOLD = 0

# Size of the in-RAM hot cache for the NTFS image metadata region.
# All NTFS metadata (MFT, bitmap, boot sector) lives within the first few MB.
# Keeping 64MB in RAM prevents page fault stalls when the underlying disk is
# under heavy I/O load from other processes.
_HOT_CACHE_SIZE = 64 * 1024 * 1024  # 64 MB


class _HotImageCache:
    """RAM-backed cache over a mmap for the NTFS image metadata region.

    The first HOT_SIZE bytes are kept in a bytearray so reads and writes
    never block on disk I/O. Beyond HOT_SIZE the underlying mmap is used
    directly (these are data-cluster regions that are rarely accessed).

    flush() writes the hot bytearray back to the mmap and syncs to disk.
    """

    def __init__(self, mm: mmap.mmap, hot_size: int):
        self._mm = mm
        self._hot_size = min(hot_size, len(mm))
        log(f"Loading {self._hot_size // (1024 * 1024)}MB image metadata into RAM...")
        self._hot = bytearray(mm[:self._hot_size])
        log("Image metadata cached (reads/writes are now RAM-speed)")

    def __len__(self) -> int:
        return len(self._mm)

    def __getitem__(self, s: slice) -> bytes:
        start = s.start if s.start is not None else 0
        stop = s.stop if s.stop is not None else len(self._mm)
        if stop <= self._hot_size:
            return bytes(self._hot[start:stop])
        if start >= self._hot_size:
            return self._mm[start:stop]
        # Spans boundary
        return bytes(self._hot[start:self._hot_size]) + self._mm[self._hot_size:stop]

    def __setitem__(self, s: slice, value: bytes):
        start = s.start if s.start is not None else 0
        stop = s.stop if s.stop is not None else start + len(value)
        if stop <= self._hot_size:
            self._hot[start:stop] = value
        elif start >= self._hot_size:
            self._mm[start:stop] = value
        else:
            # Spans boundary
            hot_end = self._hot_size
            self._hot[start:hot_end] = value[:hot_end - start]
            self._mm[hot_end:stop] = value[hot_end - start:]

    def reload(self):
        """Reload hot cache from mmap (picks up external writes, e.g. ntfs-3g populate)."""
        self._hot = bytearray(self._mm[:self._hot_size])

    def flush(self):
        """Flush hot cache to mmap and sync to disk."""
        self._mm[:self._hot_size] = bytes(self._hot)
        self._mm.flush(0, self._hot_size)

    def close(self):
        self._mm.close()


def log(msg):
    print(f"[ClusterMapper] {msg}", flush=True)


def _set_bitmap_bits(bitmap: bytearray, start: int, count: int, value: bool):
    """Set a range of bitmap bits efficiently using byte-level operations."""
    if count <= 0 or not bitmap:
        return
    end = start + count
    first_byte = start // 8
    first_bit = start % 8
    last_byte = min((end - 1) // 8, len(bitmap) - 1)
    last_bit = (end - 1) % 8

    if first_byte >= len(bitmap):
        return

    if first_byte == last_byte:
        mask = ((1 << count) - 1) << first_bit
        if value:
            bitmap[first_byte] |= mask & 0xFF
        else:
            bitmap[first_byte] &= (~mask) & 0xFF
        return

    # First partial byte
    mid_start = first_byte
    if first_bit != 0:
        mask = (0xFF << first_bit) & 0xFF
        if value:
            bitmap[first_byte] |= mask
        else:
            bitmap[first_byte] &= (~mask) & 0xFF
        mid_start = first_byte + 1

    # Middle full bytes
    if last_byte > mid_start:
        mid_count = last_byte - mid_start
        if value:
            bitmap[mid_start:last_byte] = b'\xff' * mid_count
        else:
            bitmap[mid_start:last_byte] = b'\x00' * mid_count

    # Last byte
    mask = (1 << (last_bit + 1)) - 1
    if value:
        bitmap[last_byte] |= mask & 0xFF
    else:
        bitmap[last_byte] &= (~mask) & 0xFF


class ClusterMapper:
    """Maps NTFS clusters to ext4 source files via MFT scanning.

    Reads from the image file for metadata and from ext4 files for data.
    Writes go to the image for metadata and to ext4 for data clusters.

    Resident files (small files stored directly in MFT records) are handled
    by injecting ext4 content into MFT reads, so ext4 is always the source
    of truth for file content.
    """

    def __init__(self, image_path: str, source_dir: str,
                 overflow_dir: Optional[str] = None,
                 protected_roots: Optional[Iterable[str]] = None):
        self.image_path = os.path.abspath(image_path)
        self.source_dir = os.path.abspath(source_dir)

        # Top-level subdirectories of source_dir whose contents are presented
        # read-only at the bridge level. Writes that target an MFT record
        # belonging to a file/dir under one of these roots, or a data cluster
        # belonging to such a file, are silently dropped. Writes elsewhere
        # (root-level files, subdirectories created by Windows like System
        # Volume Information, $RECYCLE.BIN, or any working dir a backup or
        # indexing tool installs at the volume root) still propagate normally.
        # Set by --protected-roots; empty set = no protection. Stored
        # lowercased and matched case-insensitively, because the synthesized
        # NTFS view is case-insensitive even when the ext4 source preserves
        # case (e.g. Documents vs documents).
        self._protected_top_dirs: Set[str] = set()
        if protected_roots:
            self._protected_top_dirs = {p.lower() for p in protected_roots if p}

        # Overflow directory for root-level items not in the source tree
        # (e.g. System Volume Information, Windows SID folders)
        if overflow_dir:
            self.overflow_dir = os.path.abspath(overflow_dir)
        else:
            self.overflow_dir = self.source_dir
        if self.overflow_dir != self.source_dir:
            os.makedirs(self.overflow_dir, exist_ok=True)
            log(f"Overflow directory: {self.overflow_dir}")

        # Known top-level entries in the source directory (for root path resolution)
        self.known_root_entries: Set[str] = set()
        try:
            self.known_root_entries = set(os.listdir(self.source_dir))
        except OSError:
            pass

        # Memory-map the image file, then wrap with a hot RAM cache.
        # The hot cache keeps the first 64MB in a bytearray so NTFS metadata
        # reads/writes are RAM-speed and never stall on disk I/O.
        self._image_file = open(image_path, 'r+b')
        self.image = _HotImageCache(
            mmap.mmap(self._image_file.fileno(), 0),
            _HOT_CACHE_SIZE,
        )

        # Parse boot sector
        boot = self.image[0:512]
        self.bytes_per_sector = struct.unpack('<H', boot[0x0B:0x0D])[0]
        self.sectors_per_cluster = boot[0x0D]
        self.cluster_size = self.bytes_per_sector * self.sectors_per_cluster
        self.mft_cluster = struct.unpack('<Q', boot[0x30:0x38])[0]
        self.mft_offset = self.mft_cluster * self.cluster_size
        self._mft_runs, self._mft_total_records = self._get_mft_runs()

        # Cluster -> (source_file_path, offset_in_file)
        self.cluster_map: Dict[int, Tuple[str, int]] = {}

        # MFT tracking
        self.mft_record_to_source: Dict[int, str] = {}
        self.source_to_clusters: Dict[str, Set[int]] = {}
        self.mft_record_to_dir: Dict[int, str] = {}
        self.path_to_mft_record: Dict[str, int] = {}
        # Sequence number at the time each directory record was first tracked.
        # NTFS increments this when a record is freed and reused for a different
        # entity.  If it changes, _check_directory_rename is looking at a
        # recycled record – not a rename – and must not move anything.
        self._dir_mft_seq: Dict[int, int] = {}
        self.dir_children: Dict[int, Set[int]] = {}
        self.removed_mft_records: Set[int] = set()

        # Resident file tracking: record_num -> {source_path, val_len_abs, data_abs, avail}
        # These are files small enough that NTFS stores data directly in the MFT record
        self.resident_file_data: Dict[int, dict] = {}

        # Track which clusters are INDX blocks (direct bytes data)
        self.dir_indx_clusters: Dict[int, List[int]] = {}

        # Thread safety
        self.lock = threading.RLock()

        # Background MFT sync queue: write() puts (offset, data) here so that
        # the NBD reply goes out immediately; ext4 operations run asynchronously.
        self._mft_queue: queue.Queue = queue.Queue()
        _t = threading.Thread(target=self._mft_worker, daemon=True,
                              name="MFTSyncWorker")
        _t.start()

        # Loop prevention sets (shared with SyncDaemon)
        # Individual set operations (add, discard, `in`) are thread-safe under
        # CPython's GIL.  A lock is provided for any compound operations or
        # future iteration that may need atomicity.
        self._sync_lock = threading.Lock()
        self.ext4_sync_in_progress: Set[str] = set()
        self.ntfs_sync_in_progress: Set[str] = set()

        # Time-based loop prevention: records when NTFS→ext4 sync last wrote
        # each file.  The instant set (ntfs_sync_in_progress) is cleared as
        # soon as the write finishes, but the FileWatcher fires asynchronously
        # later.  SyncDaemon checks these timestamps to suppress cascade events
        # for a grace period after the sync.
        self.ntfs_sync_timestamps: Dict[str, float] = {}

        # Lazy allocator (set by bridge after construction)
        self.lazy_allocator: Optional['LazyAllocator'] = None

        # Track sparse files: rel_path -> (source_path, file_size, mft_record)
        # These are large files with no data runs (not yet allocated)
        self.sparse_files: Dict[str, Tuple[str, int, int]] = {}

        # Map allocated clusters of sparse files to rel_path
        # Used to trigger allocation when these clusters are read
        self.sparse_file_clusters: Dict[int, str] = {}

        # Pending allocation: set of rel_paths currently being allocated
        self._allocating: Set[str] = set()

        # Run-based cluster map for large files (avoids per-cluster dict entries).
        # Sorted list of (start_cluster, end_cluster_exclusive, source_path, base_file_offset).
        # Used instead of cluster_map for files with >RUN_MAP_THRESHOLD clusters.
        self._direct_run_map: List[Tuple[int, int, str, int]] = []

        # Bitmap location (found during MFT scan)
        self.bitmap_clusters: List[Tuple[int, int]] = []  # (start_cluster, count)
        self.total_clusters = len(self.image) // self.cluster_size

        # Bitmap cached in RAM to avoid re-reading 500MB+ on every allocation.
        # Loaded once in _load_bitmap_cache() after MFT scan.
        self._bitmap_cache: Optional[bytearray] = None

        # Sorted list of (start_cluster, count) free run intervals.
        # Built once from the bitmap in _build_free_run_index(); maintained
        # incrementally by _free_index_remove() / _free_index_add().
        # Turns _find_free_cluster_runs() from O(4B) to O(log n).
        self._free_run_index: List[Tuple[int, int]] = []

        # Allocation watermark: highest cluster known to be used + 1.
        # New allocations start here to avoid fragmented earlier regions.
        # Set properly after _scan_mft via _map_clusters calls.
        self._alloc_watermark = max(16, self.mft_cluster + 100)

        # Set of MFT record numbers for directly-allocated files.
        # Used by _mft_write_to_image() to block external overwrites of the
        # data runs we wrote via allocate_file_direct(). Without this, ntfs-3g
        # or Windows journal replay can write stale sparse data runs back over
        # our allocations, making the files unreadable.
        self._direct_allocated_records: Set[int] = set()

        # Protected INDEX_ALLOC / INDEX_BITMAP: record_num ->
        #   (ia_offset_in_record, data_size, ib_offset_in_record, ib_val_offset, bitmap_bytes)
        # After _fix_index_alloc_data_sizes() extends a directory's data_size and
        # sets INDEX_BITMAP bits, Windows may write the old (smaller) values back
        # during journal replay or access-time updates.  _mft_write_to_image()
        # re-patches these fields after every write to a protected record so the
        # fix is never reverted.
        self._protected_ia_sizes: Dict[int, Tuple[int, int, int, int, bytes]] = {}

        # $MFTMirr sync: byte offset of the mirror cluster in the image, and
        # how many MFT records it stores. Populated by _load_mft_mirror_info().
        # _mft_write_to_image() keeps this in sync with the primary MFT so that
        # ntfs-3g and Windows never see a $MFTMirr/$MFT mismatch.
        self._mft_mirror_offset: int = -1   # -1 = not found / disabled
        self._mft_mirror_record_count: int = 0

        # Virtual file manager for live ext4→NTFS sync (set by bridge)
        self.virtual_file_manager: Optional['VirtualFileManager'] = None

        # Directory virtualization: synthesized directory data
        # dir_record_num -> {
        #   'entries': list of (filename, entry_bytes),
        #   'index_root': synthesized INDEX_ROOT attribute bytes,
        #   'indx_blocks': list of synthesized INDX block bytes,
        #   'virtual_indx_clusters': list of virtual cluster numbers for INDX
        # }
        self.virtualized_dirs: Dict[int, dict] = {}

        # Map virtual INDX cluster -> (dir_record_num, block_index)
        self.virtual_indx_map: Dict[int, Tuple[int, int]] = {}

        # Map real INDX cluster -> (dir_record_num, block_index, synthesized_data)
        # Used to intercept reads to original INDX clusters and return virtualized content
        self.virtualized_indx_clusters: Dict[int, Tuple[int, int, bytes]] = {}

        # Next available virtual cluster number for INDX blocks
        # Must be within valid cluster range (total_clusters - 1) to be readable
        # Start high within the valid range to avoid conflicts with real data
        total_clusters = len(self.image) // self.cluster_size
        # Reserve last 1000 clusters for virtual INDX (or 10% of volume, whichever is smaller)
        virtual_reserve = min(1000, total_clusters // 10)
        self.next_virtual_indx_cluster = total_clusters - virtual_reserve
        log(f"Virtual INDX cluster range: {self.next_virtual_indx_cluster}-{total_clusters-1}")

        # Scan MFT
        self._scan_mft()

        # Find $Bitmap location
        self._find_bitmap_location()

        # Cache bitmap in RAM and build free-run index (one O(n) scan at startup)
        self._load_bitmap_cache()
        self._build_free_run_index()

        # Protect NTFS system file clusters from being handed out to user files.
        # ntfs-3g may place $MFTMirr and other metadata at high LCNs in sparse
        # space without marking those clusters used in $Bitmap; this prevents
        # allocate_file_direct() from clobbering them.
        self._reserve_system_file_clusters()

        # Find $MFTMirr data cluster so _mft_write_to_image() can keep it in sync.
        self._load_mft_mirror_info()

        # Build reverse mappings
        self._build_path_mappings()

        log(f"Initialized: {len(self._direct_run_map)} runs mapped, "
            f"{len(self.mft_record_to_source)} files tracked "
            f"({len(self.resident_file_data)} resident)")

        if self._protected_top_dirs:
            protected_files = sum(
                1 for src in self.mft_record_to_source.values()
                if self._is_source_protected(src)
            )
            protected_dirs = sum(
                1 for rel in self.mft_record_to_dir.values()
                if rel and rel.split(os.sep, 1)[0].lower() in self._protected_top_dirs
            )
            log(f"Protected (read-only at bridge): {sorted(self._protected_top_dirs)} "
                f"-> {protected_files} files, {protected_dirs} dirs")

    def close(self):
        """Close the memory-mapped image file."""
        if hasattr(self, 'image') and self.image:
            self.image.flush()
            self.image.close()
            self.image = None
        if hasattr(self, '_image_file') and self._image_file:
            self._image_file.close()
            self._image_file = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    def _run_map_lookup(self, cluster: int) -> Optional[Tuple[str, int]]:
        """Binary search in _direct_run_map for a cluster.

        Returns (source_path, file_offset) or None.
        O(log n) in number of runs, not clusters.
        """
        if not self._direct_run_map:
            return None
        # Find rightmost entry with start <= cluster
        lo, hi = 0, len(self._direct_run_map)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._direct_run_map[mid][0] <= cluster:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx < 0:
            return None
        start, end, path, base_offset = self._direct_run_map[idx]
        if cluster < end:
            return (path, base_offset + (cluster - start) * self.cluster_size)
        return None

    # =========================================================================
    # Public interface
    # =========================================================================

    def read(self, offset: int, length: int) -> bytes:
        """Read bytes from the virtual volume.

        Data clusters are read from ext4 source files.
        Metadata and unmapped regions are read from the image file.
        For resident files, ext4 content is injected into MFT reads.
        For sparse files, triggers lazy allocation on first access.
        For virtual files (ext4 only), synthesizes MFT records and data on-the-fly.
        """
        # Lock MFT-region reads to prevent torn reads during concurrent writes
        if self.is_mft_region(offset, length):
            with self.lock:
                return self._read_inner(offset, length)
        return self._read_inner(offset, length)

    def _read_inner(self, offset: int, length: int) -> bytes:
        """Inner read implementation (caller handles MFT locking)."""
        # Check if this read might be for a sparse file that needs allocation
        self._check_sparse_file_read(offset, length)

        result = bytearray(length)
        pos = 0

        while pos < length:
            byte_offset = offset + pos
            remaining = length - pos
            cluster = byte_offset // self.cluster_size
            cluster_offset = byte_offset % self.cluster_size

            # Check for virtual cluster first (from VirtualFileManager)
            virtual_data = None
            if self.virtual_file_manager:
                virtual_data = self.virtual_file_manager.read_virtual_cluster(cluster)

            # Check for virtualized real INDX clusters (intercept original clusters)
            virtual_indx_data = None
            if cluster in self.virtualized_indx_clusters:
                dir_record, block_idx, indx_data = self.virtualized_indx_clusters[cluster]
                virtual_indx_data = indx_data
                log(f"  INDX intercept: cluster={cluster} dir={dir_record} block={block_idx} size={len(indx_data)}")

            # Also check for virtual INDX clusters (fallback: new virtual cluster numbers)
            if not virtual_indx_data and cluster in self.virtual_indx_map:
                dir_record, block_idx = self.virtual_indx_map[cluster]
                if dir_record in self.virtualized_dirs:
                    vdir = self.virtualized_dirs[dir_record]
                    if 'indx_blocks' in vdir and block_idx < len(vdir['indx_blocks']):
                        virtual_indx_data = vdir['indx_blocks'][block_idx]
                    else:
                        log(f"  ERROR: indx_blocks not found or block_idx out of range")

            if virtual_indx_data:
                # Read from synthesized INDX block
                chunk_len = min(remaining, self.cluster_size - cluster_offset)
                data = virtual_indx_data[cluster_offset:cluster_offset + chunk_len]
                if len(data) < chunk_len:
                    data = data + b'\x00' * (chunk_len - len(data))
                result[pos:pos + len(data)] = data
                pos += chunk_len

            elif virtual_data:
                # Read from virtual cluster (ext4 file via VirtualFileManager)
                chunk_len = min(remaining, self.cluster_size - cluster_offset)
                data = virtual_data[cluster_offset:cluster_offset + chunk_len]
                result[pos:pos + len(data)] = data
                pos += chunk_len

            elif cluster in self.cluster_map:
                mapping = self.cluster_map[cluster]
                chunk_len = min(remaining, self.cluster_size - cluster_offset)

                if isinstance(mapping, tuple) and mapping[0] == 'bytes':
                    # Direct bytes data (INDX blocks)
                    block_data = mapping[1]
                    data = block_data[cluster_offset:cluster_offset + chunk_len]
                    if len(data) < chunk_len:
                        data = data + b'\x00' * (chunk_len - len(data))
                    result[pos:pos + len(data)] = data
                else:
                    # Read from ext4 source file
                    source_path, file_offset = mapping
                    read_offset = file_offset + cluster_offset
                    try:
                        with open(source_path, 'rb') as f:
                            f.seek(read_offset)
                            data = f.read(chunk_len)
                            if len(data) < chunk_len:
                                data += b'\x00' * (chunk_len - len(data))
                            result[pos:pos + len(data)] = data
                        # Record read for lazy allocator deallocation timeout
                        if self.lazy_allocator:
                            rel_path = self._get_rel_path(source_path)
                            self.lazy_allocator.record_read(rel_path)
                    except OSError:
                        pass  # Keep zeros on error

                pos += chunk_len

            elif (run_mapping := self._run_map_lookup(cluster)) is not None:
                # Read from ext4 source file via run-based map (large files)
                source_path, file_offset = run_mapping
                chunk_len = min(remaining, self.cluster_size - cluster_offset)
                read_offset = file_offset + cluster_offset
                try:
                    with open(source_path, 'rb') as f:
                        f.seek(read_offset)
                        data = f.read(chunk_len)
                        if len(data) < chunk_len:
                            data += b'\x00' * (chunk_len - len(data))
                        result[pos:pos + len(data)] = data
                    if self.lazy_allocator:
                        rel_path = self._get_rel_path(source_path)
                        self.lazy_allocator.record_read(rel_path)
                except OSError:
                    pass  # Keep zeros on error
                pos += chunk_len

            elif byte_offset < len(self.image):
                # Read from image (metadata)
                chunk_len = min(remaining, self.cluster_size - cluster_offset,
                                len(self.image) - byte_offset)
                result[pos:pos + chunk_len] = self.image[byte_offset:byte_offset + chunk_len]
                pos += chunk_len

            else:
                # Beyond image - zeros
                chunk_len = min(remaining, self.cluster_size - cluster_offset)
                pos += chunk_len

        # Inject ext4 content for resident files in the MFT area
        self._inject_resident_data(result, offset, length)

        # Inject virtual MFT records and directory entries
        if self.virtual_file_manager:
            self._inject_virtual_entries(result, offset, length)

        return bytes(result)

    def _inject_virtual_entries(self, result: bytearray, offset: int, length: int):
        """Inject virtual MFT records and directory entries into read result.

        This handles:
        1. Virtual MFT records for files that exist only in ext4
        2. Virtual directory entries in $INDEX_ROOT for those files
        """
        if not self.virtual_file_manager:
            return

        # Log if we have virtual files
        vfm = self.virtual_file_manager
        if vfm.virtual_files and self.is_mft_region(offset, length):
            log(f"_inject_virtual_entries MFT read: vfiles={list(vfm.virtual_files.keys())}")

        # Check if read is in MFT region
        if self.is_mft_region(offset, length):
            self._inject_virtual_mft_records(result, offset, length)

        # Check if read might include directory indexes
        self._inject_virtual_dir_entries(result, offset, length)

    def _check_sparse_file_read(self, offset: int, length: int):
        """Check if read is for a sparse file and trigger allocation if needed.

        Since we can't know which file a read is for at the cluster level,
        we use a different approach: look for reads to cluster ranges that
        return all zeros from the image. This indicates a sparse region.

        When we detect such a read and have pending sparse files, we allocate
        the sparse file that best matches the read pattern.
        """
        if not self.sparse_files:
            return

        if not self.lazy_allocator:
            return

        # Skip MFT region reads
        if self.is_mft_region(offset, length):
            return

        # Skip small reads (likely metadata probes)
        if length < self.cluster_size:  # At least 4KB read
            return

        cluster = offset // self.cluster_size

        # Check if this read is to one of a sparse file's allocated clusters
        # This is the most reliable way to detect sparse file access
        rel_path = self.sparse_file_clusters.get(cluster)
        if rel_path:
            log(f"  Read to sparse file cluster {cluster}: {rel_path}")
            self._trigger_sparse_allocation(rel_path)
            return

        # Check if this read is to a cluster that's not in our cluster_map
        if cluster in self.cluster_map or self._run_map_lookup(cluster) is not None:
            return  # Already mapped to a file

        # Check if the image has zeros at this location (sparse indicator)
        if offset + 4096 <= len(self.image):
            sample = self.image[offset:offset + 4096]
            if sample != b'\x00' * 4096:
                return  # Not zeros, probably metadata
        else:
            return  # Beyond image

        # Debug: log that we detected a potential sparse read
        log(f"  Potential sparse read at offset {offset} ({length} bytes)")

        # This looks like a read to a sparse file's data region
        # Find the matching sparse file and trigger blocking allocation
        for rel_path, (source_path, file_size, record_num) in list(self.sparse_files.items()):
            if self.lazy_allocator.needs_allocation(rel_path) or rel_path in self._allocating:
                log(f"  Detected sparse read, allocating: {rel_path}")
                self._trigger_sparse_allocation(rel_path)
                # Only allocate one file at a time
                break

    def _trigger_sparse_allocation(self, rel_path: str):
        """Trigger direct allocation for a sparse file.

        Uses direct NTFS image manipulation (bitmap + MFT data runs) which
        doesn't go through ntfs-3g/NBD, so it can be done synchronously.
        First read returns correct data immediately.
        """
        if rel_path not in self.sparse_files:
            return

        # Record read to prevent deallocation
        if self.lazy_allocator:
            self.lazy_allocator.record_read(rel_path)

        if rel_path in self._allocating:
            # Already allocating
            return

        # Check lazy_allocator state if present
        if self.lazy_allocator and not self.lazy_allocator.needs_allocation(rel_path):
            return  # Already allocated according to lazy_allocator

        # Mark as allocating
        self._allocating.add(rel_path)
        log(f"  Starting direct allocation: {rel_path}")

        try:
            # Synchronous direct allocation (no ntfs-3g, no deadlock)
            success = self.allocate_file_direct(rel_path)

            if success and self.lazy_allocator:
                # Update lazy_allocator state to match
                with self.lazy_allocator.state_lock:
                    self.lazy_allocator.file_states[rel_path] = 'allocated'
                    self.lazy_allocator.last_read_time[rel_path] = time.time()
        finally:
            self._allocating.discard(rel_path)

    def write(self, offset: int, data: bytes):
        """Write bytes to the virtual volume.

        MFT writes update the image immediately (so NTFS stays consistent) and
        queue the ext4 sync to a background thread so the NBD reply goes out
        without waiting for slow filesystem operations.
        Data cluster writes go to ext4 source files synchronously.
        Other metadata writes go to the image.
        """
        if self.is_mft_region(offset, len(data)):
            with self.lock:
                # Phase 1 (fast): write MFT data to image so NTFS sees it
                self._mft_write_to_image(offset, data)
            # Phase 2 (slow): sync changes to ext4 in background thread
            self._mft_queue.put((offset, bytes(data)))
        else:
            self._write_inner(offset, data)

    def _is_record_protected(self, record_num: int) -> bool:
        """Return True if record_num's path resolves under a protected top-level dir.

        Uses mft_record_to_source / mft_record_to_dir to map the record back to
        its relative path, then checks the top-level component against
        self._protected_top_dirs. Resident files have their data inside the MFT
        record, so blocking the record write also protects the file content.
        """
        if not self._protected_top_dirs:
            return False
        rel_path = None
        if record_num in self.mft_record_to_source:
            rel_path = self._get_rel_path(self.mft_record_to_source[record_num])
        elif record_num in self.mft_record_to_dir:
            rel_path = self.mft_record_to_dir[record_num]
        if not rel_path:
            return False
        top = rel_path.split(os.sep, 1)[0]
        return top.lower() in self._protected_top_dirs

    def _is_source_protected(self, source_path: str) -> bool:
        """Return True if source_path lies under a protected top-level dir of source_dir.

        Case-insensitive comparison; see _protected_top_dirs note.
        """
        if not self._protected_top_dirs:
            return False
        prefix = self.source_dir + os.sep
        if not source_path.startswith(prefix):
            return False
        rel = source_path[len(prefix):]
        top = rel.split(os.sep, 1)[0]
        return top.lower() in self._protected_top_dirs

    def _mft_write_to_image(self, offset: int, data: bytes):
        """Write MFT record data to the image (fast path, called under self.lock).

        Only updates the image bytes. The ext4 sync is done separately by
        _mft_sync_ext4_passes() running in the background worker thread.
        """
        write_end = offset + len(data)

        # Determine which MFT records are touched by this write
        touched_records = []
        cum_records = 0
        for disk_off, run_bytes in self._mft_runs:
            run_end = disk_off + run_bytes
            if offset < run_end and write_end > disk_off:
                # This run overlaps with the write
                overlap_start = max(offset, disk_off)
                overlap_end = min(write_end, run_end)
                first_rec_in_overlap = cum_records + (overlap_start - disk_off) // MFT_RECORD_SIZE
                last_rec_in_overlap = cum_records + (overlap_end - disk_off - 1) // MFT_RECORD_SIZE
                for rn in range(first_rec_in_overlap, last_rec_in_overlap + 1):
                    touched_records.append(rn)
            cum_records += run_bytes // MFT_RECORD_SIZE

        if not touched_records:
            return

        log(f"MFT write: records {touched_records[0]}-{touched_records[-1]}")

        # Write data to image at the actual disk offset
        if write_end <= len(self.image):
            # First, write the raw data to the image
            for record_num in touched_records:
                if record_num in self.dir_indx_clusters:
                    continue
                if record_num in self._direct_allocated_records:
                    # Protect directly-allocated file records from external overwrites.
                    # ntfs-3g or Windows journal replay may write stale sparse data
                    # runs for these records, which would silently undo allocate_file_direct().
                    continue
                if self._is_record_protected(record_num):
                    # User-configured read-only top-level dir: drop the write.
                    # The image keeps its current (good) record bytes; ext4 source
                    # never sees the write either, since _mft_sync_ext4_passes
                    # re-reads the (unchanged) record from the image.
                    continue
                rec_abs = self._rec_offset(record_num)
                if rec_abs is None:
                    continue
                # Determine the overlap between the write data and this record
                rec_end_abs = rec_abs + MFT_RECORD_SIZE
                overlap_start = max(offset, rec_abs)
                overlap_end = min(write_end, rec_end_abs)
                if overlap_start >= overlap_end:
                    continue
                data_start = overlap_start - offset
                data_end = overlap_end - offset
                chunk = data[data_start:data_end]
                self.image[overlap_start:overlap_start + len(chunk)] = chunk

                # Re-patch INDEX_ALLOC data_size and INDEX_BITMAP for protected
                # directories. Windows journal replay and access-time updates write
                # back old (smaller) values, hiding INDX blocks and making the
                # directory appear "corrupted and unreadable". By re-patching
                # immediately after the write we keep the fix alive indefinitely.
                # These fields don't fall at USA fixup positions (sector-end bytes
                # 510-511 / 1022-1023), so direct bytearray patching is safe.
                if record_num in self._protected_ia_sizes:
                    ia_off_in_rec, target_ds, ib_off_in_rec, ib_val_off, bitmap = \
                        self._protected_ia_sizes[record_num]
                    # Re-patch data_size and init_size
                    ds_off = rec_abs + ia_off_in_rec + 48
                    is_off = rec_abs + ia_off_in_rec + 56
                    packed = struct.pack('<Q', target_ds)
                    self.image[ds_off:ds_off + 8] = packed
                    self.image[is_off:is_off + 8] = packed
                    # Re-patch INDEX_BITMAP (ensures Windows sees all blocks as allocated)
                    if ib_off_in_rec >= 0 and bitmap:
                        bm_abs = rec_abs + ib_off_in_rec + ib_val_off
                        self.image[bm_abs:bm_abs + len(bitmap)] = bitmap

                # Keep $MFTMirr in sync: if this record falls within the
                # mirror's range, copy the full record to the mirror cluster.
                # This prevents ntfs-3g from crashing with "$MFTMirr does not
                # match $MFT" after any write to the mirrored records.
                if (self._mft_mirror_offset >= 0
                        and record_num < self._mft_mirror_record_count):
                    # Re-read the full (now-updated) record from the primary MFT
                    primary_off = rec_abs
                    mirror_off = self._mft_mirror_offset + record_num * MFT_RECORD_SIZE
                    full_record = self.image[primary_off:primary_off + MFT_RECORD_SIZE]
                    if len(full_record) == MFT_RECORD_SIZE:
                        self.image[mirror_off:mirror_off + MFT_RECORD_SIZE] = full_record

    def _mft_worker(self):
        """Background thread: drain the MFT write queue and sync changes to ext4.

        Runs one write at a time (FIFO) so rename sequences stay ordered.
        Acquires self.lock only briefly around metadata reads/writes; releases
        it before slow filesystem operations so concurrent reads are not blocked.
        """
        while True:
            offset, data = self._mft_queue.get()
            try:
                self._mft_sync_ext4_passes(offset, data)
            except Exception as e:
                log(f"MFT sync worker error: {e}")
                traceback.print_exc()
            finally:
                self._mft_queue.task_done()

    def _mft_sync_ext4_passes(self, offset: int, data: bytes):
        """Two-pass ext4 sync for an MFT write (called from background thread).

        Called without self.lock held. Each sub-method acquires self.lock only
        for brief metadata reads/writes, releasing it before slow ext4 operations.
        """
        write_end = offset + len(data)

        # Determine which MFT records are touched by this write
        touched_records = []
        cum_records = 0
        for disk_off, run_bytes in self._mft_runs:
            run_end = disk_off + run_bytes
            if offset < run_end and write_end > disk_off:
                overlap_start = max(offset, disk_off)
                overlap_end = min(write_end, run_end)
                first_rec = cum_records + (overlap_start - disk_off) // MFT_RECORD_SIZE
                last_rec = cum_records + (overlap_end - disk_off - 1) // MFT_RECORD_SIZE
                for rn in range(first_rec, last_rec + 1):
                    touched_records.append(rn)
            cum_records += run_bytes // MFT_RECORD_SIZE

        if not touched_records:
            return

        # Pass 1: Directory operations (renames and new directories)
        for record_num in touched_records:
            with self.lock:
                in_dir = record_num in self.mft_record_to_dir
                in_source = record_num in self.mft_record_to_source
            if in_dir:
                self._check_directory_rename(record_num)
            elif not in_source:
                if record_num >= 24:
                    log(f"  Pass1: record {record_num} -> _check_new_directory")
                self._check_new_directory(record_num)

        # Pass 2: File operations (deletions, renames, new files, content updates)
        for record_num in touched_records:
            with self.lock:
                in_source = record_num in self.mft_record_to_source
                in_dir = record_num in self.mft_record_to_dir
                is_direct = self._is_directly_allocated(record_num)
            if in_source:
                if is_direct:
                    continue
                if not self._check_file_deleted(record_num):
                    self._reparse_mft_record(record_num)
            elif not in_dir:
                self._check_new_file(record_num)

    def _write_inner(self, offset: int, data: bytes):
        """Inner write implementation for non-MFT writes."""
        cluster_size = self.cluster_size

        # MFT region is handled by write() via _mft_write_to_image + queue
        if self.is_mft_region(offset, len(data)):
            return

        # Route each cluster-aligned chunk
        pos = 0
        while pos < len(data):
            byte_offset = offset + pos
            cluster = byte_offset // cluster_size
            cluster_offset = byte_offset % cluster_size
            remaining = len(data) - pos
            chunk_len = min(remaining, cluster_size - cluster_offset)
            chunk_data = data[pos:pos + chunk_len]

            if cluster in self.cluster_map:
                mapping = self.cluster_map[cluster]
                if isinstance(mapping, tuple) and mapping[0] == 'bytes':
                    # Write to INDX block
                    block_data = bytearray(mapping[1])
                    block_data[cluster_offset:cluster_offset + chunk_len] = chunk_data
                    self.cluster_map[cluster] = ('bytes', bytes(block_data))
                else:
                    # Write to ext4 source file (per-cluster mapping)
                    source_path, file_offset = mapping
                    if self._is_source_protected(source_path):
                        pos += chunk_len
                        continue
                    write_offset = file_offset + cluster_offset
                    try:
                        with open(source_path, 'r+b') as f:
                            f.seek(write_offset)
                            f.write(chunk_data)
                        if not hasattr(self, '_write_logged'):
                            self._write_logged = set()
                        if source_path not in self._write_logged:
                            log(f"  Write {chunk_len}B to {os.path.basename(source_path)} at offset {write_offset}")
                            self._write_logged.add(source_path)
                    except OSError as e:
                        log(f"Write error for {source_path}: {e}")
                        # Don't fall back to image - that would be silently lost on next read
            elif (run_mapping := self._run_map_lookup(cluster)) is not None:
                # Write to ext4 source file (run-based mapping — all files with
                # RUN_MAP_THRESHOLD=0 end up here, ensuring writes reach ext4)
                source_path, file_offset = run_mapping
                if self._is_source_protected(source_path):
                    pos += chunk_len
                    continue
                write_offset = file_offset + cluster_offset
                try:
                    with open(source_path, 'r+b') as f:
                        f.seek(write_offset)
                        f.write(chunk_data)
                    if not hasattr(self, '_write_logged'):
                        self._write_logged = set()
                    if source_path not in self._write_logged:
                        log(f"  Write {chunk_len}B to {os.path.basename(source_path)} at offset {write_offset}")
                        self._write_logged.add(source_path)
                except OSError as e:
                    log(f"Write error for {source_path}: {e}")
                    # Don't fall back to image - that would be silently lost on next read
            else:
                # Write to image (metadata region)
                if byte_offset + chunk_len <= len(self.image):
                    self.image[byte_offset:byte_offset + chunk_len] = chunk_data

            pos += chunk_len

    def get_size(self) -> int:
        """Get total volume size."""
        return len(self.image)

    def flush(self):
        """Flush image changes to disk."""
        if self.image:
            self.image.flush()

    def protect_ia_size(self, record_num: int, ia_off: int, data_size: int,
                        ib_off: int = -1, ib_val_off: int = 0, bitmap: bytes = b''):
        """Register INDEX_ALLOC data_size and INDEX_BITMAP that must survive Windows writes.

        Called by bridge._fix_index_alloc_data_sizes() for each fixed directory.
        After any Windows write to this MFT record, _mft_write_to_image() will
        re-patch data_size, init_size, and INDEX_BITMAP so the fix is never reverted.

        Args:
            record_num: MFT record number of the directory
            ia_off: Byte offset of the INDEX_ALLOC attribute within the record
            data_size: The corrected data_size value to preserve
            ib_off: Byte offset of the INDEX_BITMAP attribute within the record (-1 if absent)
            ib_val_off: Byte offset of the bitmap value within the INDEX_BITMAP attribute
            bitmap: The corrected INDEX_BITMAP value to preserve
        """
        self._protected_ia_sizes[record_num] = (ia_off, data_size, ib_off, ib_val_off, bitmap)

    def rescan_mft(self):
        """Rescan the MFT to pick up changes made through ntfs-3g.

        Called after ext4->NTFS sync operations complete.
        """
        with self.lock:
            old_cluster_count = len(self.cluster_map)
            old_file_count = len(self.mft_record_to_source)
            old_files = set(self.mft_record_to_source.values())
            self._scan_mft()
            self._build_path_mappings()
            new_cluster_count = len(self.cluster_map)
            new_file_count = len(self.mft_record_to_source)
            new_files = set(self.mft_record_to_source.values())
            added_files = new_files - old_files
            removed_files = old_files - new_files
            log(f"Rescan: {old_cluster_count}->{new_cluster_count} clusters, "
                f"{old_file_count}->{new_file_count} files")
            if added_files:
                for f in added_files:
                    log(f"  + {os.path.basename(f)}")
            if removed_files:
                for f in removed_files:
                    log(f"  - {os.path.basename(f)}")

    # =========================================================================
    # Direct allocation (no ntfs-3g, no data copy)
    # =========================================================================

    def _find_bitmap_location(self):
        """Find $Bitmap (MFT record 6) data runs to locate cluster bitmap."""
        # $Bitmap is always MFT record 6
        record_offset = self._rec_offset(6)
        if record_offset is None:
            log("Warning: $Bitmap MFT record offset not found")
            return
        record = self._undo_fixups(bytearray(self.image[record_offset:record_offset + MFT_RECORD_SIZE]))

        if record[:4] != b'FILE':
            log("Warning: $Bitmap MFT record not found")
            return

        data_runs = self._extract_data_runs(record)
        if data_runs:
            self.bitmap_clusters = data_runs
            total_bitmap_clusters = sum(count for _, count in data_runs)
            log(f"  $Bitmap: {total_bitmap_clusters} clusters")

    def _load_bitmap_cache(self):
        """Read the NTFS bitmap from the image into RAM once.

        Subsequent reads return the cached bytearray directly (no allocation).
        Writes update it in-place and flush the changed bytes to the mmap.
        This eliminates the 500MB+ allocation storm during pre-allocation of
        thousands of files.
        """
        if not self.bitmap_clusters:
            return
        bitmap = bytearray()
        for start_cluster, count in self.bitmap_clusters:
            offset = start_cluster * self.cluster_size
            length = count * self.cluster_size
            bitmap.extend(self.image[offset:offset + length])
        self._bitmap_cache = bitmap
        log(f"  Bitmap cached: {len(bitmap) // (1024 * 1024)}MB in RAM")

    def _build_free_run_index(self):
        """Scan the bitmap ONCE and build a sorted free-run index.

        self._free_run_index is a sorted list of (start_cluster, count) tuples
        representing contiguous free-cluster runs.  All subsequent allocation
        searches use bisect on this list — O(log n) instead of O(n_clusters).
        """
        if not self._bitmap_cache or not self.bitmap_clusters:
            return

        bitmap = self._bitmap_cache
        min_cluster = max(16, self.mft_cluster + 100)

        index = []
        run_start = -1
        run_length = 0

        for byte_idx in range(min_cluster // 8, (self.total_clusters + 7) // 8):
            if byte_idx >= len(bitmap):
                break
            byte_val = bitmap[byte_idx]
            # Fast-path: whole byte free or whole byte used
            if byte_val == 0x00:
                cluster = byte_idx * 8
                if cluster < min_cluster:
                    pass  # Fall through to bit-by-bit below
                elif cluster + 8 <= self.total_clusters:
                    if run_length == 0:
                        run_start = cluster
                    run_length += 8
                    continue
            if byte_val == 0xFF:
                cluster = byte_idx * 8
                if cluster >= min_cluster and run_length > 0:
                    index.append((run_start, run_length))
                    run_start = -1
                    run_length = 0
                continue
            # Bit-by-bit for partial/mixed bytes
            for bit in range(8):
                cluster = byte_idx * 8 + bit
                if cluster < min_cluster:
                    continue
                if cluster >= self.total_clusters:
                    break
                if not (byte_val & (1 << bit)):
                    if run_length == 0:
                        run_start = cluster
                    run_length += 1
                else:
                    if run_length > 0:
                        index.append((run_start, run_length))
                        run_start = -1
                        run_length = 0

        if run_length > 0:
            index.append((run_start, run_length))

        self._free_run_index = index
        total_free = sum(c for _, c in index)
        log(f"  Free-run index: {len(index):,} runs, {total_free:,} free clusters")

    def _reserve_system_file_clusters(self):
        """Remove NTFS system file clusters from the free-run index.

        MFT records 0-15 are NTFS metadata files ($MFT, $MFTMirr, $LogFile,
        $Volume, $AttrDef, root dir, $Bitmap, $Boot, etc.).  ntfs-3g sometimes
        places these at high LCNs in sparse space and does NOT mark those
        clusters as used in $Bitmap (because the sparse image has never been
        written to disk).  If allocate_file_direct() picks up one of those
        clusters for a user file, reads of that metadata cluster return user
        file data — causing $MFTMirr corruption and directory errors.

        This method scans every attribute in each of the 16 system records and
        removes any non-resident cluster range from the free-run index so they
        can never be handed out to user files.
        """
        total_reserved = 0
        for record_num in range(16):
            record_offset = self._rec_offset(record_num)
            if record_offset is None:
                continue
            raw = self.image[record_offset:record_offset + MFT_RECORD_SIZE]
            if len(raw) < MFT_RECORD_SIZE or raw[:4] != b'FILE':
                continue
            record = self._undo_fixups(bytearray(raw))

            # Walk every attribute and collect non-resident cluster runs
            first_attr = struct.unpack('<H', record[20:22])[0]
            off = first_attr
            while off < MFT_RECORD_SIZE - 8:
                attr_type = struct.unpack('<I', record[off:off + 4])[0]
                if attr_type == 0xFFFFFFFF:
                    break
                attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
                if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                    break

                non_res = record[off + 8]
                if non_res:
                    try:
                        runs_off = struct.unpack('<H', record[off + 32:off + 34])[0]
                        real_size = struct.unpack('<Q', record[off + 48:off + 56])[0]
                        runs_bytes = record[off + runs_off:off + attr_len]
                        parsed = self._parse_data_runs(runs_bytes, real_size)
                        for cluster, count in parsed:
                            if cluster >= 0 and count > 0:
                                self._free_index_remove(cluster, count)
                                total_reserved += count
                    except Exception:
                        pass

                off += attr_len

        log(f"  System file reservation: removed {total_reserved:,} clusters from free-run index")

    def _load_mft_mirror_info(self):
        """Find $MFTMirr's data cluster and record count.

        Reads MFT record 1 ($MFTMirr), extracts the non-resident $DATA run to
        locate where the mirror copy lives on disk, and stores:
          self._mft_mirror_offset       — byte offset in the image
          self._mft_mirror_record_count — number of MFT records the mirror holds

        Called once at startup so _mft_write_to_image() can keep the mirror
        in sync with the primary MFT without any external ntfsfix step.
        """
        try:
            _rec1_off = self._rec_offset(1)
            if _rec1_off is None:
                log("  MFTMirr: record 1 offset not found — mirror sync disabled")
                return
            raw = self.image[_rec1_off:_rec1_off + MFT_RECORD_SIZE]
            if len(raw) < MFT_RECORD_SIZE or raw[:4] != b'FILE':
                log("  MFTMirr: record 1 missing or invalid — mirror sync disabled")
                return
            record = self._undo_fixups(bytearray(raw))

            first_attr = struct.unpack('<H', record[20:22])[0]
            off = first_attr
            while off < MFT_RECORD_SIZE - 8:
                attr_type = struct.unpack('<I', record[off:off + 4])[0]
                if attr_type == 0xFFFFFFFF:
                    break
                attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
                if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                    break
                if attr_type == 0x80 and record[off + 8]:  # non-resident $DATA
                    runs_off = struct.unpack('<H', record[off + 32:off + 34])[0]
                    alloc_size = struct.unpack('<Q', record[off + 40:off + 48])[0]
                    runs_bytes = record[off + runs_off:off + attr_len]
                    parsed = self._parse_data_runs(runs_bytes, alloc_size)
                    if parsed:
                        mirror_cluster, mirror_count = parsed[0]
                        if mirror_cluster > 0:
                            self._mft_mirror_offset = mirror_cluster * self.cluster_size
                            self._mft_mirror_record_count = (mirror_count * self.cluster_size) // MFT_RECORD_SIZE
                            log(f"  MFTMirr: cluster={mirror_cluster}, "
                                f"offset={self._mft_mirror_offset}, "
                                f"records={self._mft_mirror_record_count}")
                            return
                off += attr_len

            log("  MFTMirr: $DATA attribute not found — mirror sync disabled")
        except Exception as e:
            log(f"  MFTMirr: error loading mirror info: {e} — mirror sync disabled")

    def _get_mft_runs(self):
        """Parse MFT $DATA runs from MFT record 0 for non-contiguous MFT support.

        Returns list of (disk_byte_offset, byte_length) and total_records.
        """
        rec0 = self._undo_fixups(bytearray(
            self.image[self.mft_offset:self.mft_offset + MFT_RECORD_SIZE]))
        mft_runs = []
        total_records = 0
        a = struct.unpack('<H', rec0[20:22])[0]
        while a < MFT_RECORD_SIZE - 8:
            at = struct.unpack('<I', rec0[a:a+4])[0]
            if at == 0xFFFFFFFF: break
            al = struct.unpack('<I', rec0[a+4:a+8])[0]
            if al == 0: break
            if at == 0x80 and rec0[a+8]:  # $DATA non-resident
                data_size = struct.unpack('<Q', rec0[a+48:a+56])[0]
                total_records = data_size // MFT_RECORD_SIZE
                ro = struct.unpack('<H', rec0[a+32:a+34])[0]
                rb = rec0[a+ro:a+al]
                pos = 0; lcn = 0
                while pos < len(rb) and rb[pos]:
                    hdr = rb[pos]; pos += 1
                    ls = hdr & 0xF; os2 = (hdr >> 4) & 0xF
                    rlen = int.from_bytes(rb[pos:pos+ls], 'little'); pos += ls
                    if os2:
                        delta = int.from_bytes(rb[pos:pos+os2], 'little', signed=True)
                        pos += os2; lcn += delta
                        mft_runs.append((lcn * self.cluster_size,
                                         rlen * self.cluster_size))
                break
            a += al
        return mft_runs, total_records

    def _mft_record_offset(self, rec_num, mft_runs):
        """Resolve MFT record number to image byte offset using data runs."""
        stream_off = rec_num * MFT_RECORD_SIZE
        cum = 0
        for disk_off, run_bytes in mft_runs:
            if cum + run_bytes > stream_off:
                return disk_off + (stream_off - cum)
            cum += run_bytes
        return None

    def _rec_offset(self, record_num):
        """Get disk byte offset for MFT record number."""
        return self._mft_record_offset(record_num, self._mft_runs)

    def _offset_to_rec(self, disk_offset):
        """Get MFT record number from disk byte offset."""
        cum_records = 0
        for disk_off, run_bytes in self._mft_runs:
            if disk_off <= disk_offset < disk_off + run_bytes:
                byte_within_run = disk_offset - disk_off
                return cum_records + byte_within_run // MFT_RECORD_SIZE
            cum_records += run_bytes // MFT_RECORD_SIZE
        return None

    def fix_indx_clusters(self):
        """Mark all directory $INDEX_ALLOC clusters as used in bitmap + free-run index.

        Called after the production ntfs-3g mount, which may have moved INDX pages
        to new cluster locations while leaving those new locations FREE in $Bitmap.
        Without this fix, Windows reports those directories as 'corrupted and
        unreadable' because the $Bitmap does not agree with $INDEX_ALLOC data runs.

        Reads MFT records from self.image (hot cache or mmap) so it sees any
        in-flight MFT changes from ntfs-3g's startup fixups.
        """
        if not self.bitmap_clusters:
            log("  INDX fix: no bitmap, skipping")
            return

        mft_runs, total_records = self._get_mft_runs()
        indx_clusters = set()

        for rec_num in range(total_records):
            record_offset = self._mft_record_offset(rec_num, mft_runs)
            if record_offset is None:
                continue
            raw = self.image[record_offset:record_offset + MFT_RECORD_SIZE]
            if len(raw) < MFT_RECORD_SIZE or raw[:4] != b'FILE':
                continue

            record = self._undo_fixups(bytearray(raw))
            flags = struct.unpack('<H', record[22:24])[0]
            if not (flags & 0x2):  # Not a directory
                continue

            first_attr = struct.unpack('<H', record[20:22])[0]
            off = first_attr
            while off < MFT_RECORD_SIZE - 8:
                attr_type = struct.unpack('<I', record[off:off + 4])[0]
                if attr_type == 0xFFFFFFFF:
                    break
                attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
                if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                    break
                if attr_type == 0xA0 and record[off + 8]:  # $INDEX_ALLOC non-res
                    try:
                        runs_off = struct.unpack('<H', record[off + 32:off + 34])[0]
                        rb = bytes(record[off + runs_off:off + attr_len])
                        parsed = self._parse_data_runs(rb, 0)
                        for cluster, count in parsed:
                            if cluster >= 0 and count > 0:
                                for c in range(cluster, cluster + count):
                                    indx_clusters.add(c)
                    except Exception:
                        pass
                off += attr_len

        if not indx_clusters:
            log("  INDX fix: no INDX clusters found")
            return

        bitmap = self._read_bitmap()
        fixed = 0
        for cluster in sorted(indx_clusters):
            if cluster >= self.total_clusters:
                continue
            byte_idx = cluster // 8
            bit = (cluster % 8)
            if byte_idx < len(bitmap) and not (bitmap[byte_idx] & (1 << bit)):
                bitmap[byte_idx] |= (1 << bit)
                self._free_index_remove(cluster, 1)
                fixed += 1
        if fixed:
            self._write_bitmap(bitmap)
        log(f"  INDX fix: scanned {rec_num} records, found {len(indx_clusters)} clusters, "
            f"fixed {fixed} FREE bits")

    def _free_index_remove(self, start: int, count: int):
        """Remove [start, start+count) from the free-run index."""
        end = start + count
        i = bisect.bisect_left(self._free_run_index, (start,))
        if i > 0 and self._free_run_index[i - 1][0] + self._free_run_index[i - 1][1] > start:
            i -= 1

        to_delete = []
        to_add = []
        j = i
        while j < len(self._free_run_index):
            s, c = self._free_run_index[j]
            if s >= end:
                break
            e = s + c
            if e <= start:
                j += 1
                continue
            to_delete.append(j)
            if s < start:
                to_add.append((s, start - s))
            if e > end:
                to_add.append((end, e - end))
            j += 1

        for idx in reversed(to_delete):
            del self._free_run_index[idx]
        for entry in to_add:
            bisect.insort(self._free_run_index, entry)

    def _free_index_add(self, start: int, count: int):
        """Add [start, start+count) to the free-run index, merging adjacent runs."""
        end = start + count
        merge_start = start
        merge_end = end

        i = bisect.bisect_left(self._free_run_index, (start,))
        if i > 0:
            ps, pc = self._free_run_index[i - 1]
            if ps + pc >= start:
                merge_start = min(merge_start, ps)
                merge_end = max(merge_end, ps + pc)
                i -= 1

        j = i
        while j < len(self._free_run_index):
            s, c = self._free_run_index[j]
            if s > merge_end:
                break
            merge_end = max(merge_end, s + c)
            j += 1

        self._free_run_index[i:j] = [(merge_start, merge_end - merge_start)]

    def _read_bitmap(self) -> bytearray:
        """Return the cached cluster bitmap (loaded once at startup)."""
        if self._bitmap_cache is not None:
            return self._bitmap_cache
        # Fallback: read from image (before cache is loaded)
        bitmap = bytearray()
        for start_cluster, count in self.bitmap_clusters:
            offset = start_cluster * self.cluster_size
            length = count * self.cluster_size
            bitmap.extend(self.image[offset:offset + length])
        return bitmap

    def _write_bitmap(self, bitmap: bytearray):
        """Update cached bitmap in-place and flush changed bytes to the mmap."""
        if self._bitmap_cache is not None and bitmap is not self._bitmap_cache:
            # Caller modified a separate bytearray — sync it back to cache
            self._bitmap_cache[:] = bitmap
        pos = 0
        for start_cluster, count in self.bitmap_clusters:
            offset = start_cluster * self.cluster_size
            length = count * self.cluster_size
            self.image[offset:offset + length] = bitmap[pos:pos + length]
            pos += length

    def _find_free_clusters(self, count: int) -> Optional[List[int]]:
        """Find 'count' free clusters in the bitmap.

        Returns list of cluster numbers, or None if not enough free space.
        Prefers contiguous blocks to minimize data run fragmentation.
        """
        if not self.bitmap_clusters:
            return None

        bitmap = self._read_bitmap()

        # Skip system clusters (first ~16 clusters are usually reserved)
        start_search = max(16, self.mft_cluster + 100)  # Start after MFT region

        # Find all contiguous free runs in the bitmap
        free_runs = []  # (start_cluster, length)
        run_start = -1
        run_length = 0
        total_free = 0

        for byte_idx in range(start_search // 8, len(bitmap)):
            byte_val = bitmap[byte_idx]
            for bit in range(8):
                cluster = byte_idx * 8 + bit
                if cluster < start_search:
                    continue
                if cluster >= self.total_clusters:
                    break
                if not (byte_val & (1 << bit)):  # Bit 0 = free
                    if run_length == 0:
                        run_start = cluster
                    run_length += 1
                    total_free += 1
                    # Fast path: single contiguous block large enough
                    if run_length >= count:
                        return list(range(run_start, run_start + count))
                else:
                    if run_length > 0:
                        free_runs.append((run_start, run_length))
                        run_length = 0
            if cluster >= self.total_clusters:
                break

        # Don't forget last run
        if run_length > 0:
            free_runs.append((run_start, run_length))

        if total_free < count:
            return None

        # Use largest runs first to minimize data run fragmentation
        free_runs.sort(key=lambda x: x[1], reverse=True)
        result = []
        for start, length in free_runs:
            take = min(length, count - len(result))
            result.extend(range(start, start + take))
            if len(result) >= count:
                return sorted(result)

        return None

    def _mark_clusters_used(self, clusters: List[int]):
        """Mark clusters as used in the bitmap and remove from free-run index."""
        if not self.bitmap_clusters:
            return
        bitmap = self._read_bitmap()
        for cluster in clusters:
            byte_idx = cluster // 8
            if byte_idx < len(bitmap):
                bitmap[byte_idx] |= (1 << (cluster % 8))
                self._free_index_remove(cluster, 1)
        self._write_bitmap(bitmap)

    def _mark_clusters_free(self, clusters: List[int]):
        """Mark clusters as free in the bitmap and add to free-run index."""
        if not self.bitmap_clusters:
            return
        bitmap = self._read_bitmap()
        for cluster in clusters:
            byte_idx = cluster // 8
            if byte_idx < len(bitmap):
                bitmap[byte_idx] &= ~(1 << (cluster % 8))
                self._free_index_add(cluster, 1)
        self._write_bitmap(bitmap)

    def _find_free_cluster_runs(self, needed: int, max_runs: int = 60) -> Optional[List[Tuple[int, int]]]:
        """Find free clusters as runs using the in-RAM free-run index.

        O(log n + result_size) instead of O(n_clusters) bitmap scan.

        Args:
            needed: Total number of clusters needed
            max_runs: Maximum number of runs (must fit in MFT record)

        Returns:
            List of (start_cluster, count) tuples, or None if not enough space
        """
        if not self.bitmap_clusters or not self._free_run_index:
            return None

        min_start = max(16, self.mft_cluster + 100)

        # Try twice: first from the alloc watermark (avoids re-scanning fragmented
        # earlier regions), then from the beginning if that doesn't yield enough.
        for search_from in [max(min_start, self._alloc_watermark), min_start]:
            result = []
            remaining = needed

            # Binary search for first run at/after search_from
            idx = bisect.bisect_left(self._free_run_index, (search_from,))
            # The previous entry may extend past search_from
            if idx > 0:
                prev_s, prev_c = self._free_run_index[idx - 1]
                if prev_s + prev_c > search_from:
                    idx -= 1

            for i in range(idx, len(self._free_run_index)):
                s, c = self._free_run_index[i]
                actual_start = max(s, search_from)
                actual_count = s + c - actual_start
                if actual_count <= 0:
                    continue

                take = min(actual_count, remaining)
                result.append((actual_start, take))
                remaining -= take

                if remaining <= 0:
                    result.sort()
                    return result

                if len(result) >= max_runs:
                    result = []  # Can't fit; try from beginning
                    break

            if not result and remaining <= 0:
                return []  # Already returned above

            if search_from <= min_start:
                break  # Already tried from beginning

        # Fallback: sequential search failed (too many small runs hitting max_runs
        # limit before accumulating enough clusters).  Pick the max_runs LARGEST
        # free runs from anywhere in the index — this maximises clusters per run
        # and minimises fragmentation in the MFT data-run list.
        sorted_by_size = sorted(
            ((s, c) for s, c in self._free_run_index if s >= min_start and c > 0),
            key=lambda x: x[1], reverse=True
        )
        result = []
        remaining = needed
        for s, c in sorted_by_size[:max_runs]:
            take = min(c, remaining)
            result.append((s, take))
            remaining -= take
            if remaining <= 0:
                result.sort()
                return result

        log(f"  ERROR: Not enough free space for {needed} clusters "
            f"(free runs: {len(self._free_run_index)})")
        return None

    def _mark_cluster_runs_used(self, runs: List[Tuple[int, int]]):
        """Mark runs as used in the bitmap cache and remove from free-run index."""
        if not self.bitmap_clusters:
            return
        bitmap = self._read_bitmap()
        for start, count in runs:
            _set_bitmap_bits(bitmap, start, count, True)
            self._free_index_remove(start, count)
        self._write_bitmap(bitmap)

    def _mark_cluster_runs_free(self, runs: List[Tuple[int, int]]):
        """Mark runs as free in the bitmap cache and add to free-run index."""
        if not self.bitmap_clusters:
            return
        bitmap = self._read_bitmap()
        for start, count in runs:
            _set_bitmap_bits(bitmap, start, count, False)
            self._free_index_add(start, count)
        self._write_bitmap(bitmap)

    def _max_data_runs_in_record(self, record_num: int) -> int:
        """Calculate max data runs that fit in this MFT record's $DATA attribute.

        Reads the record, strips non-essential attrs, finds $DATA offset,
        and calculates available space for data run entries.
        """
        record_offset = self._rec_offset(record_num)
        if record_offset is None:
            return 60  # Fallback default
        record = self._undo_fixups(bytearray(
            self.image[record_offset:record_offset + MFT_RECORD_SIZE]
        ))

        if record[:4] != b'FILE':
            return 60  # Fallback default

        # Strip to get maximum available space
        record = self._strip_nonessential_attrs(record)

        # Find where $DATA would end (or end marker position)
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr
        data_attr_off = -1

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break
            attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break
            name_len = record[off + 9]
            if attr_type == 0x80 and name_len == 0:
                data_attr_off = off
            off += attr_len

        # Available space: from $DATA start to end of record
        # Non-resident $DATA header = 64 bytes, end marker = 4 bytes, terminator = 1 byte
        if data_attr_off >= 0:
            available = MFT_RECORD_SIZE - data_attr_off - 64 - 4 - 1
        else:
            # No existing $DATA, use end-of-attrs position
            available = MFT_RECORD_SIZE - off - 64 - 4 - 1

        # Worst-case per run: 1 (header) + 4 (length) + 5 (offset) = 10 bytes
        max_runs = max(1, available // 10)
        return min(max_runs, 60)  # Cap at 60 for safety

    def _strip_nonessential_attrs(self, record: bytearray) -> bytearray:
        """Strip non-essential attributes from an MFT record to free space.

        Keeps only $STANDARD_INFORMATION (0x10), $FILE_NAME (0x30),
        and $DATA (0x80, unnamed). Removes $SECURITY_DESCRIPTOR (0x50),
        $OBJECT_ID (0x40), $LOGGED_UTILITY_STREAM (0x100), extra named
        $DATA streams, and other Windows-added attributes.
        """
        first_attr = struct.unpack('<H', record[20:22])[0]
        essential_types = {0x10, 0x30, 0x80}

        # Collect essential attributes
        kept_attrs = []
        off = first_attr
        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break
            attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break
            name_len = record[off + 9]
            # Keep essential types; for $DATA only keep unnamed
            if attr_type in essential_types:
                if attr_type == 0x80 and name_len > 0:
                    pass  # Skip named $DATA streams
                else:
                    kept_attrs.append(bytes(record[off:off + attr_len]))
            off += attr_len

        # Rebuild record with only essential attributes
        new_record = bytearray(MFT_RECORD_SIZE)
        new_record[:first_attr] = record[:first_attr]
        write_off = first_attr
        for attr in kept_attrs:
            new_record[write_off:write_off + len(attr)] = attr
            write_off += len(attr)
        # End marker
        struct.pack_into('<I', new_record, write_off, 0xFFFFFFFF)
        write_off += 4

        # Update used size
        struct.pack_into('<I', new_record, 24, write_off)
        return new_record

    def _update_mft_data_runs(self, record_num: int, data_runs: List[Tuple[int, int]],
                               file_size: int) -> bool:
        """Update or insert the $DATA attribute's data runs in an MFT record.

        Handles two cases:
        - REPLACE: existing $DATA attribute is found and updated with new data runs
        - INSERT: no $DATA attribute exists (e.g. os.truncate failed during populate)
          and a new one is inserted before the end-of-attributes marker

        Args:
            record_num: MFT record number
            data_runs: List of (cluster_count, start_cluster) tuples
            file_size: Actual file size in bytes

        Returns:
            True if successful
        """
        from .data_runs import encode_data_runs

        record_offset = self._rec_offset(record_num)
        if record_offset is None:
            return False
        record = self._undo_fixups(bytearray(self.image[record_offset:record_offset + MFT_RECORD_SIZE]))

        if record[:4] != b'FILE':
            return False

        runs_bytes = encode_data_runs(data_runs)
        total_clusters = sum(c for c, _ in data_runs)
        alloc_size = total_clusters * self.cluster_size

        # Build non-resident $DATA attribute.
        # name_offset=64 is required by NTFS spec for no-name non-resident attrs;
        # setting it to 0 causes CHKDSK to flag the attribute as corrupt.
        attr_size = 64 + len(runs_bytes)
        attr_size_aligned = (attr_size + 7) & ~7  # 8-byte aligned

        new_attr = bytearray(attr_size_aligned)
        struct.pack_into('<I', new_attr, 0, 0x80)   # Type: $DATA
        struct.pack_into('<I', new_attr, 4, attr_size_aligned)  # Length
        new_attr[8] = 1                              # Non-resident flag
        new_attr[9] = 0                              # Name length (0 = unnamed)
        struct.pack_into('<H', new_attr, 10, 64)    # Name offset (64 = standard for no name)
        struct.pack_into('<H', new_attr, 12, 0)     # Flags
        struct.pack_into('<H', new_attr, 14, 0)     # Instance ID
        struct.pack_into('<Q', new_attr, 16, 0)     # Start VCN
        struct.pack_into('<Q', new_attr, 24, total_clusters - 1 if total_clusters > 0 else 0)  # End VCN
        struct.pack_into('<H', new_attr, 32, 64)    # Data runs offset
        struct.pack_into('<H', new_attr, 34, 0)     # Compression unit
        struct.pack_into('<I', new_attr, 36, 0)     # Padding
        struct.pack_into('<Q', new_attr, 40, alloc_size)   # Allocated size
        struct.pack_into('<Q', new_attr, 48, file_size)    # Real size
        struct.pack_into('<Q', new_attr, 56, file_size)    # Initialized size
        new_attr[64:64 + len(runs_bytes)] = runs_bytes

        # Try with original record first, then with stripped attrs if it doesn't fit
        for attempt in range(2):
            if attempt == 1:
                record = self._strip_nonessential_attrs(record)

            first_attr = struct.unpack('<H', record[20:22])[0]
            off = first_attr
            found_data_off = None    # offset of existing $DATA attr (REPLACE case)
            end_of_attrs_off = None  # offset of 0xFFFFFFFF marker (INSERT case)

            while off < MFT_RECORD_SIZE - 8:
                attr_type = struct.unpack('<I', record[off:off + 4])[0]
                if attr_type == 0xFFFFFFFF:
                    end_of_attrs_off = off
                    break

                attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
                if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                    break

                name_len = record[off + 9]
                if attr_type == 0x80 and name_len == 0:  # $DATA (unnamed)
                    found_data_off = off
                    break

                off += attr_len

            if found_data_off is not None:
                # REPLACE existing $DATA attribute
                off = found_data_off
                old_attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
                used_size = struct.unpack('<I', record[24:28])[0]
                if used_size > MFT_RECORD_SIZE or used_size < off + old_attr_len:
                    used_size = MFT_RECORD_SIZE  # Fallback
                remaining = record[off + old_attr_len:used_size]

                new_used = off + len(new_attr) + len(remaining)
                if new_used > MFT_RECORD_SIZE:
                    if attempt == 0:
                        continue  # Try again after stripping attrs
                    log(f"  ERROR: Data runs too large for MFT record ({new_used} > {MFT_RECORD_SIZE})")
                    return False

                new_record = bytearray(MFT_RECORD_SIZE)
                new_record[:off] = record[:off]
                new_record[off:off + len(new_attr)] = new_attr
                new_record[off + len(new_attr):off + len(new_attr) + len(remaining)] = remaining
                struct.pack_into('<I', new_record, 24, new_used)

            elif end_of_attrs_off is not None:
                # INSERT new $DATA before end-of-attrs marker.
                # This handles records where os.truncate failed during populate,
                # leaving the MFT record without any $DATA attribute.
                new_used = end_of_attrs_off + len(new_attr) + 4  # +4 for end marker
                if new_used > MFT_RECORD_SIZE:
                    if attempt == 0:
                        continue  # Try again after stripping attrs
                    log(f"  ERROR: No room to insert $DATA in MFT record {record_num} "
                        f"({new_used} > {MFT_RECORD_SIZE})")
                    return False

                new_record = bytearray(MFT_RECORD_SIZE)
                new_record[:end_of_attrs_off] = record[:end_of_attrs_off]
                new_record[end_of_attrs_off:end_of_attrs_off + len(new_attr)] = new_attr
                # Write end-of-attributes marker
                struct.pack_into('<I', new_record, end_of_attrs_off + len(new_attr), 0xFFFFFFFF)
                struct.pack_into('<I', new_record, 24, new_used)

            else:
                # Could not find $DATA or end-of-attrs marker — record is malformed
                continue

            # Apply NTFS fixups and write back to hot cache
            self._apply_fixups_to_record(new_record)
            self.image[record_offset:record_offset + MFT_RECORD_SIZE] = new_record
            return True

        return False

    def _update_filename_sizes(self, record_num: int, alloc_size: int, data_size: int) -> bool:
        """Update $FILE_NAME attribute size fields in an MFT record.

        Called after allocate_file_direct() so Windows sees the correct file size
        in directory listings (Windows reads size from $FILE_NAME, not $DATA).
        """
        record_offset = self._rec_offset(record_num)
        if record_offset is None:
            return False
        record = self._undo_fixups(bytearray(
            self.image[record_offset:record_offset + MFT_RECORD_SIZE]))
        if record[:4] != b'FILE':
            return False

        modified = False
        attr_offset = struct.unpack('<H', record[20:22])[0]
        while attr_offset + 8 < MFT_RECORD_SIZE:
            attr_type = struct.unpack('<I', record[attr_offset:attr_offset + 4])[0]
            attr_len = struct.unpack('<I', record[attr_offset + 4:attr_offset + 8])[0]
            if attr_type == 0xFFFFFFFF or attr_len == 0:
                break
            if attr_type == 0x30:  # $FILE_NAME
                non_res = record[attr_offset + 8]
                if not non_res:
                    val_off = struct.unpack('<H', record[attr_offset + 20:attr_offset + 22])[0]
                    fn_abs = attr_offset + val_off
                    if fn_abs + 56 <= MFT_RECORD_SIZE:
                        struct.pack_into('<Q', record, fn_abs + 40, alloc_size)
                        struct.pack_into('<Q', record, fn_abs + 48, data_size)
                        modified = True
            attr_offset += attr_len

        if modified:
            self._apply_fixups_to_record(record)
            self.image[record_offset:record_offset + MFT_RECORD_SIZE] = record
        return modified

    def _update_i30_entry(self, parent_record_num: int, file_record_num: int,
                          alloc_size: int, data_size: int) -> bool:
        """Update file size in parent directory's $I30 index entry.

        Called after allocate_file_direct() so directory listings show the
        correct file size. The $I30 index is the primary source Windows uses.
        """
        from .data_runs import decode_data_runs

        par_offset = self._rec_offset(parent_record_num)
        if par_offset is None:
            return False
        par_rec = self._undo_fixups(bytearray(
            self.image[par_offset:par_offset + MFT_RECORD_SIZE]))
        if par_rec[:4] != b'FILE':
            return False

        # Find $INDEX_ALLOCATION ($A0) data runs
        ia_runs = []
        attr_offset = struct.unpack('<H', par_rec[20:22])[0]
        while attr_offset + 8 < MFT_RECORD_SIZE:
            attr_type = struct.unpack('<I', par_rec[attr_offset:attr_offset + 4])[0]
            attr_len = struct.unpack('<I', par_rec[attr_offset + 4:attr_offset + 8])[0]
            if attr_type == 0xFFFFFFFF or attr_len == 0:
                break
            if attr_type == 0xA0:  # $INDEX_ALLOCATION
                non_res = par_rec[attr_offset + 8]
                if non_res:
                    runs_off = struct.unpack('<H', par_rec[attr_offset + 32:attr_offset + 34])[0]
                    runs_bytes = par_rec[attr_offset + runs_off:attr_offset + attr_len]
                    ia_runs = decode_data_runs(bytes(runs_bytes))
            attr_offset += attr_len

        if not ia_runs:
            return False

        indx_size = self.cluster_size
        for run_count, run_lcn in ia_runs:
            if run_lcn < 0:  # sparse run
                continue
            for i in range(run_count):
                lcn = run_lcn + i
                block_offset = lcn * self.cluster_size
                if block_offset + indx_size > len(self.image):
                    continue

                indx = bytearray(self.image[block_offset:block_offset + indx_size])
                if indx[:4] != b'INDX':
                    continue

                # Undo fixups (block-size-aware)
                usa_off = struct.unpack('<H', indx[4:6])[0]
                usa_cnt = struct.unpack('<H', indx[6:8])[0]
                for j in range(1, usa_cnt):
                    sec_end = j * 512 - 2
                    if usa_off + j * 2 + 2 <= indx_size and sec_end + 2 <= indx_size:
                        orig = struct.unpack('<H',
                            indx[usa_off + j * 2:usa_off + j * 2 + 2])[0]
                        struct.pack_into('<H', indx, sec_end, orig)

                # Parse index entries
                entries_rel_off = struct.unpack('<I', indx[24:28])[0]
                entries_off = entries_rel_off + 24
                entries_size = struct.unpack('<I', indx[28:32])[0]
                pos = entries_off
                end = entries_off + entries_size
                found = False

                while pos < end and pos + 16 <= indx_size:
                    mft_ref = struct.unpack('<Q', indx[pos:pos + 8])[0] & 0xFFFFFFFFFFFF
                    entry_len = struct.unpack('<H', indx[pos + 8:pos + 10])[0]
                    key_len = struct.unpack('<H', indx[pos + 10:pos + 12])[0]
                    flags = struct.unpack('<H', indx[pos + 12:pos + 14])[0]

                    if entry_len == 0 or pos + entry_len > end:
                        break
                    if flags & 2:  # End marker
                        break

                    if mft_ref == file_record_num and key_len >= 56:
                        fn_start = pos + 16
                        if fn_start + 56 <= indx_size:
                            struct.pack_into('<Q', indx, fn_start + 40, alloc_size)
                            struct.pack_into('<Q', indx, fn_start + 48, data_size)
                            found = True

                    pos += entry_len

                if found:
                    # Re-apply fixups (block-size-aware)
                    seq_val = struct.unpack('<H', indx[usa_off:usa_off + 2])[0]
                    seq_val = (seq_val + 1) & 0xFFFF
                    if seq_val == 0:
                        seq_val = 1
                    struct.pack_into('<H', indx, usa_off, seq_val)
                    for j in range(1, usa_cnt):
                        sec_end = j * 512 - 2
                        if sec_end + 2 <= indx_size:
                            struct.pack_into('<H', indx, usa_off + j * 2,
                                          struct.unpack('<H', indx[sec_end:sec_end + 2])[0])
                            struct.pack_into('<H', indx, sec_end, seq_val)

                    self.image[block_offset:block_offset + indx_size] = indx
                    return True

        return False

    def allocate_file_direct(self, rel_path: str) -> bool:
        """Allocate clusters for a sparse file directly (no ntfs-3g).

        This updates:
        1. Cluster bitmap (marks clusters as used)
        2. MFT data runs (points to allocated clusters)
        3. cluster_map (routes reads to ext4 file)

        No data is copied - reads will return ext4 content.
        Uses run-based allocation for efficiency with large files (40GB+).

        Returns True if successful.
        """
        if rel_path not in self.sparse_files:
            return False

        source_path, file_size, record_num = self.sparse_files[rel_path]

        # Calculate needed clusters
        needed_clusters = (file_size + self.cluster_size - 1) // self.cluster_size
        if needed_clusters == 0:
            return True  # Empty file, nothing to allocate

        # Calculate max runs that fit in this MFT record
        max_runs = self._max_data_runs_in_record(record_num)

        log(f"  Direct alloc: {rel_path} ({needed_clusters} clusters, max {max_runs} runs)")

        # Find free clusters as runs (not individual list)
        runs = self._find_free_cluster_runs(needed_clusters, max_runs)
        if not runs:
            log(f"  ERROR: Not enough free clusters for {rel_path}")
            return False

        log(f"  Found {len(runs)} runs for {needed_clusters} clusters")

        # Mark clusters as used in bitmap (bulk operation)
        self._mark_cluster_runs_used(runs)

        # Convert (start, count) runs to data_runs format (count, start)
        data_runs = [(count, start) for start, count in runs]

        # Update MFT record with new data runs
        if not self._update_mft_data_runs(record_num, data_runs, file_size):
            # Rollback bitmap changes
            self._mark_cluster_runs_free(runs)
            log(f"  ERROR: Failed to update MFT for {rel_path}")
            return False

        # Clear the SPARSE flag from $STANDARD_INFORMATION.
        # ntfs-3g treats files with SPARSE flag differently and may fail
        # to read files whose data runs were replaced from sparse to normal.
        self._clear_stdinfo_sparse_flag(record_num)

        # Update $FILE_NAME sizes and parent $I30 so Windows sees correct file size.
        # allocate_file_direct() only updates $DATA; without this Windows sees
        # $FILE_NAME.real_size=0 vs $DATA size=4GB and marks file as corrupted.
        _fn_rec_off = self._rec_offset(record_num)
        _fn_rec = self._undo_fixups(bytearray(
            self.image[_fn_rec_off:_fn_rec_off + MFT_RECORD_SIZE]
        ))
        _, _parent_ref_raw = self._extract_filename_and_parent(_fn_rec)
        if _parent_ref_raw:
            _parent_rec_num = _parent_ref_raw & 0xFFFFFFFFFFFF
            _alloc_sz = sum(count for count, _ in data_runs) * self.cluster_size
            self._update_filename_sizes(record_num, _alloc_sz, file_size)
            self._update_i30_entry(_parent_rec_num, record_num, _alloc_sz, file_size)
            log(f"  Updated $FILE_NAME+$I30 for record {record_num} size={file_size}")

        # Update _direct_run_map to route these clusters to ext4 file.
        # Run-based rather than per-cluster to avoid O(n_clusters) cost for large files.
        first_cluster = runs[0][0]
        last_cluster = runs[-1][0] + runs[-1][1] - 1
        log(f"  Mapping clusters {first_cluster}-{last_cluster} to {os.path.basename(source_path)}")
        file_offset = 0
        for start, count in runs:
            bisect.insort(self._direct_run_map, (start, start + count, source_path, file_offset))
            file_offset += count * self.cluster_size
            self._alloc_watermark = max(self._alloc_watermark, start + count)

        # Ensure source_to_clusters has an entry (empty set; runs are in _direct_run_map)
        if source_path not in self.source_to_clusters:
            self.source_to_clusters[source_path] = set()

        # Remove from sparse_files and sparse_file_clusters
        old_sparse_clusters = [c for c, p in self.sparse_file_clusters.items() if p == rel_path]
        for c in old_sparse_clusters:
            del self.sparse_file_clusters[c]
        del self.sparse_files[rel_path]

        # Track allocated runs for deallocation (store runs, not individual clusters)
        if not hasattr(self, '_direct_allocated'):
            self._direct_allocated = {}
        self._direct_allocated[rel_path] = (source_path, file_size, record_num, runs)
        self._direct_allocated_records.add(record_num)

        log(f"  Direct alloc complete: {rel_path}")
        return True

    def _is_directly_allocated(self, record_num: int) -> bool:
        """Check if an MFT record belongs to a directly-allocated file."""
        return record_num in self._direct_allocated_records

    def _clear_stdinfo_sparse_flag(self, record_num: int):
        """Clear the SPARSE flag (0x0200) from $STANDARD_INFORMATION.

        When ntfs-3g creates a sparse file, it sets FILE_ATTRIBUTE_SPARSE_FILE
        in $STANDARD_INFORMATION. After we replace the data runs with normal
        (non-sparse) runs, ntfs-3g may still treat the file as sparse and
        fail reads. Clearing this flag fixes that.
        """
        record_offset = self._rec_offset(record_num)
        if record_offset is None:
            return
        record = self._undo_fixups(bytearray(
            self.image[record_offset:record_offset + MFT_RECORD_SIZE]
        ))

        if record[:4] != b'FILE':
            return

        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break
            attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            if attr_type == 0x10:  # $STANDARD_INFORMATION
                val_off = struct.unpack('<H', record[off + 20:off + 22])[0]
                flags_off = off + val_off + 32
                if flags_off + 4 <= MFT_RECORD_SIZE:
                    old_flags = struct.unpack('<I', record[flags_off:flags_off + 4])[0]
                    if old_flags & 0x0200:
                        new_flags = old_flags & ~0x0200
                        # Write directly to image (flags are in first 512 bytes,
                        # so fixups don't affect this offset)
                        abs_offset = record_offset + flags_off
                        self.image[abs_offset:abs_offset + 4] = struct.pack('<I', new_flags)
                return

            off += attr_len

    def deallocate_file_direct(self, rel_path: str) -> bool:
        """Deallocate clusters for a file (reverse of allocate_file_direct).

        Restores the file to sparse state.
        """
        if not hasattr(self, '_direct_allocated'):
            return False

        if rel_path not in self._direct_allocated:
            return False

        source_path, file_size, record_num, runs = self._direct_allocated[rel_path]

        log(f"  Direct dealloc: {rel_path}")

        # Remove cluster mappings from _direct_run_map (O(runs), not O(clusters))
        run_starts = {start for start, count in runs}
        self._direct_run_map = [r for r in self._direct_run_map if r[0] not in run_starts]
        # Also clean up per-cluster entries that may exist from a prior rescan.
        # Only do this for small files; large files won't have cluster_map entries.
        total_clusters = sum(count for _, count in runs)
        if total_clusters <= RUN_MAP_THRESHOLD:
            for start, count in runs:
                for i in range(count):
                    self.cluster_map.pop(start + i, None)

        # Mark clusters as free in bitmap
        self._mark_cluster_runs_free(runs)

        # Restore MFT to sparse (single cluster at end for the \x00 byte)
        # This creates a sparse run followed by one allocated cluster
        # Use last cluster from last run
        last_run_start, last_run_count = runs[-1]
        last_cluster = last_run_start + last_run_count - 1
        # Sparse data runs: (cluster_count-1, None) for sparse hole, (1, last_cluster) for allocated
        sparse_runs = []
        needed_clusters = (file_size + self.cluster_size - 1) // self.cluster_size
        if needed_clusters > 1:
            sparse_runs.append((needed_clusters - 1, None))  # Sparse run (no physical clusters)
        sparse_runs.append((1, last_cluster))  # Keep one cluster allocated

        # Mark the last cluster as used
        self._mark_clusters_used([last_cluster])

        self._update_mft_data_runs(record_num, sparse_runs, file_size)

        # Re-add to sparse_files
        self.sparse_files[rel_path] = (source_path, file_size, record_num)
        self.sparse_file_clusters[last_cluster] = rel_path

        # Remove from direct_allocated
        del self._direct_allocated[rel_path]
        self._direct_allocated_records.discard(record_num)

        log(f"  Direct dealloc complete: {rel_path}")
        return True

    # =========================================================================
    # MFT scanning
    # =========================================================================

    def _scan_mft(self):
        """Scan MFT to find directories and files."""
        self.cluster_map.clear()
        self.mft_record_to_source.clear()
        self.source_to_clusters.clear()
        self.mft_record_to_dir.clear()
        self.resident_file_data.clear()
        self._direct_run_map.clear()

        # First pass: find all directories
        for record_num in range(self._mft_total_records):
            offset = self._rec_offset(record_num)
            if offset is None or offset + MFT_RECORD_SIZE > len(self.image):
                continue
            record = self.image[offset:offset + MFT_RECORD_SIZE]
            sig = record[0:4]
            if sig != b'FILE':
                if sig == b'BAAD':
                    # Zero out BAAD records so Windows doesn't see corrupt fixups
                    self.image[offset:offset + MFT_RECORD_SIZE] = b'\x00' * MFT_RECORD_SIZE
                    log(f"  Zeroed BAAD record {record_num}")
                continue

            record = self._undo_fixups(bytearray(record))
            flags = struct.unpack('<H', record[22:24])[0]

            if flags & 0x01 and flags & 0x02:  # In-use directory
                self._process_directory_record(record, record_num)

        # Second pass: find all files (both resident and non-resident)
        for record_num in range(self._mft_total_records):
            offset = self._rec_offset(record_num)
            if offset is None or offset + MFT_RECORD_SIZE > len(self.image):
                continue
            record = self.image[offset:offset + MFT_RECORD_SIZE]
            sig = record[0:4]
            if sig != b'FILE':
                if sig == b'BAAD':
                    self.image[offset:offset + MFT_RECORD_SIZE] = b'\x00' * MFT_RECORD_SIZE
                    log(f"  Zeroed BAAD record {record_num}")
                continue

            record = self._undo_fixups(bytearray(record))
            flags = struct.unpack('<H', record[22:24])[0]

            if flags & 0x01 and not (flags & 0x02):  # In-use file
                self._process_file_record(record, record_num)

    def _get_mft_record_count(self) -> int:
        """Get total MFT record count from $MFT record 0 $DATA attribute."""
        record0 = bytearray(self.image[self.mft_offset:self.mft_offset + MFT_RECORD_SIZE])
        record0 = self._undo_fixups_raw(record0)
        attr_offset = struct.unpack('<H', record0[20:22])[0]
        while attr_offset < MFT_RECORD_SIZE - 4:
            attr_type = struct.unpack('<I', record0[attr_offset:attr_offset + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break
            attr_len = struct.unpack('<I', record0[attr_offset + 4:attr_offset + 8])[0]
            if attr_len == 0:
                break
            if attr_type == 0x80:  # $DATA
                non_resident = record0[attr_offset + 8]
                if non_resident:
                    data_size = struct.unpack('<Q', record0[attr_offset + 48:attr_offset + 56])[0]
                    count = data_size // MFT_RECORD_SIZE
                    log(f"MFT has {count} records ({data_size} bytes)")
                    return count
            attr_offset += attr_len
        # Fallback: estimate from image size
        count = (len(self.image) - self.mft_offset) // MFT_RECORD_SIZE
        log(f"MFT record count fallback: {count}")
        return count

    def _undo_fixups_raw(self, record: bytearray) -> bytearray:
        """Undo USA fixups without logging (for bootstrap use)."""
        usa_offset = struct.unpack('<H', record[4:6])[0]
        usa_count = struct.unpack('<H', record[6:8])[0]
        for i in range(1, usa_count):
            sector_end = i * 512 - 2
            if (usa_offset + i * 2 + 2 <= MFT_RECORD_SIZE and
                    sector_end + 2 <= MFT_RECORD_SIZE):
                original = struct.unpack('<H',
                    record[usa_offset + i * 2:usa_offset + i * 2 + 2])[0]
                struct.pack_into('<H', record, sector_end, original)
        return record

    def _undo_fixups(self, record: bytearray) -> bytearray:
        """Undo USA fixups in an MFT record."""
        usa_offset = struct.unpack('<H', record[4:6])[0]
        usa_count = struct.unpack('<H', record[6:8])[0]
        for i in range(1, usa_count):
            sector_end = i * 512 - 2
            if (usa_offset + i * 2 + 2 <= MFT_RECORD_SIZE and
                    sector_end + 2 <= MFT_RECORD_SIZE):
                original = struct.unpack('<H',
                    record[usa_offset + i * 2:usa_offset + i * 2 + 2])[0]
                struct.pack_into('<H', record, sector_end, original)
        return record

    def _process_directory_record(self, record: bytearray, record_num: int):
        """Process a directory MFT record."""
        filename, parent_ref = self._extract_filename_and_parent(record)
        if not filename:
            return

        if filename.startswith('$'):
            return

        if record_num == 5:
            self.mft_record_to_dir[5] = ''
            self.path_to_mft_record[''] = 5
            return

        parent_record = parent_ref & 0xFFFFFFFFFFFF
        if parent_record == 5:
            dir_path = filename
        elif parent_record in self.mft_record_to_dir:
            parent_path = self.mft_record_to_dir[parent_record]
            dir_path = os.path.join(parent_path, filename) if parent_path else filename
        else:
            dir_path = filename

        self.mft_record_to_dir[record_num] = dir_path
        self.path_to_mft_record[dir_path] = record_num
        seq = struct.unpack('<H', record[16:18])[0]
        self._dir_mft_seq[record_num] = seq

    def _process_file_record(self, record: bytearray, record_num: int):
        """Process a file MFT record - handles both resident and non-resident files."""
        filename, parent_ref = self._extract_filename_and_parent(record)
        if not filename:
            return

        # Skip system files
        if filename.startswith('$'):
            return

        # Determine path using directory mapping
        parent_record = parent_ref & 0xFFFFFFFFFFFF
        if parent_record == 5:
            rel_path = filename
        elif parent_record in self.mft_record_to_dir:
            parent_path = self.mft_record_to_dir[parent_record]
            rel_path = os.path.join(parent_path, filename) if parent_path else filename
        else:
            rel_path = filename

        source_path = self._resolve_source_path(rel_path)

        # Find source file
        if not os.path.isfile(source_path):
            found = self._find_source_file(filename)
            if not found:
                return
            source_path = found

        # Check for non-resident data (clusters)
        data_runs = self._extract_data_runs(record)
        if data_runs:
            # Count only real clusters (lcn != -1), not sparse holes
            real_cluster_count = sum(count for lcn, count in data_runs if lcn != -1)
            has_sparse_runs = any(lcn == -1 for lcn, _ in data_runs)

            # Check if this is a sparse file (allocated clusters < expected clusters)
            try:
                file_size = os.path.getsize(source_path)
                expected_clusters = (file_size + self.cluster_size - 1) // self.cluster_size
                # Sparse if: has any sparse holes, OR real clusters < half expected
                is_sparse = has_sparse_runs or real_cluster_count < expected_clusters // 2
            except OSError:
                is_sparse = False

            if is_sparse:
                # This is a sparse file - track it but don't map the minimal clusters
                self.sparse_files[rel_path] = (source_path, file_size, record_num)
                self.mft_record_to_source[record_num] = source_path
                # Record the allocated clusters so we can detect reads to them
                for start_cluster, count in data_runs:
                    if start_cluster > 0:  # Skip sparse runs
                        for c in range(start_cluster, start_cluster + count):
                            self.sparse_file_clusters[c] = rel_path
                log(f"  Sparse file: {rel_path} ({real_cluster_count}/{expected_clusters} clusters, {file_size} bytes)")
            else:
                # Fully allocated file - map clusters
                if record_num not in self.mft_record_to_source:
                    log(f"  Mapping new file: {rel_path} (record {record_num}, {real_cluster_count} clusters)")
                self._map_clusters(data_runs, source_path)
                self.mft_record_to_source[record_num] = source_path
                # Remove from sparse tracking if it was there
                self.sparse_files.pop(rel_path, None)
        else:
            # Check for resident data (stored in MFT record)
            resident_loc = self._find_resident_data_location(record, record_num)
            if resident_loc:
                # If the source file is larger than the available resident space,
                # the MFT has an empty/stale resident DATA attribute left over from
                # a previous session where allocate_file_direct() was undone by
                # Windows journal replay.  Treat this as sparse so pre-allocation
                # will install proper non-resident data runs.
                try:
                    source_size = os.path.getsize(source_path)
                except OSError:
                    source_size = 0
                if source_size > resident_loc[2]:  # too large to be truly resident
                    self.sparse_files[rel_path] = (source_path, source_size, record_num)
                    self.mft_record_to_source[record_num] = source_path
                else:
                    self.resident_file_data[record_num] = {
                        'source_path': source_path,
                        'val_len_abs': resident_loc[0],  # abs offset of value_length field
                        'data_abs': resident_loc[1],       # abs offset of data start
                        'available': resident_loc[2],      # max bytes available for data
                    }
                    self.mft_record_to_source[record_num] = source_path
            else:
                # No data runs and no resident data - check if it's a large sparse file
                try:
                    file_size = os.path.getsize(source_path)
                    if file_size > 700:  # Large file with no allocation = sparse
                        self.sparse_files[rel_path] = (source_path, file_size, record_num)
                        self.mft_record_to_source[record_num] = source_path
                except OSError:
                    pass

    # =========================================================================
    # Attribute parsing
    # =========================================================================

    def _extract_filename_and_parent(self, record: bytearray) -> Tuple[Optional[str], int]:
        """Extract filename and parent reference from MFT record."""
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr
        filename = None
        parent_ref = 0

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            name_len = record[off + 9]
            attr_name = ''
            if name_len > 0:
                name_offset = struct.unpack('<H', record[off + 10:off + 12])[0]
                attr_name = record[off + name_offset:off + name_offset + name_len * 2].decode(
                    'utf-16-le', errors='ignore')

            if attr_type == 0x30 and not attr_name:  # $FILE_NAME
                val_len = struct.unpack('<I', record[off + 16:off + 20])[0]
                val_off = struct.unpack('<H', record[off + 20:off + 22])[0]
                if off + val_off + val_len > MFT_RECORD_SIZE:
                    break
                fn_data = record[off + val_off:off + val_off + val_len]
                if len(fn_data) >= 66:
                    parent_ref = struct.unpack('<Q', fn_data[0:8])[0]
                    fn_len = fn_data[64]
                    fn_namespace = fn_data[65]
                    if fn_namespace in (1, 3) or filename is None:
                        filename = fn_data[66:66 + fn_len * 2].decode(
                            'utf-16-le', errors='ignore')

            off += attr_len

        return filename, parent_ref

    def _extract_data_runs(self, record: bytearray) -> Optional[List[Tuple[int, int]]]:
        """Extract data runs from MFT record's $DATA attribute."""
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            name_len = record[off + 9]
            attr_name = ''
            if name_len > 0:
                name_offset = struct.unpack('<H', record[off + 10:off + 12])[0]
                attr_name = record[off + name_offset:off + name_offset + name_len * 2].decode(
                    'utf-16-le', errors='ignore')

            if attr_type == 0x80 and not attr_name:  # $DATA (unnamed)
                non_res = record[off + 8]
                if non_res:
                    runs_off = struct.unpack('<H', record[off + 32:off + 34])[0]
                    real_size = struct.unpack('<Q', record[off + 48:off + 56])[0]
                    runs = record[off + runs_off:off + attr_len]
                    return self._parse_data_runs(runs, real_size)
                break

            off += attr_len

        return None

    def _extract_file_size(self, record: bytearray) -> Optional[int]:
        """Extract file size from $DATA attribute (resident or non-resident)."""
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            name_len = record[off + 9]
            attr_name = ''
            if name_len > 0:
                name_offset = struct.unpack('<H', record[off + 10:off + 12])[0]
                attr_name = record[off + name_offset:off + name_offset + name_len * 2].decode(
                    'utf-16-le', errors='ignore')

            if attr_type == 0x80 and not attr_name:  # $DATA (unnamed)
                non_res = record[off + 8]
                if non_res:
                    # Non-resident: real size at offset 48
                    return struct.unpack('<Q', record[off + 48:off + 56])[0]
                else:
                    # Resident: content length at offset 16
                    return struct.unpack('<I', record[off + 16:off + 20])[0]

            off += attr_len

        return None

    def _extract_resident_data(self, record: bytearray) -> Optional[bytes]:
        """Extract resident data from $DATA attribute."""
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            name_len = record[off + 9]
            attr_name = ''
            if name_len > 0:
                name_offset = struct.unpack('<H', record[off + 10:off + 12])[0]
                attr_name = record[off + name_offset:off + name_offset + name_len * 2].decode(
                    'utf-16-le', errors='ignore')

            if attr_type == 0x80 and not attr_name:  # $DATA (unnamed)
                non_res = record[off + 8]
                if not non_res:  # Resident
                    val_len = struct.unpack('<I', record[off + 16:off + 20])[0]
                    val_off = struct.unpack('<H', record[off + 20:off + 22])[0]
                    return bytes(record[off + val_off:off + val_off + val_len])
                break

            off += attr_len

        return None

    def _find_resident_data_location(self, record: bytearray, record_num: int) -> Optional[Tuple[int, int, int]]:
        """Find the byte location of resident $DATA in an MFT record.

        Returns (val_len_abs_offset, data_abs_offset, available_space) or None.
        """
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr
        record_abs = self._rec_offset(record_num)
        if record_abs is None:
            return None

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            name_len = record[off + 9]
            attr_name = ''
            if name_len > 0:
                name_offset = struct.unpack('<H', record[off + 10:off + 12])[0]
                attr_name = record[off + name_offset:off + name_offset + name_len * 2].decode(
                    'utf-16-le', errors='ignore')

            if attr_type == 0x80 and not attr_name:  # $DATA
                non_res = record[off + 8]
                if not non_res:  # Resident
                    val_off = struct.unpack('<H', record[off + 20:off + 22])[0]
                    available = attr_len - val_off  # max space for data

                    val_len_abs = record_abs + off + 16   # value_length field
                    data_abs = record_abs + off + val_off  # data start

                    # Safety: ensure data doesn't span sector boundary fixup positions
                    # (510-511 and 1022-1023 within the record)
                    # For typical small files, data is well within first 500 bytes
                    return (val_len_abs, data_abs, available)
                break

            off += attr_len

        return None

    def _parse_data_runs(self, runs: bytes, real_size: int) -> List[Tuple[int, int]]:
        """Parse data runs into list of (cluster, count) tuples.

        Sparse runs (holes) are included with cluster=-1 so that
        _map_clusters can correctly calculate file offsets.
        """
        result = []
        pos = 0
        current_lcn = 0

        while pos < len(runs):
            header = runs[pos]
            if header == 0:
                break

            len_size = header & 0x0F
            off_size = (header >> 4) & 0x0F
            pos += 1

            if pos + len_size + off_size > len(runs):
                break

            run_length = int.from_bytes(runs[pos:pos + len_size], 'little')
            pos += len_size

            if off_size > 0:
                run_offset = int.from_bytes(runs[pos:pos + off_size], 'little', signed=True)
                pos += off_size
                current_lcn += run_offset
                result.append((current_lcn, run_length))
            else:
                # Sparse run (hole) - no offset field means no physical allocation
                result.append((-1, run_length))

        return result

    def _resolve_root_path(self, filename: str) -> str:
        """Resolve a root-level filename to a base directory.

        Known entries (those present in the source directory at startup)
        resolve to source_dir. Unknown entries (e.g. System Volume
        Information, Windows SID folders) resolve to overflow_dir.
        """
        if filename in self.known_root_entries:
            return os.path.join(self.source_dir, filename)
        return os.path.join(self.overflow_dir, filename)

    def _resolve_source_path(self, rel_path: str) -> str:
        """Resolve a relative path to a full source path.

        Checks the top-level directory component to determine whether
        the path belongs in source_dir or overflow_dir.
        """
        top_level = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path
        if top_level in self.known_root_entries:
            return os.path.join(self.source_dir, rel_path)
        return os.path.join(self.overflow_dir, rel_path)

    def _validate_path(self, source_path: str, context: str = '') -> bool:
        """Validate that a resolved path stays within allowed directories.

        Returns True if the path is safe, False if it escapes the allowed
        directories (path traversal) or contains null bytes.
        """
        # Reject null bytes in the path
        if '\x00' in source_path:
            log(f"  PATH REJECTED (null byte){' in ' + context if context else ''}: {source_path!r}")
            return False

        resolved = os.path.realpath(source_path)
        source_real = os.path.realpath(self.source_dir)
        overflow_real = os.path.realpath(self.overflow_dir)

        if resolved.startswith(source_real + os.sep) or resolved == source_real:
            return True
        if overflow_real != source_real:
            if resolved.startswith(overflow_real + os.sep) or resolved == overflow_real:
                return True

        log(f"  PATH REJECTED (traversal){' in ' + context if context else ''}: {source_path} -> {resolved}")
        return False

    def _get_rel_path(self, source_path: str) -> str:
        """Get the relative path from a full source path.

        Handles paths in either source_dir or overflow_dir.
        """
        if self.overflow_dir != self.source_dir and source_path.startswith(self.overflow_dir + os.sep):
            return os.path.relpath(source_path, self.overflow_dir)
        return os.path.relpath(source_path, self.source_dir)

    def _find_source_file(self, filename: str) -> Optional[str]:
        """Find matching file in source directory."""
        path = os.path.join(self.source_dir, filename)
        if os.path.isfile(path):
            return path

        for root, dirs, files in os.walk(self.source_dir, followlinks=True):
            if filename in files:
                return os.path.join(root, filename)

        return None

    def _map_clusters(self, data_runs: List[Tuple[int, int]], source_path: str):
        """Map clusters from data runs to source file offsets.

        Sparse runs (lcn=-1) advance file_offset but don't create mappings,
        so real clusters get the correct offset into the source file.

        Large files (>RUN_MAP_THRESHOLD clusters) use _direct_run_map instead
        of per-cluster cluster_map entries to avoid O(n) memory/time.
        """
        total_real = sum(c for lc, c in data_runs if lc != -1)

        if source_path not in self.source_to_clusters:
            self.source_to_clusters[source_path] = set()

        if total_real > RUN_MAP_THRESHOLD:
            # Large file: store runs, not individual clusters
            file_offset = 0
            new_entries = []
            for lcn, count in data_runs:
                if lcn == -1:
                    file_offset += count * self.cluster_size
                    continue
                new_entries.append((lcn, lcn + count, source_path, file_offset))
                file_offset += count * self.cluster_size
                self._alloc_watermark = max(self._alloc_watermark, lcn + count)
            # Insert maintaining sort order
            for entry in new_entries:
                bisect.insort(self._direct_run_map, entry)
            return

        file_offset = 0
        for lcn, count in data_runs:
            if lcn == -1:
                # Sparse run - advance offset but don't map
                file_offset += count * self.cluster_size
                continue
            self._alloc_watermark = max(self._alloc_watermark, lcn + count)
            for i in range(count):
                cluster = lcn + i
                self.cluster_map[cluster] = (source_path, file_offset)
                self.source_to_clusters[source_path].add(cluster)
                file_offset += self.cluster_size

    def _build_path_mappings(self):
        """Build reverse path -> MFT record mappings."""
        self.path_to_mft_record.clear()
        self.dir_children.clear()

        for record_num, path in self.mft_record_to_source.items():
            rel_path = self._get_rel_path(path)
            self.path_to_mft_record[rel_path] = record_num

        for record_num, rel_path in self.mft_record_to_dir.items():
            if rel_path:
                self.path_to_mft_record[rel_path] = record_num

        # Build directory children
        for record_num, source_path in self.mft_record_to_source.items():
            rel_path = self._get_rel_path(source_path)
            parent_path = os.path.dirname(rel_path)
            parent_record = self._get_parent_record(parent_path)
            if parent_record not in self.dir_children:
                self.dir_children[parent_record] = set()
            self.dir_children[parent_record].add(record_num)

        for record_num, rel_path in self.mft_record_to_dir.items():
            if rel_path:
                parent_path = os.path.dirname(rel_path)
                parent_record = self._get_parent_record(parent_path)
                if parent_record not in self.dir_children:
                    self.dir_children[parent_record] = set()
                self.dir_children[parent_record].add(record_num)

    def _get_parent_record(self, parent_rel_path: str) -> int:
        """Get MFT record number for a parent directory path."""
        if not parent_rel_path or parent_rel_path == '.':
            return 5
        return self.path_to_mft_record.get(parent_rel_path, 5)

    # =========================================================================
    # Resident file injection (ext4 content into MFT reads)
    # =========================================================================

    def _inject_resident_data(self, result: bytearray, read_offset: int, read_length: int):
        """Replace resident file data in MFT reads with current ext4 content.

        When ntfs-3g reads MFT records, resident file data is served from
        the image. This method patches the returned data with current ext4
        content, so content changes in ext4 are immediately visible.
        """
        if not self.resident_file_data:
            return

        read_end = read_offset + read_length

        for record_num, info in self.resident_file_data.items():
            source_path = info['source_path']
            val_len_abs = info['val_len_abs']
            data_abs = info['data_abs']
            available = info['available']

            # Quick check: does this read overlap with this record's data area?
            data_end = data_abs + available
            if read_end <= val_len_abs or read_offset >= data_end:
                continue

            # Read current ext4 content (cap to available space in MFT record)
            try:
                with open(source_path, 'rb') as f:
                    ext4_data = f.read(available)
            except OSError:
                continue

            ext4_size = len(ext4_data)
            inject_size = min(ext4_size, available)

            # Patch value_length field (4 bytes LE at val_len_abs)
            if val_len_abs >= read_offset and val_len_abs + 4 <= read_end:
                dst = val_len_abs - read_offset
                struct.pack_into('<I', result, dst, inject_size)

            # Patch data bytes
            if data_abs < read_end and data_abs + available > read_offset:
                # Calculate overlap region
                patch_start = max(data_abs, read_offset)
                patch_end = min(data_abs + inject_size, read_end)

                if patch_start < patch_end:
                    src_off = patch_start - data_abs
                    dst_off = patch_start - read_offset
                    patch_len = patch_end - patch_start
                    result[dst_off:dst_off + patch_len] = ext4_data[src_off:src_off + patch_len]

                # Zero out remaining available space after ext4 data
                if ext4_size < available:
                    zero_start = max(data_abs + inject_size, read_offset)
                    zero_end = min(data_abs + available, read_end)
                    if zero_start < zero_end:
                        dst_off = zero_start - read_offset
                        zero_len = zero_end - zero_start
                        result[dst_off:dst_off + zero_len] = b'\x00' * zero_len

            # Restore NTFS USA fixup bytes that may have been overwritten.
            # Each 512-byte sector in the MFT record must end with the USA
            # check value (USA[0]) for ntfs-3g to accept the record.
            # _inject_resident_data can overwrite those bytes (at record
            # offsets 510-511 and 1022-1023) with ext4 file content.
            record_abs = self._rec_offset(record_num)
            if record_abs is None:
                continue
            try:
                usa_off = struct.unpack('<H', bytes(self.image[record_abs + 4: record_abs + 6]))[0]
                check_val = bytes(self.image[record_abs + usa_off: record_abs + usa_off + 2])
            except Exception:
                continue
            for sector_end in (512, 1024):
                fixup_abs = record_abs + sector_end - 2  # 510 or 1022
                if read_offset <= fixup_abs and fixup_abs + 2 <= read_end:
                    buf_pos = fixup_abs - read_offset
                    result[buf_pos:buf_pos + 2] = check_val

    # =========================================================================
    # Virtual file injection (ext4→NTFS live sync)
    # =========================================================================

    def _inject_virtual_mft_records(self, result: bytearray, offset: int, length: int):
        """Inject virtual MFT records into read result.

        When Windows reads the MFT, we inject synthesized FILE records
        for virtual files (those that exist in ext4 but not in NTFS image).
        """
        if not self.virtual_file_manager:
            return

        vfm = self.virtual_file_manager
        read_end = offset + length

        # Check each virtual record to see if it falls in this read
        for record_num in list(vfm.mft_to_virtual.keys()):
            record_abs_offset = self._rec_offset(record_num)
            if record_abs_offset is None:
                continue
            # Check if this record overlaps with the read range
            if record_abs_offset + MFT_RECORD_SIZE <= offset or record_abs_offset >= read_end:
                continue
            record_data = vfm.get_virtual_mft_record(record_num)
            if record_data:
                if record_abs_offset >= offset and record_abs_offset + MFT_RECORD_SIZE <= read_end:
                    # Entire record fits in this read
                    dst = record_abs_offset - offset
                    result[dst:dst + MFT_RECORD_SIZE] = record_data
                else:
                    # Partial overlap - handle carefully
                    overlap_start = max(record_abs_offset, offset)
                    overlap_end = min(record_abs_offset + MFT_RECORD_SIZE, read_end)
                    if overlap_start < overlap_end:
                        src_off = overlap_start - record_abs_offset
                        dst_off = overlap_start - offset
                        patch_len = overlap_end - overlap_start
                        result[dst_off:dst_off + patch_len] = record_data[src_off:src_off + patch_len]

    def _inject_virtual_dir_entries(self, result: bytearray, offset: int, length: int):
        """Virtualize directory listings to include virtual files.

        This implements full directory virtualization:
        1. Parse all real entries from INDEX_ROOT and INDEX_ALLOCATION
        2. Merge with virtual entries
        3. Synthesize new INDEX structures
        4. Return synthesized data

        Works for any directory size, including those with B+ tree indexes.
        """
        if not self.virtual_file_manager:
            return

        vfm = self.virtual_file_manager

        # Check if this read includes any directory MFT records
        if self.is_mft_region(offset, length):
            self._virtualize_dir_mft_records(result, offset, length)

        # Check if this read includes any virtual INDX clusters
        self._inject_virtual_indx_clusters(result, offset, length)

    def _virtualize_dir_mft_records(self, result: bytearray, offset: int, length: int):
        """Virtualize directory MFT records in the read result."""
        if not self.virtual_file_manager:
            return

        vfm = self.virtual_file_manager

        read_end = offset + length

        # Log which directories we know about (first call only)
        if not hasattr(self, '_logged_dirs'):
            self._logged_dirs = True
            log(f"Known dirs: {list(self.mft_record_to_dir.items())}")

        # Check each directory record
        for record_num, dir_path in list(self.mft_record_to_dir.items()):
            record_abs_offset = self._rec_offset(record_num)
            if record_abs_offset is None:
                continue
            # Check if this record overlaps with the read range
            if record_abs_offset + MFT_RECORD_SIZE <= offset or record_abs_offset >= read_end:
                continue

            # Get virtual children for this directory
            virtual_children = vfm.get_virtual_children(record_num)
            if virtual_children:
                log(f"Dir {record_num} ({dir_path}) has {len(virtual_children)} virtual children: {[c.rel_path for c in virtual_children]}")

            if not virtual_children:
                continue

            # Virtualize this directory
            if record_abs_offset >= offset and record_abs_offset + MFT_RECORD_SIZE <= offset + length:
                dst = record_abs_offset - offset
                record_data = bytearray(result[dst:dst + MFT_RECORD_SIZE])

                # Build or update virtualized directory
                self._ensure_dir_virtualized(record_num, record_data, virtual_children)

                # Get the virtualized MFT record
                if record_num in self.virtualized_dirs:
                    virt_record = self._build_virtualized_mft_record(record_num, record_data)
                    if virt_record and len(virt_record) == MFT_RECORD_SIZE:
                        result[dst:dst + MFT_RECORD_SIZE] = virt_record
                    elif virt_record:
                        log(f"Virtualized record size mismatch: {len(virt_record)} != {MFT_RECORD_SIZE}")
                    else:
                        log(f"Failed to build virtualized record for dir {record_num}")

    def _inject_virtual_indx_clusters(self, result: bytearray, offset: int, length: int):
        """Inject synthesized INDX blocks for virtual clusters."""
        if not self.virtual_file_manager:
            return

        read_end = offset + length
        start_cluster = offset // self.cluster_size
        end_cluster = (offset + length + self.cluster_size - 1) // self.cluster_size

        for cluster in range(start_cluster, end_cluster):
            if cluster in self.virtual_indx_map:
                dir_record, block_idx = self.virtual_indx_map[cluster]
                if dir_record in self.virtualized_dirs:
                    vdir = self.virtualized_dirs[dir_record]
                    if 'indx_blocks' in vdir and block_idx < len(vdir['indx_blocks']):
                        indx_data = vdir['indx_blocks'][block_idx]
                        cluster_offset = cluster * self.cluster_size
                        if cluster_offset >= offset and cluster_offset + self.cluster_size <= read_end:
                            dst = cluster_offset - offset
                            result[dst:dst + len(indx_data)] = indx_data

    def _ensure_dir_virtualized(self, record_num: int, record_data: bytearray,
                                 virtual_children: list):
        """Ensure a directory is virtualized with current entries."""
        if not self.virtual_file_manager:
            return

        vfm = self.virtual_file_manager

        # Check if we need to rebuild (new virtual children or not yet built)
        current_virtual = set(c.rel_path for c in virtual_children)
        if record_num in self.virtualized_dirs:
            cached_virtual = self.virtualized_dirs[record_num].get('virtual_paths', set())
            if current_virtual == cached_virtual:
                return  # Already up to date

        # Parse all real entries from this directory
        real_entries = self._parse_all_dir_entries(record_num, record_data)

        # Build virtual index entries
        virtual_entries = []
        for child in virtual_children:
            entry_data = vfm.synthesize_index_entry(child)
            # Extract filename for sorting
            if len(entry_data) >= 82:
                name_len = entry_data[80]
                name_bytes = entry_data[82:82 + name_len * 2]
                try:
                    filename = name_bytes.decode('utf-16-le')
                except:
                    filename = ''
            else:
                filename = child.rel_path
            virtual_entries.append((filename.upper(), entry_data))

        # Merge entries sorted by filename (NTFS uses uppercase comparison)
        all_entries = []
        for filename, entry_data in real_entries:
            all_entries.append((filename.upper(), entry_data))
        all_entries.extend(virtual_entries)
        all_entries.sort(key=lambda x: x[0])

        # Extract original INDX clusters from MFT record (for real cluster interception)
        original_indx_clusters = self._extract_original_indx_clusters(record_data)

        # Synthesize INDEX structures, reusing original clusters if available
        self._synthesize_dir_index_inplace(record_num, record_data, all_entries, original_indx_clusters)

        # Store virtual paths for cache invalidation
        self.virtualized_dirs[record_num]['virtual_paths'] = current_virtual

    def _parse_all_dir_entries(self, record_num: int, record_data: bytearray) -> List[Tuple[str, bytes]]:
        """Parse all index entries from a directory (INDEX_ROOT + INDEX_ALLOCATION)."""
        entries = []

        # Undo fixups
        record = self._undo_fixups(bytearray(record_data))

        if record[0:4] != b'FILE':
            return entries

        # Find INDEX_ROOT and INDEX_ALLOCATION
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr
        index_root_off = None
        index_alloc_info = None

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            if attr_type == 0x90:  # INDEX_ROOT
                index_root_off = off
            elif attr_type == 0xA0:  # INDEX_ALLOCATION (non-resident)
                # Parse data runs to find INDX clusters
                if record[off + 8] == 1:  # Non-resident
                    run_off = struct.unpack('<H', record[off + 32:off + 34])[0]
                    alloc_size = struct.unpack('<Q', record[off + 40:off + 48])[0]
                    data_runs = self._parse_data_runs(record[off + run_off:off + attr_len], alloc_size)
                    index_alloc_info = {'data_runs': data_runs, 'size': alloc_size}

            off += attr_len

        # Parse INDEX_ROOT entries
        if index_root_off is not None:
            entries.extend(self._parse_index_root_entries(record, index_root_off))

        # Parse INDEX_ALLOCATION entries (INDX blocks)
        if index_alloc_info:
            entries.extend(self._parse_index_alloc_entries(index_alloc_info))

        return entries

    def _parse_index_root_entries(self, record: bytearray, attr_off: int) -> List[Tuple[str, bytes]]:
        """Parse entries from INDEX_ROOT attribute."""
        entries = []

        attr_len = struct.unpack('<I', record[attr_off + 4:attr_off + 8])[0]
        val_off = struct.unpack('<H', record[attr_off + 20:attr_off + 22])[0]

        idx_root_start = attr_off + val_off
        idx_header_start = idx_root_start + 16
        entries_off = struct.unpack('<I', record[idx_header_start:idx_header_start + 4])[0]

        entry_off = idx_header_start + entries_off
        while entry_off < attr_off + attr_len:
            if entry_off + 16 > MFT_RECORD_SIZE:
                break

            entry_len = struct.unpack('<H', record[entry_off + 8:entry_off + 10])[0]
            entry_flags = struct.unpack('<H', record[entry_off + 12:entry_off + 14])[0]

            if entry_flags & 0x02:  # LAST_ENTRY
                break

            if entry_len == 0 or entry_len > 512:
                break

            # Extract filename for sorting
            entry_data = bytes(record[entry_off:entry_off + entry_len])
            filename = self._extract_entry_filename(entry_data)
            entries.append((filename, entry_data))

            entry_off += entry_len

        return entries

    def _parse_index_alloc_entries(self, alloc_info: dict) -> List[Tuple[str, bytes]]:
        """Parse entries from INDEX_ALLOCATION (INDX blocks)."""
        entries = []
        data_runs = alloc_info['data_runs']

        # Read each INDX block
        current_vcn = 0
        for start_cluster, count in data_runs:
            if start_cluster == -1:  # Skip sparse runs
                current_vcn += count
                continue
            for i in range(count):
                cluster = start_cluster + i
                cluster_offset = cluster * self.cluster_size

                if cluster_offset + self.cluster_size <= len(self.image):
                    indx_data = bytearray(self.image[cluster_offset:cluster_offset + self.cluster_size])

                    # Check for INDX signature
                    if indx_data[0:4] == b'INDX':
                        # Undo fixups on INDX block
                        indx_data = self._undo_indx_fixups(indx_data)
                        entries.extend(self._parse_indx_block_entries(indx_data))

                current_vcn += 1

        return entries

    def _undo_indx_fixups(self, indx: bytearray) -> bytearray:
        """Undo USA fixups on an INDX block."""
        usa_offset = struct.unpack('<H', indx[4:6])[0]
        usa_count = struct.unpack('<H', indx[6:8])[0]

        for i in range(1, usa_count):
            sector_end = i * 512 - 2
            if usa_offset + i * 2 + 2 <= len(indx) and sector_end + 2 <= len(indx):
                original = struct.unpack('<H', indx[usa_offset + i * 2:usa_offset + i * 2 + 2])[0]
                struct.pack_into('<H', indx, sector_end, original)

        return indx

    def _parse_indx_block_entries(self, indx: bytearray) -> List[Tuple[str, bytes]]:
        """Parse entries from a single INDX block."""
        entries = []

        # INDX header: entries start at offset 24 + entries_offset
        entries_off = struct.unpack('<I', indx[24:28])[0]
        entry_off = 24 + entries_off

        while entry_off < len(indx) - 16:
            entry_len = struct.unpack('<H', indx[entry_off + 8:entry_off + 10])[0]
            entry_flags = struct.unpack('<H', indx[entry_off + 12:entry_off + 14])[0]

            if entry_flags & 0x02:  # LAST_ENTRY
                break

            if entry_len == 0 or entry_len > 512 or entry_off + entry_len > len(indx):
                break

            entry_data = bytes(indx[entry_off:entry_off + entry_len])
            filename = self._extract_entry_filename(entry_data)
            entries.append((filename, entry_data))

            entry_off += entry_len

        return entries

    def _extract_entry_filename(self, entry_data: bytes) -> str:
        """Extract filename from an index entry."""
        if len(entry_data) < 82:
            return ''

        name_len = entry_data[80]
        if len(entry_data) < 82 + name_len * 2:
            return ''

        name_bytes = entry_data[82:82 + name_len * 2]
        try:
            return name_bytes.decode('utf-16-le')
        except:
            return ''

    def _extract_original_indx_clusters(self, record_data: bytearray) -> List[int]:
        """Extract the original INDX cluster numbers from a directory MFT record.

        These are the real clusters that contain INDX blocks for the directory.
        We'll intercept reads to these clusters and return synthesized content.
        """
        clusters = []
        record = self._undo_fixups(bytearray(record_data))

        if record[0:4] != b'FILE':
            return clusters

        # Find INDEX_ALLOCATION attribute
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            if attr_type == 0xA0:  # INDEX_ALLOCATION (non-resident)
                if record[off + 8] == 1:  # Non-resident flag
                    run_off = struct.unpack('<H', record[off + 32:off + 34])[0]
                    alloc_size = struct.unpack('<Q', record[off + 40:off + 48])[0]
                    data_runs = self._parse_data_runs(record[off + run_off:off + attr_len], alloc_size)

                    # Flatten data runs to get all cluster numbers
                    for start_cluster, count in data_runs:
                        if start_cluster == -1:  # Skip sparse runs
                            continue
                        for i in range(count):
                            clusters.append(start_cluster + i)
                    break  # Found INDEX_ALLOCATION, done

            off += attr_len

        return clusters

    def _synthesize_dir_index_inplace(self, record_num: int, record_data: bytearray,
                                       all_entries: List[Tuple[str, bytes]],
                                       original_clusters: List[int]):
        """Synthesize INDEX structures for a virtualized directory.

        When original_clusters is provided and has enough clusters, we intercept reads
        to those real INDX clusters and return synthesized content. This preserves the
        original INDEX_ALLOCATION data runs in the MFT record, which is critical for
        ntfs-3g compatibility during mount validation.
        """
        # Clean up any previous virtualized_indx_clusters entries for this directory
        # This is needed when re-virtualizing after adding more files
        if record_num in self.virtualized_dirs:
            old_vdir = self.virtualized_dirs[record_num]
            for cluster in old_vdir.get('original_clusters', []):
                if cluster in self.virtualized_indx_clusters:
                    del self.virtualized_indx_clusters[cluster]
            # Also clean up old virtual_indx_map entries
            for cluster in old_vdir.get('virtual_indx_clusters', []):
                if cluster in self.virtual_indx_map:
                    del self.virtual_indx_map[cluster]

        INDX_BLOCK_SIZE = self.cluster_size  # Usually 4096
        MAX_INDEX_ROOT_ENTRIES_SIZE = 400  # Leave room for other attributes

        # Calculate total entries size
        total_entries_size = sum(len(e[1]) for e in all_entries)
        total_entries_size += 16  # End entry

        # Decide if we need INDEX_ALLOCATION
        if total_entries_size <= MAX_INDEX_ROOT_ENTRIES_SIZE:
            # Fits in INDEX_ROOT only
            log(f"Dir {record_num} virtualized inline: {len(all_entries)} entries, {total_entries_size} bytes")
            self.virtualized_dirs[record_num] = {
                'use_indx': False,
                'entries': all_entries,
                'indx_blocks': [],
                'virtual_indx_clusters': [],
                'original_clusters': []
            }
        else:
            # Need INDEX_ALLOCATION with INDX blocks
            # Build INDX blocks

            # Calculate actual overhead per INDX block:
            # - Index node header starts at offset 24
            # - USA at offset 40, takes (sectors + 1) * 2 bytes
            # - Entries start after USA, aligned to 8
            SECTOR_SIZE = 512
            usa_offset = 40
            usa_count = INDX_BLOCK_SIZE // SECTOR_SIZE + 1  # 9 for 4KB
            usa_size = usa_count * 2  # 18 bytes
            entries_start = (usa_offset + usa_size + 7) & ~7  # 64 for 4KB
            indx_overhead = entries_start + 16  # +16 for end entry reserve

            indx_blocks = []
            current_block_entries = []
            current_block_size = indx_overhead

            for filename, entry_data in all_entries:
                entry_size = len(entry_data)

                # Check if entry fits in current block
                if current_block_size + entry_size > INDX_BLOCK_SIZE:
                    # Finalize current block and start new one
                    if current_block_entries:
                        indx_blocks.append(self._build_indx_block(current_block_entries, len(indx_blocks)))
                    current_block_entries = []
                    current_block_size = indx_overhead

                current_block_entries.append((filename, entry_data))
                current_block_size += entry_size

            # Finalize last block
            if current_block_entries:
                indx_blocks.append(self._build_indx_block(current_block_entries, len(indx_blocks)))

            # Decide whether to use original clusters (inplace) or allocate virtual clusters
            if original_clusters and len(original_clusters) >= len(indx_blocks):
                # Use original real clusters - intercept reads to them
                # This preserves the MFT INDEX_ALLOCATION data runs
                for i, indx_data in enumerate(indx_blocks):
                    cluster = original_clusters[i]
                    self.virtualized_indx_clusters[cluster] = (record_num, i, indx_data)

                log(f"Dir {record_num} virtualized inplace: {len(indx_blocks)} blocks at original clusters {original_clusters[:len(indx_blocks)]}")
                self.virtualized_dirs[record_num] = {
                    'use_indx': True,
                    'entries': all_entries,
                    'indx_blocks': indx_blocks,
                    'virtual_indx_clusters': [],  # Empty - not using virtual clusters
                    'original_clusters': original_clusters[:len(indx_blocks)]  # Track which real clusters we're intercepting
                }
            else:
                # Fallback: allocate virtual cluster numbers for INDX blocks
                # This approach modifies MFT data runs (may not be compatible with ntfs-3g)
                virtual_clusters = []
                for i in range(len(indx_blocks)):
                    vcluster = self.next_virtual_indx_cluster
                    self.next_virtual_indx_cluster += 1
                    virtual_clusters.append(vcluster)
                    self.virtual_indx_map[vcluster] = (record_num, i)

                log(f"Dir {record_num} virtualized with virtual clusters: {len(indx_blocks)} blocks at {virtual_clusters}")
                self.virtualized_dirs[record_num] = {
                    'use_indx': True,
                    'entries': all_entries,
                    'indx_blocks': indx_blocks,
                    'virtual_indx_clusters': virtual_clusters,
                    'original_clusters': []
                }

    def _build_indx_block(self, entries: List[Tuple[str, bytes]], block_num: int) -> bytes:
        """Build an INDX block from entries.

        INDX block layout (for 4KB cluster):
        - 0x00-0x03: "INDX" signature
        - 0x04-0x05: USA offset (40)
        - 0x06-0x07: USA count (9 for 4KB = 8 sectors + 1)
        - 0x10-0x17: VCN of this block
        - 0x18-0x1B: Index node header: entries offset (relative to 0x18)
        - 0x1C-0x1F: Index node header: total size of entries
        - 0x20-0x23: Index node header: allocated size
        - 0x24-0x27: Index node header: flags
        - 0x28-0x39: USA (18 bytes for 4KB)
        - 0x40+: Index entries (aligned to 8)
        """
        INDX_BLOCK_SIZE = self.cluster_size
        SECTOR_SIZE = 512

        indx = bytearray(INDX_BLOCK_SIZE)

        # INDX signature
        indx[0:4] = b'INDX'

        # USA: offset 40, count = sectors + 1
        usa_offset = 40
        num_sectors = INDX_BLOCK_SIZE // SECTOR_SIZE  # 8 for 4KB
        usa_count = num_sectors + 1  # 9
        usa_size = usa_count * 2  # 18 bytes

        struct.pack_into('<H', indx, 4, usa_offset)
        struct.pack_into('<H', indx, 6, usa_count)

        # VCN of this block
        struct.pack_into('<Q', indx, 16, block_num)

        # Calculate entries start: after USA, aligned to 8
        # USA ends at 40 + 18 = 58, aligned to 8 = 64
        entries_start = (usa_offset + usa_size + 7) & ~7  # = 64

        # Index node header at offset 24 (0x18)
        # Entries offset is RELATIVE to the node header start (offset 24)
        node_entries_offset = entries_start - 24  # = 40

        # Build entries
        entries_data = bytearray()
        for filename, entry_data in entries:
            entries_data.extend(entry_data)

        # Add end entry
        end_entry = bytearray(16)
        struct.pack_into('<H', end_entry, 8, 16)  # Length
        struct.pack_into('<H', end_entry, 12, 2)  # Flags: LAST_ENTRY
        entries_data.extend(end_entry)

        # Node header fields at offset 24
        struct.pack_into('<I', indx, 24, node_entries_offset)  # Entries offset from header
        struct.pack_into('<I', indx, 28, node_entries_offset + len(entries_data))  # Total size
        struct.pack_into('<I', indx, 32, INDX_BLOCK_SIZE - 24)  # Allocated size
        struct.pack_into('<I', indx, 36, 0)  # Flags (leaf node)

        # Write entries at correct position (AFTER USA)
        indx[entries_start:entries_start + len(entries_data)] = entries_data

        # Apply fixups
        self._apply_indx_fixups(indx)

        return bytes(indx)

    def _apply_indx_fixups(self, indx: bytearray):
        """Apply USA fixups to an INDX block."""
        usa_offset = struct.unpack('<H', indx[4:6])[0]
        usa_count = struct.unpack('<H', indx[6:8])[0]

        # Generate sequence value
        seq_val = 1

        # Write sequence value
        struct.pack_into('<H', indx, usa_offset, seq_val)

        # Apply to each sector end
        for i in range(1, usa_count):
            sector_end = i * 512 - 2
            if sector_end + 2 <= len(indx) and usa_offset + i * 2 + 2 <= len(indx):
                # Save original bytes
                struct.pack_into('<H', indx, usa_offset + i * 2,
                               struct.unpack('<H', indx[sector_end:sector_end + 2])[0])
                # Write sequence value
                struct.pack_into('<H', indx, sector_end, seq_val)

    def _build_virtualized_mft_record(self, record_num: int, original_record: bytearray) -> Optional[bytes]:
        """Build a virtualized MFT record for a directory.

        This function:
        1. Preserves attributes before INDEX_ROOT (STANDARD_INFO, FILE_NAME, etc.)
        2. Rebuilds INDEX_ROOT with virtual entries
        3. Adds INDEX_ALLOCATION and BITMAP if needed for large directories
        4. Preserves attributes AFTER the $I30 BITMAP (REPARSE_POINT, EA, etc.)
        5. Uses proper instance numbers to avoid conflicts
        """
        if record_num not in self.virtualized_dirs:
            return None

        vdir = self.virtualized_dirs[record_num]
        record = self._undo_fixups(bytearray(original_record))

        if record[0:4] != b'FILE':
            return None

        # Find and modify INDEX_ROOT, possibly add/modify INDEX_ALLOCATION
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr

        # Check if we're using original clusters (inplace approach)
        # When using original clusters, we preserve INDEX_ALLOCATION and BITMAP
        use_original_clusters = bool(vdir.get('original_clusters'))

        # Collect attributes before INDEX_ROOT and after $I30 BITMAP
        attrs_before = bytearray()
        attrs_after = bytearray()
        original_index_alloc = None  # Preserved when using original clusters
        original_i30_bitmap = None   # Preserved when using original clusters
        index_root_off = None
        i30_bitmap_off = None
        max_instance = 0

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            # Track max instance number
            attr_instance = struct.unpack('<H', record[off + 14:off + 16])[0]
            max_instance = max(max_instance, attr_instance)

            if attr_type == 0x90:  # INDEX_ROOT
                index_root_off = off
            elif attr_type == 0xA0:  # INDEX_ALLOCATION
                if use_original_clusters:
                    # Preserve original INDEX_ALLOCATION when using real clusters
                    original_index_alloc = bytes(record[off:off + attr_len])
                # Otherwise skip - we'll rebuild it with virtual clusters
            elif attr_type == 0xB0:  # BITMAP
                # Check if this is the $I30 bitmap (name length 4)
                name_len = record[off + 9]
                if name_len == 4:
                    i30_bitmap_off = off
                    if use_original_clusters:
                        # Preserve original $I30 bitmap when using real clusters
                        original_i30_bitmap = bytes(record[off:off + attr_len])
                    # Otherwise skip - we'll rebuild it
                else:
                    # Different bitmap, preserve it
                    if i30_bitmap_off is not None:
                        attrs_after.extend(record[off:off + attr_len])
                    elif index_root_off is None:
                        attrs_before.extend(record[off:off + attr_len])
            elif i30_bitmap_off is not None:
                # Attribute after $I30 BITMAP - preserve it
                attrs_after.extend(record[off:off + attr_len])
            elif index_root_off is None:
                # Attribute before INDEX_ROOT - preserve it
                attrs_before.extend(record[off:off + attr_len])

            off += attr_len

        if index_root_off is None:
            return None

        # Assign new instance numbers sequentially after max
        next_instance = max_instance + 1

        # Build new INDEX_ROOT
        new_index_root = self._build_virtual_index_root(record, index_root_off, vdir, next_instance)
        if new_index_root is None:
            return None
        next_instance += 1

        # Build new record - calculate sizes first to avoid overflow
        max_usable = MFT_RECORD_SIZE - 8  # Leave room for fixups and end marker

        needed = first_attr + len(attrs_before) + len(new_index_root) + 4  # +4 for end marker

        if vdir['use_indx']:
            if use_original_clusters:
                # Use preserved original INDEX_ALLOCATION and BITMAP
                # These point to the real clusters we're intercepting
                index_alloc = original_index_alloc
                bitmap = original_i30_bitmap
            else:
                # Build new INDEX_ALLOCATION pointing to virtual clusters
                index_alloc = self._build_virtual_index_allocation(vdir, next_instance)
                next_instance += 1
                bitmap = self._build_virtual_bitmap(vdir, next_instance)
                next_instance += 1

            if index_alloc:
                needed += len(index_alloc)
            if bitmap:
                needed += len(bitmap)
        else:
            index_alloc = None
            bitmap = None

        # Add space for attrs_after
        if attrs_after:
            needed += len(attrs_after)

        if needed > max_usable:
            log(f"Warning: virtualized record {record_num} too large ({needed} > {max_usable}), skipping")
            return None

        new_record = bytearray(MFT_RECORD_SIZE)
        new_record[0:first_attr] = record[0:first_attr]

        pos = first_attr
        # Copy attributes before INDEX_ROOT
        new_record[pos:pos + len(attrs_before)] = attrs_before
        pos += len(attrs_before)

        # Add new INDEX_ROOT
        new_record[pos:pos + len(new_index_root)] = new_index_root
        pos += len(new_index_root)

        # If using INDX blocks, add INDEX_ALLOCATION and BITMAP
        if vdir['use_indx']:
            if index_alloc:
                new_record[pos:pos + len(index_alloc)] = index_alloc
                pos += len(index_alloc)

            if bitmap:
                new_record[pos:pos + len(bitmap)] = bitmap
                pos += len(bitmap)

        # Add preserved attributes after $I30 BITMAP
        if attrs_after:
            new_record[pos:pos + len(attrs_after)] = attrs_after
            pos += len(attrs_after)

        # Add end marker
        struct.pack_into('<I', new_record, pos, 0xFFFFFFFF)
        pos += 4

        # Update used size
        struct.pack_into('<I', new_record, 24, pos)

        # Update next_attribute_instance in record header (offset 40)
        struct.pack_into('<H', new_record, 40, next_instance)

        # Apply fixups
        self._apply_fixups_to_record(new_record)

        return bytes(new_record)

    def _build_virtual_index_root(self, record: bytearray, attr_off: int, vdir: dict,
                                    instance: int) -> Optional[bytes]:
        """Build a virtualized INDEX_ROOT attribute."""
        # Parse original attribute header
        orig_attr_len = struct.unpack('<I', record[attr_off + 4:attr_off + 8])[0]
        name_len = record[attr_off + 9]
        name_off = struct.unpack('<H', record[attr_off + 10:attr_off + 12])[0]
        val_off = struct.unpack('<H', record[attr_off + 20:attr_off + 22])[0]

        attr_name = b''
        if name_len > 0:
            attr_name = bytes(record[attr_off + name_off:attr_off + name_off + name_len * 2])

        # Parse original index root header
        idx_root_start = attr_off + val_off
        idx_attr_type = struct.unpack('<I', record[idx_root_start:idx_root_start + 4])[0]
        collation_rule = struct.unpack('<I', record[idx_root_start + 4:idx_root_start + 8])[0]
        idx_block_size = struct.unpack('<I', record[idx_root_start + 8:idx_root_start + 12])[0]
        clusters_per_block = record[idx_root_start + 12]

        # Build entries for INDEX_ROOT
        if vdir['use_indx']:
            # INDEX_ROOT for large directory: contains root node of B+ tree
            # The end entry must have HAS_SUBNODES flag and VCN pointer to first INDX block
            # Entry structure:
            #   0-7: MFT reference (0 for end entry)
            #   8-9: Entry length (24 = 16 base + 8 VCN)
            #   10-11: Key length (0 for end entry)
            #   12-13: Flags (0x03 = LAST_ENTRY | HAS_SUBNODES)
            #   14-15: Padding
            #   16-23: Sub-node VCN (0 = first INDX block)
            entries_data = bytearray(24)  # End entry with VCN pointer
            struct.pack_into('<H', entries_data, 8, 24)  # Length (includes VCN)
            struct.pack_into('<H', entries_data, 12, 0x03)  # Flags: LAST_ENTRY | HAS_SUBNODES
            struct.pack_into('<Q', entries_data, 16, 0)  # VCN of first INDX block
            idx_flags = 0x01  # Large index
        else:
            # All entries in INDEX_ROOT
            entries_data = bytearray()
            for filename, entry_data in vdir['entries']:
                entries_data.extend(entry_data)
            # Add end entry
            end_entry = bytearray(16)
            struct.pack_into('<H', end_entry, 8, 16)
            struct.pack_into('<H', end_entry, 12, 2)
            entries_data.extend(end_entry)
            idx_flags = 0x00  # Small index

        # Build attribute
        header_size = 24
        if name_len > 0:
            header_size = (24 + name_len * 2 + 7) & ~7

        index_data_size = 16 + 16 + len(entries_data)
        new_attr_len = header_size + index_data_size
        new_attr_len = (new_attr_len + 7) & ~7

        new_attr = bytearray(new_attr_len)
        struct.pack_into('<I', new_attr, 0, 0x90)  # Type
        struct.pack_into('<I', new_attr, 4, new_attr_len)
        new_attr[8] = 0  # Resident
        new_attr[9] = name_len

        if name_len > 0:
            struct.pack_into('<H', new_attr, 10, 24)
            new_attr[24:24 + len(attr_name)] = attr_name
            val_start = (24 + name_len * 2 + 7) & ~7
        else:
            struct.pack_into('<H', new_attr, 10, 0)
            val_start = 24

        struct.pack_into('<H', new_attr, 12, 0)  # Flags
        struct.pack_into('<H', new_attr, 14, instance)  # Instance
        struct.pack_into('<I', new_attr, 16, index_data_size)
        struct.pack_into('<H', new_attr, 20, val_start)

        # Index root header
        irh = val_start
        struct.pack_into('<I', new_attr, irh, idx_attr_type)
        struct.pack_into('<I', new_attr, irh + 4, collation_rule)
        struct.pack_into('<I', new_attr, irh + 8, idx_block_size)
        new_attr[irh + 12] = clusters_per_block

        # Index header
        ihdr = irh + 16
        struct.pack_into('<I', new_attr, ihdr, 16)  # Entries offset
        struct.pack_into('<I', new_attr, ihdr + 4, 16 + len(entries_data))
        struct.pack_into('<I', new_attr, ihdr + 8, 16 + len(entries_data))
        new_attr[ihdr + 12] = idx_flags

        # Entries
        new_attr[irh + 32:irh + 32 + len(entries_data)] = entries_data

        return bytes(new_attr)

    def _build_virtual_index_allocation(self, vdir: dict, instance: int) -> Optional[bytes]:
        """Build a virtualized INDEX_ALLOCATION attribute."""
        if not vdir['use_indx'] or not vdir['virtual_indx_clusters']:
            return None

        # Build data runs pointing to virtual clusters
        clusters = vdir['virtual_indx_clusters']
        data_runs = self._encode_data_runs_simple(clusters)

        # Calculate sizes
        num_clusters = len(clusters)
        alloc_size = num_clusters * self.cluster_size
        data_size = alloc_size

        # Build non-resident attribute
        header_size = 72  # Non-resident header with name
        name = b'$\x00I\x003\x000\x00'  # "$I30" in UTF-16LE
        name_len = 4

        attr_len = header_size + len(data_runs)
        attr_len = (attr_len + 7) & ~7

        attr = bytearray(attr_len)
        struct.pack_into('<I', attr, 0, 0xA0)  # Type
        struct.pack_into('<I', attr, 4, attr_len)
        attr[8] = 1  # Non-resident
        attr[9] = name_len
        struct.pack_into('<H', attr, 10, 64)  # Name offset
        struct.pack_into('<H', attr, 12, 0)  # Flags
        struct.pack_into('<H', attr, 14, instance)  # Instance

        # Non-resident specific
        struct.pack_into('<Q', attr, 16, 0)  # Start VCN
        struct.pack_into('<Q', attr, 24, num_clusters - 1)  # End VCN
        struct.pack_into('<H', attr, 32, 64 + name_len * 2)  # Data runs offset
        struct.pack_into('<Q', attr, 40, alloc_size)  # Allocated size
        struct.pack_into('<Q', attr, 48, data_size)  # Data size
        struct.pack_into('<Q', attr, 56, data_size)  # Initialized size

        # Name
        attr[64:64 + len(name)] = name

        # Data runs
        run_off = 64 + name_len * 2
        attr[run_off:run_off + len(data_runs)] = data_runs

        return bytes(attr)

    def _build_virtual_bitmap(self, vdir: dict, instance: int) -> Optional[bytes]:
        """Build a virtualized BITMAP attribute for directory index."""
        if not vdir['use_indx']:
            return None

        num_blocks = len(vdir['indx_blocks'])
        # Bitmap: 1 bit per block, all set to 1
        bitmap_bytes = (num_blocks + 7) // 8
        bitmap_data = bytes([0xFF] * bitmap_bytes)

        # Resident attribute
        name = b'$\x00I\x003\x000\x00'  # "$I30"
        name_len = 4

        header_size = 24 + name_len * 2
        header_size = (header_size + 7) & ~7
        val_off = header_size

        attr_len = header_size + len(bitmap_data)
        attr_len = (attr_len + 7) & ~7

        attr = bytearray(attr_len)
        struct.pack_into('<I', attr, 0, 0xB0)  # Type
        struct.pack_into('<I', attr, 4, attr_len)
        attr[8] = 0  # Resident
        attr[9] = name_len
        struct.pack_into('<H', attr, 10, 24)  # Name offset
        struct.pack_into('<H', attr, 12, 0)  # Flags
        struct.pack_into('<H', attr, 14, instance)  # Instance
        struct.pack_into('<I', attr, 16, len(bitmap_data))  # Value length
        struct.pack_into('<H', attr, 20, val_off)  # Value offset

        # Name
        attr[24:24 + len(name)] = name

        # Bitmap data
        attr[val_off:val_off + len(bitmap_data)] = bitmap_data

        return bytes(attr)

    def _encode_data_runs_simple(self, clusters: List[int]) -> bytes:
        """Encode a simple list of clusters as NTFS data runs."""
        if not clusters:
            return b'\x00'

        runs = bytearray()
        prev_cluster = 0

        i = 0
        while i < len(clusters):
            # Find contiguous run
            start = clusters[i]
            count = 1
            while i + count < len(clusters) and clusters[i + count] == start + count:
                count += 1

            # Encode run
            offset = start - prev_cluster

            # Determine sizes needed (use +8 to add zero byte when MSB >= 0x80,
            # preventing sign-extension by NTFS parsers including Windows)
            count_bytes = (count.bit_length() + 8) // 8
            if offset >= 0:
                offset_bytes = (offset.bit_length() + 8) // 8  # +1 for sign
            else:
                offset_bytes = ((-offset).bit_length() + 8) // 8

            count_bytes = max(1, min(count_bytes, 4))
            offset_bytes = max(1, min(offset_bytes, 4))

            header = (offset_bytes << 4) | count_bytes
            runs.append(header)

            # Count (little-endian)
            for b in range(count_bytes):
                runs.append((count >> (b * 8)) & 0xFF)

            # Offset (little-endian, signed)
            if offset < 0:
                offset = (1 << (offset_bytes * 8)) + offset
            for b in range(offset_bytes):
                runs.append((offset >> (b * 8)) & 0xFF)

            prev_cluster = start  # Next offset is relative to this run's start
            i += count

        runs.append(0)  # End marker
        return bytes(runs)

    def _apply_fixups_to_record(self, record: bytearray):
        """Apply NTFS fixups to an MFT record."""
        usa_offset = struct.unpack('<H', record[4:6])[0]
        usa_count = struct.unpack('<H', record[6:8])[0]

        if usa_count < 2 or usa_offset + usa_count * 2 > MFT_RECORD_SIZE:
            return

        # Increment update sequence value
        seq_val = struct.unpack('<H', record[usa_offset:usa_offset + 2])[0]
        seq_val = (seq_val + 1) & 0xFFFF
        if seq_val == 0:
            seq_val = 1
        struct.pack_into('<H', record, usa_offset, seq_val)

        # Apply to each sector end
        for i in range(1, usa_count):
            sector_end = i * 512 - 2
            if sector_end + 2 <= MFT_RECORD_SIZE:
                # Save original bytes to USA
                struct.pack_into('<H', record, usa_offset + i * 2,
                               struct.unpack('<H', record[sector_end:sector_end + 2])[0])
                # Write sequence value
                struct.pack_into('<H', record, sector_end, seq_val)

    # =========================================================================
    # MFT write tracking (NTFS -> ext4 sync)
    # =========================================================================

    def is_mft_region(self, offset: int, length: int) -> bool:
        """Check if an offset affects the MFT region (including virtual MFT records)."""
        end = offset + length
        for disk_off, run_bytes in self._mft_runs:
            run_end = disk_off + run_bytes
            if offset < run_end and end > disk_off:
                return True
        # Also consider virtual MFT records (which may be at higher record numbers)
        if self.virtual_file_manager:
            vfm = self.virtual_file_manager
            if vfm.mft_to_virtual:
                for vrec in vfm.mft_to_virtual.keys():
                    vrec_off = self._rec_offset(vrec)
                    if vrec_off is not None:
                        if offset < vrec_off + MFT_RECORD_SIZE and end > vrec_off:
                            return True
        return False

    def _check_file_deleted(self, record_num: int) -> bool:
        """Check if a tracked file's MFT record was marked as deleted.

        Called from the background MFT sync thread (self.lock NOT held).
        Returns True if file was deleted.
        """
        with self.lock:
            record_offset = self._rec_offset(record_num)
            if record_offset is None:
                return False
            if record_offset + MFT_RECORD_SIZE > len(self.image):
                return False

            record = self.image[record_offset:record_offset + MFT_RECORD_SIZE]
            if record[0:4] != b'FILE':
                return False

            flags = struct.unpack('<H', record[22:24])[0]
            if flags & 0x01:  # Still in use - not deleted
                return False

            source_path = self.mft_record_to_source.get(record_num)
            if not source_path:
                return True

            rel_path = self._get_rel_path(source_path)

            if rel_path in self.ext4_sync_in_progress:
                log(f"  Skipping delete (ext4 sync in progress): {rel_path}")
                del self.mft_record_to_source[record_num]
                self.resident_file_data.pop(record_num, None)
                self.path_to_mft_record.pop(rel_path, None)
                return True

            # Remove tracking before releasing lock
            del self.mft_record_to_source[record_num]
            self.resident_file_data.pop(record_num, None)
            self.path_to_mft_record.pop(rel_path, None)
            if source_path in self.source_to_clusters:
                for cluster in self.source_to_clusters[source_path]:
                    if cluster in self.cluster_map:
                        del self.cluster_map[cluster]
                del self.source_to_clusters[source_path]
            self._direct_run_map = [r for r in self._direct_run_map if r[2] != source_path]
            do_delete = os.path.exists(source_path)
            if do_delete:
                self.ntfs_sync_in_progress.add(rel_path)
                self.ntfs_sync_timestamps[rel_path] = time.time()

        # Delete from ext4 without holding lock (os.remove is fast but avoid blocking)
        if do_delete:
            try:
                os.remove(source_path)
                log(f"  FILE DELETED: {rel_path}")
            except OSError as e:
                log(f"  Failed to delete {rel_path}: {e}")
            finally:
                with self.lock:
                    self.ntfs_sync_in_progress.discard(rel_path)

        return True

    def _check_directory_rename(self, record_num: int):
        """Check if a tracked directory was renamed.

        Called from the background MFT sync thread (self.lock NOT held).
        Acquires self.lock for metadata reads/writes; releases it before
        the slow shutil.move filesystem operation.
        """
        with self.lock:
            old_rel_path = self.mft_record_to_dir.get(record_num)
            if not old_rel_path:
                return

            record_offset = self._rec_offset(record_num)
            if record_offset is None:
                return
            if record_offset + MFT_RECORD_SIZE > len(self.image):
                return

            record = self._undo_fixups(bytearray(
                self.image[record_offset:record_offset + MFT_RECORD_SIZE]))
            if record[0:4] != b'FILE':
                return

            flags = struct.unpack('<H', record[22:24])[0]
            seq   = struct.unpack('<H', record[16:18])[0]

            # Guard: record was freed (not in-use) or reused as a non-directory.
            # Either way this is NOT a rename – it's a recycled slot.  Clean up
            # our stale tracking entry and leave the ext4 directory untouched.
            if not (flags & 0x01) or not (flags & 0x02):
                log(f"  _check_dir_rename({record_num}): record no longer in-use/dir "
                    f"(flags=0x{flags:04x}), removing stale tracking for {old_rel_path}")
                self.mft_record_to_dir.pop(record_num, None)
                self._dir_mft_seq.pop(record_num, None)
                self.path_to_mft_record.pop(old_rel_path, None)
                return

            # Guard: sequence number changed → record was freed and reused for a
            # completely different directory entity.  Removing our stale tracking
            # is the only safe action; let _check_new_directory handle the new one.
            expected_seq = self._dir_mft_seq.get(record_num)
            if expected_seq is not None and seq != expected_seq:
                log(f"  _check_dir_rename({record_num}): seq changed {expected_seq}->{seq}, "
                    f"record recycled (was {old_rel_path}), removing stale tracking")
                self.mft_record_to_dir.pop(record_num, None)
                self._dir_mft_seq.pop(record_num, None)
                self.path_to_mft_record.pop(old_rel_path, None)
                return

            filename, parent_ref = self._extract_filename_and_parent(record)
            if not filename:
                return

            parent_record = parent_ref & 0xFFFFFFFFFFFF
            if parent_record == 5:
                new_rel_path = filename
            elif parent_record in self.mft_record_to_dir:
                parent_path = self.mft_record_to_dir[parent_record]
                new_rel_path = os.path.join(parent_path, filename) if parent_path else filename
            else:
                new_rel_path = filename

            if new_rel_path == old_rel_path:
                return

            if new_rel_path in self.ext4_sync_in_progress:
                log(f"  Skipping dir rename (ext4 sync in progress): {new_rel_path}")
                self.mft_record_to_dir[record_num] = new_rel_path
                self.path_to_mft_record.pop(old_rel_path, None)
                self.path_to_mft_record[new_rel_path] = record_num
                return

            old_path = self._resolve_source_path(old_rel_path)
            new_path = self._resolve_source_path(new_rel_path)
            do_move = os.path.exists(old_path) and not os.path.exists(new_path)
            if do_move:
                self.ntfs_sync_in_progress.add(new_rel_path)
                self.ntfs_sync_in_progress.add(old_rel_path)
                now = time.time()
                self.ntfs_sync_timestamps[new_rel_path] = now
                self.ntfs_sync_timestamps[old_rel_path] = now

        # Slow filesystem op outside the lock so reads are not blocked
        if do_move:
            try:
                shutil.move(old_path, new_path)
                log(f"  DIR RENAMED: {old_rel_path} -> {new_rel_path}")
            except OSError as e:
                log(f"  Failed to rename dir {old_rel_path}: {e}")

        # Always update tracking (even on failure) to prevent infinite retry
        with self.lock:
            if do_move:
                self.ntfs_sync_in_progress.discard(new_rel_path)
                self.ntfs_sync_in_progress.discard(old_rel_path)
            self._update_child_paths_on_dir_rename(old_rel_path, new_rel_path)
            self.mft_record_to_dir[record_num] = new_rel_path
            self._dir_mft_seq[record_num] = seq  # sequence already read above
            self.path_to_mft_record.pop(old_rel_path, None)
            self.path_to_mft_record[new_rel_path] = record_num

    def _update_child_paths_on_dir_rename(self, old_dir_path: str, new_dir_path: str):
        """Update all child file/dir paths when a parent directory is renamed."""
        old_prefix = old_dir_path + os.sep
        new_prefix = new_dir_path + os.sep

        # Update mft_record_to_source (file paths)
        for record_num, source_path in list(self.mft_record_to_source.items()):
            rel_path = self._get_rel_path(source_path)
            if rel_path.startswith(old_prefix) or rel_path == old_dir_path:
                new_rel = new_dir_path + rel_path[len(old_dir_path):]
                new_source = os.path.join(self.source_dir, new_rel)
                self.mft_record_to_source[record_num] = new_source

                # Update source_to_clusters
                if source_path in self.source_to_clusters:
                    clusters = self.source_to_clusters.pop(source_path)
                    self.source_to_clusters[new_source] = clusters
                    # Update cluster_map entries
                    for cluster in clusters:
                        if cluster in self.cluster_map:
                            _, offset = self.cluster_map[cluster]
                            self.cluster_map[cluster] = (new_source, offset)
                # Update run-based mappings (large files)
                self._direct_run_map = [
                    (s, e, new_source if sp == source_path else sp, o)
                    for s, e, sp, o in self._direct_run_map
                ]

                # Update resident_file_data
                if record_num in self.resident_file_data:
                    self.resident_file_data[record_num]['source_path'] = new_source

        # Update path_to_mft_record
        for rel_path, record_num in list(self.path_to_mft_record.items()):
            if rel_path.startswith(old_prefix) or rel_path == old_dir_path:
                new_rel = new_dir_path + rel_path[len(old_dir_path):]
                del self.path_to_mft_record[rel_path]
                self.path_to_mft_record[new_rel] = record_num

        # Update mft_record_to_dir (child directories)
        for record_num, dir_path in list(self.mft_record_to_dir.items()):
            if dir_path.startswith(old_prefix):
                new_rel = new_dir_path + dir_path[len(old_dir_path):]
                self.mft_record_to_dir[record_num] = new_rel

    def _is_orphan_root_fallthrough(self, parent_record: int, rel_path: str) -> bool:
        """Detect the parent-untracked → root-fallthrough corruption pattern.

        The three workers (_check_new_directory, _check_new_file,
        _reparse_mft_record) all derive rel_path with the same branch:

            if parent_record == 5:
                rel_path = filename
            elif parent_record in mft_record_to_dir:
                rel_path = join(parent_path, filename)
            else:
                rel_path = filename   # <-- silently treated as root

        The final else fires when Windows mutates an MFT record whose parent
        directory we never successfully tracked (e.g. _check_new_directory
        rejected it earlier because _validate_path balks at symlink
        traversal). The worker then materializes the record at source_dir
        root — a corrupted duplicate of a tree file, invisible in the NTFS
        view because the MFT entry already points correctly inside its
        proper subtree.

        We only flag this specific fallthrough — *not* a legitimate root
        write where parent_record == 5. Backblaze and other apps may freely
        create new top-level entries (.bzvol, working dirs, etc) and those
        go through the normal write path.
        """
        if parent_record == 5:
            return False
        if parent_record in self.mft_record_to_dir:
            return False
        if os.sep in rel_path:
            return False
        return True

    def _check_new_directory(self, record_num: int):
        """Check if an MFT record is a new directory.

        Called from the background MFT sync thread (self.lock NOT held).
        """
        with self.lock:
            record_offset = self._rec_offset(record_num)
            if record_offset is None:
                return
            if record_offset + MFT_RECORD_SIZE > len(self.image):
                log(f"  _check_new_dir({record_num}): beyond image")
                return

            record = self._undo_fixups(bytearray(
                self.image[record_offset:record_offset + MFT_RECORD_SIZE]))
            if record[0:4] != b'FILE':
                return

            flags = struct.unpack('<H', record[22:24])[0]
            seq   = struct.unpack('<H', record[16:18])[0]
            if not (flags & 0x01) or not (flags & 0x02):
                return
            if record_num in self.mft_record_to_dir:
                return

            filename, parent_ref = self._extract_filename_and_parent(record)
            if not filename or filename.startswith('$'):
                return
            log(f"  _check_new_dir({record_num}): {filename} parent={parent_ref & 0xFFFFFFFFFFFF}")

            parent_record = parent_ref & 0xFFFFFFFFFFFF
            if parent_record == 5:
                rel_path = filename
            elif parent_record in self.mft_record_to_dir:
                parent_path = self.mft_record_to_dir[parent_record]
                rel_path = os.path.join(parent_path, filename) if parent_path else filename
            else:
                rel_path = filename

            if self._is_orphan_root_fallthrough(parent_record, rel_path):
                log(f"  Skipping orphan-root dir fallthrough: {rel_path} "
                    f"(record {record_num}, parent={parent_record} untracked)")
                return

            if rel_path in self.ext4_sync_in_progress:
                log(f"  Skipping new dir (ext4 sync in progress): {rel_path}")
                self.mft_record_to_dir[record_num] = rel_path
                self._dir_mft_seq[record_num] = seq
                self.path_to_mft_record[rel_path] = record_num
                return

            source_path = self._resolve_source_path(rel_path)
            if not self._validate_path(source_path, '_check_new_directory'):
                return
            do_create = not os.path.exists(source_path)
            if do_create:
                self.ntfs_sync_in_progress.add(rel_path)
                self.ntfs_sync_timestamps[rel_path] = time.time()

        # os.makedirs outside lock (fast but avoids holding lock during I/O)
        try:
            if do_create:
                os.makedirs(source_path, exist_ok=True)
                log(f"  NEW DIR: {rel_path} -> {source_path}")
        except OSError as e:
            log(f"  Failed to create dir {rel_path}: {e}")
        finally:
            with self.lock:
                if do_create:
                    self.ntfs_sync_in_progress.discard(rel_path)
                self.mft_record_to_dir[record_num] = rel_path
                self._dir_mft_seq[record_num] = seq
                self.path_to_mft_record[rel_path] = record_num

    def _check_new_file(self, record_num: int) -> Optional[str]:
        """Check if an MFT record is a new file.

        Called from the background MFT sync thread (self.lock NOT held).
        """
        with self.lock:
            record_offset = self._rec_offset(record_num)
            if record_offset is None:
                return None
            if record_offset + MFT_RECORD_SIZE > len(self.image):
                log(f"  _check_new_file({record_num}): beyond image")
                return None

            record = self._undo_fixups(bytearray(
                self.image[record_offset:record_offset + MFT_RECORD_SIZE]))
            if record[0:4] != b'FILE':
                return None

            flags = struct.unpack('<H', record[22:24])[0]
            if not (flags & 0x01) or (flags & 0x02):
                return None

            filename, parent_ref = self._extract_filename_and_parent(record)
            if not filename or filename.startswith('$'):
                return None
            log(f"  _check_new_file({record_num}): {filename} parent={parent_ref & 0xFFFFFFFFFFFF}")

            parent_record = parent_ref & 0xFFFFFFFFFFFF
            if parent_record == 5:
                rel_path = filename
            elif parent_record in self.mft_record_to_dir:
                parent_path = self.mft_record_to_dir[parent_record]
                rel_path = os.path.join(parent_path, filename) if parent_path else filename
            else:
                rel_path = filename

            if self._is_orphan_root_fallthrough(parent_record, rel_path):
                log(f"  Skipping orphan-root file fallthrough: {rel_path} "
                    f"(record {record_num}, parent={parent_record} untracked)")
                return None

            source_path = self._resolve_source_path(rel_path)
            if not self._validate_path(source_path, '_check_new_file'):
                return None

            if rel_path in self.ext4_sync_in_progress:
                log(f"  Skipping new file (ext4 sync in progress): {rel_path}")
                self.mft_record_to_source[record_num] = source_path
                self._track_file_data(record, record_num, source_path)
                return None

            if os.path.exists(source_path):
                self.mft_record_to_source[record_num] = source_path
                self._track_file_data(record, record_num, source_path)
                return None

            # Snapshot data runs under lock before releasing for I/O
            data_runs = self._extract_data_runs(record)
            resident_data = None if data_runs else self._extract_resident_data(record)
            file_size = self._extract_file_size(record) if data_runs else None
            self.ntfs_sync_in_progress.add(rel_path)
            self.ntfs_sync_timestamps[rel_path] = time.time()

        # Create the file in ext4 (outside lock)
        try:
            parent_dir = os.path.dirname(source_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            if data_runs:
                with open(source_path, 'wb') as f:
                    for start_cluster, num_clusters in data_runs:
                        if start_cluster == -1:
                            f.write(b'\x00' * num_clusters * self.cluster_size)
                            continue
                        for i in range(num_clusters):
                            cluster = start_cluster + i
                            cluster_offset = cluster * self.cluster_size
                            with self.lock:
                                chunk = self.image[cluster_offset:cluster_offset + self.cluster_size] \
                                    if cluster_offset + self.cluster_size <= len(self.image) else b''
                            f.write(chunk)
                if file_size is not None and file_size > 0:
                    with open(source_path, 'r+b') as f:
                        f.truncate(file_size)
                log(f"  NEW FILE (non-resident): {rel_path} ({file_size} bytes)")
            else:
                with open(source_path, 'wb') as f:
                    if resident_data:
                        f.write(resident_data)
                log(f"  NEW FILE: {rel_path}")

            with self.lock:
                self.ntfs_sync_in_progress.discard(rel_path)
                self.mft_record_to_source[record_num] = source_path
                self.path_to_mft_record[rel_path] = record_num
                self._track_file_data(record, record_num, source_path)
            return source_path
        except OSError as e:
            log(f"  Failed to create file {rel_path}: {e}")
            with self.lock:
                self.ntfs_sync_in_progress.discard(rel_path)
            return None

    def _track_file_data(self, record: bytearray, record_num: int, source_path: str):
        """Track file data - either cluster mapping or resident tracking."""
        data_runs = self._extract_data_runs(record)
        if data_runs:
            self._map_clusters(data_runs, source_path)
            # Remove from resident tracking if it was resident before
            self.resident_file_data.pop(record_num, None)
        else:
            resident_loc = self._find_resident_data_location(record, record_num)
            if resident_loc:
                self.resident_file_data[record_num] = {
                    'source_path': source_path,
                    'val_len_abs': resident_loc[0],
                    'data_abs': resident_loc[1],
                    'available': resident_loc[2],
                }

    def _reparse_mft_record(self, record_num: int):
        """Re-parse an MFT record for cluster updates, renames, and resident data.

        Called from the background MFT sync thread (self.lock NOT held).
        Acquires self.lock for metadata reads/writes; releases it before
        slow filesystem operations (shutil.move, file I/O).
        """
        # --- Phase 1: read state under lock, decide what to do ---
        with self.lock:
            source_path = self.mft_record_to_source.get(record_num)
            if not source_path:
                return

            record_offset = self._rec_offset(record_num)
            if record_offset is None:
                return
            if record_offset + MFT_RECORD_SIZE > len(self.image):
                return

            record = self._undo_fixups(bytearray(
                self.image[record_offset:record_offset + MFT_RECORD_SIZE]))
            if record[0:4] != b'FILE':
                return

            # Determine rename
            do_move = False
            old_source = source_path
            new_path = source_path
            old_rel = new_rel_path = self._get_rel_path(source_path)

            filename, parent_ref = self._extract_filename_and_parent(record)
            if filename:
                parent_record = parent_ref & 0xFFFFFFFFFFFF
                if parent_record == 5:
                    new_rel_path = filename
                elif parent_record in self.mft_record_to_dir:
                    parent_path = self.mft_record_to_dir[parent_record]
                    new_rel_path = os.path.join(parent_path, filename) if parent_path else filename
                else:
                    new_rel_path = filename

                if self._is_orphan_root_fallthrough(parent_record, new_rel_path) and os.sep in old_rel:
                    log(f"  Refusing reparse-move to orphan root: {old_rel} -> {new_rel_path} "
                        f"(record {record_num}, parent={parent_record} untracked); keeping old path")
                    new_rel_path = old_rel

                new_path = self._resolve_source_path(new_rel_path)
                if not self._validate_path(new_path, '_reparse_mft_record'):
                    new_path = source_path
                    new_rel_path = old_rel

                if new_path != source_path:
                    if os.path.exists(source_path):
                        if new_rel_path not in self.ext4_sync_in_progress:
                            if not os.path.exists(new_path):
                                do_move = True
                                self.ntfs_sync_in_progress.add(new_rel_path)
                                self.ntfs_sync_in_progress.add(old_rel)
                                now = time.time()
                                self.ntfs_sync_timestamps[new_rel_path] = now
                                self.ntfs_sync_timestamps[old_rel] = now
                            else:
                                # Target already exists - just update tracking
                                if old_rel in self.path_to_mft_record:
                                    del self.path_to_mft_record[old_rel]
                                self.path_to_mft_record[new_rel_path] = record_num
                                source_path = new_path
                                self.mft_record_to_source[record_num] = new_path
                        else:
                            # Sync in progress - just update tracking
                            if old_rel in self.path_to_mft_record:
                                del self.path_to_mft_record[old_rel]
                            self.path_to_mft_record[new_rel_path] = record_num
                            source_path = new_path
                            self.mft_record_to_source[record_num] = new_path
                    else:
                        # File doesn't exist at old path - just update tracking
                        if old_rel in self.path_to_mft_record:
                            del self.path_to_mft_record[old_rel]
                        self.path_to_mft_record[new_rel_path] = record_num
                        source_path = new_path
                        self.mft_record_to_source[record_num] = new_path

        # --- Phase 2: slow filesystem op outside the lock ---
        if do_move:
            parent_dir = os.path.dirname(new_path)
            if parent_dir and not os.path.exists(parent_dir):
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                except OSError:
                    pass
            try:
                shutil.move(old_source, new_path)
                log(f"  FILE RENAMED: {os.path.basename(old_source)} -> {filename}")
            except OSError as e:
                log(f"  Failed to rename file: {e}")

            # Update tracking under lock (always, even on failure, to prevent retry)
            with self.lock:
                self.ntfs_sync_in_progress.discard(new_rel_path)
                self.ntfs_sync_in_progress.discard(old_rel)
                if old_rel in self.path_to_mft_record:
                    del self.path_to_mft_record[old_rel]
                self.path_to_mft_record[new_rel_path] = record_num
                source_path = new_path
                self.mft_record_to_source[record_num] = new_path
                if old_source in self.source_to_clusters:
                    clusters = self.source_to_clusters.pop(old_source)
                    self.source_to_clusters[new_path] = clusters
                    for cluster in clusters:
                        if cluster in self.cluster_map:
                            self.cluster_map[cluster] = (new_path, self.cluster_map[cluster][1])
                self._direct_run_map = [
                    (s, e, new_path if sp == old_source else sp, o)
                    for s, e, sp, o in self._direct_run_map
                ]

        # --- Phase 3: cluster remapping and resident data update under lock ---
        do_write_resident = False
        resident_data = None
        rel_path = None
        with self.lock:
            # Remove old cluster mappings
            if source_path in self.source_to_clusters:
                old_clusters = self.source_to_clusters[source_path]
                for cluster in old_clusters:
                    if cluster in self.cluster_map:
                        del self.cluster_map[cluster]
                self.source_to_clusters[source_path] = set()
            self._direct_run_map = [r for r in self._direct_run_map if r[2] != source_path]

            # Extract new data runs or resident data
            do_write_resident = False
            data_runs = self._extract_data_runs(record)
            if data_runs:
                self._map_clusters(data_runs, source_path)
                self.resident_file_data.pop(record_num, None)
            else:
                rel_path = self._get_rel_path(source_path)
                resident_data = self._extract_resident_data(record)
                do_write_resident = (
                    resident_data is not None and
                    rel_path not in self.ext4_sync_in_progress
                )
                if do_write_resident:
                    self.ntfs_sync_in_progress.add(rel_path)
                    self.ntfs_sync_timestamps[rel_path] = time.time()

                resident_loc = self._find_resident_data_location(record, record_num)
                if resident_loc:
                    self.resident_file_data[record_num] = {
                        'source_path': source_path,
                        'val_len_abs': resident_loc[0],
                        'data_abs': resident_loc[1],
                        'available': resident_loc[2],
                    }

        # Write resident data outside lock (fast, but avoids holding lock during I/O)
        if do_write_resident:
            try:
                current_data = b''
                current_size = 0
                if os.path.exists(source_path):
                    current_size = os.path.getsize(source_path)
                    with open(source_path, 'rb') as f:
                        current_data = f.read(len(resident_data) + 1)
                if current_data != resident_data:
                    # Safety guard: never truncate a non-empty source file to empty.
                    # ntfs-3g truncates files to 0 before rewriting (truncate-then-write
                    # sequence). If we see empty resident_data but the source has content,
                    # skip the write — the non-zero content write will follow shortly.
                    if not resident_data and current_size > 0:
                        log(f"  Skipping empty resident write for non-empty source "
                            f"({current_size}B): {os.path.basename(source_path)}")
                        # Re-queue the file for allocation since the NTFS record was
                        # wiped (ntfs-3g truncated it). The source file is intact so
                        # we need to re-allocate clusters pointing to it.
                        with self.lock:
                            if rel_path not in self.sparse_files:
                                record_num_for_realloc = self.path_to_mft_record.get(rel_path)
                                if record_num_for_realloc is not None:
                                    self.sparse_files[rel_path] = (source_path, current_size, record_num_for_realloc)
                                    log(f"  Re-queued for allocation: {rel_path}")
                    else:
                        with open(source_path, 'wb') as f:
                            f.write(resident_data)
            except OSError as e:
                log(f"  Error writing resident data: {e}")
            finally:
                with self.lock:
                    self.ntfs_sync_in_progress.discard(rel_path)
