"""Cluster mapper for NTFS-ext4 bridge.

Scans the MFT of an NTFS image to build a cluster-to-ext4-file mapping.
Handles read/write routing: metadata from image, data from ext4.
Tracks MFT writes to detect new files, renames, deletions, and reallocations.

Handles both non-resident files (data in clusters → mapped to ext4) and
resident files (data in MFT record → injected from ext4 on reads).

Supports lazy allocation: large files can start as sparse (no clusters),
get allocated on first read, and deallocated after a timeout.
"""
import os
import struct
import threading
import time
from typing import Dict, List, Tuple, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .lazy_allocator import LazyAllocator
    from .virtual_files import VirtualFileManager

MFT_RECORD_SIZE = 1024
CLUSTER_SIZE = 4096  # Standard NTFS cluster size


def log(msg):
    print(f"[ClusterMapper] {msg}", flush=True)


class ClusterMapper:
    """Maps NTFS clusters to ext4 source files via MFT scanning.

    Reads from the image file for metadata and from ext4 files for data.
    Writes go to the image for metadata and to ext4 for data clusters.

    Resident files (small files stored directly in MFT records) are handled
    by injecting ext4 content into MFT reads, so ext4 is always the source
    of truth for file content.
    """

    def __init__(self, image_path: str, source_dir: str):
        self.image_path = os.path.abspath(image_path)
        self.source_dir = os.path.abspath(source_dir)

        # Load the image
        with open(image_path, 'rb') as f:
            self.image = bytearray(f.read())

        # Parse boot sector
        boot = self.image[0:512]
        self.bytes_per_sector = struct.unpack('<H', boot[0x0B:0x0D])[0]
        self.sectors_per_cluster = boot[0x0D]
        self.cluster_size = self.bytes_per_sector * self.sectors_per_cluster
        self.mft_cluster = struct.unpack('<Q', boot[0x30:0x38])[0]
        self.mft_offset = self.mft_cluster * self.cluster_size

        # Cluster -> (source_file_path, offset_in_file)
        self.cluster_map: Dict[int, Tuple[str, int]] = {}

        # MFT tracking
        self.mft_record_to_source: Dict[int, str] = {}
        self.source_to_clusters: Dict[str, Set[int]] = {}
        self.mft_record_to_dir: Dict[int, str] = {}
        self.path_to_mft_record: Dict[str, int] = {}
        self.dir_children: Dict[int, Set[int]] = {}
        self.removed_mft_records: Set[int] = set()

        # Resident file tracking: record_num -> {source_path, val_len_abs, data_abs, avail}
        # These are files small enough that NTFS stores data directly in the MFT record
        self.resident_file_data: Dict[int, dict] = {}

        # Track which clusters are INDX blocks (direct bytes data)
        self.dir_indx_clusters: Dict[int, List[int]] = {}

        # Thread safety
        self.lock = threading.RLock()

        # Loop prevention sets (shared with SyncDaemon)
        self.ext4_sync_in_progress: Set[str] = set()
        self.ntfs_sync_in_progress: Set[str] = set()

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

        # Bitmap location (found during MFT scan)
        self.bitmap_clusters: List[Tuple[int, int]] = []  # (start_cluster, count)
        self.total_clusters = len(self.image) // self.cluster_size

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

        # Build reverse mappings
        self._build_path_mappings()

        log(f"Initialized: {len(self.cluster_map)} clusters mapped, "
            f"{len(self.mft_record_to_source)} files tracked "
            f"({len(self.resident_file_data)} resident)")

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
                            rel_path = os.path.relpath(source_path, self.source_dir)
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
        if cluster in self.cluster_map:
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

        MFT writes update the image and trigger re-parsing.
        Data cluster writes go to ext4 source files.
        Other metadata writes go to the image.
        """
        cluster_size = self.cluster_size

        # Check if write affects MFT
        if self.is_mft_region(offset, len(data)):
            self._handle_mft_write(offset, data)

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
                    # Write to ext4 source file
                    source_path, file_offset = mapping
                    write_offset = file_offset + cluster_offset
                    try:
                        with open(source_path, 'r+b') as f:
                            f.seek(write_offset)
                            f.write(chunk_data)
                        # Debug: log first write to each file
                        if not hasattr(self, '_write_logged'):
                            self._write_logged = set()
                        if source_path not in self._write_logged:
                            log(f"  Write {chunk_len}B to {os.path.basename(source_path)} at offset {write_offset}")
                            self._write_logged.add(source_path)
                    except OSError as e:
                        log(f"Write error for {source_path}: {e}")
                        # Fall through to write to image
                        if byte_offset + chunk_len <= len(self.image):
                            self.image[byte_offset:byte_offset + chunk_len] = chunk_data
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
        with open(self.image_path, 'r+b') as f:
            f.write(self.image)

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
        record_offset = self.mft_offset + 6 * MFT_RECORD_SIZE
        record = self.image[record_offset:record_offset + MFT_RECORD_SIZE]

        if record[:4] != b'FILE':
            log("Warning: $Bitmap MFT record not found")
            return

        data_runs = self._extract_data_runs(bytearray(record))
        if data_runs:
            self.bitmap_clusters = data_runs
            total_bitmap_clusters = sum(count for _, count in data_runs)
            log(f"  $Bitmap: {total_bitmap_clusters} clusters")

    def _read_bitmap(self) -> bytearray:
        """Read the entire cluster bitmap from the image."""
        bitmap = bytearray()
        for start_cluster, count in self.bitmap_clusters:
            offset = start_cluster * self.cluster_size
            length = count * self.cluster_size
            bitmap.extend(self.image[offset:offset + length])
        return bitmap

    def _write_bitmap(self, bitmap: bytearray):
        """Write the cluster bitmap back to the image."""
        pos = 0
        for start_cluster, count in self.bitmap_clusters:
            offset = start_cluster * self.cluster_size
            length = count * self.cluster_size
            self.image[offset:offset + length] = bitmap[pos:pos + length]
            pos += length

    def _find_free_clusters(self, count: int) -> Optional[List[int]]:
        """Find 'count' free clusters in the bitmap.

        Returns list of cluster numbers, or None if not enough free space.
        Tries to find contiguous clusters for efficiency.
        """
        if not self.bitmap_clusters:
            return None

        bitmap = self._read_bitmap()
        free_clusters = []
        cluster = 0

        # Skip system clusters (first ~16 clusters are usually reserved)
        start_search = max(16, self.mft_cluster + 100)  # Start after MFT region

        for byte_idx in range(start_search // 8, len(bitmap)):
            byte_val = bitmap[byte_idx]
            for bit in range(8):
                cluster = byte_idx * 8 + bit
                if cluster >= self.total_clusters:
                    break
                if not (byte_val & (1 << bit)):  # Bit 0 = free
                    free_clusters.append(cluster)
                    if len(free_clusters) >= count:
                        return free_clusters

        return None if len(free_clusters) < count else free_clusters

    def _mark_clusters_used(self, clusters: List[int]):
        """Mark clusters as used in the bitmap."""
        if not self.bitmap_clusters:
            return

        bitmap = self._read_bitmap()
        for cluster in clusters:
            byte_idx = cluster // 8
            bit = cluster % 8
            if byte_idx < len(bitmap):
                bitmap[byte_idx] |= (1 << bit)
        self._write_bitmap(bitmap)

    def _mark_clusters_free(self, clusters: List[int]):
        """Mark clusters as free in the bitmap."""
        if not self.bitmap_clusters:
            return

        bitmap = self._read_bitmap()
        for cluster in clusters:
            byte_idx = cluster // 8
            bit = cluster % 8
            if byte_idx < len(bitmap):
                bitmap[byte_idx] &= ~(1 << bit)
        self._write_bitmap(bitmap)

    def _update_mft_data_runs(self, record_num: int, data_runs: List[Tuple[int, int]],
                               file_size: int) -> bool:
        """Update the $DATA attribute's data runs in an MFT record.

        Args:
            record_num: MFT record number
            data_runs: List of (cluster_count, start_cluster) tuples
            file_size: Actual file size in bytes

        Returns:
            True if successful
        """
        from .data_runs import encode_data_runs

        record_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
        record = bytearray(self.image[record_offset:record_offset + MFT_RECORD_SIZE])

        if record[:4] != b'FILE':
            return False

        # Find the $DATA attribute
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
            if attr_type == 0x80 and name_len == 0:  # $DATA (unnamed)
                # Encode new data runs
                runs_bytes = encode_data_runs(data_runs)

                # Check if we need non-resident attribute
                total_clusters = sum(c for c, _ in data_runs)
                alloc_size = total_clusters * self.cluster_size

                # Build non-resident $DATA attribute
                # Attribute header (non-resident): 64 bytes header + data runs
                attr_size = 64 + len(runs_bytes)
                attr_size_aligned = (attr_size + 7) & ~7  # 8-byte aligned

                new_attr = bytearray(attr_size_aligned)
                struct.pack_into('<I', new_attr, 0, 0x80)  # Type
                struct.pack_into('<I', new_attr, 4, attr_size_aligned)  # Length
                new_attr[8] = 1  # Non-resident flag
                new_attr[9] = 0  # Name length
                struct.pack_into('<H', new_attr, 10, 0)  # Name offset
                struct.pack_into('<H', new_attr, 12, 0)  # Flags
                struct.pack_into('<H', new_attr, 14, 0)  # Instance
                struct.pack_into('<Q', new_attr, 16, 0)  # Start VCN
                struct.pack_into('<Q', new_attr, 24, total_clusters - 1 if total_clusters > 0 else 0)  # End VCN
                struct.pack_into('<H', new_attr, 32, 64)  # Data runs offset
                struct.pack_into('<H', new_attr, 34, 0)  # Compression unit
                struct.pack_into('<I', new_attr, 36, 0)  # Padding
                struct.pack_into('<Q', new_attr, 40, alloc_size)  # Allocated size
                struct.pack_into('<Q', new_attr, 48, file_size)  # Real size
                struct.pack_into('<Q', new_attr, 56, file_size)  # Initialized size
                # Runs start at offset 64
                new_attr[64:64 + len(runs_bytes)] = runs_bytes

                # Replace old attribute with new one
                old_attr_len = attr_len
                remaining = record[off + old_attr_len:]

                # Fit new attribute
                record[off:off + len(new_attr)] = new_attr
                record[off + len(new_attr):off + len(new_attr) + len(remaining)] = remaining

                # Update used size in record header
                new_used = off + len(new_attr) + len(remaining)
                struct.pack_into('<I', record[24:28], 0, new_used)

                # Apply fixups and write back
                self._apply_fixups(record)
                self.image[record_offset:record_offset + MFT_RECORD_SIZE] = record

                return True

            off += attr_len

        return False

    def _apply_fixups(self, record: bytearray):
        """Apply NTFS fixups to an MFT record before writing."""
        update_seq_off = struct.unpack('<H', record[4:6])[0]
        update_seq_count = struct.unpack('<H', record[6:8])[0]

        if update_seq_count < 2:
            return

        # Get the update sequence value
        seq_val = record[update_seq_off:update_seq_off + 2]

        # Apply to end of each sector
        for i in range(1, update_seq_count):
            sector_end = i * 512 - 2
            # Save current value to update sequence array
            record[update_seq_off + i * 2:update_seq_off + i * 2 + 2] = record[sector_end:sector_end + 2]
            # Write sequence value to sector end
            record[sector_end:sector_end + 2] = seq_val

    def allocate_file_direct(self, rel_path: str) -> bool:
        """Allocate clusters for a sparse file directly (no ntfs-3g).

        This updates:
        1. Cluster bitmap (marks clusters as used)
        2. MFT data runs (points to allocated clusters)
        3. cluster_map (routes reads to ext4 file)

        No data is copied - reads will return ext4 content.

        Returns True if successful.
        """
        if rel_path not in self.sparse_files:
            return False

        source_path, file_size, record_num = self.sparse_files[rel_path]

        # Calculate needed clusters
        needed_clusters = (file_size + self.cluster_size - 1) // self.cluster_size
        if needed_clusters == 0:
            return True  # Empty file, nothing to allocate

        log(f"  Direct alloc: {rel_path} ({needed_clusters} clusters)")

        # Find free clusters
        clusters = self._find_free_clusters(needed_clusters)
        if not clusters:
            log(f"  ERROR: Not enough free clusters for {rel_path}")
            return False

        # Mark clusters as used in bitmap
        self._mark_clusters_used(clusters)

        # Build data runs (try to make contiguous runs)
        from .data_runs import compress_cluster_list
        data_runs = compress_cluster_list(sorted(clusters))

        # Update MFT record with new data runs
        if not self._update_mft_data_runs(record_num, data_runs, file_size):
            # Rollback bitmap changes
            self._mark_clusters_free(clusters)
            log(f"  ERROR: Failed to update MFT for {rel_path}")
            return False

        # Update cluster_map to route these clusters to ext4 file
        offset = 0
        for cluster in sorted(clusters):
            self.cluster_map[cluster] = (source_path, offset)
            offset += self.cluster_size

        # Remove from sparse_files and sparse_file_clusters
        old_sparse_clusters = [c for c, p in self.sparse_file_clusters.items() if p == rel_path]
        for c in old_sparse_clusters:
            del self.sparse_file_clusters[c]
        del self.sparse_files[rel_path]

        # Track allocated clusters for deallocation
        if not hasattr(self, '_direct_allocated'):
            self._direct_allocated = {}
        self._direct_allocated[rel_path] = (source_path, file_size, record_num, clusters)

        log(f"  Direct alloc complete: {rel_path}")
        return True

    def deallocate_file_direct(self, rel_path: str) -> bool:
        """Deallocate clusters for a file (reverse of allocate_file_direct).

        Restores the file to sparse state.
        """
        if not hasattr(self, '_direct_allocated'):
            return False

        if rel_path not in self._direct_allocated:
            return False

        source_path, file_size, record_num, clusters = self._direct_allocated[rel_path]

        log(f"  Direct dealloc: {rel_path}")

        # Remove cluster mappings
        for cluster in clusters:
            self.cluster_map.pop(cluster, None)

        # Mark clusters as free in bitmap
        self._mark_clusters_free(clusters)

        # Restore MFT to sparse (single cluster at end for the \x00 byte)
        # This creates a sparse run followed by one allocated cluster
        last_cluster = clusters[-1] if clusters else 0
        # Sparse data runs: (cluster_count-1, 0) for sparse, (1, last_cluster) for allocated
        sparse_runs = []
        needed_clusters = (file_size + self.cluster_size - 1) // self.cluster_size
        if needed_clusters > 1:
            sparse_runs.append((needed_clusters - 1, 0))  # Sparse run
        sparse_runs.append((1, last_cluster))  # Keep one cluster allocated

        # Actually, for simplicity, let's just mark the last cluster
        self._mark_clusters_used([last_cluster])

        self._update_mft_data_runs(record_num, sparse_runs, file_size)

        # Re-add to sparse_files
        self.sparse_files[rel_path] = (source_path, file_size, record_num)
        self.sparse_file_clusters[last_cluster] = rel_path

        # Remove from direct_allocated
        del self._direct_allocated[rel_path]

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

        offset = self.mft_offset
        record_num = 0

        # First pass: find all directories
        while offset + MFT_RECORD_SIZE <= len(self.image):
            record = self.image[offset:offset + MFT_RECORD_SIZE]
            if record[0:4] != b'FILE':
                break

            record = self._undo_fixups(bytearray(record))
            flags = struct.unpack('<H', record[22:24])[0]

            if flags & 0x01 and flags & 0x02:  # In-use directory
                self._process_directory_record(record, record_num)

            offset += MFT_RECORD_SIZE
            record_num += 1

        # Second pass: find all files (both resident and non-resident)
        offset = self.mft_offset
        record_num = 0

        while offset + MFT_RECORD_SIZE <= len(self.image):
            record = self.image[offset:offset + MFT_RECORD_SIZE]
            if record[0:4] != b'FILE':
                break

            record = self._undo_fixups(bytearray(record))
            flags = struct.unpack('<H', record[22:24])[0]

            if flags & 0x01 and not (flags & 0x02):  # In-use file
                self._process_file_record(record, record_num)

            offset += MFT_RECORD_SIZE
            record_num += 1

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

        source_path = os.path.join(self.source_dir, rel_path)

        # Find source file
        if not os.path.isfile(source_path):
            found = self._find_source_file(filename)
            if not found:
                return
            source_path = found

        # Check for non-resident data (clusters)
        data_runs = self._extract_data_runs(record)
        if data_runs:
            cluster_count = sum(count for _, count in data_runs)

            # Check if this is a sparse file (allocated clusters << expected clusters)
            try:
                file_size = os.path.getsize(source_path)
                expected_clusters = (file_size + self.cluster_size - 1) // self.cluster_size
                is_sparse = cluster_count < expected_clusters // 2  # Less than half expected = sparse
            except OSError:
                is_sparse = False

            if is_sparse:
                # This is a sparse file - track it but don't map the minimal clusters
                self.sparse_files[rel_path] = (source_path, file_size, record_num)
                self.mft_record_to_source[record_num] = source_path
                # Record the allocated clusters so we can detect reads to them
                for start_cluster, count in data_runs:
                    if start_cluster > 0:  # Skip sparse runs (cluster 0)
                        for c in range(start_cluster, start_cluster + count):
                            self.sparse_file_clusters[c] = rel_path
                log(f"  Sparse file: {rel_path} ({cluster_count}/{expected_clusters} clusters, {file_size} bytes)")
            else:
                # Fully allocated file - map clusters
                if record_num not in self.mft_record_to_source:
                    log(f"  Mapping new file: {rel_path} (record {record_num}, {cluster_count} clusters)")
                self._map_clusters(data_runs, source_path)
                self.mft_record_to_source[record_num] = source_path
                # Remove from sparse tracking if it was there
                self.sparse_files.pop(rel_path, None)
        else:
            # Check for resident data (stored in MFT record)
            resident_loc = self._find_resident_data_location(record, record_num)
            if resident_loc:
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
        record_abs = self.mft_offset + record_num * MFT_RECORD_SIZE

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
        """Parse data runs into list of (cluster, count) tuples."""
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

        return result

    def _find_source_file(self, filename: str) -> Optional[str]:
        """Find matching file in source directory."""
        path = os.path.join(self.source_dir, filename)
        if os.path.isfile(path):
            return path

        for root, dirs, files in os.walk(self.source_dir):
            if filename in files:
                return os.path.join(root, filename)

        return None

    def _map_clusters(self, data_runs: List[Tuple[int, int]], source_path: str):
        """Map clusters from data runs to source file offsets."""
        if source_path not in self.source_to_clusters:
            self.source_to_clusters[source_path] = set()

        file_offset = 0
        for lcn, count in data_runs:
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
            rel_path = os.path.relpath(path, self.source_dir)
            self.path_to_mft_record[rel_path] = record_num

        for record_num, rel_path in self.mft_record_to_dir.items():
            if rel_path:
                self.path_to_mft_record[rel_path] = record_num

        # Build directory children
        for record_num, source_path in self.mft_record_to_source.items():
            rel_path = os.path.relpath(source_path, self.source_dir)
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

            # Read current ext4 content
            try:
                with open(source_path, 'rb') as f:
                    ext4_data = f.read()
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

        # Calculate which MFT records this read covers
        rel_offset = offset - self.mft_offset
        if rel_offset < 0:
            return

        start_record = rel_offset // MFT_RECORD_SIZE
        end_record = (rel_offset + length + MFT_RECORD_SIZE - 1) // MFT_RECORD_SIZE

        # Check each virtual record to see if it falls in this read
        for record_num in list(vfm.mft_to_virtual.keys()):
            if start_record <= record_num < end_record:
                # This virtual record is within our read range
                record_data = vfm.get_virtual_mft_record(record_num)
                if record_data:
                    # Calculate where this record should go in the result
                    record_abs_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
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

        rel_offset = offset - self.mft_offset
        if rel_offset < 0:
            return

        start_record = rel_offset // MFT_RECORD_SIZE
        end_record = (rel_offset + length + MFT_RECORD_SIZE - 1) // MFT_RECORD_SIZE

        # Log which directories we know about (first call only)
        if not hasattr(self, '_logged_dirs'):
            self._logged_dirs = True
            log(f"Known dirs: {list(self.mft_record_to_dir.items())}")

        # Check each directory record
        for record_num, dir_path in list(self.mft_record_to_dir.items()):
            if start_record <= record_num < end_record:
                # Get virtual children for this directory
                virtual_children = vfm.get_virtual_children(record_num)
                if virtual_children:
                    log(f"Dir {record_num} ({dir_path}) has {len(virtual_children)} virtual children: {[c.rel_path for c in virtual_children]}")

                if not virtual_children:
                    continue

                # Virtualize this directory
                record_abs_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
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

            # Determine sizes needed
            count_bytes = (count.bit_length() + 7) // 8
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
        max_tracked = max(self.mft_record_to_source.keys()) if self.mft_record_to_source else 64
        mft_end = self.mft_offset + max(256, max_tracked + 64) * MFT_RECORD_SIZE

        # Also consider virtual MFT records (which may be at higher record numbers)
        if self.virtual_file_manager:
            vfm = self.virtual_file_manager
            if vfm.mft_to_virtual:
                max_virtual = max(vfm.mft_to_virtual.keys())
                virtual_mft_end = self.mft_offset + (max_virtual + 1) * MFT_RECORD_SIZE
                mft_end = max(mft_end, virtual_mft_end)

        write_end = offset + length
        return not (write_end <= self.mft_offset or offset >= mft_end)

    def _handle_mft_write(self, offset: int, data: bytes):
        """Handle a write to the MFT region - re-parse affected records."""
        rel_offset = offset - self.mft_offset
        if rel_offset < 0:
            data = data[-rel_offset:]
            rel_offset = 0

        start_record = rel_offset // MFT_RECORD_SIZE
        end_offset = rel_offset + len(data)
        end_record = (end_offset + MFT_RECORD_SIZE - 1) // MFT_RECORD_SIZE

        log(f"MFT write: records {start_record}-{end_record - 1}")

        # Update image with new MFT data (skip managed dir records)
        template_offset = self.mft_offset + rel_offset
        if template_offset + len(data) <= len(self.image):
            for record_num in range(start_record, end_record):
                if record_num in self.dir_indx_clusters:
                    continue
                rec_start = record_num * MFT_RECORD_SIZE - rel_offset
                rec_end = rec_start + MFT_RECORD_SIZE
                if rec_start < 0:
                    rec_start = 0
                if rec_end > len(data):
                    rec_end = len(data)
                if rec_start < rec_end:
                    tpl_start = template_offset + rec_start
                    self.image[tpl_start:tpl_start + (rec_end - rec_start)] = data[rec_start:rec_end]

        # Process affected records
        for record_num in range(start_record, end_record):
            if record_num in self.mft_record_to_source:
                # Known file - check for deletion first, then reparse
                if not self._check_file_deleted(record_num):
                    self._reparse_mft_record(record_num)
            elif record_num in self.mft_record_to_dir:
                self._check_directory_rename(record_num)
            else:
                new_path = self._check_new_file(record_num)
                if not new_path:
                    self._check_new_directory(record_num)

    def _check_file_deleted(self, record_num: int) -> bool:
        """Check if a tracked file's MFT record was marked as deleted.

        Returns True if file was deleted.
        """
        record_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.image):
            return False

        record = self.image[record_offset:record_offset + MFT_RECORD_SIZE]
        if record[0:4] != b'FILE':
            return False

        flags = struct.unpack('<H', record[22:24])[0]
        if not (flags & 0x01):  # Not in use = deleted
            source_path = self.mft_record_to_source.get(record_num)
            if source_path:
                rel_path = os.path.relpath(source_path, self.source_dir)

                # Check loop prevention
                if rel_path in self.ext4_sync_in_progress:
                    log(f"  Skipping delete (ext4 sync in progress): {rel_path}")
                    # Still clean up tracking
                    del self.mft_record_to_source[record_num]
                    self.resident_file_data.pop(record_num, None)
                    return True

                # Remove tracking
                del self.mft_record_to_source[record_num]
                self.resident_file_data.pop(record_num, None)
                if source_path in self.source_to_clusters:
                    for cluster in self.source_to_clusters[source_path]:
                        if cluster in self.cluster_map:
                            del self.cluster_map[cluster]
                    del self.source_to_clusters[source_path]

                # Delete from ext4
                if os.path.exists(source_path):
                    self.ntfs_sync_in_progress.add(rel_path)
                    try:
                        os.remove(source_path)
                        log(f"  FILE DELETED: {rel_path}")
                    except OSError as e:
                        log(f"  Failed to delete {rel_path}: {e}")
                    finally:
                        self.ntfs_sync_in_progress.discard(rel_path)

                return True

        return False

    def _check_directory_rename(self, record_num: int):
        """Check if a tracked directory was renamed."""
        old_rel_path = self.mft_record_to_dir.get(record_num)
        if not old_rel_path:
            return

        record_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.image):
            return

        record = self._undo_fixups(bytearray(
            self.image[record_offset:record_offset + MFT_RECORD_SIZE]))

        if record[0:4] != b'FILE':
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

        if new_rel_path != old_rel_path:
            # Check loop prevention
            if new_rel_path in self.ext4_sync_in_progress:
                log(f"  Skipping dir rename (ext4 sync in progress): {new_rel_path}")
                self.mft_record_to_dir[record_num] = new_rel_path
                return

            old_path = os.path.join(self.source_dir, old_rel_path)
            new_path = os.path.join(self.source_dir, new_rel_path)

            try:
                if os.path.exists(old_path) and not os.path.exists(new_path):
                    self.ntfs_sync_in_progress.add(new_rel_path)
                    self.ntfs_sync_in_progress.add(old_rel_path)
                    try:
                        os.rename(old_path, new_path)
                        log(f"  DIR RENAMED: {old_rel_path} -> {new_rel_path}")
                    finally:
                        self.ntfs_sync_in_progress.discard(new_rel_path)
                        self.ntfs_sync_in_progress.discard(old_rel_path)
                self.mft_record_to_dir[record_num] = new_rel_path
            except OSError as e:
                log(f"  Failed to rename dir {old_rel_path}: {e}")

    def _check_new_directory(self, record_num: int):
        """Check if an MFT record is a new directory."""
        record_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.image):
            return

        record = self._undo_fixups(bytearray(
            self.image[record_offset:record_offset + MFT_RECORD_SIZE]))

        if record[0:4] != b'FILE':
            return

        flags = struct.unpack('<H', record[22:24])[0]
        if not (flags & 0x01) or not (flags & 0x02):
            return

        if record_num in self.mft_record_to_dir:
            return

        filename, parent_ref = self._extract_filename_and_parent(record)
        if not filename or filename.startswith('$'):
            return

        parent_record = parent_ref & 0xFFFFFFFFFFFF
        if parent_record == 5:
            rel_path = filename
        elif parent_record in self.mft_record_to_dir:
            parent_path = self.mft_record_to_dir[parent_record]
            rel_path = os.path.join(parent_path, filename) if parent_path else filename
        else:
            rel_path = filename

        # Check loop prevention
        if rel_path in self.ext4_sync_in_progress:
            log(f"  Skipping new dir (ext4 sync in progress): {rel_path}")
            self.mft_record_to_dir[record_num] = rel_path
            return

        source_path = os.path.join(self.source_dir, rel_path)
        try:
            if not os.path.exists(source_path):
                self.ntfs_sync_in_progress.add(rel_path)
                try:
                    os.makedirs(source_path, exist_ok=True)
                    log(f"  NEW DIR: {rel_path}")
                finally:
                    self.ntfs_sync_in_progress.discard(rel_path)

            self.mft_record_to_dir[record_num] = rel_path
        except OSError as e:
            log(f"  Failed to create dir {rel_path}: {e}")

    def _check_new_file(self, record_num: int) -> Optional[str]:
        """Check if an MFT record is a new file."""
        record_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.image):
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

        parent_record = parent_ref & 0xFFFFFFFFFFFF
        if parent_record == 5:
            rel_path = filename
        elif parent_record in self.mft_record_to_dir:
            parent_path = self.mft_record_to_dir[parent_record]
            rel_path = os.path.join(parent_path, filename) if parent_path else filename
        else:
            rel_path = filename

        source_path = os.path.join(self.source_dir, rel_path)

        # Check loop prevention
        if rel_path in self.ext4_sync_in_progress:
            log(f"  Skipping new file (ext4 sync in progress): {rel_path}")
            self.mft_record_to_source[record_num] = source_path
            self._track_file_data(record, record_num, source_path)
            return None

        if os.path.exists(source_path):
            # File already exists - just track and map
            self.mft_record_to_source[record_num] = source_path
            self._track_file_data(record, record_num, source_path)
            return None

        # Create the file in ext4
        try:
            parent_dir = os.path.dirname(source_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            self.ntfs_sync_in_progress.add(rel_path)
            try:
                # Extract resident data if present to write to ext4
                resident_data = self._extract_resident_data(record)
                with open(source_path, 'wb') as f:
                    if resident_data:
                        f.write(resident_data)
                log(f"  NEW FILE: {rel_path}")
            finally:
                self.ntfs_sync_in_progress.discard(rel_path)

            self.mft_record_to_source[record_num] = source_path
            self._track_file_data(record, record_num, source_path)

            return source_path
        except OSError as e:
            log(f"  Failed to create file {rel_path}: {e}")
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
        """Re-parse an MFT record for cluster updates, renames, and resident data."""
        source_path = self.mft_record_to_source.get(record_num)
        if not source_path:
            return

        record_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.image):
            return

        record = self._undo_fixups(bytearray(
            self.image[record_offset:record_offset + MFT_RECORD_SIZE]))

        if record[0:4] != b'FILE':
            return

        # Check for rename
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

            new_path = os.path.join(self.source_dir, new_rel_path)

            if new_path != source_path and os.path.exists(source_path):
                if new_rel_path not in self.ext4_sync_in_progress:
                    try:
                        parent_dir = os.path.dirname(new_path)
                        if parent_dir and not os.path.exists(parent_dir):
                            os.makedirs(parent_dir, exist_ok=True)

                        if not os.path.exists(new_path):
                            old_rel = os.path.relpath(source_path, self.source_dir)
                            self.ntfs_sync_in_progress.add(new_rel_path)
                            self.ntfs_sync_in_progress.add(old_rel)
                            try:
                                os.rename(source_path, new_path)
                                log(f"  FILE RENAMED: {os.path.basename(source_path)} -> {filename}")
                            finally:
                                self.ntfs_sync_in_progress.discard(new_rel_path)
                                self.ntfs_sync_in_progress.discard(old_rel)

                            source_path = new_path
                            self.mft_record_to_source[record_num] = new_path

                            if source_path in self.source_to_clusters:
                                clusters = self.source_to_clusters.pop(source_path)
                                self.source_to_clusters[new_path] = clusters
                                for cluster in clusters:
                                    if cluster in self.cluster_map:
                                        self.cluster_map[cluster] = (new_path, self.cluster_map[cluster][1])
                    except OSError as e:
                        log(f"  Failed to rename file: {e}")
                else:
                    source_path = new_path
                    self.mft_record_to_source[record_num] = new_path

        # Remove old cluster mappings
        if source_path in self.source_to_clusters:
            old_clusters = self.source_to_clusters[source_path]
            for cluster in old_clusters:
                if cluster in self.cluster_map:
                    del self.cluster_map[cluster]
            self.source_to_clusters[source_path] = set()

        # Extract new data runs or handle resident data
        data_runs = self._extract_data_runs(record)
        if data_runs:
            self._map_clusters(data_runs, source_path)
            # No longer resident
            self.resident_file_data.pop(record_num, None)
        else:
            # Resident data - extract and write to ext4 if changed
            rel_path = os.path.relpath(source_path, self.source_dir)
            resident_data = self._extract_resident_data(record)

            if resident_data is not None and rel_path not in self.ext4_sync_in_progress:
                # Only write if content actually changed (avoids cascading
                # rewrites when ntfs-3g flushes neighboring MFT records)
                try:
                    current_data = b''
                    if os.path.exists(source_path):
                        with open(source_path, 'rb') as f:
                            current_data = f.read()
                    if current_data != resident_data:
                        self.ntfs_sync_in_progress.add(rel_path)
                        try:
                            with open(source_path, 'wb') as f:
                                f.write(resident_data)
                        finally:
                            self.ntfs_sync_in_progress.discard(rel_path)
                except OSError as e:
                    log(f"  Error writing resident data: {e}")

            # Update resident tracking
            resident_loc = self._find_resident_data_location(record, record_num)
            if resident_loc:
                self.resident_file_data[record_num] = {
                    'source_path': source_path,
                    'val_len_abs': resident_loc[0],
                    'data_abs': resident_loc[1],
                    'available': resident_loc[2],
                }
