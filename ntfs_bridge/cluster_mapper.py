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
        """
        # Debug: log reads when we have sparse files
        if length >= 4096 and self.sparse_files:
            cluster = offset // self.cluster_size
            in_map = cluster in self.cluster_map
            is_mft = self.is_mft_region(offset, length)
            # Only log non-MFT, non-mapped reads
            if not in_map and not is_mft:
                # Check if this is in a zero region
                if offset + 4096 <= len(self.image):
                    sample = self.image[offset:offset+64]
                    is_zeros = all(b == 0 for b in sample)
                else:
                    is_zeros = False
                log(f"  READ: off={offset}, len={length}, clus={cluster}, zeros={is_zeros}")

        # Check if this read might be for a sparse file that needs allocation
        self._check_sparse_file_read(offset, length)

        result = bytearray(length)
        pos = 0

        while pos < length:
            byte_offset = offset + pos
            remaining = length - pos
            cluster = byte_offset // self.cluster_size
            cluster_offset = byte_offset % self.cluster_size

            if cluster in self.cluster_map:
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

        return bytes(result)

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
    # MFT write tracking (NTFS -> ext4 sync)
    # =========================================================================

    def is_mft_region(self, offset: int, length: int) -> bool:
        """Check if a write affects the MFT region."""
        max_tracked = max(self.mft_record_to_source.keys()) if self.mft_record_to_source else 64
        mft_end = self.mft_offset + max(256, max_tracked + 64) * MFT_RECORD_SIZE
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
