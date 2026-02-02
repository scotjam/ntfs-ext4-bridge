"""Template-based NTFS synthesizer with MFT tracking.

Uses a pre-populated NTFS template and swaps file content from ext4 at read time.
Tracks MFT changes to dynamically update cluster mappings when ntfs-3g reallocates.
"""
import os
import struct
from typing import Dict, List, Tuple, Optional, Set

MFT_RECORD_SIZE = 1024


def log(msg):
    """Debug logging."""
    print(f"[Synth] {msg}", flush=True)


class TemplateSynthesizer:
    """Synthesizes NTFS by swapping file content from ext4 source."""

    def __init__(self, template_path: str, source_dir: str):
        """
        Initialize with template NTFS and source directory.

        Args:
            template_path: Path to NTFS image with pre-created file structure
            source_dir: Path to ext4 directory with actual file content
        """
        self.source_dir = os.path.abspath(source_dir)

        # Load the template
        with open(template_path, 'rb') as f:
            self.template = bytearray(f.read())

        # Parse boot sector
        boot = self.template[0:512]
        self.bytes_per_sector = struct.unpack('<H', boot[0x0B:0x0D])[0]
        self.sectors_per_cluster = boot[0x0D]
        self.cluster_size = self.bytes_per_sector * self.sectors_per_cluster
        self.mft_cluster = struct.unpack('<Q', boot[0x30:0x38])[0]
        self.mft_offset = self.mft_cluster * self.cluster_size

        # Map of cluster -> (source_file_path, offset_in_file)
        self.cluster_map: Dict[int, Tuple[str, int]] = {}

        # MFT tracking: record_num -> source_file_path
        self.mft_record_to_source: Dict[int, str] = {}

        # Track which clusters belong to which MFT record (for cleanup on remap)
        self.source_to_clusters: Dict[str, Set[int]] = {}

        # Directory tracking: record_num -> directory path (relative to source_dir)
        self.mft_record_to_dir: Dict[int, str] = {}

        # Calculate MFT extent (may span multiple clusters)
        # We'll update this when we see MFT $DATA attribute
        self.mft_clusters: Set[int] = set()

        # Scan template MFT to find directories first, then files
        self._scan_template_mft()

        log(f"MFT tracking initialized: {len(self.mft_record_to_source)} files tracked")

        print(f"Template Synthesizer initialized:")
        print(f"  Cluster size: {self.cluster_size}")
        print(f"  Mapped clusters: {len(self.cluster_map)}")

    def _scan_template_mft(self):
        """Scan MFT in template to find directories and files."""
        offset = self.mft_offset
        record_num = 0

        # First pass: find all directories
        while offset + MFT_RECORD_SIZE <= len(self.template):
            record = self.template[offset:offset + MFT_RECORD_SIZE]

            if record[0:4] != b'FILE':
                break

            record = self._undo_fixups(bytearray(record))
            flags = struct.unpack('<H', record[22:24])[0]

            if flags & 0x01 and flags & 0x02:  # In-use and is directory
                self._process_directory_record(record, record_num)

            offset += MFT_RECORD_SIZE
            record_num += 1

        # Second pass: find all files
        offset = self.mft_offset
        record_num = 0

        while offset + MFT_RECORD_SIZE <= len(self.template):
            record = self.template[offset:offset + MFT_RECORD_SIZE]

            if record[0:4] != b'FILE':
                break

            record = self._undo_fixups(bytearray(record))
            flags = struct.unpack('<H', record[22:24])[0]

            if flags & 0x01 and not (flags & 0x02):  # In-use, not directory
                self._process_file_record(record, record_num)

            offset += MFT_RECORD_SIZE
            record_num += 1

    def _undo_fixups(self, record: bytearray) -> bytearray:
        """Undo USA fixups in an MFT record."""
        usa_offset = struct.unpack('<H', record[4:6])[0]
        usa_count = struct.unpack('<H', record[6:8])[0]
        for i in range(1, usa_count):
            sector_end = i * 512 - 2
            if usa_offset + i*2 + 2 <= MFT_RECORD_SIZE and sector_end + 2 <= MFT_RECORD_SIZE:
                original = struct.unpack('<H', record[usa_offset + i*2:usa_offset + i*2 + 2])[0]
                struct.pack_into('<H', record, sector_end, original)
        return record

    def _process_directory_record(self, record: bytearray, record_num: int):
        """Process a directory record to build directory mapping."""
        filename, parent_ref = self._extract_filename_and_parent(record)
        if not filename:
            return

        # Skip system directories
        if filename.startswith('$'):
            return

        # Root directory (record 5) maps to source_dir
        if record_num == 5:
            self.mft_record_to_dir[5] = ''
            return

        # Get parent path
        parent_record = parent_ref & 0xFFFFFFFFFFFF  # Lower 48 bits
        if parent_record == 5:
            # Direct child of root
            dir_path = filename
        elif parent_record in self.mft_record_to_dir:
            parent_path = self.mft_record_to_dir[parent_record]
            dir_path = os.path.join(parent_path, filename) if parent_path else filename
        else:
            # Parent not found yet, use just filename
            dir_path = filename

        self.mft_record_to_dir[record_num] = dir_path

    def _extract_filename_and_parent(self, record: bytearray) -> Tuple[Optional[str], int]:
        """Extract filename and parent directory reference from MFT record."""
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr
        filename = None
        parent_ref = 0

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off+4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off+4:off+8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            name_len = record[off+9]
            attr_name = ''
            if name_len > 0:
                name_offset = struct.unpack('<H', record[off+10:off+12])[0]
                attr_name = record[off+name_offset:off+name_offset+name_len*2].decode('utf-16-le', errors='ignore')

            if attr_type == 0x30 and not attr_name:  # $FILE_NAME
                val_len = struct.unpack('<I', record[off+16:off+20])[0]
                val_off = struct.unpack('<H', record[off+20:off+22])[0]
                fn_data = record[off+val_off:off+val_off+val_len]
                if len(fn_data) >= 66:
                    # Parent directory reference (first 8 bytes)
                    parent_ref = struct.unpack('<Q', fn_data[0:8])[0]
                    fn_len = fn_data[64]
                    fn_namespace = fn_data[65]
                    # Prefer Win32 or Win32+DOS namespace (1 or 3)
                    if fn_namespace in (1, 3) or filename is None:
                        filename = fn_data[66:66+fn_len*2].decode('utf-16-le', errors='ignore')

            off += attr_len

        return filename, parent_ref

    def _extract_data_runs(self, record: bytearray) -> Optional[List[Tuple[int, int]]]:
        """Extract data runs from MFT record's $DATA attribute."""
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off+4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off+4:off+8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            name_len = record[off+9]
            attr_name = ''
            if name_len > 0:
                name_offset = struct.unpack('<H', record[off+10:off+12])[0]
                attr_name = record[off+name_offset:off+name_offset+name_len*2].decode('utf-16-le', errors='ignore')

            if attr_type == 0x80 and not attr_name:  # $DATA (unnamed)
                non_res = record[off+8]
                if non_res:
                    runs_off = struct.unpack('<H', record[off+32:off+34])[0]
                    real_size = struct.unpack('<Q', record[off+48:off+56])[0]
                    runs = record[off+runs_off:off+attr_len]
                    return self._parse_data_runs(runs, real_size)
                break

            off += attr_len

        return None

    def _process_file_record(self, record: bytearray, record_num: int):
        """Process a file record to extract filename and data run mapping."""
        filename, parent_ref = self._extract_filename_and_parent(record)
        data_runs = self._extract_data_runs(record)

        # If we found a file with non-resident data, try to map it
        if filename and data_runs:
            source_path = self._find_source_file(filename)
            if source_path:
                self._map_clusters(data_runs, source_path)
                # Track MFT record -> source file mapping for dynamic updates
                self.mft_record_to_source[record_num] = source_path
                log(f"Tracking MFT record {record_num} -> {os.path.basename(source_path)}")

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

            run_length = int.from_bytes(runs[pos:pos+len_size], 'little')
            pos += len_size

            if off_size > 0:
                run_offset = int.from_bytes(runs[pos:pos+off_size], 'little', signed=True)
                pos += off_size
                current_lcn += run_offset
                result.append((current_lcn, run_length))

        return result

    def _find_source_file(self, filename: str) -> Optional[str]:
        """Find matching file in source directory."""
        # Try exact match in root
        path = os.path.join(self.source_dir, filename)
        if os.path.isfile(path):
            return path

        # Search recursively
        for root, dirs, files in os.walk(self.source_dir):
            if filename in files:
                return os.path.join(root, filename)

        return None

    def _map_clusters(self, data_runs: List[Tuple[int, int]], source_path: str):
        """Map clusters from data runs to source file offsets."""
        # Track clusters for this source file
        if source_path not in self.source_to_clusters:
            self.source_to_clusters[source_path] = set()

        file_offset = 0
        for lcn, count in data_runs:
            for i in range(count):
                cluster = lcn + i
                self.cluster_map[cluster] = (source_path, file_offset)
                self.source_to_clusters[source_path].add(cluster)
                file_offset += self.cluster_size

    def read(self, offset: int, length: int) -> bytes:
        """Read bytes from the synthesized volume."""
        result = bytearray(length)
        pos = 0

        while pos < length:
            byte_offset = offset + pos
            remaining = length - pos
            cluster = byte_offset // self.cluster_size
            cluster_offset = byte_offset % self.cluster_size

            if cluster in self.cluster_map:
                # Read from source file instead of template
                source_path, file_offset = self.cluster_map[cluster]
                read_offset = file_offset + cluster_offset
                chunk_len = min(remaining, self.cluster_size - cluster_offset)

                try:
                    with open(source_path, 'rb') as f:
                        f.seek(read_offset)
                        data = f.read(chunk_len)
                        # Pad with zeros if file is shorter
                        if len(data) < chunk_len:
                            data += b'\x00' * (chunk_len - len(data))
                        result[pos:pos+len(data)] = data
                except OSError:
                    pass  # Keep zeros on error

                pos += chunk_len

            elif byte_offset < len(self.template):
                # Read from template - but limit to current cluster to re-check map
                chunk_len = min(remaining, self.cluster_size - cluster_offset, len(self.template) - byte_offset)
                result[pos:pos+chunk_len] = self.template[byte_offset:byte_offset+chunk_len]
                pos += chunk_len

            else:
                # Beyond template - zeros
                chunk_len = min(remaining, self.cluster_size - cluster_offset)
                pos += chunk_len

        return bytes(result)

    def get_size(self) -> int:
        """Get total volume size."""
        return len(self.template)

    def is_mft_region(self, offset: int, length: int) -> bool:
        """Check if a write affects the MFT region."""
        # MFT starts at mft_offset
        # Check if write range overlaps with MFT
        # Cover a generous range to catch new file creation (256 records)
        max_tracked_record = max(self.mft_record_to_source.keys()) if self.mft_record_to_source else 64
        mft_end = self.mft_offset + max(256, max_tracked_record + 64) * MFT_RECORD_SIZE
        write_end = offset + length
        result = not (write_end <= self.mft_offset or offset >= mft_end)
        if result:
            log(f"Write at offset {offset} (len {length}) is in MFT region")
        return result

    def handle_mft_write(self, offset: int, data: bytes) -> List[str]:
        """Handle a write to the MFT region - re-parse affected records.

        Returns list of newly created file paths.
        """
        new_files = []

        # Calculate which MFT records are affected
        rel_offset = offset - self.mft_offset
        if rel_offset < 0:
            # Write starts before MFT
            data = data[-rel_offset:]
            rel_offset = 0

        start_record = rel_offset // MFT_RECORD_SIZE
        end_offset = rel_offset + len(data)
        end_record = (end_offset + MFT_RECORD_SIZE - 1) // MFT_RECORD_SIZE

        log(f"MFT write detected: records {start_record} to {end_record-1}")

        # Update template with new MFT data
        template_offset = self.mft_offset + rel_offset
        if template_offset + len(data) <= len(self.template):
            self.template[template_offset:template_offset + len(data)] = data

        # Process affected records
        for record_num in range(start_record, end_record):
            if record_num in self.mft_record_to_source:
                # Known file - reparse for cluster updates and check for rename
                self._reparse_mft_record(record_num)
            elif record_num in self.mft_record_to_dir:
                # Known directory - check for rename
                self._check_directory_rename(record_num)
            else:
                # Check if this is a new file or directory
                new_path = self._check_new_file(record_num)
                if new_path:
                    new_files.append(new_path)
                else:
                    self._check_new_directory(record_num)

        return new_files

    def _check_directory_rename(self, record_num: int):
        """Check if a tracked directory was renamed and update ext4."""
        old_rel_path = self.mft_record_to_dir.get(record_num)
        if not old_rel_path:
            return

        record_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.template):
            return

        record = self._undo_fixups(bytearray(
            self.template[record_offset:record_offset + MFT_RECORD_SIZE]
        ))

        if record[0:4] != b'FILE':
            return

        filename, parent_ref = self._extract_filename_and_parent(record)
        if not filename:
            return

        # Determine new path
        parent_record = parent_ref & 0xFFFFFFFFFFFF
        if parent_record == 5:
            new_rel_path = filename
        elif parent_record in self.mft_record_to_dir:
            parent_path = self.mft_record_to_dir[parent_record]
            new_rel_path = os.path.join(parent_path, filename) if parent_path else filename
        else:
            new_rel_path = filename

        # Check if renamed
        if new_rel_path != old_rel_path:
            old_path = os.path.join(self.source_dir, old_rel_path)
            new_path = os.path.join(self.source_dir, new_rel_path)

            try:
                if os.path.exists(old_path) and not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    self.mft_record_to_dir[record_num] = new_rel_path
                    log(f"  DIR RENAMED: {old_rel_path} -> {new_rel_path}")
            except OSError as e:
                log(f"  Failed to rename dir {old_rel_path}: {e}")

    def _check_new_directory(self, record_num: int):
        """Check if an MFT record is a new directory and create it in ext4."""
        record_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.template):
            return

        record = self._undo_fixups(bytearray(
            self.template[record_offset:record_offset + MFT_RECORD_SIZE]
        ))

        if record[0:4] != b'FILE':
            return

        flags = struct.unpack('<H', record[22:24])[0]

        # Check if in-use and IS a directory
        if not (flags & 0x01) or not (flags & 0x02):
            return

        # Already tracked?
        if record_num in self.mft_record_to_dir:
            return

        filename, parent_ref = self._extract_filename_and_parent(record)
        if not filename:
            return

        # Skip system directories
        if filename.startswith('$'):
            return

        # Determine the target path
        parent_record = parent_ref & 0xFFFFFFFFFFFF

        if parent_record == 5:
            rel_path = filename
        elif parent_record in self.mft_record_to_dir:
            parent_path = self.mft_record_to_dir[parent_record]
            rel_path = os.path.join(parent_path, filename) if parent_path else filename
        else:
            rel_path = filename

        source_path = os.path.join(self.source_dir, rel_path)

        # Create directory if it doesn't exist
        try:
            if not os.path.exists(source_path):
                os.makedirs(source_path, exist_ok=True)
                log(f"  NEW DIR created: {rel_path}")

            # Track this directory
            self.mft_record_to_dir[record_num] = rel_path

        except OSError as e:
            log(f"  Failed to create dir {rel_path}: {e}")

    def _check_new_file(self, record_num: int) -> Optional[str]:
        """Check if an MFT record is a new file and create it in ext4."""
        record_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.template):
            return None

        record = self._undo_fixups(bytearray(
            self.template[record_offset:record_offset + MFT_RECORD_SIZE]
        ))

        if record[0:4] != b'FILE':
            return None

        flags = struct.unpack('<H', record[22:24])[0]

        # Check if in-use and is a file (not directory)
        if not (flags & 0x01) or (flags & 0x02):
            return None

        filename, parent_ref = self._extract_filename_and_parent(record)
        if not filename:
            return None

        # Skip system files
        if filename.startswith('$'):
            return None

        # Determine the target path in ext4
        parent_record = parent_ref & 0xFFFFFFFFFFFF

        if parent_record == 5:
            # Direct child of root
            rel_path = filename
        elif parent_record in self.mft_record_to_dir:
            parent_path = self.mft_record_to_dir[parent_record]
            rel_path = os.path.join(parent_path, filename) if parent_path else filename
        else:
            # Unknown parent, put in root
            rel_path = filename

        source_path = os.path.join(self.source_dir, rel_path)

        # Check if file already exists
        if os.path.exists(source_path):
            log(f"  File already exists: {rel_path}")
            return None

        # Create the file
        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(source_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            # Create empty file
            with open(source_path, 'wb') as f:
                pass

            log(f"  NEW FILE created: {rel_path}")

            # Track this file
            self.mft_record_to_source[record_num] = source_path

            # Check for data runs and map clusters
            data_runs = self._extract_data_runs(record)
            if data_runs:
                self._map_clusters(data_runs, source_path)
                log(f"    Mapped {len(data_runs)} data runs")

            return source_path

        except OSError as e:
            log(f"  Failed to create file {rel_path}: {e}")
            return None

    def _reparse_mft_record(self, record_num: int):
        """Re-parse an MFT record and update cluster mappings, check for renames."""
        source_path = self.mft_record_to_source.get(record_num)
        if not source_path:
            return

        log(f"Re-parsing MFT record {record_num} for {os.path.basename(source_path)}")

        # Read the MFT record to check for rename
        record_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.template):
            return

        record = self._undo_fixups(bytearray(
            self.template[record_offset:record_offset + MFT_RECORD_SIZE]
        ))

        if record[0:4] != b'FILE':
            log(f"  Record {record_num} is not a valid FILE record")
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
                try:
                    # Ensure parent directory exists
                    parent_dir = os.path.dirname(new_path)
                    if parent_dir and not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)

                    if not os.path.exists(new_path):
                        os.rename(source_path, new_path)
                        log(f"  FILE RENAMED: {os.path.basename(source_path)} -> {filename}")
                        source_path = new_path
                        self.mft_record_to_source[record_num] = new_path

                        # Update source_to_clusters mapping
                        if source_path in self.source_to_clusters:
                            clusters = self.source_to_clusters.pop(source_path)
                            self.source_to_clusters[new_path] = clusters
                            for cluster in clusters:
                                if cluster in self.cluster_map:
                                    self.cluster_map[cluster] = (new_path, self.cluster_map[cluster][1])
                except OSError as e:
                    log(f"  Failed to rename file: {e}")

        # Remove old cluster mappings for this file
        if source_path in self.source_to_clusters:
            old_clusters = self.source_to_clusters[source_path]
            for cluster in old_clusters:
                if cluster in self.cluster_map:
                    del self.cluster_map[cluster]
            log(f"  Removed {len(old_clusters)} old cluster mappings")
            self.source_to_clusters[source_path] = set()

        # Find $DATA attribute (record already read and fixups undone above)
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr
        data_runs = None

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off+4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off+4:off+8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            name_len = record[off+9]
            attr_name = ''
            if name_len > 0:
                name_offset = struct.unpack('<H', record[off+10:off+12])[0]
                attr_name = record[off+name_offset:off+name_offset+name_len*2].decode('utf-16-le', errors='ignore')

            if attr_type == 0x80 and not attr_name:  # $DATA (unnamed)
                non_res = record[off+8]
                if non_res:
                    runs_off = struct.unpack('<H', record[off+32:off+34])[0]
                    real_size = struct.unpack('<Q', record[off+48:off+56])[0]
                    runs = record[off+runs_off:off+attr_len]
                    data_runs = self._parse_data_runs(runs, real_size)
                break

            off += attr_len

        if data_runs:
            self._map_clusters(data_runs, source_path)
            log(f"  Mapped {len(self.source_to_clusters.get(source_path, set()))} new clusters")
        else:
            # File is resident - extract data from MFT record and write to source
            resident_data = self._extract_resident_data(record)
            if resident_data is not None:
                log(f"  File is resident ({len(resident_data)} bytes) - writing to source")
                try:
                    with open(source_path, 'wb') as f:
                        f.write(resident_data)
                    log(f"  -> Resident data written to source")
                except OSError as e:
                    log(f"  -> Error writing resident data: {e}")
            else:
                log(f"  No data runs and no resident data found")

    def _extract_resident_data(self, record: bytearray) -> Optional[bytes]:
        """Extract resident data from a $DATA attribute."""
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off+4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off+4:off+8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            name_len = record[off+9]
            attr_name = ''
            if name_len > 0:
                name_offset = struct.unpack('<H', record[off+10:off+12])[0]
                attr_name = record[off+name_offset:off+name_offset+name_len*2].decode('utf-16-le', errors='ignore')

            if attr_type == 0x80 and not attr_name:  # $DATA (unnamed)
                non_res = record[off+8]
                if not non_res:  # Resident
                    val_len = struct.unpack('<I', record[off+16:off+20])[0]
                    val_off = struct.unpack('<H', record[off+20:off+22])[0]
                    return bytes(record[off+val_off:off+val_off+val_len])
                break

            off += attr_len

        return None
