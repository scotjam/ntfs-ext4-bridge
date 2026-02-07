"""Template-based NTFS synthesizer with MFT tracking.

Uses a pre-populated NTFS template and swaps file content from ext4 at read time.
Tracks MFT changes to dynamically update cluster mappings when ntfs-3g reallocates.
Supports dynamic file addition/removal from ext4 side.
"""
import os
import struct
import threading
import time
from typing import Dict, List, Tuple, Optional, Set

MFT_RECORD_SIZE = 1024
CLUSTER_SIZE = 4096  # Standard NTFS cluster size


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
        self.dir_indx_clusters: Dict[int, List[int]] = {}  # dir_record -> list of INDX cluster numbers
        self.dir_template_entries: Dict[int, List[Tuple[bytes, int]]] = {}  # dir_record -> cached template entries

        # MFT tracking: record_num -> source_file_path
        self.mft_record_to_source: Dict[int, str] = {}

        # Track which clusters belong to which MFT record (for cleanup on remap)
        self.source_to_clusters: Dict[str, Set[int]] = {}

        # Directory tracking: record_num -> directory path (relative to source_dir)
        self.mft_record_to_dir: Dict[int, str] = {}

        # Calculate MFT extent (may span multiple clusters)
        # We'll update this when we see MFT $DATA attribute
        self.mft_clusters: Set[int] = set()

        # Thread safety lock for all state modifications
        self.lock = threading.RLock()

        # Dynamic allocation tracking
        self.next_mft_record = 0  # Will be set after MFT scan
        self.next_free_cluster = 0  # Will be set after bitmap scan

        # Reverse mapping: rel_path -> mft_record_num
        self.path_to_mft_record: Dict[str, int] = {}

        # Directory children tracking: parent_record -> set of child records
        self.dir_children: Dict[int, Set[int]] = {}

        # Track removed MFT records (so we don't re-add them from template)
        self.removed_mft_records: Set[int] = set()

        # Scan template MFT to find directories first, then files
        self._scan_template_mft()

        # Initialize dynamic allocation state
        self._init_allocation_state()

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

        # Debug: log reads of INDX clusters for root directory
        start_cluster = offset // self.cluster_size
        end_cluster = (offset + length - 1) // self.cluster_size
        for c in range(start_cluster, end_cluster + 1):
            if c in self.cluster_map:
                mapping = self.cluster_map[c]
                if isinstance(mapping, tuple) and mapping[0] == 'bytes':
                    log(f"Reading INDX cluster {c} (offset {offset}, len {length})")

        while pos < length:
            byte_offset = offset + pos
            remaining = length - pos
            cluster = byte_offset // self.cluster_size
            cluster_offset = byte_offset % self.cluster_size

            if cluster in self.cluster_map:
                mapping = self.cluster_map[cluster]
                chunk_len = min(remaining, self.cluster_size - cluster_offset)

                if isinstance(mapping, tuple) and mapping[0] == 'bytes':
                    # Direct bytes data (e.g., INDEX_ALLOCATION blocks)
                    block_data = mapping[1]
                    read_offset = cluster_offset
                    data = block_data[read_offset:read_offset + chunk_len]
                    if len(data) < chunk_len:
                        data = data + b'\x00' * (chunk_len - len(data))
                    result[pos:pos+len(data)] = data
                else:
                    # Read from source file instead of template
                    source_path, file_offset = mapping
                    read_offset = file_offset + cluster_offset

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

        # Debug: if this read covers MFT record 5, log the INDEX_ALLOCATION data runs
        record_5_offset = self.mft_offset + 5 * 1024
        if offset <= record_5_offset < offset + length:
            rec5_start = record_5_offset - offset
            rec5_data = result[rec5_start:rec5_start + 1024]
            if len(rec5_data) >= 512:
                # Find INDEX_ALLOCATION (0xA0) attribute
                import struct as s
                first_attr = s.unpack('<H', rec5_data[20:22])[0]
                attr_off = first_attr
                while attr_off < 900:
                    attr_type = s.unpack('<I', rec5_data[attr_off:attr_off+4])[0]
                    if attr_type == 0xFFFFFFFF:
                        break
                    attr_len = s.unpack('<I', rec5_data[attr_off+4:attr_off+8])[0]
                    if attr_type == 0xA0 and rec5_data[attr_off + 8] == 1:  # Non-resident INDEX_ALLOCATION
                        dr_off = s.unpack('<H', rec5_data[attr_off+32:attr_off+34])[0]
                        dr_data = rec5_data[attr_off + dr_off:attr_off + dr_off + 10]
                        log(f"MFT record 5 INDEX_ALLOCATION data runs: {dr_data.hex()}")
                        break
                    if attr_len == 0:
                        break
                    attr_off += attr_len

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

        # Update template with new MFT data, but preserve synthesized directory records
        template_offset = self.mft_offset + rel_offset
        if template_offset + len(data) <= len(self.template):
            # Selectively update: skip records that have synthesized B+ trees
            for record_num in range(start_record, end_record):
                if record_num in self.dir_indx_clusters:
                    log(f"  Skipping template update for managed dir record {record_num}")
                    continue
                rec_start = record_num * MFT_RECORD_SIZE - rel_offset
                rec_end = rec_start + MFT_RECORD_SIZE
                if rec_start < 0:
                    rec_start = 0
                if rec_end > len(data):
                    rec_end = len(data)
                if rec_start < rec_end:
                    tpl_start = template_offset + rec_start
                    self.template[tpl_start:tpl_start + (rec_end - rec_start)] = data[rec_start:rec_end]

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

    # =========================================================================
    # Dynamic Allocation Methods (ext4 -> NTFS)
    # =========================================================================

    def _init_allocation_state(self):
        """Initialize dynamic allocation tracking after MFT scan."""
        # Find highest used MFT record by scanning template MFT
        # We need to find ALL in-use records, not just tracked ones
        highest_record = 15  # System records are 0-15
        offset = self.mft_offset
        record_num = 0

        while offset + MFT_RECORD_SIZE <= len(self.template):
            record = self.template[offset:offset + MFT_RECORD_SIZE]
            if record[0:4] != b'FILE':
                break

            # Check if in use
            flags = struct.unpack('<H', record[22:24])[0]
            if flags & 0x01:  # IN_USE
                highest_record = record_num

            offset += MFT_RECORD_SIZE
            record_num += 1

        # Also check tracked records (in case some weren't found in scan)
        all_records = set(self.mft_record_to_source.keys()) | set(self.mft_record_to_dir.keys())
        if all_records:
            highest_record = max(highest_record, max(all_records))

        # Start after highest used record, but never before system records
        self.next_mft_record = max(highest_record + 1, 24)

        # Build path -> record mapping
        for record_num, path in self.mft_record_to_source.items():
            rel_path = os.path.relpath(path, self.source_dir)
            self.path_to_mft_record[rel_path] = record_num

        for record_num, rel_path in self.mft_record_to_dir.items():
            if rel_path:  # Skip root
                self.path_to_mft_record[rel_path] = record_num

        # Build directory children mapping
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

        # Find next free cluster by scanning cluster bitmap
        self._scan_cluster_bitmap()

        log(f"Dynamic allocation initialized:")
        log(f"  Next MFT record: {self.next_mft_record}")
        log(f"  Next free cluster: {self.next_free_cluster}")

    def _get_parent_record(self, parent_rel_path: str) -> int:
        """Get MFT record number for a parent directory path."""
        if not parent_rel_path or parent_rel_path == '.':
            return 5  # Root directory
        return self.path_to_mft_record.get(parent_rel_path, 5)

    def _scan_cluster_bitmap(self):
        """Scan the cluster allocation bitmap to find free clusters."""
        # Find $Bitmap MFT record (record 6) and its data
        # For simplicity, start after a safe offset based on template size
        template_clusters = len(self.template) // self.cluster_size
        used_clusters = set(self.cluster_map.keys())

        # Find first free cluster after used area
        # Start after MFT region
        mft_clusters = (self.next_mft_record * MFT_RECORD_SIZE + self.cluster_size - 1) // self.cluster_size
        start_search = self.mft_cluster + mft_clusters + 10  # Some padding

        for cluster in range(start_search, template_clusters):
            if cluster not in used_clusters:
                self.next_free_cluster = cluster
                return

        # If no free cluster found, extend beyond template
        self.next_free_cluster = template_clusters

    def allocate_mft_record(self) -> int:
        """Allocate the next available MFT record number.

        Returns:
            The allocated MFT record number
        """
        with self.lock:
            record_num = self.next_mft_record
            self.next_mft_record += 1

            # Ensure template is large enough for new MFT record
            required_size = self.mft_offset + (self.next_mft_record * MFT_RECORD_SIZE)
            if required_size > len(self.template):
                # Extend template with zeros
                extension = required_size - len(self.template)
                self.template.extend(b'\x00' * extension)

            # Update MFT's $DATA attribute to reflect new record count
            self._update_mft_size(self.next_mft_record)

            return record_num

    def _update_mft_size(self, num_records: int):
        """Update MFT's $DATA attribute to reflect the number of valid records.

        This is necessary because ntfs-3g checks the MFT's real_size and
        initialized_size to determine how many records can be read.
        """
        # Read MFT record 0 (the $MFT file itself)
        record = bytearray(self.template[self.mft_offset:self.mft_offset + MFT_RECORD_SIZE])
        record = self._undo_fixups(record)

        if record[0:4] != b'FILE':
            return

        # Find $DATA attribute
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off+4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off+4:off+8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE - off:
                break

            if attr_type == 0x80:  # $DATA
                non_resident = record[off+8]
                if non_resident:
                    # Update real_size and initialized_size
                    new_size = num_records * MFT_RECORD_SIZE
                    alloc_size = struct.unpack('<Q', record[off+40:off+48])[0]

                    # Only update if we're within allocated size
                    if new_size <= alloc_size:
                        struct.pack_into('<Q', record, off+48, new_size)  # real_size
                        struct.pack_into('<Q', record, off+56, new_size)  # initialized_size

                        # Re-apply fixups and write back
                        usa_offset = struct.unpack('<H', record[4:6])[0]
                        usa_count = struct.unpack('<H', record[6:8])[0]
                        usn = struct.unpack('<H', record[usa_offset:usa_offset+2])[0]
                        new_usn = (usn + 1) & 0xFFFF
                        if new_usn == 0:
                            new_usn = 1
                        struct.pack_into('<H', record, usa_offset, new_usn)

                        for i in range(1, usa_count):
                            sector_end = (i * 512) - 2
                            if sector_end < MFT_RECORD_SIZE:
                                original = struct.unpack('<H', record[sector_end:sector_end+2])[0]
                                struct.pack_into('<H', record, usa_offset + (i * 2), original)
                                struct.pack_into('<H', record, sector_end, new_usn)

                        self.template[self.mft_offset:self.mft_offset + MFT_RECORD_SIZE] = record

                        # Also update MFT mirror ($MFTMirr stores first few records)
                        # MFTMirr is at a different location - find it from boot sector
                        self._update_mft_mirror(record)
                break

            off += attr_len

    def _update_mft_mirror(self, mft_record_0: bytearray):
        """Update the MFT mirror with the updated MFT record 0.

        The MFT mirror ($MFTMirr) contains copies of the first 4 MFT records.
        Its location is stored in the boot sector at offset 0x38.
        """
        # Get MFTMirr cluster from boot sector
        boot = self.template[0:512]
        mftmirr_cluster = struct.unpack('<Q', boot[0x38:0x40])[0]
        mftmirr_offset = mftmirr_cluster * self.cluster_size

        if mftmirr_offset + MFT_RECORD_SIZE <= len(self.template):
            # Copy the updated MFT record 0 to the mirror
            self.template[mftmirr_offset:mftmirr_offset + MFT_RECORD_SIZE] = mft_record_0

    def allocate_clusters(self, count: int) -> List[int]:
        """Allocate contiguous clusters for file data.

        Args:
            count: Number of clusters to allocate

        Returns:
            List of allocated cluster numbers
        """
        if count <= 0:
            return []

        with self.lock:
            clusters = []
            start_cluster = self.next_free_cluster

            for i in range(count):
                cluster = start_cluster + i
                clusters.append(cluster)

            self.next_free_cluster = start_cluster + count

            # Ensure template is large enough
            required_size = (self.next_free_cluster + 1) * self.cluster_size
            if required_size > len(self.template):
                extension = required_size - len(self.template)
                self.template.extend(b'\x00' * extension)

            return clusters

    def add_file(self, rel_path: str) -> Optional[int]:
        """Add a new file from ext4 to the NTFS view.

        Args:
            rel_path: Path relative to source_dir

        Returns:
            MFT record number if successful, None otherwise
        """
        with self.lock:
            source_path = os.path.join(self.source_dir, rel_path)

            if not os.path.exists(source_path):
                log(f"add_file: Source not found: {rel_path}")
                return None

            if rel_path in self.path_to_mft_record:
                log(f"add_file: Already tracked: {rel_path}")
                return self.path_to_mft_record[rel_path]

            is_dir = os.path.isdir(source_path)

            # Get file stats
            try:
                stat = os.stat(source_path)
                size = stat.st_size if not is_dir else 0
                ctime = stat.st_ctime
                mtime = stat.st_mtime
                atime = stat.st_atime
            except OSError as e:
                log(f"add_file: Failed to stat {rel_path}: {e}")
                return None

            # Find parent directory
            parent_path = os.path.dirname(rel_path)
            parent_record = self._get_parent_record(parent_path)

            # Allocate MFT record
            record_num = self.allocate_mft_record()
            filename = os.path.basename(rel_path)

            # Allocate clusters for file data (if needed)
            clusters = []
            if not is_dir and size > 0:
                cluster_count = (size + self.cluster_size - 1) // self.cluster_size
                clusters = self.allocate_clusters(cluster_count)

            # Create MFT record
            mft_record = self._create_mft_record(
                record_num=record_num,
                filename=filename,
                parent_record=parent_record,
                is_dir=is_dir,
                size=size,
                ctime=ctime,
                mtime=mtime,
                atime=atime,
                clusters=clusters
            )

            # Write MFT record to template
            record_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
            self.template[record_offset:record_offset + MFT_RECORD_SIZE] = mft_record

            # Update tracking
            self.path_to_mft_record[rel_path] = record_num

            if is_dir:
                self.mft_record_to_dir[record_num] = rel_path
                self.dir_children[record_num] = set()
            else:
                self.mft_record_to_source[record_num] = source_path
                if clusters:
                    self._map_clusters_from_list(clusters, source_path)

            # Add to parent's children
            if parent_record not in self.dir_children:
                self.dir_children[parent_record] = set()
            self.dir_children[parent_record].add(record_num)

            # Update parent directory's $INDEX_ROOT
            self._update_directory_index(parent_record)

            log(f"add_file: Added {'directory' if is_dir else 'file'} {rel_path} as MFT record {record_num}")
            return record_num

    def remove_file(self, rel_path: str) -> bool:
        """Remove a file from the NTFS view.

        Args:
            rel_path: Path relative to source_dir

        Returns:
            True if successfully removed
        """
        with self.lock:
            if rel_path not in self.path_to_mft_record:
                log(f"remove_file: Not tracked: {rel_path}")
                return False

            record_num = self.path_to_mft_record[rel_path]

            # Get parent for index update
            parent_path = os.path.dirname(rel_path)
            parent_record = self._get_parent_record(parent_path)

            # Mark MFT record as not in use (clear IN_USE flag)
            record_offset = self.mft_offset + record_num * MFT_RECORD_SIZE
            if record_offset + MFT_RECORD_SIZE <= len(self.template):
                # Read flags and clear IN_USE bit
                flags = struct.unpack('<H', self.template[record_offset + 22:record_offset + 24])[0]
                flags &= ~0x01  # Clear IN_USE flag
                struct.pack_into('<H', self.template, record_offset + 22, flags)

            # Free clusters
            source_path = self.mft_record_to_source.get(record_num)
            if source_path and source_path in self.source_to_clusters:
                clusters = self.source_to_clusters.pop(source_path)
                for cluster in clusters:
                    if cluster in self.cluster_map:
                        del self.cluster_map[cluster]

            # Remove from tracking
            del self.path_to_mft_record[rel_path]
            if record_num in self.mft_record_to_source:
                del self.mft_record_to_source[record_num]
            if record_num in self.mft_record_to_dir:
                del self.mft_record_to_dir[record_num]
            if record_num in self.dir_children:
                del self.dir_children[record_num]

            # Track as removed (so we don't resurrect it from template entries)
            self.removed_mft_records.add(record_num)

            # Remove from parent's children
            if parent_record in self.dir_children:
                self.dir_children[parent_record].discard(record_num)

            # Update parent directory's $INDEX_ROOT
            self._update_directory_index(parent_record)

            log(f"remove_file: Removed {rel_path} (MFT record {record_num})")
            return True

    def _map_clusters_from_list(self, clusters: List[int], source_path: str):
        """Map a list of clusters to a source file."""
        if source_path not in self.source_to_clusters:
            self.source_to_clusters[source_path] = set()

        for i, cluster in enumerate(clusters):
            file_offset = i * self.cluster_size
            self.cluster_map[cluster] = (source_path, file_offset)
            self.source_to_clusters[source_path].add(cluster)

    def _create_mft_record(self, record_num: int, filename: str, parent_record: int,
                           is_dir: bool, size: int, ctime: float, mtime: float,
                           atime: float, clusters: List[int]) -> bytes:
        """Create a complete MFT record."""
        record = bytearray(MFT_RECORD_SIZE)

        # FILE signature
        record[0:4] = b'FILE'

        # USA offset and count
        usa_offset = 48
        usa_count = 3  # 1 USN + 2 sector fixups for 1024-byte record
        struct.pack_into('<H', record, 4, usa_offset)
        struct.pack_into('<H', record, 6, usa_count)

        # $LogFile sequence number
        struct.pack_into('<Q', record, 8, 0)

        # Sequence number
        struct.pack_into('<H', record, 16, 1)

        # Hard link count
        struct.pack_into('<H', record, 18, 1)

        # First attribute offset (aligned to 8 after USA)
        first_attr_offset = (usa_offset + usa_count * 2 + 7) & ~7
        struct.pack_into('<H', record, 20, first_attr_offset)

        # Flags
        flags = 0x01  # IN_USE
        if is_dir:
            flags |= 0x02  # IS_DIRECTORY
        struct.pack_into('<H', record, 22, flags)

        # Allocated size
        struct.pack_into('<I', record, 28, MFT_RECORD_SIZE)

        # MFT record number
        struct.pack_into('<I', record, 44, record_num)

        # Write USA
        usn = 1
        struct.pack_into('<H', record, usa_offset, usn)

        # Build attributes
        attr_offset = first_attr_offset

        # $STANDARD_INFORMATION (0x10)
        std_info = self._create_standard_info(ctime, mtime, atime, is_dir)
        record[attr_offset:attr_offset + len(std_info)] = std_info
        attr_offset += len(std_info)

        # $FILE_NAME (0x30)
        file_name = self._create_file_name(filename, parent_record, ctime, mtime, atime, size, is_dir)
        record[attr_offset:attr_offset + len(file_name)] = file_name
        attr_offset += len(file_name)

        # $DATA (0x80) for files, $INDEX_ROOT (0x90) for directories
        if is_dir:
            index_root = self._create_empty_index_root()
            record[attr_offset:attr_offset + len(index_root)] = index_root
            attr_offset += len(index_root)
        else:
            data_attr = self._create_data_attr(size, clusters)
            record[attr_offset:attr_offset + len(data_attr)] = data_attr
            attr_offset += len(data_attr)

        # End marker
        struct.pack_into('<I', record, attr_offset, 0xFFFFFFFF)
        attr_offset += 4

        # Update used size
        struct.pack_into('<I', record, 24, attr_offset)

        # Apply fixups
        for i in range(1, usa_count):
            sector_end = (i * 512) - 2
            if sector_end < MFT_RECORD_SIZE:
                original = struct.unpack('<H', record[sector_end:sector_end + 2])[0]
                struct.pack_into('<H', record, usa_offset + (i * 2), original)
                struct.pack_into('<H', record, sector_end, usn)

        return bytes(record)

    def _create_standard_info(self, ctime: float, mtime: float, atime: float, is_dir: bool) -> bytes:
        """Create $STANDARD_INFORMATION attribute."""
        # Convert times to NTFS format (100ns intervals since 1601-01-01)
        EPOCH_DIFF = 11644473600
        ctime_ntfs = int((ctime + EPOCH_DIFF) * 10000000)
        mtime_ntfs = int((mtime + EPOCH_DIFF) * 10000000)
        atime_ntfs = int((atime + EPOCH_DIFF) * 10000000)

        data = bytearray(48)
        struct.pack_into('<Q', data, 0, ctime_ntfs)   # Creation time
        struct.pack_into('<Q', data, 8, mtime_ntfs)   # Modification time
        struct.pack_into('<Q', data, 16, mtime_ntfs)  # MFT modification time
        struct.pack_into('<Q', data, 24, atime_ntfs)  # Access time

        file_attrs = 0x10 if is_dir else 0x20  # DIRECTORY or ARCHIVE
        struct.pack_into('<I', data, 32, file_attrs)

        # Build attribute header
        header = bytearray(24)
        total_len = (24 + len(data) + 7) & ~7
        struct.pack_into('<I', header, 0, 0x10)       # Type
        struct.pack_into('<I', header, 4, total_len)  # Length
        header[8] = 0                                  # Resident
        struct.pack_into('<I', header, 16, len(data)) # Value length
        struct.pack_into('<H', header, 20, 24)        # Value offset

        result = bytearray(total_len)
        result[:24] = header
        result[24:24+len(data)] = data
        return bytes(result)

    def _create_file_name(self, filename: str, parent_record: int, ctime: float,
                          mtime: float, atime: float, size: int, is_dir: bool) -> bytes:
        """Create $FILE_NAME attribute."""
        EPOCH_DIFF = 11644473600
        ctime_ntfs = int((ctime + EPOCH_DIFF) * 10000000)
        mtime_ntfs = int((mtime + EPOCH_DIFF) * 10000000)
        atime_ntfs = int((atime + EPOCH_DIFF) * 10000000)

        name_utf16 = filename.encode('utf-16-le')
        name_len = len(filename)

        # $FILE_NAME attribute data
        data = bytearray(66 + len(name_utf16))

        # Parent directory MFT reference (record + sequence)
        parent_ref = (parent_record & 0xFFFFFFFFFFFF) | (1 << 48)
        struct.pack_into('<Q', data, 0, parent_ref)

        struct.pack_into('<Q', data, 8, ctime_ntfs)   # Creation time
        struct.pack_into('<Q', data, 16, mtime_ntfs)  # Modification time
        struct.pack_into('<Q', data, 24, mtime_ntfs)  # MFT modification time
        struct.pack_into('<Q', data, 32, atime_ntfs)  # Access time

        alloc_size = ((size + self.cluster_size - 1) // self.cluster_size) * self.cluster_size if size > 0 else 0
        struct.pack_into('<Q', data, 40, alloc_size)  # Allocated size
        struct.pack_into('<Q', data, 48, size)        # Real size

        file_attrs = 0x10 if is_dir else 0x20
        struct.pack_into('<I', data, 56, file_attrs)  # Flags

        data[64] = name_len                           # Filename length
        data[65] = 3                                  # Namespace (Win32+DOS)
        data[66:66+len(name_utf16)] = name_utf16      # Filename

        # Build attribute header
        header = bytearray(24)
        total_len = (24 + len(data) + 7) & ~7
        struct.pack_into('<I', header, 0, 0x30)       # Type
        struct.pack_into('<I', header, 4, total_len)  # Length
        header[8] = 0                                  # Resident
        struct.pack_into('<I', header, 16, len(data)) # Value length
        struct.pack_into('<H', header, 20, 24)        # Value offset
        header[22] = 1                                 # Indexed

        result = bytearray(total_len)
        result[:24] = header
        result[24:24+len(data)] = data
        return bytes(result)

    def _create_data_attr(self, size: int, clusters: List[int]) -> bytes:
        """Create $DATA attribute."""
        if size == 0 or not clusters:
            # Resident empty data
            header = bytearray(24)
            total_len = 24
            struct.pack_into('<I', header, 0, 0x80)       # Type
            struct.pack_into('<I', header, 4, total_len)  # Length
            header[8] = 0                                  # Resident
            struct.pack_into('<I', header, 16, 0)         # Value length
            struct.pack_into('<H', header, 20, 24)        # Value offset
            return bytes(header)

        # Non-resident data
        data_runs = self._encode_data_runs(clusters)

        header = bytearray(64)
        total_len = (64 + len(data_runs) + 7) & ~7

        struct.pack_into('<I', header, 0, 0x80)       # Type
        struct.pack_into('<I', header, 4, total_len)  # Length
        header[8] = 1                                  # Non-resident

        alloc_size = len(clusters) * self.cluster_size
        ending_vcn = len(clusters) - 1 if clusters else 0

        struct.pack_into('<Q', header, 16, 0)              # Starting VCN
        struct.pack_into('<Q', header, 24, ending_vcn)     # Ending VCN
        struct.pack_into('<H', header, 32, 64)             # Data runs offset
        struct.pack_into('<Q', header, 40, alloc_size)     # Allocated size
        struct.pack_into('<Q', header, 48, size)           # Real size
        struct.pack_into('<Q', header, 56, size)           # Initialized size

        result = bytearray(total_len)
        result[:64] = header
        result[64:64+len(data_runs)] = data_runs
        return bytes(result)

    def _encode_data_runs(self, clusters: List[int]) -> bytes:
        """Encode cluster list into NTFS data runs format."""
        if not clusters:
            return b'\x00'

        # Compress consecutive clusters into runs
        runs = []
        run_start = clusters[0]
        run_length = 1

        for i in range(1, len(clusters)):
            if clusters[i] == clusters[i-1] + 1:
                run_length += 1
            else:
                runs.append((run_length, run_start))
                run_start = clusters[i]
                run_length = 1
        runs.append((run_length, run_start))

        # Encode runs
        encoded = bytearray()
        prev_lcn = 0

        for length, lcn in runs:
            lcn_offset = lcn - prev_lcn

            # Bytes needed for length (unsigned)
            len_bytes = (length.bit_length() + 7) // 8
            if len_bytes == 0:
                len_bytes = 1

            # Bytes needed for offset (signed)
            if lcn_offset >= 0:
                off_bytes = (lcn_offset.bit_length() + 8) // 8
            else:
                off_bytes = ((-lcn_offset - 1).bit_length() + 8) // 8
            if off_bytes == 0:
                off_bytes = 1

            header = (off_bytes << 4) | len_bytes
            encoded.append(header)
            encoded.extend(length.to_bytes(len_bytes, 'little', signed=False))
            encoded.extend(lcn_offset.to_bytes(off_bytes, 'little', signed=True))

            prev_lcn = lcn

        encoded.append(0x00)  # Terminator
        return bytes(encoded)

    def _create_empty_index_root(self) -> bytes:
        """Create an empty $INDEX_ROOT attribute for a directory."""
        # Index root header (16 bytes)
        index_header = bytearray(16)
        struct.pack_into('<I', index_header, 0, 0x30)   # Indexed attr type ($FILE_NAME)
        struct.pack_into('<I', index_header, 4, 1)     # Collation rule
        struct.pack_into('<I', index_header, 8, 4096)  # Index block size
        index_header[12] = 1                            # Clusters per index block

        # Node header (16 bytes) + end entry (16 bytes)
        node_header = bytearray(16)
        struct.pack_into('<I', node_header, 0, 16)     # Offset to first entry
        struct.pack_into('<I', node_header, 4, 32)     # Total entries size
        struct.pack_into('<I', node_header, 8, 32)     # Allocated size
        node_header[12] = 0                             # Flags (small index)

        # End entry
        end_entry = bytearray(16)
        struct.pack_into('<H', end_entry, 8, 16)       # Entry length
        struct.pack_into('<I', end_entry, 12, 2)       # Flags (end)

        data = index_header + node_header + end_entry

        # Attribute header with $I30 name
        name_bytes = "$I30".encode('utf-16-le')
        name_offset = 24
        value_offset = (name_offset + len(name_bytes) + 7) & ~7
        total_len = (value_offset + len(data) + 7) & ~7

        header = bytearray(total_len)
        struct.pack_into('<I', header, 0, 0x90)        # Type ($INDEX_ROOT)
        struct.pack_into('<I', header, 4, total_len)   # Length
        header[8] = 0                                   # Resident
        header[9] = 4                                   # Name length
        struct.pack_into('<H', header, 10, name_offset) # Name offset
        struct.pack_into('<I', header, 16, len(data))  # Value length
        struct.pack_into('<H', header, 20, value_offset) # Value offset

        header[name_offset:name_offset+len(name_bytes)] = name_bytes
        header[value_offset:value_offset+len(data)] = data

        return bytes(header)

    def _update_directory_index(self, dir_record: int):
        """Update a directory's $INDEX_ROOT with current children.

        Note: This implementation supports up to ~30 entries (small index).
        Directories with more files need B+ tree ($INDEX_ALLOCATION).

        For template directories (like root), we merge existing entries with
        new dynamic entries. For directories we created, we rebuild from scratch.
        """
        if dir_record not in self.dir_children:
            return

        # Don't modify system metadata directories (records 0-4, 6-15)
        # Record 5 is root directory - we CAN and SHOULD modify it
        if dir_record < 5 or (dir_record > 5 and dir_record <= 15):
            return

        dynamic_children = self.dir_children.get(dir_record, set())
        dir_path = self.mft_record_to_dir.get(dir_record, "(unknown)")
        log(f"_update_directory_index: record {dir_record} ({dir_path}), {len(dynamic_children)} children: {dynamic_children}")

        # For the root directory or any template directory, we need to
        # preserve existing entries and merge with new dynamic entries
        entries = []
        dynamic_mft_records = set(dynamic_children)

        # Check if this is a template directory (existed before we started adding files)
        # Root (record 5) is always a template directory
        is_template_dir = (dir_record == 5) or (dir_record not in self.mft_record_to_dir)
        log(f"_update_directory_index: record {dir_record}: is_template_dir={is_template_dir}")

        if is_template_dir:
            # Get existing entries from template, excluding any we're replacing
            # Cache template entries on first read (before B+ tree modifies INDEX_ROOT)
            if dir_record not in self.dir_template_entries:
                self.dir_template_entries[dir_record] = self._get_existing_index_entries(dir_record)
                log(f"_update_directory_index: cached {len(self.dir_template_entries[dir_record])} template entries for record {dir_record}")
            existing_entries = self.dir_template_entries[dir_record]
            for entry_bytes, child_mft in existing_entries:
                # Skip if this entry is for a file we're dynamically managing
                if child_mft in dynamic_mft_records:
                    continue
                # Also skip if entry is for a file we've removed
                if child_mft in self.removed_mft_records:
                    continue
                entries.append(entry_bytes)

        # Add entries for dynamic children
        for child_record in sorted(dynamic_children):
            entry = self._create_index_entry(child_record)
            if entry:
                entries.append(entry)

        # Sort entries by filename (uppercase for NTFS collation)
        entries.sort(key=lambda e: self._get_entry_filename(e).upper())

        # Check if the total entries would exceed what INDEX_ROOT can hold
        # INDEX_ROOT can hold about 600 bytes of entries in a typical MFT record
        total_size = sum(len(e) for e in entries)

        if total_size > 600:
            # Use B+ tree (INDEX_ALLOCATION) for large directories
            log(f"_update_directory_index: {len(entries)} entries ({total_size} bytes), using B+ tree")
            self._update_directory_with_btree(dir_record, entries)
            return

        # Build new $INDEX_ROOT for small directory
        new_index_root = self._build_index_root(entries)

        # Find and replace $INDEX_ROOT in directory's MFT record
        self._replace_index_root(dir_record, new_index_root)

    def _create_index_entry(self, child_record: int) -> Optional[bytes]:
        """Create an index entry for a child file/directory."""
        # Get filename and attributes from child's MFT record
        record_offset = self.mft_offset + child_record * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.template):
            return None

        record = self._undo_fixups(bytearray(
            self.template[record_offset:record_offset + MFT_RECORD_SIZE]
        ))

        if record[0:4] != b'FILE':
            return None

        # Extract $FILE_NAME attribute data
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off+4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off+4:off+8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE:
                break

            if attr_type == 0x30:  # $FILE_NAME
                val_len = struct.unpack('<I', record[off+16:off+20])[0]
                val_off = struct.unpack('<H', record[off+20:off+22])[0]
                fn_data = bytes(record[off+val_off:off+val_off+val_len])

                # Debug: extract file attributes and filename
                if len(fn_data) >= 66:
                    file_attrs = struct.unpack('<I', fn_data[56:60])[0]
                    fn_len = fn_data[64]
                    filename = fn_data[66:66+fn_len*2].decode('utf-16-le', errors='ignore')
                    is_dir = bool(file_attrs & 0x10)
                    log(f"_create_index_entry: record {child_record} '{filename}': file_attrs=0x{file_attrs:08x} is_dir={is_dir}")

                # Build index entry
                entry_len = (16 + val_len + 7) & ~7
                entry = bytearray(entry_len)

                # MFT reference (child record + sequence)
                mft_ref = (child_record & 0xFFFFFFFFFFFF) | (1 << 48)
                struct.pack_into('<Q', entry, 0, mft_ref)
                struct.pack_into('<H', entry, 8, entry_len)
                struct.pack_into('<H', entry, 10, val_len)
                struct.pack_into('<I', entry, 12, 0)  # Flags
                entry[16:16+val_len] = fn_data

                return bytes(entry)

            off += attr_len

        return None

    def _get_existing_index_entries(self, dir_record: int) -> List[Tuple[bytes, int]]:
        """Extract existing index entries from a directory's INDEX_ROOT.

        Returns list of (entry_bytes, child_mft_record) tuples.
        This allows us to preserve template entries when adding new files.

        Note: Entries are converted to leaf format (HAS_SUBNODES flag and VCN
        pointers stripped) so they can be used in flat INDX blocks.
        """
        entries = []
        record_offset = self.mft_offset + dir_record * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.template):
            return entries

        record = self._undo_fixups(bytearray(
            self.template[record_offset:record_offset + MFT_RECORD_SIZE]
        ))

        if record[0:4] != b'FILE':
            return entries

        # Find $INDEX_ROOT attribute
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr
        has_index_allocation = False

        # First pass: check for INDEX_ALLOCATION
        scan_off = first_attr
        while scan_off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[scan_off:scan_off+4])[0]
            if attr_type == 0xFFFFFFFF:
                break
            attr_len = struct.unpack('<I', record[scan_off+4:scan_off+8])[0]
            if attr_len == 0:
                break
            if attr_type == 0xA0:  # $INDEX_ALLOCATION
                has_index_allocation = True
                break
            scan_off += attr_len

        if has_index_allocation:
            # Template uses B+ tree - read entries from INDEX_ALLOCATION
            # For now, read from the raw template at the cluster locations
            return self._get_entries_from_index_allocation(dir_record, record)

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off+4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off+4:off+8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE - off:
                break

            if attr_type == 0x90:  # $INDEX_ROOT
                # Parse the INDEX_ROOT to extract entries
                # Attribute header
                resident = record[off + 8]
                if resident != 0:  # Must be resident
                    break

                value_len = struct.unpack('<I', record[off+16:off+20])[0]
                value_off = struct.unpack('<H', record[off+20:off+22])[0]
                data_start = off + value_off
                data_end = data_start + value_len

                if data_end > MFT_RECORD_SIZE:
                    break

                # INDEX_ROOT structure:
                # [0:16] Index root header (attr_type, collation, block_size, clusters)
                # [16:32] Node header (entries_offset, index_size, alloc_size, flags)
                # [32+] Index entries

                index_data = record[data_start:data_end]
                if len(index_data) < 32:
                    break

                entries_offset = struct.unpack('<I', index_data[16:20])[0]

                # Parse entries starting at entries_offset within the node
                # Node header is at offset 16 in index_data
                entry_start = 16 + entries_offset
                while entry_start < len(index_data) - 16:
                    entry_len = struct.unpack('<H', index_data[entry_start+8:entry_start+10])[0]
                    entry_flags = struct.unpack('<I', index_data[entry_start+12:entry_start+16])[0]

                    if entry_len == 0:
                        break

                    if entry_flags & 2:  # INDEX_ENTRY_END
                        break

                    # Extract child MFT reference (lower 48 bits)
                    mft_ref = struct.unpack('<Q', index_data[entry_start:entry_start+8])[0]
                    child_mft = mft_ref & 0xFFFFFFFFFFFF

                    # Copy entry, stripping HAS_SUBNODES flag and VCN pointer if present
                    entry_bytes = self._strip_entry_subnode(
                        bytes(index_data[entry_start:entry_start+entry_len]),
                        entry_flags
                    )
                    entries.append((entry_bytes, child_mft))

                    entry_start += entry_len

                break

            off += attr_len

        return entries

    def _strip_entry_subnode(self, entry: bytes, flags: int) -> bytes:
        """Strip HAS_SUBNODES flag and VCN pointer from an index entry.

        Converts an interior node entry to a leaf entry format.
        """
        HAS_SUBNODES = 0x01

        if not (flags & HAS_SUBNODES):
            # Already a leaf entry
            return entry

        # Entry has a VCN pointer at the end (8 bytes)
        # Remove it and clear the HAS_SUBNODES flag
        entry_len = struct.unpack('<H', entry[8:10])[0]
        key_len = struct.unpack('<H', entry[10:12])[0]

        # Calculate new entry length without VCN pointer
        # New length = old length - 8 (VCN), aligned to 8 bytes
        new_len = entry_len - 8
        new_len = (new_len + 7) & ~7

        new_entry = bytearray(new_len)
        # Copy MFT reference
        new_entry[0:8] = entry[0:8]
        # Set new length
        struct.pack_into('<H', new_entry, 8, new_len)
        # Copy key length
        struct.pack_into('<H', new_entry, 10, key_len)
        # Clear flags (no HAS_SUBNODES, no INDEX_ENTRY_END)
        struct.pack_into('<I', new_entry, 12, 0)
        # Copy key data (FILE_NAME)
        new_entry[16:16+key_len] = entry[16:16+key_len]

        return bytes(new_entry)

    def _get_entries_from_index_allocation(self, dir_record: int, record: bytearray) -> List[Tuple[bytes, int]]:
        """Extract entries from INDEX_ROOT and INDEX_ALLOCATION blocks (B+ tree).

        This handles the case where the template directory uses B+ tree indexing.
        Reads both INDEX_ROOT intermediate entries (split keys) and INDX block entries.
        """
        entries = []

        # First pass: extract INDEX_ROOT split key entries (these are real file entries too)
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off+4])[0]
            if attr_type == 0xFFFFFFFF:
                break
            attr_len = struct.unpack('<I', record[off+4:off+8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE - off:
                break

            if attr_type == 0x90:  # $INDEX_ROOT
                if record[off + 8] == 0:  # Resident
                    value_len = struct.unpack('<I', record[off+16:off+20])[0]
                    value_off = struct.unpack('<H', record[off+20:off+22])[0]
                    data_start = off + value_off
                    index_data = record[data_start:data_start + value_len]

                    if len(index_data) >= 32:
                        entries_offset = struct.unpack('<I', index_data[16:20])[0]
                        entry_start = 16 + entries_offset

                        while entry_start < len(index_data) - 16:
                            entry_len = struct.unpack('<H', index_data[entry_start+8:entry_start+10])[0]
                            entry_flags = struct.unpack('<I', index_data[entry_start+12:entry_start+16])[0]
                            if entry_len == 0:
                                break
                            if entry_flags & 2:  # INDEX_ENTRY_END
                                break
                            # This is a split key entry - extract it
                            mft_ref = struct.unpack('<Q', index_data[entry_start:entry_start+8])[0]
                            child_mft = mft_ref & 0xFFFFFFFFFFFF
                            entry_bytes = self._strip_entry_subnode(
                                bytes(index_data[entry_start:entry_start+entry_len]),
                                entry_flags
                            )
                            entries.append((entry_bytes, child_mft))
                            if entry_len > 80:
                                fn_len = index_data[entry_start + 80]
                                fn_start = entry_start + 82
                                fn_end = fn_start + fn_len * 2
                                if fn_end <= entry_start + entry_len:
                                    filename = index_data[fn_start:fn_end].decode('utf-16-le', errors='ignore')
                                    log(f"  Read INDEX_ROOT split key: MFT {child_mft} '{filename}'")
                            entry_start += entry_len
                break
            off += attr_len

        # Second pass: extract entries from INDEX_ALLOCATION INDX blocks
        off = first_attr

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off+4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off+4:off+8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE - off:
                break

            if attr_type == 0xA0:  # $INDEX_ALLOCATION
                # Non-resident attribute
                if record[off + 8] != 1:  # Must be non-resident
                    break

                # Get data runs offset
                data_runs_offset = struct.unpack('<H', record[off+32:off+34])[0]
                lowest_vcn = struct.unpack('<Q', record[off+16:off+24])[0]
                highest_vcn = struct.unpack('<Q', record[off+24:off+32])[0]

                # Parse data runs to get cluster numbers
                clusters = self._decode_data_runs_to_clusters(record[off + data_runs_offset:off + attr_len])

                # Read each INDX block from cluster_map or template
                for vcn, cluster in enumerate(clusters):
                    # Check cluster_map first (for our synthesized INDX blocks)
                    if cluster in self.cluster_map:
                        mapping = self.cluster_map[cluster]
                        if isinstance(mapping, tuple) and mapping[0] == 'bytes':
                            block = bytearray(mapping[1])
                        else:
                            # File mapping, not INDX data - skip
                            continue
                    else:
                        # Read from template
                        block_offset = cluster * self.cluster_size
                        if block_offset + 4096 > len(self.template):
                            continue
                        block = bytearray(self.template[block_offset:block_offset + 4096])

                    # Undo fixups in INDX block
                    if block[0:4] != b'INDX':
                        continue

                    usa_offset = struct.unpack('<H', block[4:6])[0]
                    usa_count = struct.unpack('<H', block[6:8])[0]

                    for i in range(1, usa_count):
                        sector_end = (i * 512) - 2
                        if usa_offset + i*2 + 2 <= 4096 and sector_end + 2 <= 4096:
                            original = struct.unpack('<H', block[usa_offset + i*2:usa_offset + i*2 + 2])[0]
                            struct.pack_into('<H', block, sector_end, original)

                    # Parse entries from this INDX block
                    # Node header is at offset 24
                    node_offset = 24
                    entries_offset = struct.unpack('<I', block[node_offset:node_offset+4])[0]
                    entry_start = node_offset + entries_offset

                    while entry_start < 4096 - 16:
                        entry_len = struct.unpack('<H', block[entry_start+8:entry_start+10])[0]
                        entry_flags = struct.unpack('<I', block[entry_start+12:entry_start+16])[0]

                        if entry_len == 0:
                            break

                        if entry_flags & 2:  # INDEX_ENTRY_END
                            break

                        mft_ref = struct.unpack('<Q', block[entry_start:entry_start+8])[0]
                        child_mft = mft_ref & 0xFFFFFFFFFFFF
                        seq_num = (mft_ref >> 48) & 0xFFFF

                        # Debug: extract filename and log entry
                        if entry_len > 90:
                            fn_len = block[entry_start + 80]
                            fn_start = entry_start + 82
                            fn_end = fn_start + fn_len * 2
                            if fn_end <= entry_start + entry_len:
                                filename = block[fn_start:fn_end].decode('utf-16-le', errors='ignore')
                                log(f"  Read entry: MFT {child_mft} seq {seq_num} '{filename}'")

                        # Strip subnode info for leaf format
                        entry_bytes = self._strip_entry_subnode(
                            bytes(block[entry_start:entry_start+entry_len]),
                            entry_flags
                        )
                        entries.append((entry_bytes, child_mft))

                        entry_start += entry_len

                break

            off += attr_len

        return entries

    def _decode_data_runs_to_clusters(self, data: bytes) -> List[int]:
        """Decode NTFS data runs to get a flat list of cluster numbers."""
        clusters = []
        pos = 0
        current_cluster = 0

        while pos < len(data) and data[pos] != 0:
            header = data[pos]
            pos += 1

            len_size = header & 0x0F
            offset_size = (header >> 4) & 0x0F

            if pos + len_size + offset_size > len(data):
                break

            # Read run length
            run_length = int.from_bytes(data[pos:pos+len_size], 'little')
            pos += len_size

            # Read offset (signed)
            if offset_size > 0:
                offset_bytes = data[pos:pos+offset_size]
                if offset_bytes[-1] & 0x80:  # Negative
                    offset = int.from_bytes(offset_bytes, 'little', signed=False)
                    offset -= (1 << (offset_size * 8))
                else:
                    offset = int.from_bytes(offset_bytes, 'little')
                pos += offset_size
                current_cluster += offset
            else:
                # Sparse run
                pass

            # Add clusters
            for i in range(run_length):
                clusters.append(current_cluster + i)

        return clusters

    def _get_entry_filename(self, entry: bytes) -> str:
        """Extract filename from an index entry."""
        if len(entry) < 18:
            return ""
        key_len = struct.unpack('<H', entry[10:12])[0]
        if len(entry) < 16 + key_len or key_len < 66:
            return ""
        fn_len = entry[16 + 64]
        fn_start = 16 + 66
        fn_end = fn_start + fn_len * 2
        if fn_end > len(entry):
            return ""
        return entry[fn_start:fn_end].decode('utf-16-le', errors='ignore')

    def _build_index_root(self, entries: List[bytes]) -> bytes:
        """Build $INDEX_ROOT attribute with given entries."""
        # Index root header
        index_header = bytearray(16)
        struct.pack_into('<I', index_header, 0, 0x30)   # Indexed attr type
        struct.pack_into('<I', index_header, 4, 1)     # Collation rule
        struct.pack_into('<I', index_header, 8, 4096)  # Index block size
        index_header[12] = 1

        # Combine entries
        entries_data = bytearray()
        for entry in entries:
            entries_data.extend(entry)

        # End entry
        end_entry = bytearray(16)
        struct.pack_into('<H', end_entry, 8, 16)
        struct.pack_into('<I', end_entry, 12, 2)  # INDEX_ENTRY_END
        entries_data.extend(end_entry)

        # Node header
        node_header = bytearray(16)
        entries_offset = 16
        total_size = entries_offset + len(entries_data)
        struct.pack_into('<I', node_header, 0, entries_offset)
        struct.pack_into('<I', node_header, 4, total_size)
        struct.pack_into('<I', node_header, 8, total_size)
        node_header[12] = 0  # Small index (no INDEX_ALLOCATION)

        data = index_header + node_header + entries_data

        # Build attribute with $I30 name
        name_bytes = "$I30".encode('utf-16-le')
        name_offset = 24
        value_offset = (name_offset + len(name_bytes) + 7) & ~7
        total_len = (value_offset + len(data) + 7) & ~7

        result = bytearray(total_len)
        struct.pack_into('<I', result, 0, 0x90)
        struct.pack_into('<I', result, 4, total_len)
        result[8] = 0
        result[9] = 4
        struct.pack_into('<H', result, 10, name_offset)
        struct.pack_into('<I', result, 16, len(data))
        struct.pack_into('<H', result, 20, value_offset)

        result[name_offset:name_offset+len(name_bytes)] = name_bytes
        result[value_offset:value_offset+len(data)] = data

        return bytes(result)

    def _build_index_root_large(self, split_keys: List[Tuple[bytes, int]] = None) -> bytes:
        """Build $INDEX_ROOT for large directory (points to INDEX_ALLOCATION).

        Args:
            split_keys: List of (entry_bytes, left_child_vcn) tuples.
                        Each split key entry points to a child with entries < this key.
                        If None, creates a simple root pointing to VCN=0.

        In a proper B+ tree with N leaf blocks:
        - INDEX_ROOT contains (N-1) split key entries + 1 end entry
        - split_key[i] has VCN pointer to block i (entries < split_key[i])
        - end entry has VCN pointer to block N-1 (entries > all split keys)
        """
        # Index root header
        index_header = bytearray(16)
        struct.pack_into('<I', index_header, 0, 0x30)   # Indexed attr type ($FILE_NAME)
        struct.pack_into('<I', index_header, 4, 1)     # Collation rule (FILENAME)
        struct.pack_into('<I', index_header, 8, 4096)  # Index block size
        index_header[12] = 1  # Clusters per index block

        # Build intermediate entries (split keys with VCN pointers)
        intermediate_data = bytearray()
        last_vcn = 0  # VCN for the end entry (right-most child)

        if split_keys:
            for entry_bytes, child_vcn in split_keys:
                # Convert entry to have HAS_SUBNODES flag and VCN pointer
                entry = bytearray(entry_bytes)
                old_len = struct.unpack('<H', entry[8:10])[0]
                new_len = old_len + 8  # Add 8 bytes for VCN
                struct.pack_into('<H', entry, 8, new_len)  # Update entry length
                # Set HAS_SUBNODES flag (bit 0)
                old_flags = struct.unpack('<I', entry[12:16])[0]
                struct.pack_into('<I', entry, 12, old_flags | 1)  # INDEX_ENTRY_NODE
                # Append VCN at end
                vcn_bytes = struct.pack('<Q', child_vcn)
                entry = entry[:old_len] + bytearray(vcn_bytes)
                intermediate_data += entry

            last_vcn = len(split_keys)  # Last block VCN
            log(f"_build_index_root_large: {len(split_keys)} split keys, end VCN={last_vcn}")

        # Node header
        node_header = bytearray(16)
        entries_offset = 16

        # End entry with VCN pointer to last child block
        end_entry = bytearray(24)
        struct.pack_into('<H', end_entry, 8, 24)  # Entry length
        struct.pack_into('<I', end_entry, 12, 3)  # INDEX_ENTRY_END | HAS_SUBNODES
        struct.pack_into('<Q', end_entry, 16, last_vcn)  # VCN of right-most child

        entries_data = intermediate_data + end_entry
        total_size = entries_offset + len(entries_data)
        struct.pack_into('<I', node_header, 0, entries_offset)
        struct.pack_into('<I', node_header, 4, total_size)
        struct.pack_into('<I', node_header, 8, total_size)
        node_header[12] = 1  # LARGE_INDEX flag - has INDEX_ALLOCATION

        data = index_header + node_header + entries_data

        # Build attribute with $I30 name
        name_bytes = "$I30".encode('utf-16-le')
        name_offset = 24
        value_offset = (name_offset + len(name_bytes) + 7) & ~7
        total_len = (value_offset + len(data) + 7) & ~7

        result = bytearray(total_len)
        struct.pack_into('<I', result, 0, 0x90)  # $INDEX_ROOT
        struct.pack_into('<I', result, 4, total_len)
        result[8] = 0  # Resident
        result[9] = 4  # Name length
        struct.pack_into('<H', result, 10, name_offset)
        struct.pack_into('<I', result, 16, len(data))
        struct.pack_into('<H', result, 20, value_offset)

        result[name_offset:name_offset+len(name_bytes)] = name_bytes
        result[value_offset:value_offset+len(data)] = data

        return bytes(result)

    def _build_index_block(self, entries: List[bytes], vcn: int, is_leaf: bool = True) -> bytes:
        """Build a single INDX block for INDEX_ALLOCATION.

        Args:
            entries: List of index entries to include
            vcn: Virtual Cluster Number of this block
            is_leaf: True if this is a leaf node (no child pointers)

        Returns:
            4096-byte INDX block with USA fixups applied

        INDX block structure (4096 bytes):
        - 0x00-0x03: "INDX" signature
        - 0x04-0x05: USA offset (0x28 = 40)
        - 0x06-0x07: USA count (9 for 4KB block)
        - 0x08-0x0F: $LogFile LSN
        - 0x10-0x17: VCN of this block
        - 0x18-0x27: Index node header (16 bytes)
        - 0x28-0x39: USA array (18 bytes)
        - 0x40+: Index entries start (8-byte aligned after USA)
        """
        block = bytearray(4096)

        # INDX signature
        block[0:4] = b'INDX'

        # USA offset and count (for 4096 byte block with 512 byte sectors)
        usa_offset = 0x28  # 40
        usa_count = 9  # 1 USN + 8 sector fixups (18 bytes total)
        struct.pack_into('<H', block, 4, usa_offset)
        struct.pack_into('<H', block, 6, usa_count)

        # $LogFile sequence number
        struct.pack_into('<Q', block, 8, 0)

        # VCN of this index block
        struct.pack_into('<Q', block, 16, vcn)

        # Index node header at offset 0x18 (24)
        node_header_offset = 0x18  # 24

        # USA ends at offset 40 + 18 = 58, round up to 64 for 8-byte alignment
        # Entries start at offset 0x40 (64)
        entries_start_abs = 0x40  # 64
        entries_offset = entries_start_abs - node_header_offset  # 40 (relative to node header)

        # Write entries
        current_offset = entries_start_abs
        for entry in entries:
            if current_offset + len(entry) > 4096 - 24:  # Leave room for end entry
                break
            block[current_offset:current_offset + len(entry)] = entry
            current_offset += len(entry)

        # Write end entry (leaf block - no VCN pointers)
        end_entry = bytearray(16)
        struct.pack_into('<H', end_entry, 8, 16)  # Entry length
        struct.pack_into('<I', end_entry, 12, 2)  # INDEX_ENTRY_END only
        block[current_offset:current_offset + 16] = end_entry
        current_offset += 16

        # Calculate sizes for node header
        # index_length: total size of entries area (from entries_offset to end of entries)
        index_length = current_offset - node_header_offset
        # allocated_size: total allocated space for entries area
        allocated_size = 4096 - node_header_offset

        # Write index node header at offset 0x18
        struct.pack_into('<I', block, node_header_offset + 0, entries_offset)   # Offset to first entry
        struct.pack_into('<I', block, node_header_offset + 4, index_length)     # Total size used
        struct.pack_into('<I', block, node_header_offset + 8, allocated_size)   # Allocated size
        # Flags: 0 = leaf node (entries don't have child pointers, only end entry may have sibling VCN)
        block[node_header_offset + 12] = 0 if is_leaf else 1

        # Apply USA fixups - save original bytes at sector boundaries, write USN
        usn = 1
        struct.pack_into('<H', block, usa_offset, usn)

        for i in range(1, usa_count):
            sector_end = (i * 512) - 2
            if sector_end < 4096:
                original = struct.unpack('<H', block[sector_end:sector_end + 2])[0]
                struct.pack_into('<H', block, usa_offset + (i * 2), original)
                struct.pack_into('<H', block, sector_end, usn)

        # Debug: verify block structure
        log(f"_build_index_block: VCN={vcn}, entries_offset={entries_offset}, index_length={index_length}, allocated={allocated_size}")

        return bytes(block)

    def _build_index_allocation(self, entries: List[bytes]) -> Tuple[bytes, int, List[Tuple[bytes, int]]]:
        """Build $INDEX_ALLOCATION attribute with INDX blocks.

        Uses proper B+ tree structure: entries are split into leaf blocks,
        with split keys extracted for the INDEX_ROOT intermediate node.

        Args:
            entries: List of all index entries (sorted)

        Returns:
            Tuple of (index_data, num_clusters, split_keys)
            split_keys: List of (entry_bytes, child_vcn) for INDEX_ROOT
        """
        # Calculate how many blocks we need
        # Each block can hold ~3900 bytes of entries (4096 - headers - USA)
        max_entries_per_block = 3800

        # Split entries into groups for each block
        entry_groups = []
        current_entries = []
        current_size = 0

        for entry in entries:
            if current_size + len(entry) > max_entries_per_block and current_entries:
                entry_groups.append(current_entries)
                current_entries = []
                current_size = 0

            current_entries.append(entry)
            current_size += len(entry)

        if current_entries:
            entry_groups.append(current_entries)

        # Extract split keys: last entry of each group (except the last group)
        # becomes a split key in INDEX_ROOT. Remove it from the leaf block.
        split_keys = []
        for i in range(len(entry_groups) - 1):
            # Take the last entry from this group as the split key
            split_entry = entry_groups[i].pop()  # Remove from leaf
            split_keys.append((split_entry, i))  # (entry_bytes, left_child_vcn)
            log(f"_build_index_allocation: Split key for VCN={i}: {len(split_entry)} bytes")

        # Build leaf INDX blocks (no VCN pointers, just entries + end marker)
        blocks = []
        for i, group in enumerate(entry_groups):
            blocks.append(self._build_index_block(group, i))

        # Debug
        log(f"_build_index_allocation: Split {len(entries)} entries into {len(blocks)} leaf blocks, {len(split_keys)} split keys")

        # Concatenate all blocks
        index_data = b''.join(blocks)
        num_clusters = len(blocks)  # Each block is 4096 bytes = 1 cluster

        # Build non-resident attribute header for $INDEX_ALLOCATION
        name_bytes = "$I30".encode('utf-16-le')
        name_offset = 64  # Non-resident header is larger
        data_runs_offset = (name_offset + len(name_bytes) + 7) & ~7

        # Return data, cluster count, and split keys for INDEX_ROOT
        return index_data, num_clusters, split_keys

    def _build_index_bitmap(self, num_blocks: int) -> bytes:
        """Build $BITMAP attribute for INDEX_ALLOCATION.

        Args:
            num_blocks: Number of index blocks allocated

        Returns:
            Resident $BITMAP attribute bytes
        """
        # Bitmap data - one bit per block, all set to 1 (allocated)
        num_bytes = (num_blocks + 7) // 8
        bitmap_data = bytearray(num_bytes)

        for i in range(num_blocks):
            bitmap_data[i // 8] |= (1 << (i % 8))

        # Build resident attribute
        name_bytes = "$I30".encode('utf-16-le')
        name_offset = 24
        value_offset = (name_offset + len(name_bytes) + 7) & ~7
        total_len = (value_offset + len(bitmap_data) + 7) & ~7

        result = bytearray(total_len)
        struct.pack_into('<I', result, 0, 0xB0)  # $BITMAP
        struct.pack_into('<I', result, 4, total_len)
        result[8] = 0  # Resident
        result[9] = 4  # Name length
        struct.pack_into('<H', result, 10, name_offset)
        struct.pack_into('<I', result, 16, len(bitmap_data))
        struct.pack_into('<H', result, 20, value_offset)

        result[name_offset:name_offset+len(name_bytes)] = name_bytes
        result[value_offset:value_offset+len(bitmap_data)] = bitmap_data

        return bytes(result)

    def _build_index_allocation_attr(self, clusters: List[int]) -> bytes:
        """Build the $INDEX_ALLOCATION attribute header (non-resident).

        Args:
            clusters: List of cluster numbers (may be non-contiguous)

        Returns:
            Non-resident attribute header bytes
        """
        num_clusters = len(clusters)
        name_bytes = "$I30".encode('utf-16-le')

        # Encode data runs (handles non-contiguous clusters)
        data_runs = self._encode_data_runs(clusters)
        log(f"_build_index_allocation_attr: clusters={clusters} data_runs={data_runs.hex()}")

        # Non-resident attribute header
        name_offset = 64
        data_runs_offset = (name_offset + len(name_bytes) + 7) & ~7
        total_len = (data_runs_offset + len(data_runs) + 7) & ~7

        result = bytearray(total_len)
        struct.pack_into('<I', result, 0, 0xA0)  # $INDEX_ALLOCATION
        struct.pack_into('<I', result, 4, total_len)
        result[8] = 1  # Non-resident
        result[9] = 4  # Name length
        struct.pack_into('<H', result, 10, name_offset)
        struct.pack_into('<H', result, 14, 0)  # Flags

        # Non-resident specific fields
        data_size = num_clusters * 4096
        struct.pack_into('<Q', result, 16, 0)  # Lowest VCN
        struct.pack_into('<Q', result, 24, num_clusters - 1)  # Highest VCN
        struct.pack_into('<H', result, 32, data_runs_offset)  # Data runs offset
        struct.pack_into('<Q', result, 40, data_size)  # Allocated size
        struct.pack_into('<Q', result, 48, data_size)  # Real size
        struct.pack_into('<Q', result, 56, data_size)  # Initialized size

        result[name_offset:name_offset+len(name_bytes)] = name_bytes
        result[data_runs_offset:data_runs_offset+len(data_runs)] = data_runs

        return bytes(result)

    def _update_directory_with_btree(self, dir_record: int, entries: List[bytes]):
        """Update directory using B+ tree (INDEX_ALLOCATION) for large directories.

        Args:
            dir_record: MFT record number of directory
            entries: List of index entries (already sorted)
        """
        log(f"_update_directory_with_btree: Building B+ tree for record {dir_record} with {len(entries)} entries")

        # Build INDEX_ALLOCATION data (INDX blocks) with split keys for B+ tree
        index_data, num_clusters, split_keys = self._build_index_allocation(entries)

        if num_clusters == 0:
            log(f"_update_directory_with_btree: No clusters needed")
            return

        # Reuse existing INDX clusters if possible, allocate more if needed
        old_clusters = self.dir_indx_clusters.get(dir_record, [])
        if len(old_clusters) >= num_clusters:
            # Reuse existing clusters (remove extras from cluster_map)
            index_clusters = old_clusters[:num_clusters]
            for c in old_clusters[num_clusters:]:
                if c in self.cluster_map:
                    del self.cluster_map[c]
        elif old_clusters:
            # Need more clusters - keep old ones and allocate additional
            extra = self.allocate_clusters(num_clusters - len(old_clusters))
            index_clusters = old_clusters + extra
        else:
            # First time - allocate all
            index_clusters = self.allocate_clusters(num_clusters)

        if len(index_clusters) < num_clusters:
            log(f"_update_directory_with_btree: Failed to get {num_clusters} clusters")
            return

        # Track INDX clusters for this directory
        self.dir_indx_clusters[dir_record] = index_clusters

        # Write INDEX_ALLOCATION data to cluster map (overwrite in-place)
        for i, cluster in enumerate(index_clusters):
            block_offset = i * 4096
            block_data = index_data[block_offset:block_offset + 4096]
            self.cluster_map[cluster] = ('bytes', block_data)

        log(f"_update_directory_with_btree: INDX clusters for record {dir_record}: {index_clusters} (reused={len(old_clusters)})")

        # Build the three attributes (INDEX_ROOT with split keys for proper B+ tree)
        index_root = self._build_index_root_large(split_keys)
        index_alloc = self._build_index_allocation_attr(index_clusters)
        bitmap = self._build_index_bitmap(num_clusters)

        # Replace all index attributes in MFT record
        self._replace_directory_index_attrs(dir_record, index_root, index_alloc, bitmap)

        log(f"_update_directory_with_btree: Created {num_clusters} index blocks at clusters {index_clusters}")

    def _replace_directory_index_attrs(self, dir_record: int, index_root: bytes,
                                        index_alloc: Optional[bytes], bitmap: Optional[bytes]):
        """Replace directory index attributes (INDEX_ROOT, INDEX_ALLOCATION, BITMAP).

        This handles the full replacement of directory indexing attributes.
        """
        record_offset = self.mft_offset + dir_record * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.template):
            return

        record = bytearray(self.template[record_offset:record_offset + MFT_RECORD_SIZE])
        record = self._undo_fixups(record)

        if record[0:4] != b'FILE':
            return

        # Parse header
        usa_offset = struct.unpack('<H', record[4:6])[0]
        usa_count = struct.unpack('<H', record[6:8])[0]
        first_attr = struct.unpack('<H', record[20:22])[0]

        usn = struct.unpack('<H', record[usa_offset:usa_offset+2])[0]
        new_usn = (usn + 1) & 0xFFFF
        if new_usn == 0:
            new_usn = 1

        # Parse all attributes, removing old index attrs; track highest attr_id
        attributes = []
        max_attr_id = 0
        off = first_attr

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off+4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off+4:off+8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE - off:
                break

            attr_id = struct.unpack('<H', record[off+14:off+16])[0]
            if attr_id > max_attr_id:
                max_attr_id = attr_id

            # Skip old index attributes (we'll add new ones)
            if attr_type not in (0x90, 0xA0, 0xB0):
                attributes.append((attr_type, bytes(record[off:off + attr_len])))

            off += attr_len

        # Assign unique attribute IDs to new index attributes
        next_id = max_attr_id + 1
        index_root_with_id = bytearray(index_root)
        struct.pack_into('<H', index_root_with_id, 14, next_id)
        next_id += 1

        if index_alloc:
            index_alloc_with_id = bytearray(index_alloc)
            struct.pack_into('<H', index_alloc_with_id, 14, next_id)
            next_id += 1
        if bitmap:
            bitmap_with_id = bytearray(bitmap)
            struct.pack_into('<H', bitmap_with_id, 14, next_id)
            next_id += 1

        # Add new index attributes in order
        attributes.append((0x90, bytes(index_root_with_id)))
        if index_alloc:
            attributes.append((0xA0, bytes(index_alloc_with_id)))
        if bitmap:
            attributes.append((0xB0, bytes(bitmap_with_id)))

        # Sort by attribute type (NTFS requirement)
        attributes.sort(key=lambda x: x[0])

        # Calculate total size
        total_attr_size = sum(len(data) for _, data in attributes)
        new_used_size = first_attr + total_attr_size + 4

        if new_used_size > MFT_RECORD_SIZE:
            log(f"_replace_directory_index_attrs: Record too large ({new_used_size} bytes)")
            return

        # Build new record
        new_record = bytearray(MFT_RECORD_SIZE)
        new_record[0:first_attr] = record[0:first_attr]
        struct.pack_into('<H', new_record, usa_offset, new_usn)

        for i in range(1, usa_count):
            struct.pack_into('<H', new_record, usa_offset + (i * 2), 0)

        # Write attributes
        attr_offset = first_attr
        for attr_type, attr_data in attributes:
            new_record[attr_offset:attr_offset + len(attr_data)] = attr_data
            attr_offset += len(attr_data)

        # End marker
        struct.pack_into('<I', new_record, attr_offset, 0xFFFFFFFF)
        attr_offset += 4

        # Update used size
        struct.pack_into('<I', new_record, 24, attr_offset)

        # Update next attribute ID in record header (offset 40)
        struct.pack_into('<H', new_record, 40, next_id)

        # Apply fixups
        for i in range(1, usa_count):
            sector_end = (i * 512) - 2
            if sector_end < MFT_RECORD_SIZE:
                original = struct.unpack('<H', new_record[sector_end:sector_end + 2])[0]
                struct.pack_into('<H', new_record, usa_offset + (i * 2), original)
                struct.pack_into('<H', new_record, sector_end, new_usn)

        self.template[record_offset:record_offset + MFT_RECORD_SIZE] = new_record
        log(f"_replace_directory_index_attrs: record {dir_record}: wrote {attr_offset} bytes, used_size={attr_offset}")

        # Debug: dump record hex for validation (first and last 256 bytes)
        if dir_record == 5:
            hex1 = ' '.join(f'{b:02x}' for b in new_record[:256])
            hex2 = ' '.join(f'{b:02x}' for b in new_record[256:512])
            hex3 = ' '.join(f'{b:02x}' for b in new_record[496:752])
            log(f"  Record 5 hex [0:256]: {hex1}")
            log(f"  Record 5 hex [256:512]: {hex2}")
            log(f"  Record 5 hex [496:752]: {hex3}")

    def _replace_index_root(self, dir_record: int, new_index_root: bytes):
        """Replace $INDEX_ROOT attribute in a directory's MFT record.

        This function properly handles USA fixups by:
        1. Reading and parsing the entire record
        2. Extracting all attributes
        3. Rebuilding the record with the new INDEX_ROOT
        4. Properly applying fixups to the rebuilt record
        """
        record_offset = self.mft_offset + dir_record * MFT_RECORD_SIZE
        if record_offset + MFT_RECORD_SIZE > len(self.template):
            return

        record = bytearray(self.template[record_offset:record_offset + MFT_RECORD_SIZE])

        # Undo fixups for reading
        record = self._undo_fixups(record)

        if record[0:4] != b'FILE':
            return

        # Parse header fields we need to preserve
        usa_offset = struct.unpack('<H', record[4:6])[0]
        usa_count = struct.unpack('<H', record[6:8])[0]
        first_attr = struct.unpack('<H', record[20:22])[0]
        flags = struct.unpack('<H', record[22:24])[0]

        # Get the current USN and increment it for the modified record
        usn = struct.unpack('<H', record[usa_offset:usa_offset+2])[0]
        new_usn = (usn + 1) & 0xFFFF
        if new_usn == 0:
            new_usn = 1  # USN should never be 0

        # Parse ALL attributes into a list
        attributes = []
        off = first_attr
        found_index_root = False

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off+4])[0]
            if attr_type == 0xFFFFFFFF:
                break

            attr_len = struct.unpack('<I', record[off+4:off+8])[0]
            if attr_len == 0 or attr_len > MFT_RECORD_SIZE - off:
                break

            if attr_type == 0x90:  # $INDEX_ROOT
                # Replace with new INDEX_ROOT
                attributes.append((0x90, new_index_root))
                found_index_root = True
                log(f"_replace_index_root: record {dir_record}: replacing INDEX_ROOT (old len={attr_len}, new len={len(new_index_root)})")
            else:
                # Keep original attribute
                attributes.append((attr_type, bytes(record[off:off + attr_len])))

            off += attr_len

        log(f"_replace_index_root: record {dir_record}: found attrs {[hex(t) for t, _ in attributes]}")

        if not found_index_root:
            log(f"_replace_index_root: No $INDEX_ROOT found in record {dir_record}")
            return

        # Calculate total size needed
        total_attr_size = sum(len(data) for _, data in attributes)
        new_used_size = first_attr + total_attr_size + 4  # +4 for end marker

        if new_used_size > MFT_RECORD_SIZE:
            log(f"_replace_index_root: New index too large for record {dir_record}")
            return

        # Build new record - copy header verbatim (up to first attribute)
        new_record = bytearray(MFT_RECORD_SIZE)
        new_record[0:first_attr] = record[0:first_attr]

        # Update the USN in the header
        struct.pack_into('<H', new_record, usa_offset, new_usn)

        # Clear the old USA values (they'll be recalculated)
        for i in range(1, usa_count):
            struct.pack_into('<H', new_record, usa_offset + (i * 2), 0)

        # Write attributes
        attr_offset = first_attr
        for attr_type, attr_data in attributes:
            new_record[attr_offset:attr_offset + len(attr_data)] = attr_data
            attr_offset += len(attr_data)

        # Write end marker
        struct.pack_into('<I', new_record, attr_offset, 0xFFFFFFFF)
        attr_offset += 4

        # Update used size in header
        struct.pack_into('<I', new_record, 24, attr_offset)

        # Apply fixups - save original bytes at sector boundaries, write USN
        for i in range(1, usa_count):
            sector_end = (i * 512) - 2
            if sector_end < MFT_RECORD_SIZE:
                # Save the actual data at this position
                original = struct.unpack('<H', new_record[sector_end:sector_end+2])[0]
                struct.pack_into('<H', new_record, usa_offset + (i * 2), original)
                # Write USN at sector boundary
                struct.pack_into('<H', new_record, sector_end, new_usn)

        # Write back to template
        self.template[record_offset:record_offset + MFT_RECORD_SIZE] = new_record
