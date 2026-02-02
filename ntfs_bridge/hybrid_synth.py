"""Hybrid NTFS synthesizer - uses base template for system files, dynamically adds user files."""

import os
import struct
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

MFT_RECORD_SIZE = 1024
CLUSTER_SIZE = 4096

# Attribute types
ATTR_STANDARD_INFO = 0x10
ATTR_FILE_NAME = 0x30
ATTR_DATA = 0x80
ATTR_INDEX_ROOT = 0x90
ATTR_INDEX_ALLOCATION = 0xA0
ATTR_BITMAP = 0xB0
ATTR_END = 0xFFFFFFFF

FILE_RECORD_IN_USE = 0x01
FILE_RECORD_IS_DIRECTORY = 0x02


def log(msg):
    print(f"[HybridSynth] {msg}", flush=True)


@dataclass
class FileEntry:
    """Represents a file/directory."""
    name: str
    source_path: str
    is_directory: bool
    size: int
    mft_record: int = 0
    parent_record: int = 5
    clusters: List[int] = field(default_factory=list)


class HybridSynthesizer:
    """Uses base NTFS template, dynamically generates user file records."""

    def __init__(self, base_template_path: str, source_dir: str):
        self.source_dir = os.path.abspath(source_dir)

        # Load base template
        with open(base_template_path, 'rb') as f:
            self.template = bytearray(f.read())

        # Parse boot sector
        boot = self.template[0:512]
        self.bytes_per_sector = struct.unpack('<H', boot[0x0B:0x0D])[0]
        self.sectors_per_cluster = boot[0x0D]
        self.cluster_size = self.bytes_per_sector * self.sectors_per_cluster
        self.mft_cluster = struct.unpack('<Q', boot[0x30:0x38])[0]
        self.mft_offset = self.mft_cluster * self.cluster_size
        self.total_size = len(self.template)

        log(f"Base template: {self.total_size // (1024*1024)} MB, cluster size {self.cluster_size}")

        # File entries by MFT record number
        self.files: Dict[int, FileEntry] = {}

        # Directory children
        self.dir_children: Dict[int, List[int]] = {5: []}

        # Cluster map: cluster -> (source_path, offset_in_file)
        self.cluster_map: Dict[int, Tuple[str, int]] = {}

        # MFT record tracking for source files
        self.mft_record_to_source: Dict[int, str] = {}
        self.source_to_clusters: Dict[str, Set[int]] = {}

        # Find first free cluster (after template content)
        self.next_data_cluster = self._find_first_free_cluster()

        # Scan source and build file entries
        self._scan_source_directory()

        log(f"Mapped {len(self.files)} files, {len(self.cluster_map)} clusters")

    def _find_first_free_cluster(self) -> int:
        """Find first cluster after template's used area."""
        # Scan backwards from end to find last non-zero cluster
        cluster_count = self.total_size // self.cluster_size
        for cluster in range(cluster_count - 1, 0, -1):
            offset = cluster * self.cluster_size
            data = self.template[offset:offset + self.cluster_size]
            if any(b != 0 for b in data):
                return cluster + 10  # Leave some gap
        return 100

    def _scan_source_directory(self):
        """Scan source directory and create file entries."""
        next_record = 24  # User files start at 24

        for root, dirs, files in os.walk(self.source_dir):
            rel_path = os.path.relpath(root, self.source_dir)

            if rel_path == '.':
                parent_record = 5
            else:
                # For subdirectories, parent is the directory itself (not its parent)
                parent_record = self._find_record_for_path(rel_path)
                if parent_record is None:
                    parent_record = 5

            # Process directories
            for d in sorted(dirs):
                dir_path = os.path.join(root, d)
                entry = FileEntry(
                    name=d,
                    source_path=dir_path,
                    is_directory=True,
                    size=0,
                    mft_record=next_record,
                    parent_record=parent_record
                )
                self.files[next_record] = entry
                self.mft_record_to_source[next_record] = dir_path

                if parent_record not in self.dir_children:
                    self.dir_children[parent_record] = []
                self.dir_children[parent_record].append(next_record)
                self.dir_children[next_record] = []

                next_record += 1

            # Process files
            for f in sorted(files):
                file_path = os.path.join(root, f)
                try:
                    file_size = os.path.getsize(file_path)
                except OSError:
                    continue

                entry = FileEntry(
                    name=f,
                    source_path=file_path,
                    is_directory=False,
                    size=file_size,
                    mft_record=next_record,
                    parent_record=parent_record
                )

                # Allocate clusters for non-resident data
                if file_size > 700:  # Non-resident threshold
                    clusters_needed = (file_size + self.cluster_size - 1) // self.cluster_size
                    self.source_to_clusters[file_path] = set()

                    for i in range(clusters_needed):
                        cluster = self.next_data_cluster
                        self.next_data_cluster += 1
                        entry.clusters.append(cluster)
                        self.cluster_map[cluster] = (file_path, i * self.cluster_size)
                        self.source_to_clusters[file_path].add(cluster)

                self.files[next_record] = entry
                self.mft_record_to_source[next_record] = file_path

                if parent_record not in self.dir_children:
                    self.dir_children[parent_record] = []
                self.dir_children[parent_record].append(next_record)

                next_record += 1

    def _find_record_for_path(self, rel_path: str) -> Optional[int]:
        """Find MFT record for a relative path."""
        if not rel_path or rel_path == '.':
            return 5

        parts = rel_path.replace('\\', '/').split('/')
        current = 5

        for part in parts:
            found = False
            for child in self.dir_children.get(current, []):
                if child in self.files and self.files[child].name == part:
                    current = child
                    found = True
                    break
            if not found:
                return None

        return current

    def get_size(self) -> int:
        return self.total_size

    def read(self, offset: int, length: int) -> bytes:
        """Read from synthesized volume."""
        result = bytearray(length)
        pos = 0

        while pos < length:
            current_offset = offset + pos
            remaining = length - pos

            # Check if reading root directory MFT record (record 5)
            root_record_offset = self.mft_offset + 5 * MFT_RECORD_SIZE
            if root_record_offset <= current_offset < root_record_offset + MFT_RECORD_SIZE:
                record_offset = current_offset - root_record_offset
                chunk_len = min(remaining, MFT_RECORD_SIZE - record_offset)
                patched_root = self._build_patched_root_directory()
                result[pos:pos + chunk_len] = patched_root[record_offset:record_offset + chunk_len]
                pos += chunk_len
                continue

            # Check if in MFT region - might need to generate user record
            if self._is_user_mft_region(current_offset):
                record_num = (current_offset - self.mft_offset) // MFT_RECORD_SIZE
                record_offset = (current_offset - self.mft_offset) % MFT_RECORD_SIZE

                if record_num in self.files:
                    chunk_len = min(remaining, MFT_RECORD_SIZE - record_offset)
                    record_data = self._build_file_record(self.files[record_num])
                    result[pos:pos + chunk_len] = record_data[record_offset:record_offset + chunk_len]
                    pos += chunk_len
                    continue

            # Check if in root directory index region - need to inject our files
            if self._is_root_index_region(current_offset):
                chunk_len = min(remaining, self.cluster_size)
                index_data = self._build_root_index()
                rel_offset = current_offset - self._get_root_index_offset()
                if rel_offset < len(index_data):
                    copy_len = min(chunk_len, len(index_data) - rel_offset)
                    result[pos:pos + copy_len] = index_data[rel_offset:rel_offset + copy_len]
                pos += chunk_len
                continue

            # Check if reading file data from mapped cluster
            cluster = current_offset // self.cluster_size
            cluster_offset = current_offset % self.cluster_size

            if cluster in self.cluster_map:
                source_path, file_offset = self.cluster_map[cluster]
                chunk_len = min(remaining, self.cluster_size - cluster_offset)

                try:
                    with open(source_path, 'rb') as f:
                        f.seek(file_offset + cluster_offset)
                        data = f.read(chunk_len)
                        result[pos:pos + len(data)] = data
                except (OSError, IOError):
                    pass

                pos += chunk_len
                continue

            # Fall back to template - but don't cross special region boundaries
            chunk_len = min(remaining, self.cluster_size - cluster_offset)

            # Check if this chunk would cross into root directory record
            root_record_offset = self.mft_offset + 5 * MFT_RECORD_SIZE
            if current_offset < root_record_offset < current_offset + chunk_len:
                chunk_len = root_record_offset - current_offset

            # Check if chunk crosses into user MFT region
            user_mft_start = self.mft_offset + 24 * MFT_RECORD_SIZE
            if current_offset < user_mft_start < current_offset + chunk_len:
                chunk_len = user_mft_start - current_offset

            if current_offset + chunk_len <= len(self.template):
                result[pos:pos + chunk_len] = self.template[current_offset:current_offset + chunk_len]
            pos += chunk_len

        return bytes(result)

    def _build_patched_root_directory(self) -> bytes:
        """Build root directory MFT record with our files in the index."""
        # Start with original record from template to preserve all attributes
        root_offset = self.mft_offset + 5 * MFT_RECORD_SIZE
        record = bytearray(self.template[root_offset:root_offset + MFT_RECORD_SIZE])

        # Undo fixups
        self._undo_fixups(record)

        # Find $INDEX_ROOT and $INDEX_ALLOCATION attributes
        first_attr = struct.unpack('<H', record[20:22])[0]
        off = first_attr
        index_root_off = None
        index_alloc_off = None
        bitmap_off = None

        while off < MFT_RECORD_SIZE - 8:
            attr_type = struct.unpack('<I', record[off:off + 4])[0]
            if attr_type == 0xFFFFFFFF:
                break
            attr_len = struct.unpack('<I', record[off + 4:off + 8])[0]
            if attr_len == 0:
                break

            if attr_type == ATTR_INDEX_ROOT:
                index_root_off = off
            elif attr_type == ATTR_INDEX_ALLOCATION:
                index_alloc_off = off
            elif attr_type == ATTR_BITMAP:
                # Check if it's the $I30 bitmap
                name_len = record[off + 9]
                if name_len == 4:
                    bitmap_off = off

            off += attr_len

        if index_root_off is None:
            self._apply_fixups(record)
            return bytes(record)

        # Build new index entries
        children = self.dir_children.get(5, [])
        new_entries = bytearray()
        for child_rec in sorted(children, key=lambda r: self.files[r].name.upper() if r in self.files else ""):
            if child_rec in self.files:
                new_entries.extend(self._build_index_entry(self.files[child_rec]))

        # End entry
        end_entry = bytearray(24)  # Larger end entry
        struct.pack_into('<H', end_entry, 8, 24)  # Entry length
        struct.pack_into('<I', end_entry, 12, 2)  # Last entry flag
        new_entries.extend(end_entry)

        # Get index root content offset
        content_offset = struct.unpack('<H', record[index_root_off + 20:index_root_off + 22])[0]
        content_start = index_root_off + content_offset

        # New content size: index root header (16) + node header (16) + entries
        new_entries_size = 16 + len(new_entries)
        new_content_size = 16 + new_entries_size

        # Calculate space available (up to next attr or end)
        old_attr_len = struct.unpack('<I', record[index_root_off + 4:index_root_off + 8])[0]
        new_attr_size = ((content_offset + new_content_size) + 7) & ~7

        # Update attribute if it fits
        if new_attr_size <= old_attr_len + 200:  # Allow some growth
            # Update attribute length
            struct.pack_into('<I', record, index_root_off + 4, new_attr_size)

            # Update content length
            struct.pack_into('<I', record, index_root_off + 16, new_content_size)

            # Update node header
            node_hdr_off = content_start + 16
            struct.pack_into('<I', record, node_hdr_off, 16)  # Entries offset
            struct.pack_into('<I', record, node_hdr_off + 4, new_entries_size)  # Total size
            struct.pack_into('<I', record, node_hdr_off + 8, new_entries_size)  # Allocated
            record[node_hdr_off + 12] = 0  # Flags - small index (no subnodes)

            # Write entries
            entries_start = content_start + 32
            record[entries_start:entries_start + len(new_entries)] = new_entries

        self._apply_fixups(record)
        return bytes(record)

    def _undo_fixups(self, record: bytearray):
        """Undo NTFS fixup array."""
        usa_offset = struct.unpack('<H', record[4:6])[0]
        usa_count = struct.unpack('<H', record[6:8])[0]

        for i in range(1, usa_count):
            sector_end = i * 512 - 2
            if usa_offset + i * 2 + 2 <= MFT_RECORD_SIZE and sector_end + 2 <= MFT_RECORD_SIZE:
                original = struct.unpack('<H', record[usa_offset + i * 2:usa_offset + i * 2 + 2])[0]
                struct.pack_into('<H', record, sector_end, original)

    def _is_user_mft_region(self, offset: int) -> bool:
        """Check if offset is in user MFT region (records 24+)."""
        if offset < self.mft_offset:
            return False
        record_num = (offset - self.mft_offset) // MFT_RECORD_SIZE
        return record_num >= 24 and record_num in self.files

    def _is_root_index_region(self, offset: int) -> bool:
        """Check if offset is in root directory index region."""
        # For simplicity, we'll patch the root directory record instead
        return False

    def _get_root_index_offset(self) -> int:
        """Get offset of root directory index."""
        return 0

    def _build_root_index(self) -> bytes:
        """Build root directory index."""
        return b''

    def _build_file_record(self, entry: FileEntry) -> bytes:
        """Build MFT record for a file."""
        record = bytearray(MFT_RECORD_SIZE)

        # FILE signature
        record[0:4] = b'FILE'

        # Update sequence array offset and count
        struct.pack_into('<H', record, 4, 48)
        struct.pack_into('<H', record, 6, 3)

        # Sequence number
        struct.pack_into('<H', record, 16, 1)

        # Link count
        struct.pack_into('<H', record, 18, 1)

        # First attribute offset
        struct.pack_into('<H', record, 20, 56)

        # Flags
        flags = FILE_RECORD_IN_USE
        if entry.is_directory:
            flags |= FILE_RECORD_IS_DIRECTORY
        struct.pack_into('<H', record, 22, flags)

        # Used size and allocated size
        struct.pack_into('<I', record, 24, MFT_RECORD_SIZE)
        struct.pack_into('<I', record, 28, MFT_RECORD_SIZE)

        # Record number
        struct.pack_into('<I', record, 44, entry.mft_record)

        # Build attributes
        attr_offset = 56

        # $STANDARD_INFORMATION
        attr_offset = self._add_standard_info(record, attr_offset)

        # $FILE_NAME
        attr_offset = self._add_filename(record, attr_offset, entry)

        if entry.is_directory:
            # $INDEX_ROOT for directories
            attr_offset = self._add_index_root(record, attr_offset, entry.mft_record)
        elif entry.size > 0:
            # $DATA for files
            attr_offset = self._add_data_attr(record, attr_offset, entry)

        # End marker
        struct.pack_into('<I', record, attr_offset, ATTR_END)

        # Apply fixups
        self._apply_fixups(record)

        return bytes(record)

    def _add_standard_info(self, record: bytearray, offset: int) -> int:
        """Add $STANDARD_INFORMATION attribute."""
        struct.pack_into('<I', record, offset, ATTR_STANDARD_INFO)
        struct.pack_into('<I', record, offset + 4, 96)
        record[offset + 8] = 0  # Resident
        struct.pack_into('<I', record, offset + 16, 48)
        struct.pack_into('<H', record, offset + 20, 24)

        now = self._windows_time()
        content_off = offset + 24
        struct.pack_into('<Q', record, content_off, now)
        struct.pack_into('<Q', record, content_off + 8, now)
        struct.pack_into('<Q', record, content_off + 16, now)
        struct.pack_into('<Q', record, content_off + 24, now)
        struct.pack_into('<I', record, content_off + 32, 0x20)  # Archive

        return offset + 96

    def _add_filename(self, record: bytearray, offset: int, entry: FileEntry) -> int:
        """Add $FILE_NAME attribute."""
        name_bytes = entry.name.encode('utf-16-le')
        content_size = 66 + len(name_bytes)
        attr_size = ((24 + content_size) + 7) & ~7

        struct.pack_into('<I', record, offset, ATTR_FILE_NAME)
        struct.pack_into('<I', record, offset + 4, attr_size)
        record[offset + 8] = 0  # Resident
        struct.pack_into('<I', record, offset + 16, content_size)
        struct.pack_into('<H', record, offset + 20, 24)

        content_off = offset + 24
        struct.pack_into('<Q', record, content_off, entry.parent_record | (1 << 48))

        now = self._windows_time()
        struct.pack_into('<Q', record, content_off + 8, now)
        struct.pack_into('<Q', record, content_off + 16, now)
        struct.pack_into('<Q', record, content_off + 24, now)
        struct.pack_into('<Q', record, content_off + 32, now)
        struct.pack_into('<Q', record, content_off + 40, len(entry.clusters) * self.cluster_size if entry.clusters else 0)
        struct.pack_into('<Q', record, content_off + 48, entry.size)

        flags = 0x10 if entry.is_directory else 0x20
        struct.pack_into('<I', record, content_off + 56, flags)

        record[content_off + 64] = len(entry.name)
        record[content_off + 65] = 3  # Win32 namespace
        record[content_off + 66:content_off + 66 + len(name_bytes)] = name_bytes

        return offset + attr_size

    def _add_data_attr(self, record: bytearray, offset: int, entry: FileEntry) -> int:
        """Add $DATA attribute."""
        if entry.size <= 700 and not entry.clusters:
            # Resident data
            try:
                with open(entry.source_path, 'rb') as f:
                    data = f.read()
            except:
                data = b''

            attr_size = ((24 + len(data)) + 7) & ~7
            struct.pack_into('<I', record, offset, ATTR_DATA)
            struct.pack_into('<I', record, offset + 4, attr_size)
            record[offset + 8] = 0  # Resident
            struct.pack_into('<I', record, offset + 16, len(data))
            struct.pack_into('<H', record, offset + 20, 24)
            record[offset + 24:offset + 24 + len(data)] = data
            return offset + attr_size

        # Non-resident data with data runs
        data_runs = self._encode_data_runs(entry.clusters)
        attr_size = ((64 + len(data_runs)) + 7) & ~7

        struct.pack_into('<I', record, offset, ATTR_DATA)
        struct.pack_into('<I', record, offset + 4, attr_size)
        record[offset + 8] = 1  # Non-resident

        struct.pack_into('<Q', record, offset + 16, 0)  # Lowest VCN
        struct.pack_into('<Q', record, offset + 24, len(entry.clusters) - 1 if entry.clusters else 0)  # Highest VCN
        struct.pack_into('<H', record, offset + 32, 64)  # Data runs offset
        struct.pack_into('<Q', record, offset + 40, len(entry.clusters) * self.cluster_size)  # Allocated
        struct.pack_into('<Q', record, offset + 48, entry.size)  # Real size
        struct.pack_into('<Q', record, offset + 56, entry.size)  # Initialized

        record[offset + 64:offset + 64 + len(data_runs)] = data_runs

        return offset + attr_size

    def _add_index_root(self, record: bytearray, offset: int, dir_record: int) -> int:
        """Add $INDEX_ROOT for directory."""
        children = self.dir_children.get(dir_record, [])

        # Build index entries
        index_entries = bytearray()
        for child_rec in sorted(children, key=lambda r: self.files[r].name.upper() if r in self.files else ""):
            if child_rec in self.files:
                index_entries.extend(self._build_index_entry(self.files[child_rec]))

        # End entry
        end_entry = bytearray(16)
        struct.pack_into('<H', end_entry, 8, 16)
        struct.pack_into('<I', end_entry, 12, 2)  # Last entry flag
        index_entries.extend(end_entry)

        # Index root header (16) + node header (16) + entries
        content_size = 32 + len(index_entries)
        attr_size = ((32 + content_size) + 7) & ~7

        struct.pack_into('<I', record, offset, ATTR_INDEX_ROOT)
        struct.pack_into('<I', record, offset + 4, attr_size)
        record[offset + 8] = 0  # Resident
        record[offset + 9] = 4  # Name length
        struct.pack_into('<H', record, offset + 10, 24)  # Name offset
        struct.pack_into('<I', record, offset + 16, content_size)
        struct.pack_into('<H', record, offset + 20, 32)  # Content offset

        # Name "$I30"
        record[offset + 24:offset + 32] = b'$\x00I\x003\x000\x00'

        content_off = offset + 32

        # Index root header
        struct.pack_into('<I', record, content_off, ATTR_FILE_NAME)  # Indexed type
        struct.pack_into('<I', record, content_off + 4, 1)  # Collation
        struct.pack_into('<I', record, content_off + 8, 4096)  # Index block size
        record[content_off + 12] = 1  # Clusters per block

        # Node header
        struct.pack_into('<I', record, content_off + 16, 16)  # Entries offset
        struct.pack_into('<I', record, content_off + 20, 16 + len(index_entries))  # Total size
        struct.pack_into('<I', record, content_off + 24, 16 + len(index_entries))  # Allocated

        # Entries
        record[content_off + 32:content_off + 32 + len(index_entries)] = index_entries

        return offset + attr_size

    def _build_index_entry(self, entry: FileEntry) -> bytes:
        """Build directory index entry."""
        name_bytes = entry.name.encode('utf-16-le')
        key_len = 66 + len(name_bytes)
        entry_len = ((16 + key_len) + 7) & ~7

        data = bytearray(entry_len)
        struct.pack_into('<Q', data, 0, entry.mft_record | (1 << 48))
        struct.pack_into('<H', data, 8, entry_len)
        struct.pack_into('<H', data, 10, key_len)

        key_off = 16
        struct.pack_into('<Q', data, key_off, entry.parent_record | (1 << 48))

        now = self._windows_time()
        struct.pack_into('<Q', data, key_off + 8, now)
        struct.pack_into('<Q', data, key_off + 16, now)
        struct.pack_into('<Q', data, key_off + 24, now)
        struct.pack_into('<Q', data, key_off + 32, now)
        struct.pack_into('<Q', data, key_off + 40, 0)
        struct.pack_into('<Q', data, key_off + 48, entry.size)

        flags = 0x10 if entry.is_directory else 0x20
        struct.pack_into('<I', data, key_off + 56, flags)

        data[key_off + 64] = len(entry.name)
        data[key_off + 65] = 3
        data[key_off + 66:key_off + 66 + len(name_bytes)] = name_bytes

        return bytes(data)

    def _encode_data_runs(self, clusters: List[int]) -> bytes:
        """Encode clusters as NTFS data runs."""
        if not clusters:
            return b'\x00'

        runs = bytearray()
        prev_lcn = 0
        i = 0

        while i < len(clusters):
            start = clusters[i]
            length = 1
            while i + length < len(clusters) and clusters[i + length] == start + length:
                length += 1

            lcn_offset = start - prev_lcn

            len_size = max(1, (length.bit_length() + 7) // 8)
            if lcn_offset >= 0:
                off_size = max(1, (lcn_offset.bit_length() + 7) // 8)
            else:
                off_size = max(1, ((~lcn_offset).bit_length() + 8) // 8)

            header = (off_size << 4) | len_size
            runs.append(header)

            for b in range(len_size):
                runs.append((length >> (b * 8)) & 0xFF)

            if lcn_offset >= 0:
                for b in range(off_size):
                    runs.append((lcn_offset >> (b * 8)) & 0xFF)
            else:
                val = lcn_offset & ((1 << (off_size * 8)) - 1)
                for b in range(off_size):
                    runs.append((val >> (b * 8)) & 0xFF)

            prev_lcn = start
            i += length

        runs.append(0)
        return bytes(runs)

    def _apply_fixups(self, record: bytearray):
        """Apply NTFS fixup array."""
        usa_offset = struct.unpack('<H', record[4:6])[0]
        usa_count = struct.unpack('<H', record[6:8])[0]

        check = 1
        struct.pack_into('<H', record, usa_offset, check)

        for i in range(1, usa_count):
            sector_end = i * 512 - 2
            if sector_end + 2 <= len(record):
                orig = struct.unpack('<H', record[sector_end:sector_end + 2])[0]
                struct.pack_into('<H', record, usa_offset + i * 2, orig)
                struct.pack_into('<H', record, sector_end, check)

    def _windows_time(self) -> int:
        """Get Windows FILETIME."""
        return int((time.time() + 11644473600) * 10000000)

    def is_mft_region(self, offset: int, length: int) -> bool:
        """Check if offset is in MFT region."""
        mft_end = self.mft_offset + (max(self.files.keys()) + 1) * MFT_RECORD_SIZE if self.files else self.mft_offset
        return offset < mft_end and offset + length > self.mft_offset


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python hybrid_synth.py <base_template> <source_dir>")
        sys.exit(1)

    synth = HybridSynthesizer(sys.argv[1], sys.argv[2])
    print(f"Files: {list(synth.files.keys())}")
    print(f"Clusters: {len(synth.cluster_map)}")
