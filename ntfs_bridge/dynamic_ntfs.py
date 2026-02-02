"""Dynamic NTFS synthesizer - generates NTFS structures on-the-fly from source directory."""

import os
import struct
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

# NTFS Constants
SECTOR_SIZE = 512
CLUSTER_SIZE = 4096  # 4KB clusters
MFT_RECORD_SIZE = 1024
BYTES_PER_FILE_RECORD = 1024

# Attribute types
ATTR_STANDARD_INFORMATION = 0x10
ATTR_FILE_NAME = 0x30
ATTR_DATA = 0x80
ATTR_INDEX_ROOT = 0x90
ATTR_INDEX_ALLOCATION = 0xA0
ATTR_BITMAP = 0xB0
ATTR_END = 0xFFFFFFFF

# MFT record flags
FILE_RECORD_IN_USE = 0x01
FILE_RECORD_IS_DIRECTORY = 0x02

# File attribute flags
FILE_ATTR_READONLY = 0x0001
FILE_ATTR_HIDDEN = 0x0002
FILE_ATTR_SYSTEM = 0x0004
FILE_ATTR_DIRECTORY = 0x0010
FILE_ATTR_ARCHIVE = 0x0020
FILE_ATTR_NORMAL = 0x0080


def log(msg):
    print(f"[DynamicNTFS] {msg}", flush=True)


@dataclass
class FileEntry:
    """Represents a file/directory in the virtual NTFS."""
    name: str
    source_path: str  # Path on ext4
    is_directory: bool
    size: int
    mft_record: int = 0
    parent_record: int = 5  # Root directory by default
    clusters: List[int] = field(default_factory=list)
    created_time: int = 0
    modified_time: int = 0


class DynamicNTFSSynthesizer:
    """Synthesizes NTFS filesystem dynamically from source directory."""

    def __init__(self, source_dir: str, volume_label: str = "DynamicNTFS", size_mb: int = 100):
        self.source_dir = os.path.abspath(source_dir)
        self.volume_label = volume_label
        self.cluster_size = CLUSTER_SIZE

        # Calculate volume geometry
        self.total_size = size_mb * 1024 * 1024
        self.total_sectors = self.total_size // SECTOR_SIZE
        self.total_clusters = self.total_size // CLUSTER_SIZE

        # MFT location (cluster 4, typical for small volumes)
        self.mft_cluster = 4
        self.mft_offset = self.mft_cluster * CLUSTER_SIZE

        # Reserved MFT records (0-15 are system, 16-23 reserved, user files start at 24)
        self.first_user_record = 24

        # File entries indexed by MFT record number
        self.files: Dict[int, FileEntry] = {}

        # Directory children: parent_record -> [child_records]
        self.dir_children: Dict[int, List[int]] = {5: []}  # Root starts empty

        # Cluster allocation: cluster -> (file_record, offset_in_file)
        self.cluster_map: Dict[int, Tuple[int, int]] = {}

        # Next available cluster for file data (after MFT and system areas)
        self.next_data_cluster = 1000  # Leave room for MFT growth

        # Scan source directory and build file entries
        self._scan_source_directory()

        # Pre-build static structures
        self._build_boot_sector()
        self._build_system_mft_records()

        log(f"Initialized: {len(self.files)} files, {self.total_size // (1024*1024)} MB volume")

    def _scan_source_directory(self):
        """Scan source directory and create file entries."""
        next_record = self.first_user_record

        # Walk the source directory
        for root, dirs, files in os.walk(self.source_dir):
            rel_path = os.path.relpath(root, self.source_dir)

            # Determine parent MFT record
            if rel_path == '.':
                parent_record = 5  # Root directory
            else:
                # Find parent directory's record
                parent_record = self._find_record_for_path(os.path.dirname(rel_path))
                if parent_record is None:
                    parent_record = 5

            # Process subdirectories
            for d in dirs:
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

                # Add to parent's children
                if parent_record not in self.dir_children:
                    self.dir_children[parent_record] = []
                self.dir_children[parent_record].append(next_record)

                # Initialize this dir's children list
                self.dir_children[next_record] = []

                next_record += 1

            # Process files
            for f in files:
                file_path = os.path.join(root, f)
                try:
                    stat = os.stat(file_path)
                    file_size = stat.st_size
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

                # Allocate clusters for file data
                if file_size > 0:
                    clusters_needed = (file_size + CLUSTER_SIZE - 1) // CLUSTER_SIZE
                    for i in range(clusters_needed):
                        cluster = self.next_data_cluster
                        self.next_data_cluster += 1
                        entry.clusters.append(cluster)
                        self.cluster_map[cluster] = (next_record, i * CLUSTER_SIZE)

                self.files[next_record] = entry

                # Add to parent's children
                if parent_record not in self.dir_children:
                    self.dir_children[parent_record] = []
                self.dir_children[parent_record].append(next_record)

                next_record += 1

        log(f"Scanned {len(self.files)} files/directories")

    def _find_record_for_path(self, rel_path: str) -> Optional[int]:
        """Find MFT record number for a relative path."""
        if not rel_path or rel_path == '.':
            return 5

        parts = rel_path.replace('\\', '/').split('/')
        current_record = 5  # Start at root

        for part in parts:
            found = False
            for child_record in self.dir_children.get(current_record, []):
                if child_record in self.files and self.files[child_record].name == part:
                    current_record = child_record
                    found = True
                    break
            if not found:
                return None

        return current_record

    def _build_boot_sector(self):
        """Build NTFS boot sector."""
        self.boot_sector = bytearray(SECTOR_SIZE)

        # Jump instruction
        self.boot_sector[0:3] = b'\xeb\x52\x90'

        # OEM ID
        self.boot_sector[3:11] = b'NTFS    '

        # BPB (BIOS Parameter Block)
        struct.pack_into('<H', self.boot_sector, 0x0B, SECTOR_SIZE)  # Bytes per sector
        self.boot_sector[0x0D] = CLUSTER_SIZE // SECTOR_SIZE  # Sectors per cluster
        struct.pack_into('<H', self.boot_sector, 0x0E, 0)  # Reserved sectors
        self.boot_sector[0x10:0x13] = b'\x00\x00\x00'  # Always 0 for NTFS
        struct.pack_into('<H', self.boot_sector, 0x13, 0)  # Unused
        self.boot_sector[0x15] = 0xF8  # Media descriptor (hard disk)
        struct.pack_into('<H', self.boot_sector, 0x16, 0)  # Unused
        struct.pack_into('<H', self.boot_sector, 0x18, 63)  # Sectors per track
        struct.pack_into('<H', self.boot_sector, 0x1A, 255)  # Number of heads
        struct.pack_into('<I', self.boot_sector, 0x1C, 0)  # Hidden sectors

        # Extended BPB
        struct.pack_into('<Q', self.boot_sector, 0x28, self.total_sectors - 1)  # Total sectors
        struct.pack_into('<Q', self.boot_sector, 0x30, self.mft_cluster)  # MFT cluster
        struct.pack_into('<Q', self.boot_sector, 0x38, self.total_clusters - 1)  # MFT mirror cluster

        # Clusters per MFT record (negative means 2^|n| bytes)
        mft_record_clusters = MFT_RECORD_SIZE // CLUSTER_SIZE
        if mft_record_clusters == 0:
            # Record size < cluster size, encode as power of 2
            self.boot_sector[0x40] = 256 - 10  # 2^10 = 1024
        else:
            self.boot_sector[0x40] = mft_record_clusters

        # Clusters per index record
        self.boot_sector[0x44] = 256 - 12  # 4096 bytes

        # Volume serial number
        struct.pack_into('<Q', self.boot_sector, 0x48, int(time.time()) & 0xFFFFFFFFFFFFFFFF)

        # Boot signature
        self.boot_sector[0x1FE:0x200] = b'\x55\xAA'

    def _build_system_mft_records(self):
        """Build system MFT records (0-15)."""
        self.system_records = {}

        # Record 0: $MFT
        self.system_records[0] = self._build_mft_record(0, "$MFT", is_system=True)

        # Record 1: $MFTMirr
        self.system_records[1] = self._build_mft_record(1, "$MFTMirr", is_system=True)

        # Record 2: $LogFile
        self.system_records[2] = self._build_mft_record(2, "$LogFile", is_system=True)

        # Record 3: $Volume
        self.system_records[3] = self._build_mft_record(3, "$Volume", is_system=True)

        # Record 4: $AttrDef
        self.system_records[4] = self._build_mft_record(4, "$AttrDef", is_system=True)

        # Record 5: Root directory (.)
        self.system_records[5] = self._build_directory_record(5, ".", parent=5)

        # Record 6: $Bitmap
        self.system_records[6] = self._build_mft_record(6, "$Bitmap", is_system=True)

        # Record 7: $Boot
        self.system_records[7] = self._build_mft_record(7, "$Boot", is_system=True)

        # Record 8: $BadClus
        self.system_records[8] = self._build_mft_record(8, "$BadClus", is_system=True)

        # Record 9: $Secure
        self.system_records[9] = self._build_mft_record(9, "$Secure", is_system=True)

        # Record 10: $UpCase
        self.system_records[10] = self._build_mft_record(10, "$UpCase", is_system=True)

        # Record 11: $Extend
        self.system_records[11] = self._build_mft_record(11, "$Extend", is_system=True, is_dir=True)

        # Records 12-15: Reserved (empty but valid)
        for i in range(12, 16):
            self.system_records[i] = self._build_empty_record(i)

    def _build_mft_record(self, record_num: int, name: str, is_system: bool = False, is_dir: bool = False) -> bytes:
        """Build a basic MFT record."""
        record = bytearray(MFT_RECORD_SIZE)

        # FILE signature
        record[0:4] = b'FILE'

        # Update sequence offset and count
        struct.pack_into('<H', record, 4, 48)  # USA offset
        struct.pack_into('<H', record, 6, 3)   # USA count (1 + 2 sectors)

        # LSN (Log Sequence Number)
        struct.pack_into('<Q', record, 8, 0)

        # Sequence number
        struct.pack_into('<H', record, 16, 1)

        # Link count
        struct.pack_into('<H', record, 18, 1)

        # First attribute offset
        struct.pack_into('<H', record, 20, 56)

        # Flags
        flags = FILE_RECORD_IN_USE
        if is_dir:
            flags |= FILE_RECORD_IS_DIRECTORY
        struct.pack_into('<H', record, 22, flags)

        # Used size of record
        struct.pack_into('<I', record, 24, MFT_RECORD_SIZE)

        # Allocated size
        struct.pack_into('<I', record, 28, MFT_RECORD_SIZE)

        # Base record (0 for base records)
        struct.pack_into('<Q', record, 32, 0)

        # Next attribute ID
        struct.pack_into('<H', record, 40, 3)

        # Record number (for XP+)
        struct.pack_into('<I', record, 44, record_num)

        # Update sequence array
        record[48:50] = b'\x00\x00'  # USA check value
        record[50:52] = b'\x00\x00'  # Sector 1 original
        record[52:54] = b'\x00\x00'  # Sector 2 original

        # Build attributes starting at offset 56
        attr_offset = 56

        # $STANDARD_INFORMATION attribute
        attr_offset = self._add_standard_info(record, attr_offset)

        # $FILE_NAME attribute
        attr_offset = self._add_filename(record, attr_offset, name, 5, is_dir)

        # End marker
        struct.pack_into('<I', record, attr_offset, ATTR_END)

        # Apply fixups
        self._apply_fixups(record)

        return bytes(record)

    def _build_empty_record(self, record_num: int) -> bytes:
        """Build an empty (not in use) MFT record."""
        record = bytearray(MFT_RECORD_SIZE)
        record[0:4] = b'FILE'
        struct.pack_into('<H', record, 4, 48)
        struct.pack_into('<H', record, 6, 3)
        struct.pack_into('<H', record, 16, 0)  # Sequence 0
        struct.pack_into('<H', record, 22, 0)  # Not in use
        struct.pack_into('<I', record, 44, record_num)
        self._apply_fixups(record)
        return bytes(record)

    def _build_directory_record(self, record_num: int, name: str, parent: int) -> bytes:
        """Build MFT record for a directory with index."""
        record = bytearray(MFT_RECORD_SIZE)

        # FILE signature
        record[0:4] = b'FILE'
        struct.pack_into('<H', record, 4, 48)
        struct.pack_into('<H', record, 6, 3)
        struct.pack_into('<Q', record, 8, 0)
        struct.pack_into('<H', record, 16, 1)
        struct.pack_into('<H', record, 18, 1)
        struct.pack_into('<H', record, 20, 56)
        struct.pack_into('<H', record, 22, FILE_RECORD_IN_USE | FILE_RECORD_IS_DIRECTORY)
        struct.pack_into('<I', record, 24, MFT_RECORD_SIZE)
        struct.pack_into('<I', record, 28, MFT_RECORD_SIZE)
        struct.pack_into('<Q', record, 32, 0)
        struct.pack_into('<H', record, 40, 5)
        struct.pack_into('<I', record, 44, record_num)
        record[48:54] = b'\x00' * 6

        attr_offset = 56

        # $STANDARD_INFORMATION
        attr_offset = self._add_standard_info(record, attr_offset)

        # $FILE_NAME
        attr_offset = self._add_filename(record, attr_offset, name, parent, is_dir=True)

        # $INDEX_ROOT for $I30 (filename index)
        attr_offset = self._add_index_root(record, attr_offset, record_num)

        # End marker
        struct.pack_into('<I', record, attr_offset, ATTR_END)

        self._apply_fixups(record)
        return bytes(record)

    def _build_file_record(self, entry: FileEntry) -> bytes:
        """Build MFT record for a user file."""
        record = bytearray(MFT_RECORD_SIZE)

        record[0:4] = b'FILE'
        struct.pack_into('<H', record, 4, 48)
        struct.pack_into('<H', record, 6, 3)
        struct.pack_into('<Q', record, 8, 0)
        struct.pack_into('<H', record, 16, 1)
        struct.pack_into('<H', record, 18, 1)
        struct.pack_into('<H', record, 20, 56)

        flags = FILE_RECORD_IN_USE
        if entry.is_directory:
            flags |= FILE_RECORD_IS_DIRECTORY
        struct.pack_into('<H', record, 22, flags)

        struct.pack_into('<I', record, 24, MFT_RECORD_SIZE)
        struct.pack_into('<I', record, 28, MFT_RECORD_SIZE)
        struct.pack_into('<Q', record, 32, 0)
        struct.pack_into('<H', record, 40, 5)
        struct.pack_into('<I', record, 44, entry.mft_record)
        record[48:54] = b'\x00' * 6

        attr_offset = 56

        # $STANDARD_INFORMATION
        attr_offset = self._add_standard_info(record, attr_offset)

        # $FILE_NAME
        attr_offset = self._add_filename(record, attr_offset, entry.name, entry.parent_record, entry.is_directory)

        if entry.is_directory:
            # $INDEX_ROOT
            attr_offset = self._add_index_root(record, attr_offset, entry.mft_record)
        else:
            # $DATA attribute
            attr_offset = self._add_data_attribute(record, attr_offset, entry)

        # End marker
        struct.pack_into('<I', record, attr_offset, ATTR_END)

        self._apply_fixups(record)
        return bytes(record)

    def _add_standard_info(self, record: bytearray, offset: int) -> int:
        """Add $STANDARD_INFORMATION attribute."""
        # Attribute header
        struct.pack_into('<I', record, offset, ATTR_STANDARD_INFORMATION)
        struct.pack_into('<I', record, offset + 4, 96)  # Attribute length
        record[offset + 8] = 0  # Resident
        record[offset + 9] = 0  # Name length
        struct.pack_into('<H', record, offset + 10, 0)  # Name offset
        struct.pack_into('<H', record, offset + 12, 0)  # Flags
        struct.pack_into('<H', record, offset + 14, 0)  # Attribute ID
        struct.pack_into('<I', record, offset + 16, 48)  # Content length
        struct.pack_into('<H', record, offset + 20, 24)  # Content offset

        # Content: timestamps and flags
        now = self._windows_time()
        content_off = offset + 24
        struct.pack_into('<Q', record, content_off, now)      # Created
        struct.pack_into('<Q', record, content_off + 8, now)  # Modified
        struct.pack_into('<Q', record, content_off + 16, now) # MFT modified
        struct.pack_into('<Q', record, content_off + 24, now) # Accessed
        struct.pack_into('<I', record, content_off + 32, FILE_ATTR_ARCHIVE)  # Flags

        return offset + 96

    def _add_filename(self, record: bytearray, offset: int, name: str, parent: int, is_dir: bool) -> int:
        """Add $FILE_NAME attribute."""
        name_bytes = name.encode('utf-16-le')
        name_len = len(name)

        # Content size: 66 + name_bytes
        content_size = 66 + len(name_bytes)
        # Attribute size (aligned to 8)
        attr_size = ((24 + content_size) + 7) & ~7

        struct.pack_into('<I', record, offset, ATTR_FILE_NAME)
        struct.pack_into('<I', record, offset + 4, attr_size)
        record[offset + 8] = 0  # Resident
        record[offset + 9] = 0  # Name length
        struct.pack_into('<H', record, offset + 10, 0)
        struct.pack_into('<H', record, offset + 12, 0)
        struct.pack_into('<H', record, offset + 14, 1)  # Attribute ID
        struct.pack_into('<I', record, offset + 16, content_size)
        struct.pack_into('<H', record, offset + 20, 24)

        content_off = offset + 24
        # Parent directory reference (MFT record + sequence)
        struct.pack_into('<Q', record, content_off, parent | (1 << 48))

        now = self._windows_time()
        struct.pack_into('<Q', record, content_off + 8, now)   # Created
        struct.pack_into('<Q', record, content_off + 16, now)  # Modified
        struct.pack_into('<Q', record, content_off + 24, now)  # MFT modified
        struct.pack_into('<Q', record, content_off + 32, now)  # Accessed
        struct.pack_into('<Q', record, content_off + 40, 0)    # Allocated size
        struct.pack_into('<Q', record, content_off + 48, 0)    # Real size

        flags = FILE_ATTR_ARCHIVE
        if is_dir:
            flags = FILE_ATTR_DIRECTORY
        struct.pack_into('<I', record, content_off + 56, flags)
        struct.pack_into('<I', record, content_off + 60, 0)    # Reparse value

        record[content_off + 64] = name_len  # Filename length
        record[content_off + 65] = 3  # Namespace (3 = Win32 & DOS)

        record[content_off + 66:content_off + 66 + len(name_bytes)] = name_bytes

        return offset + attr_size

    def _add_data_attribute(self, record: bytearray, offset: int, entry: FileEntry) -> int:
        """Add $DATA attribute for a file."""
        if entry.size == 0 or not entry.clusters:
            # Empty or resident data
            attr_size = 24
            struct.pack_into('<I', record, offset, ATTR_DATA)
            struct.pack_into('<I', record, offset + 4, attr_size)
            record[offset + 8] = 0  # Resident
            record[offset + 9] = 0
            struct.pack_into('<H', record, offset + 10, 0)
            struct.pack_into('<H', record, offset + 12, 0)
            struct.pack_into('<H', record, offset + 14, 2)
            struct.pack_into('<I', record, offset + 16, 0)
            struct.pack_into('<H', record, offset + 20, 24)
            return offset + attr_size

        # Non-resident data with data runs
        data_runs = self._encode_data_runs(entry.clusters)

        # Non-resident header is 64 bytes + data runs
        attr_size = ((64 + len(data_runs)) + 7) & ~7

        struct.pack_into('<I', record, offset, ATTR_DATA)
        struct.pack_into('<I', record, offset + 4, attr_size)
        record[offset + 8] = 1  # Non-resident
        record[offset + 9] = 0  # Name length
        struct.pack_into('<H', record, offset + 10, 0)
        struct.pack_into('<H', record, offset + 12, 0)
        struct.pack_into('<H', record, offset + 14, 2)  # Attribute ID

        # Non-resident specific fields
        struct.pack_into('<Q', record, offset + 16, 0)  # Lowest VCN
        struct.pack_into('<Q', record, offset + 24, len(entry.clusters) - 1)  # Highest VCN
        struct.pack_into('<H', record, offset + 32, 64)  # Data runs offset
        struct.pack_into('<H', record, offset + 34, 0)   # Compression unit
        struct.pack_into('<I', record, offset + 36, 0)   # Padding
        struct.pack_into('<Q', record, offset + 40, len(entry.clusters) * CLUSTER_SIZE)  # Allocated size
        struct.pack_into('<Q', record, offset + 48, entry.size)  # Real size
        struct.pack_into('<Q', record, offset + 56, entry.size)  # Initialized size

        # Data runs
        record[offset + 64:offset + 64 + len(data_runs)] = data_runs

        return offset + attr_size

    def _add_index_root(self, record: bytearray, offset: int, dir_record: int) -> int:
        """Add $INDEX_ROOT attribute for directory."""
        # Get children for this directory
        children = self.dir_children.get(dir_record, [])

        # Build index entries
        index_entries = bytearray()
        for child_rec in sorted(children, key=lambda r: self.files[r].name.upper() if r in self.files else ""):
            if child_rec in self.files:
                entry_data = self._build_index_entry(self.files[child_rec])
                index_entries.extend(entry_data)

        # Add end entry
        end_entry = bytearray(16)
        struct.pack_into('<H', end_entry, 8, 16)  # Entry length
        struct.pack_into('<H', end_entry, 12, 2)  # Flags: last entry
        index_entries.extend(end_entry)

        # Index header (16 bytes) + entries
        index_size = 16 + len(index_entries)

        # Attribute: header (32) + index root header (16) + index header (16) + entries
        content_size = 16 + 16 + len(index_entries)
        attr_size = ((32 + content_size) + 7) & ~7

        struct.pack_into('<I', record, offset, ATTR_INDEX_ROOT)
        struct.pack_into('<I', record, offset + 4, attr_size)
        record[offset + 8] = 0  # Resident
        record[offset + 9] = 4  # Name length ($I30)
        struct.pack_into('<H', record, offset + 10, 24)  # Name offset
        struct.pack_into('<H', record, offset + 12, 0)
        struct.pack_into('<H', record, offset + 14, 3)
        struct.pack_into('<I', record, offset + 16, content_size)
        struct.pack_into('<H', record, offset + 20, 32)  # Content offset

        # Attribute name "$I30"
        record[offset + 24:offset + 32] = "$I30".encode('utf-16-le')

        content_off = offset + 32

        # Index root header
        struct.pack_into('<I', record, content_off, ATTR_FILE_NAME)  # Indexed attribute type
        struct.pack_into('<I', record, content_off + 4, 1)  # Collation rule (filename)
        struct.pack_into('<I', record, content_off + 8, 4096)  # Index block size
        record[content_off + 12] = 1  # Clusters per index block

        # Index header
        idx_hdr_off = content_off + 16
        struct.pack_into('<I', record, idx_hdr_off, 16)  # Entries offset (relative to this header)
        struct.pack_into('<I', record, idx_hdr_off + 4, 16 + len(index_entries))  # Total size
        struct.pack_into('<I', record, idx_hdr_off + 8, 16 + len(index_entries))  # Allocated size
        record[idx_hdr_off + 12] = 0  # Flags (small index, fits in root)

        # Index entries
        record[idx_hdr_off + 16:idx_hdr_off + 16 + len(index_entries)] = index_entries

        return offset + attr_size

    def _build_index_entry(self, entry: FileEntry) -> bytes:
        """Build an index entry for a file/directory."""
        name_bytes = entry.name.encode('utf-16-le')

        # Entry: file ref (8) + entry length (2) + key length (2) + flags (4) +
        #        filename attr content (66 + name)
        key_length = 66 + len(name_bytes)
        entry_length = ((16 + key_length) + 7) & ~7

        data = bytearray(entry_length)

        # File reference
        struct.pack_into('<Q', data, 0, entry.mft_record | (1 << 48))

        struct.pack_into('<H', data, 8, entry_length)
        struct.pack_into('<H', data, 10, key_length)
        struct.pack_into('<I', data, 12, 0)  # Flags

        # Filename attribute content (same as $FILE_NAME content)
        key_off = 16
        struct.pack_into('<Q', data, key_off, entry.parent_record | (1 << 48))

        now = self._windows_time()
        struct.pack_into('<Q', data, key_off + 8, now)
        struct.pack_into('<Q', data, key_off + 16, now)
        struct.pack_into('<Q', data, key_off + 24, now)
        struct.pack_into('<Q', data, key_off + 32, now)
        struct.pack_into('<Q', data, key_off + 40, 0)
        struct.pack_into('<Q', data, key_off + 48, entry.size)

        flags = FILE_ATTR_DIRECTORY if entry.is_directory else FILE_ATTR_ARCHIVE
        struct.pack_into('<I', data, key_off + 56, flags)
        struct.pack_into('<I', data, key_off + 60, 0)

        data[key_off + 64] = len(entry.name)
        data[key_off + 65] = 3
        data[key_off + 66:key_off + 66 + len(name_bytes)] = name_bytes

        return bytes(data)

    def _encode_data_runs(self, clusters: List[int]) -> bytes:
        """Encode cluster list as NTFS data runs."""
        if not clusters:
            return b'\x00'

        runs = bytearray()
        prev_lcn = 0

        i = 0
        while i < len(clusters):
            # Find contiguous run
            start = clusters[i]
            length = 1
            while i + length < len(clusters) and clusters[i + length] == start + length:
                length += 1

            # Encode run
            lcn_offset = start - prev_lcn

            # Determine sizes needed
            len_size = (length.bit_length() + 8) // 8
            off_size = ((lcn_offset.bit_length() if lcn_offset >= 0 else (lcn_offset + 1).bit_length()) + 8) // 8
            if lcn_offset < 0:
                off_size = max(off_size, 1)

            header = (off_size << 4) | len_size
            runs.append(header)

            # Length (little-endian)
            for b in range(len_size):
                runs.append((length >> (b * 8)) & 0xFF)

            # Offset (little-endian, signed)
            if lcn_offset >= 0:
                for b in range(off_size):
                    runs.append((lcn_offset >> (b * 8)) & 0xFF)
            else:
                # Two's complement for negative
                val = lcn_offset & ((1 << (off_size * 8)) - 1)
                for b in range(off_size):
                    runs.append((val >> (b * 8)) & 0xFF)

            prev_lcn = start
            i += length

        runs.append(0)  # End marker
        return bytes(runs)

    def _apply_fixups(self, record: bytearray):
        """Apply NTFS fixup array to record."""
        usa_offset = struct.unpack('<H', record[4:6])[0]
        usa_count = struct.unpack('<H', record[6:8])[0]

        # Generate check value
        check = struct.unpack('<H', record[usa_offset:usa_offset + 2])[0]
        if check == 0:
            check = 1
        struct.pack_into('<H', record, usa_offset, check)

        # Save and replace sector end bytes
        for i in range(1, usa_count):
            sector_end = i * 512 - 2
            if sector_end + 2 <= len(record):
                # Save original
                orig = struct.unpack('<H', record[sector_end:sector_end + 2])[0]
                struct.pack_into('<H', record, usa_offset + i * 2, orig)
                # Write check value
                struct.pack_into('<H', record, sector_end, check)

    def _windows_time(self) -> int:
        """Get current time in Windows FILETIME format."""
        # Windows FILETIME: 100ns intervals since 1601-01-01
        # Unix epoch is 1970-01-01, difference is 11644473600 seconds
        unix_time = time.time()
        return int((unix_time + 11644473600) * 10000000)

    def get_size(self) -> int:
        """Get total volume size."""
        return self.total_size

    def read(self, offset: int, length: int) -> bytes:
        """Read from the synthesized NTFS volume."""
        result = bytearray(length)
        pos = 0

        while pos < length:
            current_offset = offset + pos
            remaining = length - pos

            # Boot sector (first sector)
            if current_offset < SECTOR_SIZE:
                chunk_len = min(remaining, SECTOR_SIZE - current_offset)
                result[pos:pos + chunk_len] = self.boot_sector[current_offset:current_offset + chunk_len]
                pos += chunk_len
                continue

            # MFT region
            if self.mft_offset <= current_offset < self.mft_offset + self._get_mft_size():
                mft_rel = current_offset - self.mft_offset
                record_num = mft_rel // MFT_RECORD_SIZE
                record_offset = mft_rel % MFT_RECORD_SIZE

                chunk_len = min(remaining, MFT_RECORD_SIZE - record_offset)
                record_data = self._get_mft_record(record_num)
                result[pos:pos + chunk_len] = record_data[record_offset:record_offset + chunk_len]
                pos += chunk_len
                continue

            # File data clusters
            cluster = current_offset // CLUSTER_SIZE
            cluster_offset = current_offset % CLUSTER_SIZE

            if cluster in self.cluster_map:
                file_record, file_offset = self.cluster_map[cluster]
                if file_record in self.files:
                    entry = self.files[file_record]
                    read_offset = file_offset + cluster_offset
                    chunk_len = min(remaining, CLUSTER_SIZE - cluster_offset)

                    try:
                        with open(entry.source_path, 'rb') as f:
                            f.seek(read_offset)
                            data = f.read(chunk_len)
                            result[pos:pos + len(data)] = data
                            pos += chunk_len
                            continue
                    except (OSError, IOError):
                        pass

            # Default: zeros
            chunk_len = min(remaining, CLUSTER_SIZE - cluster_offset)
            pos += chunk_len

        return bytes(result)

    def _get_mft_size(self) -> int:
        """Get total MFT size in bytes."""
        max_record = max(list(self.files.keys()) + [15]) + 1
        return max_record * MFT_RECORD_SIZE

    def _get_mft_record(self, record_num: int) -> bytes:
        """Get MFT record by number."""
        if record_num in self.system_records:
            return self.system_records[record_num]

        if record_num in self.files:
            return self._build_file_record(self.files[record_num])

        # Empty record
        return self._build_empty_record(record_num)

    def is_mft_region(self, offset: int, length: int) -> bool:
        """Check if offset is in MFT region."""
        mft_end = self.mft_offset + self._get_mft_size()
        return offset < mft_end and offset + length > self.mft_offset


# Test
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dynamic_ntfs.py <source_dir>")
        sys.exit(1)

    synth = DynamicNTFSSynthesizer(sys.argv[1])

    print(f"\nVolume size: {synth.get_size()} bytes")
    print(f"Files mapped:")
    for rec, entry in sorted(synth.files.items()):
        print(f"  Record {rec}: {entry.name} ({entry.size} bytes, clusters: {entry.clusters[:3]}{'...' if len(entry.clusters) > 3 else ''})")
