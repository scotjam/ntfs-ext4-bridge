"""NTFS attribute generation."""

import struct
import time
from typing import List, Optional, Tuple
from .constants import (
    ATTR_STANDARD_INFORMATION, ATTR_FILE_NAME, ATTR_DATA,
    ATTR_INDEX_ROOT, ATTR_INDEX_ALLOCATION, ATTR_BITMAP,
    ATTR_VOLUME_NAME, ATTR_VOLUME_INFORMATION, ATTR_END,
    FILE_ATTR_ARCHIVE, FILE_ATTR_DIRECTORY, FILE_ATTR_NORMAL,
    FILE_ATTR_SYSTEM, FILE_ATTR_HIDDEN,
    FILENAME_WIN32_AND_DOS, NTFS_VERSION_MAJOR, NTFS_VERSION_MINOR
)


def unix_to_ntfs_time(unix_time: float) -> int:
    """Convert Unix timestamp to NTFS timestamp (100ns intervals since 1601-01-01)."""
    # Seconds between 1601-01-01 and 1970-01-01
    EPOCH_DIFF = 11644473600
    return int((unix_time + EPOCH_DIFF) * 10000000)


def current_ntfs_time() -> int:
    """Get current time in NTFS format."""
    return unix_to_ntfs_time(time.time())


class AttributeHeader:
    """Base class for attribute headers."""

    def __init__(self, attr_type: int, name: str = "", resident: bool = True):
        self.attr_type = attr_type
        self.name = name
        self.resident = resident

    def build_header(self, data_size: int, name_bytes: bytes = b"") -> bytes:
        """Build common attribute header."""
        header = bytearray(24 if self.resident else 64)

        # Attribute type (4 bytes)
        struct.pack_into('<I', header, 0, self.attr_type)

        name_len = len(self.name)
        name_offset = 24 if self.resident else 64

        if self.resident:
            # Total length including header, name, and data
            total_len = 24 + len(name_bytes) + data_size
            # Align to 8 bytes
            total_len = (total_len + 7) & ~7

            struct.pack_into('<I', header, 4, total_len)  # Record length
            header[8] = 0  # Resident flag (0 = resident)
            header[9] = name_len  # Name length (in characters)
            struct.pack_into('<H', header, 10, name_offset)  # Name offset
            struct.pack_into('<H', header, 12, 0)  # Flags
            struct.pack_into('<H', header, 14, 0)  # Instance

            # Resident-specific fields
            struct.pack_into('<I', header, 16, data_size)  # Value length
            value_offset = name_offset + len(name_bytes)
            value_offset = (value_offset + 7) & ~7  # Align
            struct.pack_into('<H', header, 20, value_offset)  # Value offset
            header[22] = 0  # Indexed flag
            header[23] = 0  # Padding
        else:
            # Non-resident header is handled separately
            pass

        return bytes(header)


class StandardInformation:
    """$STANDARD_INFORMATION attribute (0x10)."""

    def __init__(self, ctime: float, mtime: float, atime: float,
                 file_attrs: int = FILE_ATTR_ARCHIVE):
        self.ctime = unix_to_ntfs_time(ctime)
        self.mtime = unix_to_ntfs_time(mtime)
        self.atime = unix_to_ntfs_time(atime)
        self.file_attrs = file_attrs

    def to_bytes(self) -> bytes:
        """Generate complete attribute."""
        # Attribute data (48 bytes for NTFS 3.x)
        data = bytearray(48)

        # Creation time (8 bytes)
        struct.pack_into('<Q', data, 0, self.ctime)
        # Modification time (8 bytes)
        struct.pack_into('<Q', data, 8, self.mtime)
        # MFT modification time (8 bytes)
        struct.pack_into('<Q', data, 16, self.mtime)
        # Access time (8 bytes)
        struct.pack_into('<Q', data, 24, self.atime)
        # File attributes (4 bytes)
        struct.pack_into('<I', data, 32, self.file_attrs)
        # Maximum versions (4 bytes)
        struct.pack_into('<I', data, 36, 0)
        # Version number (4 bytes)
        struct.pack_into('<I', data, 40, 0)
        # Class ID (4 bytes)
        struct.pack_into('<I', data, 44, 0)

        # Build header
        header = AttributeHeader(ATTR_STANDARD_INFORMATION)
        header_bytes = header.build_header(len(data))

        # Combine header and data
        result = bytearray(header_bytes)
        # Pad to align
        while len(result) < 24:
            result.append(0)
        result.extend(data)

        # Align to 8 bytes
        while len(result) % 8 != 0:
            result.append(0)

        return bytes(result)


class FileName:
    """$FILE_NAME attribute (0x30)."""

    def __init__(self, parent_ref: int, parent_seq: int, name: str,
                 ctime: float, mtime: float, atime: float,
                 file_size: int = 0, alloc_size: int = 0,
                 file_attrs: int = FILE_ATTR_ARCHIVE,
                 namespace: int = FILENAME_WIN32_AND_DOS):
        self.parent_ref = parent_ref
        self.parent_seq = parent_seq
        self.name = name
        self.ctime = unix_to_ntfs_time(ctime)
        self.mtime = unix_to_ntfs_time(mtime)
        self.atime = unix_to_ntfs_time(atime)
        self.file_size = file_size
        self.alloc_size = alloc_size
        self.file_attrs = file_attrs
        self.namespace = namespace

    def to_bytes(self) -> bytes:
        """Generate complete attribute."""
        name_utf16 = self.name.encode('utf-16-le')
        name_len = len(self.name)

        # Attribute data (66 bytes base + variable name)
        data = bytearray(66 + len(name_utf16))

        # Parent directory MFT reference (8 bytes)
        # Low 48 bits = record number, high 16 bits = sequence
        parent_mft_ref = (self.parent_ref & 0xFFFFFFFFFFFF) | ((self.parent_seq & 0xFFFF) << 48)
        struct.pack_into('<Q', data, 0, parent_mft_ref)

        # Creation time (8 bytes)
        struct.pack_into('<Q', data, 8, self.ctime)
        # Modification time (8 bytes)
        struct.pack_into('<Q', data, 16, self.mtime)
        # MFT modification time (8 bytes)
        struct.pack_into('<Q', data, 24, self.mtime)
        # Access time (8 bytes)
        struct.pack_into('<Q', data, 32, self.atime)
        # Allocated size (8 bytes)
        struct.pack_into('<Q', data, 40, self.alloc_size)
        # Real size (8 bytes)
        struct.pack_into('<Q', data, 48, self.file_size)
        # Flags (4 bytes)
        struct.pack_into('<I', data, 56, self.file_attrs)
        # Reparse value / EA (4 bytes)
        struct.pack_into('<I', data, 60, 0)
        # Filename length in characters (1 byte)
        data[64] = name_len
        # Filename namespace (1 byte)
        data[65] = self.namespace
        # Filename in UTF-16LE
        data[66:66 + len(name_utf16)] = name_utf16

        # Build header
        header = bytearray(24)
        data_len = len(data)
        total_len = (24 + data_len + 7) & ~7

        struct.pack_into('<I', header, 0, ATTR_FILE_NAME)  # Type
        struct.pack_into('<I', header, 4, total_len)  # Length
        header[8] = 0  # Resident
        header[9] = 0  # Name length
        struct.pack_into('<H', header, 10, 24)  # Name offset
        struct.pack_into('<H', header, 12, 0)  # Flags
        struct.pack_into('<H', header, 14, 0)  # Instance
        struct.pack_into('<I', header, 16, data_len)  # Value length
        struct.pack_into('<H', header, 20, 24)  # Value offset
        header[22] = 1  # Indexed
        header[23] = 0  # Padding

        result = bytearray(header)
        result.extend(data)

        # Align to 8 bytes
        while len(result) % 8 != 0:
            result.append(0)

        return bytes(result)


class DataAttribute:
    """$DATA attribute (0x80)."""

    def __init__(self, data: bytes = b"", is_resident: bool = True,
                 data_runs: bytes = b"", real_size: int = 0, alloc_size: int = 0):
        self.data = data
        self.is_resident = is_resident
        self.data_runs = data_runs
        self.real_size = real_size
        self.alloc_size = alloc_size

    def to_bytes(self) -> bytes:
        """Generate complete attribute."""
        if self.is_resident:
            # Resident $DATA
            header = bytearray(24)
            data_len = len(self.data)
            total_len = (24 + data_len + 7) & ~7

            struct.pack_into('<I', header, 0, ATTR_DATA)  # Type
            struct.pack_into('<I', header, 4, total_len)  # Length
            header[8] = 0  # Resident
            header[9] = 0  # Name length
            struct.pack_into('<H', header, 10, 24)  # Name offset
            struct.pack_into('<H', header, 12, 0)  # Flags
            struct.pack_into('<H', header, 14, 0)  # Instance
            struct.pack_into('<I', header, 16, data_len)  # Value length
            struct.pack_into('<H', header, 20, 24)  # Value offset

            result = bytearray(header)
            result.extend(self.data)
        else:
            # Non-resident $DATA
            header = bytearray(64)
            runs_len = len(self.data_runs)
            total_len = (64 + runs_len + 7) & ~7

            struct.pack_into('<I', header, 0, ATTR_DATA)  # Type
            struct.pack_into('<I', header, 4, total_len)  # Length
            header[8] = 1  # Non-resident
            header[9] = 0  # Name length
            struct.pack_into('<H', header, 10, 64)  # Name offset
            struct.pack_into('<H', header, 12, 0)  # Flags
            struct.pack_into('<H', header, 14, 0)  # Instance

            # Non-resident specific fields
            struct.pack_into('<Q', header, 16, 0)  # Starting VCN
            struct.pack_into('<Q', header, 24, (self.alloc_size // 4096) - 1 if self.alloc_size else 0)  # Ending VCN
            struct.pack_into('<H', header, 32, 64)  # Data runs offset
            struct.pack_into('<H', header, 34, 0)  # Compression unit
            struct.pack_into('<I', header, 36, 0)  # Padding
            struct.pack_into('<Q', header, 40, self.alloc_size)  # Allocated size
            struct.pack_into('<Q', header, 48, self.real_size)  # Real size
            struct.pack_into('<Q', header, 56, self.real_size)  # Initialized size

            result = bytearray(header)
            result.extend(self.data_runs)

        # Align to 8 bytes
        while len(result) % 8 != 0:
            result.append(0)

        return bytes(result)


class BitmapAttribute:
    """$BITMAP attribute (0xB0) - for MFT record allocation bitmap."""

    def __init__(self, data: bytes = b"", is_resident: bool = True,
                 data_runs: bytes = b"", real_size: int = 0, alloc_size: int = 0):
        self.data = data
        self.is_resident = is_resident
        self.data_runs = data_runs
        self.real_size = real_size
        self.alloc_size = alloc_size

    def to_bytes(self) -> bytes:
        """Generate complete attribute."""
        if self.is_resident:
            header = bytearray(24)
            data_len = len(self.data)
            total_len = (24 + data_len + 7) & ~7

            struct.pack_into('<I', header, 0, ATTR_BITMAP)  # Type
            struct.pack_into('<I', header, 4, total_len)  # Length
            header[8] = 0  # Resident
            header[9] = 0  # Name length
            struct.pack_into('<H', header, 10, 24)  # Name offset
            struct.pack_into('<H', header, 12, 0)  # Flags
            struct.pack_into('<H', header, 14, 0)  # Instance
            struct.pack_into('<I', header, 16, data_len)  # Value length
            struct.pack_into('<H', header, 20, 24)  # Value offset

            result = bytearray(header)
            result.extend(self.data)
        else:
            # Non-resident $BITMAP
            header = bytearray(64)
            runs_len = len(self.data_runs)
            total_len = (64 + runs_len + 7) & ~7

            struct.pack_into('<I', header, 0, ATTR_BITMAP)  # Type
            struct.pack_into('<I', header, 4, total_len)  # Length
            header[8] = 1  # Non-resident
            header[9] = 0  # Name length
            struct.pack_into('<H', header, 10, 64)  # Name offset
            struct.pack_into('<H', header, 12, 0)  # Flags
            struct.pack_into('<H', header, 14, 0)  # Instance

            struct.pack_into('<Q', header, 16, 0)  # Starting VCN
            struct.pack_into('<Q', header, 24, (self.alloc_size // 4096) - 1 if self.alloc_size else 0)
            struct.pack_into('<H', header, 32, 64)  # Data runs offset
            struct.pack_into('<H', header, 34, 0)  # Compression unit
            struct.pack_into('<I', header, 36, 0)  # Padding
            struct.pack_into('<Q', header, 40, self.alloc_size)
            struct.pack_into('<Q', header, 48, self.real_size)
            struct.pack_into('<Q', header, 56, self.real_size)

            result = bytearray(header)
            result.extend(self.data_runs)

        while len(result) % 8 != 0:
            result.append(0)

        return bytes(result)


class VolumeName:
    """$VOLUME_NAME attribute (0x60)."""

    def __init__(self, name: str = "NTFS-Bridge"):
        self.name = name

    def to_bytes(self) -> bytes:
        """Generate complete attribute."""
        name_utf16 = self.name.encode('utf-16-le')

        header = bytearray(24)
        data_len = len(name_utf16)
        total_len = (24 + data_len + 7) & ~7

        struct.pack_into('<I', header, 0, ATTR_VOLUME_NAME)  # Type
        struct.pack_into('<I', header, 4, total_len)  # Length
        header[8] = 0  # Resident
        header[9] = 0  # Name length
        struct.pack_into('<H', header, 10, 24)  # Name offset
        struct.pack_into('<H', header, 12, 0)  # Flags
        struct.pack_into('<H', header, 14, 0)  # Instance
        struct.pack_into('<I', header, 16, data_len)  # Value length
        struct.pack_into('<H', header, 20, 24)  # Value offset

        result = bytearray(header)
        result.extend(name_utf16)

        while len(result) % 8 != 0:
            result.append(0)

        return bytes(result)


class VolumeInformation:
    """$VOLUME_INFORMATION attribute (0x70)."""

    def __init__(self, major: int = NTFS_VERSION_MAJOR, minor: int = NTFS_VERSION_MINOR):
        self.major = major
        self.minor = minor

    def to_bytes(self) -> bytes:
        """Generate complete attribute."""
        data = bytearray(12)
        struct.pack_into('<Q', data, 0, 0)  # Reserved
        data[8] = self.major  # Major version
        data[9] = self.minor  # Minor version
        struct.pack_into('<H', data, 10, 0)  # Flags

        header = bytearray(24)
        data_len = len(data)
        total_len = (24 + data_len + 7) & ~7

        struct.pack_into('<I', header, 0, ATTR_VOLUME_INFORMATION)
        struct.pack_into('<I', header, 4, total_len)
        header[8] = 0  # Resident
        struct.pack_into('<I', header, 16, data_len)
        struct.pack_into('<H', header, 20, 24)

        result = bytearray(header)
        result.extend(data)

        while len(result) % 8 != 0:
            result.append(0)

        return bytes(result)


class IndexRoot:
    """$INDEX_ROOT attribute (0x90) for directory."""

    def __init__(self, entries: List[bytes] = None):
        self.entries = entries or []

    def to_bytes(self) -> bytes:
        """Generate complete attribute with index header."""
        # Index root header (16 bytes)
        index_header = bytearray(16)
        struct.pack_into('<I', index_header, 0, ATTR_FILE_NAME)  # Indexed attr type
        struct.pack_into('<I', index_header, 4, 1)  # Collation rule (filename)
        struct.pack_into('<I', index_header, 8, 4096)  # Index block size
        index_header[12] = 1  # Clusters per index block

        # Node header (16 bytes)
        node_header = bytearray(16)
        entries_offset = 16  # Offset to first entry from start of node header

        # Calculate entries size
        entries_data = bytearray()
        for entry in self.entries:
            entries_data.extend(entry)

        # Add end entry
        end_entry = self._create_end_entry()
        entries_data.extend(end_entry)

        total_entries_size = len(entries_data)
        allocated_size = total_entries_size + entries_offset

        struct.pack_into('<I', node_header, 0, entries_offset)  # Offset to first entry
        struct.pack_into('<I', node_header, 4, total_entries_size + entries_offset)  # Total size of entries
        struct.pack_into('<I', node_header, 8, allocated_size)  # Allocated size
        node_header[12] = 0  # Flags (0 = small index, no INDEX_ALLOCATION needed)

        # Combine all parts
        data = bytearray()
        data.extend(index_header)
        data.extend(node_header)
        data.extend(entries_data)

        # Build attribute header
        header = bytearray(24)
        data_len = len(data)
        total_len = (24 + data_len + 7) & ~7

        struct.pack_into('<I', header, 0, ATTR_INDEX_ROOT)
        struct.pack_into('<I', header, 4, total_len)
        header[8] = 0  # Resident
        header[9] = 4  # Name length ($I30)
        struct.pack_into('<H', header, 10, 24)  # Name offset
        struct.pack_into('<I', header, 16, data_len)
        name_bytes = "$I30".encode('utf-16-le')
        value_offset = 24 + len(name_bytes)
        value_offset = (value_offset + 7) & ~7
        struct.pack_into('<H', header, 20, value_offset)

        result = bytearray(header)
        result.extend(name_bytes)
        while len(result) < value_offset:
            result.append(0)
        result.extend(data)

        # Recalculate total length
        total_len = (len(result) + 7) & ~7
        struct.pack_into('<I', result, 4, total_len)

        while len(result) % 8 != 0:
            result.append(0)

        return bytes(result)

    def _create_end_entry(self) -> bytes:
        """Create end-of-index entry."""
        entry = bytearray(16)
        struct.pack_into('<Q', entry, 0, 0)  # MFT reference (unused for end entry)
        struct.pack_into('<H', entry, 8, 16)  # Entry length
        struct.pack_into('<H', entry, 10, 0)  # Key length
        struct.pack_into('<I', entry, 12, 2)  # Flags (INDEX_ENTRY_END)
        return bytes(entry)


def create_index_entry(mft_ref: int, seq_num: int, filename_attr: bytes) -> bytes:
    """Create an index entry for a directory."""
    # Index entry structure
    key_len = len(filename_attr) - 24  # Subtract attribute header size
    entry_len = 16 + key_len  # 16-byte header + key
    entry_len = (entry_len + 7) & ~7  # Align to 8 bytes

    entry = bytearray(entry_len)

    # MFT reference (8 bytes)
    mft_reference = (mft_ref & 0xFFFFFFFFFFFF) | ((seq_num & 0xFFFF) << 48)
    struct.pack_into('<Q', entry, 0, mft_reference)

    # Entry length (2 bytes)
    struct.pack_into('<H', entry, 8, entry_len)

    # Key length (2 bytes) - length of $FILE_NAME content
    struct.pack_into('<H', entry, 10, key_len)

    # Flags (4 bytes)
    struct.pack_into('<I', entry, 12, 0)

    # Copy filename attribute content (without header) as key
    # The key is the $FILE_NAME attribute value
    fn_attr = filename_attr[24:]  # Skip 24-byte header
    entry[16:16 + len(fn_attr)] = fn_attr

    return bytes(entry)


class NamedDataAttribute:
    """Named $DATA attribute for streams like $SDS."""

    def __init__(self, name: str, data: bytes = b"", is_resident: bool = True,
                 data_runs: bytes = b"", real_size: int = 0, alloc_size: int = 0):
        self.name = name
        self.data = data
        self.is_resident = is_resident
        self.data_runs = data_runs
        self.real_size = real_size
        self.alloc_size = alloc_size

    def to_bytes(self) -> bytes:
        """Generate complete attribute with name."""
        name_utf16 = self.name.encode('utf-16-le')
        name_len = len(self.name)

        if self.is_resident:
            # Header + name + data
            name_offset = 24
            value_offset = (name_offset + len(name_utf16) + 7) & ~7
            data_len = len(self.data)
            total_len = (value_offset + data_len + 7) & ~7

            header = bytearray(total_len)
            struct.pack_into('<I', header, 0, ATTR_DATA)
            struct.pack_into('<I', header, 4, total_len)
            header[8] = 0  # Resident
            header[9] = name_len
            struct.pack_into('<H', header, 10, name_offset)
            struct.pack_into('<H', header, 12, 0)  # Flags
            struct.pack_into('<H', header, 14, 0)  # Instance
            struct.pack_into('<I', header, 16, data_len)
            struct.pack_into('<H', header, 20, value_offset)
            header[name_offset:name_offset + len(name_utf16)] = name_utf16
            header[value_offset:value_offset + data_len] = self.data
            return bytes(header)
        else:
            # Non-resident with name
            name_offset = 64
            runs_offset = (name_offset + len(name_utf16) + 7) & ~7
            runs_len = len(self.data_runs)
            total_len = (runs_offset + runs_len + 7) & ~7

            header = bytearray(total_len)
            struct.pack_into('<I', header, 0, ATTR_DATA)
            struct.pack_into('<I', header, 4, total_len)
            header[8] = 1  # Non-resident
            header[9] = name_len
            struct.pack_into('<H', header, 10, name_offset)
            struct.pack_into('<H', header, 12, 0)  # Flags
            struct.pack_into('<H', header, 14, 0)  # Instance
            struct.pack_into('<Q', header, 16, 0)  # Starting VCN
            ending_vcn = (self.alloc_size // 4096) - 1 if self.alloc_size else 0
            struct.pack_into('<Q', header, 24, ending_vcn)
            struct.pack_into('<H', header, 32, runs_offset)
            struct.pack_into('<H', header, 34, 0)  # Compression
            struct.pack_into('<I', header, 36, 0)  # Padding
            struct.pack_into('<Q', header, 40, self.alloc_size)
            struct.pack_into('<Q', header, 48, self.real_size)
            struct.pack_into('<Q', header, 56, self.real_size)
            header[name_offset:name_offset + len(name_utf16)] = name_utf16
            header[runs_offset:runs_offset + runs_len] = self.data_runs
            return bytes(header)


class NamedIndexRoot:
    """Named $INDEX_ROOT attribute for indexes like $SDH, $SII."""

    def __init__(self, name: str, indexed_attr_type: int = 0, collation: int = 0):
        self.name = name
        self.indexed_attr_type = indexed_attr_type
        self.collation = collation

    def to_bytes(self) -> bytes:
        """Generate named index root with empty index."""
        name_utf16 = self.name.encode('utf-16-le')
        name_len = len(self.name)

        # Index root header (16 bytes) + node header (16 bytes) + end entry (16 bytes)
        index_data = bytearray(48)
        struct.pack_into('<I', index_data, 0, self.indexed_attr_type)
        struct.pack_into('<I', index_data, 4, self.collation)
        struct.pack_into('<I', index_data, 8, 4096)  # Index block size
        index_data[12] = 1  # Clusters per index block

        # Node header
        struct.pack_into('<I', index_data, 16, 16)  # Offset to first entry
        struct.pack_into('<I', index_data, 20, 32)  # Total size
        struct.pack_into('<I', index_data, 24, 32)  # Allocated size
        index_data[28] = 0  # Flags (small index)

        # End entry
        struct.pack_into('<Q', index_data, 32, 0)  # MFT ref
        struct.pack_into('<H', index_data, 40, 16)  # Entry length
        struct.pack_into('<H', index_data, 42, 0)  # Key length
        struct.pack_into('<I', index_data, 44, 2)  # Flags (end)

        # Attribute header
        name_offset = 24
        value_offset = (name_offset + len(name_utf16) + 7) & ~7
        data_len = len(index_data)
        total_len = (value_offset + data_len + 7) & ~7

        header = bytearray(total_len)
        struct.pack_into('<I', header, 0, ATTR_INDEX_ROOT)
        struct.pack_into('<I', header, 4, total_len)
        header[8] = 0  # Resident
        header[9] = name_len
        struct.pack_into('<H', header, 10, name_offset)
        struct.pack_into('<H', header, 12, 0)
        struct.pack_into('<H', header, 14, 0)
        struct.pack_into('<I', header, 16, data_len)
        struct.pack_into('<H', header, 20, value_offset)
        header[name_offset:name_offset + len(name_utf16)] = name_utf16
        header[value_offset:value_offset + data_len] = index_data

        return bytes(header)


def end_marker() -> bytes:
    """Return attribute end marker."""
    return struct.pack('<I', ATTR_END)
