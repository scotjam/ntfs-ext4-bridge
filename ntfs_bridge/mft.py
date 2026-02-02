"""MFT record generation."""

import struct
from typing import List, Optional, Dict, Tuple
from .constants import (
    MFT_RECORD_SIZE, FILE_SIGNATURE, CLUSTER_SIZE,
    MFT_RECORD_MFT, MFT_RECORD_MFTMIRR, MFT_RECORD_LOGFILE,
    MFT_RECORD_VOLUME, MFT_RECORD_ATTRDEF, MFT_RECORD_ROOT,
    MFT_RECORD_BITMAP, MFT_RECORD_BOOT, MFT_RECORD_BADCLUS,
    MFT_RECORD_SECURE, MFT_RECORD_UPCASE, MFT_RECORD_EXTEND,
    MFT_RECORD_USER_START,
    MFT_RECORD_IN_USE, MFT_RECORD_IS_DIRECTORY,
    FILE_ATTR_ARCHIVE, FILE_ATTR_DIRECTORY, FILE_ATTR_SYSTEM, FILE_ATTR_HIDDEN
)
from .attributes import (
    StandardInformation, FileName, DataAttribute, VolumeName,
    VolumeInformation, IndexRoot, create_index_entry, end_marker,
    current_ntfs_time, unix_to_ntfs_time
)


class MFTRecord:
    """Represents a single MFT record."""

    def __init__(self, record_num: int, is_directory: bool = False,
                 sequence: int = 1, base_record: int = 0):
        self.record_num = record_num
        self.is_directory = is_directory
        self.sequence = sequence
        self.base_record = base_record
        self.attributes: List[bytes] = []
        self.hard_link_count = 1

    def add_attribute(self, attr_bytes: bytes):
        """Add a serialized attribute."""
        self.attributes.append(attr_bytes)

    def to_bytes(self) -> bytes:
        """Serialize to MFT_RECORD_SIZE bytes."""
        record = bytearray(MFT_RECORD_SIZE)

        # FILE signature (4 bytes)
        record[0:4] = FILE_SIGNATURE

        # Update sequence offset (2 bytes) - points to fixup array
        usa_offset = 48  # Standard offset after header
        struct.pack_into('<H', record, 4, usa_offset)

        # Update sequence count (2 bytes) - number of fixup entries
        # For 1024-byte record with 512-byte sectors: 1 USN + 2 entries = 3
        usa_count = 3
        struct.pack_into('<H', record, 6, usa_count)

        # $LogFile sequence number (8 bytes)
        struct.pack_into('<Q', record, 8, 0)

        # Sequence number (2 bytes)
        struct.pack_into('<H', record, 16, self.sequence)

        # Hard link count (2 bytes)
        struct.pack_into('<H', record, 18, self.hard_link_count)

        # Offset to first attribute (2 bytes)
        first_attr_offset = usa_offset + (usa_count * 2)
        first_attr_offset = (first_attr_offset + 7) & ~7  # Align to 8
        struct.pack_into('<H', record, 20, first_attr_offset)

        # Flags (2 bytes)
        flags = MFT_RECORD_IN_USE
        if self.is_directory:
            flags |= MFT_RECORD_IS_DIRECTORY
        struct.pack_into('<H', record, 22, flags)

        # Used size of MFT record (4 bytes) - will update later
        # struct.pack_into('<I', record, 24, used_size)

        # Allocated size of MFT record (4 bytes)
        struct.pack_into('<I', record, 28, MFT_RECORD_SIZE)

        # Base MFT record (8 bytes)
        struct.pack_into('<Q', record, 32, self.base_record)

        # Next attribute ID (2 bytes)
        struct.pack_into('<H', record, 40, len(self.attributes))

        # Padding (2 bytes) - for XP and later
        struct.pack_into('<H', record, 42, 0)

        # MFT record number (4 bytes) - for XP and later
        struct.pack_into('<I', record, 44, self.record_num)

        # Update sequence array
        # First entry is the USN (update sequence number)
        usn = 1  # Simple incrementing value
        struct.pack_into('<H', record, usa_offset, usn)
        # Following entries are the original values at end of each sector
        # We'll set them to match the USN for now
        for i in range(1, usa_count):
            struct.pack_into('<H', record, usa_offset + (i * 2), 0)

        # Write attributes
        attr_offset = first_attr_offset
        for attr in self.attributes:
            if attr_offset + len(attr) > MFT_RECORD_SIZE - 4:
                break  # Not enough space
            record[attr_offset:attr_offset + len(attr)] = attr
            attr_offset += len(attr)

        # Write end marker
        end = end_marker()
        record[attr_offset:attr_offset + len(end)] = end
        attr_offset += len(end)

        # Update used size
        struct.pack_into('<I', record, 24, attr_offset)

        # Apply fixups - replace last 2 bytes of each sector with USN
        # and store original bytes in fixup array
        for i in range(1, usa_count):
            sector_end = (i * 512) - 2
            # Store original bytes
            original = struct.unpack('<H', record[sector_end:sector_end + 2])[0]
            struct.pack_into('<H', record, usa_offset + (i * 2), original)
            # Replace with USN
            struct.pack_into('<H', record, sector_end, usn)

        return bytes(record)


class MFTGenerator:
    """Generates MFT records for a filesystem."""

    def __init__(self, mft_start_cluster: int):
        self.mft_start_cluster = mft_start_cluster
        self.records: Dict[int, MFTRecord] = {}
        self.next_record = MFT_RECORD_USER_START
        self.path_to_record: Dict[str, int] = {}
        self.record_to_path: Dict[int, str] = {}

        # Create system records
        self._create_system_records()

    def _create_system_records(self):
        """Create the 16 system MFT records."""
        now = current_ntfs_time()

        # $MFT (record 0)
        mft = MFTRecord(MFT_RECORD_MFT, is_directory=False)
        mft.add_attribute(StandardInformation(0, 0, 0, FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        mft.add_attribute(FileName(MFT_RECORD_MFT, 1, "$MFT", 0, 0, 0,
                                   file_attrs=FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        # $MFT's $DATA will be set later when we know total size
        self.records[MFT_RECORD_MFT] = mft

        # $MFTMirr (record 1)
        mftmirr = MFTRecord(MFT_RECORD_MFTMIRR, is_directory=False)
        mftmirr.add_attribute(StandardInformation(0, 0, 0, FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        mftmirr.add_attribute(FileName(MFT_RECORD_ROOT, 1, "$MFTMirr", 0, 0, 0,
                                       file_attrs=FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        self.records[MFT_RECORD_MFTMIRR] = mftmirr

        # $LogFile (record 2)
        logfile = MFTRecord(MFT_RECORD_LOGFILE, is_directory=False)
        logfile.add_attribute(StandardInformation(0, 0, 0, FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        logfile.add_attribute(FileName(MFT_RECORD_ROOT, 1, "$LogFile", 0, 0, 0,
                                       file_attrs=FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        self.records[MFT_RECORD_LOGFILE] = logfile

        # $Volume (record 3)
        volume = MFTRecord(MFT_RECORD_VOLUME, is_directory=False)
        volume.add_attribute(StandardInformation(0, 0, 0, FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        volume.add_attribute(FileName(MFT_RECORD_ROOT, 1, "$Volume", 0, 0, 0,
                                      file_attrs=FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        volume.add_attribute(VolumeName("NTFS-Bridge").to_bytes())
        volume.add_attribute(VolumeInformation().to_bytes())
        self.records[MFT_RECORD_VOLUME] = volume

        # $AttrDef (record 4)
        attrdef = MFTRecord(MFT_RECORD_ATTRDEF, is_directory=False)
        attrdef.add_attribute(StandardInformation(0, 0, 0, FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        attrdef.add_attribute(FileName(MFT_RECORD_ROOT, 1, "$AttrDef", 0, 0, 0,
                                       file_attrs=FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        self.records[MFT_RECORD_ATTRDEF] = attrdef

        # Root directory (record 5)
        root = MFTRecord(MFT_RECORD_ROOT, is_directory=True)
        root.add_attribute(StandardInformation(0, 0, 0, FILE_ATTR_DIRECTORY).to_bytes())
        root.add_attribute(FileName(MFT_RECORD_ROOT, 1, ".", 0, 0, 0,
                                    file_attrs=FILE_ATTR_DIRECTORY).to_bytes())
        root.add_attribute(IndexRoot([]).to_bytes())  # Will be updated
        self.records[MFT_RECORD_ROOT] = root
        self.path_to_record["."] = MFT_RECORD_ROOT
        self.record_to_path[MFT_RECORD_ROOT] = "."

        # $Bitmap (record 6)
        bitmap = MFTRecord(MFT_RECORD_BITMAP, is_directory=False)
        bitmap.add_attribute(StandardInformation(0, 0, 0, FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        bitmap.add_attribute(FileName(MFT_RECORD_ROOT, 1, "$Bitmap", 0, 0, 0,
                                      file_attrs=FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        self.records[MFT_RECORD_BITMAP] = bitmap

        # $Boot (record 7)
        boot = MFTRecord(MFT_RECORD_BOOT, is_directory=False)
        boot.add_attribute(StandardInformation(0, 0, 0, FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        boot.add_attribute(FileName(MFT_RECORD_ROOT, 1, "$Boot", 0, 0, 0,
                                    file_attrs=FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        self.records[MFT_RECORD_BOOT] = boot

        # $BadClus (record 8)
        badclus = MFTRecord(MFT_RECORD_BADCLUS, is_directory=False)
        badclus.add_attribute(StandardInformation(0, 0, 0, FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        badclus.add_attribute(FileName(MFT_RECORD_ROOT, 1, "$BadClus", 0, 0, 0,
                                       file_attrs=FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        self.records[MFT_RECORD_BADCLUS] = badclus

        # $Secure (record 9) - needs $SDS, $SDH, $SII streams
        from .attributes import NamedDataAttribute, NamedIndexRoot
        secure = MFTRecord(MFT_RECORD_SECURE, is_directory=False)
        secure.add_attribute(StandardInformation(0, 0, 0, FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        secure.add_attribute(FileName(MFT_RECORD_ROOT, 1, "$Secure", 0, 0, 0,
                                      file_attrs=FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        # $SDS - Security Descriptor Stream (empty resident for now)
        secure.add_attribute(NamedDataAttribute("$SDS", data=b"").to_bytes())
        # $SDH - Security Descriptor Hash index (collation=0x10 for security hash)
        secure.add_attribute(NamedIndexRoot("$SDH", indexed_attr_type=0, collation=0x10).to_bytes())
        # $SII - Security ID Index (collation=0x11 for security ID)
        secure.add_attribute(NamedIndexRoot("$SII", indexed_attr_type=0, collation=0x11).to_bytes())
        self.records[MFT_RECORD_SECURE] = secure

        # $UpCase (record 10)
        upcase = MFTRecord(MFT_RECORD_UPCASE, is_directory=False)
        upcase.add_attribute(StandardInformation(0, 0, 0, FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        upcase.add_attribute(FileName(MFT_RECORD_ROOT, 1, "$UpCase", 0, 0, 0,
                                      file_attrs=FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM).to_bytes())
        self.records[MFT_RECORD_UPCASE] = upcase

        # $Extend (record 11)
        extend = MFTRecord(MFT_RECORD_EXTEND, is_directory=True)
        extend.add_attribute(StandardInformation(0, 0, 0, FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM | FILE_ATTR_DIRECTORY).to_bytes())
        extend.add_attribute(FileName(MFT_RECORD_ROOT, 1, "$Extend", 0, 0, 0,
                                      file_attrs=FILE_ATTR_HIDDEN | FILE_ATTR_SYSTEM | FILE_ATTR_DIRECTORY).to_bytes())
        extend.add_attribute(IndexRoot([]).to_bytes())
        self.records[MFT_RECORD_EXTEND] = extend

        # Records 12-15 are reserved
        for i in range(12, 16):
            reserved = MFTRecord(i, is_directory=False)
            # Empty reserved records
            self.records[i] = reserved

    def add_file(self, rel_path: str, ctime: float, mtime: float, atime: float,
                 size: int, data_runs: bytes, alloc_size: int) -> int:
        """Add a file and return its MFT record number."""
        record_num = self.next_record
        self.next_record += 1

        # Get parent directory
        import os.path
        parent_path = os.path.dirname(rel_path)
        if not parent_path or parent_path == ".":
            parent_record = MFT_RECORD_ROOT
        else:
            parent_record = self.path_to_record.get(parent_path, MFT_RECORD_ROOT)

        filename = os.path.basename(rel_path)

        record = MFTRecord(record_num, is_directory=False)
        record.add_attribute(StandardInformation(ctime, mtime, atime, FILE_ATTR_ARCHIVE).to_bytes())
        record.add_attribute(FileName(parent_record, 1, filename, ctime, mtime, atime,
                                      file_size=size, alloc_size=alloc_size,
                                      file_attrs=FILE_ATTR_ARCHIVE).to_bytes())

        # Add $DATA attribute
        if size == 0:
            record.add_attribute(DataAttribute(b"", is_resident=True).to_bytes())
        elif len(data_runs) > 0:
            record.add_attribute(DataAttribute(is_resident=False, data_runs=data_runs,
                                               real_size=size, alloc_size=alloc_size).to_bytes())
        else:
            # Small file - resident data (would need actual data)
            record.add_attribute(DataAttribute(b"\x00" * size, is_resident=True).to_bytes())

        self.records[record_num] = record
        self.path_to_record[rel_path] = record_num
        self.record_to_path[record_num] = rel_path

        return record_num

    def add_directory(self, rel_path: str, ctime: float, mtime: float, atime: float) -> int:
        """Add a directory and return its MFT record number."""
        if rel_path == "." or rel_path == "":
            return MFT_RECORD_ROOT

        record_num = self.next_record
        self.next_record += 1

        import os.path
        parent_path = os.path.dirname(rel_path)
        if not parent_path or parent_path == ".":
            parent_record = MFT_RECORD_ROOT
        else:
            parent_record = self.path_to_record.get(parent_path, MFT_RECORD_ROOT)

        dirname = os.path.basename(rel_path)

        record = MFTRecord(record_num, is_directory=True)
        record.add_attribute(StandardInformation(ctime, mtime, atime, FILE_ATTR_DIRECTORY).to_bytes())
        record.add_attribute(FileName(parent_record, 1, dirname, ctime, mtime, atime,
                                      file_attrs=FILE_ATTR_DIRECTORY).to_bytes())
        record.add_attribute(IndexRoot([]).to_bytes())  # Empty for now

        self.records[record_num] = record
        self.path_to_record[rel_path] = record_num
        self.record_to_path[record_num] = rel_path

        return record_num

    def get_record_bytes(self, record_num: int) -> Optional[bytes]:
        """Get serialized MFT record."""
        record = self.records.get(record_num)
        if record:
            return record.to_bytes()
        return None

    def get_mft_size(self) -> int:
        """Get total size of MFT in bytes."""
        return len(self.records) * MFT_RECORD_SIZE

    def get_mft_clusters(self) -> int:
        """Get number of clusters needed for MFT."""
        size = self.get_mft_size()
        return (size + CLUSTER_SIZE - 1) // CLUSTER_SIZE

    def _has_data_attribute(self, record: MFTRecord) -> bool:
        """Check if a record already has a $DATA attribute."""
        import struct
        for attr in record.attributes:
            if len(attr) >= 4:
                attr_type = struct.unpack('<I', attr[0:4])[0]
                if attr_type == 0x80:  # ATTR_DATA
                    return True
        return False

    def finalize(self):
        """
        Finalize the MFT after all records have been added.
        Adds $DATA attributes to system files that don't already have them.
        """
        from .data_runs import make_data_runs

        mft_size = self.get_mft_size()
        mft_clusters = self.get_mft_clusters()
        mft_alloc = mft_clusters * CLUSTER_SIZE

        # Add $DATA to $MFT (record 0) - points to the entire MFT
        mft_record = self.records[MFT_RECORD_MFT]
        if not self._has_data_attribute(mft_record):
            mft_data_runs = make_data_runs(self.mft_start_cluster, mft_clusters)
            mft_record.add_attribute(DataAttribute(
                is_resident=False,
                data_runs=mft_data_runs,
                real_size=mft_size,
                alloc_size=mft_alloc
            ).to_bytes())

        # Add $BITMAP to $MFT (record 0) - tracks which MFT records are in use
        from .attributes import BitmapAttribute
        num_records = len(self.records)
        # Create bitmap: 1 bit per MFT record, bits set for records in use
        bitmap_bytes = (num_records + 7) // 8
        mft_bitmap = bytearray(bitmap_bytes)
        for record_num in self.records.keys():
            if record_num < num_records * 8:
                byte_idx = record_num // 8
                bit_idx = record_num % 8
                if byte_idx < len(mft_bitmap):
                    mft_bitmap[byte_idx] |= (1 << bit_idx)
        mft_record.add_attribute(BitmapAttribute(
            data=bytes(mft_bitmap),
            is_resident=True
        ).to_bytes())

        # Add $DATA to other system files only if they don't have one yet
        # Note: MFT_RECORD_MFTMIRR is set up by synthesizer with proper data runs
        # Note: MFT_RECORD_SECURE has $SDS, $SDH, $SII set up in _create_system_records
        system_records = [
            MFT_RECORD_LOGFILE, MFT_RECORD_ATTRDEF,
            MFT_RECORD_BITMAP, MFT_RECORD_BOOT, MFT_RECORD_BADCLUS,
            MFT_RECORD_UPCASE
        ]

        for record_num in system_records:
            record = self.records.get(record_num)
            if record and not self._has_data_attribute(record):
                record.add_attribute(DataAttribute(
                    data=b"",
                    is_resident=True
                ).to_bytes())
