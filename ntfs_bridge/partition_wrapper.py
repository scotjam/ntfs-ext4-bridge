"""Partition table wrapper for NTFS-ext4 bridge.

Wraps a raw NTFS filesystem with a partition table so Windows
sees it as a properly partitioned disk. This is required because
Windows expects disks to have partition tables, not raw filesystems.

Uses MBR for volumes <=2TB, GPT for volumes >2TB.

The wrapper:
1. Synthesizes a virtual partition table at the start of the disk
2. Offsets all I/O by the partition start offset
3. Returns the virtual partition table for reads to the header area
"""
import struct
import uuid
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .cluster_mapper import ClusterMapper

# Standard sector size
SECTOR_SIZE = 512

# Partition offset: 1 MiB (modern alignment for SSDs)
# This is 2048 sectors of 512 bytes each
PARTITION_OFFSET_SECTORS = 2048
PARTITION_OFFSET_BYTES = PARTITION_OFFSET_SECTORS * SECTOR_SIZE  # 1048576 bytes

# MBR max: 2TB (32-bit sector count * 512)
MBR_MAX_BYTES = 0xFFFFFFFF * SECTOR_SIZE

# GPT constants
GPT_HEADER_SIGNATURE = b'EFI PART'
GPT_REVISION = 0x00010000
GPT_HEADER_SIZE = 92
GPT_ENTRY_SIZE = 128

# Microsoft Basic Data GUID: EBD0A0A2-B9E5-4433-87C0-68B6B72699C7
BASIC_DATA_GUID = uuid.UUID('EBD0A0A2-B9E5-4433-87C0-68B6B72699C7')


def _uuid_to_mixed_endian(u: uuid.UUID) -> bytes:
    """Convert UUID to GPT mixed-endian format.
    GPT stores UUIDs with the first three components in little-endian."""
    b = u.bytes
    return (b[3::-1] + b[5:3:-1] + b[7:5:-1] + b[8:])


def _crc32(data: bytes) -> int:
    """Compute CRC32 as unsigned 32-bit value."""
    import binascii
    return binascii.crc32(data) & 0xFFFFFFFF


def log(msg):
    print(f"[Partition] {msg}", flush=True)


class PartitionWrapper:
    """Wraps a ClusterMapper with a partition table (MBR or GPT).

    Presents the raw NTFS filesystem as a partitioned disk by:
    - Adding a virtual partition table at sector 0+
    - Offsetting all partition I/O by PARTITION_OFFSET_BYTES
    - Reporting total size as partition size + offset
    """

    def __init__(self, mapper: 'ClusterMapper'):
        """Initialize the partition wrapper.

        Args:
            mapper: The ClusterMapper containing the raw NTFS filesystem
        """
        self.mapper = mapper
        self.partition_size = mapper.get_size()
        self.total_size = self.partition_size + PARTITION_OFFSET_BYTES

        # Decide MBR vs GPT based on size
        self.use_gpt = self.partition_size > MBR_MAX_BYTES

        if self.use_gpt:
            self._build_gpt()
        else:
            self.header_area = self._build_mbr()
            self.header_size = SECTOR_SIZE
            self.trailer_area = b''
            self.trailer_offset = 0

        # Virtual size for NBD advertisement
        self.advertised_size = self.total_size

        log(f"Initialized: partition={self.partition_size} bytes, "
            f"total={self.total_size} bytes, "
            f"offset={PARTITION_OFFSET_BYTES} bytes")

    def _build_mbr(self) -> bytes:
        """Build a valid MBR with one NTFS partition entry."""
        mbr = bytearray(512)

        partition_sectors = self.partition_size // SECTOR_SIZE

        # First partition entry at offset 446
        entry = bytearray(16)
        entry[0] = 0x80  # Active/bootable partition
        entry[1] = 0xFE; entry[2] = 0xFF; entry[3] = 0xFF  # CHS start
        entry[4] = 0x07  # NTFS
        entry[5] = 0xFE; entry[6] = 0xFF; entry[7] = 0xFF  # CHS end
        struct.pack_into('<I', entry, 8, PARTITION_OFFSET_SECTORS)
        # MBR partition table uses 32-bit LBA sector counts; cap at 0xFFFFFFFF
        # for volumes > 2TB.  Windows will use the NTFS boot sector's actual
        # cluster count rather than the MBR entry for the true volume size.
        struct.pack_into('<I', entry, 12, min(partition_sectors, 0xFFFFFFFF))
        mbr[446:462] = entry

        mbr[510] = 0x55
        mbr[511] = 0xAA

        log(f"Built MBR: partition starts at sector {PARTITION_OFFSET_SECTORS}, "
            f"size {partition_sectors} sectors")

        return bytes(mbr)

    def _build_gpt(self):
        """Build GPT partition table (protective MBR + GPT header + entries).

        GPT layout:
        - Sector 0: Protective MBR
        - Sector 1: Primary GPT header
        - Sectors 2-33: Primary partition entries (128 entries * 128 bytes = 32 sectors)
        - Sector 2048+: Partition data (1 MiB offset)
        - Last 33 sectors: Backup GPT entries + header
        """
        total_sectors = self.total_size // SECTOR_SIZE
        partition_sectors = self.partition_size // SECTOR_SIZE
        partition_start_lba = PARTITION_OFFSET_SECTORS
        partition_end_lba = partition_start_lba + partition_sectors - 1

        # Add space for backup GPT (33 sectors at the end)
        backup_sectors = 33
        self.total_size += backup_sectors * SECTOR_SIZE
        total_sectors = self.total_size // SECTOR_SIZE

        last_usable_lba = total_sectors - backup_sectors - 1
        first_usable_lba = 34  # After primary GPT entries

        # Generate stable GUIDs from partition size (deterministic)
        disk_guid = uuid.uuid5(uuid.NAMESPACE_DNS, f"ntfs-bridge-disk-{self.partition_size}")
        part_guid = uuid.uuid5(uuid.NAMESPACE_DNS, f"ntfs-bridge-part-{self.partition_size}")

        # Build partition entry (one NTFS partition)
        entry = bytearray(GPT_ENTRY_SIZE)
        entry[0:16] = _uuid_to_mixed_endian(BASIC_DATA_GUID)
        entry[16:32] = _uuid_to_mixed_endian(part_guid)
        struct.pack_into('<Q', entry, 32, partition_start_lba)   # First LBA
        struct.pack_into('<Q', entry, 40, partition_end_lba)     # Last LBA
        struct.pack_into('<Q', entry, 48, 0)                     # Attributes
        # Partition name: "NTFS Bridge" in UTF-16LE
        name = "NTFS Bridge".encode('utf-16-le')
        entry[56:56 + len(name)] = name

        # Build entries area (128 entries, each 128 bytes = 32 sectors)
        entries_area = bytearray(128 * GPT_ENTRY_SIZE)
        entries_area[0:GPT_ENTRY_SIZE] = entry
        entries_crc = _crc32(bytes(entries_area))

        # Build primary GPT header (sector 1)
        primary_header = bytearray(SECTOR_SIZE)
        primary_header[0:8] = GPT_HEADER_SIGNATURE
        struct.pack_into('<I', primary_header, 8, GPT_REVISION)
        struct.pack_into('<I', primary_header, 12, GPT_HEADER_SIZE)
        # CRC32 at offset 16 - fill in after
        struct.pack_into('<I', primary_header, 16, 0)  # placeholder
        struct.pack_into('<I', primary_header, 20, 0)  # reserved
        struct.pack_into('<Q', primary_header, 24, 1)  # My LBA
        struct.pack_into('<Q', primary_header, 32, total_sectors - 1)  # Alternate LBA
        struct.pack_into('<Q', primary_header, 40, first_usable_lba)
        struct.pack_into('<Q', primary_header, 48, last_usable_lba)
        primary_header[56:72] = _uuid_to_mixed_endian(disk_guid)
        struct.pack_into('<Q', primary_header, 72, 2)  # Partition entries start LBA
        struct.pack_into('<I', primary_header, 80, 128)  # Number of entries
        struct.pack_into('<I', primary_header, 84, GPT_ENTRY_SIZE)
        struct.pack_into('<I', primary_header, 88, entries_crc)
        # Now compute header CRC
        header_crc = _crc32(bytes(primary_header[:GPT_HEADER_SIZE]))
        struct.pack_into('<I', primary_header, 16, header_crc)

        # Build protective MBR
        pmbr = bytearray(512)
        pmbr_entry = bytearray(16)
        pmbr_entry[4] = 0xEE  # GPT protective
        pmbr_entry[1] = 0x00; pmbr_entry[2] = 0x02; pmbr_entry[3] = 0x00  # CHS start
        pmbr_entry[5] = 0xFE; pmbr_entry[6] = 0xFF; pmbr_entry[7] = 0xFF  # CHS end
        struct.pack_into('<I', pmbr_entry, 8, 1)  # LBA start
        mbr_sectors = min(total_sectors - 1, 0xFFFFFFFF)
        struct.pack_into('<I', pmbr_entry, 12, mbr_sectors)
        pmbr[446:462] = pmbr_entry
        pmbr[510] = 0x55
        pmbr[511] = 0xAA

        # Assemble header area: protective MBR + primary GPT header + entries
        # Pad entries to fill up to PARTITION_OFFSET_BYTES
        header_data = bytearray(PARTITION_OFFSET_BYTES)
        header_data[0:512] = pmbr
        header_data[512:1024] = primary_header
        header_data[1024:1024 + len(entries_area)] = entries_area
        self.header_area = bytes(header_data)
        self.header_size = PARTITION_OFFSET_BYTES

        # Build backup GPT (entries + header) at end of disk
        backup_header = bytearray(SECTOR_SIZE)
        backup_header[0:8] = GPT_HEADER_SIGNATURE
        struct.pack_into('<I', backup_header, 8, GPT_REVISION)
        struct.pack_into('<I', backup_header, 12, GPT_HEADER_SIZE)
        struct.pack_into('<I', backup_header, 16, 0)  # placeholder
        struct.pack_into('<I', backup_header, 20, 0)
        struct.pack_into('<Q', backup_header, 24, total_sectors - 1)  # My LBA (last sector)
        struct.pack_into('<Q', backup_header, 32, 1)  # Alternate LBA (primary)
        struct.pack_into('<Q', backup_header, 40, first_usable_lba)
        struct.pack_into('<Q', backup_header, 48, last_usable_lba)
        backup_header[56:72] = _uuid_to_mixed_endian(disk_guid)
        struct.pack_into('<Q', backup_header, 72, total_sectors - 33)  # Backup entries start
        struct.pack_into('<I', backup_header, 80, 128)
        struct.pack_into('<I', backup_header, 84, GPT_ENTRY_SIZE)
        struct.pack_into('<I', backup_header, 88, entries_crc)
        backup_crc = _crc32(bytes(backup_header[:GPT_HEADER_SIZE]))
        struct.pack_into('<I', backup_header, 16, backup_crc)

        # Backup trailer: 32 sectors of entries + 1 sector header
        trailer = bytearray(backup_sectors * SECTOR_SIZE)
        trailer[0:len(entries_area)] = entries_area
        trailer[32 * SECTOR_SIZE:33 * SECTOR_SIZE] = backup_header
        self.trailer_area = bytes(trailer)
        self.trailer_offset = self.total_size - backup_sectors * SECTOR_SIZE

        log(f"Built GPT: partition starts at sector {partition_start_lba}, "
            f"size {partition_sectors} sectors, total {total_sectors} sectors")

    def get_size(self) -> int:
        """Get advertised disk size for NBD."""
        return self.advertised_size

    def set_virtual_size(self, max_cluster: int, cluster_size: int):
        """Set advertised size to accommodate virtual clusters.

        Args:
            max_cluster: Highest virtual cluster number that may be accessed
            cluster_size: Cluster size in bytes
        """
        needed_size = (max_cluster + 1) * cluster_size + PARTITION_OFFSET_BYTES
        if needed_size > self.advertised_size:
            self.advertised_size = needed_size
            log(f"Advertised size increased to {needed_size} bytes for virtual clusters")

    @property
    def cluster_size(self) -> int:
        """Pass through cluster size from underlying mapper."""
        return self.mapper.cluster_size

    def read(self, offset: int, length: int) -> bytes:
        """Read data, handling partition table and partition offset.

        Args:
            offset: Byte offset from start of disk
            length: Number of bytes to read

        Returns:
            bytes: Data read from the appropriate location
        """
        result = bytearray(length)
        pos = 0

        while pos < length:
            current_offset = offset + pos
            remaining = length - pos

            if current_offset < PARTITION_OFFSET_BYTES:
                # Reading from header area (MBR/GPT + gap)
                hdr_bytes = min(remaining, PARTITION_OFFSET_BYTES - current_offset)
                if current_offset < len(self.header_area):
                    src_end = min(current_offset + hdr_bytes, len(self.header_area))
                    copy_len = src_end - current_offset
                    result[pos:pos + copy_len] = self.header_area[current_offset:src_end]
                # Rest stays zeros (gap)
                pos += hdr_bytes

            elif self.trailer_area and current_offset >= self.trailer_offset:
                # Reading from GPT backup trailer
                trailer_pos = current_offset - self.trailer_offset
                if trailer_pos < len(self.trailer_area):
                    copy_len = min(remaining, len(self.trailer_area) - trailer_pos)
                    result[pos:pos + copy_len] = self.trailer_area[trailer_pos:trailer_pos + copy_len]
                    pos += copy_len
                else:
                    pos += remaining

            else:
                # Reading from partition - offset and pass to mapper.
                # Clamp so a read that starts in the partition body doesn't
                # run past it into the backup-GPT trailer (which would then
                # be served as zeros from the mapper instead of trailer_area).
                partition_offset = current_offset - PARTITION_OFFSET_BYTES
                part_read = remaining
                if self.trailer_area and current_offset < self.trailer_offset:
                    part_read = min(remaining, self.trailer_offset - current_offset)
                data = self.mapper.read(partition_offset, part_read)
                if not data or len(data) == 0:
                    break  # Prevent infinite loop
                result[pos:pos + len(data)] = data
                pos += len(data)

        return bytes(result)

    def write(self, offset: int, data: bytes):
        """Write data, handling MBR and partition offset.

        Args:
            offset: Byte offset from start of disk
            data: Data to write
        """
        length = len(data)
        pos = 0

        while pos < length:
            current_offset = offset + pos
            remaining = length - pos

            if current_offset < PARTITION_OFFSET_BYTES:
                # Writing to MBR or gap - ignore (read-only area)
                # Windows may try to update the MBR, but we don't persist it
                skip_bytes = min(remaining, PARTITION_OFFSET_BYTES - current_offset)
                pos += skip_bytes

            else:
                # Writing to partition - offset and pass to mapper
                partition_offset = current_offset - PARTITION_OFFSET_BYTES
                partition_remaining = self.partition_size - partition_offset

                if partition_remaining <= 0:
                    # Beyond partition end - ignore
                    pos += remaining
                else:
                    write_bytes = min(remaining, partition_remaining)
                    self.mapper.write(partition_offset, data[pos:pos + write_bytes])
                    pos += write_bytes

    def flush(self):
        """Flush any pending writes to the underlying mapper."""
        self.mapper.flush()

    def flush_all(self):
        """Pass through full-image flush to the underlying mapper."""
        self.mapper.flush_all()

    def durability_barrier(self):
        """Pass through the NBD FLUSH durability barrier to the mapper."""
        self.mapper.durability_barrier()

    def clear_dirty_bit(self):
        """Pass through the volume dirty-bit clear to the mapper."""
        self.mapper.clear_dirty_bit()

    def rescan_mft(self):
        """Pass through MFT rescan to underlying mapper."""
        self.mapper.rescan_mft()
