"""MBR partition table generation for NTFS volumes."""

import struct
from .constants import SECTOR_SIZE


class MBRPartitionTable:
    """
    Generates an MBR partition table with a single NTFS partition.

    Disk layout:
    - Sector 0: MBR with partition table
    - Sector 1 to partition_start-1: Empty/reserved
    - Sector partition_start onwards: NTFS volume
    """

    # Partition type for NTFS
    PARTITION_TYPE_NTFS = 0x07

    def __init__(self, total_disk_size: int, partition_start_sector: int = 2048):
        """
        Initialize MBR generator.

        Args:
            total_disk_size: Total disk size in bytes
            partition_start_sector: First sector of NTFS partition (default 2048 = 1MB offset)
        """
        self.total_disk_size = total_disk_size
        self.total_sectors = total_disk_size // SECTOR_SIZE
        self.partition_start = partition_start_sector
        self.partition_sectors = self.total_sectors - partition_start_sector

        # Calculate CHS values (for compatibility, use LBA values primarily)
        # Standard values for modern drives
        self.heads = 255
        self.sectors_per_track = 63

    def generate_mbr(self) -> bytes:
        """Generate 512-byte MBR sector."""
        mbr = bytearray(SECTOR_SIZE)

        # Boot code area (bytes 0-445) - leave as zeros (not bootable)

        # Partition table starts at offset 446
        # Each entry is 16 bytes, 4 entries total

        # Partition 1: NTFS
        entry1 = self._create_partition_entry(
            bootable=False,
            partition_type=self.PARTITION_TYPE_NTFS,
            start_lba=self.partition_start,
            sector_count=self.partition_sectors
        )
        mbr[446:462] = entry1

        # Partitions 2-4: Empty
        # Already zeros

        # MBR signature
        mbr[510] = 0x55
        mbr[511] = 0xAA

        return bytes(mbr)

    def _create_partition_entry(self, bootable: bool, partition_type: int,
                                  start_lba: int, sector_count: int) -> bytes:
        """Create a 16-byte partition table entry."""
        entry = bytearray(16)

        # Boot indicator
        entry[0] = 0x80 if bootable else 0x00

        # Starting CHS (use LBA-to-CHS conversion or max values)
        start_chs = self._lba_to_chs(start_lba)
        entry[1] = start_chs[0]  # Head
        entry[2] = start_chs[1]  # Sector (bits 0-5) + Cylinder high (bits 6-7)
        entry[3] = start_chs[2]  # Cylinder low

        # Partition type
        entry[4] = partition_type

        # Ending CHS
        end_lba = start_lba + sector_count - 1
        end_chs = self._lba_to_chs(end_lba)
        entry[5] = end_chs[0]  # Head
        entry[6] = end_chs[1]  # Sector + Cylinder high
        entry[7] = end_chs[2]  # Cylinder low

        # Starting LBA (4 bytes, little-endian)
        struct.pack_into('<I', entry, 8, start_lba)

        # Sector count (4 bytes, little-endian)
        struct.pack_into('<I', entry, 12, sector_count)

        return bytes(entry)

    def _lba_to_chs(self, lba: int) -> tuple:
        """
        Convert LBA to CHS.

        For large drives, CHS values max out at 1023/254/63.
        """
        if lba >= 1024 * 255 * 63:
            # Max CHS values
            return (254, 0xFF, 0xFF)  # Head=254, Sector=63+Cyl_high, Cyl_low=255

        cylinder = lba // (self.heads * self.sectors_per_track)
        temp = lba % (self.heads * self.sectors_per_track)
        head = temp // self.sectors_per_track
        sector = (temp % self.sectors_per_track) + 1

        if cylinder > 1023:
            cylinder = 1023

        # Pack into CHS format
        # Byte 0: Head
        # Byte 1: Sector (bits 0-5) | Cylinder high bits (bits 6-7)
        # Byte 2: Cylinder low bits
        chs_head = head & 0xFF
        chs_sector = (sector & 0x3F) | ((cylinder >> 2) & 0xC0)
        chs_cylinder = cylinder & 0xFF

        return (chs_head, chs_sector, chs_cylinder)

    def get_partition_offset(self) -> int:
        """Get byte offset of NTFS partition."""
        return self.partition_start * SECTOR_SIZE

    def get_partition_size(self) -> int:
        """Get size of NTFS partition in bytes."""
        return self.partition_sectors * SECTOR_SIZE


class PartitionedDisk:
    """
    Wraps an NTFS synthesizer to add MBR partition table.
    """

    def __init__(self, source_dir: str, min_size: int = 10 * 1024 * 1024):
        """
        Initialize partitioned disk.

        Args:
            source_dir: Source directory for NTFS content
            min_size: Minimum disk size in bytes (default 10MB)
        """
        from .synthesizer import NTFSSynthesizer

        # Calculate total disk size first
        partition_start_sector = 2048  # 1MB offset (standard for modern disks)
        partition_offset = partition_start_sector * SECTOR_SIZE

        # Ensure minimum size
        total_size = max(min_size, partition_offset + 1024 * 1024)

        # Round up to MB boundary
        total_size = ((total_size + 1024 * 1024 - 1) // (1024 * 1024)) * (1024 * 1024)

        self.total_size = total_size
        partition_size = total_size - partition_offset

        # Now create NTFS synthesizer with the correct partition size
        self.ntfs = NTFSSynthesizer(source_dir, volume_size=partition_size,
                                     hidden_sectors=partition_start_sector)

        self.mbr = MBRPartitionTable(total_size, partition_start_sector)
        self._mbr_bytes = self.mbr.generate_mbr()

        print(f"Partitioned disk created:")
        print(f"  Total size: {total_size / (1024*1024):.1f} MB")
        print(f"  Partition offset: {partition_offset} bytes (sector {partition_start_sector})")
        print(f"  NTFS partition size: {partition_size} bytes")

    def read(self, offset: int, length: int) -> bytes:
        """Read bytes from the partitioned disk."""
        result = bytearray(length)
        pos = 0

        partition_offset = self.mbr.get_partition_offset()

        while pos < length:
            byte_offset = offset + pos
            remaining = length - pos

            if byte_offset < SECTOR_SIZE:
                # MBR sector
                mbr_offset = byte_offset
                chunk_len = min(remaining, SECTOR_SIZE - mbr_offset)
                result[pos:pos + chunk_len] = self._mbr_bytes[mbr_offset:mbr_offset + chunk_len]
                pos += chunk_len

            elif byte_offset < partition_offset:
                # Gap between MBR and partition (zeros)
                chunk_len = min(remaining, partition_offset - byte_offset)
                pos += chunk_len  # Already zeros

            else:
                # NTFS partition
                ntfs_offset = byte_offset - partition_offset
                ntfs_remaining = self.ntfs.get_size() - ntfs_offset
                if ntfs_remaining <= 0:
                    # Past end of NTFS volume
                    pos += remaining
                else:
                    chunk_len = min(remaining, ntfs_remaining)
                    chunk = self.ntfs.read(ntfs_offset, chunk_len)
                    result[pos:pos + len(chunk)] = chunk
                    pos += len(chunk)

        return bytes(result)

    def get_size(self) -> int:
        """Get total disk size in bytes."""
        return self.total_size
