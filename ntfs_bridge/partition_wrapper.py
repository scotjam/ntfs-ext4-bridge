"""Partition table wrapper for NTFS-ext4 bridge.

Wraps a raw NTFS filesystem with an MBR partition table so Windows
sees it as a properly partitioned disk. This is required because
Windows expects disks to have partition tables, not raw filesystems.

The wrapper:
1. Synthesizes a virtual MBR at sector 0 with one NTFS partition
2. Offsets all I/O by the partition start offset
3. Returns the virtual MBR for reads to sector 0
"""
import struct
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .cluster_mapper import ClusterMapper

# Standard sector size
SECTOR_SIZE = 512

# Partition offset: 1 MiB (modern alignment for SSDs)
# This is 2048 sectors of 512 bytes each
PARTITION_OFFSET_SECTORS = 2048
PARTITION_OFFSET_BYTES = PARTITION_OFFSET_SECTORS * SECTOR_SIZE  # 1048576 bytes


def log(msg):
    print(f"[Partition] {msg}", flush=True)


class PartitionWrapper:
    """Wraps a ClusterMapper with an MBR partition table.

    Presents the raw NTFS filesystem as a partitioned disk by:
    - Adding a virtual MBR at sector 0
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

        # Build the virtual MBR
        self.mbr = self._build_mbr()

        # Pre-calculate sectors for the gap between MBR and partition
        # This is all zeros (empty space before partition)
        self.gap_size = PARTITION_OFFSET_BYTES - SECTOR_SIZE

        # Virtual size for NBD advertisement (may be larger than real size
        # to accommodate virtual clusters with high cluster numbers)
        self.advertised_size = self.total_size

        log(f"Initialized: partition={self.partition_size} bytes, "
            f"total={self.total_size} bytes, "
            f"offset={PARTITION_OFFSET_BYTES} bytes")

    def _build_mbr(self) -> bytes:
        """Build a valid MBR with one NTFS partition entry."""
        mbr = bytearray(512)

        # Boot code area (first 446 bytes) - leave as zeros
        # A real bootloader would go here, but we don't need one

        # Partition table starts at offset 446
        # Each entry is 16 bytes, there are 4 entries (446-509)
        # Entry format:
        #   0: Boot flag (0x80 = active, 0x00 = inactive)
        #   1-3: CHS start (we'll use LBA, so set to 0xFE, 0xFF, 0xFF for large disks)
        #   4: Partition type (0x07 = NTFS)
        #   5-7: CHS end (same as start for large disks)
        #   8-11: LBA start (little-endian)
        #   12-15: LBA size in sectors (little-endian)

        # Calculate partition size in sectors
        partition_sectors = self.partition_size // SECTOR_SIZE

        # First partition entry at offset 446
        entry = bytearray(16)
        entry[0] = 0x80  # Active/bootable partition

        # CHS start - use max values to indicate LBA mode
        entry[1] = 0xFE  # Head
        entry[2] = 0xFF  # Sector (bits 0-5) + Cylinder high (bits 6-7)
        entry[3] = 0xFF  # Cylinder low

        # Partition type: 0x07 = NTFS/exFAT/HPFS
        entry[4] = 0x07

        # CHS end - same as start for large disks
        entry[5] = 0xFE
        entry[6] = 0xFF
        entry[7] = 0xFF

        # LBA start (partition offset in sectors)
        struct.pack_into('<I', entry, 8, PARTITION_OFFSET_SECTORS)

        # LBA size (partition size in sectors)
        struct.pack_into('<I', entry, 12, partition_sectors)

        # Copy entry to MBR
        mbr[446:462] = entry

        # Other 3 partition entries remain zeros (unused)

        # MBR signature at offset 510
        mbr[510] = 0x55
        mbr[511] = 0xAA

        log(f"Built MBR: partition starts at sector {PARTITION_OFFSET_SECTORS}, "
            f"size {partition_sectors} sectors")

        return bytes(mbr)

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
        """Read data, handling MBR and partition offset.

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

            if current_offset < SECTOR_SIZE:
                # Reading from MBR
                mbr_start = current_offset
                mbr_bytes = min(remaining, SECTOR_SIZE - mbr_start)
                result[pos:pos + mbr_bytes] = self.mbr[mbr_start:mbr_start + mbr_bytes]
                pos += mbr_bytes

            elif current_offset < PARTITION_OFFSET_BYTES:
                # Reading from gap between MBR and partition (zeros)
                gap_start = current_offset - SECTOR_SIZE
                gap_bytes = min(remaining, PARTITION_OFFSET_BYTES - current_offset)
                # result is already zeros from bytearray initialization
                pos += gap_bytes

            else:
                # Reading from partition - offset and pass to mapper
                partition_offset = current_offset - PARTITION_OFFSET_BYTES

                # Always pass to mapper - it handles virtual clusters beyond partition size
                # The mapper will return zeros for truly out-of-range reads
                data = self.mapper.read(partition_offset, remaining)
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

    def rescan_mft(self):
        """Pass through MFT rescan to underlying mapper."""
        self.mapper.rescan_mft()
