"""NTFS boot sector generation."""

import struct
import random
from .constants import SECTOR_SIZE, CLUSTER_SIZE, SECTORS_PER_CLUSTER, MFT_RECORD_SIZE


class BootSector:
    """Generates NTFS boot sector (sector 0)."""

    def __init__(self, total_size: int, mft_cluster: int, mft_mirror_cluster: int,
                 hidden_sectors: int = 0):
        """
        Initialize boot sector generator.

        Args:
            total_size: Total volume size in bytes
            mft_cluster: Starting cluster of $MFT
            mft_mirror_cluster: Starting cluster of $MFTMirr
            hidden_sectors: Number of sectors before this volume (partition offset)
        """
        self.total_size = total_size
        self.total_sectors = total_size // SECTOR_SIZE
        self.mft_cluster = mft_cluster
        self.mft_mirror_cluster = mft_mirror_cluster
        self.hidden_sectors = hidden_sectors
        self.volume_serial = random.randint(0, 0xFFFFFFFFFFFFFFFF)

    def generate(self) -> bytes:
        """Generate 512-byte NTFS boot sector."""
        boot = bytearray(SECTOR_SIZE)

        # Jump instruction (3 bytes)
        boot[0:3] = b'\xEB\x52\x90'

        # OEM ID (8 bytes)
        boot[3:11] = b'NTFS    '

        # BIOS Parameter Block (BPB)
        # Bytes per sector (2 bytes) - offset 0x0B
        struct.pack_into('<H', boot, 0x0B, SECTOR_SIZE)

        # Sectors per cluster (1 byte) - offset 0x0D
        boot[0x0D] = SECTORS_PER_CLUSTER

        # Reserved sectors (2 bytes) - offset 0x0E - must be 0 for NTFS
        struct.pack_into('<H', boot, 0x0E, 0)

        # Always 0 for NTFS (3 bytes) - offset 0x10
        boot[0x10:0x13] = b'\x00\x00\x00'

        # Unused (2 bytes) - offset 0x13
        struct.pack_into('<H', boot, 0x13, 0)

        # Media descriptor (1 byte) - offset 0x15 - 0xF8 for hard disk
        boot[0x15] = 0xF8

        # Always 0 for NTFS (2 bytes) - offset 0x16
        struct.pack_into('<H', boot, 0x16, 0)

        # Sectors per track (2 bytes) - offset 0x18
        struct.pack_into('<H', boot, 0x18, 63)

        # Number of heads (2 bytes) - offset 0x1A
        struct.pack_into('<H', boot, 0x1A, 255)

        # Hidden sectors (4 bytes) - offset 0x1C - sectors before this volume
        struct.pack_into('<I', boot, 0x1C, self.hidden_sectors)

        # Unused (4 bytes) - offset 0x20
        struct.pack_into('<I', boot, 0x20, 0)

        # Usually 0x80 (4 bytes) - offset 0x24
        struct.pack_into('<I', boot, 0x24, 0x00800080)

        # Total sectors (8 bytes) - offset 0x28
        struct.pack_into('<Q', boot, 0x28, self.total_sectors - 1)

        # $MFT cluster number (8 bytes) - offset 0x30
        struct.pack_into('<Q', boot, 0x30, self.mft_cluster)

        # $MFTMirr cluster number (8 bytes) - offset 0x38
        struct.pack_into('<Q', boot, 0x38, self.mft_mirror_cluster)

        # Clusters per MFT record (1 byte signed) - offset 0x40
        # Negative value means 2^|value| bytes per record
        # For 1024 bytes: -10 (2^10 = 1024)
        # But if positive, it's clusters per record
        # Since MFT_RECORD_SIZE (1024) < CLUSTER_SIZE (4096), we use negative encoding
        mft_record_clusters = MFT_RECORD_SIZE // CLUSTER_SIZE
        if mft_record_clusters >= 1:
            boot[0x40] = mft_record_clusters
        else:
            # Calculate log2 of MFT_RECORD_SIZE
            import math
            boot[0x40] = (256 - int(math.log2(MFT_RECORD_SIZE))) & 0xFF  # -10 for 1024

        # Unused (3 bytes) - offset 0x41
        boot[0x41:0x44] = b'\x00\x00\x00'

        # Clusters per index block (1 byte signed) - offset 0x44
        boot[0x44] = 1  # 1 cluster per index block (4096 bytes)

        # Unused (3 bytes) - offset 0x45
        boot[0x45:0x48] = b'\x00\x00\x00'

        # Volume serial number (8 bytes) - offset 0x48
        struct.pack_into('<Q', boot, 0x48, self.volume_serial)

        # Checksum (4 bytes) - offset 0x50 - unused, set to 0
        struct.pack_into('<I', boot, 0x50, 0)

        # Bootstrap code would go here (bytes 0x54 to 0x1FD)
        # We leave it as zeros since we're not bootable

        # End of sector signature (2 bytes) - offset 0x1FE
        boot[0x1FE:0x200] = b'\x55\xAA'

        return bytes(boot)

    def get_volume_serial(self) -> int:
        """Return the volume serial number."""
        return self.volume_serial
