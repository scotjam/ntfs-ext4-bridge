"""NTFS filesystem synthesizer - generates NTFS from a directory on-the-fly."""

import os
import struct
from typing import Dict, List, Tuple, Optional
from .constants import (
    SECTOR_SIZE, CLUSTER_SIZE, MFT_RECORD_SIZE,
    MFT_RECORD_ROOT, MFT_RECORD_USER_START,
    MFT_RECORD_UPCASE, MFT_RECORD_BITMAP, MFT_RECORD_ATTRDEF,
    MFT_RECORD_MFT, MFT_RECORD_MFTMIRR
)
from .boot_sector import BootSector
from .mft import MFTGenerator
from .data_runs import make_data_runs
from .system_files import generate_upcase_table, generate_bitmap, generate_attrdef


class ClusterMap:
    """Maps NTFS clusters to source file regions."""

    def __init__(self, source_dir: str, data_start_cluster: int):
        self.source_dir = source_dir
        self.data_start_cluster = data_start_cluster
        self.next_cluster = data_start_cluster

        # cluster_num -> (file_path, offset_in_file)
        self.cluster_to_file: Dict[int, Tuple[str, int]] = {}

        # file_path -> (start_cluster, cluster_count)
        self.file_to_clusters: Dict[str, Tuple[int, int]] = {}

    def allocate_file(self, rel_path: str, size: int) -> Tuple[int, int, bytes]:
        """
        Allocate clusters for a file.

        Returns:
            (start_cluster, cluster_count, data_runs_bytes)
        """
        if size == 0:
            return (0, 0, b'\x00')

        cluster_count = (size + CLUSTER_SIZE - 1) // CLUSTER_SIZE
        start_cluster = self.next_cluster

        # Map clusters to file
        full_path = os.path.join(self.source_dir, rel_path)
        for i in range(cluster_count):
            cluster = start_cluster + i
            offset = i * CLUSTER_SIZE
            self.cluster_to_file[cluster] = (full_path, offset)

        self.file_to_clusters[rel_path] = (start_cluster, cluster_count)
        self.next_cluster += cluster_count

        # Generate data runs (contiguous allocation)
        data_runs = make_data_runs(start_cluster, cluster_count)

        return (start_cluster, cluster_count, data_runs)

    def allocate_clusters(self, count: int) -> Tuple[int, bytes]:
        """Allocate clusters without mapping to a file. Returns (start_cluster, data_runs)."""
        start_cluster = self.next_cluster
        self.next_cluster += count
        data_runs = make_data_runs(start_cluster, count)
        return (start_cluster, data_runs)

    def read_cluster(self, cluster: int) -> bytes:
        """Read a cluster's data from the source file."""
        if cluster not in self.cluster_to_file:
            return b'\x00' * CLUSTER_SIZE

        file_path, offset = self.cluster_to_file[cluster]

        try:
            with open(file_path, 'rb') as f:
                f.seek(offset)
                data = f.read(CLUSTER_SIZE)
                # Pad to cluster size if needed
                if len(data) < CLUSTER_SIZE:
                    data = data + b'\x00' * (CLUSTER_SIZE - len(data))
                return data
        except (IOError, OSError):
            return b'\x00' * CLUSTER_SIZE

    def get_total_clusters(self) -> int:
        """Get total number of allocated data clusters."""
        return self.next_cluster - self.data_start_cluster


class NTFSSynthesizer:
    """Synthesizes an NTFS filesystem from a source directory."""

    def __init__(self, source_dir: str, volume_size: int = None, hidden_sectors: int = 0):
        """
        Initialize the synthesizer.

        Args:
            source_dir: Path to source directory (will be presented as NTFS root)
            volume_size: Total volume size in bytes (auto-calculated if None)
        """
        self.source_dir = os.path.abspath(source_dir)
        self.hidden_sectors = hidden_sectors

        # Verify source exists
        if not os.path.isdir(self.source_dir):
            raise ValueError(f"Source directory does not exist: {self.source_dir}")

        # Calculate sizes
        self.total_data_size = self._calculate_data_size()

        # Layout:
        # Cluster 0: Boot sector (in first sector)
        # Clusters 1-7: Reserved
        # Cluster 8+: MFT
        # After MFT: System file data ($UpCase, $Bitmap, etc.)
        # After system files: User data

        self.mft_start_cluster = 8

        # Initialize MFT generator
        self.mft = MFTGenerator(self.mft_start_cluster)

        # Scan directory and add entries to MFT
        self._scan_directory()

        # Calculate data start (after MFT)
        mft_clusters = self.mft.get_mft_clusters()
        self.data_start_cluster = self.mft_start_cluster + mft_clusters + 10  # Some padding

        # Initialize cluster map
        self.cluster_map = ClusterMap(self.source_dir, self.data_start_cluster)

        # Generate and allocate system files first
        self._setup_system_files()

        # Allocate clusters for user files
        self._allocate_file_clusters()

        # Calculate MFT mirror position - place it right after all data
        self.mft_mirror_cluster = self.cluster_map.next_cluster + 1

        # Calculate total volume size first (needed for bitmap)
        total_clusters = self.mft_mirror_cluster + 10  # Include mirror + padding
        if volume_size:
            self.volume_size = volume_size
        else:
            self.volume_size = total_clusters * CLUSTER_SIZE

        # Now setup bitmap with all allocation info
        self._setup_bitmap()

        # Update $MFTMirr $DATA to point to the mirror location
        self._setup_mftmirr()

        # Now finalize MFT (add $DATA to $MFT record pointing to MFT location)
        self.mft.finalize()

        # Initialize boot sector
        self.boot_sector = BootSector(
            self.volume_size,
            self.mft_start_cluster,
            self.mft_mirror_cluster,
            self.hidden_sectors
        )

        # Cache generated boot sector
        self._boot_sector_bytes = self.boot_sector.generate()

        print(f"NTFS Synthesizer initialized:")
        print(f"  Source: {self.source_dir}")
        print(f"  Volume size: {self.volume_size / (1024*1024):.1f} MB")
        print(f"  MFT records: {len(self.mft.records)}")
        print(f"  Data clusters: {self.cluster_map.get_total_clusters()}")
        print(f"  $UpCase at cluster: {self.upcase_cluster}")
        print(f"  MFT mirror at cluster: {self.mft_mirror_cluster}")

    def _calculate_data_size(self) -> int:
        """Calculate total size of files in source directory."""
        total = 0
        for root, dirs, files in os.walk(self.source_dir):
            for name in files:
                try:
                    path = os.path.join(root, name)
                    total += os.path.getsize(path)
                except OSError:
                    pass
        return total

    def _setup_system_files(self):
        """Generate and allocate system file data."""
        from .attributes import DataAttribute
        from .constants import FILE_ATTR_HIDDEN, FILE_ATTR_SYSTEM

        # Generate $UpCase table (128KB = 32 clusters)
        self.upcase_data = generate_upcase_table()
        upcase_clusters = (len(self.upcase_data) + CLUSTER_SIZE - 1) // CLUSTER_SIZE
        self.upcase_cluster, upcase_runs = self.cluster_map.allocate_clusters(upcase_clusters)

        # Update $UpCase MFT record with proper $DATA
        upcase_record = self.mft.records[MFT_RECORD_UPCASE]
        # Remove the placeholder empty $DATA and add real one
        upcase_record.attributes = [a for a in upcase_record.attributes
                                    if struct.unpack('<I', a[0:4])[0] != 0x80]
        upcase_record.add_attribute(DataAttribute(
            is_resident=False,
            data_runs=upcase_runs,
            real_size=len(self.upcase_data),
            alloc_size=upcase_clusters * CLUSTER_SIZE
        ).to_bytes())

        # Generate $AttrDef
        self.attrdef_data = generate_attrdef()
        attrdef_clusters = (len(self.attrdef_data) + CLUSTER_SIZE - 1) // CLUSTER_SIZE
        self.attrdef_cluster, attrdef_runs = self.cluster_map.allocate_clusters(attrdef_clusters)

        # Update $AttrDef MFT record
        attrdef_record = self.mft.records[MFT_RECORD_ATTRDEF]
        attrdef_record.attributes = [a for a in attrdef_record.attributes
                                     if struct.unpack('<I', a[0:4])[0] != 0x80]
        attrdef_record.add_attribute(DataAttribute(
            is_resident=False,
            data_runs=attrdef_runs,
            real_size=len(self.attrdef_data),
            alloc_size=attrdef_clusters * CLUSTER_SIZE
        ).to_bytes())

        # Store cluster ranges for reading (bitmap added later after we know all allocations)
        self.system_file_ranges = {
            'upcase': (self.upcase_cluster, upcase_clusters, self.upcase_data),
            'attrdef': (self.attrdef_cluster, attrdef_clusters, self.attrdef_data),
        }

    def _setup_bitmap(self):
        """Generate and allocate $Bitmap after all other allocations are done."""
        from .attributes import DataAttribute

        # Calculate total clusters in volume
        total_clusters = self.volume_size // CLUSTER_SIZE

        # Collect all allocated clusters
        allocated = []

        # Boot sector / reserved (clusters 0-7)
        for i in range(self.mft_start_cluster):
            allocated.append(i)

        # MFT clusters
        mft_clusters = self.mft.get_mft_clusters()
        for i in range(self.mft_start_cluster, self.mft_start_cluster + mft_clusters):
            allocated.append(i)

        # System file clusters (upcase, attrdef)
        for name, (start, count, data) in self.system_file_ranges.items():
            for i in range(start, start + count):
                allocated.append(i)

        # User file clusters
        for cluster in self.cluster_map.cluster_to_file.keys():
            allocated.append(cluster)

        # MFT mirror
        allocated.append(self.mft_mirror_cluster)

        # Generate bitmap data
        self.bitmap_data = generate_bitmap(total_clusters, allocated)
        bitmap_clusters = (len(self.bitmap_data) + CLUSTER_SIZE - 1) // CLUSTER_SIZE

        # Allocate clusters for bitmap
        self.bitmap_cluster, bitmap_runs = self.cluster_map.allocate_clusters(bitmap_clusters)

        # Also mark bitmap clusters as allocated in the bitmap itself
        for i in range(bitmap_clusters):
            cluster = self.bitmap_cluster + i
            if cluster < total_clusters:
                byte_idx = cluster // 8
                bit_idx = cluster % 8
                if byte_idx < len(self.bitmap_data):
                    # Need to modify - convert to bytearray
                    self.bitmap_data = bytearray(self.bitmap_data)
                    self.bitmap_data[byte_idx] |= (1 << bit_idx)
                    self.bitmap_data = bytes(self.bitmap_data)

        # Update $Bitmap MFT record
        bitmap_record = self.mft.records[MFT_RECORD_BITMAP]
        bitmap_record.attributes = [a for a in bitmap_record.attributes
                                    if struct.unpack('<I', a[0:4])[0] != 0x80]
        bitmap_record.add_attribute(DataAttribute(
            is_resident=False,
            data_runs=bitmap_runs,
            real_size=len(self.bitmap_data),
            alloc_size=bitmap_clusters * CLUSTER_SIZE
        ).to_bytes())

        # Add to system file ranges
        self.system_file_ranges['bitmap'] = (self.bitmap_cluster, bitmap_clusters, self.bitmap_data)

    def _setup_mftmirr(self):
        """Setup $MFTMirr $DATA attribute to point to the mirror location."""
        from .attributes import DataAttribute
        from .constants import MFT_RECORD_MFTMIRR

        # MFT mirror contains first 4 MFT records = 4KB = 1 cluster
        mirror_size = 4 * MFT_RECORD_SIZE  # 4096 bytes
        mirror_clusters = 1  # Fits in 1 cluster

        # Create data runs pointing to the mirror cluster
        mirror_runs = make_data_runs(self.mft_mirror_cluster, mirror_clusters)

        # Update $MFTMirr record - remove old $DATA and add new one
        mftmirr_record = self.mft.records[MFT_RECORD_MFTMIRR]
        mftmirr_record.attributes = [a for a in mftmirr_record.attributes
                                     if struct.unpack('<I', a[0:4])[0] != 0x80]
        mftmirr_record.add_attribute(DataAttribute(
            is_resident=False,
            data_runs=mirror_runs,
            real_size=mirror_size,
            alloc_size=mirror_clusters * CLUSTER_SIZE
        ).to_bytes())

    def _scan_directory(self):
        """Scan source directory and create MFT entries."""
        for root, dirs, files in os.walk(self.source_dir):
            rel_root = os.path.relpath(root, self.source_dir)
            if rel_root == ".":
                rel_root = ""

            # Add subdirectories
            for dirname in dirs:
                if rel_root:
                    rel_path = os.path.join(rel_root, dirname)
                else:
                    rel_path = dirname

                full_path = os.path.join(root, dirname)
                try:
                    stat = os.stat(full_path)
                    self.mft.add_directory(
                        rel_path,
                        ctime=stat.st_ctime,
                        mtime=stat.st_mtime,
                        atime=stat.st_atime
                    )
                except OSError:
                    pass

            # Add files (without data allocation yet)
            for filename in files:
                if rel_root:
                    rel_path = os.path.join(rel_root, filename)
                else:
                    rel_path = filename

                full_path = os.path.join(root, filename)
                try:
                    stat = os.stat(full_path)
                    # We'll update with real data runs after allocation
                    self.mft.add_file(
                        rel_path,
                        ctime=stat.st_ctime,
                        mtime=stat.st_mtime,
                        atime=stat.st_atime,
                        size=stat.st_size,
                        data_runs=b'\x00',  # Placeholder
                        alloc_size=0
                    )
                except OSError:
                    pass

    def _allocate_file_clusters(self):
        """Allocate clusters for all files and update MFT records."""
        from .attributes import StandardInformation, FileName, DataAttribute
        from .constants import FILE_ATTR_ARCHIVE

        for rel_path, record_num in list(self.mft.path_to_record.items()):
            if record_num < MFT_RECORD_USER_START:
                continue  # Skip system records

            record = self.mft.records.get(record_num)
            if not record or record.is_directory:
                continue

            full_path = os.path.join(self.source_dir, rel_path)
            try:
                stat = os.stat(full_path)
                size = stat.st_size

                if size > 0:
                    start_cluster, cluster_count, data_runs = self.cluster_map.allocate_file(
                        rel_path, size
                    )
                    alloc_size = cluster_count * CLUSTER_SIZE

                    # Rebuild MFT record with correct data runs
                    record.attributes.clear()
                    record.add_attribute(StandardInformation(
                        stat.st_ctime, stat.st_mtime, stat.st_atime, FILE_ATTR_ARCHIVE
                    ).to_bytes())

                    parent_record = MFT_RECORD_ROOT
                    parent_path = os.path.dirname(rel_path)
                    if parent_path:
                        parent_record = self.mft.path_to_record.get(parent_path, MFT_RECORD_ROOT)

                    record.add_attribute(FileName(
                        parent_record, 1, os.path.basename(rel_path),
                        stat.st_ctime, stat.st_mtime, stat.st_atime,
                        file_size=size, alloc_size=alloc_size, file_attrs=FILE_ATTR_ARCHIVE
                    ).to_bytes())

                    record.add_attribute(DataAttribute(
                        is_resident=False,
                        data_runs=data_runs,
                        real_size=size,
                        alloc_size=alloc_size
                    ).to_bytes())

            except OSError:
                pass

    def read(self, offset: int, length: int) -> bytes:
        """
        Read bytes from the synthesized NTFS volume.

        Args:
            offset: Byte offset from start of volume
            length: Number of bytes to read

        Returns:
            Requested bytes
        """
        result = bytearray(length)
        pos = 0

        while pos < length:
            byte_offset = offset + pos
            remaining = length - pos

            # Determine which region this offset falls into
            cluster = byte_offset // CLUSTER_SIZE
            cluster_offset = byte_offset % CLUSTER_SIZE

            if byte_offset < SECTOR_SIZE:
                # Boot sector
                boot_offset = byte_offset
                chunk_len = min(remaining, SECTOR_SIZE - boot_offset)
                result[pos:pos + chunk_len] = self._boot_sector_bytes[boot_offset:boot_offset + chunk_len]
                pos += chunk_len

            elif cluster < self.mft_start_cluster:
                # Reserved area - zeros
                chunk_len = min(remaining, (self.mft_start_cluster * CLUSTER_SIZE) - byte_offset)
                pos += chunk_len  # Already zeros in bytearray

            elif cluster < self.data_start_cluster:
                # MFT region
                chunk = self._read_mft_region(byte_offset, remaining)
                result[pos:pos + len(chunk)] = chunk
                pos += len(chunk)

            elif cluster >= self.mft_mirror_cluster and cluster < self.mft_mirror_cluster + 1:
                # MFT Mirror region - contains first 4 MFT records
                chunk = self._read_mft_mirror(cluster, cluster_offset, remaining)
                result[pos:pos + len(chunk)] = chunk
                pos += len(chunk)

            else:
                # Data region (includes system files and user files)
                chunk = self._read_data_region(cluster, cluster_offset, remaining)
                result[pos:pos + len(chunk)] = chunk
                pos += len(chunk)

        return bytes(result)

    def _read_mft_region(self, byte_offset: int, max_length: int) -> bytes:
        """Read from MFT region."""
        mft_start_byte = self.mft_start_cluster * CLUSTER_SIZE
        mft_offset = byte_offset - mft_start_byte

        record_num = mft_offset // MFT_RECORD_SIZE
        record_offset = mft_offset % MFT_RECORD_SIZE

        record_bytes = self.mft.get_record_bytes(record_num)
        if record_bytes is None:
            # Empty/unused record area
            record_bytes = b'\x00' * MFT_RECORD_SIZE

        chunk_len = min(max_length, MFT_RECORD_SIZE - record_offset)
        return record_bytes[record_offset:record_offset + chunk_len]

    def _read_mft_mirror(self, cluster: int, cluster_offset: int, max_length: int) -> bytes:
        """Read from MFT mirror region (first 4 MFT records)."""
        # MFT mirror contains copies of records 0-3
        mirror_data = bytearray(CLUSTER_SIZE)

        for i in range(4):
            record_bytes = self.mft.get_record_bytes(i)
            if record_bytes:
                start = i * MFT_RECORD_SIZE
                mirror_data[start:start + MFT_RECORD_SIZE] = record_bytes

        chunk_len = min(max_length, CLUSTER_SIZE - cluster_offset)
        return bytes(mirror_data[cluster_offset:cluster_offset + chunk_len])

    def _read_data_region(self, cluster: int, cluster_offset: int, max_length: int) -> bytes:
        """Read from data region (system files or user files)."""
        # Check if this is a system file cluster
        for name, (start_cluster, count, data) in self.system_file_ranges.items():
            if start_cluster <= cluster < start_cluster + count:
                # This is a system file cluster
                data_offset = (cluster - start_cluster) * CLUSTER_SIZE + cluster_offset
                chunk_len = min(max_length, CLUSTER_SIZE - cluster_offset)
                if data_offset + chunk_len <= len(data):
                    return data[data_offset:data_offset + chunk_len]
                else:
                    # Pad with zeros if past end of data
                    available = max(0, len(data) - data_offset)
                    result = data[data_offset:data_offset + available] if available > 0 else b''
                    result += b'\x00' * (chunk_len - len(result))
                    return result

        # Regular user file cluster
        cluster_data = self.cluster_map.read_cluster(cluster)
        chunk_len = min(max_length, CLUSTER_SIZE - cluster_offset)
        return cluster_data[cluster_offset:cluster_offset + chunk_len]

    def get_size(self) -> int:
        """Get total volume size in bytes."""
        return self.volume_size
