"""Virtual file overlay for live ext4→NTFS sync.

Synthesizes NTFS MFT records and directory entries on-the-fly for files
that exist in ext4 but not yet in the NTFS image. This allows Windows
to see new ext4 files without actually writing to the NTFS image.

Key concepts:
- Virtual MFT records: Synthesized FILE records for ext4 files
- Virtual directory entries: Injected into $INDEX_ROOT/$INDEX_ALLOCATION
- Virtual clusters: Mapped to ext4 file content
- No actual NTFS modification = no corruption risk
"""

import os
import struct
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .cluster_mapper import ClusterMapper

MFT_RECORD_SIZE = 1024
CLUSTER_SIZE = 4096


def log(msg):
    print(f"[VirtualFiles] {msg}", flush=True)


@dataclass
class VirtualFile:
    """Represents a virtual file (exists in ext4, not in NTFS image)."""
    rel_path: str
    source_path: str
    file_size: int
    is_directory: bool
    mft_record_num: int
    parent_mft_record: int
    # Virtual cluster allocation (for non-resident files)
    virtual_clusters: List[int] = field(default_factory=list)
    # Creation time (Windows FILETIME format)
    creation_time: int = 0

    def __post_init__(self):
        # Set creation time to now if not set
        if self.creation_time == 0:
            # Windows FILETIME: 100-nanosecond intervals since 1601-01-01
            # Unix epoch is 11644473600 seconds after Windows epoch
            unix_time = time.time()
            self.creation_time = int((unix_time + 11644473600) * 10000000)


@dataclass
class VirtualDirectory:
    """Represents a virtual directory."""
    rel_path: str
    source_path: str
    mft_record_num: int
    parent_mft_record: int
    children: Set[int] = field(default_factory=set)  # MFT record numbers
    creation_time: int = 0

    def __post_init__(self):
        if self.creation_time == 0:
            unix_time = time.time()
            self.creation_time = int((unix_time + 11644473600) * 10000000)


class VirtualFileManager:
    """Manages virtual file overlay for ext4→NTFS visibility.

    When a new file appears in ext4, it's added as a "virtual" file.
    The ClusterMapper consults this manager during reads to inject
    virtual MFT records and directory entries.
    """

    # Start virtual MFT records just after the expected real files.
    # System files use 0-23, user files start at 24. For small volumes, the MFT
    # may only have ~66 records allocated. We start at 30 to leave room for a few
    # real files while staying within the allocated MFT range.
    VIRTUAL_MFT_START = 30

    # Start virtual clusters at a high number
    VIRTUAL_CLUSTER_START = 500000

    def __init__(self, source_dir: str, cluster_size: int = CLUSTER_SIZE):
        self.source_dir = os.path.abspath(source_dir)
        self.cluster_size = cluster_size

        # Virtual files: rel_path -> VirtualFile
        self.virtual_files: Dict[str, VirtualFile] = {}

        # Virtual directories: rel_path -> VirtualDirectory
        self.virtual_dirs: Dict[str, VirtualDirectory] = {}

        # MFT record number -> VirtualFile or VirtualDirectory
        self.mft_to_virtual: Dict[int, VirtualFile | VirtualDirectory] = {}

        # Virtual cluster -> (source_path, offset)
        self.virtual_cluster_map: Dict[int, Tuple[str, int]] = {}

        # Next available virtual MFT record number
        self._next_mft_record = self.VIRTUAL_MFT_START

        # Next available virtual cluster
        self._next_cluster = self.VIRTUAL_CLUSTER_START

        # Thread safety
        self.lock = threading.RLock()

        # Reference to ClusterMapper (set after construction)
        self.mapper: Optional['ClusterMapper'] = None

    def set_mapper(self, mapper: 'ClusterMapper'):
        """Set reference to ClusterMapper for directory lookups."""
        self.mapper = mapper

    def add_file(self, rel_path: str) -> Optional[VirtualFile]:
        """Add a virtual file for an ext4 file.

        Returns the VirtualFile if added, None if file doesn't exist or
        is already tracked (virtual or real).
        """
        source_path = os.path.join(self.source_dir, rel_path)

        if not os.path.isfile(source_path):
            return None

        with self.lock:
            # Already tracked as virtual?
            if rel_path in self.virtual_files:
                return self.virtual_files[rel_path]

            # Already tracked as real file in ClusterMapper?
            if self.mapper and rel_path in self.mapper.path_to_mft_record:
                return None

            # Get file size
            try:
                file_size = os.path.getsize(source_path)
            except OSError:
                return None

            # Find parent directory's MFT record
            parent_mft = self._get_parent_mft_record(rel_path)

            # Allocate virtual MFT record
            mft_record = self._next_mft_record
            self._next_mft_record += 1

            # Allocate virtual clusters for non-resident files
            virtual_clusters = []
            if file_size > 700:  # Non-resident threshold
                num_clusters = (file_size + self.cluster_size - 1) // self.cluster_size
                for i in range(num_clusters):
                    cluster = self._next_cluster
                    self._next_cluster += 1
                    virtual_clusters.append(cluster)
                    self.virtual_cluster_map[cluster] = (source_path, i * self.cluster_size)

            vf = VirtualFile(
                rel_path=rel_path,
                source_path=source_path,
                file_size=file_size,
                is_directory=False,
                mft_record_num=mft_record,
                parent_mft_record=parent_mft,
                virtual_clusters=virtual_clusters,
            )

            self.virtual_files[rel_path] = vf
            self.mft_to_virtual[mft_record] = vf

            # Add to parent's children
            self._add_to_parent(vf)

            log(f"Added virtual file: {rel_path} (MFT {mft_record}, {file_size} bytes)")
            return vf

    def add_directory(self, rel_path: str) -> Optional[VirtualDirectory]:
        """Add a virtual directory."""
        source_path = os.path.join(self.source_dir, rel_path)

        if not os.path.isdir(source_path):
            return None

        with self.lock:
            if rel_path in self.virtual_dirs:
                return self.virtual_dirs[rel_path]

            # Already in ClusterMapper?
            if self.mapper and rel_path in self.mapper.path_to_mft_record:
                return None

            parent_mft = self._get_parent_mft_record(rel_path)

            mft_record = self._next_mft_record
            self._next_mft_record += 1

            vd = VirtualDirectory(
                rel_path=rel_path,
                source_path=source_path,
                mft_record_num=mft_record,
                parent_mft_record=parent_mft,
            )

            self.virtual_dirs[rel_path] = vd
            self.mft_to_virtual[mft_record] = vd

            # Add to parent's children
            self._add_to_parent(vd)

            log(f"Added virtual directory: {rel_path} (MFT {mft_record})")
            return vd

    def remove_file(self, rel_path: str):
        """Remove a virtual file (e.g., when it's been deleted from ext4)."""
        with self.lock:
            vf = self.virtual_files.pop(rel_path, None)
            if vf:
                self.mft_to_virtual.pop(vf.mft_record_num, None)
                for cluster in vf.virtual_clusters:
                    self.virtual_cluster_map.pop(cluster, None)
                self._remove_from_parent(vf)
                log(f"Removed virtual file: {rel_path}")

    def remove_directory(self, rel_path: str):
        """Remove a virtual directory."""
        with self.lock:
            vd = self.virtual_dirs.pop(rel_path, None)
            if vd:
                self.mft_to_virtual.pop(vd.mft_record_num, None)
                self._remove_from_parent(vd)
                log(f"Removed virtual directory: {rel_path}")

    def promote_to_real(self, rel_path: str):
        """Called when a virtual file becomes real (written to NTFS image).

        This happens when Windows creates the file through normal NTFS
        operations and we detect it in MFT tracking.
        """
        with self.lock:
            if rel_path in self.virtual_files:
                self.remove_file(rel_path)
                log(f"Promoted to real: {rel_path}")
            elif rel_path in self.virtual_dirs:
                self.remove_directory(rel_path)
                log(f"Promoted dir to real: {rel_path}")

    def _get_parent_mft_record(self, rel_path: str) -> int:
        """Get MFT record number for parent directory."""
        parent_path = os.path.dirname(rel_path)

        if not parent_path or parent_path == '.':
            return 5  # Root directory is always MFT record 5

        # Check if parent is a virtual directory
        if parent_path in self.virtual_dirs:
            return self.virtual_dirs[parent_path].mft_record_num

        # Check ClusterMapper for real directory
        if self.mapper and parent_path in self.mapper.path_to_mft_record:
            return self.mapper.path_to_mft_record[parent_path]

        # Parent doesn't exist - need to create virtual parent first
        # This handles nested new directories
        parent_source = os.path.join(self.source_dir, parent_path)
        if os.path.isdir(parent_source):
            vd = self.add_directory(parent_path)
            if vd:
                return vd.mft_record_num

        return 5  # Fallback to root

    def _add_to_parent(self, item: VirtualFile | VirtualDirectory):
        """Add item to its parent directory's children set."""
        parent_mft = item.parent_mft_record

        # Is parent a virtual directory?
        parent = self.mft_to_virtual.get(parent_mft)
        if isinstance(parent, VirtualDirectory):
            parent.children.add(item.mft_record_num)

    def _remove_from_parent(self, item: VirtualFile | VirtualDirectory):
        """Remove item from its parent directory's children set."""
        parent_mft = item.parent_mft_record
        parent = self.mft_to_virtual.get(parent_mft)
        if isinstance(parent, VirtualDirectory):
            parent.children.discard(item.mft_record_num)

    # =========================================================================
    # MFT record synthesis
    # =========================================================================

    def get_virtual_mft_record(self, record_num: int) -> Optional[bytes]:
        """Get synthesized MFT record for a virtual file/directory.

        Returns None if record_num is not a virtual record.
        """
        with self.lock:
            item = self.mft_to_virtual.get(record_num)
            if not item:
                return None

            if isinstance(item, VirtualFile):
                return self._synthesize_file_record(item)
            else:
                return self._synthesize_dir_record(item)

    def _synthesize_file_record(self, vf: VirtualFile) -> bytes:
        """Synthesize an MFT FILE record for a virtual file."""
        record = bytearray(MFT_RECORD_SIZE)

        # FILE signature
        record[0:4] = b'FILE'

        # Update sequence offset and count
        struct.pack_into('<H', record, 4, 48)   # USA offset
        struct.pack_into('<H', record, 6, 3)    # USA count (1 + 2 sectors)

        # $LogFile sequence number
        struct.pack_into('<Q', record, 8, 0)

        # Sequence number
        struct.pack_into('<H', record, 16, 1)

        # Link count
        struct.pack_into('<H', record, 18, 1)

        # First attribute offset (after USA)
        first_attr_off = 56
        struct.pack_into('<H', record, 20, first_attr_off)

        # Flags: in-use (0x01)
        struct.pack_into('<H', record, 22, 0x01)

        # Used size of record (filled in later)
        # Allocated size
        struct.pack_into('<I', record, 28, MFT_RECORD_SIZE)

        # Base record (0 for base record)
        struct.pack_into('<Q', record, 32, 0)

        # Next attribute ID
        struct.pack_into('<H', record, 40, 4)

        # Record number (for NTFS 3.1+)
        struct.pack_into('<I', record, 44, vf.mft_record_num)

        # Update sequence array placeholder
        record[48:50] = b'\x00\x00'  # USA value
        record[50:52] = b'\x00\x00'  # Sector 1 original
        record[52:54] = b'\x00\x00'  # Sector 2 original

        off = first_attr_off

        # $STANDARD_INFORMATION (0x10)
        off = self._add_standard_info(record, off, vf.creation_time)

        # $FILE_NAME (0x30)
        off = self._add_file_name(record, off, vf.rel_path, vf.parent_mft_record,
                                   vf.file_size, vf.creation_time, is_dir=False)

        # $DATA (0x80)
        if vf.file_size <= 700 and not vf.virtual_clusters:
            # Resident data
            off = self._add_resident_data(record, off, vf.source_path, vf.file_size)
        else:
            # Non-resident data
            off = self._add_nonresident_data(record, off, vf.virtual_clusters,
                                              vf.file_size, self.cluster_size)

        # End marker
        struct.pack_into('<I', record, off, 0xFFFFFFFF)
        off += 4

        # Update used size
        struct.pack_into('<I', record, 24, off)

        # Apply fixups
        self._apply_fixups(record)

        return bytes(record)

    def _synthesize_dir_record(self, vd: VirtualDirectory) -> bytes:
        """Synthesize an MFT FILE record for a virtual directory."""
        record = bytearray(MFT_RECORD_SIZE)

        # FILE signature
        record[0:4] = b'FILE'

        # Update sequence offset and count
        struct.pack_into('<H', record, 4, 48)
        struct.pack_into('<H', record, 6, 3)

        # Sequence number
        struct.pack_into('<H', record, 16, 1)

        # Link count
        struct.pack_into('<H', record, 18, 1)

        # First attribute offset
        first_attr_off = 56
        struct.pack_into('<H', record, 20, first_attr_off)

        # Flags: in-use (0x01) + directory (0x02)
        struct.pack_into('<H', record, 22, 0x03)

        # Allocated size
        struct.pack_into('<I', record, 28, MFT_RECORD_SIZE)

        # Next attribute ID
        struct.pack_into('<H', record, 40, 4)

        # Record number
        struct.pack_into('<I', record, 44, vd.mft_record_num)

        # USA placeholder
        record[48:54] = b'\x00' * 6

        off = first_attr_off

        # $STANDARD_INFORMATION
        off = self._add_standard_info(record, off, vd.creation_time)

        # $FILE_NAME
        off = self._add_file_name(record, off, vd.rel_path, vd.parent_mft_record,
                                   0, vd.creation_time, is_dir=True)

        # $INDEX_ROOT with children
        children = self.get_virtual_children(vd.mft_record_num)
        off = self._add_index_root_with_children(record, off, children)

        # End marker
        struct.pack_into('<I', record, off, 0xFFFFFFFF)
        off += 4

        # Update used size
        struct.pack_into('<I', record, 24, off)

        # Apply fixups
        self._apply_fixups(record)

        return bytes(record)

    def _add_standard_info(self, record: bytearray, off: int, creation_time: int) -> int:
        """Add $STANDARD_INFORMATION attribute (NTFS 3.0+ extended version)."""
        # NTFS 3.0+ STANDARD_INFORMATION has 72 bytes:
        # 0-7: Creation time
        # 8-15: Modified time
        # 16-23: MFT modified time
        # 24-31: Access time
        # 32-35: Flags
        # 36-39: Max versions
        # 40-43: Version
        # 44-47: Class ID
        # 48-51: Owner ID
        # 52-55: Security ID (index into $Secure)
        # 56-63: Quota charged
        # 64-71: USN

        # Attribute header (resident)
        struct.pack_into('<I', record, off, 0x10)      # Type
        struct.pack_into('<I', record, off + 4, 96)    # Length (24 header + 72 data = 96)
        record[off + 8] = 0                             # Non-resident flag (resident)
        record[off + 9] = 0                             # Name length
        struct.pack_into('<H', record, off + 10, 0)    # Name offset
        struct.pack_into('<H', record, off + 12, 0)    # Flags
        struct.pack_into('<H', record, off + 14, 0)    # Instance
        struct.pack_into('<I', record, off + 16, 72)   # Value length (extended)
        struct.pack_into('<H', record, off + 20, 24)   # Value offset

        # $STANDARD_INFORMATION data (72 bytes for NTFS 3.0+)
        data_off = off + 24
        struct.pack_into('<Q', record, data_off, creation_time)       # Creation time
        struct.pack_into('<Q', record, data_off + 8, creation_time)   # Modified time
        struct.pack_into('<Q', record, data_off + 16, creation_time)  # MFT modified
        struct.pack_into('<Q', record, data_off + 24, creation_time)  # Access time
        struct.pack_into('<I', record, data_off + 32, 0x20)           # Flags (ARCHIVE)
        struct.pack_into('<I', record, data_off + 36, 0)              # Max versions
        struct.pack_into('<I', record, data_off + 40, 0)              # Version
        struct.pack_into('<I', record, data_off + 44, 0)              # Class ID
        struct.pack_into('<I', record, data_off + 48, 0)              # Owner ID
        struct.pack_into('<I', record, data_off + 52, 0x100)          # Security ID (256 = default)
        struct.pack_into('<Q', record, data_off + 56, 0)              # Quota charged
        struct.pack_into('<Q', record, data_off + 64, 0)              # USN

        return off + 96

    def _add_file_name(self, record: bytearray, off: int, rel_path: str,
                        parent_mft: int, file_size: int, creation_time: int,
                        is_dir: bool) -> int:
        """Add $FILE_NAME attribute."""
        filename = os.path.basename(rel_path)
        filename_bytes = filename.encode('utf-16-le')
        filename_len = len(filename)

        # $FILE_NAME is always resident
        # Data: 66 bytes header + filename
        data_len = 66 + len(filename_bytes)
        attr_len = 24 + data_len  # Header + data
        attr_len = (attr_len + 7) & ~7  # 8-byte align

        # Attribute header
        struct.pack_into('<I', record, off, 0x30)           # Type
        struct.pack_into('<I', record, off + 4, attr_len)   # Length
        record[off + 8] = 0                                  # Resident
        record[off + 9] = 0                                  # Name length
        struct.pack_into('<H', record, off + 10, 0)         # Name offset
        struct.pack_into('<H', record, off + 12, 0)         # Flags
        struct.pack_into('<H', record, off + 14, 1)         # Instance
        struct.pack_into('<I', record, off + 16, data_len)  # Value length
        struct.pack_into('<H', record, off + 20, 24)        # Value offset

        # $FILE_NAME data
        data_off = off + 24
        # Parent directory reference (48-bit record + 16-bit sequence)
        parent_ref = parent_mft | (1 << 48)  # Sequence 1
        struct.pack_into('<Q', record, data_off, parent_ref)

        # Timestamps
        struct.pack_into('<Q', record, data_off + 8, creation_time)   # Creation
        struct.pack_into('<Q', record, data_off + 16, creation_time)  # Modified
        struct.pack_into('<Q', record, data_off + 24, creation_time)  # MFT modified
        struct.pack_into('<Q', record, data_off + 32, creation_time)  # Access

        # Allocated size
        if is_dir:
            struct.pack_into('<Q', record, data_off + 40, 0)
            struct.pack_into('<Q', record, data_off + 48, 0)
        else:
            alloc_size = ((file_size + self.cluster_size - 1) // self.cluster_size) * self.cluster_size
            struct.pack_into('<Q', record, data_off + 40, alloc_size)  # Allocated
            struct.pack_into('<Q', record, data_off + 48, file_size)   # Real size

        # Flags
        flags = 0x10000000 if is_dir else 0x20  # Directory or Archive
        struct.pack_into('<I', record, data_off + 56, flags)

        # Reparse value (0)
        struct.pack_into('<I', record, data_off + 60, 0)

        # Filename length and namespace
        record[data_off + 64] = filename_len
        record[data_off + 65] = 3  # WIN32_AND_DOS namespace

        # Filename
        record[data_off + 66:data_off + 66 + len(filename_bytes)] = filename_bytes

        return off + attr_len

    def _add_resident_data(self, record: bytearray, off: int,
                            source_path: str, file_size: int) -> int:
        """Add resident $DATA attribute with actual file content."""
        # Read file content
        try:
            with open(source_path, 'rb') as f:
                content = f.read(min(file_size, 700))
        except OSError:
            content = b''

        data_len = len(content)
        attr_len = 24 + data_len
        attr_len = (attr_len + 7) & ~7

        # Attribute header
        struct.pack_into('<I', record, off, 0x80)           # Type ($DATA)
        struct.pack_into('<I', record, off + 4, attr_len)   # Length
        record[off + 8] = 0                                  # Resident
        record[off + 9] = 0                                  # Name length
        struct.pack_into('<H', record, off + 10, 0)         # Name offset
        struct.pack_into('<H', record, off + 12, 0)         # Flags
        struct.pack_into('<H', record, off + 14, 2)         # Instance
        struct.pack_into('<I', record, off + 16, data_len)  # Value length
        struct.pack_into('<H', record, off + 20, 24)        # Value offset

        # Data content
        record[off + 24:off + 24 + data_len] = content

        return off + attr_len

    def _add_nonresident_data(self, record: bytearray, off: int,
                               clusters: List[int], file_size: int,
                               cluster_size: int) -> int:
        """Add non-resident $DATA attribute with data runs."""
        from .data_runs import compress_cluster_list, encode_data_runs

        if not clusters:
            # Empty file
            runs_bytes = b'\x00'
        else:
            runs = compress_cluster_list(clusters)
            runs_bytes = encode_data_runs(runs)

        total_clusters = len(clusters)
        alloc_size = total_clusters * cluster_size

        # Non-resident header is 64 bytes + runs
        attr_len = 64 + len(runs_bytes)
        attr_len = (attr_len + 7) & ~7

        # Attribute header
        struct.pack_into('<I', record, off, 0x80)           # Type ($DATA)
        struct.pack_into('<I', record, off + 4, attr_len)   # Length
        record[off + 8] = 1                                  # Non-resident
        record[off + 9] = 0                                  # Name length
        struct.pack_into('<H', record, off + 10, 0)         # Name offset
        struct.pack_into('<H', record, off + 12, 0)         # Flags
        struct.pack_into('<H', record, off + 14, 2)         # Instance

        # Non-resident specific fields
        struct.pack_into('<Q', record, off + 16, 0)                       # Start VCN
        struct.pack_into('<Q', record, off + 24, max(0, total_clusters - 1))  # End VCN
        struct.pack_into('<H', record, off + 32, 64)                      # Runs offset
        struct.pack_into('<H', record, off + 34, 0)                       # Compression unit
        struct.pack_into('<I', record, off + 36, 0)                       # Padding
        struct.pack_into('<Q', record, off + 40, alloc_size)              # Allocated size
        struct.pack_into('<Q', record, off + 48, file_size)               # Real size
        struct.pack_into('<Q', record, off + 56, file_size)               # Initialized size

        # Data runs
        record[off + 64:off + 64 + len(runs_bytes)] = runs_bytes

        return off + attr_len

    def _add_empty_index_root(self, record: bytearray, off: int) -> int:
        """Add empty $INDEX_ROOT attribute for directories."""
        # Index root header is complex, but for empty dir we need minimal structure
        # $INDEX_ROOT contains: index root header + index header + end entry

        # Index root header: 16 bytes
        # Index header: 16 bytes
        # End entry: 16 bytes
        data_len = 48
        attr_len = 24 + data_len
        attr_len = (attr_len + 7) & ~7

        # Attribute header
        struct.pack_into('<I', record, off, 0x90)           # Type ($INDEX_ROOT)
        struct.pack_into('<I', record, off + 4, attr_len)   # Length
        record[off + 8] = 0                                  # Resident
        record[off + 9] = 4                                  # Name length ($I30)
        struct.pack_into('<H', record, off + 10, 24)        # Name offset
        struct.pack_into('<H', record, off + 12, 0)         # Flags
        struct.pack_into('<H', record, off + 14, 3)         # Instance
        struct.pack_into('<I', record, off + 16, data_len)  # Value length
        struct.pack_into('<H', record, off + 20, 32)        # Value offset (after name)

        # Attribute name: $I30 (index on $FILE_NAME)
        record[off + 24:off + 32] = '$I30'.encode('utf-16-le')

        data_off = off + 32

        # Index root header
        struct.pack_into('<I', record, data_off, 0x30)      # Attribute type ($FILE_NAME)
        struct.pack_into('<I', record, data_off + 4, 1)     # Collation rule (filename)
        struct.pack_into('<I', record, data_off + 8, 4096)  # Index block size
        record[data_off + 12] = 1                            # Clusters per index block
        # 3 bytes padding

        # Index header (at data_off + 16)
        idx_off = data_off + 16
        struct.pack_into('<I', record, idx_off, 16)         # Entries offset
        struct.pack_into('<I', record, idx_off + 4, 32)     # Total size
        struct.pack_into('<I', record, idx_off + 8, 32)     # Allocated size
        record[idx_off + 12] = 0                             # Flags (small index, no $INDEX_ALLOCATION)
        # 3 bytes padding

        # End entry (at data_off + 32)
        end_off = data_off + 32
        struct.pack_into('<I', record, end_off + 8, 16)     # Entry length
        struct.pack_into('<H', record, end_off + 12, 0)     # Key length
        struct.pack_into('<H', record, end_off + 14, 2)     # Flags: LAST_ENTRY

        return off + attr_len

    def _add_index_root_with_children(self, record: bytearray, off: int,
                                       children: list) -> int:
        """Add $INDEX_ROOT attribute with child entries for virtual directories."""
        # Build all child entries first
        entries_data = bytearray()
        for child in sorted(children, key=lambda c: os.path.basename(c.rel_path).upper()):
            entry = self.synthesize_index_entry(child)
            entries_data.extend(entry)

        # Add end entry (16 bytes)
        end_entry = bytearray(16)
        struct.pack_into('<H', end_entry, 8, 16)    # Entry length
        struct.pack_into('<H', end_entry, 12, 0)    # Key length
        struct.pack_into('<H', end_entry, 14, 2)    # Flags: LAST_ENTRY
        entries_data.extend(end_entry)

        # Index root header: 16 bytes
        # Index header: 16 bytes
        # Entries: variable
        index_header_size = 16
        entries_offset = 16  # Relative to index header start
        total_entries_size = len(entries_data)
        index_data_size = index_header_size + total_entries_size

        # Total data: index root header (16) + index header (16) + entries
        data_len = 16 + index_data_size

        # Check if it fits in MFT record (leave room for other attrs)
        max_index_size = 400  # Conservative limit for inline index
        if data_len > max_index_size:
            # Too large - fall back to empty index (would need INDEX_ALLOCATION)
            return self._add_empty_index_root(record, off)

        attr_len = 24 + 8 + data_len  # header + name ($I30) + data
        attr_len = (attr_len + 7) & ~7

        # Attribute header
        struct.pack_into('<I', record, off, 0x90)           # Type ($INDEX_ROOT)
        struct.pack_into('<I', record, off + 4, attr_len)   # Length
        record[off + 8] = 0                                  # Resident
        record[off + 9] = 4                                  # Name length ($I30)
        struct.pack_into('<H', record, off + 10, 24)        # Name offset
        struct.pack_into('<H', record, off + 12, 0)         # Flags
        struct.pack_into('<H', record, off + 14, 3)         # Instance
        struct.pack_into('<I', record, off + 16, data_len)  # Value length
        struct.pack_into('<H', record, off + 20, 32)        # Value offset (after name)

        # Attribute name: $I30
        record[off + 24:off + 32] = '$I30'.encode('utf-16-le')

        data_off = off + 32

        # Index root header (16 bytes)
        struct.pack_into('<I', record, data_off, 0x30)      # Attribute type ($FILE_NAME)
        struct.pack_into('<I', record, data_off + 4, 1)     # Collation rule (filename)
        struct.pack_into('<I', record, data_off + 8, 4096)  # Index block size
        record[data_off + 12] = 1                            # Clusters per index block
        # 3 bytes padding

        # Index header (at data_off + 16)
        idx_off = data_off + 16
        struct.pack_into('<I', record, idx_off, entries_offset)          # Entries offset
        struct.pack_into('<I', record, idx_off + 4, index_data_size)     # Total size
        struct.pack_into('<I', record, idx_off + 8, index_data_size)     # Allocated size
        record[idx_off + 12] = 0                             # Flags (small index)
        # 3 bytes padding

        # Copy entries
        entries_start = data_off + 32
        record[entries_start:entries_start + len(entries_data)] = entries_data

        return off + attr_len

    def _apply_fixups(self, record: bytearray):
        """Apply NTFS fixups to MFT record."""
        usa_offset = struct.unpack('<H', record[4:6])[0]
        usa_count = struct.unpack('<H', record[6:8])[0]

        if usa_count < 2:
            return

        # Generate update sequence value
        seq_val = struct.pack('<H', 0x0001)
        record[usa_offset:usa_offset + 2] = seq_val

        # Save and replace sector end bytes
        for i in range(1, usa_count):
            sector_end = i * 512 - 2
            if sector_end + 2 <= len(record):
                # Save original bytes
                record[usa_offset + i * 2:usa_offset + i * 2 + 2] = record[sector_end:sector_end + 2]
                # Write sequence value
                record[sector_end:sector_end + 2] = seq_val

    # =========================================================================
    # Directory entry synthesis (for $INDEX_ROOT injection)
    # =========================================================================

    def get_virtual_children(self, parent_mft: int) -> List[VirtualFile | VirtualDirectory]:
        """Get all virtual files/dirs that are children of a directory.

        This is used to inject entries into directory index reads.
        """
        with self.lock:
            children = []

            for vf in self.virtual_files.values():
                if vf.parent_mft_record == parent_mft:
                    children.append(vf)

            for vd in self.virtual_dirs.values():
                if vd.parent_mft_record == parent_mft:
                    children.append(vd)

            return children

    def synthesize_index_entry(self, item: VirtualFile | VirtualDirectory) -> bytes:
        """Synthesize an NTFS index entry for a virtual file/directory.

        This entry can be injected into $INDEX_ROOT reads.
        """
        filename = os.path.basename(item.rel_path)
        filename_bytes = filename.encode('utf-16-le')
        filename_len = len(filename)

        # Index entry structure:
        # 0-7: File reference (MFT record + sequence)
        # 8-9: Entry length
        # 10-11: Key length
        # 12-13: Flags
        # 14-15: Padding
        # 16+: Key ($FILE_NAME attribute content)

        # $FILE_NAME content (same as in MFT attribute)
        key_len = 66 + len(filename_bytes)
        entry_len = 16 + key_len
        entry_len = (entry_len + 7) & ~7  # 8-byte align

        entry = bytearray(entry_len)

        # File reference
        file_ref = item.mft_record_num | (1 << 48)  # Sequence 1
        struct.pack_into('<Q', entry, 0, file_ref)

        # Entry length
        struct.pack_into('<H', entry, 8, entry_len)

        # Key length
        struct.pack_into('<H', entry, 10, key_len)

        # Flags (0 = has sub-node pointer if in B+ tree, but we don't use that)
        struct.pack_into('<H', entry, 12, 0)

        # Key: $FILE_NAME content
        key_off = 16

        # Parent reference
        parent_ref = item.parent_mft_record | (1 << 48)
        struct.pack_into('<Q', entry, key_off, parent_ref)

        # Timestamps
        if isinstance(item, VirtualFile):
            creation_time = item.creation_time
            file_size = item.file_size
            is_dir = False
        else:
            creation_time = item.creation_time
            file_size = 0
            is_dir = True

        struct.pack_into('<Q', entry, key_off + 8, creation_time)
        struct.pack_into('<Q', entry, key_off + 16, creation_time)
        struct.pack_into('<Q', entry, key_off + 24, creation_time)
        struct.pack_into('<Q', entry, key_off + 32, creation_time)

        # Sizes
        if is_dir:
            struct.pack_into('<Q', entry, key_off + 40, 0)
            struct.pack_into('<Q', entry, key_off + 48, 0)
        else:
            alloc_size = ((file_size + self.cluster_size - 1) // self.cluster_size) * self.cluster_size
            struct.pack_into('<Q', entry, key_off + 40, alloc_size)
            struct.pack_into('<Q', entry, key_off + 48, file_size)

        # Flags
        flags = 0x10000000 if is_dir else 0x20
        struct.pack_into('<I', entry, key_off + 56, flags)

        # Reparse
        struct.pack_into('<I', entry, key_off + 60, 0)

        # Filename
        entry[key_off + 64] = filename_len
        entry[key_off + 65] = 3  # WIN32_AND_DOS
        entry[key_off + 66:key_off + 66 + len(filename_bytes)] = filename_bytes

        return bytes(entry)

    # =========================================================================
    # Read interception
    # =========================================================================

    def read_virtual_cluster(self, cluster: int) -> Optional[bytes]:
        """Read data from a virtual cluster.

        Returns the cluster data if this is a virtual cluster, None otherwise.
        """
        with self.lock:
            mapping = self.virtual_cluster_map.get(cluster)
            if not mapping:
                return None

            source_path, offset = mapping
            try:
                with open(source_path, 'rb') as f:
                    f.seek(offset)
                    data = f.read(self.cluster_size)
                    if len(data) < self.cluster_size:
                        data += b'\x00' * (self.cluster_size - len(data))
                    return data
            except OSError:
                return b'\x00' * self.cluster_size

    def is_virtual_mft_region(self, offset: int, length: int, mft_offset: int) -> bool:
        """Check if a read might include virtual MFT records."""
        if not self.mft_to_virtual:
            return False

        # Calculate MFT record range being read
        rel_offset = offset - mft_offset
        if rel_offset < 0:
            return False

        start_record = rel_offset // MFT_RECORD_SIZE
        end_record = (rel_offset + length + MFT_RECORD_SIZE - 1) // MFT_RECORD_SIZE

        # Check if any virtual records fall in this range
        for record_num in self.mft_to_virtual.keys():
            if start_record <= record_num < end_record:
                return True

        return False
