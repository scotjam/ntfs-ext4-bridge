"""NTFS-ext4 Bridge main entry point.

Creates an NTFS image from an ext4 directory, starts the NBD server,
and runs the sync daemon for bidirectional synchronization.

Supports lazy allocation mode: large files start as sparse and are
only allocated when first read, then deallocated after a timeout.
This minimizes disk usage when backing up large file collections.

Usage:
    sudo python3 -m ntfs_bridge.bridge \
        --source /path/to/ext4/dir \
        --image /path/to/image.raw \
        --mount /mnt/ntfs-bridge \
        --port 10809 \
        --lazy  # Enable lazy allocation for large files
"""
import argparse
import fnmatch
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

from .cluster_mapper import ClusterMapper
from .nbd_server import NBDServer
from .sync_daemon import SyncDaemon
from .lazy_allocator import LazyAllocator
from .partition_wrapper import PartitionWrapper
from .virtual_files import VirtualFileManager
from .file_watcher import create_watcher, EVENT_CREATE, EVENT_DELETE


def log(msg):
    print(f"[Bridge] {msg}", flush=True)


class NTFSBridge:
    """Main bridge tying together ClusterMapper, NBD server, and SyncDaemon."""

    def __init__(self, image_path: str, source_dir: str,
                 ntfs_mount: str, port: int = 10809,
                 image_size_mb: int = 256,
                 lazy_alloc: bool = False,
                 dealloc_timeout: float = 60.0,
                 partitioned: bool = False,
                 virtual_mode: bool = False,
                 overflow_dir=None,
                 exclude_patterns=None):
        self.image_path = os.path.abspath(image_path)
        self.source_dir = os.path.abspath(source_dir)
        self.ntfs_mount = os.path.abspath(ntfs_mount)
        self.port = port
        self.image_size_mb = image_size_mb
        self.lazy_alloc = lazy_alloc
        self.dealloc_timeout = dealloc_timeout
        self.partitioned = partitioned
        self.overflow_dir = overflow_dir
        self.virtual_mode = virtual_mode
        self.exclude_patterns = list(exclude_patterns) if exclude_patterns else []

        self.mapper = None
        self.partition_wrapper = None
        self.nbd_server = None
        self.sync_daemon = None
        self.lazy_allocator = None
        self.virtual_file_manager = None
        self._file_watcher = None
        self._nbd_thread = None
        self._stopping = False

    def setup(self):
        """Set up the bridge: create image, populate it, initialize components."""
        log(f"Source directory: {self.source_dir}")
        log(f"Image path: {self.image_path}")
        log(f"NTFS mount point: {self.ntfs_mount}")

        # Validate source directory
        if not os.path.isdir(self.source_dir):
            log(f"ERROR: Source directory does not exist: {self.source_dir}")
            sys.exit(1)

        # Dynamically calculate image size from ext4 source content.
        # The image must be large enough to represent all file clusters in the
        # NTFS bitmap. Since we use truncate (sparse file), the actual disk
        # usage stays small — only metadata is written.
        total_bytes = 0
        file_count = 0
        for dirpath, dirnames, filenames in os.walk(self.source_dir, followlinks=True):
            rel_root = os.path.relpath(dirpath, self.source_dir)
            dirnames[:] = [d for d in dirnames
                           if not self._should_exclude(os.path.join(rel_root, d) if rel_root != '.' else d)]
            for f in filenames:
                rel_file = os.path.join(rel_root, f) if rel_root != '.' else f
                if self._should_exclude(rel_file):
                    continue
                try:
                    total_bytes += os.path.getsize(os.path.join(dirpath, f))
                    file_count += 1
                except OSError:
                    pass
        # Add 25% overhead for NTFS metadata (MFT zone reserves 12.5% by default) + round up to nearest 64MB
        needed_mb = int((total_bytes * 1.25) / (1024 * 1024)) + 64
        if needed_mb > self.image_size_mb:
            log(f"Auto-sizing image: {total_bytes/(1024**3):.1f}GB in {file_count} files -> {needed_mb}MB virtual image")
            self.image_size_mb = needed_mb

        if self.image_size_mb < 64:
            self.image_size_mb = 64
            log(f"Adjusted image size to minimum {self.image_size_mb}MB")

        # Step 1: Create NTFS image if it doesn't exist or is too small.
        #
        # IMPORTANT: Never recreate an existing image whose size has decreased
        # because the source symlinks were partially inaccessible at startup.
        # Recreating the image generates a new NTFS volume serial → new partition
        # GUID → Windows loses the F: drive mapping and requires a full VM reboot.
        #
        # Safe recreation rules:
        #   (a) If image does not exist → create it.
        #   (b) If image exists AND is smaller than the existing image has ever
        #       been (i.e. needed_mb > existing_size_mb by a large margin that
        #       cannot be explained by symlink staleness) → recreate only if
        #       needed_mb > existing_size_mb * 1.5  (source genuinely grew 50%+).
        #   (c) In all other cases, keep the existing image regardless of whether
        #       the freshly-calculated needed_mb is smaller than existing.
        image_is_fresh = False
        if os.path.exists(self.image_path):
            existing_size_mb = os.path.getsize(self.image_path) // (1024 * 1024)
            if needed_mb > existing_size_mb * 1.5:
                # Source has genuinely grown to more than 150% of existing image.
                log(f"Existing image too small ({existing_size_mb}MB, need {needed_mb}MB for "
                    f"{file_count} files), recreating...")
                os.remove(self.image_path)
                self._create_ntfs_image()
                image_is_fresh = True
            else:
                log(f"Using existing image: {self.image_path} ({existing_size_mb}MB, "
                    f"calculated need: {needed_mb}MB for {file_count} files)")
        else:
            self._create_ntfs_image()
            image_is_fresh = True

        # Step 2: Populate image from ext4 source
        self._populate_image(needs_fsfix=not image_is_fresh)

        # Step 2b: Fix $Bitmap for INDX clusters that ntfs-3g deallocated on
        # unmount but whose data runs still exist in the MFT.  Without this,
        # Windows/ntfs-3g reports those directories as "corrupted and unreadable"
        # because it finds $INDEX_ALLOC data runs pointing to FREE clusters.
        self._fix_indx_bitmap()

        # Step 3: Initialize ClusterMapper
        log("Initializing ClusterMapper...")
        self.mapper = ClusterMapper(self.image_path, self.source_dir,
                                     overflow_dir=self.overflow_dir)

        # Step 4: Create LazyAllocator if enabled
        if self.lazy_alloc:
            log(f"Enabling lazy allocation (dealloc timeout: {self.dealloc_timeout}s)")
            self.lazy_allocator = LazyAllocator(
                self.source_dir, self.ntfs_mount, self.mapper
            )
            self.lazy_allocator.DEALLOC_TIMEOUT = self.dealloc_timeout
            self.mapper.lazy_allocator = self.lazy_allocator

            # Pre-allocate all sparse files during setup
            # This is fast (no data copy) and ensures ntfs-3g sees allocated files
            # Sort largest files first: they need large contiguous free regions that
            # will be consumed by smaller files if those are allocated first.
            sparse_files = sorted(
                (p for p in self.mapper.sparse_files.keys()
                 if not self._should_exclude(p)),
                key=lambda p: self.mapper.sparse_files[p][1],  # file_size
                reverse=True
            )
            if sparse_files:
                log(f"Pre-allocating {len(sparse_files)} sparse files (largest first)...")
                for rel_path in sparse_files:
                    success = self.mapper.allocate_file_direct(rel_path)
                    if success:
                        self.lazy_allocator.register_file(
                            rel_path,
                            os.path.join(self.source_dir, rel_path),
                            is_allocated=True
                        )
                    else:
                        log(f"  Warning: Failed to pre-allocate {rel_path}")

                # Flush the mmap to disk and re-run ntfsfix to clear the journal
                # that ntfs-3g wrote during _populate_image().  Without this,
                # Windows replays ntfs-3g's journal on first mount and reverts all
                # the non-resident DATA attributes we just installed back to the
                # empty-resident state, making the files appear empty/corrupted.
                log("Flushing image and re-running ntfsfix to clear post-populate journal...")
                self.mapper.image.flush()
                fix_result = subprocess.run(
                    ['ntfsfix', self.image_path],
                    capture_output=True, text=True
                )
                log(f"ntfsfix (post-alloc): {fix_result.stdout.strip()}")
                if fix_result.returncode != 0:
                    log(f"ntfsfix warning: {fix_result.stderr.strip()}")

            # Register existing allocated files
            for record_num, source_path in self.mapper.mft_record_to_source.items():
                rel_path = os.path.relpath(source_path, self.source_dir)
                if rel_path not in self.mapper.sparse_files:
                    self.lazy_allocator.register_file(rel_path, source_path, is_allocated=True)

        # Step 5: Create VirtualFileManager if virtual mode enabled
        if self.virtual_mode:
            log("Enabling virtual file mode (live ext4→NTFS sync)")
            self.virtual_file_manager = VirtualFileManager(
                self.source_dir, self.mapper.cluster_size
            )
            self.virtual_file_manager.set_mapper(self.mapper)
            self.mapper.virtual_file_manager = self.virtual_file_manager

        # Step 5b: Fix INDEX_ALLOC data_size for directories where ntfs-3g shrinks
        # it on unmount but leaves valid INDX blocks beyond data_size.  Done LAST
        # (after allocate_file_direct and flush) so allocate_file_direct cannot
        # overwrite the fix via its MFT writes.  Writes via hot_cache so the NBD
        # server (which starts next) immediately serves the corrected data.
        self._fix_index_alloc_data_sizes()

        # Step 6: Create NBD server
        # Use PartitionWrapper for Windows VM mode (adds MBR partition table)
        if self.partitioned:
            log("Enabling partitioned mode (MBR wrapper for Windows VM)")
            self.partition_wrapper = PartitionWrapper(self.mapper)
            # If virtual mode is enabled, advertise larger size for virtual clusters
            if self.virtual_mode:
                # Advertise size to accommodate highest possible virtual cluster
                # VIRTUAL_CLUSTER_START (500000) + generous headroom for large files
                max_virtual_cluster = 600000  # VirtualFileManager starts at 500000
                self.partition_wrapper.set_virtual_size(
                    max_virtual_cluster, self.mapper.cluster_size
                )
            nbd_backend = self.partition_wrapper
        else:
            nbd_backend = self.mapper

        self.nbd_server = NBDServer(
            mapper=nbd_backend,
            host='0.0.0.0',  # Listen on all interfaces for VM access
            port=self.port
        )

        # Step 7: Create mount point
        os.makedirs(self.ntfs_mount, exist_ok=True)

        log("Setup complete")

    def run(self):
        """Run the bridge (blocking)."""
        # Start NBD server in background thread
        self._nbd_thread = threading.Thread(
            target=self.nbd_server.start,
            daemon=True,
            name="NBD-Server"
        )
        self._nbd_thread.start()
        log(f"NBD server started on port {self.port}")

        # Wait for server to be ready
        time.sleep(0.5)

        # Start lazy allocator background thread if enabled (before mount attempt)
        if self.lazy_allocator:
            self.lazy_allocator.start()

        # Start virtual file watcher if virtual mode enabled
        # This runs independently of ntfs-3g mount
        if self.virtual_mode:
            self._start_virtual_file_watcher()
            log("Virtual file watcher started (live ext4→NTFS sync)")

        # Connect nbd-client and mount
        # Skip in partitioned+virtual mode (VM connects directly to NBD)
        mount_success = False
        if self.partitioned and self.virtual_mode:
            log("VM mode: skipping local nbd-client/mount (VM connects directly)")
        else:
            mount_success = self._connect_and_mount()

            if mount_success:
                # After the production ntfs-3g mount, repair any INDX clusters
                # that ntfs-3g left FREE in $Bitmap during its startup fixups.
                # ntfs-3g may relocate $INDEX_ALLOC pages without correctly
                # updating $Bitmap, causing Windows to report directories as
                # "corrupted and unreadable".
                log("Post-mount: fixing INDX cluster bitmap entries...")
                self.mapper.fix_indx_clusters()

                # Start sync daemon for ntfs-3g based sync
                if not self.virtual_mode:
                    self.sync_daemon = SyncDaemon(
                        self.source_dir, self.ntfs_mount, self.mapper,
                        lazy_allocator=self.lazy_allocator
                    )
                    self.sync_daemon.start()
                    log("Sync daemon started")

                # Run catch-up populate in background via production mount.
                # The temp-mount populate may have failed for files in large
                # directories (EIO from INDX B-tree splits when bitmap is full).
                # Writing via the production ntfs-3g mount (through NBD) works
                # fine for those cases, so we do a second pass here.
                import threading as _threading
                t = _threading.Thread(
                    target=self._post_startup_populate,
                    daemon=True,
                    name="post-startup-populate"
                )
                t.start()

        log("="*60)
        log("NTFS-ext4 Bridge is running")
        log(f"  ext4 source: {self.source_dir}")
        log(f"  NBD port: {self.port}")
        if mount_success:
            log(f"  NTFS mount: {self.ntfs_mount}")
        if self.partitioned:
            log(f"  Partitioned mode: ENABLED (for Windows VM)")
        if self.lazy_alloc:
            log(f"  Lazy allocation: ENABLED (timeout: {self.dealloc_timeout}s)")
        if self.virtual_mode:
            log(f"  Virtual mode: ENABLED (no ntfs-3g mount required)")
        if not mount_success and not self.virtual_mode:
            log("  WARNING: ntfs-3g mount failed, ext4→NTFS sync disabled")
            log(f"  Connect manually: sudo nbd-client -N '' 127.0.0.1 {self.port} /dev/nbdX")
        log("  Press Ctrl+C to stop")
        log("="*60)

        # Wait for shutdown
        try:
            while not self._stopping:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        self.stop()

    def _start_virtual_file_watcher(self):
        """Start file watcher for virtual file mode.

        Watches ext4 source directory and adds/removes virtual files
        when changes are detected.
        """
        def on_file_event(event_type: str, rel_path: str):
            """Handle file events from watcher."""
            if not self.virtual_file_manager:
                return

            # Skip hidden/system files
            basename = os.path.basename(rel_path)
            if basename.startswith('.') or basename.startswith('$'):
                return

            # Skip if this file was just created by Windows (NTFS→ext4 sync)
            if rel_path in self.mapper.ntfs_sync_in_progress:
                return

            # Skip if this is already a real file in NTFS
            if rel_path in self.mapper.path_to_mft_record:
                return

            source_path = os.path.join(self.source_dir, rel_path)

            if event_type == EVENT_CREATE:
                if os.path.isdir(source_path):
                    self.virtual_file_manager.add_directory(rel_path)
                elif os.path.isfile(source_path):
                    self.virtual_file_manager.add_file(rel_path)

            elif event_type == EVENT_DELETE:
                # Remove virtual file/dir
                self.virtual_file_manager.remove_file(rel_path)
                self.virtual_file_manager.remove_directory(rel_path)

        self._file_watcher = create_watcher(self.source_dir, on_file_event)
        self._file_watcher.start()

    def stop(self):
        """Stop all components."""
        if self._stopping:
            return
        self._stopping = True

        log("Stopping bridge...")

        if self._file_watcher:
            self._file_watcher.stop()

        if self.sync_daemon:
            self.sync_daemon.stop()

        if self.lazy_allocator:
            self.lazy_allocator.stop()

        self._unmount_and_disconnect()

        if self.nbd_server:
            self.nbd_server.stop()

        # Save image changes
        if self.mapper:
            log("Saving image...")
            self.mapper.flush()

        log("Bridge stopped")

    def _create_ntfs_image(self):
        """Create a new NTFS image file."""
        log(f"Creating {self.image_size_mb}MB NTFS image...")

        # Create sparse image file
        size_bytes = self.image_size_mb * 1024 * 1024
        result = subprocess.run(
            ['truncate', '-s', str(size_bytes), self.image_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # Fallback: create with dd
            result = subprocess.run(
                ['dd', 'if=/dev/zero', f'of={self.image_path}',
                 'bs=1M', f'count={self.image_size_mb}'],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                log(f"ERROR: Failed to create image: {result.stderr}")
                sys.exit(1)

        # Format as NTFS
        result = subprocess.run(
            ['mkfs.ntfs', '-F', '-Q', '-c', '65536', self.image_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log(f"ERROR: mkfs.ntfs failed: {result.stderr}")
            sys.exit(1)

        # mkfs.ntfs on large sparse images sets the boot sector's MFTMirr LCN
        # to the volume midpoint, but MFT record 1 ($MFTMirr) stores the data
        # runs at a different cluster.  This mismatch causes Windows/ntfsprogs
        # to report "$MFTMirr does not match $MFT" and refuse to open directories.
        # Fix: read MFT record 1 data runs and patch the boot sector to match.
        self._fix_mftmirr_boot_sector()

        log("NTFS image created")

    def _fix_mftmirr_boot_sector(self):
        """Patch boot sector MFTMirr LCN to match MFT record 1 data runs.

        mkfs.ntfs computes the boot sector MFTMirr LCN as total_clusters//2,
        but places the actual $MFTMirr record at a different cluster.  The
        inconsistency triggers 'Bad $MFTMirr lcn' errors and makes directories
        unreadable in Windows.
        """
        import struct as _struct
        try:
            with open(self.image_path, 'r+b') as f:
                # Read boot sector
                f.seek(0)
                bs = bytearray(f.read(512))
                bytes_per_sector = _struct.unpack_from('<H', bs, 11)[0]
                sectors_per_cluster = bs[13]
                cluster_size = bytes_per_sector * sectors_per_cluster
                mft_cluster = _struct.unpack_from('<q', bs, 48)[0]
                boot_mftmirr = _struct.unpack_from('<q', bs, 56)[0]

                # Read MFT record 1 ($MFTMirr) — undo fixups, find $DATA run
                rec_offset = mft_cluster * cluster_size + 1 * 1024
                f.seek(rec_offset)
                raw = bytearray(f.read(1024))
                if raw[:4] != b'FILE':
                    log(f"  MFTMirr fix: record 1 is not FILE, skipping")
                    return

                # Undo update sequence array fixups
                usa_off = _struct.unpack_from('<H', raw, 4)[0]
                usa_count = _struct.unpack_from('<H', raw, 6)[0]
                seq_num = _struct.unpack_from('<H', raw, usa_off)[0]
                for i in range(1, usa_count):
                    sector_end = i * 512 - 2
                    if raw[sector_end:sector_end + 2] == _struct.pack('<H', seq_num):
                        raw[sector_end:sector_end + 2] = raw[usa_off + i * 2:usa_off + i * 2 + 2]

                # Find first non-sparse run in $DATA
                first_attr = _struct.unpack_from('<H', raw, 20)[0]
                off = first_attr
                mft1_lcn = None
                while off < 1024 - 8:
                    atype = _struct.unpack_from('<I', raw, off)[0]
                    if atype == 0xFFFFFFFF:
                        break
                    alen = _struct.unpack_from('<I', raw, off + 4)[0]
                    if alen == 0 or alen > 1024:
                        break
                    if atype == 0x80 and not raw[off + 9]:  # unnamed $DATA
                        non_res = raw[off + 8]
                        if non_res:
                            runs_off = _struct.unpack_from('<H', raw, off + 32)[0]
                            runs = raw[off + runs_off:off + alen]
                            pos = 0; lcn = 0
                            while pos < len(runs) and runs[pos]:
                                hdr = runs[pos]; pos += 1
                                ls = hdr & 0xF; os2 = (hdr >> 4) & 0xF
                                pos += ls
                                if os2:
                                    roff = int.from_bytes(runs[pos:pos + os2], 'little', signed=True)
                                    pos += os2
                                    lcn += roff
                                    mft1_lcn = lcn
                                    break
                    off += alen

                if mft1_lcn is None or mft1_lcn == boot_mftmirr:
                    return  # Nothing to fix

                log(f"  Fixing boot sector MFTMirr LCN: {boot_mftmirr:#x} → {mft1_lcn:#x}")
                _struct.pack_into('<q', bs, 56, mft1_lcn)
                f.seek(0)
                f.write(bytes(bs))

                # Also fix the backup boot sector at the end of the volume
                f.seek(0, 2)
                vol_size = f.tell()
                backup_offset = vol_size - 512
                if backup_offset > 512:
                    f.seek(backup_offset)
                    bbs = bytearray(f.read(512))
                    if bbs[:8] == bs[:8]:  # Same OEM ID
                        _struct.pack_into('<q', bbs, 56, mft1_lcn)
                        f.seek(backup_offset)
                        f.write(bytes(bbs))
        except Exception as e:
            log(f"  MFTMirr fix error (non-fatal): {e}")

    def _fix_indx_bitmap(self):
        """Mark directory INDX clusters as used in $Bitmap.

        ntfs-3g allocates clusters for directory $INDEX_ALLOC pages during
        _populate_image(), writes valid INDX data there, but then frees those
        clusters in $Bitmap on unmount (compact/trim behaviour on sparse images).
        The $INDEX_ALLOC data runs in the MFT still reference those clusters,
        creating an inconsistency: Windows/ntfs-3g sees FREE clusters referenced
        by $INDEX_ALLOC and reports 'corrupted and unreadable'.

        Fix: scan every MFT record for $INDEX_ALLOC attributes and mark all
        their clusters as USED in $Bitmap before ClusterMapper reads the bitmap.
        """
        import struct as _struct

        MFT_RECORD_SIZE = 1024
        log("Fixing $Bitmap for INDX clusters (ntfs-3g unmount may have freed them)...")

        try:
            with open(self.image_path, 'r+b') as f:
                # Read NTFS params from boot sector
                f.seek(0)
                bs = f.read(512)
                bytes_per_sector = _struct.unpack_from('<H', bs, 11)[0]
                sectors_per_cluster = bs[13]
                cluster_size = bytes_per_sector * sectors_per_cluster
                mft_cluster = _struct.unpack_from('<q', bs, 48)[0]
                total_sectors = _struct.unpack_from('<Q', bs, 40)[0]
                total_clusters = total_sectors // sectors_per_cluster
                mft_offset = mft_cluster * cluster_size

                # Find $Bitmap location (MFT record 6)
                f.seek(mft_offset + 6 * MFT_RECORD_SIZE)
                raw = bytearray(f.read(MFT_RECORD_SIZE))
                if raw[:4] != b'FILE':
                    log("  Cannot find $Bitmap record, skipping")
                    return
                raw = self._undo_fixups_raw(raw)

                # Parse $Bitmap $DATA runs
                bitmap_runs = []
                first_attr = _struct.unpack_from('<H', raw, 20)[0]
                off = first_attr
                while off < MFT_RECORD_SIZE - 8:
                    atype = _struct.unpack_from('<I', raw, off)[0]
                    if atype == 0xFFFFFFFF: break
                    alen = _struct.unpack_from('<I', raw, off + 4)[0]
                    if alen == 0 or alen > MFT_RECORD_SIZE: break
                    if atype == 0x80 and not raw[off + 9] and raw[off + 8]:
                        bitmap_real_size = _struct.unpack_from('<Q', raw, off + 48)[0]
                        runs_off = _struct.unpack_from('<H', raw, off + 32)[0]
                        rb = raw[off + runs_off:off + alen]
                        pos = 0; lcn = 0
                        while pos < len(rb) and rb[pos]:
                            hdr = rb[pos]; pos += 1
                            ls = hdr & 0xF; os2 = (hdr >> 4) & 0xF
                            rlen = int.from_bytes(rb[pos:pos + ls], 'little'); pos += ls
                            if os2:
                                roff = int.from_bytes(rb[pos:pos + os2], 'little', signed=True)
                                pos += os2; lcn += roff
                                bitmap_runs.append((lcn, rlen))
                            else:
                                bitmap_runs.append((-1, rlen))
                    off += alen

                if not bitmap_runs:
                    log("  Cannot parse $Bitmap runs, skipping")
                    return

                def read_bitmap_byte(cluster_num):
                    """Read the bitmap byte containing the bit for cluster_num."""
                    byte_pos = cluster_num // 8
                    cum = 0
                    for run_lcn, run_len in bitmap_runs:
                        if run_lcn < 0:
                            cum += run_len * cluster_size
                            continue
                        run_bytes = run_len * cluster_size
                        if cum <= byte_pos < cum + run_bytes:
                            f.seek(run_lcn * cluster_size + (byte_pos - cum))
                            return f.read(1)[0]
                        cum += run_bytes
                    return None

                def write_bitmap_byte(cluster_num, byte_val):
                    """Write the bitmap byte containing the bit for cluster_num."""
                    byte_pos = cluster_num // 8
                    cum = 0
                    for run_lcn, run_len in bitmap_runs:
                        if run_lcn < 0:
                            cum += run_len * cluster_size
                            continue
                        run_bytes = run_len * cluster_size
                        if cum <= byte_pos < cum + run_bytes:
                            f.seek(run_lcn * cluster_size + (byte_pos - cum))
                            f.write(bytes([byte_val]))
                            return True
                        cum += run_bytes
                    return False

                # Scan ALL MFT records for $INDEX_ALLOC attributes.
                # Stop after 200 consecutive non-FILE records (past end of MFT).
                indx_clusters = set()
                rec_num = 0
                consecutive_non_file = 0
                while consecutive_non_file < 200:
                    offset = mft_offset + rec_num * MFT_RECORD_SIZE
                    f.seek(offset)
                    raw = f.read(MFT_RECORD_SIZE)
                    if not raw or len(raw) < MFT_RECORD_SIZE:
                        break
                    if raw[:4] != b'FILE':
                        consecutive_non_file += 1
                        rec_num += 1
                        continue
                    consecutive_non_file = 0

                    raw = self._undo_fixups_raw(bytearray(raw))
                    flags = _struct.unpack_from('<H', raw, 22)[0]
                    if not (flags & 0x2):  # Not a directory
                        rec_num += 1
                        continue

                    # Walk attributes looking for $INDEX_ALLOC (0xA0)
                    first_attr = _struct.unpack_from('<H', raw, 20)[0]
                    off = first_attr
                    while off < MFT_RECORD_SIZE - 8:
                        atype = _struct.unpack_from('<I', raw, off)[0]
                        if atype == 0xFFFFFFFF: break
                        alen = _struct.unpack_from('<I', raw, off + 4)[0]
                        if alen == 0 or alen > MFT_RECORD_SIZE: break
                        if atype == 0xA0 and raw[off + 8]:  # $INDEX_ALLOC non-res
                            runs_off = _struct.unpack_from('<H', raw, off + 32)[0]
                            rb = raw[off + runs_off:off + alen]
                            pos = 0; lcn = 0
                            while pos < len(rb) and rb[pos]:
                                hdr = rb[pos]; pos += 1
                                ls = hdr & 0xF; os2 = (hdr >> 4) & 0xF
                                rlen = int.from_bytes(rb[pos:pos + ls], 'little'); pos += ls
                                if os2:
                                    roff = int.from_bytes(rb[pos:pos + os2], 'little', signed=True)
                                    pos += os2; lcn += roff
                                    for c in range(lcn, lcn + rlen):
                                        indx_clusters.add(c)
                                else:
                                    pos += 0  # sparse $INDEX_ALLOC run, no clusters
                        off += alen
                    rec_num += 1

                log(f"  Found {len(indx_clusters)} INDX clusters across all directories")

                # Mark them all as USED in $Bitmap (set the bit)
                fixed = 0
                for cluster in sorted(indx_clusters):
                    if cluster < 0 or cluster >= total_clusters:
                        continue
                    byte_val = read_bitmap_byte(cluster)
                    if byte_val is None:
                        continue
                    bit = (byte_val >> (cluster % 8)) & 1
                    if not bit:  # Was FREE, needs fixing
                        new_byte = byte_val | (1 << (cluster % 8))
                        write_bitmap_byte(cluster, new_byte)
                        fixed += 1

                log(f"  Fixed {fixed} INDX clusters in $Bitmap")

        except Exception as e:
            log(f"  INDX bitmap fix error (non-fatal): {e}")
            import traceback
            traceback.print_exc()

    def _fix_index_alloc_data_sizes(self):
        """Extend INDEX_ALLOC data_size to cover all valid INDX blocks in allocated clusters.

        ntfs-3g on unmount may set data_size to only cover INDX blocks it considers
        'active' (those with INDEX_BITMAP bits set), while leaving valid INDX data
        (with valid INDX signatures and entries) in allocated clusters beyond data_size.
        If an internal B+ tree node has a child pointer to such a beyond-data_size block,
        Windows reports 'The file or directory is corrupted and unreadable'.

        This function scans every directory MFT record, checks if any valid INDX blocks
        exist beyond the current data_size in the allocated clusters, and if so:
          - Extends data_size (and init_size) to cover them
          - Sets the corresponding bit in the per-directory INDEX_BITMAP attribute
          - Writes the corrected MFT record back via self.mapper.image (hot_cache)

        Must be called AFTER ClusterMapper is initialized (uses self.mapper.image).
        Writing through the hot_cache ensures the NBD server immediately serves the
        corrected data, and the fix persists to disk on the next image flush().
        """
        import struct as _struct

        INDX_RECORD_SIZE = 4096   # bytes per index record
        MFT_RECORD_SIZE = 1024
        log("Checking INDEX_ALLOC data_size for hidden INDX blocks...")

        def redo_fixups(record: bytearray) -> bytearray:
            """Re-apply USA fixups to an MFT record before writing."""
            record = bytearray(record)
            usa_off = _struct.unpack_from('<H', record, 4)[0]
            usa_count = _struct.unpack_from('<H', record, 6)[0]
            seq = _struct.unpack_from('<H', record, usa_off)[0]
            for i in range(1, usa_count):
                sec_end = i * 512 - 2
                _struct.pack_into('<H', record, usa_off + i * 2,
                                  _struct.unpack_from('<H', record, sec_end)[0])
                _struct.pack_into('<H', record, sec_end, seq)
            return record

        def decode_runs(rb):
            runs = []
            pos = 0; lcn = 0
            while pos < len(rb) and rb[pos]:
                hdr = rb[pos]; pos += 1
                ls = hdr & 0xF; os2 = (hdr >> 4) & 0xF
                rlen = int.from_bytes(rb[pos:pos + ls], 'little'); pos += ls
                if os2:
                    delta = int.from_bytes(rb[pos:pos + os2], 'little', signed=True)
                    pos += os2; lcn += delta
                    runs.append((lcn, rlen))
            return runs

        try:
            img = self.mapper.image
            cluster_size = self.mapper.cluster_size
            bps = self.mapper.bytes_per_sector
            mft_offset = self.mapper.mft_offset
            img_len = len(img)

            vcn_step = INDX_RECORD_SIZE // bps  # VCNs per INDX block (e.g. 8 for 512B sectors)

            def stream_to_file_offset(stream_byte_off, runs):
                cur = 0
                for lcn, length in runs:
                    run_bytes = length * cluster_size
                    if cur + run_bytes > stream_byte_off:
                        return lcn * cluster_size + (stream_byte_off - cur)
                    cur += run_bytes
                return None

            fixed_count = 0
            rec_num = 0
            consecutive_non_file = 0

            while consecutive_non_file < 200:
                mft_file_off = mft_offset + rec_num * MFT_RECORD_SIZE
                if mft_file_off + MFT_RECORD_SIZE > img_len:
                    break
                raw = bytes(img[mft_file_off:mft_file_off + MFT_RECORD_SIZE])
                if len(raw) < MFT_RECORD_SIZE:
                    break
                if raw[:4] != b'FILE':
                    consecutive_non_file += 1
                    rec_num += 1
                    continue
                consecutive_non_file = 0

                rec = self._undo_fixups_raw(bytearray(raw))
                flags = _struct.unpack_from('<H', rec, 22)[0]
                if not (flags & 0x2):  # Not a directory
                    rec_num += 1
                    continue

                # Parse INDEX_ALLOC (0xA0) and INDEX_BITMAP (0xB0) attrs
                ia_off = None; ia_alloc = 0; ia_data = 0; ia_runs = []
                ib_off = None; ib_val_off = 0; ib_val_len = 0

                p = _struct.unpack_from('<H', rec, 20)[0]
                while p < MFT_RECORD_SIZE - 8:
                    at = _struct.unpack_from('<I', rec, p)[0]
                    al = _struct.unpack_from('<I', rec, p + 4)[0]
                    if at == 0xFFFFFFFF or al == 0: break
                    if at == 0xA0 and rec[p + 8]:  # nonresident INDEX_ALLOC
                        ia_off = p
                        ia_alloc = _struct.unpack_from('<Q', rec, p + 40)[0]
                        ia_data  = _struct.unpack_from('<Q', rec, p + 48)[0]
                        dr_off = _struct.unpack_from('<H', rec, p + 32)[0]
                        ia_runs = decode_runs(rec[p + dr_off:p + al])
                    if at == 0xB0 and not rec[p + 8]:  # resident INDEX_BITMAP
                        ib_off = p
                        ib_val_off = _struct.unpack_from('<H', rec, p + 20)[0]
                        ib_val_len = _struct.unpack_from('<I', rec, p + 16)[0]
                    p += al

                if ia_off is None or not ia_runs or ia_data >= ia_alloc:
                    rec_num += 1
                    continue

                # Read current INDEX_BITMAP before scanning so we can start
                # new_bitmap from the existing bits (scan only ADDs bits, never
                # removes them — prevents clearing bits for directories whose
                # INDX cluster data isn't in the hot image but was written by
                # ntfs-3g and is still valid).
                old_bitmap = (bytes(rec[ib_off + ib_val_off:
                                        ib_off + ib_val_off + ib_val_len])
                              if ib_off is not None else None)

                # Scan ALL INDX blocks in the allocated space to compute the
                # full expected data_size and INDEX_BITMAP.
                new_data = ia_data
                new_bitmap = (bytearray(old_bitmap) if old_bitmap is not None
                              else (bytearray(ib_val_len)
                                    if ib_off is not None else None))
                block_idx = 0
                vcn = 0
                while vcn * bps < ia_alloc:
                    stream_byte_off = vcn * bps
                    foff = stream_to_file_offset(stream_byte_off, ia_runs)
                    if foff is not None and foff + INDX_RECORD_SIZE <= img_len:
                        blk = bytes(img[foff:foff + INDX_RECORD_SIZE])
                        if len(blk) == INDX_RECORD_SIZE and blk[:4] == b'INDX':
                            end = stream_byte_off + INDX_RECORD_SIZE
                            if end > new_data:
                                new_data = end
                            if (new_bitmap is not None
                                    and block_idx // 8 < len(new_bitmap)):
                                new_bitmap[block_idx // 8] |= (
                                    1 << (block_idx % 8))
                    vcn += vcn_step
                    block_idx += 1

                bitmap_changed = (old_bitmap is not None
                                  and new_bitmap is not None
                                  and bytes(new_bitmap) != old_bitmap)

                # Always register protection so _mft_write_to_image() re-patches
                # data_size and INDEX_BITMAP whenever Windows writes back stale
                # smaller values (journal replay, access-time updates, etc.).
                # Done regardless of whether an extension was needed this startup,
                # so the protection is active even when the disk is already correct.
                _ib_off_reg = ib_off if ib_off is not None else -1
                _ib_val_off_reg = ib_val_off if ib_off is not None else 0
                _bitmap_reg = (bytes(new_bitmap) if new_bitmap is not None
                               else b'')
                self.mapper.protect_ia_size(
                    rec_num, ia_off, new_data,
                    _ib_off_reg, _ib_val_off_reg, _bitmap_reg)

                if new_data > ia_data or bitmap_changed:
                    # Get dir name for log
                    name = ""
                    p2 = _struct.unpack_from('<H', rec, 20)[0]
                    while p2 < MFT_RECORD_SIZE - 8:
                        at = _struct.unpack_from('<I', rec, p2)[0]
                        al = _struct.unpack_from('<I', rec, p2 + 4)[0]
                        if at == 0xFFFFFFFF or al == 0: break
                        if at == 0x30:
                            nl = rec[p2 + 88]
                            name = rec[p2 + 90:p2 + 90 + nl * 2].decode(
                                'utf-16-le', errors='replace')
                            break
                        p2 += al

                    # Patch the logical (fixup-undone) record
                    _struct.pack_into('<Q', rec, ia_off + 48, new_data)  # data_size
                    _struct.pack_into('<Q', rec, ia_off + 56, new_data)  # init_size
                    if new_bitmap is not None and ib_off is not None:
                        bm_start = ib_off + ib_val_off
                        rec[bm_start:bm_start + ib_val_len] = new_bitmap

                    # Re-apply fixups and write via hot_cache (NBD server reads this)
                    on_disk = redo_fixups(rec)
                    img[mft_file_off:mft_file_off + MFT_RECORD_SIZE] = bytes(on_disk)

                    # Verify the write landed in hot_cache correctly
                    check = bytes(img[mft_file_off + ia_off + 48:
                                      mft_file_off + ia_off + 56])
                    got = _struct.unpack_from('<Q', bytes(check))[0]
                    bm_desc = ""
                    if bitmap_changed and new_bitmap is not None and old_bitmap is not None:
                        bm_desc = (f" bitmap {old_bitmap.hex()}"
                                   f"->{bytes(new_bitmap).hex()}")
                    if got == new_data:
                        log(f"  Record {rec_num} ({name!r}): "
                            f"data_size {ia_data} -> {new_data}{bm_desc} [OK]")
                    else:
                        log(f"  Record {rec_num} ({name!r}): "
                            f"data_size write FAILED (expected {new_data}, got {got})")

                    fixed_count += 1

                rec_num += 1

            log(f"  Fixed INDEX_ALLOC/bitmap for {fixed_count} director(ies)")
            if fixed_count:
                # Flush to disk so the fix survives a bridge restart without
                # needing to re-scan every time.
                self.mapper.image.flush()
                log(f"  Flushed {fixed_count} INDEX_ALLOC fix(es) to disk")

        except Exception as e:
            log(f"  Index alloc data_size fix error (non-fatal): {e}")
            import traceback
            traceback.print_exc()

    def _undo_fixups_raw(self, record: bytearray) -> bytearray:
        """Undo USA fixups without logging."""
        import struct as _struct
        usa_off = _struct.unpack_from('<H', record, 4)[0]
        usa_count = _struct.unpack_from('<H', record, 6)[0]
        seq_num = _struct.unpack_from('<H', record, usa_off)[0]
        for i in range(1, usa_count):
            sector_end = i * 512 - 2
            if record[sector_end:sector_end + 2] == _struct.pack('<H', seq_num):
                record[sector_end:sector_end + 2] = record[usa_off + i * 2:usa_off + i * 2 + 2]
        return record

    def _populate_image(self, needs_fsfix: bool = False):
        """Populate the NTFS image with files from ext4 source."""
        # Clean up any stale ntfs-init mounts from previous runs
        # (if the bridge was killed, the finally block may not have run)
        import glob as globmod
        for stale_dir in globmod.glob('/root/ntfs-init-*'):
            subprocess.run(['umount', stale_dir], capture_output=True)
            subprocess.run(['fusermount', '-u', stale_dir], capture_output=True)
            try:
                os.rmdir(stale_dir)
                log(f"Cleaned up stale mount: {stale_dir}")
            except OSError:
                pass

        # Mount image directly and copy directory structure + sparse files
        # Use /root for temp mount to avoid tmpfs space issues with large files
        tmp_mount = tempfile.mkdtemp(prefix='ntfs-init-', dir='/root')

        try:
            if needs_fsfix:
                # Fix dirty NTFS filesystem before mounting.
                #
                # Run full ntfsfix (no -d flag) so it both:
                #   1. Empties the $LogFile (prevents Windows from hanging on
                #      log replay after an unclean bridge shutdown)
                #   2. Clears the dirty bit (allows ntfs-3g rw mount)
                #
                # ntfsfix -d only clears the dirty bit but leaves the $LogFile
                # intact. Windows replays any entries it finds, which can cause
                # an indefinite hang when the log contains stale transactions
                # from a bridge that was killed mid-write.
                fix_result = subprocess.run(
                    ['ntfsfix', self.image_path],
                    capture_output=True, text=True
                )
                log(f"ntfsfix: {fix_result.stdout.strip()}")
                if fix_result.returncode != 0:
                    log(f"ntfsfix warning: {fix_result.stderr.strip()}")

            # Mount the image with ntfs-3g.
            # Use 'recover' to force read-write even if the volume has a
            # dirty/scheduled-check flag from allocate_file_direct's direct
            # NTFS writes (which bypass the ntfs-3g journal).
            result = subprocess.run(
                ['mount', '-t', 'ntfs-3g', '-o', 'rw,big_writes,recover',
                 self.image_path, tmp_mount],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                log(f"WARNING: Could not mount image for population: {result.stderr}")
                log("Image may need manual population")
                return

            log("Populating NTFS image from ext4 source...")
            files_created = 0
            files_skipped = 0
            dirs_created = 0

            for root, dirs, files in os.walk(self.source_dir, followlinks=True):
                rel_root = os.path.relpath(root, self.source_dir)
                dirs[:] = [d for d in dirs
                           if not self._should_exclude(os.path.join(rel_root, d) if rel_root != '.' else d)]

                # Create directories
                for d in dirs:
                    rel_dir = os.path.join(rel_root, d) if rel_root != '.' else d
                    ntfs_dir = os.path.join(tmp_mount, rel_dir)
                    try:
                        os.makedirs(ntfs_dir, exist_ok=True)
                        dirs_created += 1
                    except OSError as e:
                        log(f"  Warning: could not create dir {rel_dir}: {e}")

                # Create files - sparse for large files if lazy_alloc enabled
                for f in files:
                    rel_file = os.path.join(rel_root, f) if rel_root != '.' else f
                    if self._should_exclude(rel_file):
                        continue
                    source_file = os.path.join(root, f)
                    ntfs_file = os.path.join(tmp_mount, rel_file)

                    # Skip files that already exist on the NTFS mount
                    # (avoids re-creating sparse files that were already allocated)
                    if os.path.exists(ntfs_file):
                        files_skipped += 1
                        continue

                    try:
                        file_size = os.path.getsize(source_file)

                        if self.lazy_alloc and file_size > 700:
                            # Large file with lazy alloc - create truly sparse NTFS entry.
                            # os.truncate (rather than seek+write) sets the file size in
                            # the NTFS MFT without allocating any clusters. The seek+write
                            # approach allocated 1 trailing cluster per file, which with
                            # thousands of sparse files fragments the NTFS bitmap completely.
                            with open(ntfs_file, 'wb'):
                                pass  # Create empty file
                            os.truncate(ntfs_file, file_size)
                        else:
                            # Small file or no lazy alloc - copy content
                            shutil.copy2(source_file, ntfs_file)
                        files_created += 1
                    except OSError as e:
                        log(f"  Warning: could not create file {rel_file}: {e}")

            log(f"Populated: {dirs_created} dirs, {files_created} files"
                f"{f', {files_skipped} skipped (existing)' if files_skipped else ''}")

            # Also populate overflow_dir items at NTFS root
            # These are root-level items created by Windows (System Volume Information,
            # SID folders, user files) that survive restarts but were previously lost
            # on image recreation.
            if self.overflow_dir and os.path.isdir(self.overflow_dir) and \
                    os.path.abspath(self.overflow_dir) != os.path.abspath(self.source_dir):
                overflow_created = 0
                overflow_skipped = 0
                for root, dirs, files in os.walk(self.overflow_dir, followlinks=True):
                    rel_root = os.path.relpath(root, self.overflow_dir)

                    for d in dirs:
                        rel_dir = os.path.join(rel_root, d) if rel_root != '.' else d
                        ntfs_dir = os.path.join(tmp_mount, rel_dir)
                        try:
                            os.makedirs(ntfs_dir, exist_ok=True)
                        except OSError as e:
                            log(f"  Warning: could not create overflow dir {rel_dir}: {e}")

                    for f in files:
                        rel_file = os.path.join(rel_root, f) if rel_root != '.' else f
                        source_file = os.path.join(root, f)
                        ntfs_file = os.path.join(tmp_mount, rel_file)

                        if os.path.exists(ntfs_file):
                            overflow_skipped += 1
                            continue

                        try:
                            file_size = os.path.getsize(source_file)
                            if self.lazy_alloc and file_size > 700:
                                with open(ntfs_file, 'wb'):
                                    pass
                                os.truncate(ntfs_file, file_size)
                            else:
                                shutil.copy2(source_file, ntfs_file)
                            overflow_created += 1
                        except OSError as e:
                            log(f"  Warning: could not create overflow file {rel_file}: {e}")

                if overflow_created or overflow_skipped:
                    log(f"Overflow dir: {overflow_created} items restored"
                        f"{f', {overflow_skipped} skipped (existing)' if overflow_skipped else ''}")

            # Sync and unmount
            subprocess.run(['sync'], capture_output=True)
            result = subprocess.run(
                ['umount', tmp_mount],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                # Try fusermount
                subprocess.run(
                    ['fusermount', '-u', tmp_mount],
                    capture_output=True, text=True
                )

        except Exception as e:
            log(f"Population error: {e}")
            # Try to unmount on error
            subprocess.run(['umount', tmp_mount], capture_output=True)
        finally:
            try:
                os.rmdir(tmp_mount)
            except OSError:
                pass

    def _post_startup_populate(self):
        """Catch-up populate via production ntfs-3g mount.

        The pre-startup temp-mount populate fails with EIO for files in large
        directories when the NTFS cluster bitmap is full (no room for INDX
        B-tree splits). The production mount (via NBD) handles these correctly.

        This runs in a background thread after the production mount is active
        and copies any ext4 source files that are absent from the NTFS volume.
        """
        import time as _time
        # Give the production mount a moment to settle
        _time.sleep(2)

        log("Post-startup populate: checking for missing files via production mount...")
        files_created = 0
        files_failed = 0

        for root, dirs, files in os.walk(self.source_dir, followlinks=True):
            rel_root = os.path.relpath(root, self.source_dir)
            dirs[:] = [d for d in dirs
                       if not self._should_exclude(os.path.join(rel_root, d) if rel_root != '.' else d)]

            # Ensure directories exist
            for d in dirs:
                rel_dir = os.path.join(rel_root, d) if rel_root != '.' else d
                ntfs_dir = os.path.join(self.ntfs_mount, rel_dir)
                try:
                    os.makedirs(ntfs_dir, exist_ok=True)
                except OSError:
                    pass

            for f in files:
                rel_file = os.path.join(rel_root, f) if rel_root != '.' else f
                if self._should_exclude(rel_file):
                    continue
                source_file = os.path.join(root, f)
                ntfs_file = os.path.join(self.ntfs_mount, rel_file)

                if os.path.exists(ntfs_file):
                    continue

                try:
                    file_size = os.path.getsize(source_file)
                    if self.lazy_alloc and file_size > 700:
                        with open(ntfs_file, 'wb'):
                            pass  # Create empty file
                        os.truncate(ntfs_file, file_size)
                    else:
                        shutil.copy2(source_file, ntfs_file)
                    files_created += 1
                    if files_created % 50 == 0:
                        log(f"Post-startup populate: {files_created} files added so far...")
                except OSError as e:
                    log(f"  Post-populate warning: could not create {rel_file}: {e}")
                    files_failed += 1

        if files_created or files_failed:
            log(f"Post-startup populate complete: {files_created} files added"
                f"{f', {files_failed} failed' if files_failed else ''}")
        else:
            log("Post-startup populate: no missing files found")

    def _connect_and_mount(self) -> bool:
        """Connect nbd-client and mount ntfs-3g."""
        # Find a free NBD device
        nbd_device = self._find_free_nbd()
        if not nbd_device:
            log("No free NBD device found")
            return False

        self._nbd_device = nbd_device

        # Connect nbd-client
        log(f"Connecting nbd-client to {nbd_device}...")
        result = subprocess.run(
            ['nbd-client', '-N', '', '127.0.0.1', str(self.port), nbd_device],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log(f"nbd-client failed: {result.stderr}")
            return False

        # Wait for device to be ready
        time.sleep(1)

        # Determine mount device
        if self.partitioned:
            # In partitioned mode, mount the partition (p1), not the whole disk
            # First, trigger partition table read
            subprocess.run(['partprobe', nbd_device], capture_output=True)
            time.sleep(1)
            mount_device = f"{nbd_device}p1"
            # Check if partition device exists
            if not os.path.exists(mount_device):
                log(f"Partition device {mount_device} not found, trying {nbd_device}1")
                mount_device = f"{nbd_device}1"  # Some systems use nbd0p1, others nbd01
        else:
            mount_device = nbd_device

        # Mount with ntfs-3g
        log(f"Mounting ntfs-3g {mount_device} on {self.ntfs_mount}...")
        result = subprocess.run(
            ['mount', '-t', 'ntfs-3g', '-o', 'rw,big_writes,recover',
             mount_device, self.ntfs_mount],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log(f"ntfs-3g mount failed: {result.stderr}")
            # Disconnect nbd-client
            subprocess.run(['nbd-client', '-d', nbd_device], capture_output=True)
            return False

        log(f"Mounted on {self.ntfs_mount}")
        return True

    def _unmount_and_disconnect(self):
        """Unmount ntfs-3g and disconnect nbd-client."""
        nbd_device = getattr(self, '_nbd_device', None)

        # Unmount
        if os.path.ismount(self.ntfs_mount):
            log("Unmounting...")
            subprocess.run(['umount', self.ntfs_mount], capture_output=True)
            time.sleep(0.5)
            if os.path.ismount(self.ntfs_mount):
                subprocess.run(['fusermount', '-u', self.ntfs_mount], capture_output=True)

        # Disconnect NBD
        if nbd_device:
            log(f"Disconnecting {nbd_device}...")
            subprocess.run(['nbd-client', '-d', nbd_device], capture_output=True)

    def _find_free_nbd(self) -> str:
        """Find a free /dev/nbdX device."""
        for i in range(16):
            device = f'/dev/nbd{i}'
            if os.path.exists(device):
                # Check if already in use
                result = subprocess.run(
                    ['nbd-client', '-c', device],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    return device
        return ''

    def _should_exclude(self, rel_path: str) -> bool:
        """Return True if rel_path matches any --exclude pattern."""
        if not self.exclude_patterns:
            return False
        name = os.path.basename(rel_path)
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern):
                return True
        return False

    @staticmethod
    def _get_dir_size(path: str) -> int:
        """Get total size of files in a directory."""
        total = 0
        for root, dirs, files in os.walk(path, followlinks=True):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total


def main():
    parser = argparse.ArgumentParser(
        description='NTFS-ext4 Bridge: present ext4 directory as NTFS via NBD'
    )
    parser.add_argument('--source', required=True,
                        help='Path to ext4 source directory')
    parser.add_argument('--image', required=True,
                        help='Path to NTFS image file (created if missing)')
    parser.add_argument('--mount', required=True,
                        help='Path to mount NTFS via ntfs-3g')
    parser.add_argument('--port', type=int, default=10809,
                        help='NBD server port (default: 10809)')
    parser.add_argument('--size', type=int, default=256,
                        help='Minimum image size in MB; auto-increased based on source content (default: 256)')
    parser.add_argument('--lazy', action='store_true',
                        help='Enable lazy allocation for large files (saves disk space)')
    parser.add_argument('--dealloc-timeout', type=float, default=60.0,
                        help='Seconds after last read before deallocating (default: 60)')
    parser.add_argument('--partitioned', action='store_true',
                        help='Add MBR partition table (required for Windows VM)')
    parser.add_argument('--virtual', action='store_true',
                        help='Enable virtual file mode for live ext4→NTFS sync '
                             '(synthesizes NTFS entries on-the-fly, no ntfs-3g mount needed)')
    parser.add_argument('--overflow-dir',
                        help='Directory for root-level files created by Windows '
                             '(e.g. System Volume Information). Should be on the data disk. '
                             'Default: source directory')
    parser.add_argument('--exclude', action='append', metavar='PATTERN', dest='exclude',
                        help='Exclude files/dirs matching this glob pattern (can repeat). '
                             'Matched against filename and relative path. '
                             'Example: --exclude "*.raw" --exclude "bridge-test"')

    args = parser.parse_args()

    bridge = NTFSBridge(
        image_path=args.image,
        source_dir=args.source,
        ntfs_mount=args.mount,
        port=args.port,
        image_size_mb=args.size,
        lazy_alloc=args.lazy,
        dealloc_timeout=args.dealloc_timeout,
        partitioned=args.partitioned,
        virtual_mode=args.virtual,
        overflow_dir=args.overflow_dir,
        exclude_patterns=args.exclude,
    )

    # Handle signals
    def signal_handler(sig, frame):
        log("Signal received, stopping...")
        bridge.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bridge.setup()
    bridge.run()


if __name__ == '__main__':
    main()
