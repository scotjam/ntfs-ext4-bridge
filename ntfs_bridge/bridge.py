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
                 virtual_mode: bool = False):
        self.image_path = os.path.abspath(image_path)
        self.source_dir = os.path.abspath(source_dir)
        self.ntfs_mount = os.path.abspath(ntfs_mount)
        self.port = port
        self.image_size_mb = image_size_mb
        self.lazy_alloc = lazy_alloc
        self.dealloc_timeout = dealloc_timeout
        self.partitioned = partitioned
        self.virtual_mode = virtual_mode

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

        # Calculate image size: at least 2x source size, minimum 64MB
        source_size = self._get_dir_size(self.source_dir)
        min_size_mb = max(64, (source_size * 3) // (1024 * 1024))
        if self.image_size_mb < min_size_mb:
            self.image_size_mb = min_size_mb
            log(f"Adjusted image size to {self.image_size_mb}MB (source is {source_size // 1024 // 1024}MB)")

        # Step 1: Create NTFS image if it doesn't exist
        if not os.path.exists(self.image_path):
            self._create_ntfs_image()
        else:
            log(f"Using existing image: {self.image_path}")

        # Step 2: Populate image from ext4 source
        self._populate_image()

        # Step 3: Initialize ClusterMapper
        log("Initializing ClusterMapper...")
        self.mapper = ClusterMapper(self.image_path, self.source_dir)

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
            sparse_files = list(self.mapper.sparse_files.keys())
            if sparse_files:
                log(f"Pre-allocating {len(sparse_files)} sparse files...")
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
                # Start sync daemon for ntfs-3g based sync
                if not self.virtual_mode:
                    self.sync_daemon = SyncDaemon(
                        self.source_dir, self.ntfs_mount, self.mapper,
                        lazy_allocator=self.lazy_allocator
                    )
                    self.sync_daemon.start()
                    log("Sync daemon started")

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
            ['mkfs.ntfs', '-F', '-Q', self.image_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log(f"ERROR: mkfs.ntfs failed: {result.stderr}")
            sys.exit(1)

        log("NTFS image created")

    def _populate_image(self):
        """Populate the NTFS image with files from ext4 source."""
        # Mount image directly and copy directory structure + sparse files
        tmp_mount = tempfile.mkdtemp(prefix='ntfs-init-')

        try:
            # Mount the image with ntfs-3g
            result = subprocess.run(
                ['mount', '-t', 'ntfs-3g', '-o', 'rw,big_writes',
                 self.image_path, tmp_mount],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                log(f"WARNING: Could not mount image for population: {result.stderr}")
                log("Image may need manual population")
                return

            log("Populating NTFS image from ext4 source...")
            files_created = 0
            dirs_created = 0

            for root, dirs, files in os.walk(self.source_dir):
                rel_root = os.path.relpath(root, self.source_dir)

                # Create directories
                for d in dirs:
                    if d.startswith('.'):
                        continue
                    rel_dir = os.path.join(rel_root, d) if rel_root != '.' else d
                    ntfs_dir = os.path.join(tmp_mount, rel_dir)
                    try:
                        os.makedirs(ntfs_dir, exist_ok=True)
                        dirs_created += 1
                    except OSError as e:
                        log(f"  Warning: could not create dir {rel_dir}: {e}")

                # Create files - sparse for large files if lazy_alloc enabled
                for f in files:
                    if f.startswith('.'):
                        continue
                    rel_file = os.path.join(rel_root, f) if rel_root != '.' else f
                    source_file = os.path.join(root, f)
                    ntfs_file = os.path.join(tmp_mount, rel_file)

                    try:
                        file_size = os.path.getsize(source_file)

                        if self.lazy_alloc and file_size > 700:
                            # Large file with lazy alloc - create sparse
                            with open(ntfs_file, 'wb') as nf:
                                if file_size > 0:
                                    nf.seek(file_size - 1)
                                    nf.write(b'\x00')
                        else:
                            # Small file or no lazy alloc - copy content
                            shutil.copy2(source_file, ntfs_file)
                        files_created += 1
                    except OSError as e:
                        log(f"  Warning: could not create file {rel_file}: {e}")

            log(f"Populated: {dirs_created} dirs, {files_created} files")

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
            ['mount', '-t', 'ntfs-3g', '-o', 'rw,big_writes',
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

    @staticmethod
    def _get_dir_size(path: str) -> int:
        """Get total size of files in a directory."""
        total = 0
        for root, dirs, files in os.walk(path):
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
                        help='Image size in MB (default: 256)')
    parser.add_argument('--lazy', action='store_true',
                        help='Enable lazy allocation for large files (saves disk space)')
    parser.add_argument('--dealloc-timeout', type=float, default=60.0,
                        help='Seconds after last read before deallocating (default: 60)')
    parser.add_argument('--partitioned', action='store_true',
                        help='Add MBR partition table (required for Windows VM)')
    parser.add_argument('--virtual', action='store_true',
                        help='Enable virtual file mode for live ext4→NTFS sync '
                             '(synthesizes NTFS entries on-the-fly, no ntfs-3g mount needed)')

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
        virtual_mode=args.virtual
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
