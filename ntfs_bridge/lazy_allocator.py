"""Lazy allocation manager for NTFS-ext4 bridge.

Manages on-demand allocation and deallocation of large files to minimize
disk space usage. Files start as sparse (metadata only), get allocated
when first read, and are deallocated after a timeout period of no access.

This allows backing up hundreds of GB of files while only using disk space
for one file at a time.
"""
import os
import shutil
import subprocess
import threading
import time
from typing import Dict, Set, Optional, Callable


def log(msg):
    print(f"[LazyAlloc] {msg}", flush=True)


# File states
STATE_SPARSE = 'sparse'      # No clusters allocated, reads return zeros
STATE_ALLOCATING = 'allocating'  # Currently being allocated
STATE_ALLOCATED = 'allocated'    # Clusters allocated and mapped
STATE_DEALLOCATING = 'deallocating'  # Currently being deallocated


class LazyAllocator:
    """Manages lazy allocation/deallocation of large files.

    Large files (>700 bytes, non-resident) start as sparse entries in NTFS.
    When a read is detected for a sparse file, we:
    1. Write the file content through ntfs-3g to allocate clusters
    2. Rescan MFT to map the new clusters to ext4
    3. Serve reads from ext4

    After a period of no reads, we deallocate by:
    1. Truncating the file to 0 through ntfs-3g
    2. Restoring the file size (sparse again)
    3. Removing cluster mappings
    """

    # Threshold for "large" files that use lazy allocation
    LARGE_FILE_THRESHOLD = 700  # bytes - files larger than this are non-resident

    # Time to wait after last read before deallocating
    DEALLOC_TIMEOUT = 60.0  # seconds

    # How often to check for files to deallocate
    DEALLOC_CHECK_INTERVAL = 10.0  # seconds

    def __init__(self, source_dir: str, ntfs_mount: str, cluster_mapper,
                 on_allocated: Optional[Callable[[str], None]] = None,
                 on_deallocated: Optional[Callable[[str], None]] = None):
        """
        Args:
            source_dir: Path to ext4 source directory
            ntfs_mount: Path where ntfs-3g mounts the NTFS volume
            cluster_mapper: ClusterMapper instance for rescanning
            on_allocated: Callback when a file is allocated (for logging)
            on_deallocated: Callback when a file is deallocated (for logging)
        """
        self.source_dir = os.path.abspath(source_dir)
        self.ntfs_mount = os.path.abspath(ntfs_mount)
        self.mapper = cluster_mapper
        self.on_allocated = on_allocated
        self.on_deallocated = on_deallocated

        # Track file states: rel_path -> state
        self.file_states: Dict[str, str] = {}

        # Track last read time for allocated files: rel_path -> timestamp
        self.last_read_time: Dict[str, float] = {}

        # Files currently being allocated (to avoid double-allocation)
        self.allocating_lock = threading.Lock()

        # Lock for state changes
        self.state_lock = threading.RLock()

        # Background deallocation thread
        self._running = False
        self._dealloc_thread: Optional[threading.Thread] = None

        # Track which source paths are "large" and eligible for lazy alloc
        self._large_files: Set[str] = set()

    def start(self):
        """Start the background deallocation thread."""
        if self._running:
            return
        self._running = True
        self._dealloc_thread = threading.Thread(
            target=self._dealloc_loop,
            daemon=True,
            name="LazyAlloc-Dealloc"
        )
        self._dealloc_thread.start()
        log("Started deallocation thread")

    def stop(self):
        """Stop the background thread."""
        self._running = False
        if self._dealloc_thread:
            self._dealloc_thread.join(timeout=5.0)
            self._dealloc_thread = None
        log("Stopped")

    def register_file(self, rel_path: str, source_path: str, is_allocated: bool):
        """Register a file for lazy allocation tracking.

        Args:
            rel_path: Relative path from source_dir
            source_path: Absolute path to ext4 source file
            is_allocated: True if file already has clusters allocated
        """
        try:
            file_size = os.path.getsize(source_path)
        except OSError:
            return

        # Only track large files
        if file_size <= self.LARGE_FILE_THRESHOLD:
            return

        with self.state_lock:
            self._large_files.add(rel_path)
            if is_allocated:
                self.file_states[rel_path] = STATE_ALLOCATED
                self.last_read_time[rel_path] = time.time()
            else:
                self.file_states[rel_path] = STATE_SPARSE

    def unregister_file(self, rel_path: str):
        """Remove a file from tracking (e.g., when deleted)."""
        with self.state_lock:
            self._large_files.discard(rel_path)
            self.file_states.pop(rel_path, None)
            self.last_read_time.pop(rel_path, None)

    def is_large_file(self, rel_path: str) -> bool:
        """Check if a file is tracked as a large file."""
        return rel_path in self._large_files

    def get_state(self, rel_path: str) -> Optional[str]:
        """Get the current state of a file."""
        return self.file_states.get(rel_path)

    def is_allocated(self, rel_path: str) -> bool:
        """Check if a file is currently allocated."""
        return self.file_states.get(rel_path) == STATE_ALLOCATED

    def record_read(self, rel_path: str):
        """Record that a file was read (updates deallocation timer)."""
        with self.state_lock:
            if rel_path in self.file_states:
                self.last_read_time[rel_path] = time.time()

    def needs_allocation(self, rel_path: str) -> bool:
        """Check if a file needs to be allocated before reading."""
        state = self.file_states.get(rel_path)
        return state == STATE_SPARSE

    def allocate_file(self, rel_path: str) -> bool:
        """Allocate clusters for a sparse file.

        Writes the file content through ntfs-3g to force cluster allocation,
        then triggers an MFT rescan to map the clusters.

        Returns True if allocation succeeded.
        """
        with self.allocating_lock:
            with self.state_lock:
                state = self.file_states.get(rel_path)
                if state == STATE_ALLOCATED:
                    return True  # Already allocated
                if state == STATE_ALLOCATING:
                    return False  # In progress
                if state != STATE_SPARSE:
                    return False  # Unknown state
                self.file_states[rel_path] = STATE_ALLOCATING

            source_path = os.path.join(self.source_dir, rel_path)
            ntfs_path = os.path.join(self.ntfs_mount, rel_path)

            try:
                if not os.path.isfile(source_path):
                    log(f"  Source file not found: {rel_path}")
                    with self.state_lock:
                        self.file_states[rel_path] = STATE_SPARSE
                    return False

                file_size = os.path.getsize(source_path)
                log(f"  Allocating: {rel_path} ({file_size / 1024 / 1024:.1f} MB)")

                # Copy content through ntfs-3g to allocate clusters
                # This is the expensive operation but only happens once per read session
                shutil.copy2(source_path, ntfs_path)

                # Sync to ensure data is flushed
                subprocess.run(['sync'], capture_output=True)

                # Rescan MFT to pick up new cluster mappings
                self.mapper.rescan_mft()

                with self.state_lock:
                    self.file_states[rel_path] = STATE_ALLOCATED
                    self.last_read_time[rel_path] = time.time()

                log(f"  Allocated: {rel_path}")
                if self.on_allocated:
                    self.on_allocated(rel_path)
                return True

            except Exception as e:
                log(f"  Allocation failed for {rel_path}: {e}")
                with self.state_lock:
                    self.file_states[rel_path] = STATE_SPARSE
                return False

    def deallocate_file(self, rel_path: str) -> bool:
        """Deallocate clusters for an allocated file.

        Uses direct NTFS image manipulation if the file was directly allocated,
        otherwise falls back to ntfs-3g (truncate + restore).

        Returns True if deallocation succeeded.
        """
        with self.state_lock:
            state = self.file_states.get(rel_path)
            if state != STATE_ALLOCATED:
                return False
            self.file_states[rel_path] = STATE_DEALLOCATING

        source_path = os.path.join(self.source_dir, rel_path)

        try:
            if not os.path.isfile(source_path):
                with self.state_lock:
                    self.file_states.pop(rel_path, None)
                return False

            log(f"  Deallocating: {rel_path}")

            # Try direct deallocation first (for directly-allocated files)
            if hasattr(self.mapper, 'deallocate_file_direct'):
                success = self.mapper.deallocate_file_direct(rel_path)
                if success:
                    with self.state_lock:
                        self.file_states[rel_path] = STATE_SPARSE
                        self.last_read_time.pop(rel_path, None)
                    log(f"  Deallocated: {rel_path}")
                    if self.on_deallocated:
                        self.on_deallocated(rel_path)
                    return True

            # Fall back to ntfs-3g method for files allocated through ntfs-3g
            ntfs_path = os.path.join(self.ntfs_mount, rel_path)
            file_size = os.path.getsize(source_path)

            # Mark as ext4 sync in progress to prevent sync daemon interference
            self.mapper.ext4_sync_in_progress.add(rel_path)
            try:
                # Truncate to 0 to free clusters
                with open(ntfs_path, 'wb') as f:
                    pass  # Empty file

                # Restore size as sparse
                with open(ntfs_path, 'r+b') as f:
                    if file_size > 0:
                        f.seek(file_size - 1)
                        f.write(b'\x00')

                subprocess.run(['sync'], capture_output=True)

                # Rescan MFT - clusters will no longer be mapped
                self.mapper.rescan_mft()

            finally:
                self.mapper.ext4_sync_in_progress.discard(rel_path)

            with self.state_lock:
                self.file_states[rel_path] = STATE_SPARSE
                self.last_read_time.pop(rel_path, None)

            log(f"  Deallocated: {rel_path}")
            if self.on_deallocated:
                self.on_deallocated(rel_path)
            return True

        except Exception as e:
            log(f"  Deallocation failed for {rel_path}: {e}")
            with self.state_lock:
                self.file_states[rel_path] = STATE_ALLOCATED
            return False

    def _dealloc_loop(self):
        """Background loop to deallocate files after timeout."""
        while self._running:
            time.sleep(self.DEALLOC_CHECK_INTERVAL)

            if not self._running:
                break

            # Find files to deallocate
            now = time.time()
            to_dealloc = []

            with self.state_lock:
                for rel_path, state in list(self.file_states.items()):
                    if state != STATE_ALLOCATED:
                        continue
                    last_read = self.last_read_time.get(rel_path, 0)
                    if now - last_read > self.DEALLOC_TIMEOUT:
                        to_dealloc.append(rel_path)

            # Deallocate outside the lock
            for rel_path in to_dealloc:
                self.deallocate_file(rel_path)

    def create_sparse_file(self, rel_path: str, source_path: str) -> bool:
        """Create a sparse file entry in NTFS (metadata only, no data).

        Used by SyncDaemon when a new large file is detected in ext4.
        Creates the file with correct size but no allocated clusters.
        """
        ntfs_path = os.path.join(self.ntfs_mount, rel_path)

        try:
            file_size = os.path.getsize(source_path)

            if file_size <= self.LARGE_FILE_THRESHOLD:
                # Small file - copy normally (will be resident)
                return False  # Let caller handle normally

            # Ensure parent directory exists
            parent = os.path.dirname(ntfs_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)

            # Create sparse file (no data copy)
            with open(ntfs_path, 'wb') as f:
                if file_size > 0:
                    f.seek(file_size - 1)
                    f.write(b'\x00')

            subprocess.run(['sync'], capture_output=True)

            # Register for lazy allocation
            self.register_file(rel_path, source_path, is_allocated=False)

            log(f"  Created sparse: {rel_path} ({file_size / 1024 / 1024:.1f} MB)")
            return True

        except Exception as e:
            log(f"  Failed to create sparse file {rel_path}: {e}")
            return False
