"""Sync daemon for ext4 → NTFS structural synchronization.

Watches the ext4 source directory and mirrors structural changes
(create, delete, rename) to the NTFS mount via ntfs-3g.
Content changes are handled transparently by the ClusterMapper
(reads go directly to ext4 files).

For large files, uses lazy allocation - files are created as sparse
and only allocated when first read.
"""
import os
import shutil
import subprocess
import threading
import time
from typing import Optional, TYPE_CHECKING

from .file_watcher import create_watcher, EVENT_CREATE, EVENT_DELETE, EVENT_MODIFY

if TYPE_CHECKING:
    from .lazy_allocator import LazyAllocator


def log(msg):
    print(f"[SyncDaemon] {msg}", flush=True)


# Threshold for large files that use lazy allocation (bytes)
LARGE_FILE_THRESHOLD = 700


class SyncDaemon:
    """Syncs ext4 source directory changes to NTFS mount.

    Structural changes (create, delete, rename) are performed through
    the ntfs-3g mount point so ntfs-3g handles all NTFS metadata.
    Content changes need no sync - the ClusterMapper reads directly
    from ext4 files.

    Large files (>700 bytes) use lazy allocation - created as sparse
    files and only allocated when first read.
    """

    def __init__(self, source_dir: str, ntfs_mount: str, cluster_mapper,
                 lazy_allocator: Optional['LazyAllocator'] = None):
        """
        Args:
            source_dir: Path to ext4 source directory
            ntfs_mount: Path where ntfs-3g mounts the NTFS volume
            cluster_mapper: ClusterMapper instance for loop prevention and rescanning
            lazy_allocator: Optional LazyAllocator for large file handling
        """
        self.source_dir = os.path.abspath(source_dir)
        self.ntfs_mount = os.path.abspath(ntfs_mount)
        self.mapper = cluster_mapper
        self.lazy_allocator = lazy_allocator
        self._watcher = None
        self._running = False

        # Debounce: track recently synced paths to avoid rapid re-sync
        self._recent_syncs: dict = {}  # path -> timestamp
        self._recent_lock = threading.Lock()
        self.DEBOUNCE_SECONDS = 2.0

    def start(self):
        """Start watching and syncing."""
        if self._running:
            return

        self._running = True
        self._watcher = create_watcher(self.source_dir, self._on_event)
        self._watcher.start()
        log(f"Started: source={self.source_dir}, mount={self.ntfs_mount}")

    def stop(self):
        """Stop watching."""
        self._running = False
        if self._watcher:
            self._watcher.stop()
            self._watcher = None
        log("Stopped")

    def _on_event(self, event_type: str, rel_path: str):
        """Handle a file system event from the watcher."""
        # Skip hidden/system files
        basename = os.path.basename(rel_path)
        if basename.startswith('.') or basename.startswith('$'):
            return

        # Check loop prevention: skip if NTFS→ext4 sync created this
        if rel_path in self.mapper.ntfs_sync_in_progress:
            log(f"  Skipping (NTFS sync in progress): {rel_path}")
            return

        # Debounce check
        with self._recent_lock:
            last_sync = self._recent_syncs.get(rel_path, 0)
            if time.time() - last_sync < self.DEBOUNCE_SECONDS:
                return

        if event_type == EVENT_CREATE:
            self._handle_create(rel_path)
        elif event_type == EVENT_DELETE:
            self._handle_delete(rel_path)
        elif event_type == EVENT_MODIFY:
            # Content modifications need no sync - ClusterMapper reads
            # directly from ext4 files. But we should drop caches.
            self._handle_modify(rel_path)

    def _handle_create(self, rel_path: str):
        """Handle file/directory creation in ext4."""
        source_path = os.path.join(self.source_dir, rel_path)
        ntfs_path = os.path.join(self.ntfs_mount, rel_path)

        if not os.path.exists(source_path):
            return

        if os.path.exists(ntfs_path):
            # File exists - treat as modify (content or size change)
            self._handle_modify(rel_path)
            return

        # Mark as ext4 sync in progress (prevents MFT tracker from
        # creating the file again in ext4)
        self.mapper.ext4_sync_in_progress.add(rel_path)
        try:
            if os.path.isdir(source_path):
                self._sync_create_dir(rel_path, source_path, ntfs_path)
            else:
                self._sync_create_file(rel_path, source_path, ntfs_path)

            # Record sync timestamp
            with self._recent_lock:
                self._recent_syncs[rel_path] = time.time()

            # Rescan MFT to pick up new cluster mappings
            self.mapper.rescan_mft()

        except Exception as e:
            log(f"  Create sync failed for {rel_path}: {e}")
        finally:
            self.mapper.ext4_sync_in_progress.discard(rel_path)

    def _sync_create_dir(self, rel_path: str, source_path: str, ntfs_path: str):
        """Create a directory in the NTFS mount."""
        # Ensure parent exists
        parent = os.path.dirname(ntfs_path)
        if parent and not os.path.exists(parent):
            parent_rel = os.path.dirname(rel_path)
            parent_source = os.path.join(self.source_dir, parent_rel)
            self._sync_create_dir(parent_rel, parent_source, parent)

        try:
            os.makedirs(ntfs_path, exist_ok=True)
            log(f"  DIR created in NTFS: {rel_path}")
        except OSError as e:
            log(f"  Failed to create dir in NTFS: {rel_path}: {e}")

    def _sync_create_file(self, rel_path: str, source_path: str, ntfs_path: str):
        """Create a file in the NTFS mount.

        For large files (>700 bytes) with lazy allocation enabled:
        - Create as sparse (metadata only, no data copy)
        - File will be allocated on first read

        For small files or without lazy allocation:
        - Copy actual content so NTFS allocates real clusters
        """
        # Ensure parent directory exists in NTFS mount
        parent = os.path.dirname(ntfs_path)
        if parent and not os.path.exists(parent):
            parent_rel = os.path.dirname(rel_path)
            parent_source = os.path.join(self.source_dir, parent_rel)
            if os.path.isdir(parent_source):
                self._sync_create_dir(parent_rel, parent_source, parent)

        try:
            file_size = os.path.getsize(source_path)

            # Use lazy allocation for large files if available
            if self.lazy_allocator and file_size > LARGE_FILE_THRESHOLD:
                # Create as sparse file (will be allocated on first read)
                if self.lazy_allocator.create_sparse_file(rel_path, source_path):
                    subprocess.run(['sync'], capture_output=True)
                    return
                # Fall through to full copy if sparse creation failed

            # Small file or no lazy allocator - copy actual content
            shutil.copy2(source_path, ntfs_path)

            log(f"  FILE created in NTFS: {rel_path} ({file_size} bytes)")
            subprocess.run(['sync'], capture_output=True)
        except OSError as e:
            log(f"  Failed to create file in NTFS: {rel_path}: {e}")

    def _handle_delete(self, rel_path: str):
        """Handle file/directory deletion in ext4."""
        ntfs_path = os.path.join(self.ntfs_mount, rel_path)

        if not os.path.exists(ntfs_path):
            return

        self.mapper.ext4_sync_in_progress.add(rel_path)
        try:
            if os.path.isdir(ntfs_path):
                shutil.rmtree(ntfs_path, ignore_errors=True)
                log(f"  DIR deleted from NTFS: {rel_path}")
            else:
                os.remove(ntfs_path)
                log(f"  FILE deleted from NTFS: {rel_path}")

            with self._recent_lock:
                self._recent_syncs[rel_path] = time.time()

            self.mapper.rescan_mft()

        except OSError as e:
            log(f"  Delete sync failed for {rel_path}: {e}")
        finally:
            self.mapper.ext4_sync_in_progress.discard(rel_path)

    def _handle_modify(self, rel_path: str):
        """Handle content modification in ext4.

        For non-resident files, content is served transparently through
        ClusterMapper (reads from ext4 directly). We just need to update
        the NTFS file size if it changed.

        For resident files (small, data stored in MFT), we also need to
        write content through ntfs-3g so the MFT record stays in sync.
        The ClusterMapper injects ext4 content on reads, but the MFT
        record's value_length needs updating for size changes.
        """
        source_path = os.path.join(self.source_dir, rel_path)
        ntfs_path = os.path.join(self.ntfs_mount, rel_path)

        if not os.path.isfile(source_path) or not os.path.isfile(ntfs_path):
            return

        try:
            source_size = os.path.getsize(source_path)
            ntfs_size = os.path.getsize(ntfs_path)

            # For small/resident files, always sync content through ntfs-3g
            # so MFT record and ntfs-3g's cache stay current.
            # For large files, only sync if size changed (content served from ext4).
            needs_sync = False
            if source_size <= 700:
                # Small file likely resident - always sync content
                needs_sync = True
            elif source_size != ntfs_size:
                needs_sync = True

            if needs_sync:
                self.mapper.ext4_sync_in_progress.add(rel_path)
                try:
                    if source_size <= 700:
                        with open(source_path, 'rb') as src:
                            content = src.read()
                        with open(ntfs_path, 'wb') as dst:
                            dst.write(content)
                        log(f"  FILE content synced to NTFS: {rel_path} ({source_size} bytes)")
                    else:
                        with open(ntfs_path, 'r+b') as f:
                            f.truncate(source_size)
                        log(f"  FILE resized in NTFS: {rel_path} ({ntfs_size} -> {source_size})")
                    self.mapper.rescan_mft()
                finally:
                    self.mapper.ext4_sync_in_progress.discard(rel_path)
        except OSError:
            pass  # File may have been deleted

    def on_ext4_rename(self, old_path: str, new_path: str):
        """Handle a file/directory rename in ext4.

        Called externally when a rename is detected (inotify MOVED_FROM + MOVED_TO).
        """
        old_ntfs = os.path.join(self.ntfs_mount, old_path)
        new_ntfs = os.path.join(self.ntfs_mount, new_path)

        if not os.path.exists(old_ntfs):
            return

        if os.path.exists(new_ntfs):
            log(f"  Rename target already exists in NTFS: {new_path}")
            return

        self.mapper.ext4_sync_in_progress.add(old_path)
        self.mapper.ext4_sync_in_progress.add(new_path)
        try:
            # Ensure parent exists
            parent = os.path.dirname(new_ntfs)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)

            os.rename(old_ntfs, new_ntfs)
            log(f"  RENAMED in NTFS: {old_path} -> {new_path}")

            with self._recent_lock:
                self._recent_syncs[old_path] = time.time()
                self._recent_syncs[new_path] = time.time()

            self.mapper.rescan_mft()

        except OSError as e:
            log(f"  Rename sync failed: {old_path} -> {new_path}: {e}")
        finally:
            self.mapper.ext4_sync_in_progress.discard(old_path)
            self.mapper.ext4_sync_in_progress.discard(new_path)
