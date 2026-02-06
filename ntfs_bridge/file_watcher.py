"""Inotify-based file watcher for ext4 -> NTFS synchronization.

Monitors a directory for file system changes and notifies the synthesizer
to update the NTFS view dynamically.
"""

import os
import threading
import time
from typing import Callable, Dict, Optional, Set
from collections import defaultdict

# Try to import inotify - will fail on Windows but that's OK for development
try:
    import inotify.adapters
    import inotify.constants
    INOTIFY_AVAILABLE = True
except ImportError:
    INOTIFY_AVAILABLE = False


def log(msg):
    """Debug logging."""
    print(f"[FileWatcher] {msg}", flush=True)


# Event types
EVENT_CREATE = 'create'
EVENT_DELETE = 'delete'
EVENT_MODIFY = 'modify'
EVENT_MOVED_FROM = 'moved_from'
EVENT_MOVED_TO = 'moved_to'


class FileWatcher:
    """Watches a directory tree for changes using inotify.

    Events are debounced to avoid rapid-fire callbacks during bulk operations.
    """

    DEBOUNCE_MS = 100  # Debounce window in milliseconds

    def __init__(self, watch_dir: str, callback: Callable[[str, str], None]):
        """
        Initialize the file watcher.

        Args:
            watch_dir: Root directory to watch (recursively)
            callback: Function called with (event_type, rel_path) on changes
        """
        self.watch_dir = os.path.abspath(watch_dir)
        self.callback = callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Debouncing: path -> (event_type, timestamp)
        self._pending_events: Dict[str, tuple] = {}
        self._debounce_thread: Optional[threading.Thread] = None

        # Track move cookies to pair IN_MOVED_FROM/TO events
        self._move_cookies: Dict[int, str] = {}

        if not INOTIFY_AVAILABLE:
            log("WARNING: inotify not available - file watching disabled")

    def start(self):
        """Start watching for file system events."""
        if not INOTIFY_AVAILABLE:
            log("Cannot start watcher - inotify not available")
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

        self._debounce_thread = threading.Thread(target=self._debounce_loop, daemon=True)
        self._debounce_thread.start()

        log(f"Started watching: {self.watch_dir}")

    def stop(self):
        """Stop watching for events."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._debounce_thread:
            self._debounce_thread.join(timeout=1.0)
            self._debounce_thread = None
        log("Stopped watching")

    def _watch_loop(self):
        """Main inotify event loop."""
        try:
            # Create recursive inotify watcher
            notifier = inotify.adapters.InotifyTree(self.watch_dir)

            # Watch for: create, delete, close_write (modify complete), move
            mask = (
                inotify.constants.IN_CREATE |
                inotify.constants.IN_DELETE |
                inotify.constants.IN_CLOSE_WRITE |
                inotify.constants.IN_MOVED_FROM |
                inotify.constants.IN_MOVED_TO |
                inotify.constants.IN_ISDIR
            )

            for event in notifier.event_gen(yield_nones=False):
                if not self._running:
                    break

                (_, type_names, path, filename) = event

                if not filename:
                    continue

                full_path = os.path.join(path, filename)
                rel_path = os.path.relpath(full_path, self.watch_dir)

                # Skip hidden files and system files
                if filename.startswith('.'):
                    continue

                self._handle_event(type_names, rel_path, event)

        except Exception as e:
            log(f"Watch loop error: {e}")

    def _handle_event(self, type_names: list, rel_path: str, raw_event):
        """Process a single inotify event."""
        event_type = None

        if 'IN_CREATE' in type_names:
            if 'IN_ISDIR' in type_names:
                event_type = EVENT_CREATE
                log(f"Directory created: {rel_path}")
            else:
                # For files, wait for IN_CLOSE_WRITE to get complete file
                return

        elif 'IN_DELETE' in type_names:
            event_type = EVENT_DELETE
            is_dir = 'IN_ISDIR' in type_names
            log(f"{'Directory' if is_dir else 'File'} deleted: {rel_path}")

        elif 'IN_CLOSE_WRITE' in type_names:
            # File write completed
            event_type = EVENT_CREATE  # Treat as create/modify
            log(f"File created/modified: {rel_path}")

        elif 'IN_MOVED_FROM' in type_names:
            # Start of a move operation
            cookie = raw_event[0].cookie if hasattr(raw_event[0], 'cookie') else 0
            self._move_cookies[cookie] = rel_path
            event_type = EVENT_DELETE  # Treat as delete from old location
            log(f"Move from: {rel_path}")

        elif 'IN_MOVED_TO' in type_names:
            # End of a move operation
            cookie = raw_event[0].cookie if hasattr(raw_event[0], 'cookie') else 0
            old_path = self._move_cookies.pop(cookie, None)
            event_type = EVENT_CREATE  # Treat as create at new location
            log(f"Move to: {rel_path} (from {old_path})")

        if event_type:
            self._queue_event(rel_path, event_type)

    def _queue_event(self, rel_path: str, event_type: str):
        """Queue an event for debouncing."""
        now = time.time()
        with self._lock:
            # If there's already a pending event for this path, update it
            # Priority: delete > create > modify
            existing = self._pending_events.get(rel_path)
            if existing:
                existing_type, _ = existing
                # Delete takes precedence
                if existing_type == EVENT_DELETE and event_type == EVENT_CREATE:
                    # Recreated - use create
                    pass
                elif event_type != EVENT_DELETE and existing_type == EVENT_DELETE:
                    # Keep delete
                    return

            self._pending_events[rel_path] = (event_type, now)

    def _debounce_loop(self):
        """Process debounced events."""
        while self._running:
            time.sleep(self.DEBOUNCE_MS / 1000.0)

            now = time.time()
            threshold = now - (self.DEBOUNCE_MS / 1000.0)

            events_to_fire = []

            with self._lock:
                paths_to_remove = []
                for path, (event_type, timestamp) in self._pending_events.items():
                    if timestamp <= threshold:
                        events_to_fire.append((path, event_type))
                        paths_to_remove.append(path)

                for path in paths_to_remove:
                    del self._pending_events[path]

            # Fire events outside the lock
            for path, event_type in events_to_fire:
                try:
                    self.callback(event_type, path)
                except Exception as e:
                    log(f"Callback error for {path}: {e}")


class PollingFileWatcher:
    """Fallback file watcher using polling (for when inotify is not available).

    This is less efficient but works on any platform.
    """

    POLL_INTERVAL = 1.0  # Seconds between polls

    def __init__(self, watch_dir: str, callback: Callable[[str, str], None]):
        """
        Initialize the polling watcher.

        Args:
            watch_dir: Root directory to watch (recursively)
            callback: Function called with (event_type, rel_path) on changes
        """
        self.watch_dir = os.path.abspath(watch_dir)
        self.callback = callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._known_files: Dict[str, float] = {}  # rel_path -> mtime
        self._lock = threading.Lock()

    def start(self):
        """Start polling for file system changes."""
        if self._running:
            return

        # Initial scan
        self._scan_directory()

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        log(f"Started polling: {self.watch_dir}")

    def stop(self):
        """Stop polling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        log("Stopped polling")

    def _scan_directory(self) -> Dict[str, float]:
        """Scan directory and return dict of rel_path -> mtime."""
        files = {}
        try:
            for root, dirs, filenames in os.walk(self.watch_dir):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]

                rel_root = os.path.relpath(root, self.watch_dir)
                if rel_root != '.':
                    # Track directories too
                    try:
                        files[rel_root] = os.path.getmtime(root)
                    except OSError:
                        pass

                for filename in filenames:
                    if filename.startswith('.'):
                        continue

                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, self.watch_dir)

                    try:
                        files[rel_path] = os.path.getmtime(full_path)
                    except OSError:
                        pass
        except OSError as e:
            log(f"Scan error: {e}")

        return files

    def _poll_loop(self):
        """Polling loop."""
        while self._running:
            time.sleep(self.POLL_INTERVAL)

            if not self._running:
                break

            current_files = self._scan_directory()

            with self._lock:
                old_files = self._known_files
                self._known_files = current_files

            # Find new files
            for path in current_files:
                if path not in old_files:
                    try:
                        self.callback(EVENT_CREATE, path)
                    except Exception as e:
                        log(f"Callback error for create {path}: {e}")
                elif current_files[path] > old_files[path]:
                    # Modified
                    try:
                        self.callback(EVENT_MODIFY, path)
                    except Exception as e:
                        log(f"Callback error for modify {path}: {e}")

            # Find deleted files
            for path in old_files:
                if path not in current_files:
                    try:
                        self.callback(EVENT_DELETE, path)
                    except Exception as e:
                        log(f"Callback error for delete {path}: {e}")


def create_watcher(watch_dir: str, callback: Callable[[str, str], None]) -> 'FileWatcher | PollingFileWatcher':
    """Create the best available file watcher for the platform.

    Returns an inotify-based watcher on Linux, or a polling watcher elsewhere.
    """
    if INOTIFY_AVAILABLE:
        return FileWatcher(watch_dir, callback)
    else:
        log("Using polling watcher (inotify not available)")
        return PollingFileWatcher(watch_dir, callback)
