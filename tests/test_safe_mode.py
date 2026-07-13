"""Unit tests for --safe-mode write protection.

Safe mode makes every pre-existing ext4 file and directory strictly read-only
at the bridge: only objects Windows creates this session (tracked in
_windows_created_sources) stay writable, and because every existing directory
is read-only, new content can only attach at the volume root. This guarantees
existing ext4 data can never be corrupted.

Pure logic — binds the real protection predicates to a fake object carrying
just the mapping state they touch, so no NTFS image or VM is needed.

Run: python -m pytest tests/test_safe_mode.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import ClusterMapper


SRC = '/root/bridge-source'


class FakeMapper:
    """Minimal stand-in carrying only what the protection predicates touch."""

    def __init__(self, safe_mode=True, protected_top_dirs=None):
        self.source_dir = SRC
        self.overflow_dir = SRC
        self.known_root_entries = {'Share', 'Docs'}
        self._safe_mode = safe_mode
        self._windows_created_sources = set()
        self._protected_top_dirs = protected_top_dirs or set()
        self.mft_record_to_source = {}
        self.mft_record_to_dir = {}

    # Reuse the real logic under test.
    _is_source_protected = ClusterMapper._is_source_protected
    _is_record_protected = ClusterMapper._is_record_protected
    _resolve_source_path = ClusterMapper._resolve_source_path
    _get_rel_path = ClusterMapper._get_rel_path


def s(*parts):
    return os.path.join(SRC, *parts)


# --- _is_source_protected (data-cluster writes) ---------------------------

def test_existing_source_is_protected():
    m = FakeMapper()
    # Any pre-existing ext4 file (not created by Windows) is read-only.
    assert m._is_source_protected(s('Share', 'report.pdf')) is True
    assert m._is_source_protected(s('root-level-existing.txt')) is True


def test_windows_created_source_is_writable():
    m = FakeMapper()
    new = s('dropped-from-windows.txt')
    m._windows_created_sources.add(new)
    assert m._is_source_protected(new) is False
    # A different (pre-existing) file remains protected.
    assert m._is_source_protected(s('Share', 'report.pdf')) is True


# --- _is_record_protected (MFT record writes) -----------------------------

def test_root_record_never_protected():
    # Root dir (record 5) must accept new entries so Windows can create files.
    m = FakeMapper()
    m.mft_record_to_dir[5] = ''
    assert m._is_record_protected(5) is False


def test_existing_file_record_protected():
    m = FakeMapper()
    m.mft_record_to_source[42] = s('Share', 'report.pdf')
    assert m._is_record_protected(42) is True


def test_windows_created_file_record_writable():
    m = FakeMapper()
    new = s('note.txt')
    m.mft_record_to_source[43] = new
    m._windows_created_sources.add(new)
    assert m._is_record_protected(43) is False


def test_existing_directory_record_protected():
    # An existing subdirectory is read-only, so Windows cannot insert entries
    # into it (new content can only land at the root).
    m = FakeMapper()
    m.mft_record_to_dir[7] = 'Share'
    assert m._is_record_protected(7) is True


def test_windows_created_directory_record_writable():
    m = FakeMapper()
    m.mft_record_to_dir[8] = 'WinNewDir'
    m._windows_created_sources.add(m._resolve_source_path('WinNewDir'))
    assert m._is_record_protected(8) is False


def test_free_record_writable():
    # A record Windows is using for a brand-new file isn't tracked yet; it must
    # be writable so the create can proceed (it becomes Windows-created once
    # _check_new_file materializes it).
    m = FakeMapper()
    assert m._is_record_protected(9999) is False


# --- safe mode OFF: falls back to protected-roots behavior ----------------

def test_safe_mode_off_uses_protected_top_dirs():
    m = FakeMapper(safe_mode=False, protected_top_dirs={'share'})
    # Only files under a protected top dir are read-only; everything else
    # (including an untracked existing file) is writable.
    assert m._is_source_protected(s('Share', 'report.pdf')) is True
    assert m._is_source_protected(s('Other', 'file.bin')) is False
    m.mft_record_to_source[42] = s('Share', 'report.pdf')
    assert m._is_record_protected(42) is True
    m.mft_record_to_source[43] = s('Other', 'file.bin')
    assert m._is_record_protected(43) is False


def test_safe_mode_off_no_protection_when_unset():
    m = FakeMapper(safe_mode=False)
    assert m._is_source_protected(s('anything.bin')) is False
    m.mft_record_to_source[1] = s('anything.bin')
    assert m._is_record_protected(1) is False
