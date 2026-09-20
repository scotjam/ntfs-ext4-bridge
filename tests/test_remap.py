"""Unit tests for ClusterMapper.remap_source_path (file and directory
renames). Pure logic — binds the method to a fake object with the mapping
dicts, so no NTFS image or VM is needed.

Run: python -m pytest tests/test_remap.py -v
"""

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import ClusterMapper


class FakeMapper:
    """Minimal stand-in carrying just what remap_source_path touches."""

    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.overflow_dir = source_dir
        self.known_root_entries = {'Share'}
        self.lock = threading.RLock()
        self.source_to_clusters = {}
        self.cluster_map = {}
        self._direct_run_map = []
        self.mft_record_to_source = {}
        self.mft_record_to_dir = {}
        self.path_to_mft_record = {}
        self.sparse_files = {}
        self.sparse_file_clusters = {}
        self.resident_file_data = {}

    # Reuse the real resolution + remap logic.
    _resolve_source_path = ClusterMapper._resolve_source_path
    remap_source_path = ClusterMapper.remap_source_path


def src(*parts):
    return os.path.join('/root/bridge-source', *parts)


def test_file_rename_remaps_all_structures():
    m = FakeMapper('/root/bridge-source')
    old = os.path.join('Share', 'Docs', 'report.pdf')
    new = os.path.join('Share', 'Docs', 'report-renamed.pdf')
    m.source_to_clusters[src(old)] = {10, 11}
    m.cluster_map[10] = (src(old), 0)
    m._direct_run_map = [(100, 174, src(old), 0)]
    m.mft_record_to_source[42] = src(old)
    m.path_to_mft_record[old] = 42
    m.resident_file_data[42] = {'source_path': src(old)}

    m.remap_source_path(old, new)

    assert src(new) in m.source_to_clusters and src(old) not in m.source_to_clusters
    assert m.cluster_map[10] == (src(new), 0)
    assert m._direct_run_map[0][2] == src(new)
    assert m.mft_record_to_source[42] == src(new)
    assert m.path_to_mft_record.get(new) == 42 and old not in m.path_to_mft_record
    assert m.resident_file_data[42]['source_path'] == src(new)


def test_directory_rename_remaps_children():
    m = FakeMapper('/root/bridge-source')
    old_dir = os.path.join('Share', 'Videos')
    new_dir = os.path.join('Share', 'Clips')
    child_a = os.path.join(old_dir, 'a.mkv')
    child_b = os.path.join(old_dir, 'sub', 'b.avi')

    # The directory itself
    m.mft_record_to_dir[5] = old_dir
    m.path_to_mft_record[old_dir] = 5
    # A subdir
    m.mft_record_to_dir[6] = os.path.join(old_dir, 'sub')
    # Two child files (one run-mapped, one sparse)
    m._direct_run_map = [(200, 300, src(child_a), 0)]
    m.mft_record_to_source[7] = src(child_a)
    m.path_to_mft_record[child_a] = 7
    m.source_to_clusters[src(child_a)] = {200}
    m.cluster_map[200] = (src(child_a), 0)
    m.sparse_files[child_b] = (src(child_b), 999999, 8)
    m.sparse_file_clusters[500] = child_b
    m.mft_record_to_source[8] = src(child_b)
    m.path_to_mft_record[child_b] = 8

    m.remap_source_path(old_dir, new_dir)

    new_child_a = os.path.join(new_dir, 'a.mkv')
    new_child_b = os.path.join(new_dir, 'sub', 'b.avi')

    # Directory + subdir rel paths
    assert m.mft_record_to_dir[5] == new_dir
    assert m.mft_record_to_dir[6] == os.path.join(new_dir, 'sub')
    assert m.path_to_mft_record.get(new_dir) == 5
    # Child A (run-mapped)
    assert m._direct_run_map[0][2] == src(new_child_a)
    assert m.mft_record_to_source[7] == src(new_child_a)
    assert m.path_to_mft_record.get(new_child_a) == 7
    assert m.cluster_map[200] == (src(new_child_a), 0)
    assert src(new_child_a) in m.source_to_clusters
    # Child B (sparse, nested)
    assert m.sparse_files.get(new_child_b) == (src(new_child_b), 999999, 8)
    assert m.sparse_file_clusters[500] == new_child_b
    assert m.mft_record_to_source[8] == src(new_child_b)
    # Nothing left under the old paths
    assert not any(k == old_dir or k.startswith(old_dir + os.sep)
                   for k in m.path_to_mft_record)
    assert all(src(old_dir) not in e[2] for e in m._direct_run_map)


def test_unrelated_paths_untouched():
    m = FakeMapper('/root/bridge-source')
    keep = os.path.join('Share', 'Other', 'keep.bin')
    m.mft_record_to_source[9] = src(keep)
    m.path_to_mft_record[keep] = 9
    m._direct_run_map = [(1, 2, src(keep), 0)]
    m.remap_source_path(os.path.join('Share', 'Videos'),
                        os.path.join('Share', 'Clips'))
    assert m.mft_record_to_source[9] == src(keep)
    assert m.path_to_mft_record[keep] == 9
    assert m._direct_run_map[0][2] == src(keep)
