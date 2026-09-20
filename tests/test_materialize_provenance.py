"""Materialisation may only use clusters a client actually wrote.

_check_new_file rebuilds an ext4 file by copying image bytes for the clusters in
the record's data runs. It never asked whether anything had put data in those
clusters. Started against a stale image, with the VM powered off, it copied the
image's own empty space over four intact files - each came back at
its exact original size in all-zero bytes.

Provenance fixes the question at its source: a bit per cluster, set on every
client write, and materialisation refuses any run containing a cluster nobody
wrote. Separately, a reused image is not a description of ext4 until the volume
has mounted cleanly, so reconciliation is held off entirely in that window.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import ClusterMapper  # noqa: E402

CS = 4096


class FakeImage(bytearray):
    def flush(self):
        pass

    def close(self):
        pass


def make_mapper(tmp="/tmp"):
    m = ClusterMapper.__new__(ClusterMapper)
    m.lock = threading.RLock()
    m.cluster_size = CS
    m.image = FakeImage(256 * CS)
    m.source_dir = tmp
    m.overflow_dir = tmp
    m._metadata_clusters = set()
    m.cluster_map = {}
    m._direct_run_map = []
    m._protected_top_dirs = set()
    # safe mode came in with the two-way branch; these harnesses build
    # mappers via __new__, so the attributes it reads must be set here too.
    m._safe_mode = False
    m._windows_created_sources = set()
    m._protect_refused = set()
    m._attempt_log = None
    m.record_only = False
    m._materialize_refused = set()
    m._guest_written = bytearray()
    m.ext4_authoritative = False
    m.lazy_allocator = None
    m._mft_runs = []
    m._bitmap_cache = None
    m.bitmap_clusters = []
    m.resident_file_data = {}
    m.sparse_files = {}
    m._unmapped_reported = set()
    m._mft_mirror_offset = -1
    m._mft_mirror_record_count = 0
    m._metadata_clusters = set()
    m.virtualized_indx_clusters = {}
    m.virtual_indx_map = {}
    m.virtualized_dirs = {}
    m.virtual_file_manager = None
    return m


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    return cond


def main():
    ok = True

    print("\n[1] provenance bitmap")
    m = make_mapper()
    ok &= check("unwritten cluster reads False", m._is_guest_written(7) is False)
    m._mark_guest_written(7)
    ok &= check("marked cluster reads True", m._is_guest_written(7) is True)
    ok &= check("neighbour untouched", m._is_guest_written(8) is False)
    m._mark_guest_written(900000)
    ok &= check("bitmap grows for a far cluster", m._is_guest_written(900000) is True)
    ok &= check("growth did not set anything else", m._is_guest_written(899999) is False)

    print("\n[2] _first_unwritten_cluster")
    m = make_mapper()
    runs = [(10, 3)]
    ok &= check("all unwritten -> reports the first", m._first_unwritten_cluster(runs) == 10)
    for c in (10, 11):
        m._mark_guest_written(c)
    ok &= check("partially written -> reports the gap", m._first_unwritten_cluster(runs) == 12)
    m._mark_guest_written(12)
    ok &= check("fully written -> None", m._first_unwritten_cluster(runs) is None)
    ok &= check("sparse run (-1) is a real hole, skipped",
                m._first_unwritten_cluster([(-1, 5)]) is None)

    print("\n[3] the write path records provenance")
    m = make_mapper()
    payload = b"\xab" * (CS * 2)
    m._write_inner(20 * CS, payload)
    ok &= check("both written clusters marked",
                m._is_guest_written(20) and m._is_guest_written(21))
    ok &= check("next cluster not marked", m._is_guest_written(22) is False)
    ok &= check("a run over those clusters now materialises",
                m._first_unwritten_cluster([(20, 2)]) is None)
    ok &= check("a run reaching past them does not",
                m._first_unwritten_cluster([(20, 3)]) == 22)

    print("\n[4] reused image: ext4 is authoritative until the volume mounts")
    m = make_mapper()
    p = os.path.join("/tmp", "Media", "x.mkv")
    ok &= check("nothing refused while the image is trusted",
                m._refuse_ext4_mutation(p, "delete", "Media/x.mkv") is False)
    m.ext4_authoritative = True
    ok &= check("refused while the image is stale",
                m._refuse_ext4_mutation(p, "delete", "Media/x.mkv") is True)
    ok &= check("refused even outside any protected root",
                m._refuse_ext4_mutation("/tmp/Scratch/y.mkv", "materialize") is True)
    m.ext4_authoritative = False
    ok &= check("allowed again once the volume mounts cleanly",
                m._refuse_ext4_mutation("/tmp/Scratch/y.mkv", "materialize") is False)

    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
