"""An allocated cluster the bridge cannot account for is an I/O error, not a hole.

The read path used to fall through to "read from image" for any cluster nothing
had mapped. For a cluster that $Bitmap says is IN USE but which the image has
never been populated with, that means handing the client a block of zeros and
calling it data.

ntfs-3g believed it. The live log from 2026-09-18 reads:

    ntfs_mst_post_read_fixup_warn: magic: 0x00000000 size: 4096 usa_ofs: 0
    Corrupt index block signature: vcn 24 inode 5810
    Index lookup failed, inode 5810: Input/output error

- an index block read back as zeros. Mounted with -o recover, ntfs-3g then
"repaired" what it saw, and those repairs flowed back through reconciliation
into ext4 as deletions and zero-filled rewrites. Fabricating the zeros is the
first domino; everything after it is downstream.

Free clusters still read as zeros, because there a hole is the truth.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import ClusterMapper  # noqa: E402

CS = 4096
NCLUSTERS = 64

ALLOC_EMPTY = 10     # allocated, image empty, not metadata  -> must EIO
FREE_EMPTY = 11      # free, image empty                      -> zeros are correct
ALLOC_DATA = 12      # allocated, image has real bytes        -> serve them
META_EMPTY = 13      # allocated, image empty, but metadata   -> serve (may be zero)


class FakeImage(bytearray):
    def flush(self):
        pass

    def close(self):
        pass


def make_mapper():
    m = ClusterMapper.__new__(ClusterMapper)
    m.lock = threading.RLock()
    m.cluster_size = CS
    m.image = FakeImage(NCLUSTERS * CS)
    m._metadata_clusters = {META_EMPTY}
    m.virtual_file_manager = None
    m.virtualized_indx_clusters = {}
    m.virtual_indx_map = {}
    m.virtualized_dirs = {}
    m.cluster_map = {}
    m._direct_run_map = []
    m.lazy_allocator = None
    m.resident_file_data = {}
    m.sparse_files = {}
    m._unmapped_reported = set()
    m._check_sparse_file_read = lambda *a, **k: None

    bitmap = bytearray((NCLUSTERS + 7) // 8)
    for c in (ALLOC_EMPTY, ALLOC_DATA, META_EMPTY):
        bitmap[c // 8] |= 1 << (c % 8)
    m._bitmap_cache = bitmap

    m.image[ALLOC_DATA * CS:ALLOC_DATA * CS + 8] = b"INDX\x28\x00\x09\x00"
    return m


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    return cond


def read_cluster(m, c):
    return m._read_inner(c * CS, CS)


def main():
    ok = True
    m = make_mapper()

    print("\n[1] allocated in $Bitmap, unmapped, image empty")
    try:
        data = read_cluster(m, ALLOC_EMPTY)
        ok &= check("raised IOError instead of serving zeros", False,
                    "returned %d bytes, all-zero=%s" % (len(data), not any(data)))
    except IOError as e:
        ok &= check("raised IOError instead of serving zeros", True)
        ok &= check("message names the cluster", str(ALLOC_EMPTY) in str(e), str(e))

    print("\n[2] FREE in $Bitmap, image empty - a hole is the truth")
    try:
        data = read_cluster(m, FREE_EMPTY)
        ok &= check("served zeros without error", len(data) == CS and not any(data))
    except IOError as e:
        ok &= check("served zeros without error", False, "raised %s" % e)

    print("\n[3] allocated with real bytes in the image - served as before")
    try:
        data = read_cluster(m, ALLOC_DATA)
        ok &= check("content served unchanged", data[:4] == b"INDX", repr(data[:4]))
    except IOError as e:
        ok &= check("content served unchanged", False, "raised %s" % e)

    print("\n[4] metadata cluster, allocated and empty - still served")
    try:
        data = read_cluster(m, META_EMPTY)
        ok &= check("metadata may legitimately be zero", len(data) == CS and not any(data))
    except IOError as e:
        ok &= check("metadata may legitimately be zero", False, "raised %s" % e)

    print("\n[5] no bitmap loaded yet - unknown must never escalate")
    m2 = make_mapper()
    m2._bitmap_cache = None
    ok &= check("_cluster_is_allocated False without a cache",
                m2._cluster_is_allocated(ALLOC_EMPTY) is False)
    try:
        data = read_cluster(m2, ALLOC_EMPTY)
        ok &= check("read still succeeds during startup", len(data) == CS)
    except IOError as e:
        ok &= check("read still succeeds during startup", False, "raised %s" % e)

    print("\n[6] the failure is reported once per cluster")
    m3 = make_mapper()
    for _ in range(4):
        try:
            read_cluster(m3, ALLOC_EMPTY)
        except IOError:
            pass
    ok &= check("one report after four attempts", len(m3._unmapped_reported) == 1,
                "got %d" % len(m3._unmapped_reported))

    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
