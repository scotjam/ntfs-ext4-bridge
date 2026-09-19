"""The read path answers only for clusters it can account for.

An unmapped cluster is served straight out of the image. Ownership decides
whether that is legitimate, not what the bytes happen to look like:

  metadata        the image is authoritative - NTFS structure lives there,
                  including every directory's $INDEX_ALLOCATION blocks
  guest-written   a client write to an unmapped cluster lands in the image,
                  so those bytes are the client's own
  free            a hole, and zeros are the truth
  anything else   file data ext4 owns and we have lost track of -> EIO

The first version of this guard tested the content instead, asking "is it all
zeros?". That caught the case behind the 2026-09-18 loss, where the image had
never been populated and ntfs-3g read an index block as

    ntfs_mst_post_read_fixup_warn: magic: 0x00000000 size: 4096 usa_ofs: 0

believed it, and "repaired" the volume into ext4. But a cluster left unmapped
mid-rescan or mid-reparse still holds its previous, plausible-looking bytes.
Serving those is the same silent corruption in better disguise, and a content
test cannot see it. Ownership can.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import ClusterMapper  # noqa: E402

CS = 4096
NCLUSTERS = 64

ALLOC_EMPTY = 10     # allocated, unmapped, empty          -> EIO
FREE_EMPTY = 11      # free                                -> zeros are correct
ALLOC_STALE = 12     # allocated, unmapped, stale bytes    -> EIO (the new case)
META_STALE = 13      # allocated, metadata                 -> image is authoritative
ALLOC_WRITTEN = 14   # allocated, unmapped, client wrote it -> serve it back


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
    m._metadata_clusters = {META_STALE}
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
    m._guest_written = bytearray()
    m._mft_runs = []
    m.bitmap_clusters = []
    m._mft_mirror_offset = -1
    m._mft_mirror_record_count = 0
    m._protected_top_dirs = set()
    m._protect_refused = set()
    m.ext4_authoritative = False
    m._check_sparse_file_read = lambda *a, **k: None

    bitmap = bytearray((NCLUSTERS + 7) // 8)
    for c in (ALLOC_EMPTY, ALLOC_STALE, META_STALE, ALLOC_WRITTEN):
        bitmap[c // 8] |= 1 << (c % 8)
    m._bitmap_cache = bitmap

    # Plausible leftover content in the two "stale" clusters.
    for c in (ALLOC_STALE, META_STALE):
        m.image[c * CS:c * CS + 8] = b"INDX\x28\x00\x09\x00"
    return m


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    return cond


def expect_eio(m, c, label):
    try:
        data = m._read_inner(c * CS, CS)
        return check(label, False, "served %d bytes instead" % len(data))
    except IOError:
        return check(label, True)


def expect_served(m, c, label, first4=None):
    try:
        data = m._read_inner(c * CS, CS)
        if first4 is not None:
            return check(label, data[:4] == first4, repr(data[:4]))
        return check(label, len(data) == CS)
    except IOError as e:
        return check(label, False, "raised %s" % e)


def main():
    ok = True
    m = make_mapper()

    print("\n[1] allocated, unmapped, empty - the original failure")
    ok &= expect_eio(m, ALLOC_EMPTY, "EIO rather than fabricated zeros")

    print("\n[2] allocated, unmapped, STALE NON-ZERO bytes - the window this closes")
    ok &= expect_eio(m, ALLOC_STALE, "EIO rather than plausible leftovers")

    print("\n[3] free in $Bitmap - a hole is the truth")
    ok &= expect_served(m, FREE_EMPTY, "zeros served without error")

    print("\n[4] metadata cluster - the image is authoritative there")
    ok &= expect_served(m, META_STALE, "served unchanged", b"INDX")

    print("\n[5] a client wrote this unmapped cluster - give its bytes back")
    m2 = make_mapper()
    m2._write_inner(ALLOC_WRITTEN * CS, b"\xcd" * CS)
    ok &= check("write recorded provenance", m2._is_guest_written(ALLOC_WRITTEN))
    ok &= expect_served(m2, ALLOC_WRITTEN, "served the client's own bytes", b"\xcd" * 4)

    print("\n[6] no bitmap loaded yet - unknown must never escalate")
    m3 = make_mapper()
    m3._bitmap_cache = None
    ok &= check("_cluster_is_allocated False without a cache",
                m3._cluster_is_allocated(ALLOC_EMPTY) is False)
    ok &= expect_served(m3, ALLOC_EMPTY, "read still succeeds during startup")

    print("\n[7] reported once per cluster")
    m4 = make_mapper()
    for _ in range(4):
        try:
            m4._read_inner(ALLOC_EMPTY * CS, CS)
        except IOError:
            pass
    ok &= check("one report after four attempts", len(m4._unmapped_reported) == 1,
                "got %d" % len(m4._unmapped_reported))

    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
