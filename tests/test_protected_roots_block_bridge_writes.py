"""protected_roots must stop the BRIDGE mutating ext4, not just the guest.

--protected-roots was only ever consulted on the NBD write path (_write_inner,
_mft_write_to_image), i.e. "the guest tried to write". The bridge's own MFT
reconciliation mutates ext4 directly, and none of it looked. With a stale image
and no guest attached at all, _check_file_deleted saw records it could no longer
account for and unlinked the ext4 files behind them; _check_new_file rebuilt
others out of image clusters holding nothing but zeros.

That is how a protected Shows tree lost a file and had four files
rewritten to their exact original size in all-zero bytes, with the VM powered
off the whole time.

The delete path is driven here for real over a temp source tree - no root, no
NBD, no ntfs-3g - for a file under a protected root and an identical one
outside it, so the guard is shown to be targeted rather than blanket.
"""
import os
import struct
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import ClusterMapper, MFT_RECORD_SIZE  # noqa: E402

TOTAL_RECORDS = 32
PROTECTED_REC = 20
OPEN_REC = 21


class FakeImage(bytearray):
    """Stands in for _HotImageCache: a buffer with a no-op flush/close."""

    def flush(self):
        pass

    def close(self):
        pass


def make_mapper(source_dir, image):
    m = ClusterMapper.__new__(ClusterMapper)
    m.lock = threading.RLock()
    m.image = image
    m.source_dir = source_dir
    m.overflow_dir = source_dir
    m._mft_runs = [(0, TOTAL_RECORDS * MFT_RECORD_SIZE)]
    m._protected_top_dirs = {"shows"}
    m._protect_refused = set()
    m.mft_record_to_source = {}
    m.mft_record_to_dir = {}
    m.path_to_mft_record = {}
    m.resident_file_data = {}
    m.source_to_clusters = {}
    m.cluster_map = {}
    m._direct_run_map = []
    m.ext4_sync_in_progress = set()
    m.ntfs_sync_in_progress = set()
    m.ntfs_sync_timestamps = {}
    return m


def deleted_record():
    """An MFT record whose in-use flag is clear - i.e. 'this file is gone'."""
    rec = bytearray(MFT_RECORD_SIZE)
    rec[0:4] = b"FILE"
    struct.pack_into("<H", rec, 20, 0x38)
    struct.pack_into("<H", rec, 22, 0x00)   # in-use bit CLEAR
    return rec


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    return cond


def main():
    ok = True
    tmp = tempfile.mkdtemp(prefix="bridge-protect-")

    # A source_dir shaped like the real one: top-level trees, one protected.
    prot_dir = os.path.join(tmp, "Shows", "Some Show")
    open_dir = os.path.join(tmp, "Scratch", "Some Show")
    os.makedirs(prot_dir)
    os.makedirs(open_dir)
    prot_file = os.path.join(prot_dir, "episode.mkv")
    open_file = os.path.join(open_dir, "episode.mkv")
    for p in (prot_file, open_file):
        with open(p, "wb") as f:
            f.write(b"real content" * 1000)

    image = FakeImage(TOTAL_RECORDS * MFT_RECORD_SIZE)
    rec = deleted_record()
    for rn in (PROTECTED_REC, OPEN_REC):
        image[rn * MFT_RECORD_SIZE:(rn + 1) * MFT_RECORD_SIZE] = rec

    m = make_mapper(tmp, image)
    m.mft_record_to_source[PROTECTED_REC] = prot_file
    m.mft_record_to_source[OPEN_REC] = open_file

    # --- 1. protected root: the bridge must not unlink
    print("\n[1] record says deleted, file lives under a protected root")
    ret = m._check_file_deleted(PROTECTED_REC)
    ok &= check("ext4 file still exists", os.path.exists(prot_file))
    ok &= check("file still has its content",
                os.path.getsize(prot_file) == 12000 if os.path.exists(prot_file) else False)
    ok &= check("returned False (caller re-reads instead of deleting)", ret is False)
    ok &= check("mapping kept", m.mft_record_to_source.get(PROTECTED_REC) == prot_file)
    ok &= check("refusal recorded", any(w == "delete" for _, w in m._protect_refused))

    # --- 2. unprotected root: behaviour unchanged
    print("\n[2] same record shape, file outside any protected root")
    ret = m._check_file_deleted(OPEN_REC)
    ok &= check("ext4 file deleted as before", not os.path.exists(open_file))
    ok &= check("returned True", ret is True)
    ok &= check("mapping dropped", OPEN_REC not in m.mft_record_to_source)

    # --- 3. the predicate itself
    print("\n[3] _refuse_ext4_mutation targeting")
    m2 = make_mapper(tmp, FakeImage(TOTAL_RECORDS * MFT_RECORD_SIZE))
    ok &= check("protected path refused",
                m2._refuse_ext4_mutation(prot_file, "materialize", "Shows/x") is True)
    ok &= check("unprotected path allowed",
                m2._refuse_ext4_mutation(open_file, "materialize", "Scratch/x") is False)
    ok &= check("path outside source_dir allowed",
                m2._refuse_ext4_mutation("/elsewhere/Shows/x.mkv", "materialize") is False)
    m2._protected_top_dirs = set()
    ok &= check("no protected roots configured -> never refuses",
                m2._refuse_ext4_mutation(prot_file, "materialize") is False)

    # --- 4. logged once per (path, op), not per call
    print("\n[4] refusal is logged once per path+operation")
    m3 = make_mapper(tmp, FakeImage(TOTAL_RECORDS * MFT_RECORD_SIZE))
    for _ in range(5):
        m3._refuse_ext4_mutation(prot_file, "materialize", "Shows/x")
    ok &= check("one entry after five calls", len(m3._protect_refused) == 1,
                "got %d" % len(m3._protect_refused))

    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
