"""When may the protected INDEX_BITMAP be written back?

Two writes to a directory record look alike at the byte level and mean the
opposite:

  journal replay     Windows replays an OLD copy of the whole record. data_size
                     is smaller and the bitmap has fewer bits. Both are stale
                     and both must be restored, or INDX blocks vanish and the
                     directory reads as corrupt.

  directory shrink   Windows removed entries and freed an index block. NTFS
                     never shrinks $INDEX_ALLOCATION; it clears that block's bit
                     in $BITMAP and leaves data_size alone. Nothing is stale.
                     Forcing the bit back marks a freed block as in use.

The only signal separating them is whether data_size shrank. Restoring the
bitmap unconditionally gets the second case wrong; restoring it only alongside
a data_size restore gets both right. This pins that scoping so it cannot drift
back in either direction.
"""
import os
import struct
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import ClusterMapper, MFT_RECORD_SIZE  # noqa: E402

REC = 20
N = 32
IA_OFF, IB_OFF, IB_VAL_OFF = 0x180, 0x1D0, 0x20
BITMAP = b"\xff" * 12            # every INDX block in use
TARGET_DS = 8192


class FakeImage(bytearray):
    def flush(self):
        pass

    def close(self):
        pass


def mapper():
    m = ClusterMapper.__new__(ClusterMapper)
    m.lock = threading.RLock()
    m.image = FakeImage(N * MFT_RECORD_SIZE)
    m._mft_runs = [(0, N * MFT_RECORD_SIZE)]
    m.dir_indx_clusters = set()
    m._direct_allocated_records = set()
    m._protected_top_dirs = set()
    m._safe_mode = False
    m._windows_created_sources = set()
    m._protect_refused = set()
    m._protected_ia_sizes = {}
    m._ia_protect_warned = set()
    m._mft_mirror_offset = -1
    m._mft_mirror_record_count = 0
    m._guest_written = bytearray()
    m.ext4_authoritative = False
    m.cluster_size = 4096
    m._metadata_clusters = set()
    m.cluster_map = {}
    m.protect_ia_size(REC, IA_OFF, TARGET_DS, IB_OFF, IB_VAL_OFF, BITMAP)
    return m


def record(data_size, bitmap):
    rec = bytearray(MFT_RECORD_SIZE)
    rec[0:4] = b"FILE"
    struct.pack_into("<H", rec, 20, IA_OFF)
    struct.pack_into("<H", rec, 22, 0x3)
    struct.pack_into("<I", rec, IA_OFF, 0xA0)
    struct.pack_into("<I", rec, IA_OFF + 4, 0x50)
    rec[IA_OFF + 8] = 1
    struct.pack_into("<Q", rec, IA_OFF + 40, 65536)
    struct.pack_into("<Q", rec, IA_OFF + 48, data_size)
    struct.pack_into("<Q", rec, IA_OFF + 56, data_size)
    struct.pack_into("<I", rec, IB_OFF, 0xB0)
    struct.pack_into("<I", rec, IB_OFF + 4, 0x38)
    rec[IB_OFF + 8] = 0
    struct.pack_into("<I", rec, IB_OFF + 16, len(bitmap))
    struct.pack_into("<H", rec, IB_OFF + 20, IB_VAL_OFF)
    rec[IB_OFF + IB_VAL_OFF:IB_OFF + IB_VAL_OFF + len(bitmap)] = bitmap
    return rec


def write(m, rec):
    off = REC * MFT_RECORD_SIZE
    m._mft_write_to_image(off, bytes(rec))
    bm = bytes(m.image[off + IB_OFF + IB_VAL_OFF: off + IB_OFF + IB_VAL_OFF + len(BITMAP)])
    ds = struct.unpack_from("<Q", m.image, off + IA_OFF + 48)[0]
    return ds, bm


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    return cond


def main():
    ok = True

    print("\n[1] journal replay: whole old record, data_size AND bitmap stale")
    ds, bm = write(mapper(), record(512, bytes(12)))
    ok &= check("data_size restored", ds == TARGET_DS, "got %d" % ds)
    ok &= check("bitmap restored", bm == BITMAP, "got %r" % bm[:4])

    print("\n[2] directory shrink: a freed block's bit cleared, data_size intact")
    freed = bytearray(BITMAP)
    freed[0] &= ~0x01                 # block 0 released by Windows
    ds, bm = write(mapper(), record(TARGET_DS, bytes(freed)))
    ok &= check("data_size untouched", ds == TARGET_DS, "got %d" % ds)
    ok &= check("freed block NOT resurrected", bm == bytes(freed),
                "bit forced back on: %r" % bm[:2])

    print("\n[3] growth: Windows added a block, data_size larger than protected")
    grown = BITMAP + b"\x01"
    rec = record(TARGET_DS + 4096, grown)
    struct.pack_into("<I", rec, IB_OFF + 16, len(grown))
    ds, bm = write(mapper(), rec)
    ok &= check("larger data_size kept", ds == TARGET_DS + 4096, "got %d" % ds)

    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
