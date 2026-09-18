"""Protecting a directory's INDEX_ALLOC must never corrupt its INDEX_BITMAP.

_protected_ia_sizes records where a directory's $INDEX_ALLOCATION and (resident)
$INDEX_BITMAP sat when the MFT was scanned at startup, and _mft_write_to_image()
re-patches those offsets after every Windows write so the fix can't be reverted.

But Windows rewrites records, and once a directory outgrows its record it
converts $INDEX_BITMAP from resident to non-resident. The two layouts are
different shapes: a resident attribute's value lives at +value_offset, while a
non-resident one has allocated_size at +40. Re-patching blind then wrote the
bitmap bytes into the non-resident header, and an all-ones bitmap left
allocated_size = 0xFFFFFFFF. That is the on-disk state behind ntfs-3g's
"Corrupt non resident attribute 0xb0 in MFT record N", and a directory whose
index is mis-described that way hands Windows the wrong file records - which is
how files end up listed under a folder they don't belong to.

No root, no NBD, no ntfs-3g: the mapper's write path is driven directly over a
bytearray standing in for the image.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import ClusterMapper, MFT_RECORD_SIZE  # noqa: E402

REC_NUM = 20                      # not $Volume (3), not a reserved record
TOTAL_RECORDS = 32
IA_OFF = 0x180                    # 8-byte aligned, as real attributes are
IB_OFF = 0x1D0
IB_VAL_OFF = 0x20                 # resident value offset within the attribute
BITMAP = b"\xff" * 12             # "every INDX block allocated"
# With IB_VAL_OFF = 0x20, a 12-byte value spans +32..+44 of the attribute. On a
# NON-resident header that covers allocated_size at +40..+43 and stops before
# its high half - leaving exactly the 4294967295 seen on the live volume.
GOOD_ALLOC = 65536                # a sane non-resident allocated_size
TARGET_DS = 8192


class FakeImage(bytearray):
    """Stands in for _HotImageCache: a buffer with a no-op flush/close."""

    def flush(self):
        pass

    def close(self):
        pass


def make_mapper(image):
    """A ClusterMapper with only the state _mft_write_to_image() touches."""
    m = ClusterMapper.__new__(ClusterMapper)
    m.image = image
    m._mft_runs = [(0, TOTAL_RECORDS * MFT_RECORD_SIZE)]
    m.dir_indx_clusters = set()
    m._direct_allocated_records = set()
    m._protected_top_dirs = set()
    m._protected_ia_sizes = {}
    m._ia_protect_warned = set()
    m._mft_mirror_offset = -1
    m._mft_mirror_record_count = 0
    return m


def base_record():
    """A directory record: INDEX_ALLOC at IA_OFF, INDEX_BITMAP at IB_OFF."""
    rec = bytearray(MFT_RECORD_SIZE)
    rec[0:4] = b"FILE"
    struct.pack_into("<H", rec, 20, IA_OFF)      # first attribute offset
    struct.pack_into("<H", rec, 22, 0x3)         # in use + directory

    # $INDEX_ALLOCATION, non-resident
    struct.pack_into("<I", rec, IA_OFF + 0, 0xA0)
    struct.pack_into("<I", rec, IA_OFF + 4, 0x50)
    rec[IA_OFF + 8] = 1
    struct.pack_into("<Q", rec, IA_OFF + 40, GOOD_ALLOC)
    struct.pack_into("<Q", rec, IA_OFF + 48, TARGET_DS)
    struct.pack_into("<Q", rec, IA_OFF + 56, TARGET_DS)
    return rec


def add_resident_bitmap(rec):
    struct.pack_into("<I", rec, IB_OFF + 0, 0xB0)
    struct.pack_into("<I", rec, IB_OFF + 4, 0x38)
    rec[IB_OFF + 8] = 0                                   # resident
    struct.pack_into("<I", rec, IB_OFF + 16, len(BITMAP))  # value length
    struct.pack_into("<H", rec, IB_OFF + 20, IB_VAL_OFF)   # value offset
    rec[IB_OFF + IB_VAL_OFF:IB_OFF + IB_VAL_OFF + len(BITMAP)] = BITMAP
    return rec


def add_nonresident_bitmap(rec):
    """What Windows leaves behind once the directory outgrows the record."""
    struct.pack_into("<I", rec, IB_OFF + 0, 0xB0)
    struct.pack_into("<I", rec, IB_OFF + 4, 0x50)
    rec[IB_OFF + 8] = 1                                    # NON-resident
    struct.pack_into("<Q", rec, IB_OFF + 40, GOOD_ALLOC)   # allocated_size
    struct.pack_into("<Q", rec, IB_OFF + 48, 4096)         # data_size
    struct.pack_into("<Q", rec, IB_OFF + 56, 4096)         # init_size
    return rec


def run_write(mapper, rec):
    off = REC_NUM * MFT_RECORD_SIZE
    mapper._mft_write_to_image(off, bytes(rec))
    return off


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    return cond


def main():
    ok = True

    # --- 1. regression: bitmap must NOT be written into a non-resident header
    image = FakeImage(TOTAL_RECORDS * MFT_RECORD_SIZE)
    m = make_mapper(image)
    m.protect_ia_size(REC_NUM, IA_OFF, TARGET_DS, IB_OFF, IB_VAL_OFF, BITMAP)
    rec = add_nonresident_bitmap(base_record())
    off = run_write(m, rec)

    alloc = struct.unpack_from("<Q", image, off + IB_OFF + 40)[0]
    print("\n[1] $INDEX_BITMAP turned non-resident under us")
    ok &= check("allocated_size not clobbered", alloc == GOOD_ALLOC,
                "got %d (0x%X), want %d" % (alloc, alloc, GOOD_ALLOC))
    ok &= check("allocated_size is not the 0xFFFFFFFF corruption",
                alloc != 0xFFFFFFFF)
    ok &= check("staleness was reported",
                (REC_NUM, 'INDEX_BITMAP') in m._ia_protect_warned)

    # INDEX_ALLOC protection still has to work in that same record
    ds = struct.unpack_from("<Q", image, off + IA_OFF + 48)[0]
    ok &= check("INDEX_ALLOC data_size still re-patched", ds == TARGET_DS)

    # --- 2. the resident case still gets protected
    image = FakeImage(TOTAL_RECORDS * MFT_RECORD_SIZE)
    m = make_mapper(image)
    m.protect_ia_size(REC_NUM, IA_OFF, TARGET_DS, IB_OFF, IB_VAL_OFF, BITMAP)
    rec = add_resident_bitmap(base_record())
    rec[IB_OFF + IB_VAL_OFF:IB_OFF + IB_VAL_OFF + len(BITMAP)] = b"\x00" * len(BITMAP)
    rec[IA_OFF + 8] = 0                       # Windows clears the non-res flag
    struct.pack_into("<Q", rec, IA_OFF + 48, 512)   # and shrinks data_size
    off = run_write(m, rec)

    print("\n[2] $INDEX_BITMAP still resident where we recorded it")
    got_bm = bytes(image[off + IB_OFF + IB_VAL_OFF:
                         off + IB_OFF + IB_VAL_OFF + len(BITMAP)])
    ok &= check("bitmap re-patched", got_bm == BITMAP, "got %r" % (got_bm,))
    ok &= check("INDEX_ALLOC non-resident flag restored",
                image[off + IA_OFF + 8] == 1)
    ok &= check("data_size restored",
                struct.unpack_from("<Q", image, off + IA_OFF + 48)[0] == TARGET_DS)
    ok &= check("no staleness reported", not m._ia_protect_warned)

    # --- 3. the attribute at the recorded offset is something else entirely
    image = FakeImage(TOTAL_RECORDS * MFT_RECORD_SIZE)
    m = make_mapper(image)
    m.protect_ia_size(REC_NUM, IA_OFF, TARGET_DS, IB_OFF, IB_VAL_OFF, BITMAP)
    rec = base_record()
    struct.pack_into("<I", rec, IA_OFF + 0, 0x80)   # DATA now sits here
    sentinel = bytes(rec[IA_OFF:IA_OFF + 64])
    off = run_write(m, rec)

    print("\n[3] a different attribute now sits at the INDEX_ALLOC offset")
    ok &= check("attribute left untouched",
                bytes(image[off + IA_OFF:off + IA_OFF + 64]) == sentinel)
    ok &= check("staleness was reported",
                (REC_NUM, 'INDEX_ALLOC') in m._ia_protect_warned)

    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
