"""Record-only must catalogue what a refused MFT record write was trying to do.

The guest's delete, rename and truncate never reach ext4 in record-only mode:
the record write itself is dropped in _mft_write_to_image. Without a
classifier the catalogue showed data writes and nothing else. The classifier
rebuilds the record the guest wanted and diffs it against the image:

  in-use flag cleared            -> delete
  sequence number bumped         -> delete (record freed; the kernel page cache
                                    coalesces a delete + a create that reused
                                    the slot into one flush that looks like a
                                    rename - it is not)
  $FILE_NAME name/parent changed -> rename
  unnamed $DATA size changed     -> truncate / extend
  anything else                  -> record_update, once per record

Also: a record whose path already exists on ext4 is linked, not refused, in
_check_new_file - the post-startup populate re-creates through the mount
whatever the guest removed, and an unlinked record EIOs on every read.
"""
import json
import os
import struct
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import (  # noqa: E402
    ClusterMapper, Ext4AttemptLog, MFT_RECORD_SIZE)

MFT_OFF = 0
IMAGE = 64 * MFT_RECORD_SIZE
REC = 40


def fn_attr_len(name):
    return (0x18 + 0x42 + len(name.encode("utf-16-le")) + 7) & ~7


def file_record(name, parent=5, size=100, in_use=True, seq=1, resident=False,
                fill=b"\x00"):
    rec = bytearray(MFT_RECORD_SIZE)
    rec[0:4] = b"FILE"
    struct.pack_into("<H", rec, 4, 48)
    struct.pack_into("<H", rec, 6, 3)
    struct.pack_into("<H", rec, 16, seq)
    struct.pack_into("<H", rec, 20, 0x38)
    struct.pack_into("<H", rec, 22, 0x1 if in_use else 0x0)
    off = 0x38
    # $FILE_NAME (0x30), resident
    nm = name.encode("utf-16-le")
    val_len = 0x42 + len(nm)
    attr_len = fn_attr_len(name)
    struct.pack_into("<I", rec, off, 0x30)
    struct.pack_into("<I", rec, off + 4, attr_len)
    struct.pack_into("<I", rec, off + 16, val_len)
    struct.pack_into("<H", rec, off + 20, 0x18)
    v = off + 0x18
    struct.pack_into("<Q", rec, v, parent | (1 << 48))
    rec[v + 0x40] = len(name)
    rec[v + 0x41] = 3
    rec[v + 0x42:v + 0x42 + len(nm)] = nm
    off += attr_len
    # unnamed $DATA (0x80)
    struct.pack_into("<I", rec, off, 0x80)
    if resident:
        pad = (size + 7) & ~7
        struct.pack_into("<I", rec, off + 4, 0x18 + pad)
        rec[off + 8] = 0
        struct.pack_into("<I", rec, off + 16, size)
        struct.pack_into("<H", rec, off + 20, 0x18)
        rec[off + 0x18:off + 0x18 + size] = fill * size
        off += 0x18 + pad
    else:
        struct.pack_into("<I", rec, off + 4, 0x48)
        rec[off + 8] = 1
        struct.pack_into("<H", rec, off + 32, 0x40)
        struct.pack_into("<Q", rec, off + 40, (size + 4095) & ~4095)
        struct.pack_into("<Q", rec, off + 48, size)
        struct.pack_into("<Q", rec, off + 56, size)
        # one data run: 1 cluster at LCN 20
        rec[off + 0x40:off + 0x43] = bytes([0x11, 0x01, 0x14])
        off += 0x48
    struct.pack_into("<I", rec, off, 0xFFFFFFFF)
    struct.pack_into("<H", rec, 24, off + 8)
    struct.pack_into("<H", rec, 48, 7)
    struct.pack_into("<H", rec, 510, 7)
    struct.pack_into("<H", rec, 1022, 7)
    return bytes(rec)


class FakeImage(bytearray):
    def flush(self):
        pass

    def close(self):
        pass


def make_mapper(tmp):
    m = ClusterMapper.__new__(ClusterMapper)
    m.lock = threading.RLock()
    m.image = FakeImage(IMAGE)
    m._mft_runs = [(MFT_OFF, 64 * MFT_RECORD_SIZE)]
    m._mft_mirror_offset = -1
    m._mft_mirror_record_count = 0
    m.cluster_size = 4096
    m.source_dir = tmp
    m.overflow_dir = os.path.join(tmp, "_overflow")
    m._metadata_clusters = set()
    m.cluster_map = {}
    m.dir_indx_clusters = set()
    m._direct_allocated_records = set()
    m._protected_top_dirs = set()
    m._safe_mode = True
    m.record_only = True
    m._windows_created_sources = set()
    m._protect_refused = set()
    m._protected_ia_sizes = {}
    m._ia_protect_warned = set()
    m._guest_written = bytearray()
    m.ext4_authoritative = False
    m.mft_record_to_source = {}
    m.mft_record_to_dir = {}
    m._file_mft_seq = {}
    m._dir_mft_seq = {}
    m.path_to_mft_record = {}
    m.log_path = os.path.join(tmp, "attempts.jsonl")
    m._attempt_log = Ext4AttemptLog(m.log_path)
    return m


def entries(m):
    m._attempt_log._fh.flush()
    out = []
    for line in open(m.log_path, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("header"):
            out.append(r)
    return out


def seed(m, rec, num=REC, source="a.bin"):
    off = MFT_OFF + num * MFT_RECORD_SIZE
    m.image[off:off + MFT_RECORD_SIZE] = rec
    sp = os.path.join(m.source_dir, source)
    open(sp, "wb").write(b"x" * 100)
    m.mft_record_to_source[num] = sp
    return off


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    return cond


def run_case(label, before, after, want_op, extra=None):
    with tempfile.TemporaryDirectory() as tmp:
        m = make_mapper(tmp)
        off = seed(m, before)
        m._mft_write_to_image(off, after)
        ok = check(label + ": image record unchanged",
                   bytes(m.image[off:off + MFT_RECORD_SIZE]) == before)
        es = entries(m)
        ok &= check(label + ": one entry", len(es) == 1, str([e["op"] for e in es]))
        if es:
            ok &= check(label + ": op == %s" % want_op, es[0]["op"] == want_op, es[0]["op"])
            ok &= check(label + ": path recorded", es[0]["path"] == "a.bin", es[0]["path"])
            for k, v in (extra or {}).items():
                ok &= check(label + ": %s == %r" % (k, v), es[0].get(k) == v,
                            repr(es[0].get(k)))
        return ok


def main():
    ok = True
    print("\n[classifier]")
    base = file_record("a.bin")
    ok &= run_case("in-use cleared", base, file_record("a.bin", in_use=False), "delete")
    ok &= run_case("seq bump + slot reused", base,
                   file_record("new.bin", seq=2), "delete", {"reused_for": "new.bin"})
    ok &= run_case("name changed", base, file_record("b.bin"), "rename",
                   {"from": "a.bin", "to": "b.bin"})
    ok &= run_case("parent changed", base, file_record("a.bin", parent=70), "rename")
    ok &= run_case("size shrunk", base, file_record("a.bin", size=10), "truncate",
                   {"size_from": 100, "size_to": 10})
    ok &= run_case("size grown", base, file_record("a.bin", size=5000), "extend")
    ok &= run_case("resident data rewritten",
                   file_record("a.bin", size=8, resident=True),
                   file_record("a.bin", size=8, resident=True, fill=b"\xff"),
                   "record_update")

    print("\n[record_update is counted once per record]")
    with tempfile.TemporaryDirectory() as tmp:
        m = make_mapper(tmp)
        off = seed(m, base)
        churn = bytearray(base)
        struct.pack_into("<Q", churn, 0x38 + 0x18 + 8, 12345)   # a timestamp in $FILE_NAME
        m._mft_write_to_image(off, bytes(churn))
        struct.pack_into("<Q", churn, 0x38 + 0x18 + 8, 67890)
        m._mft_write_to_image(off, bytes(churn))
        es = entries(m)
        ok &= check("two churn writes -> one entry",
                    len(es) == 1 and es[0]["op"] == "record_update",
                    str([e["op"] for e in es]))

    print("\n[directly allocated records are catalogued too]")
    with tempfile.TemporaryDirectory() as tmp:
        m = make_mapper(tmp)
        m._safe_mode = False
        off = seed(m, base)
        m._direct_allocated_records.add(REC)
        m._mft_write_to_image(off, file_record("a.bin", size=10))
        es = entries(m)
        ok &= check("truncate of lazy file catalogued",
                    [e["op"] for e in es] == ["truncate"], str([e["op"] for e in es]))
        ok &= check("record still protected",
                    bytes(m.image[off:off + MFT_RECORD_SIZE]) == base)

    print("\n[_check_new_file links an existing ext4 path instead of refusing]")
    with tempfile.TemporaryDirectory() as tmp:
        m = make_mapper(tmp)
        m.ext4_sync_in_progress = set()
        m.source_to_clusters = {}
        m._direct_run_map = []
        m.resident_file_data = {}
        m.ntfs_sync_in_progress = set()
        m.ntfs_sync_timestamps = {}
        sp = os.path.join(tmp, "exists.bin")
        open(sp, "wb").write(b"y" * 100)
        off = MFT_OFF + REC * MFT_RECORD_SIZE
        m.image[off:off + MFT_RECORD_SIZE] = file_record("exists.bin")
        m._resolve_source_path = lambda rel: os.path.join(tmp, rel)
        m._validate_path = lambda p, who: True
        m._is_orphan_root_fallthrough = lambda pr, rp: False
        tracked = []
        m._track_file_data = lambda record, num, src: tracked.append((num, src))
        r = m._check_new_file(REC)
        ok &= check("returns None (nothing created)", r is None)
        ok &= check("record now tracked to ext4 file",
                    m.mft_record_to_source.get(REC) == sp,
                    repr(m.mft_record_to_source.get(REC)))
        ok &= check("data runs tracked", tracked == [(REC, sp)], repr(tracked))
        ok &= check("no materialize catalogued", entries(m) == [], str(entries(m)))
        ok &= check("ext4 file untouched", open(sp, "rb").read() == b"y" * 100)

    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
