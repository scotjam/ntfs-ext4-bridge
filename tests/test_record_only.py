"""Record-only: guest writes to pre-existing ext4 are catalogued, never applied.

The direction from Windows back to ext4 is the one that has destroyed data.
Before it is trusted, every write it would make should be reviewable as
evidence - what path, what bytes, would it have been valid. So in record-only
mode each such write is refused and written to a JSONL catalogue instead, with
no de-duplication, while objects Windows itself created this session (which
live in the overflow dir, not in the library) remain writable.

Driven directly over a temp ext4 tree: the file must be byte-identical
afterwards and the catalogue must describe exactly what was attempted.
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

CS = 4096
REC = 20


class FakeImage(bytearray):
    def flush(self):
        pass

    def close(self):
        pass


def make_mapper(source_dir, logpath):
    m = ClusterMapper.__new__(ClusterMapper)
    m.lock = threading.RLock()
    m.cluster_size = CS
    m.image = FakeImage(64 * CS)
    m.source_dir = source_dir
    m.overflow_dir = source_dir
    m._mft_runs = [(0, 32 * MFT_RECORD_SIZE)]
    m._protected_top_dirs = set()
    m.record_only = True
    m._safe_mode = True
    m._windows_created_sources = set()
    m._attempt_log = Ext4AttemptLog(logpath)
    m._protect_refused = set()
    m.ext4_authoritative = False
    m._metadata_clusters = set()
    m.cluster_map = {}
    m._direct_run_map = []
    m._guest_written = bytearray()
    m.dir_indx_clusters = set()
    m._direct_allocated_records = set()
    m.mft_record_to_source = {}
    m.mft_record_to_dir = {}
    m.path_to_mft_record = {}
    m.resident_file_data = {}
    m.source_to_clusters = {}
    m.ext4_sync_in_progress = set()
    m.ntfs_sync_in_progress = set()
    m.ntfs_sync_timestamps = {}
    m._file_mft_seq = {}
    m._dir_mft_seq = {}
    m._mft_mirror_offset = -1
    m._mft_mirror_record_count = 0
    m.lazy_allocator = None
    m._write_logged = set()
    m._dirty_sources = set()
    m._dirty_records = set()
    m.virtual_file_manager = None
    m.virtualized_indx_clusters = {}
    m.virtual_indx_map = {}
    m.virtualized_dirs = {}
    m._bitmap_cache = None
    m.bitmap_clusters = []
    m.sparse_files = {}
    m._unmapped_reported = set()
    return m


def deleted_record():
    rec = bytearray(MFT_RECORD_SIZE)
    rec[0:4] = b"FILE"
    struct.pack_into("<H", rec, 20, 0x38)
    struct.pack_into("<H", rec, 22, 0x00)     # in-use bit clear
    return rec


def entries(logpath):
    out = []
    for line in open(logpath, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    return cond


def main():
    ok = True
    tmp = tempfile.mkdtemp(prefix="recordonly-")
    src = os.path.join(tmp, "source")
    os.makedirs(os.path.join(src, "Share"))
    existing = os.path.join(src, "Share", "existing.bin")
    original = b"ORIGINAL" * (CS // 8) * 2       # two clusters
    with open(existing, "wb") as f:
        f.write(original)
    logpath = os.path.join(tmp, "attempts.jsonl")
    m = make_mapper(src, logpath)

    # clusters 10,11 map to existing.bin
    m.cluster_map[10] = (existing, 0)
    m.cluster_map[11] = (existing, CS)

    print("\n[1] a data write to a pre-existing file")
    m._write_inner(10 * CS, b"\xab" * CS)
    ok &= check("file unchanged on ext4", open(existing, "rb").read() == original)
    recs = [r for r in entries(logpath) if r.get("op") == "data_write"]
    ok &= check("one data_write catalogued", len(recs) == 1, "got %d" % len(recs))
    r = recs[0] if recs else {}
    ok &= check("bytes and span recorded", r.get("bytes") == CS and r.get("spans") == [[0, CS]],
                "%s %s" % (r.get("bytes"), r.get("spans")))
    ok &= check("preview shows the bytes", (r.get("preview_hex") or "").startswith("abab"))
    ok &= check("not flagged all-zero", r.get("all_zero") is False)

    print("\n[2] a zero-filled write is flagged as such")
    m._write_inner(11 * CS, bytes(CS))
    recs = [r for r in entries(logpath) if r.get("op") == "data_write"]
    ok &= check("second record present", len(recs) == 2)
    ok &= check("all_zero flagged", recs[-1].get("all_zero") is True)
    ok &= check("file still unchanged", open(existing, "rb").read() == original)

    print("\n[3] a delete of a pre-existing file")
    m.image[REC * MFT_RECORD_SIZE:(REC + 1) * MFT_RECORD_SIZE] = deleted_record()
    m.mft_record_to_source[REC] = existing
    ret = m._check_file_deleted(REC)
    ok &= check("file survives", os.path.exists(existing))
    ok &= check("returned False so the caller re-reads", ret is False)
    dels = [r for r in entries(logpath) if r.get("op") == "delete"]
    ok &= check("delete catalogued with reason", len(dels) == 1 and dels[0].get("reason") == "record-only",
                str(dels[:1]))

    print("\n[4] no de-duplication: the same attempt twice is two records")
    m.mft_record_to_source[REC] = existing
    m._check_file_deleted(REC)
    dels = [r for r in entries(logpath) if r.get("op") == "delete"]
    ok &= check("two delete records", len(dels) == 2, "got %d" % len(dels))

    print("\n[5] an object Windows created this session stays writable")
    created = os.path.join(src, "windows-made.bin")
    with open(created, "wb") as f:
        f.write(bytes(CS))
    m._windows_created_sources.add(created)
    m.cluster_map[12] = (created, 0)
    before = len(entries(logpath))
    m._write_inner(12 * CS, b"\xcd" * CS)
    ok &= check("write applied", open(created, "rb").read() == b"\xcd" * CS)
    ok &= check("nothing catalogued for it", len(entries(logpath)) == before)

    print("\n[6] catalogue shape")
    allr = entries(logpath)
    ok &= check("header first", allr and allr[0].get("header") is True)
    ok &= check("every record has ts/op/path",
                all(("ts" in r and "op" in r and "path" in r) for r in allr[1:]))

    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
