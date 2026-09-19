"""Malformed MFT records must never make the bridge write outside the record.

Every structure bug so far was a write that landed somewhere it did not belong:
a bitmap written at a stale offset overwrote allocated_size, and a re-patch
trusted offsets captured before Windows restructured the record. Neither needed
an exotic input - just a record whose shape had changed.

So the invariant under fuzz is spatial, not semantic. The image is filled with a
canary; only the bytes of the record under test may differ afterwards. A single
byte changed outside it is a memory-safety-class bug in the patching path, and
an unhandled exception is a denial of service on a volume the guest is using.

    python3 fuzz_mft_structures.py [iterations] [--seed N]
"""
import os
import random
import struct
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import ClusterMapper, MFT_RECORD_SIZE  # noqa: E402

TOTAL_RECORDS = 16
REC = 8
CANARY = 0xA5
IA_OFF = 0x180
IB_OFF = 0x1D0
IB_VAL_OFF = 0x20
BITMAP = b"\xff" * 12
TARGET_DS = 8192


class FakeImage(bytearray):
    def flush(self):
        pass

    def close(self):
        pass


def make_mapper():
    m = ClusterMapper.__new__(ClusterMapper)
    m.lock = threading.RLock()
    m.image = FakeImage(bytes([CANARY]) * (TOTAL_RECORDS * MFT_RECORD_SIZE))
    m._mft_runs = [(0, TOTAL_RECORDS * MFT_RECORD_SIZE)]
    m.dir_indx_clusters = set()
    m._direct_allocated_records = set()
    m._protected_top_dirs = set()
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
    return m


def valid_record():
    rec = bytearray(MFT_RECORD_SIZE)
    rec[0:4] = b"FILE"
    struct.pack_into("<H", rec, 4, 48)      # usa offset
    struct.pack_into("<H", rec, 6, 3)       # usa count
    struct.pack_into("<H", rec, 20, IA_OFF)
    struct.pack_into("<H", rec, 22, 0x3)
    # $INDEX_ALLOCATION, non-resident
    struct.pack_into("<I", rec, IA_OFF + 0, 0xA0)
    struct.pack_into("<I", rec, IA_OFF + 4, 0x50)
    rec[IA_OFF + 8] = 1
    struct.pack_into("<Q", rec, IA_OFF + 40, 65536)
    struct.pack_into("<Q", rec, IA_OFF + 48, TARGET_DS)
    struct.pack_into("<Q", rec, IA_OFF + 56, TARGET_DS)
    # $INDEX_BITMAP, resident
    struct.pack_into("<I", rec, IB_OFF + 0, 0xB0)
    struct.pack_into("<I", rec, IB_OFF + 4, 0x38)
    rec[IB_OFF + 8] = 0
    struct.pack_into("<I", rec, IB_OFF + 16, len(BITMAP))
    struct.pack_into("<H", rec, IB_OFF + 20, IB_VAL_OFF)
    rec[IB_OFF + IB_VAL_OFF:IB_OFF + IB_VAL_OFF + len(BITMAP)] = BITMAP
    struct.pack_into("<I", rec, IB_OFF + 0x38, 0xFFFFFFFF)   # end marker
    return rec


def mutate(rec, rng):
    """Bit flips, byte stomps, and field-targeted corruption."""
    rec = bytearray(rec)
    how = rng.choice(["bitflip", "stomp", "field", "truncate_attr", "type_swap"])
    if how == "bitflip":
        for _ in range(rng.randint(1, 8)):
            i = rng.randrange(len(rec))
            rec[i] ^= 1 << rng.randrange(8)
    elif how == "stomp":
        i = rng.randrange(len(rec))
        n = rng.randint(1, 64)
        rec[i:i + n] = bytes(rng.randrange(256) for _ in range(min(n, len(rec) - i)))
    elif how == "field":
        off = rng.choice([IA_OFF, IB_OFF])
        fld = rng.choice([0, 4, 8, 16, 20, 40, 48, 56])
        width = rng.choice([1, 2, 4, 8])
        if off + fld + width <= len(rec):
            rec[off + fld:off + fld + width] = bytes(
                rng.randrange(256) for _ in range(width))
    elif how == "truncate_attr":
        struct.pack_into("<I", rec, IA_OFF + 4, rng.randrange(0, 0x200))
    else:
        struct.pack_into("<I", rec, IA_OFF + 0, rng.choice(
            [0x10, 0x30, 0x80, 0x90, 0xB0, 0xFFFFFFFF, rng.randrange(1 << 32)]))
    return rec, how


def main():
    iters = 20000
    seed = 1234
    args = [a for a in sys.argv[1:]]
    if args and args[0].isdigit():
        iters = int(args[0])
    if "--seed" in args:
        seed = int(args[args.index("--seed") + 1])
    rng = random.Random(seed)
    print("fuzzing %d records (seed=%d)" % (iters, seed))

    rec_abs = REC * MFT_RECORD_SIZE
    base = valid_record()
    crashes = []
    escapes = []
    by_how = {}

    for i in range(iters):
        m = make_mapper()
        m.protect_ia_size(REC, IA_OFF, TARGET_DS, IB_OFF, IB_VAL_OFF, BITMAP)
        rec, how = mutate(base, rng)
        by_how[how] = by_how.get(how, 0) + 1
        try:
            m._mft_write_to_image(rec_abs, bytes(rec))
        except Exception as e:                      # noqa: BLE001
            crashes.append((i, how, type(e).__name__, str(e)[:120]))
            continue
        img = m.image
        before = img[:rec_abs]
        after = img[rec_abs + MFT_RECORD_SIZE:]
        if any(b != CANARY for b in before) or any(b != CANARY for b in after):
            bad = next((j for j, b in enumerate(before) if b != CANARY), None)
            if bad is None:
                bad = rec_abs + MFT_RECORD_SIZE + next(
                    j for j, b in enumerate(after) if b != CANARY)
            escapes.append((i, how, bad))

    print("\nmutation mix: %s" % ", ".join("%s=%d" % kv for kv in sorted(by_how.items())))
    print("writes escaping the record : %d" % len(escapes))
    print("unhandled exceptions       : %d" % len(crashes))
    if escapes:
        print("\nESCAPES (first 10):")
        for i, how, off in escapes[:10]:
            print("   iter %-6d %-14s wrote at byte %d (record spans %d..%d)"
                  % (i, how, off, rec_abs, rec_abs + MFT_RECORD_SIZE))
    if crashes:
        print("\nCRASHES (first 10):")
        for i, how, exc, msg in crashes[:10]:
            print("   iter %-6d %-14s %s: %s" % (i, how, exc, msg))

    ok = not escapes and not crashes
    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
