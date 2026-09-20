"""Clearing the volume dirty bit must reach $MFTMirr too.

ntfsfix deliberately sets VOLUME_IS_DIRTY so Windows runs chkdsk; the bridge
clears it again so the guest mounts clean. Record 3 ($Volume) is one of the
records $MFTMirr duplicates, and ntfs-3g refuses to mount a volume whose mirror
disagrees with $MFT:

    ntfs-3g mount failed: $MFTMirr does not match $MFT (record 3).

The two-way branch's clear_dirty_bit() patched the flags word in $MFT only.
Under --two-way that went unnoticed - the local ntfs-3g mount is skipped and
Windows tolerates the mismatch - and it broke every other configuration on the
first start after merging. Both copies were dumped from the failed sandbox
image: $MFT flags 0x0000, mirror flags 0x0001, one byte apart at offset 418.

Driven directly over an in-memory image holding both an $MFT and a mirror.
"""
import os
import struct
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import (  # noqa: E402
    ClusterMapper, MFT_RECORD_SIZE, MFT_RECORD_VOLUME)

MFT_OFF = 0                      # $MFT at the start of the image
MIRROR_OFF = 32 * MFT_RECORD_SIZE
IMAGE = 64 * MFT_RECORD_SIZE
ATTR_OFF = 0x38
VAL_OFF = 0x18
FLAGS_IN_REC = ATTR_OFF + VAL_OFF + 0x0A       # == 418 for these offsets? see check


class FakeImage(bytearray):
    def flush(self):
        pass

    def close(self):
        pass


def volume_record(dirty):
    """Record 3 with a resident $VOLUME_INFORMATION and a valid USA."""
    rec = bytearray(MFT_RECORD_SIZE)
    rec[0:4] = b"FILE"
    struct.pack_into("<H", rec, 4, 48)           # usa offset
    struct.pack_into("<H", rec, 6, 3)            # usa count (seq + 2 sectors)
    struct.pack_into("<H", rec, 20, ATTR_OFF)    # first attribute
    struct.pack_into("<H", rec, 22, 0x1)         # in use
    struct.pack_into("<I", rec, ATTR_OFF, 0x70)  # $VOLUME_INFORMATION
    struct.pack_into("<I", rec, ATTR_OFF + 4, 0x28)
    rec[ATTR_OFF + 8] = 0                        # resident
    struct.pack_into("<I", rec, ATTR_OFF + 16, 12)
    struct.pack_into("<H", rec, ATTR_OFF + 20, VAL_OFF)
    val = ATTR_OFF + VAL_OFF
    rec[val + 8] = 3                             # major
    rec[val + 9] = 1                             # minor
    struct.pack_into("<H", rec, val + 10, 0x0001 if dirty else 0x0000)
    struct.pack_into("<I", rec, ATTR_OFF + 0x28, 0xFFFFFFFF)
    # USA: seq=7 at 48; originals (0,0) at 50/52; sector ends carry seq
    struct.pack_into("<H", rec, 48, 7)
    struct.pack_into("<H", rec, 510, 7)
    struct.pack_into("<H", rec, 1022, 7)
    return rec


def make_mapper():
    m = ClusterMapper.__new__(ClusterMapper)
    m.lock = threading.RLock()
    m.image = FakeImage(IMAGE)
    m._mft_runs = [(MFT_OFF, 16 * MFT_RECORD_SIZE)]
    m._mft_mirror_offset = MIRROR_OFF
    m._mft_mirror_record_count = 4
    m.cluster_size = 4096
    m._metadata_clusters = set()
    m.cluster_map = {}
    m.dir_indx_clusters = set()
    m._direct_allocated_records = set()
    m._protected_top_dirs = set()
    m._safe_mode = False
    m._windows_created_sources = set()
    m._protect_refused = set()
    m._protected_ia_sizes = {}
    m._ia_protect_warned = set()
    m._guest_written = bytearray()
    m.ext4_authoritative = False
    m._attempt_log = None
    m.record_only = False
    m.mft_record_to_source = {}
    m.mft_record_to_dir = {}
    m._file_mft_seq = {}
    m._dir_mft_seq = {}
    return m


def seed(m, dirty=True):
    rec = volume_record(dirty)
    a = MFT_OFF + MFT_RECORD_VOLUME * MFT_RECORD_SIZE
    b = MIRROR_OFF + MFT_RECORD_VOLUME * MFT_RECORD_SIZE
    m.image[a:a + MFT_RECORD_SIZE] = rec
    m.image[b:b + MFT_RECORD_SIZE] = rec
    return a, b


def flags(m, base):
    return struct.unpack_from("<H", m.image, base + FLAGS_IN_REC)[0]


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    return cond


def main():
    ok = True

    print("\n[1] clear_dirty_bit(): the startup path after ntfsfix")
    m = make_mapper()
    a, b = seed(m)
    ok &= check("both copies start dirty", flags(m, a) == 1 and flags(m, b) == 1)
    m.clear_dirty_bit()
    ok &= check("$MFT copy clean", flags(m, a) == 0, "flags=%#x" % flags(m, a))
    ok &= check("$MFTMirr copy clean too", flags(m, b) == 0, "flags=%#x" % flags(m, b))
    ok &= check("mirror record byte-identical to $MFT",
                bytes(m.image[a:a + MFT_RECORD_SIZE]) == bytes(m.image[b:b + MFT_RECORD_SIZE]))

    print("\n[2] a guest write of $Volume through _mft_write_to_image")
    m = make_mapper()
    a, b = seed(m, dirty=False)
    m._mft_write_to_image(a, bytes(volume_record(dirty=True)))
    ok &= check("dirty bit masked on $MFT", flags(m, a) == 0, "flags=%#x" % flags(m, a))
    ok &= check("and on the mirror", flags(m, b) == 0, "flags=%#x" % flags(m, b))
    ok &= check("copies identical",
                bytes(m.image[a:a + MFT_RECORD_SIZE]) == bytes(m.image[b:b + MFT_RECORD_SIZE]))

    print("\n[3] a record the mirror does not cover is left alone")
    m = make_mapper()
    m._mft_mirror_record_count = 2          # mirror holds records 0-1 only
    a, b = seed(m)
    m.clear_dirty_bit()
    ok &= check("$MFT cleared", flags(m, a) == 0)
    ok &= check("mirror untouched (not in range)", flags(m, b) == 1)
    ok &= check("_sync_mirror_record reports no copy",
                m._sync_mirror_record(MFT_RECORD_VOLUME) is False)

    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
