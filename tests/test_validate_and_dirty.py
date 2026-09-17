#!/usr/bin/env python3
"""Tests for symlink-aware _validate_path and VOLUME_IS_DIRTY masking.

Both exercise ClusterMapper methods that only need a handful of attributes,
so they run against a bare instance rather than a full 2.4TB image.

    python3 tests/test_validate_and_dirty.py
"""
import os
import shutil
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import (
    ClusterMapper, MFT_RECORD_SIZE, MFT_RECORD_VOLUME,
    ATTR_VOLUME_INFORMATION, VOLUME_IS_DIRTY,
)

failures = []


def check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
        failures.append(name)


def bare_mapper(source_dir, overflow_dir):
    m = object.__new__(ClusterMapper)
    m.source_dir = os.path.abspath(source_dir)
    m.overflow_dir = os.path.abspath(overflow_dir)
    m._real_roots_key = None
    m._real_roots = frozenset()
    return m


def test_validate_path():
    print("_validate_path (symlink farm)")
    tmp = tempfile.mkdtemp(prefix='vp-')
    try:
        source = os.path.join(tmp, 'bridge-source')
        overflow = os.path.join(tmp, 'bridge-overflow')
        media = os.path.join(tmp, 'realdisk', 'movies')
        secret = os.path.join(tmp, 'elsewhere')
        os.makedirs(source)
        os.makedirs(overflow)
        os.makedirs(os.path.join(media, 'Example Title'))
        os.makedirs(secret)
        open(os.path.join(media, 'Example Title', 'a.mkv'), 'w').close()
        open(os.path.join(secret, 'shadow'), 'w').close()

        # The exported root is a symlink to another filesystem - the design.
        os.symlink(media, os.path.join(source, 'Movies'))

        m = bare_mapper(source, overflow)

        # A real file under the root symlink must be accepted. realpath() lands
        # outside source_dir, which the old check rejected.
        check("file under root symlink",
              m._validate_path(os.path.join(source, 'Movies', 'Example Title', 'a.mkv')),
              True)
        check("the root symlink itself",
              m._validate_path(os.path.join(source, 'Movies')), True)
        check("source_dir itself", m._validate_path(source), True)
        check("overflow entry",
              m._validate_path(os.path.join(overflow, 'System Volume Information')),
              True)

        # Traversal out of the tree must still be rejected.
        check("dotdot escape",
              m._validate_path(os.path.join(source, '..', '..', 'etc', 'shadow')),
              False)
        check("absolute outside",
              m._validate_path('/etc/shadow'), False)
        check("null byte",
              m._validate_path(os.path.join(source, 'a\x00b')), False)

        # A hostile symlink *inside* the media tree still gets rejected: it is
        # not one of the declared roots.
        os.symlink(secret, os.path.join(media, 'escape'))
        check("symlink inside tree pointing out",
              m._validate_path(os.path.join(source, 'Movies', 'escape', 'shadow')),
              False)

        # A newly added root symlink is picked up without a restart.
        os.symlink(secret, os.path.join(source, 'NewRoot'))
        check("newly added root symlink",
              m._validate_path(os.path.join(source, 'NewRoot', 'shadow')), True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_volume_record(flags):
    """Build a minimal $Volume MFT record with the given volume flags."""
    rec = bytearray(MFT_RECORD_SIZE)
    rec[0:4] = b'FILE'
    struct.pack_into('<H', rec, 4, 48)    # USA offset
    struct.pack_into('<H', rec, 6, 3)     # USA count (1 + 2 sectors)
    struct.pack_into('<H', rec, 20, 56)   # first attribute offset

    attr = 56
    # Value is 12 bytes at offset 24, so the attribute needs 36 bytes rounded
    # up to the 8-byte boundary: 40. Using 32 would put the end marker on top
    # of the flags field and the test would pass for the wrong reason.
    attr_len = 40
    struct.pack_into('<I', rec, attr, ATTR_VOLUME_INFORMATION)
    struct.pack_into('<I', rec, attr + 4, attr_len)
    rec[attr + 8] = 0                            # resident
    struct.pack_into('<H', rec, attr + 16, 12)   # value length
    struct.pack_into('<H', rec, attr + 20, 24)   # value offset
    struct.pack_into('<H', rec, attr + 24 + 10, flags)

    struct.pack_into('<I', rec, attr + attr_len, 0xFFFFFFFF)  # end marker
    return rec


class FakeImage:
    """Just enough of _HotImageCache for the helper under test."""

    def __init__(self, data):
        self.data = bytearray(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, s):
        return bytes(self.data[s])

    def __setitem__(self, s, v):
        self.data[s] = v

    def flush(self):
        pass

    def close(self):
        pass


def dirty_mapper(record):
    m = object.__new__(ClusterMapper)
    rec_abs = MFT_RECORD_VOLUME * MFT_RECORD_SIZE
    image = bytearray(MFT_RECORD_SIZE * 8)
    image[rec_abs:rec_abs + MFT_RECORD_SIZE] = record
    m.image = FakeImage(image)
    m._rec_offset = lambda n: n * MFT_RECORD_SIZE
    return m, rec_abs


def read_flags(m, rec_abs):
    rec = m._undo_fixups(bytearray(m.image[rec_abs:rec_abs + MFT_RECORD_SIZE]))
    attr = struct.unpack('<H', rec[20:22])[0]
    val_off = struct.unpack('<H', rec[attr + 20:attr + 22])[0]
    return struct.unpack('<H', rec[attr + val_off + 10:attr + val_off + 12])[0]


def test_clear_volume_dirty():
    print("_clear_volume_dirty_flag")

    # Dirty volume, with an unrelated flag that must survive.
    rec = make_volume_record(VOLUME_IS_DIRTY | 0x0008)
    m, rec_abs = dirty_mapper(rec)
    check("reports a clear", m._clear_volume_dirty_flag(), True)
    check("dirty bit gone", read_flags(m, rec_abs) & VOLUME_IS_DIRTY, 0)
    check("other flags kept", read_flags(m, rec_abs) & 0x0008, 0x0008)

    # Already clean: nothing to do, and the record must not be rewritten.
    rec = make_volume_record(0x0008)
    m, rec_abs = dirty_mapper(rec)
    before = m.image[rec_abs:rec_abs + MFT_RECORD_SIZE]
    check("clean volume is a no-op", m._clear_volume_dirty_flag(), False)
    check("record untouched", m.image[rec_abs:rec_abs + MFT_RECORD_SIZE], before)

    # Garbage record must not raise.
    m, rec_abs = dirty_mapper(bytearray(MFT_RECORD_SIZE))
    check("non-FILE record is a no-op", m._clear_volume_dirty_flag(), False)


if __name__ == '__main__':
    test_validate_path()
    test_clear_volume_dirty()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed")
