"""Unit test for $ATTRIBUTE_LIST-following in _extract_data_runs.

A fragmented file's $DATA runs spill into extension MFT records referenced
by an $ATTRIBUTE_LIST (0x20). This verifies the bridge gathers all extents,
ordered by VCN, instead of only the base record's first extent.

Run: python -m pytest tests/test_attribute_list.py -v
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import ClusterMapper, MFT_RECORD_SIZE
from ntfs_bridge.data_runs import encode_data_runs

CS = 4096


def build_data_record(runs):
    """A minimal 1024-byte MFT record with one unnamed non-resident $DATA.

    runs: list of (lcn, count).
    """
    rec = bytearray(MFT_RECORD_SIZE)
    rec[0:4] = b'FILE'
    first_attr = 56
    struct.pack_into('<H', rec, 20, first_attr)
    off = first_attr
    encoded = encode_data_runs([(count, lcn) for lcn, count in runs])
    runs_off = 64
    attr_len = runs_off + len(encoded)
    attr_len = (attr_len + 7) & ~7
    struct.pack_into('<I', rec, off, 0x80)       # type $DATA
    struct.pack_into('<I', rec, off + 4, attr_len)
    rec[off + 8] = 1                             # non-resident
    rec[off + 9] = 0                             # name_len
    struct.pack_into('<H', rec, off + 32, runs_off)
    total = sum(c for _, c in runs) * CS
    struct.pack_into('<Q', rec, off + 48, total)  # real_size
    rec[off + runs_off:off + runs_off + len(encoded)] = encoded
    struct.pack_into('<I', rec, off + attr_len, 0xFFFFFFFF)  # end marker
    return rec


def build_attribute_list(entries):
    """entries: list of (start_vcn, mft_ref) for unnamed $DATA (type 0x80)."""
    buf = bytearray()
    for start_vcn, mft_ref in entries:
        rec_len = 26  # no name
        e = bytearray(rec_len)
        struct.pack_into('<I', e, 0, 0x80)        # type
        struct.pack_into('<H', e, 4, rec_len)     # record length
        e[6] = 0                                  # name length
        e[7] = 26                                 # name offset
        struct.pack_into('<Q', e, 8, start_vcn)
        struct.pack_into('<Q', e, 16, mft_ref)    # base ref (low bits used)
        struct.pack_into('<H', e, 24, 0)          # attr id
        buf += e
    return bytes(buf)


class FakeMapper:
    def __init__(self):
        self.cluster_size = CS
        # Two extension records at fixed image offsets.
        self.image = bytearray(200 * MFT_RECORD_SIZE)
        self._recs = {}

    def place(self, rec_num, record):
        off = rec_num * MFT_RECORD_SIZE
        self.image[off:off + MFT_RECORD_SIZE] = record
        self._recs[rec_num] = off

    def _rec_offset(self, rec_num):
        return self._recs.get(rec_num)

    def _undo_fixups(self, rec):
        return rec  # no fixups in synthetic records

    _parse_data_runs = ClusterMapper._parse_data_runs
    _extract_data_runs_base = ClusterMapper._extract_data_runs_base
    _extract_data_runs_via_attrlist = ClusterMapper._extract_data_runs_via_attrlist


def test_attrlist_gathers_all_extents_in_vcn_order():
    m = FakeMapper()
    # extent 0 (base, record 30): clusters 1000..1099 ; VCN 0
    m.place(30, build_data_record([(1000, 100)]))
    # extent 1 (record 31): clusters 5000..5049 ; VCN 100
    m.place(31, build_data_record([(5000, 50)]))
    # extent 2 (record 32): clusters 200..219 ; VCN 150
    m.place(32, build_data_record([(200, 20)]))

    # Attribute list in scrambled order — must be sorted by VCN.
    al = build_attribute_list([(150, 32), (0, 30), (100, 31)])
    runs = m._extract_data_runs_via_attrlist(al)

    # Reconstruct absolute clusters, must be extent0 ++ extent1 ++ extent2.
    got = []
    for lcn, length in runs:
        got.extend(range(lcn, lcn + length))
    expected = list(range(1000, 1100)) + list(range(5000, 5050)) + list(range(200, 220))
    assert got == expected


def test_base_only_when_no_attrlist():
    m = FakeMapper()
    m.place(30, build_data_record([(1000, 100)]))
    base = m.image[30 * MFT_RECORD_SIZE:31 * MFT_RECORD_SIZE]
    runs = m._extract_data_runs_base(bytearray(base))
    got = []
    for lcn, length in runs:
        got.extend(range(lcn, lcn + length))
    assert got == list(range(1000, 1100))


def test_missing_extension_record_falls_back_to_none():
    m = FakeMapper()
    m.place(30, build_data_record([(1000, 100)]))
    # references record 99 which was never placed -> anomaly -> None
    al = build_attribute_list([(0, 30), (100, 99)])
    assert m._extract_data_runs_via_attrlist(al) is None
