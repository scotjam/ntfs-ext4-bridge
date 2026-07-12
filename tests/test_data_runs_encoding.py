"""Regression tests for data-run LCN encoding on large (>2^31 cluster)
volumes — the 17TB target. `_encode_data_runs_simple` previously capped the
LCN-offset width at 4 bytes, silently misdirecting virtual-INDX runs (which
sit at the TOP of the volume) to a mid-volume cluster.

Run: python -m pytest tests/test_data_runs_encoding.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.data_runs import decode_data_runs
from ntfs_bridge.cluster_mapper import ClusterMapper

_encode = ClusterMapper._encode_data_runs_simple


class _Dummy:
    pass


def _roundtrip(clusters):
    raw = _encode(_Dummy(), clusters)
    return decode_data_runs(raw)


def test_small_lcn():
    assert _roundtrip([100]) == [(1, 100)]


def test_lcn_above_2gb_boundary():
    # Just past 2^31 clusters — the old 4-byte cap broke here.
    lcn = 2**31 + 12345
    assert _roundtrip([lcn]) == [(1, lcn)]


def test_lcn_near_17tb_top():
    # ~4.56e9 clusters = top of a 17TB volume at 4KB clusters.
    for lcn in (4_563_000_000, 5_000_000_000):
        assert _roundtrip([lcn]) == [(1, lcn)], f"lcn={lcn}"


def test_contiguous_run_large_lcn():
    lcn = 4_000_000_000
    clusters = list(range(lcn, lcn + 50))
    decoded = _roundtrip(clusters)
    assert decoded == [(50, lcn)]


def test_multiple_runs_large_lcn():
    a, b = 4_000_000_000, 4_200_000_000
    clusters = [a, a + 1, a + 2, b, b + 1]
    decoded = _roundtrip(clusters)
    # Reconstruct absolute cluster list from (count, lcn) pairs.
    got = []
    for count, lcn in decoded:
        got.extend(range(lcn, lcn + count))
    assert got == clusters
