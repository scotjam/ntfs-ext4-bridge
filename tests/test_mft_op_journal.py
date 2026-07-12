"""Unit tests for the persistent MFT->ext4 op journal (crash recovery).

Run: python -m pytest tests/test_mft_op_journal.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge import mft_op_journal
from ntfs_bridge.mft_op_journal import MftOpJournal, recover


def test_append_and_recover_pending(tmp_path):
    p = str(tmp_path / 'j.mftops')
    j = MftOpJournal(p)
    s1 = j.append_op(1024, b'A' * 1024)
    s2 = j.append_op(2048, b'B' * 512)
    j.append_done(s1)          # s1 materialized, s2 still pending
    j.sync()
    j.close()

    pending = recover(p)
    assert len(pending) == 1
    seq, off, data = pending[0]
    assert seq == s2 and off == 2048 and data == b'B' * 512


def test_all_done_truncates(tmp_path):
    p = str(tmp_path / 'j.mftops')
    j = MftOpJournal(p)
    a = j.append_op(0, b'x' * 100)
    b = j.append_op(4096, b'y' * 100)
    j.append_done(a)
    j.append_done(b)           # nothing pending -> file truncated to empty
    j.sync()
    assert os.path.getsize(p) == 0
    j.close()
    assert recover(p) == []


def test_recover_preserves_seq_order(tmp_path):
    p = str(tmp_path / 'j.mftops')
    j = MftOpJournal(p)
    seqs = [j.append_op(i * 1024, bytes([i]) * 10) for i in range(5)]
    j.append_done(seqs[1])
    j.append_done(seqs[3])
    j.close()
    pending = recover(p)
    got = [s for s, _, _ in pending]
    assert got == [seqs[0], seqs[2], seqs[4]]  # ascending, done ones removed


def test_recover_tolerates_torn_tail(tmp_path):
    p = str(tmp_path / 'j.mftops')
    j = MftOpJournal(p)
    s1 = j.append_op(0, b'good' * 4)
    j.close()
    # Simulate a crash mid-append: append a partial op record.
    with open(p, 'ab') as f:
        f.write(b'O' + b'\x01\x02\x03')  # truncated header
    pending = recover(p)
    assert len(pending) == 1 and pending[0][0] == s1


def test_recover_missing_file(tmp_path):
    assert recover(str(tmp_path / 'nope.mftops')) == []
