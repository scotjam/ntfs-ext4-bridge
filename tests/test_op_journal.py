"""Unit tests for the two-way sync op journal (coalescing + persistence).

Run: python -m pytest tests/test_op_journal.py -v
No root, no VM, no inotify needed: events are injected directly and the
flush is driven synchronously via _flush_ready().
"""

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.op_journal import OpJournal
from ntfs_bridge.file_watcher import (EVENT_CREATE, EVENT_DELETE,
                                      EVENT_MODIFY, EVENT_MOVE)


class FakeMapper:
    def __init__(self):
        self.ntfs_sync_in_progress = set()
        self.ntfs_sync_timestamps = {}
        self.path_to_mft_record = {}
        self.known_root_entries = {'Share'}
        self._sync_lock = threading.Lock()
        self.ext4_sync_in_progress = set()
        self.echo_observed_callback = None
        self.remapped = []

    def remap_source_path(self, old_rel, new_rel):
        self.remapped.append((old_rel, new_rel))


@pytest.fixture
def env(tmp_path):
    source = tmp_path / 'source'
    share = source / 'Share'
    share.mkdir(parents=True)
    mapper = FakeMapper()
    journal = OpJournal(
        str(tmp_path / 'journal.jsonl'), str(source), mapper,
        quiesce_s=0.0, max_hold_s=0.0,
        gate_threshold_ops=10, gate_threshold_age=5.0,
    )
    yield journal, mapper, share
    with journal._cond:
        journal._journal_file.close()


def flush(journal):
    time.sleep(0.01)  # let last_ts age past quiesce_s=0
    journal._flush_ready()
    with journal._cond:
        return list(journal._ops)


def op_kinds(ops):
    return [op['op'] for op in ops]


def test_create_becomes_create_sized(env):
    journal, mapper, share = env
    f = share / 'movie.bin'
    f.write_bytes(b'x' * 1000)
    journal.on_event(EVENT_CREATE, 'Share/movie.bin')
    ops = flush(journal)
    assert op_kinds(ops) == ['create_sized', 'flush_volume']
    assert ops[0]['path'] == 'Share\\movie.bin'
    assert ops[0]['size'] == 1000
    assert ops[0]['mtime_ms'] > 0
    assert ops[0]['seq'] == 1


def test_known_path_modify_becomes_resize(env):
    journal, mapper, share = env
    f = share / 'doc.txt'
    f.write_bytes(b'y' * 2000)
    mapper.path_to_mft_record['Share/doc.txt'] = 42
    journal.on_event(EVENT_MODIFY, 'Share/doc.txt')
    ops = flush(journal)
    assert op_kinds(ops) == ['resize', 'flush_volume']
    assert ops[0]['size'] == 2000


def test_create_delete_annihilates_unknown_path(env):
    journal, mapper, share = env
    journal.on_event(EVENT_CREATE, 'Share/temp.tmp')
    journal.on_event(EVENT_DELETE, 'Share/temp.tmp')
    assert flush(journal) == []


def test_delete_of_known_path_emits_rm(env):
    journal, mapper, share = env
    mapper.path_to_mft_record['Share/old.txt'] = 7
    journal.on_event(EVENT_DELETE, 'Share/old.txt')
    ops = flush(journal)
    assert op_kinds(ops) == ['rm', 'flush_volume']
    assert ops[0]['recurse'] is True


def test_move_emits_mv(env):
    journal, mapper, share = env
    journal.on_event(EVENT_MOVE, 'Share/a.txt', 'Share/b.txt')
    ops = flush(journal)
    assert op_kinds(ops) == ['mv', 'flush_volume']
    assert ops[0]['path'] == 'Share\\a.txt'
    assert ops[0]['dst'] == 'Share\\b.txt'
    # The bridge remaps its own cluster mappings before dispatching the mv.
    assert ('Share/a.txt', 'Share/b.txt') in mapper.remapped


def test_dir_delete_subsumes_children(env):
    journal, mapper, share = env
    mapper.path_to_mft_record['Share/dir'] = 8
    mapper.path_to_mft_record['Share/dir/child.txt'] = 9
    journal.on_event(EVENT_DELETE, 'Share/dir/child.txt')
    journal.on_event(EVENT_DELETE, 'Share/dir')
    ops = flush(journal)
    rm_ops = [op for op in ops if op['op'] == 'rm']
    assert len(rm_ops) == 1
    assert rm_ops[0]['path'] == 'Share\\dir'


def test_dir_create_catches_up_children(env):
    journal, mapper, share = env
    d = share / 'newdir'
    d.mkdir()
    (d / 'inside.bin').write_bytes(b'z' * 50)
    journal.on_event(EVENT_CREATE, 'Share/newdir')
    ops = flush(journal)
    assert 'mkdir' in op_kinds(ops)
    # Child was re-enqueued by the catch-up walk; second flush emits it
    ops2 = flush(journal)
    kinds = op_kinds(ops2)
    assert 'create_sized' in kinds


def test_echo_filter_drops_bridge_own_writes(env):
    journal, mapper, share = env
    mapper.ntfs_sync_in_progress.add('Share/echo.txt')
    journal.on_event(EVENT_CREATE, 'Share/echo.txt')
    assert flush(journal) == []
    mapper.ntfs_sync_in_progress.clear()
    mapper.ntfs_sync_timestamps['Share/echo2.txt'] = time.time()
    journal.on_event(EVENT_CREATE, 'Share/echo2.txt')
    assert flush(journal) == []


def test_ack_advances_cursor_and_marks_dirty(env):
    journal, mapper, share = env
    f = share / 'a.bin'
    f.write_bytes(b'a' * 10)
    journal.on_event(EVENT_CREATE, 'Share/a.bin')
    ops = flush(journal)
    seqs = [op['seq'] for op in ops]
    journal.ack([{'seq': seqs[0], 'status': 'error', 'code': 'EFAIL',
                  'message': 'boom'},
                 {'seq': seqs[1], 'status': 'ok'}])
    stats = journal.stats()
    assert stats['undelivered'] == 0
    assert stats['acked_seq'] == seqs[-1]
    assert 'Share/a.bin' in journal.dirty_paths


def test_dirty_set_persists(env, tmp_path):
    journal, mapper, share = env
    journal.dirty_paths.add('Share/broken.bin')
    journal._save_dirty()
    journal2 = OpJournal(
        str(tmp_path / 'journal.jsonl'), str(share.parent), mapper)
    assert 'Share/broken.bin' in journal2.dirty_paths


def test_inject_drain_and_epoch_reset(env):
    journal, mapper, share = env
    journal.pause()
    journal.inject_op({'op': 'gate_begin', 'path': '', 'gate_id': 'g1',
                       '_rel': ''})
    ops = journal.ops_after(0)
    assert op_kinds(ops) == ['gate_begin']
    journal.ack([{'seq': ops[0]['seq'], 'status': 'ok'}])
    assert journal.drain_barrier(timeout=1.0) is True
    old_epoch = journal.epoch
    old_next = journal.stats()['next_seq']
    journal.reset_epoch()
    assert journal.epoch != old_epoch
    assert journal.stats()['next_seq'] == old_next  # seq stays monotonic
    assert journal.stats()['undelivered'] == 0


def test_escalation_fires_on_queue_depth(env):
    journal, mapper, share = env
    reasons = []
    journal.escalation_callback = reasons.append
    for i in range(15):
        f = share / f'f{i}.bin'
        f.write_bytes(b'q')
        journal.on_event(EVENT_CREATE, f'Share/f{i}.bin')
    flush(journal)
    journal._check_escalation()
    assert reasons, "escalation callback should have fired"


def test_long_poll_wakeup(env):
    journal, mapper, share = env

    def inject_later():
        time.sleep(0.2)
        journal.inject_op({'op': 'flush_volume', 'path': '', '_rel': ''})

    t = threading.Thread(target=inject_later)
    t.start()
    start = time.time()
    woke = journal.wait_for_ops(cursor=0, timeout=5.0)
    elapsed = time.time() - start
    t.join()
    assert woke is True
    assert elapsed < 2.0, "long-poll should wake on injection, not timeout"
