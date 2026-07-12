"""Integration test: a real ClusterMapper recovers and replays an MFT op
that a previous (crashed) run acknowledged but never materialized.

Requires Linux + root + ntfs-3g/mkntfs. Skipped elsewhere.

Simulates the ack-before-durable crash: append an op to the journal + queue
it, then abandon the mapper WITHOUT letting the worker mark it done (as a
SIGKILL would). A fresh mapper over the same image must recover the pending
op and replay it through _mft_sync_ext4_passes.

Run (Linux host): sudo python -m pytest tests/test_op_journal_recovery.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

requires_linux_root = pytest.mark.skipif(
    sys.platform != 'linux' or (hasattr(os, 'geteuid') and os.geteuid() != 0),
    reason='needs Linux root + ntfs-3g')


@requires_linux_root
def test_pending_op_replayed_on_restart(tmp_path, monkeypatch):
    from ntfs_bridge.bridge import NTFSBridge
    from ntfs_bridge.cluster_mapper import ClusterMapper

    source = tmp_path / 'source'
    (source / 'Share').mkdir(parents=True)
    (source / 'Share' / 'seed.bin').write_bytes(os.urandom(4096))
    image = str(tmp_path / 'image.raw')
    mount = str(tmp_path / 'mnt'); os.makedirs(mount)

    bridge = NTFSBridge(image_path=image, source_dir=str(source),
                        ntfs_mount=mount, port=0, lazy_alloc=True,
                        dealloc_timeout=10**9, roots=['Share'])
    bridge.setup()
    mapper = bridge.mapper

    # A real MFT-region write (offset within the MFT). Use a valid FILE
    # record region so is_mft_region() routes it through the journal path.
    mft_off = mapper.mft_offset + 32 * 1024  # some user-record slot
    payload = mapper.image[mft_off:mft_off + 1024]  # real current bytes

    # Simulate: op acked+journaled+queued, but the process is killed before
    # the worker marks it done. We append directly and DON'T call append_done.
    seq = mapper._op_journal.append_op(mft_off, bytes(payload))
    assert seq >= 1
    # journal now has a pending op
    from ntfs_bridge.mft_op_journal import recover
    mapper._op_journal.sync()
    assert len(recover(mapper._op_journal_path)) == 1

    # Abandon this mapper as a crash would (don't drain the worker).
    mapper.close()

    # "Restart": a fresh mapper over the same image must recover + replay it.
    replayed = []
    orig = ClusterMapper._mft_sync_ext4_passes

    def spy(self, offset, data):
        replayed.append((offset, len(data)))
        return orig(self, offset, data)

    monkeypatch.setattr(ClusterMapper, '_mft_sync_ext4_passes', spy)

    mapper2 = ClusterMapper(image, str(source), roots=['Share'])
    try:
        assert (mft_off, 1024) in replayed, \
            "restart did not replay the pending MFT op"
        # After a clean run the journal is empty again.
        assert recover(mapper2._op_journal_path) == []
    finally:
        mapper2.close()
