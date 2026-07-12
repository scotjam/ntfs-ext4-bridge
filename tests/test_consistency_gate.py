"""Integration test for the consistency gate (no VM required).

Requires: Linux, root, ntfs-3g, mkntfs. Skipped elsewhere.

Builds a small source tree, creates an NTFS image via the bridge's own
populate path, runs a gate cycle with a protocol-level fake agent (nothing
to take offline — no VM), mutates ext4 in between, and verifies the NTFS
view matches ext4 afterwards by mounting the image read-only.

Run (on the Linux host):
  sudo python -m pytest tests/test_consistency_gate.py -v
"""

import hashlib
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

requires_linux_root = pytest.mark.skipif(
    sys.platform != 'linux' or os.geteuid() != 0
    if hasattr(os, 'geteuid') else True,
    reason='needs Linux root + ntfs-3g')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@requires_linux_root
def test_gate_reconciles_ext4_changes(tmp_path):
    from ntfs_bridge.bridge import NTFSBridge

    source = tmp_path / 'source'
    share = source / 'Share'
    share.mkdir(parents=True)
    (share / 'keep.bin').write_bytes(os.urandom(4096))
    (share / 'gone.bin').write_bytes(os.urandom(2048))
    (share / 'grow.bin').write_bytes(os.urandom(1024))

    image = str(tmp_path / 'image.raw')
    mount = str(tmp_path / 'mnt')
    os.makedirs(mount)

    bridge = NTFSBridge(
        image_path=image, source_dir=str(source), ntfs_mount=mount,
        port=0, lazy_alloc=True, dealloc_timeout=10**9,
        roots=['Share'], two_way=True,
    )
    bridge.setup()
    bridge._start_two_way()
    try:
        # Mutate ext4 while "the VM" would normally be attached
        (share / 'gone.bin').unlink()
        (share / 'new.bin').write_bytes(os.urandom(8192))
        with open(share / 'grow.bin', 'ab') as f:
            f.write(os.urandom(4096))
        newdir = share / 'newdir'
        newdir.mkdir()
        (newdir / 'nested.bin').write_bytes(os.urandom(512))

        # Run the gate directly (fake agent: no disk to offline). Force the
        # offline step to no-op by pre-confirming the agent phase.
        gate = bridge.consistency_gate
        gate.journal = bridge.op_journal
        gate._agent_confirmed.set()          # pretend agent confirmed
        gate._agent_online_confirmed.set()
        bridge.op_journal.pause()
        gate.gate_id = 'test-gate'
        gate.mapper._mft_queue.join()
        gate.mapper.gate_active.set()
        gate.mapper.flush()
        try:
            gate._apply_offline()
            gate.mapper.image.reload()
            gate.mapper.rescan_mft()
            bridge._allocate_new_sparse_files()
        finally:
            gate.mapper.gate_active.clear()

        # Verify by mounting the image read-only
        verify_mnt = str(tmp_path / 'verify')
        os.makedirs(verify_mnt)
        subprocess.run(['ntfsfix', image], capture_output=True)
        r = subprocess.run(['mount', '-t', 'ntfs-3g', '-o', 'ro',
                            image, verify_mnt],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        try:
            ntfs_share = os.path.join(verify_mnt, 'Share')
            names = set(os.listdir(ntfs_share))
            assert 'gone.bin' not in names
            assert {'keep.bin', 'new.bin', 'grow.bin', 'newdir'} <= names
            assert os.path.getsize(os.path.join(ntfs_share, 'grow.bin')) \
                == os.path.getsize(share / 'grow.bin')
            assert os.path.isfile(
                os.path.join(ntfs_share, 'newdir', 'nested.bin'))
            # NOTE: content hashes are validated through the NBD path in the
            # live runbook; the direct image mount only holds metadata for
            # bridge-mapped files, so only sizes/names are asserted here.
        finally:
            subprocess.run(['umount', verify_mnt], capture_output=True)
    finally:
        bridge.stop()
