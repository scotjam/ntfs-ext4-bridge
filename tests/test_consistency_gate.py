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
        control_host='127.0.0.1', control_port=0,
    )
    bridge.setup()
    bridge._start_two_way()
    try:
        # Mutate ext4 while "the VM" would normally be attached.
        (share / 'gone.bin').unlink()
        (share / 'new.bin').write_bytes(os.urandom(8192))
        with open(share / 'grow.bin', 'ab') as f:
            f.write(os.urandom(4096))
        newdir = share / 'newdir'
        newdir.mkdir()
        (newdir / 'nested.bin').write_bytes(os.urandom(512))

        # Bulk create: enough files that the offline ntfs-3g apply may grow
        # or relocate the $MFT. This is the case that exposed the "10->3
        # files" geometry-staleness bug — the gate refresh must re-derive
        # MFT geometry, not reuse the geometry captured at __init__.
        bulk = share / 'bulk'
        bulk.mkdir()
        for i in range(60):
            (bulk / f'file{i:03d}.bin').write_bytes(os.urandom(1500))

        expected_files = {'keep.bin', 'new.bin', 'grow.bin'}
        expected_files |= {f'file{i:03d}.bin' for i in range(60)}

        # Drive the REAL gate cycle. Stub the agent-side offline/online steps
        # (no VM here to take a disk offline); everything else — barrier,
        # quiesce, full msync, offline ntfs-3g apply, reload_from_image,
        # re-allocate, epoch bump — runs for real.
        gate = bridge.consistency_gate
        gate._go_offline = lambda: True
        gate._go_online = lambda via_agent: None
        gate.run_gate('test-bulk')

        # After a real gate, the mapper must track every ext4 file under the
        # share (this is what regressed to 3 before the fix).
        tracked = {os.path.basename(p)
                   for p in bridge.mapper.mft_record_to_source.values()}
        tracked |= {os.path.basename(bridge.mapper.source_dir + '/' + p)
                    for p in bridge.mapper.sparse_files}
        missing = expected_files - tracked
        assert not missing, (
            f"gate dropped {len(missing)} tracked files (geometry-staleness "
            f"regression): {sorted(missing)[:8]}")
        assert 'gone.bin' not in tracked

        # Verify the NTFS image itself is consistent and complete.
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
            assert {'keep.bin', 'new.bin', 'grow.bin', 'newdir', 'bulk'} <= names
            assert os.path.getsize(os.path.join(ntfs_share, 'grow.bin')) \
                == os.path.getsize(share / 'grow.bin')
            assert os.path.isfile(
                os.path.join(ntfs_share, 'newdir', 'nested.bin'))
            bulk_names = set(os.listdir(os.path.join(ntfs_share, 'bulk')))
            assert len(bulk_names) == 60, f"bulk dir has {len(bulk_names)}/60"
        finally:
            subprocess.run(['umount', verify_mnt], capture_output=True)
    finally:
        bridge.stop()
