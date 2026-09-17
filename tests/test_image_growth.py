#!/usr/bin/env python3
"""Growing the image must not change the volume's identity.

Recreating the image would mint a new NTFS serial, which makes Windows remap
the drive letter and Backblaze re-upload every file on the volume. This drives
NTFSBridge._grow_ntfs_image over a real (small) NTFS filesystem and checks the
serial, the contents and the free space.

Needs root (mount) and mkfs.ntfs/ntfsresize.

    python3 tests/test_image_growth.py
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.bridge import NTFSBridge

failures = []


def check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
        failures.append(name)


def serial(path):
    """NTFS volume serial number: boot sector offset 0x48, 8 bytes."""
    with open(path, 'rb') as f:
        return f.read(512)[0x48:0x50].hex()


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main():
    if os.geteuid() != 0:
        print("SKIP: needs root to mount the test image")
        return 0

    tmp = tempfile.mkdtemp(prefix='grow-')
    img = os.path.join(tmp, 'test.img')
    mnt = os.path.join(tmp, 'mnt')
    os.makedirs(mnt)
    try:
        run(['truncate', '-s', str(512 * 1024 * 1024), img])
        r = run(['mkfs.ntfs', '-F', '-Q', '-c', '65536', img])
        if r.returncode != 0:
            print(f"SKIP: mkfs.ntfs unavailable: {r.stderr.strip()}")
            return 0

        before = serial(img)

        r = run(['mount', '-t', 'ntfs-3g', '-o', 'rw', img, mnt])
        if r.returncode != 0:
            print(f"SKIP: cannot mount ntfs-3g: {r.stderr.strip()}")
            return 0
        payload = b'x' * (3 * 1024 * 1024)
        with open(os.path.join(mnt, 'marker.bin'), 'wb') as f:
            f.write(payload)
        os.makedirs(os.path.join(mnt, '.bzvol'))
        with open(os.path.join(mnt, '.bzvol', 'bzvol_id.xml'), 'w') as f:
            f.write('<bzvolume vguid="v-test-identity" />')
        run(['umount', mnt])

        print("_grow_ntfs_image")

        bridge = object.__new__(NTFSBridge)
        bridge.image_path = img
        bridge.image_size_mb = 512

        # 512MB -> asked for 800MB, so 800 * GROW_HEADROOM.
        check("reports success", bridge._grow_ntfs_image(800), True)

        grown_mb = os.path.getsize(img) // (1024 * 1024)
        expected_mb = int(800 * 1024 * 1024 * NTFSBridge.GROW_HEADROOM) // (1024 * 1024)
        check("file is the headroomed size", grown_mb, expected_mb)
        check("image_size_mb updated", bridge.image_size_mb, expected_mb)

        # The whole point: same volume, not a new one.
        check("volume serial preserved", serial(img), before)

        run(['mount', '-t', 'ntfs-3g', '-o', 'rw', img, mnt])
        with open(os.path.join(mnt, 'marker.bin'), 'rb') as f:
            check("file contents intact", f.read(), payload)
        with open(os.path.join(mnt, '.bzvol', 'bzvol_id.xml')) as f:
            check("bzvol marker intact", 'v-test-identity' in f.read(), True)
        st = os.statvfs(mnt)
        total_mb = st.f_blocks * st.f_frsize // (1024 * 1024)
        check("filesystem really expanded", total_mb > 700, True)
        run(['umount', mnt])

        # Asking for no more than it already has must be a no-op.
        size_before = os.path.getsize(img)
        check("no-op when already big enough", bridge._grow_ntfs_image(100), True)
        check("untouched by the no-op", os.path.getsize(img), size_before)
        check("serial still preserved", serial(img), before)
    finally:
        run(['umount', mnt])
        run(['rm', '-rf', tmp])

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
