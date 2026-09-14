"""Populate must refresh an entry whose source changed size.

Reads always come from ext4, but the size Windows sees is the one recorded
in the MFT when the entry was made. With live sync down for days, sixteen
files drifted: a pointer file that grew by one byte was served one byte
short, and a spreadsheet that grew by 9KB was served truncated. Populate
used to skip any existing entry outright; now it compares sizes.

Needs root, mkfs.ntfs and ntfs-3g; skips otherwise.
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.bridge import NTFSBridge  # noqa: E402


def sh(*cmd):
    return subprocess.run(list(cmd), check=True, capture_output=True, text=True)


def sizes(image, mnt, *rels):
    sh('mount', '-t', 'ntfs-3g', '-o', 'ro', image, mnt)
    try:
        return [os.path.getsize(os.path.join(mnt, r)) for r in rels]
    finally:
        sh('umount', mnt)


def main():
    if os.geteuid() != 0 or not shutil.which('mkfs.ntfs') or not shutil.which('ntfs-3g'):
        print("SKIP test_populate_resize: needs root, mkfs.ntfs and ntfs-3g")
        return
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, 'src'); os.makedirs(os.path.join(src, 'Share'))
        image = os.path.join(tmp, 'image.raw'); mnt = os.path.join(tmp, 'mnt'); os.makedirs(mnt)
        small = os.path.join(src, 'Share', 'pointer.strm')     # < 700 B: copied
        big = os.path.join(src, 'Share', 'sheet.xlsx')         # > 700 B: sparse/truncate
        open(small, 'wb').write(b'x' * 251)
        open(big, 'wb').write(b'y' * 133468)
        sh('truncate', '-s', '256M', image); sh('mkfs.ntfs', '-F', '-q', image)

        bridge = NTFSBridge(image_path=image, source_dir=src, ntfs_mount=mnt,
                            port=10999, lazy_alloc=True, protected_roots=['Share'])
        bridge._populate_image(needs_fsfix=False)
        assert sizes(image, mnt, 'Share/pointer.strm', 'Share/sheet.xlsx') == [251, 133468]
        print("  ok: first populate creates entries at source size")

        # The source changes: the pointer grows by one byte, the sheet by 9KB,
        # and a third file shrinks.
        open(small, 'wb').write(b'x' * 252)
        open(big, 'wb').write(b'y' * 142644)
        bridge._populate_image(needs_fsfix=False)
        got = sizes(image, mnt, 'Share/pointer.strm', 'Share/sheet.xlsx')
        assert got == [252, 142644], f"sizes not refreshed: {got}"
        print("  ok: second populate refreshes both a copied and a sparse entry to the new size")

        open(big, 'wb').write(b'y' * 70000)
        bridge._populate_image(needs_fsfix=False)
        assert sizes(image, mnt, 'Share/sheet.xlsx') == [70000]
        print("  ok: a file that shrank is shrunk")

        r = subprocess.run(['ntfsfix', '-n', image], capture_output=True, text=True)
        assert 'does not match' not in r.stdout + r.stderr
        print("  ok: volume consistent")
    finally:
        subprocess.run(['umount', os.path.join(tmp, 'mnt')], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS test_populate_resize")


if __name__ == '__main__':
    main()
