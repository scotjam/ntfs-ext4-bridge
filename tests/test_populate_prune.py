"""Populate must remove NTFS entries whose ext4 source is gone - but only
under the exposed roots, and never when "gone" might mean "unavailable".

Runs prune_stale_entries() against a real ntfs-3g mount of a small image,
because the thing being deleted is NTFS state. Needs root, mkfs.ntfs and
ntfs-3g; skips (exit 0 with a message) if they are missing.
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.bridge import (  # noqa: E402
    prune_stale_entries, PRUNE_MAX_FRACTION, PRUNE_REFUSE_FLOOR)

LOG = []


def log(msg):
    LOG.append(msg)
    print("   ", msg)


def touch(path, data=b'x'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)


def main():
    if os.geteuid() != 0 or not shutil.which('mkfs.ntfs') or not shutil.which('ntfs-3g'):
        print("SKIP test_populate_prune: needs root, mkfs.ntfs and ntfs-3g")
        return

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'src')
        image = os.path.join(d, 'image.raw')
        mnt = os.path.join(d, 'mnt')
        os.makedirs(mnt)

        # ---- ext4 source: what the volume SHOULD contain -------------------
        touch(os.path.join(src, 'Share', 'keep.mkv'))
        touch(os.path.join(src, 'Share', 'Season 1', 'e01.mkv'))
        touch(os.path.join(src, 'Empty', 'only.txt'))       # exposed, nearly all gone
        os.makedirs(os.path.join(src, 'Other'))              # NOT exposed
        os.symlink(os.path.join(d, 'nowhere'), os.path.join(src, 'Dangling'))

        # ---- NTFS volume: what it currently holds --------------------------
        subprocess.run(['truncate', '-s', '64M', image], check=True)
        subprocess.run(['mkfs.ntfs', '-F', '-q', image], check=True,
                       capture_output=True)
        subprocess.run(['mount', '-t', 'ntfs-3g', '-o', 'rw', image, mnt], check=True)
        try:
            # live entries (must survive)
            touch(os.path.join(mnt, 'Share', 'keep.mkv'))
            touch(os.path.join(mnt, 'Share', 'Season 1', 'e01.mkv'))
            # stale entries under an exposed root (must go)
            touch(os.path.join(mnt, 'Share', 'old.part1.rar'))
            touch(os.path.join(mnt, 'Share', 'OldFolder', 'gone.mkv'))
            touch(os.path.join(mnt, 'Share', 'Season 1', 'removed.mkv'))
            # volume-root items: Windows' space, never touched
            touch(os.path.join(mnt, 'System Volume Information', 'WPSettings.dat'))
            touch(os.path.join(mnt, '.bzvol', 'bzvol_id.xml'))
            touch(os.path.join(mnt, 'loose-at-root.txt'))
            # a root that is NOT exposed: never touched even though it differs
            touch(os.path.join(mnt, 'Other', 'stale-but-not-ours.txt'))
            # a root whose source is a dangling symlink: skipped, nothing removed
            touch(os.path.join(mnt, 'Dangling', 'a.mkv'))
            touch(os.path.join(mnt, 'Dangling', 'b.mkv'))
            # a root where pruning would remove >25% AND more than the floor:
            # refused. (Share above loses 3 of 6 - over the fraction but under
            # the floor - and must still be tidied.)
            n_gone = PRUNE_REFUSE_FLOOR + 10
            for i in range(n_gone):
                touch(os.path.join(mnt, 'Empty', f'vanished{i}.mkv'))
            touch(os.path.join(mnt, 'Empty', 'only.txt'))

            removed = prune_stale_entries(
                mnt, src, ['Share', 'Empty', 'Dangling'], log)

            # --- Share: exactly the three stale entries went -----------------
            assert not os.path.exists(os.path.join(mnt, 'Share', 'old.part1.rar'))
            assert not os.path.exists(os.path.join(mnt, 'Share', 'OldFolder'))
            assert not os.path.exists(os.path.join(mnt, 'Share', 'Season 1', 'removed.mkv'))
            assert os.path.exists(os.path.join(mnt, 'Share', 'keep.mkv'))
            assert os.path.exists(os.path.join(mnt, 'Share', 'Season 1', 'e01.mkv'))
            print("  ok: stale file, stale dir and stale nested file removed; live entries kept")

            # --- volume root and non-exposed roots untouched -----------------
            for p in ('System Volume Information/WPSettings.dat', '.bzvol/bzvol_id.xml',
                      'loose-at-root.txt', 'Other/stale-but-not-ours.txt'):
                assert os.path.exists(os.path.join(mnt, p)), f"{p} was removed"
            print("  ok: volume-root items and non-exposed roots never touched")

            # --- dangling source: treated as unavailable ---------------------
            assert os.path.exists(os.path.join(mnt, 'Dangling', 'a.mkv'))
            assert os.path.exists(os.path.join(mnt, 'Dangling', 'b.mkv'))
            assert any('skipping Dangling' in m for m in LOG), "no skip logged for Dangling"
            print("  ok: a root whose source is a dangling symlink is skipped, not emptied")

            # --- mass loss: refused --------------------------------------------
            assert all(os.path.exists(os.path.join(mnt, 'Empty', f'vanished{i}.mkv'))
                       for i in range(n_gone)), "refused root lost entries anyway"
            assert any('REFUSING' in m and 'Empty' in m for m in LOG), "no refusal logged"
            assert not any('REFUSING' in m and 'Share' in m for m in LOG), \
                "a small share losing a few entries was refused"
            print(f"  ok: pruning >{int(PRUNE_MAX_FRACTION*100)}% AND >{PRUNE_REFUSE_FLOOR} "
                  f"entries of a root is refused and logged; a small tidy is not")

            assert removed == 3, f"expected 3 removals, got {removed}"
        finally:
            subprocess.run(['umount', mnt], capture_output=True)

        # The volume must still be consistent after pruning through ntfs-3g.
        r = subprocess.run(['ntfsfix', '-n', image], capture_output=True, text=True)
        assert 'does not match' not in r.stdout + r.stderr, r.stdout + r.stderr
        print("  ok: volume consistent after prune")

    print("PASS test_populate_prune")


if __name__ == '__main__':
    main()
