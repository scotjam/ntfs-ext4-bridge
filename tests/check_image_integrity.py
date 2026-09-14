#!/usr/bin/env python3
"""Report structural problems in a bridge NTFS image. Read-only.

    check_image_integrity.py IMAGE [--offset BYTES]

IMAGE is the bare volume file the bridge serves, a backup of it, or the
partition device of a running bridge (/dev/nbd0p1). With the bridge running,
prefer the device: the file may lag the bridge's RAM cache.

Exit status is 1 if anything is found, so it can gate a deploy.
See ntfs_bridge/imgcheck.py for what each class of finding means.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge import imgcheck  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        sys.exit(__doc__)
    off = 0
    if '--offset' in sys.argv:
        off = int(sys.argv[sys.argv.index('--offset') + 1])
    vol = imgcheck.Volume(args[0], part_offset=off)
    print(f"cluster={vol.csz} mft_lcn={vol.mft_lcn} clusters={vol.total_clusters} "
          f"records={len(vol.recs)} ($MFT in {len(vol.mft_map)} runs)")
    problems = 0

    cross, hits = imgcheck.find_cross_links(vol)
    n_clusters = sum(len(v) for v in cross.values())
    print(f"\n=== A. cross-linked clusters: {n_clusters} clusters across {len(cross)} files ===")
    for n, lst in sorted(cross.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"  file rec {n:6} {len(lst):5} clusters  {vol.fullpath(n)[:70]}")
    if hits:
        print(f"  metadata records hit: {len(hits)}")
        for mn, k in sorted(hits.items(), key=lambda kv: -kv[1])[:10]:
            kind = 'DIR ' if vol.recs[mn].is_dir else 'SYS '
            print(f"    rec {mn:6} {k:4} clusters  {kind}{vol.fullpath(mn)[:70]}")
    problems += n_clusters

    broken = imgcheck.find_broken_attrlist_dirs(vol)
    print(f"\n=== B. directories with a broken $ATTRIBUTE_LIST: {len(broken)} ===")
    for n, exts in broken:
        print(f"  rec {n:6}  exts={exts[:4]}  {vol.fullpath(n)[:70]}")
    problems += len(broken)

    ext_free = [n for n, r in vol.recs.items() if r.base and not r.in_use]
    print(f"\n=== C. extension records: {sum(1 for r in vol.recs.values() if r.base)}, "
          f"not in use: {len(ext_free)} ===")

    bad_idx = imgcheck.find_broken_index_dirs(vol)
    print(f"\n=== D. directories with a broken $I30 index: {len(bad_idx)} ===")
    for n, reasons in bad_idx:
        print(f"  rec {n:6}  {vol.fullpath(n)[:70]}")
        for why in reasons[:4]:
            print(f"        {why}")
    problems += len(bad_idx)

    vol.close()
    print(f"\n{'CLEAN' if not problems else f'{problems} problem(s)'}")
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
