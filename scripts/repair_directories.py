#!/usr/bin/env python3
"""Rebuild directories the bridge cannot serve correctly, so populate can
refill them from ext4.

    repair_directories.py IMAGE            # report what would be rebuilt
    repair_directories.py IMAGE --apply    # rewrite them

Run with the bridge STOPPED, on the bare volume file it serves. Two classes
are repaired, both by rewriting the directory record as an empty directory
that keeps its $STANDARD_INFORMATION and $FILE_NAME:

  - a $ATTRIBUTE_LIST pointing at extension records that are gone, so the
    directory has no reachable $INDEX_ROOT ("Index root attribute missing");
  - a $I30 index holding an entry that resolves to nothing (listed, but
    stat fails and unlink cannot remove it) or the same name twice.

Both are what a directory looks like after its index block or attribute
list was served as, or overwritten by, file data. The children live on
ext4; the next bridge start recreates them. Take a backup first: the script
does not.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge import imgcheck  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        sys.exit(__doc__)
    apply = '--apply' in sys.argv
    roots = []
    if '--roots' in sys.argv:
        roots = [r.strip().lower() for r in sys.argv[sys.argv.index('--roots') + 1].split(',') if r.strip()]
    if apply and not roots:
        sys.exit("--apply needs --roots A,B,C: only directories under those top-level "
                 "roots are rebuilt. Their children live on ext4 and come back; "
                 "anything Windows keeps at the volume root does not, so it is never touched.")
    vol = imgcheck.Volume(args[0], writable=apply)

    attrlist = imgcheck.find_broken_attrlist_dirs(vol)
    index = imgcheck.find_broken_index_dirs(vol)
    todo = {}
    for n, exts in attrlist:
        todo.setdefault(n, []).append(f"attribute list -> missing extension records {exts[:3]}")
    for n, reasons in index:
        todo.setdefault(n, []).extend(reasons[:3])

    def under_roots(n):
        top = vol.fullpath(n).split('/', 1)[0].lower()
        return top in roots

    print(f"{len(todo)} director{'y' if len(todo) == 1 else 'ies'} with a broken attribute list or index")
    skipped = []
    for n in sorted(todo):
        tag = '' if not roots or under_roots(n) else '   [outside --roots: left alone]'
        if tag:
            skipped.append(n)
        print(f"  rec {n:6}  {vol.fullpath(n)}{tag}")
        for why in todo[n]:
            print(f"        {why}")

    if not apply:
        print("\nReport only. Re-run with --apply --roots A,B,C to rebuild them as empty directories.")
        vol.close()
        return
    for n in skipped:
        todo.pop(n, None)

    done = 0
    for n in sorted(todo):
        if imgcheck.rebuild_dir_as_empty(vol, n):
            done += 1
            print(f"  rebuilt rec {n}: {vol.fullpath(n)}")
        else:
            print(f"  rec {n}: no resident $STANDARD_INFORMATION/$FILE_NAME to keep - skipped")
    vol.close()
    print(f"\nRebuilt {done}. Verify with: ntfsfix -n IMAGE ; then start the bridge so populate refills them.")


if __name__ == '__main__':
    main()
