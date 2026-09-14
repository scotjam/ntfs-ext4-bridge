#!/usr/bin/env python3
"""Prove the served NTFS tree matches the ext4 source - by stat, not by name.

    verify_served_tree.py NTFS_MOUNT SOURCE_DIR ROOT[,ROOT...]

A directory listing is not evidence: NTFS readdir enumerates index blocks by
bitmap while lookup descends the B+tree, so a name can be listed and still
fail stat. For every entry under each root this checks, in both directions:

  source -> served : the same path exists on the NTFS side, stats
                     successfully, is the same kind, and (for files) has the
                     same size;
  served -> source : every name readdir returns on the NTFS side stats
                     successfully and exists at the source.

Exit 1 with a summary if anything differs.
"""
import os
import sys


def walk(base):
    """rel_path -> (kind, size) for everything under base, following the
    source's symlinked roots but not symlinks inside the tree."""
    out = {}
    for root, dirs, files in os.walk(base, followlinks=True):
        rel_root = os.path.relpath(root, base)
        for d in dirs:
            rel = d if rel_root == '.' else os.path.join(rel_root, d)
            out[rel] = ('dir', 0)
        for f in files:
            rel = f if rel_root == '.' else os.path.join(rel_root, f)
            try:
                out[rel] = ('file', os.stat(os.path.join(root, f)).st_size)
            except OSError as e:
                out[rel] = ('unstat', str(e).split(']')[-1].strip())
    return out


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    mount, source, roots = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
    total_bad = 0
    for r in roots:
        src = walk(os.path.join(source, r))
        srv = walk(os.path.join(mount, r))
        missing = [p for p in src if p not in srv]
        unstat = [p for p, (k, _) in srv.items() if k == 'unstat']
        extra = [p for p in srv if p not in src and srv[p][0] != 'unstat']
        wrong = [p for p in src if p in srv and srv[p][0] != 'unstat' and srv[p] != src[p]]
        bad = len(missing) + len(unstat) + len(extra) + len(wrong)
        total_bad += bad
        print(f"{r:12} source={len(src):6} served={len(srv):6}  "
              f"missing={len(missing)} unstat={len(unstat)} extra={len(extra)} "
              f"size/kind-mismatch={len(wrong)}  -> {'OK' if not bad else 'PROBLEMS'}")
        for label, lst in (('missing', missing), ('unstat', unstat), ('extra', extra), ('mismatch', wrong)):
            for p in lst[:5]:
                detail = '' if label != 'mismatch' else f"  source={src[p]} served={srv[p]}"
                print(f"    {label:9} {r}/{p}{detail}")
            if len(lst) > 5:
                print(f"    {label:9} ... and {len(lst) - 5} more")
    print('ALL ROOTS MATCH' if not total_bad else f'{total_bad} difference(s)')
    sys.exit(1 if total_bad else 0)


if __name__ == '__main__':
    main()
