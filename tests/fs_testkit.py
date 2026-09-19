"""Test kit for the bridge: build an edge-case ext4 tree, and prove it never changes.

The bridge has two sources of truth - NTFS structure in the image, file data in
ext4 - and every corruption seen so far came from resolving a disagreement
between them the wrong way, with no crash and (twice) with no guest attached.
So the invariant worth enforcing is not "the journal replays correctly", it is:

    ext4 is unchanged, byte for byte, except where a guest legitimately wrote.

manifest() captures enough to catch every failure mode seen: content hash
(zero-filled rewrites, tail truncation), size (cluster-rounding inflation),
inode (a file replaced rather than modified) and the set of paths (deletions,
and files re-parented into the wrong directory).

Usage:
    python3 fs_testkit.py build   <dir>              build the edge-case tree
    python3 fs_testkit.py snap    <dir> <out.json>   record a manifest
    python3 fs_testkit.py compare <a.json> <b.json>  diff two manifests
"""
import hashlib
import io
import json
import os
import stat
import sys

CLUSTER = 65536


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def build_tree(root):
    """An ext4 tree shaped to hit the specific things that have gone wrong."""
    os.makedirs(root, exist_ok=True)
    made = []

    def d(*parts):
        p = os.path.join(root, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    def f(path, data):
        _write(path, data)
        made.append(path)

    # --- sizes around the cluster boundary.
    # Tail truncation inflated files to an exact cluster multiple and zeroed
    # the added tail, so sizes either side of the boundary are the interesting
    # ones. 0 and 1 byte files are resident in the MFT, a different code path.
    sizes = d("sizes")
    for n in (0, 1, 2, 511, 512, 4095, 4096, 4097,
              CLUSTER - 1, CLUSTER, CLUSTER + 1,
              2 * CLUSTER - 1, 2 * CLUSTER, 2 * CLUSTER + 1):
        f(os.path.join(sizes, "size_%d.bin" % n),
          bytes((i * 7 + 11) & 0xFF for i in range(n)))

    # --- content that looks like damage, so a guard keying on content is
    # forced to distinguish "legitimately all zeros" from "we lost this".
    content = d("content")
    f(os.path.join(content, "all_zero_64k.bin"), bytes(CLUSTER))
    f(os.path.join(content, "all_zero_small.bin"), bytes(100))
    f(os.path.join(content, "all_ff_64k.bin"), b"\xff" * CLUSTER)
    f(os.path.join(content, "zero_then_data.bin"), bytes(CLUSTER) + b"REAL" * 100)
    f(os.path.join(content, "data_then_zero.bin"), b"REAL" * 100 + bytes(CLUSTER))
    # An NTFS index block signature in file data: a naive scan must not mistake
    # file content for structure.
    f(os.path.join(content, "looks_like_indx.bin"), b"INDX\x28\x00\x09\x00" + bytes(4088))
    f(os.path.join(content, "looks_like_file_record.bin"), b"FILE\x30\x00\x03\x00" + bytes(1016))

    # --- a directory big enough to push $INDEX_BITMAP non-resident, which is
    # exactly the transition that corrupted allocated_size.
    big = d("big_dir")
    for i in range(1200):
        f(os.path.join(big, "entry_%04d.txt" % i), b"x%04d" % i)

    # --- names. A Windows-facing layer has to cope with names Windows itself
    # would refuse, because ext4 allows them.
    names = d("names")
    f(os.path.join(names, "with spaces.txt"), b"a")
    f(os.path.join(names, "trailing.space .txt"), b"b")
    f(os.path.join(names, "dots...txt"), b"c")
    f(os.path.join(names, ".hidden"), b"d")
    f(os.path.join(names, "éèê-accents.txt"), b"e")
    f(os.path.join(names, "日本語.txt"), b"f")
    f(os.path.join(names, "emoji-\U0001f600.txt"), b"g")
    f(os.path.join(names, "CON.txt"), b"h")       # reserved on Windows
    f(os.path.join(names, "NUL.txt"), b"i")
    f(os.path.join(names, "a" * 200 + ".txt"), b"j")
    f(os.path.join(names, "UPPER.TXT"), b"k")
    f(os.path.join(names, "upper.txt"), b"l")     # case-collides on NTFS

    # --- depth, and empty directories (no INDX allocation at all).
    deep = root
    for i in range(20):
        deep = d(os.path.relpath(deep, root), "lvl%02d" % i) if i else d("deep", "lvl00")
    f(os.path.join(deep, "bottom.txt"), b"deep")
    d("empty_dir")
    d("empty_parent", "empty_child")

    # --- sparse file: a real hole, which must stay a hole.
    sp = os.path.join(root, "sparse.bin")
    with open(sp, "wb") as fh:
        fh.truncate(4 * CLUSTER)
        fh.seek(3 * CLUSTER)
        fh.write(b"END")
    made.append(sp)

    # --- modes and links.
    links = d("links")
    target = os.path.join(links, "target.txt")
    f(target, b"target")
    os.link(target, os.path.join(links, "hardlink.txt"))
    os.symlink("target.txt", os.path.join(links, "symlink.txt"))
    ro = os.path.join(links, "readonly.txt")
    f(ro, b"ro")
    os.chmod(ro, 0o444)

    return len(made)


def manifest(root):
    """Everything about the tree that must survive a bridge run untouched."""
    out = {"root": root, "files": {}, "dirs": [], "symlinks": {}}
    for dp, dn, fn in os.walk(root):
        dn.sort()
        rel_d = os.path.relpath(dp, root)
        out["dirs"].append(rel_d)
        for name in sorted(fn):
            p = os.path.join(dp, name)
            rel = os.path.relpath(p, root)
            try:
                st = os.lstat(p)
            except OSError as e:
                out["files"][rel] = {"error": str(e)}
                continue
            if stat.S_ISLNK(st.st_mode):
                out["symlinks"][rel] = os.readlink(p)
                continue
            h = hashlib.sha256()
            try:
                with open(p, "rb") as fh:
                    while True:
                        b = fh.read(1 << 20)
                        if not b:
                            break
                        h.update(b)
            except OSError as e:
                out["files"][rel] = {"error": str(e)}
                continue
            out["files"][rel] = {
                "size": st.st_size,
                "sha256": h.hexdigest(),
                "inode": st.st_ino,
                "mode": stat.filemode(st.st_mode),
                "nlink": st.st_nlink,
                "blocks": st.st_blocks,
            }
    out["dirs"].sort()
    return out


def compare(a, b, allow_changed=()):
    """Report every difference. allow_changed lists paths a guest may alter."""
    allow = set(allow_changed)
    problems = []

    fa, fb = a["files"], b["files"]
    for rel in sorted(set(fa) - set(fb)):
        problems.append(("DELETED", rel, "%d bytes" % fa[rel].get("size", -1)))
    for rel in sorted(set(fb) - set(fa)):
        problems.append(("APPEARED", rel, "%d bytes" % fb[rel].get("size", -1)))
    for rel in sorted(set(fa) & set(fb)):
        if rel in allow:
            continue
        x, y = fa[rel], fb[rel]
        if x.get("sha256") != y.get("sha256"):
            problems.append(("CONTENT", rel,
                             "%s(%s B) -> %s(%s B)" % (str(x.get("sha256"))[:12],
                                                       x.get("size"),
                                                       str(y.get("sha256"))[:12],
                                                       y.get("size"))))
        elif x.get("size") != y.get("size"):
            problems.append(("SIZE", rel, "%s -> %s" % (x.get("size"), y.get("size"))))
        if x.get("inode") != y.get("inode"):
            problems.append(("REPLACED", rel,
                             "inode %s -> %s" % (x.get("inode"), y.get("inode"))))
        if x.get("mode") != y.get("mode"):
            problems.append(("MODE", rel, "%s -> %s" % (x.get("mode"), y.get("mode"))))

    for rel in sorted(set(a["symlinks"]) ^ set(b["symlinks"])):
        problems.append(("SYMLINK", rel, "added or removed"))
    for rel in sorted(set(a["symlinks"]) & set(b["symlinks"])):
        if a["symlinks"][rel] != b["symlinks"][rel]:
            problems.append(("SYMLINK", rel,
                             "%s -> %s" % (a["symlinks"][rel], b["symlinks"][rel])))

    da, db = set(a["dirs"]), set(b["dirs"])
    for rel in sorted(da - db):
        problems.append(("DIR DELETED", rel, ""))
    for rel in sorted(db - da):
        problems.append(("DIR APPEARED", rel, ""))
    return problems


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "build":
        n = build_tree(sys.argv[2])
        print("built %d files under %s" % (n, sys.argv[2]))
        return 0
    if cmd == "snap":
        m = manifest(sys.argv[2])
        io.open(sys.argv[3], "w", encoding="utf-8").write(json.dumps(m, indent=1))
        print("snapshot: %d files, %d dirs, %d symlinks -> %s"
              % (len(m["files"]), len(m["dirs"]), len(m["symlinks"]), sys.argv[3]))
        return 0
    if cmd == "compare":
        a = json.load(io.open(sys.argv[2], encoding="utf-8"))
        b = json.load(io.open(sys.argv[3], encoding="utf-8"))
        probs = compare(a, b)
        if not probs:
            print("ext4 UNCHANGED: %d files verified byte-for-byte" % len(a["files"]))
            return 0
        print("ext4 CHANGED - %d problem(s):" % len(probs))
        for kind, rel, detail in probs[:60]:
            print("   %-12s %-60s %s" % (kind, rel[:60], detail))
        if len(probs) > 60:
            print("   ... and %d more" % (len(probs) - 60))
        return 1
    print("unknown command %r" % cmd)
    return 2


if __name__ == "__main__":
    sys.exit(main())
