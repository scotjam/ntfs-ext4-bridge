"""Run the bridge against the real library with every ext4 write intercepted.

The library is the lowerdir of an overlayfs. The kernel never writes to a lower
layer, so the real files cannot be touched whatever the bridge does - but the
bridge is not told that, so every write path runs normally instead of hitting
EROFS and taking an error branch that might hide the bug.

Afterwards the upperdir IS the report: every file the bridge created, rewrote or
(via a whiteout) tried to delete. An empty upperdir means it attempted nothing.

Two independent checks, because either alone can mislead:
  - the real tree's stat manifest must be unchanged (it cannot change, but
    prove it rather than assume the mount was set up right)
  - the upperdir must contain nothing but the directories overlay itself needs

Hashing 1.1 TB would take hours, so the manifest is stat-only: size, mtime,
inode, nlink, mode. Content cannot change without one of those moving, since
the bridge cannot write in place through a lower layer.

    setup    build the overlay and snapshot the library
    run      serve it, read everything, stop
    verify   compare manifests and report the upperdir
    teardown unmount the overlay
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import time

D2 = "/srv/dev-disk-by-uuid-0d919a0b-021a-4259-8c43-3696c1486ab7"
LOWER = os.path.join(D2, "kidstv2")
BASE = os.path.join(D2, "bridge-ro")
SHARE = "Shows"                      # neutral name; the real one stays private
PORT = 10810

UPPER = os.path.join(BASE, "upper", SHARE)
WORK = os.path.join(BASE, "work", SHARE)
MERGED = os.path.join(BASE, "merged", SHARE)
SOURCE = os.path.join(BASE, "source")
OVERFLOW = os.path.join(BASE, "overflow")
IMAGE = os.path.join(BASE, "image.raw")
MNT = os.path.join(BASE, "mnt")
LOGP = os.path.join(BASE, "bridge.log")
REPO = "/root/ntfs-ext4-bridge"


def log(m):
    print(m, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def stat_manifest(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for name in sorted(fn):
            p = os.path.join(dp, name)
            rel = os.path.relpath(p, root)
            try:
                st = os.lstat(p)
            except OSError as e:
                out[rel] = {"error": str(e)}
                continue
            out[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                        "inode": st.st_ino, "nlink": st.st_nlink,
                        "mode": stat.filemode(st.st_mode)}
    return out


def diff_manifest(a, b):
    probs = []
    for rel in sorted(set(a) - set(b)):
        probs.append(("DISAPPEARED", rel, str(a[rel].get("size"))))
    for rel in sorted(set(b) - set(a)):
        probs.append(("APPEARED", rel, str(b[rel].get("size"))))
    for rel in sorted(set(a) & set(b)):
        x, y = a[rel], b[rel]
        for k in ("size", "mtime_ns", "inode", "nlink", "mode"):
            if x.get(k) != y.get(k):
                probs.append((k.upper(), rel, "%s -> %s" % (x.get(k), y.get(k))))
    return probs


def mounted(p):
    return os.path.ismount(p)


def cmd_setup():
    if mounted(MERGED):
        log("overlay already mounted at %s" % MERGED)
    else:
        for d in (UPPER, WORK, MERGED, SOURCE, OVERFLOW, MNT):
            os.makedirs(d, exist_ok=True)
        r = run(["mount", "-t", "overlay", "overlay", "-o",
                 "lowerdir=%s,upperdir=%s,workdir=%s" % (LOWER, UPPER, WORK),
                 MERGED])
        if r.returncode != 0:
            raise SystemExit("overlay mount failed: %s" % r.stderr.strip())
        log("overlay mounted: lower=%s (read-only to the kernel)" % LOWER)
    link = os.path.join(SOURCE, SHARE)
    if not os.path.lexists(link):
        os.symlink(MERGED, link)
    log("source farm: %s -> %s" % (link, MERGED))

    log("snapshotting the real library (stat only)...")
    t0 = time.time()
    m = stat_manifest(LOWER)
    json.dump(m, open(os.path.join(BASE, "before.json"), "w"))
    log("  %d files in %.0fs" % (len(m), time.time() - t0))


def cmd_run():
    cmd = [sys.executable, "-m", "ntfs_bridge.bridge",
           "--source", SOURCE, "--image", IMAGE, "--mount", MNT,
           "--port", str(PORT), "--partitioned", "--lazy",
           "--dealloc-timeout", "31536000",
           "--protected-roots", SHARE, "--overflow-dir", OVERFLOW]
    log("starting: %s" % " ".join(cmd[2:]))
    fh = open(LOGP, "ab")
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 3600
        while time.time() < deadline:
            if proc.poll() is not None:
                raise SystemExit("bridge exited early rc=%s; see %s"
                                 % (proc.returncode, LOGP))
            if mounted(MNT):
                log("  mounted: %s" % sorted(os.listdir(MNT))[:10])
                break
            time.sleep(5)
        else:
            raise SystemExit("did not mount within 3600s; see %s" % LOGP)

        log("  reading every served file...")
        n = nbytes = 0
        errors = []
        for dp, dn, fn in os.walk(MNT):
            if os.path.basename(dp) in ("System Volume Information", "$RECYCLE.BIN"):
                dn[:] = []
                continue
            for name in fn:
                p = os.path.join(dp, name)
                try:
                    # Sample rather than stream: it is the per-file mapping and
                    # materialisation paths that can write to ext4, and every
                    # file exercises them. Streaming 1.1 TiB adds hours and no
                    # additional write-path coverage.
                    sz = os.path.getsize(p)
                    with open(p, "rb") as f:
                        for off in (0, max(0, sz // 2 - (1 << 19)),
                                    max(0, sz - (1 << 20))):
                            f.seek(off)
                            b = f.read(1 << 20)
                            nbytes += len(b)
                    n += 1
                except OSError as e:
                    errors.append((os.path.relpath(p, MNT), str(e)))
        log("  read %d files, %.1f GiB, %d error(s)" % (n, nbytes / 2 ** 30, len(errors)))
        for rel, e in errors[:15]:
            log("     READ ERROR %s: %s" % (rel, e))
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
        run(["fusermount", "-uz", MNT])
        run(["umount", "-l", MNT])
        fh.close()
        log("  bridge stopped")


def cmd_verify():
    before = json.load(open(os.path.join(BASE, "before.json")))
    log("re-snapshotting the real library...")
    after = stat_manifest(LOWER)
    probs = diff_manifest(before, after)
    log("")
    if probs:
        log("REAL LIBRARY CHANGED - %d problem(s):" % len(probs))
        for k, rel, d in probs[:40]:
            log("   %-12s %-60s %s" % (k, rel[:60], d))
    else:
        log("REAL LIBRARY UNCHANGED: %d files, every stat field identical" % len(before))

    log("")
    log("what the bridge tried to write (overlay upperdir):")
    entries = []
    whiteouts = []
    for dp, dn, fn in os.walk(UPPER):
        for name in fn:
            p = os.path.join(dp, name)
            st = os.lstat(p)
            rel = os.path.relpath(p, UPPER)
            if stat.S_ISCHR(st.st_mode) and os.major(st.st_rdev) == 0 \
                    and os.minor(st.st_rdev) == 0:
                whiteouts.append(rel)          # overlay's "deleted" marker
            else:
                entries.append((rel, st.st_size))
    log("   files written : %d" % len(entries))
    for rel, sz in entries[:30]:
        log("      %10d  %s" % (sz, rel))
    log("   deletions attempted (whiteouts): %d" % len(whiteouts))
    for rel in whiteouts[:30]:
        log("      %s" % rel)
    ok = not probs and not entries and not whiteouts
    log("")
    log("VERDICT: %s" % ("no writes reached, none attempted" if ok
                         else "see above - the bridge attempted writes"))
    return 0 if ok else 1


def _upper_snapshot():
    """Everything currently in the upperdir, with whiteouts called out.

    My own mutations land here too, so the bridge's writes can only be seen as
    a DIFFERENCE against the state right after those mutations.
    """
    files, whiteouts = {}, set()
    for dp, dn, fn in os.walk(UPPER):
        for name in fn:
            p = os.path.join(dp, name)
            rel = os.path.relpath(p, UPPER)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if stat.S_ISCHR(st.st_mode) and os.major(st.st_rdev) == 0 \
                    and os.minor(st.st_rdev) == 0:
                whiteouts.add(rel)
            else:
                files[rel] = st.st_size
    return files, whiteouts


def cmd_stale():
    """Restart against a reused image after the library changed underneath.

    This is the shape that actually destroyed data: the image describes the
    tree as it was, reconciliation decides ext4 disagrees, and it "repairs"
    ext4 to match. Every mutation below lands in the overlay upper layer, so
    the real library is never altered even while the bridge is told it was.
    """
    share = os.path.join(MERGED)
    if not os.path.exists(IMAGE):
        raise SystemExit("no image at %s - run 'run' first so there is a "
                         "reused image to go stale" % IMAGE)
    before_lower = stat_manifest(LOWER)
    log("library baseline: %d files" % len(before_lower))

    log("")
    log("-- changing the served tree underneath the stopped bridge --")
    changed = []
    tops = sorted(d for d in os.listdir(share)
                  if os.path.isdir(os.path.join(share, d)))
    if len(tops) < 3:
        raise SystemExit("need at least 3 top-level shows to mutate")

    victim_dir = os.path.join(share, tops[1])
    shutil.rmtree(victim_dir)
    changed.append(("deleted dir", tops[1]))

    some_files = []
    for dp, dn, fn in os.walk(os.path.join(share, tops[0])):
        for name in sorted(fn):
            some_files.append(os.path.join(dp, name))
        if len(some_files) >= 4:
            break
    if len(some_files) >= 3:
        os.remove(some_files[0])
        changed.append(("deleted file", os.path.relpath(some_files[0], share)))
        os.rename(some_files[1], some_files[1] + ".renamed")
        changed.append(("renamed file", os.path.relpath(some_files[1], share)))
        with open(some_files[2], "r+b") as f:
            f.truncate(4096)
        changed.append(("truncated", os.path.relpath(some_files[2], share)))

    added = os.path.join(share, tops[2], "added_while_down.bin")
    with open(added, "wb") as f:
        f.write(b"added while the bridge was down" * 1000)
    changed.append(("added file", os.path.relpath(added, share)))

    for what, rel in changed:
        log("     %-14s %s" % (what, rel))

    base_files, base_whiteouts = _upper_snapshot()
    log("")
    log("upper after my changes: %d files, %d whiteouts" %
        (len(base_files), len(base_whiteouts)))

    log("")
    log("-- restarting against the REUSED image --")
    cmd_run()

    after_lower = stat_manifest(LOWER)
    probs = diff_manifest(before_lower, after_lower)
    now_files, now_whiteouts = _upper_snapshot()
    new_files = {k: v for k, v in now_files.items() if k not in base_files
                 or base_files[k] != v}
    new_whiteouts = now_whiteouts - base_whiteouts

    log("")
    if probs:
        log("REAL LIBRARY CHANGED - %d problem(s):" % len(probs))
        for k, rel, d in probs[:40]:
            log("   %-12s %-58s %s" % (k, rel[:58], d))
    else:
        log("REAL LIBRARY UNCHANGED: %d files, every stat field identical"
            % len(before_lower))

    log("")
    log("what the BRIDGE wrote, beyond my own changes:")
    log("   files written or altered : %d" % len(new_files))
    for rel, sz in sorted(new_files.items())[:30]:
        log("      %12d  %s" % (sz, rel))
    log("   deletions attempted      : %d" % len(new_whiteouts))
    for rel in sorted(new_whiteouts)[:30]:
        log("      %s" % rel)

    ok = not probs and not new_files and not new_whiteouts
    log("")
    log("VERDICT: %s" % ("the stale image drove no write into ext4" if ok
                         else "see above"))
    return 0 if ok else 1


def cmd_teardown():
    run(["umount", "-l", MERGED])
    log("overlay unmounted (upperdir kept at %s for inspection)" % UPPER)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    return {"setup": cmd_setup, "run": cmd_run,
            "verify": cmd_verify, "stale": cmd_stale,
            "teardown": cmd_teardown}[sys.argv[1]]() or 0


if __name__ == "__main__":
    sys.exit(main())
