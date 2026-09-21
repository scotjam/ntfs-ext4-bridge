"""Two-way sync against the real library, with every ext4 write intercepted.

The library is the read-only lowerdir of an overlayfs (see
test_real_library_overlay.py); the kernel never writes to a lower layer, so
whatever the bridge, the guest or the agent do, the real files cannot change.
The upperdir is the ledger of everything that WOULD have been written.

On top of that overlay this runs the full two-way stack from
test_two_way_live.py - bridge in --two-way, an ntfs-3g guest on the NBD
export, the agent emulator - and performs guest-side and host-side changes
inside one scratch subtree, plus one rename-and-back of a small real file.

Then three checks:
  1. the real tree's stat manifest is unchanged (it cannot be, prove it)
  2. the upperdir holds nothing but the scratch subtree and that one file
  3. the served volume matches ext4: names and sizes everywhere, bytes in
     the scratch subtree; every change mirrored within a budget

Usage: test_real_library_two_way.py
"""
import os
import stat
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import test_real_library_overlay as ov  # noqa: E402
import test_two_way_live as tw           # noqa: E402

# point the two-way harness at the overlay farm
tw.SANDBOX = ov.BASE
tw.SOURCE = ov.SOURCE
tw.IMAGE = ov.IMAGE
tw.MOUNT = ov.MNT
tw.GUEST = os.path.join(ov.BASE, "guest")
tw.OVERFLOW = ov.OVERFLOW
tw.TOKEN = os.path.join(ov.BASE, "agent.token")
tw.LOG = ov.LOGP
tw.NBD_PORT = 10814
tw.CTRL_PORT = 10815

SCRATCH = ov.SHARE + "/_twoway_scratch"
CLUSTER = tw.CLUSTER
BUDGET = 90.0
FAILS = []


def log(m):
    print(m, flush=True)


def fail(m):
    FAILS.append(m)
    log("  FAIL  " + m)


def ok(m):
    log("  ok    " + m)


def ext4_scoped(root):
    """Like tw.ext4_snapshot but follows the share symlinks and hashes only
    the scratch subtree."""
    out = {}
    for dp, dn, fn in os.walk(root, followlinks=True):
        rel_d = os.path.relpath(dp, root)
        if rel_d != ".":
            out[rel_d] = ("dir", 0, "")
        for n in fn:
            p = os.path.join(dp, n)
            rel = os.path.relpath(p, root)
            try:
                if not rel.startswith(SCRATCH):
                    out[rel] = ("file", os.stat(p).st_size, "")
                    continue
                with open(p, "rb") as fh:
                    data = fh.read()
                out[rel] = ("file", len(data), tw.sha(data))
            except OSError as e:
                out[rel] = ("file", -1, "ERR:%s" % e)
    return out


def upper_entries():
    """Every entry the overlay wrote: (rel, kind) with kind file/dir/whiteout."""
    out = {}
    base = os.path.join(ov.BASE, "upper")
    for dp, dn, fn in os.walk(base):
        for d in dn:
            out[os.path.relpath(os.path.join(dp, d), base)] = "dir"
        for f in fn:
            p = os.path.join(dp, f)
            st = os.lstat(p)
            kind = "whiteout" if stat.S_ISCHR(st.st_mode) and st.st_rdev == 0 else "file"
            out[os.path.relpath(p, base)] = kind
    return out


class Gate:
    def __init__(self):
        self.ignore = set()

    def wait(self, guest, label, expect=None, timeout=BUDGET):
        t0 = time.time()
        last = []
        while time.time() - t0 < timeout:
            e = ext4_scoped(ov.SOURCE)
            if expect is not None:
                drift = tw.diff_snapshots(expect, e, "written", "ext4-now")
                if drift:
                    fail("%s: the bridge ALTERED ext4 after a host-side op; %d difference(s):"
                         % (label, len(drift)))
                    for d in drift[:12]:
                        log("        " + d)
                    return None
            try:
                v = guest.view()
                try:
                    n = v.snapshot(hash_prefix=SCRATCH)
                finally:
                    v.close()
            except (tw.ReadError, OSError, RuntimeError) as err:
                last = ["volume unreadable: %s" % err]
                time.sleep(2.0)
                continue
            last = [d for d in tw.diff_snapshots(e, n, "ext4", "ntfs")
                    if not any(k in d for k in self.ignore)]
            if not last:
                ok("%-58s mirrored in %5.1fs" % (label, time.time() - t0))
                return time.time() - t0
            time.sleep(1.0)
        fail("%s: not mirrored after %.0fs; %d difference(s):" % (label, timeout, len(last)))
        for d in last[:12]:
            log("        " + d)
        return None


def g(rel):
    return os.path.join(tw.GUEST, rel)


def s(rel):
    return os.path.join(ov.SOURCE, rel)


def smallest_real_file():
    best = None
    root = os.path.join(ov.SOURCE, ov.SHARE)
    for dp, dn, fn in os.walk(root, followlinks=True):
        if SCRATCH.split("/", 1)[1] in dp:
            continue
        for f in fn:
            p = os.path.join(dp, f)
            try:
                sz = os.stat(p).st_size
            except OSError:
                continue
            if 0 < sz < 512 * 1024 and "." in f and not f.startswith("."):
                if best is None or sz < best[1]:
                    best = (os.path.relpath(p, ov.SOURCE), sz)
    return best


def main():
    if not ov.mounted(ov.MERGED):
        ov.cmd_setup()
    before = ov.stat_manifest(ov.LOWER)
    log("real library baseline: %d files (stat only)" % len(before))
    upper0 = upper_entries()
    log("upperdir before: %d entries" % len(upper0))
    # a clean scratch, wherever a previous run left it
    scratch_src = s(SCRATCH)
    if os.path.isdir(scratch_src):
        import shutil
        shutil.rmtree(scratch_src)
    for p in (ov.IMAGE, ov.IMAGE + ".op-journal.jsonl", ov.IMAGE + ".agent-token"):
        try:
            os.remove(p)
        except OSError:
            pass
    os.makedirs(tw.OVERFLOW, exist_ok=True)

    bridge = tw.Bridge()
    guest = tw.Guest()
    agent = None
    gate = Gate()
    try:
        bridge.start(timeout=1800)
        guest.connect()
        agent = tw.Agent(bridge, guest)
        agent.start()

        # paths the volume does not carry at all (names NTFS refuses, excluded
        # patterns) are a pre-existing exposure gap, not a sync result: note
        # them once and ignore them below
        e = ext4_scoped(ov.SOURCE)
        v = guest.view()
        try:
            n = v.snapshot(hash_prefix=SCRATCH)
        finally:
            v.close()
        initial = tw.diff_snapshots(e, n, "ext4", "ntfs")
        for d in initial:
            gate.ignore.add(d.split(": ", 1)[1].split("  ")[0] if ": " in d else d)
        log("startup: %d ext4 paths, %d served; %d not exposed (ignored from here on)"
            % (len(e), len(n), len(initial)))
        gate.wait(guest, "startup> served volume equals ext4 (names/sizes)")

        log("\n=== guest-side changes inside the scratch subtree ===")
        tw.write(g(SCRATCH + "/g/a.bin"), tw.pattern(100, 1))
        tw.write(g(SCRATCH + "/g/b.bin"), tw.pattern(CLUSTER + 5, 2))
        tw.write(g(SCRATCH + "/g/c.txt"), b"guest text")
        guest.flush()
        gate.wait(guest, "ntfs> mkdir + 3 files")
        tw.patch(g(SCRATCH + "/g/b.bin"), 10, b"GUEST-EDIT")
        guest.flush()
        gate.wait(guest, "ntfs> edit in place")
        os.rename(g(SCRATCH + "/g/a.bin"), g(SCRATCH + "/g/a2.bin"))
        os.makedirs(g(SCRATCH + "/g2"), exist_ok=True)
        os.rename(g(SCRATCH + "/g/c.txt"), g(SCRATCH + "/g2/c.txt"))
        guest.flush()
        gate.wait(guest, "ntfs> rename + move")
        os.remove(g(SCRATCH + "/g/b.bin"))
        guest.flush()
        gate.wait(guest, "ntfs> delete")

        log("\n=== host-side changes inside the scratch subtree ===")

        def host(label, fn):
            fn()
            gate.wait(guest, "ext4> " + label, expect=ext4_scoped(ov.SOURCE))
        host("mkdir + 2 files", lambda: [tw.write(s(SCRATCH + "/h/x.bin"), tw.pattern(4096, 3)),
                                        tw.write(s(SCRATCH + "/h/y.bin"), tw.pattern(2 * CLUSTER + 1, 4))])
        host("truncate", lambda: os.truncate(s(SCRATCH + "/h/y.bin"), 100))
        host("rename", lambda: os.rename(s(SCRATCH + "/h/x.bin"), s(SCRATCH + "/h/x2.bin")))
        host("delete", lambda: os.remove(s(SCRATCH + "/h/x2.bin")))
        host("rename dir", lambda: os.rename(s(SCRATCH + "/h"), s(SCRATCH + "/h_renamed")))

        log("\n=== one small real file: guest renames it and renames it back ===")
        real = smallest_real_file()
        if real is None:
            fail("no small real file found to rename")
        else:
            rel, sz = real
            log("  candidate: %d bytes (path withheld)" % sz)
            tmp = rel + ".twoway-renamed"
            os.rename(g(rel), g(tmp))
            guest.flush()
            gate.wait(guest, "ntfs> rename a real file")
            os.rename(g(tmp), g(rel))
            guest.flush()
            gate.wait(guest, "ntfs> rename it back")
            lower_path = os.path.join(ov.LOWER, os.path.relpath(rel, ov.SHARE))
            if not os.path.exists(s(rel)):
                fail("the renamed-back file is not at its original ext4 path")
            elif open(s(rel), "rb").read() != open(lower_path, "rb").read():
                fail("the renamed-back file differs from the real one")
            else:
                ok("renamed-back file byte-identical to the real one")
    finally:
        if agent:
            agent.running = False
        guest.disconnect()
        bridge.stop()

    log("\n=== what reached the real disk, and what the bridge would have written ===")
    after = ov.stat_manifest(ov.LOWER)
    d = ov.diff_manifest(before, after)
    if d:
        fail("REAL LIBRARY CHANGED: %d difference(s)" % len(d))
        for x in list(d)[:10]:
            log("        %s" % (x,))
    else:
        ok("REAL LIBRARY UNCHANGED: %d files, every stat field identical" % len(before))
    upper1 = upper_entries()
    new = {k: v for k, v in upper1.items() if k not in upper0}
    allowed_prefix = SCRATCH
    stray = []
    for k, v in sorted(new.items()):
        if k.startswith(allowed_prefix) or (k + "/") == allowed_prefix + "/" or allowed_prefix.startswith(k + "/"):
            continue
        if real and (k == real[0] or k == real[0] + ".twoway-renamed"
                     or real[0].startswith(k + "/")):
            continue
        stray.append((k, v))
    log("  upperdir entries added: %d (scratch subtree and the one renamed file expected)" % len(new))
    if stray:
        fail("%d upperdir entr(ies) outside the scratch subtree and the renamed file:" % len(stray))
        for k, v in stray[:10]:
            log("        %-9s %s" % (v, k))
    else:
        ok("nothing outside the scratch subtree and the renamed file was written")
    stray_ov = [os.path.join(dp, f) for dp, dn, fn in os.walk(ov.OVERFLOW) for f in fn]
    if stray_ov:
        fail("%d file(s) landed in the overflow dir" % len(stray_ov))
    else:
        ok("overflow dir empty")
    if agent and agent.errors:
        fail("agent reported %d error(s): %s" % (len(agent.errors), agent.errors[:3]))

    if FAILS:
        log("\nFAILURES (%d):" % len(FAILS))
        for f in FAILS:
            log("  - " + f)
        log("\nREAL-LIBRARY TWO-WAY VERDICT: FAIL")
        return 1
    log("\nREAL-LIBRARY TWO-WAY VERDICT: mirrored both ways; the real library untouched; "
        "only the scratch subtree and the renamed file in the ledger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
