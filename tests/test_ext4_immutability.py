"""Run the bridge over a sandbox tree and prove ext4 came out untouched.

Two scenarios, both of which produced real data loss on 2026-09-18:

  readonly       serve the volume, read every file through the NTFS mount,
                 stop. ext4 must be byte-identical afterwards. Reads alone
                 rewrote files, because materialisation copied image space
                 over good data.

  stale-restart  serve once, stop, then change ext4 underneath the bridge
                 (delete, move, add, truncate) and start again against the
                 REUSED image. This is the 15:57 scenario exactly: the image
                 describes the old tree, reconciliation "repairs" ext4 to
                 match, and files are unlinked or rewritten as zeros. ext4
                 must change in exactly the ways we asked for and no others.

Refuses to run against anything outside the sandbox root, so it can never be
pointed at a real library.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import fs_testkit  # noqa: E402

SANDBOX = "/root/bridge-test"
PORT = 10810


def log(msg):
    print(msg, flush=True)


def guard(path):
    real = os.path.realpath(path)
    if not real.startswith(os.path.realpath(SANDBOX) + os.sep):
        raise SystemExit("REFUSING: %s is outside the sandbox %s" % (real, SANDBOX))
    return real


class Bridge:
    def __init__(self, source, image, mount, protected_roots="", overflow=None):
        self.source = guard(source)
        self.image = guard(image)
        self.mount = guard(mount)
        self.protected_roots = protected_roots
        self.overflow = guard(overflow) if overflow else None
        self.proc = None
        self.root_entries = []
        self.log_path = os.path.join(SANDBOX, "bridge.log")

    def start(self, timeout=600):
        os.makedirs(self.mount, exist_ok=True)
        if self.protected_roots == "auto":
            # What an operator would configure right now: every top-level
            # directory the source actually has. A share renamed since the
            # last run is named by its new name, and its old name is simply
            # no longer configured - which is what lets the ghost be pruned.
            self.protected_roots = ",".join(sorted(
                d for d in os.listdir(self.source)
                if os.path.isdir(os.path.join(self.source, d))))
            log("  roots: %s" % self.protected_roots)
        cmd = [sys.executable, "-m", "ntfs_bridge.bridge",
               "--source", self.source, "--image", self.image,
               "--mount", self.mount, "--port", str(PORT),
               "--partitioned", "--lazy", "--dealloc-timeout", "31536000"]
        if self.protected_roots:
            cmd += ["--protected-roots", self.protected_roots]
        if self.overflow:
            cmd += ["--overflow-dir", self.overflow]
        log("  starting bridge: %s" % " ".join(cmd[2:]))
        self.fh = open(self.log_path, "ab")
        self.fh.write(b"\n==== start %s ====\n" % time.ctime().encode())
        self.proc = subprocess.Popen(cmd, cwd=REPO, stdout=self.fh,
                                     stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit("bridge exited early (rc=%s); see %s"
                                 % (self.proc.returncode, self.log_path))
            try:
                if os.path.ismount(self.mount):
                    try:
                        entries = os.listdir(self.mount)
                    except OSError as e:
                        entries = ["<listdir failed: %s>" % e]
                    self.root_entries = sorted(entries)
                    log("  mounted; %d top-level entries: %s"
                        % (len(entries), self.root_entries[:10]))
                    return True
            except OSError:
                pass
            time.sleep(2)
        raise SystemExit("bridge did not mount within %ds; see %s"
                         % (timeout, self.log_path))

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=60)
        subprocess.run(["fusermount", "-uz", self.mount],
                       capture_output=True)
        subprocess.run(["umount", "-l", self.mount], capture_output=True)
        try:
            self.fh.close()
        except Exception:
            pass
        time.sleep(2)
        log("  bridge stopped")


def read_everything(mount):
    """Walk the served volume and read every byte. Returns (files, bytes, errors)."""
    files = nbytes = 0
    errors = []
    for dp, dn, fn in os.walk(mount):
        base = os.path.basename(dp)
        if base in ("System Volume Information", "$RECYCLE.BIN"):
            dn[:] = []
            continue
        for name in fn:
            p = os.path.join(dp, name)
            try:
                with open(p, "rb") as f:
                    while True:
                        b = f.read(1 << 20)
                        if not b:
                            break
                        nbytes += len(b)
                files += 1
            except OSError as e:
                errors.append((os.path.relpath(p, mount), str(e)))
    return files, nbytes, errors


def scenario_readonly(args):
    src = os.path.join(SANDBOX, "source")
    before = fs_testkit.manifest(src)

    b = Bridge(src, os.path.join(SANDBOX, "image.raw"),
               os.path.join(SANDBOX, "mnt"),
               protected_roots=(args.protected_roots or "auto"))
    try:
        b.start()
        log("  reading every file through the NTFS mount...")
        n, nbytes, errors = read_everything(b.mount)
        log("  read %d files, %.1f MiB, %d read error(s)"
            % (n, nbytes / 2 ** 20, len(errors)))
        for rel, e in errors[:10]:
            log("     READ ERROR %s: %s" % (rel, e))
    finally:
        b.stop()

    after = fs_testkit.manifest(src)
    probs = fs_testkit.compare(before, after)
    report("readonly", probs, before)
    return 0 if not probs else 1


def scenario_stale_restart(args):
    src = os.path.join(SANDBOX, "source")
    image = os.path.join(SANDBOX, "image.raw")
    mount = os.path.join(SANDBOX, "mnt")

    log("\n-- pass 1: build the image from the tree --")
    b = Bridge(src, image, mount, protected_roots=(args.protected_roots or "auto"))
    try:
        b.start()
        read_everything(b.mount)
    finally:
        b.stop()

    log("\n-- changing ext4 underneath the stopped bridge --")
    changed = []
    victim = os.path.join(src, "sizes", "size_65536.bin")
    os.remove(victim)
    changed.append(("deleted", "sizes/size_65536.bin"))

    os.rename(os.path.join(src, "content"), os.path.join(src, "content_moved"))
    changed.append(("moved", "content -> content_moved"))

    newf = os.path.join(src, "sizes", "added_after.bin")
    with open(newf, "wb") as f:
        f.write(b"added while the bridge was down" * 100)
    changed.append(("added", "sizes/added_after.bin"))

    trunc = os.path.join(src, "sizes", "size_131072.bin")
    with open(trunc, "r+b") as f:
        f.truncate(1234)
    changed.append(("truncated", "sizes/size_131072.bin"))
    for what, rel in changed:
        log("     %-10s %s" % (what, rel))

    before = fs_testkit.manifest(src)

    log("\n-- pass 2: restart against the REUSED image --")
    b = Bridge(src, image, mount, protected_roots=(args.protected_roots or "auto"))
    try:
        b.start()
        log("  reading every file through the NTFS mount...")
        n, nbytes, errors = read_everything(b.mount)
        log("  read %d files, %.1f MiB, %d read error(s)"
            % (n, nbytes / 2 ** 20, len(errors)))
        for rel, e in errors[:10]:
            log("     READ ERROR %s: %s" % (rel, e))
        time.sleep(10)   # let any background reconciliation act
    finally:
        b.stop()

    after = fs_testkit.manifest(src)
    probs = fs_testkit.compare(before, after)
    report("stale-restart", probs, before)

    # ext4 being untouched is necessary but not sufficient: the served view
    # must also stop advertising a share that ext4 no longer has, or every
    # read under it can only fail.
    ext4_roots = sorted(d for d in os.listdir(src)
                        if os.path.isdir(os.path.join(src, d)))
    served = [e for e in b.root_entries
              if e.lower() not in ("system volume information", "$recycle.bin",
                                   ".bzvol", "desktop.ini")]
    ghosts = [e for e in served if e not in ext4_roots]
    missing = [e for e in ext4_roots if e not in served]
    log("")
    log("  ext4 roots  : %s" % ext4_roots)
    log("  served roots: %s" % served)
    if ghosts:
        log("  GHOST ROOTS still served but absent from ext4: %s" % ghosts)
    if missing:
        log("  ext4 roots NOT served: %s" % missing)
    if not ghosts and not missing:
        log("  served view matches ext4")
    return 0 if (not probs and not ghosts and not missing) else 1


def report(name, probs, before):
    log("")
    if not probs:
        log("RESULT %s: ext4 UNCHANGED - %d files verified byte-for-byte"
            % (name, len(before["files"])))
        return
    log("RESULT %s: ext4 CHANGED - %d problem(s)" % (name, len(probs)))
    kinds = {}
    for kind, rel, detail in probs:
        kinds[kind] = kinds.get(kind, 0) + 1
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        log("   %-14s %d" % (k, v))
    log("")
    for kind, rel, detail in probs[:40]:
        log("   %-12s %-55s %s" % (kind, rel[:55], detail))
    if len(probs) > 40:
        log("   ... and %d more" % (len(probs) - 40))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["readonly", "stale-restart"])
    ap.add_argument("--protected-roots", default="")
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(SANDBOX, "source")):
        raise SystemExit("build the tree first: "
                         "python3 tests/fs_testkit.py build %s/source" % SANDBOX)
    log("=== scenario: %s   protected-roots=%r ==="
        % (args.scenario, args.protected_roots))
    if args.scenario == "readonly":
        return scenario_readonly(args)
    return scenario_stale_restart(args)


if __name__ == "__main__":
    sys.exit(main())
