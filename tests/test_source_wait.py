"""Wait for the data to arrive rather than requiring someone to have listed it.

A source dir of symlinks usually spans several disks. RequiresMountsFor= in the
unit only covers the mounts an operator remembered to name, and naming one of
two is the easy mistake - the bridge then starts while the other is still
unmounted and aborts on the dangling-symlink guard instead of waiting for it.
That was bd 16w, and it was a deployment error the code could not detect.

Asking the source directory what it points at needs no per-machine config and
behaves the same for one disk or five, encrypted or not, however the source is
laid out. The wait is bounded: a disk that is never coming back must not hang
the bridge forever, and the existing refuse-to-serve guard still applies once
the wait runs out.
"""
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.bridge import NTFSBridge  # noqa: E402


def make(source_dir, wait):
    b = NTFSBridge.__new__(NTFSBridge)
    b.source_dir = source_dir
    b.source_wait = wait
    return b


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    return cond


def main():
    ok = True
    tmp = tempfile.mkdtemp(prefix="srcwait-")
    src = os.path.join(tmp, "source")
    os.makedirs(src)
    data = os.path.join(tmp, "disk", "Share")

    print("\n[1] everything already resolves")
    real = os.path.join(tmp, "present")
    os.makedirs(real)
    os.symlink(real, os.path.join(src, "Present"))
    b = make(src, 30)
    t0 = time.time()
    b._wait_for_source_roots()
    ok &= check("returns immediately", time.time() - t0 < 1.0,
                "%.2fs" % (time.time() - t0))
    ok &= check("no dangling reported", b._dangling_source_roots() == [])

    print("\n[2] a disk that mounts late")
    os.symlink(data, os.path.join(src, "Late"))
    ok &= check("dangling seen before it appears",
                any("Late" in d for d in b._dangling_source_roots()))

    def appear():
        time.sleep(4)
        os.makedirs(data, exist_ok=True)

    threading.Thread(target=appear, daemon=True).start()
    b = make(src, 60)
    t0 = time.time()
    b._wait_for_source_roots()
    waited = time.time() - t0
    ok &= check("waited for it rather than aborting", 3 < waited < 30,
                "%.1fs" % waited)
    ok &= check("nothing dangling afterwards", b._dangling_source_roots() == [])

    print("\n[3] a disk that never arrives - bounded, not forever")
    os.symlink(os.path.join(tmp, "never"), os.path.join(src, "Never"))
    b = make(src, 5)
    t0 = time.time()
    b._wait_for_source_roots()
    waited = time.time() - t0
    ok &= check("gave up near the timeout", 4 < waited < 20, "%.1fs" % waited)
    ok &= check("still reports it dangling, so the caller can refuse",
                any("Never" in d for d in b._dangling_source_roots()))

    print("\n[4] waiting disabled")
    b = make(src, 0)
    t0 = time.time()
    b._wait_for_source_roots()
    ok &= check("returns at once with source_wait=0", time.time() - t0 < 1.0)

    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
