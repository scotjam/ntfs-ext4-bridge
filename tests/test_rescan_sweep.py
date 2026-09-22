"""The op journal's reconciliation sweep catches what inotify did not report.

Drives OpJournal._rescan_once directly over a temp tree and a fake mapper,
with the watchers never started, so every change is by definition one
inotify "missed":

  * a file or directory the bridge does not track -> create
  * a tracked file whose size differs from the record -> upsert
  * a tracked file whose mtime moved since the last sweep (same size) -> upsert
  * a tracked path gone from ext4 -> delete, but only on the second sweep in
    a row, and never in safe/record-only mode
  * paths with an op pending or in flight, or written by the bridge within
    the last seconds, are left alone
  * the first sweep only records mtimes
"""
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.op_journal import OpJournal  # noqa: E402
from ntfs_bridge.file_watcher import EVENT_CREATE, EVENT_DELETE  # noqa: E402


class FakeMapper:
    def __init__(self):
        self.lock = threading.RLock()
        self.path_to_mft_record = {}
        self.sizes = {}
        self.known_root_entries = {"Share"}
        self.ntfs_sync_in_progress = set()
        self.ext4_sync_in_progress = set()
        self.ntfs_sync_timestamps = {}
        self._safe_mode = False

    def record_data_size(self, rel):
        return self.sizes.get(rel)


def make(tmp, safe=False):
    m = FakeMapper()
    m._safe_mode = safe
    j = OpJournal(os.path.join(tmp, "journal.jsonl"), tmp, m, rescan_interval=0)
    events = []
    j.on_event = lambda kind, rel, new=None: events.append((kind, rel))
    return m, j, events


def write(path, data, age=10.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    t = time.time() - age
    os.utime(path, (t, t))


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    return cond


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        share = os.path.join(tmp, "Share")
        write(os.path.join(share, "known.bin"), b"x" * 100)
        write(os.path.join(share, "same.bin"), b"y" * 50)
        m, j, ev = make(tmp)
        m.path_to_mft_record = {"Share/known.bin": 20, "Share/same.bin": 21}
        m.sizes = {"Share/known.bin": 100, "Share/same.bin": 50}

        print("\n[first sweep records only]")
        write(os.path.join(share, "new_before_first.bin"), b"z")
        j._rescan_once(record_only=True)
        ok &= check("no events on the first sweep", ev == [], str(ev))

        print("\n[creates, size changes, same-size edits]")
        write(os.path.join(share, "sub", "fresh.bin"), b"fresh")
        write(os.path.join(share, "known.bin"), b"x" * 200)           # size differs
        write(os.path.join(share, "same.bin"), b"Y" * 50, age=5.0)     # same size, new mtime
        ev.clear()
        j._rescan_once()
        got = set(ev)
        ok &= check("untracked file -> create",
                    (EVENT_CREATE, "Share/sub/fresh.bin") in got, str(sorted(got)))
        ok &= check("untracked directory -> create",
                    (EVENT_CREATE, "Share/sub") in got)
        ok &= check("file untracked since before the first sweep -> create",
                    (EVENT_CREATE, "Share/new_before_first.bin") in got)
        ok &= check("size differs from the record -> upsert",
                    (EVENT_CREATE, "Share/known.bin") in got)
        ok &= check("same size, mtime moved -> upsert",
                    (EVENT_CREATE, "Share/same.bin") in got)

        print("\n[a file still being written is left for the next sweep]")
        write(os.path.join(share, "writing.bin"), b"w", age=0.0)
        ev.clear()
        j._rescan_once()
        ok &= check("mtime < 2 s old -> not yet", (EVENT_CREATE, "Share/writing.bin") not in ev, str(ev))

        print("\n[busy paths are left alone]")
        write(os.path.join(share, "busy.bin"), b"b")
        m.ext4_sync_in_progress.add("Share/busy.bin")
        write(os.path.join(share, "echo.bin"), b"e")
        m.ntfs_sync_timestamps["Share/echo.bin"] = time.time()
        ev.clear()
        j._rescan_once()
        ok &= check("in-flight op -> skipped", (EVENT_CREATE, "Share/busy.bin") not in ev)
        ok &= check("bridge wrote it just now -> skipped", (EVENT_CREATE, "Share/echo.bin") not in ev)

        print("\n[deletes need two misses in a row]")
        m.path_to_mft_record["Share/gone.bin"] = 30
        ev.clear()
        j._rescan_once()
        ok &= check("first miss: no delete yet", (EVENT_DELETE, "Share/gone.bin") not in ev, str(ev))
        j._rescan_once()
        ok &= check("second miss: delete", (EVENT_DELETE, "Share/gone.bin") in ev, str(ev))
        m.path_to_mft_record["Share/back.bin"] = 31
        j._rescan_once()
        write(os.path.join(share, "back.bin"), b"back")
        m.sizes["Share/back.bin"] = 4
        ev.clear()
        j._rescan_once()
        ok &= check("reappeared before the second miss: no delete",
                    (EVENT_DELETE, "Share/back.bin") not in ev, str(ev))
        m.path_to_mft_record["Share/outside/x"] = 32
        m.path_to_mft_record["Other/y"] = 33
        ev.clear()
        j._rescan_once(); j._rescan_once()
        ok &= check("tracked paths outside the shares are never deleted",
                    (EVENT_DELETE, "Other/y") not in ev)

        print("\n[a path that never becomes tracked is asked for once per version]")
        write(os.path.join(share, "CON.txt"), b"reserved name")
        ev.clear()
        j._rescan_once(); j._rescan_once(); j._rescan_once()
        n = sum(1 for e in ev if e == (EVENT_CREATE, "Share/CON.txt"))
        ok &= check("three sweeps, one request", n == 1, "requests=%d" % n)
        write(os.path.join(share, "CON.txt"), b"reserved name, edited")
        ev.clear()
        j._rescan_once()
        ok &= check("a new version is asked for again",
                    (EVENT_CREATE, "Share/CON.txt") in ev, str(ev))

    print("\n[safe / record-only mode never sweeps deletes]")
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "Share"))
        m, j, ev = make(tmp, safe=True)
        m.path_to_mft_record = {"Share/guest_only.bin": 40}
        j._rescan_once(); j._rescan_once(); j._rescan_once()
        ok &= check("no delete in safe mode", all(k != EVENT_DELETE for k, _ in ev), str(ev))

    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
