"""Echo suppression for two-way sync.

When the guest agent executes an op we sent (e.g. creates a file because it
appeared on ext4), Windows writes MFT records for it and those writes arrive
back at the bridge over NBD. Without suppression the MFT worker would treat
them as genuine Windows-side changes and re-materialize them to ext4 —
worst case deleting or overwriting the very file that triggered the op.

The coordinator marks every path touched by a dispatched op in the mapper's
`ext4_sync_in_progress` set. The MFT worker's skip-branches consult that set
and only *track* the records (mapping clusters to the existing ext4 source)
instead of materializing. Suppression is released when the echo is observed
(mapper.echo_observed_callback) or after a timeout following the agent's ack
— Windows flushes MFT lazily, so the ack always precedes the echo.
"""

import threading
import time
from typing import Dict, List

DEFAULT_ECHO_TIMEOUT_S = 30.0


def log(msg):
    print(f"[SyncCoordinator] {msg}", flush=True)


class SyncCoordinator:
    """Tracks dispatched guest ops and manages echo suppression windows."""

    def __init__(self, mapper, journal,
                 echo_timeout_s: float = DEFAULT_ECHO_TIMEOUT_S):
        self.mapper = mapper
        self.journal = journal
        self.echo_timeout_s = echo_timeout_s

        self._lock = threading.Lock()
        # rel_path -> expiry timestamp (0.0 = no expiry yet: op dispatched
        # but not acked; the sweep only clears entries past a real expiry)
        self._suppressed: Dict[str, float] = {}
        # seq -> [rel paths] so acks can start the expiry clock
        self._seq_paths: Dict[int, List[str]] = {}
        # paths whose echo arrived before the agent acked the batch
        self._echoed_unacked: set = set()

        if not hasattr(mapper, 'ext4_sync_kinds'):
            mapper.ext4_sync_kinds = {}
        mapper.echo_observed_callback = self.on_echo_observed
        journal.dispatch_callback = self.on_dispatch
        journal.release_callback = self._release

        self._running = True
        self._sweeper = threading.Thread(target=self._sweep_loop,
                                         daemon=True,
                                         name="SyncCoordinator-Sweep")
        self._sweeper.start()

    def stop(self):
        self._running = False
        if self.mapper.echo_observed_callback is self.on_echo_observed:
            self.mapper.echo_observed_callback = None

    # ------------------------------------------------------------------

    def on_dispatch(self, ops: List[dict]):
        """Journal handed ops to the agent: open suppression windows."""
        with self._lock:
            for op in ops:
                if op.get('op') in ('set_mtime', 'flush_volume', 'gate_begin'):
                    # No namespace or size change: the echo (a timestamp in
                    # the record) is harmless to process as a guest change,
                    # and a window here would drop concurrent guest writes.
                    continue
                paths = [p for p in (op.get('_rel'), op.get('_rel_old')) if p]
                for rel in paths:
                    # Always (re)open: a path can be dispatched again while
                    # an earlier op's window is counting down to expiry (a
                    # mkdir echoed and acked, then an mv of the same dir a
                    # second later). Leaving the old expiry in place let the
                    # sweep close the window under the new op, and the next
                    # record re-read renamed the ext4 tree back.
                    self._suppressed[rel] = 0.0
                    self._echoed_unacked.discard(rel)
                    with self.mapper._sync_lock:
                        self.mapper.ext4_sync_in_progress.add(rel)
                    # which kinds of op hold this path: a record freed
                    # while only a create/resize is in flight is the
                    # guest's own delete, not an echo (only rm/mv free
                    # records)
                    kinds = self.mapper.ext4_sync_kinds.setdefault(rel, set())
                    kinds.add(op.get('op'))
                    log(f"dispatch {op.get('op')} seq={op.get('seq')} {rel}")
                if paths:
                    self._seq_paths.setdefault(op['seq'], []).extend(paths)

    def on_ack(self, results: List[dict]):
        """Agent finished executing: start the echo-timeout clock."""
        now = time.time()
        with self._lock:
            for r in results:
                for rel in self._seq_paths.pop(r.get('seq'), []):
                    if rel not in self._suppressed:
                        continue
                    if rel in self._echoed_unacked:
                        # The echo came in mid-batch. The window stays open
                        # until the batch is acked - the driver's trailing
                        # writes for the op (the flush at the end of the
                        # batch) can follow the record by seconds - and
                        # closes after the grace.
                        self._echoed_unacked.discard(rel)
                        self._suppressed[rel] = now + self.ECHO_GRACE_S
                    elif self._suppressed[rel] == 0.0:
                        self._suppressed[rel] = now + self.echo_timeout_s

    ECHO_GRACE_S = 2.0

    def on_echo_observed(self, rel_path: str):
        """MFT worker saw the echo: release suppression shortly.

        Not immediately: the record reaching the bridge does not mean the
        driver is done with the file - data pages it wrote as part of the
        same op (a zero-fill of the extension, say) can trail the record by
        a moment, and those must still be dropped rather than land in ext4.
        Before the agent has acked the batch the trailing writes may still
        be seconds away (a 50-op batch flushes at its end), so the window
        then stays open until the ack.
        """
        with self._lock:
            if rel_path not in self._suppressed:
                return
            unacked = any(rel_path in paths for paths in self._seq_paths.values())
            if unacked:
                self._echoed_unacked.add(rel_path)
            else:
                self._suppressed[rel_path] = time.time() + self.ECHO_GRACE_S

    def release_all(self):
        """Drop every suppression window (used around consistency gates)."""
        with self._lock:
            paths = list(self._suppressed)
        for rel in paths:
            self._release(rel)

    # ------------------------------------------------------------------

    def _release(self, rel_path: str):
        with self._lock:
            self._suppressed.pop(rel_path, None)
            self._echoed_unacked.discard(rel_path)
        with self.mapper._sync_lock:
            self.mapper.ext4_sync_in_progress.discard(rel_path)
            self.mapper.ext4_sync_kinds.pop(rel_path, None)

    def _sweep_loop(self):
        while self._running:
            time.sleep(1.0)
            now = time.time()
            try:
                with self._lock:
                    expired = [rel for rel, exp in self._suppressed.items()
                               if 0.0 < exp <= now]
                for rel in expired:
                    self._release(rel)
            except Exception as e:
                log(f"sweep error: {e}")

    def stats(self) -> dict:
        with self._lock:
            return {'suppressed': len(self._suppressed),
                    'awaiting_ack': sum(1 for v in self._suppressed.values()
                                        if v == 0.0)}
