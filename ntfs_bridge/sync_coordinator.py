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

        mapper.echo_observed_callback = self.on_echo_observed
        journal.dispatch_callback = self.on_dispatch

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
                paths = [p for p in (op.get('_rel'), op.get('_rel_old')) if p]
                for rel in paths:
                    if rel not in self._suppressed:
                        self._suppressed[rel] = 0.0
                        with self.mapper._sync_lock:
                            self.mapper.ext4_sync_in_progress.add(rel)
                if paths:
                    self._seq_paths.setdefault(op['seq'], []).extend(paths)

    def on_ack(self, results: List[dict]):
        """Agent finished executing: start the echo-timeout clock."""
        expiry = time.time() + self.echo_timeout_s
        with self._lock:
            for r in results:
                for rel in self._seq_paths.pop(r.get('seq'), []):
                    if rel in self._suppressed:
                        self._suppressed[rel] = expiry

    def on_echo_observed(self, rel_path: str):
        """MFT worker saw the echo: release suppression immediately."""
        self._release(rel_path)

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
        with self.mapper._sync_lock:
            self.mapper.ext4_sync_in_progress.discard(rel_path)

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
