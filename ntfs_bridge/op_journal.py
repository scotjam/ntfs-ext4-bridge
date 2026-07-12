"""ext4 change journal for two-way sync.

Watches the exposed share roots for ext4 changes, coalesces the raw
file-watcher events into a minimal sequence of idempotent guest operations,
persists them to an append-only JSONL journal, and serves them to the guest
agent (via ControlServer) with at-least-once delivery semantics.

Guest op format (JSON dict):
    {"seq": 1041, "op": "create_sized", "path": "Share\\dir\\file.bin",
     "size": 1048576, "mtime_ms": 1783900000123}
Op types: mkdir, rm (recurse flag), mv (dst), create_sized (size, mtime_ms),
resize (size, mtime_ms), set_mtime (mtime_ms), flush_volume, gate_begin,
gate_end. `path`/`dst` are NTFS-relative with backslashes. The private
`_rel` key (POSIX rel path) rides along for the SyncCoordinator; the agent
ignores underscore-prefixed keys.

IMPORTANT: share roots are commonly symlinks and inotify's recursive tree
watch does not traverse symlinks, so this module creates one watcher per
exposed root, watching the resolved target, and maps events back to
"<ShareName>/<rel>" paths.
"""

import json
import os
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Set, Tuple

from .file_watcher import (create_watcher, EVENT_CREATE, EVENT_DELETE,
                           EVENT_MODIFY, EVENT_MOVE)


def log(msg):
    print(f"[OpJournal] {msg}", flush=True)


def _to_ntfs(rel: str) -> str:
    return rel.replace('/', '\\')


class OpJournal:
    """Coalesces ext4 events into sequenced, idempotent guest ops."""

    FLUSH_TICK_S = 0.2

    def __init__(self, journal_path: str, source_dir: str, mapper,
                 exclude_cb: Optional[Callable[[str], bool]] = None,
                 quiesce_s: float = 0.5, max_hold_s: float = 5.0,
                 gate_threshold_ops: int = 500,
                 gate_threshold_age: float = 600.0):
        self.journal_path = journal_path
        self.source_dir = os.path.abspath(source_dir)
        self.mapper = mapper
        self.exclude_cb = exclude_cb or (lambda rel: False)
        self.quiesce_s = quiesce_s
        self.max_hold_s = max_hold_s
        self.gate_threshold_ops = gate_threshold_ops
        self.gate_threshold_age = gate_threshold_age

        self.epoch = uuid.uuid4().hex
        self._next_seq = 1
        self._acked_seq = 0          # highest contiguously-acked seq

        # RLock: _ops_for_upsert's directory catch-up walk re-enters
        # on_event() while _flush_ready already holds the lock.
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)

        # Pending (not yet flushed to ops) per-path coalescing state:
        # rel(posix) -> {kind: 'upsert'|'delete', first_ts, last_ts,
        #                prior_delete: bool, recurse: bool}
        self._pending: Dict[str, dict] = {}
        # Moves preserve arrival order; (old_rel, new_rel, ts)
        self._moves: List[Tuple[str, str, float]] = []

        # Flushed ops awaiting delivery/ack (list of op dicts, seq ascending)
        self._ops: List[dict] = []
        self._first_unacked_ts: Optional[float] = None

        # Paths whose guest op failed; repaired by the next consistency gate.
        self.dirty_paths: Set[str] = set()
        self._dirty_path_file = journal_path + '.dirty'
        self._load_dirty()

        # Called with the list of op dicts each time ops are handed to the
        # agent (set by SyncCoordinator for echo suppression).
        self.dispatch_callback: Optional[Callable[[List[dict]], None]] = None
        # Called when escalation thresholds trip (set by ConsistencyGate).
        self.escalation_callback: Optional[Callable[[str], None]] = None

        self._paused = False
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None
        self._watchers = []

        # Fresh epoch -> fresh journal file
        self._journal_file = open(self.journal_path, 'w', encoding='utf-8')
        self._journal_write({'header': True, 'epoch': self.epoch,
                             'base_seq': self._next_seq})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start watchers (one per exposed share root) and the flush loop."""
        self._running = True
        roots = sorted(self.mapper.known_root_entries)
        for entry in roots:
            share_path = os.path.join(self.source_dir, entry)
            target = os.path.realpath(share_path)
            if not os.path.isdir(target):
                log(f"WARNING: share root {entry} -> {target} not a directory; not watching")
                continue
            watcher = create_watcher(
                target,
                self._make_share_callback(entry),
                move_events=True,
            )
            watcher.start()
            self._watchers.append(watcher)
            log(f"Watching share '{entry}' at {target}")

        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="OpJournal-Flush")
        self._flush_thread.start()

    def stop(self):
        self._running = False
        for w in self._watchers:
            try:
                w.stop()
            except Exception:
                pass
        self._watchers = []
        if self._flush_thread:
            self._flush_thread.join(timeout=2.0)
            self._flush_thread = None
        with self._lock:
            self._journal_file.close()

    def _make_share_callback(self, share: str):
        def cb(event_type: str, payload):
            try:
                if event_type == EVENT_MOVE:
                    old_rel, new_rel = payload
                    self.on_event(EVENT_MOVE,
                                  os.path.join(share, old_rel),
                                  os.path.join(share, new_rel))
                else:
                    self.on_event(event_type, os.path.join(share, payload))
            except Exception as e:
                log(f"event error ({event_type}, {payload}): {e}")
        return cb

    # ------------------------------------------------------------------
    # Ingestion + coalescing
    # ------------------------------------------------------------------

    def on_event(self, event_type: str, rel_path: str,
                 new_rel_path: Optional[str] = None):
        """Ingest a watcher event (rel paths are POSIX, share-prefixed)."""
        rel_path = rel_path.replace('\\', '/').strip('/')
        if self.exclude_cb(rel_path):
            return

        # Echo filter: skip events caused by the bridge's own Windows->ext4
        # materialization (same rule the old SyncDaemon used).
        m = self.mapper
        if rel_path in m.ntfs_sync_in_progress:
            return
        if time.time() - m.ntfs_sync_timestamps.get(rel_path, 0) < 2.0:
            return

        now = time.time()
        with self._cond:
            if event_type == EVENT_MOVE and new_rel_path:
                new_rel_path = new_rel_path.replace('\\', '/').strip('/')
                if self.exclude_cb(new_rel_path):
                    # Renamed to an excluded name: treat as delete of old
                    self._ingest_delete(rel_path, now)
                else:
                    self._pending.pop(new_rel_path, None)
                    self._moves.append((rel_path, new_rel_path, now))
            elif event_type == EVENT_DELETE:
                self._ingest_delete(rel_path, now)
            else:  # create / modify
                st = self._pending.get(rel_path)
                if st and st['kind'] == 'delete':
                    st['kind'] = 'upsert'
                    st['prior_delete'] = True
                    st['last_ts'] = now
                elif st:
                    st['last_ts'] = now
                else:
                    self._pending[rel_path] = {
                        'kind': 'upsert', 'first_ts': now, 'last_ts': now,
                        'prior_delete': False, 'recurse': False,
                    }

    def _ingest_delete(self, rel_path: str, now: float):
        st = self._pending.get(rel_path)
        known_to_ntfs = rel_path in self.mapper.path_to_mft_record
        if st and st['kind'] == 'upsert' and not st['prior_delete'] \
                and not known_to_ntfs:
            # Brand-new path that NTFS never saw: annihilate.
            del self._pending[rel_path]
            return
        self._pending[rel_path] = {
            'kind': 'delete', 'first_ts': now, 'last_ts': now,
            'prior_delete': False, 'recurse': True,
        }

    # ------------------------------------------------------------------
    # Flush: pending -> sequenced ops
    # ------------------------------------------------------------------

    def _flush_loop(self):
        while self._running:
            time.sleep(self.FLUSH_TICK_S)
            try:
                self._flush_ready()
                self._check_escalation()
            except Exception as e:
                log(f"flush error: {e}")

    def _flush_ready(self):
        now = time.time()
        with self._cond:
            if self._paused:
                return

            ready_moves = list(self._moves)
            self._moves = []

            ready: List[Tuple[str, dict]] = []
            for rel, st in list(self._pending.items()):
                if (now - st['last_ts'] >= self.quiesce_s
                        or now - st['first_ts'] >= self.max_hold_s):
                    ready.append((rel, st))
                    del self._pending[rel]
            if not ready and not ready_moves:
                return

            # Directory-delete subsumption: drop anything under a deleted dir
            delete_roots = [rel for rel, st in ready if st['kind'] == 'delete']
            def subsumed(rel):
                return any(rel != d and rel.startswith(d + '/')
                           for d in delete_roots)
            ready = [(rel, st) for rel, st in ready if not subsumed(rel)]
            ready_moves = [mv for mv in ready_moves
                           if not subsumed(mv[0]) and not subsumed(mv[1])]

            new_ops: List[dict] = []

            # 1) moves, in arrival order
            for old_rel, new_rel, _ts in ready_moves:
                new_ops.append({'op': 'mv', 'path': _to_ntfs(old_rel),
                                'dst': _to_ntfs(new_rel),
                                '_rel': new_rel, '_rel_old': old_rel})

            # 2) upserts, parents before children
            upserts = sorted((r for r in ready if r[1]['kind'] == 'upsert'),
                             key=lambda r: r[0].count('/'))
            for rel, st in upserts:
                new_ops.extend(self._ops_for_upsert(rel, st))

            # 3) deletes, deepest first
            deletes = sorted((r for r in ready if r[1]['kind'] == 'delete'),
                             key=lambda r: -r[0].count('/'))
            for rel, st in deletes:
                new_ops.append({'op': 'rm', 'path': _to_ntfs(rel),
                                'recurse': True, '_rel': rel})

            if not new_ops:
                return

            # One volume flush per batch so MFT echoes reach NBD promptly.
            new_ops.append({'op': 'flush_volume', 'path': '', '_rel': ''})

            for op in new_ops:
                op['seq'] = self._next_seq
                self._next_seq += 1
                self._journal_write(op)
                self._ops.append(op)
            self._journal_file.flush()
            os.fsync(self._journal_file.fileno())
            if self._first_unacked_ts is None:
                self._first_unacked_ts = now
            self._cond.notify_all()

    def _ops_for_upsert(self, rel: str, st: dict) -> List[dict]:
        ops: List[dict] = []
        source = os.path.join(self.source_dir, rel)
        try:
            stat = os.stat(source)
        except OSError:
            return ops  # vanished; a delete event will follow
        mtime_ms = int(stat.st_mtime * 1000)
        ntfs = _to_ntfs(rel)

        if st['prior_delete']:
            ops.append({'op': 'rm', 'path': ntfs, 'recurse': True,
                        '_rel': rel})

        if os.path.isdir(source):
            ops.append({'op': 'mkdir', 'path': ntfs, '_rel': rel})
            # Catch-up walk: children created before the watch attached to
            # a brand-new directory are otherwise lost.
            try:
                for name in os.listdir(source):
                    child = rel + '/' + name
                    if not self.exclude_cb(child):
                        self.on_event(EVENT_CREATE, child)
            except OSError:
                pass
            return ops

        known = rel in self.mapper.path_to_mft_record
        if known and not st['prior_delete']:
            ops.append({'op': 'resize', 'path': ntfs, 'size': stat.st_size,
                        'mtime_ms': mtime_ms, '_rel': rel})
        else:
            ops.append({'op': 'create_sized', 'path': ntfs,
                        'size': stat.st_size, 'mtime_ms': mtime_ms,
                        '_rel': rel})
        return ops

    # ------------------------------------------------------------------
    # Serving (control server API)
    # ------------------------------------------------------------------

    def ops_after(self, cursor: int, limit: int = 64) -> List[dict]:
        with self._cond:
            out = [op for op in self._ops if op['seq'] > cursor][:limit]
        if out and self.dispatch_callback:
            try:
                self.dispatch_callback(out)
            except Exception as e:
                log(f"dispatch_callback error: {e}")
        return out

    def wait_for_ops(self, cursor: int, timeout: float) -> bool:
        deadline = time.time() + timeout
        with self._cond:
            while True:
                if any(op['seq'] > cursor for op in self._ops):
                    return True
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)

    def ack(self, results: List[dict]):
        """Process per-op results from the agent.

        Both ok and error results consume the op (the agent executed it;
        re-delivery would not change the outcome). Errors put the path in
        the dirty set for the next consistency gate to repair.
        """
        now = time.time()
        with self._cond:
            processed = {r['seq'] for r in results if 'seq' in r}
            dirty_changed = False
            for r in results:
                if r.get('status') != 'ok':
                    op = next((o for o in self._ops
                               if o['seq'] == r.get('seq')), None)
                    rel = op.get('_rel') if op else None
                    log(f"guest op {r.get('seq')} failed: "
                        f"{r.get('code')} {r.get('message')} ({rel})")
                    if rel:
                        self.dirty_paths.add(rel)
                        dirty_changed = True
            if dirty_changed:
                self._save_dirty()
            # Ops are executed in seq order; advance the cursor across the
            # contiguous processed prefix and drop those ops.
            while self._ops and self._ops[0]['seq'] in processed:
                self._acked_seq = self._ops[0]['seq']
                self._ops.pop(0)
            self._first_unacked_ts = now if self._ops else None

    # ------------------------------------------------------------------
    # Gate interactions / escalation
    # ------------------------------------------------------------------

    def inject_op(self, op: dict):
        """Append a control op (e.g. gate_begin) directly to the stream.

        Delivered even while dispatch is paused — gate control must flow
        during the barrier.
        """
        with self._cond:
            op = dict(op)
            op['seq'] = self._next_seq
            self._next_seq += 1
            self._journal_write(op)
            self._journal_file.flush()
            self._ops.append(op)
            if self._first_unacked_ts is None:
                self._first_unacked_ts = time.time()
            self._cond.notify_all()

    def pause(self):
        with self._cond:
            self._paused = True

    def resume(self):
        with self._cond:
            self._paused = False

    def drain_barrier(self, timeout: float) -> bool:
        """Wait until every flushed op is acked. Unacked leftovers go dirty."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._cond:
                if not self._ops:
                    return True
            time.sleep(0.25)
        with self._cond:
            for op in self._ops:
                rel = op.get('_rel')
                if rel:
                    self.dirty_paths.add(rel)
            self._save_dirty()
            self._ops = []
        return False

    def reset_epoch(self) -> str:
        """New epoch: truncate journal, drop pending state, keep dirty set."""
        with self._cond:
            self.epoch = uuid.uuid4().hex
            self._pending.clear()
            self._moves.clear()
            self._ops.clear()
            self._first_unacked_ts = None
            self._journal_file.close()
            self._journal_file = open(self.journal_path, 'w',
                                      encoding='utf-8')
            self._journal_write({'header': True, 'epoch': self.epoch,
                                 'base_seq': self._next_seq})
            self._journal_file.flush()
            self._cond.notify_all()
        log(f"epoch reset -> {self.epoch}")
        return self.epoch

    def clear_dirty(self):
        with self._cond:
            self.dirty_paths.clear()
            self._save_dirty()

    def _check_escalation(self):
        reason = None
        with self._cond:
            if len(self._ops) + len(self._pending) > self.gate_threshold_ops:
                reason = (f"queue depth "
                          f"{len(self._ops) + len(self._pending)}")
            elif (self._first_unacked_ts is not None
                    and time.time() - self._first_unacked_ts
                    > self.gate_threshold_age):
                reason = "oldest unacked op too old (agent down?)"
            elif len(self.dirty_paths) > 50:
                reason = f"{len(self.dirty_paths)} dirty paths"
        if reason and self.escalation_callback:
            try:
                self.escalation_callback(reason)
            except Exception as e:
                log(f"escalation_callback error: {e}")

    def stats(self) -> dict:
        with self._cond:
            return {
                'epoch': self.epoch,
                'next_seq': self._next_seq,
                'acked_seq': self._acked_seq,
                'undelivered': len(self._ops),
                'pending': len(self._pending),
                'dirty': len(self.dirty_paths),
                'oldest_unacked_age':
                    (time.time() - self._first_unacked_ts)
                    if self._first_unacked_ts else 0.0,
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _journal_write(self, record: dict):
        self._journal_file.write(json.dumps(record) + '\n')

    def _load_dirty(self):
        try:
            with open(self._dirty_path_file, encoding='utf-8') as f:
                self.dirty_paths = set(json.load(f))
            if self.dirty_paths:
                log(f"loaded {len(self.dirty_paths)} dirty paths")
        except (OSError, ValueError):
            self.dirty_paths = set()

    def _save_dirty(self):
        try:
            tmp = self._dirty_path_file + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(sorted(self.dirty_paths), f)
            os.replace(tmp, self._dirty_path_file)
        except OSError as e:
            log(f"could not save dirty set: {e}")
