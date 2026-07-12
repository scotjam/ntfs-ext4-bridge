"""Consistency gate: dismount-apply-remount reconciliation for two-way sync.

Used when the live guest-op path can't keep up (bulk ext4 changes), failed
(dirty paths), or is unavailable (agent down). Sequence:

  1. barrier   - stop dispatching ops, drain in-flight acks
  2. offline   - guest takes the disk offline (agent op; WinRM fallback),
                 guaranteeing Windows holds no cached NTFS state
  3. quiesce   - drain the MFT worker queue, set mapper.gate_active,
                 flush the image
  4. apply     - ntfsfix; mount the image file with ntfs-3g (safe: no other
                 NTFS driver can touch it); reconcile every exposed root
                 against ext4 (create/delete/resize/mtime); unmount; ntfsfix
  5. refresh   - hot-cache reload, rescan_mft, re-allocate new sparse files
  6. online    - bump journal epoch, signal the agent to bring the disk
                 back online; Windows re-reads all metadata from scratch

A phase file persists progress so a crash mid-gate is visible; the guest
disk stays offline until a successful gate end, so Windows never sees a
half-applied image.
"""

import json
import os
import subprocess
import threading
import time
import uuid
from typing import Optional

GATE_OP_TIMEOUT_S = 60.0
OFFLINE_CONFIRM_TIMEOUT_S = 120.0
MTIME_TOLERANCE_S = 2.0
RESIDENT_LIMIT = 700  # files <= this are resident in the MFT; don't truncate


def log(msg):
    print(f"[ConsistencyGate] {msg}", flush=True)


class ConsistencyGate:
    """Runs offline reconciliation cycles for the two-way bridge."""

    def __init__(self, bridge):
        self.bridge = bridge
        self.mapper = bridge.mapper
        self.journal = bridge.op_journal
        self.coordinator = bridge.sync_coordinator

        self.state_path = bridge.image_path + '.gate-state.json'
        self.gate_id: Optional[str] = None
        self._phase = 'idle'
        self._agent_confirmed = threading.Event()
        self._agent_online_confirmed = threading.Event()
        self._end_signaled = False

        self._request_event = threading.Event()
        self._request_reason = ''
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_result = ''

        if self.journal:
            self.journal.escalation_callback = self.request

        # A crash mid-gate leaves a stale state file; startup populate
        # already reconciled the image, so just report and clear it.
        try:
            with open(self.state_path, encoding='utf-8') as f:
                stale = json.load(f)
            log(f"WARNING: found stale gate state from a previous run: "
                f"{stale} (startup populate has reconciled; clearing)")
            os.remove(self.state_path)
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="ConsistencyGate")
        self._thread.start()

    def stop(self):
        self._running = False
        self._request_event.set()

    def request(self, reason: str):
        """Ask for a gate run (thread-safe, coalesces repeat requests)."""
        if self._phase != 'idle':
            return
        self._request_reason = reason
        self._request_event.set()

    def on_agent_phase(self, gate_id: str, phase: str) -> dict:
        """Called by ControlServer for agent /v1/gate posts."""
        if gate_id != self.gate_id:
            return {'done': True}  # unknown/old gate: let the agent move on
        if phase == 'offline_confirmed':
            self._agent_confirmed.set()
            return {}
        if phase == 'await_end':
            return {'done': self._end_signaled}
        if phase == 'online_confirmed':
            self._agent_online_confirmed.set()
            return {}
        return {}

    def stats(self) -> dict:
        return {'phase': self._phase, 'gate_id': self.gate_id,
                'last_result': self._last_result}

    # ------------------------------------------------------------------
    # Gate cycle
    # ------------------------------------------------------------------

    def _loop(self):
        while self._running:
            self._request_event.wait(timeout=5.0)
            if not self._running:
                return
            if not self._request_event.is_set():
                continue
            self._request_event.clear()
            reason = self._request_reason
            try:
                self.run_gate(reason)
                self._last_result = 'ok'
            except Exception as e:
                self._last_result = f'failed: {e}'
                log(f"GATE FAILED ({reason}): {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._phase = 'idle'

    def _set_phase(self, phase: str):
        self._phase = phase
        log(f"phase: {phase}")
        try:
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump({'gate_id': self.gate_id, 'phase': phase,
                           'ts': time.time()}, f)
        except OSError:
            pass

    def run_gate(self, reason: str):
        log(f"gate requested: {reason}")
        self.gate_id = uuid.uuid4().hex
        self._agent_confirmed.clear()
        self._agent_online_confirmed.clear()
        self._end_signaled = False

        # 1. barrier
        self._set_phase('barrier')
        self.journal.pause()
        self.journal.drain_barrier(timeout=30.0)
        if self.coordinator:
            self.coordinator.release_all()

        # 2. offline
        self._set_phase('offline')
        agent_did_offline = self._go_offline()

        try:
            # 3. quiesce
            self._set_phase('quiesce')
            self.mapper._mft_queue.join()
            self.mapper.gate_active.set()
            self.mapper.flush()

            # 4. apply
            self._set_phase('apply')
            self._apply_offline()

            # 5. refresh
            self._set_phase('refresh')
            self.mapper.image.reload()
            self.mapper.rescan_mft()
            self.bridge._allocate_new_sparse_files()
            self.bridge._fix_index_alloc_data_sizes()
            self.journal.clear_dirty()
        finally:
            self.mapper.gate_active.clear()

        # 6. online
        self._set_phase('online')
        self.journal.reset_epoch()
        self.journal.resume()
        self._end_signaled = True
        self._go_online(agent_did_offline)

        try:
            os.remove(self.state_path)
        except OSError:
            pass
        self.gate_id = None
        log("gate complete")

    # ------------------------------------------------------------------
    # Offline / online
    # ------------------------------------------------------------------

    def _go_offline(self) -> bool:
        """Take the guest disk offline. Returns True if the agent did it."""
        self.journal.inject_op({'op': 'gate_begin', 'path': '',
                                'gate_id': self.gate_id, '_rel': ''})
        if self._agent_confirmed.wait(timeout=OFFLINE_CONFIRM_TIMEOUT_S):
            log("agent confirmed disk offline")
            return True
        log("agent did not confirm offline; trying WinRM fallback")
        if self._winrm_set_disk_offline(True):
            return False
        raise RuntimeError(
            "could not take the guest disk offline (agent and WinRM both "
            "unavailable) - aborting gate; volume left untouched")

    def _go_online(self, via_agent: bool):
        if via_agent:
            if self._agent_online_confirmed.wait(timeout=180.0):
                log("agent confirmed disk online")
                return
            log("agent did not confirm online; trying WinRM fallback")
        if not self._winrm_set_disk_offline(False):
            log("WARNING: could not bring the guest disk back online "
                "automatically - do it manually (Set-Disk -IsOffline $false)")

    def _winrm_set_disk_offline(self, offline: bool) -> bool:
        b = self.bridge
        if not (b.winrm_url and b.winrm_user):
            return False
        try:
            import winrm
        except ImportError:
            log("pywinrm not installed; WinRM fallback unavailable")
            return False
        try:
            session = winrm.Session(
                b.winrm_url, auth=(b.winrm_user, b.winrm_password or ''),
                transport='ntlm')
            serial = self.bridge.control_server.volume_serial \
                if self.bridge.control_server else ''
            flag = '$true' if offline else '$false'
            # Identify the disk by NBD/virtio friendly name fallback: apply
            # to every non-boot virtio disk. Volume-serial matching is not
            # available at disk level; the agent path is preferred.
            ps = (
                "Get-Disk | Where-Object { $_.Number -ne 0 -and "
                "$_.FriendlyName -match 'VirtIO|NBD' } | "
                f"Set-Disk -IsOffline {flag}"
            )
            r = session.run_ps(ps)
            ok = r.status_code == 0
            log(f"WinRM Set-Disk IsOffline={offline}: "
                f"{'ok' if ok else r.std_err.decode()[:200]}")
            return ok
        except Exception as e:
            log(f"WinRM fallback failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Offline apply (reconciliation)
    # ------------------------------------------------------------------

    def _apply_offline(self):
        image = self.bridge.image_path
        mnt = self.bridge.ntfs_mount + '-gate'
        os.makedirs(mnt, exist_ok=True)

        subprocess.run(['ntfsfix', image], capture_output=True, text=True)
        r = subprocess.run(['mount', '-t', 'ntfs-3g',
                            '-o', 'rw,big_writes', image, mnt],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"gate mount failed: {r.stderr.strip()}")

        try:
            stats = {'created_dirs': 0, 'created_files': 0, 'deleted': 0,
                     'resized': 0, 'retimed': 0}
            for entry in sorted(self.mapper.known_root_entries):
                self._reconcile_root(entry, mnt, stats)
            log(f"apply: {stats}")
        finally:
            subprocess.run(['umount', mnt], capture_output=True, text=True)
            subprocess.run(['ntfsfix', image], capture_output=True, text=True)

    def _reconcile_root(self, entry: str, mnt: str, stats: dict):
        src_root = os.path.realpath(os.path.join(self.bridge.source_dir,
                                                 entry))
        ntfs_root = os.path.join(mnt, entry)
        exclude = self.bridge._should_exclude

        # Pass 1: ext4 -> NTFS (create/resize/retime)
        for dirpath, dirnames, filenames in os.walk(src_root,
                                                    followlinks=True):
            rel_dir = os.path.relpath(dirpath, src_root)
            rel_dir = '' if rel_dir == '.' else rel_dir
            share_rel = os.path.join(entry, rel_dir) if rel_dir else entry
            dirnames[:] = [d for d in dirnames
                           if not exclude(os.path.join(share_rel, d))]
            ntfs_dir = os.path.join(ntfs_root, rel_dir) if rel_dir \
                else ntfs_root
            if not os.path.isdir(ntfs_dir):
                os.makedirs(ntfs_dir, exist_ok=True)
                stats['created_dirs'] += 1
            for name in filenames:
                if exclude(os.path.join(share_rel, name)):
                    continue
                src = os.path.join(dirpath, name)
                dst = os.path.join(ntfs_dir, name)
                try:
                    s = os.stat(src)
                except OSError:
                    continue
                try:
                    d = os.stat(dst)
                except OSError:
                    d = None
                if d is None:
                    # Sparse create: no data copied; the bridge maps the
                    # clusters to ext4 after allocation.
                    with open(dst, 'wb') as f:
                        if s.st_size > 0:
                            f.truncate(s.st_size)
                    os.utime(dst, (s.st_atime, s.st_mtime))
                    stats['created_files'] += 1
                    continue
                if d.st_size != s.st_size and s.st_size > RESIDENT_LIMIT:
                    with open(dst, 'r+b') as f:
                        f.truncate(s.st_size)
                    os.utime(dst, (s.st_atime, s.st_mtime))
                    stats['resized'] += 1
                elif abs(d.st_mtime - s.st_mtime) > MTIME_TOLERANCE_S:
                    os.utime(dst, (s.st_atime, s.st_mtime))
                    stats['retimed'] += 1

        # Pass 2: NTFS -> ext4 deletions (bottom-up so dirs empty out)
        for dirpath, dirnames, filenames in os.walk(ntfs_root,
                                                    topdown=False):
            rel_dir = os.path.relpath(dirpath, ntfs_root)
            rel_dir = '' if rel_dir == '.' else rel_dir
            src_dir = os.path.join(src_root, rel_dir) if rel_dir \
                else src_root
            for name in filenames:
                if not os.path.exists(os.path.join(src_dir, name)):
                    try:
                        os.remove(os.path.join(dirpath, name))
                        stats['deleted'] += 1
                    except OSError as e:
                        log(f"  delete failed: {dirpath}/{name}: {e}")
            for name in dirnames:
                if not os.path.isdir(os.path.join(src_dir, name)):
                    try:
                        os.rmdir(os.path.join(dirpath, name))
                        stats['deleted'] += 1
                    except OSError as e:
                        log(f"  rmdir failed: {dirpath}/{name}: {e}")
