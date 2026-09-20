# Two-Way Live Sync: Design, Operations, and Runbook

## Corruption-safety status (2026-07-12)

A four-dimension corruption review (durability, concurrency, NTFS structure,
two-way sync) plus crash-consistency research drove a hardening pass. Fixed
and verified:

- **Reads never silently return zeros** — a failed mapped-source read is EIO,
  not zero-fill.
- **Renames** (file and directory, incl. nested children) remap the bridge's
  cluster mappings proactively; overwrite-renames drop the orphaned target
  record (no wrong-file delete).
- **Concurrency** — lock-free run-map reads snapshot an immutable list and
  check both bounds; the direct allocators take the lock; the gate blocks I/O
  before draining.
- **Durability** — a persistent MFT op journal replays acked-but-unmaterialized
  ops on restart; NBD FLUSH is a real barrier (drain queue → fsync ext4
  sources → msync image incl. dirty beyond-64MB chunks); `stop()` drains +
  full-flushes.
- **NTFS structure** — data-run LCN encoding handles >2^31-cluster (>8.8TB)
  volumes; `$ATTRIBUTE_LIST` is followed so fragmented file tails map fully;
  the volume dirty bit is cleared so Windows mounts clean and never
  auto-chkdsks the bridge volume.

Remaining known item: a narrow, transient stale-read window right after a
new file's `setvaliddata` and before the bridge maps its clusters
(mitigated, tracked separately). See the bug tracker for full details.

## The problem

The bridge presents ext4 data to Windows as a real local NTFS block device.
Windows' NTFS driver assumes exclusive ownership of that device: it caches
the MFT, `$Bitmap`, and directory indexes indefinitely and offers **no
protocol for invalidating those caches** while the volume is mounted. Any
change made to the blocks underneath a live mount is therefore a coherence
race — the historical SyncDaemon (a second, concurrent ntfs-3g mount)
corrupted volumes by design, and `--virtual` mode collided with Windows'
own MFT/cluster allocations.

Windows offers exactly two reliable invalidation points:

1. **Volume dismount / disk offline-online** — on remount NTFS re-reads
   everything from the device.
2. **Windows itself makes the change** — its caches are authoritative by
   definition.

The two-way architecture uses both.

## Architecture

```
ext4 change (inotify, per share target)
      │
      ▼
 OpJournal ── coalesce ──► sequenced idempotent guest ops (JSONL journal)
      │                            │  HTTP long-poll (token auth)
      │                            ▼
      │                     bridge-agent.ps1 (SYSTEM task in guest)
      │                            │  executes natively on the volume:
      │                            │  New-Item / Remove-Item / Move-Item /
      │                            │  fsutil createnew + setvaliddata /
      │                            │  FileStream.SetLength / mtime set
      │                            ▼
      │                     Windows NTFS driver writes MFT via NBD
      │                            │
      ▼                            ▼
 SyncCoordinator ◄── echo ── ClusterMapper MFT worker
 (suppression)               (maps new records to existing ext4 sources —
                              no data is ever copied)
```

**Why `fsutil createnew + setvaliddata`:** `createnew` creates a file of
the right size without writing data; `setvaliddata` raises the Valid Data
Length so reads go to the *device* instead of returning zeros from the
filesystem layer. The bridge maps those device reads to the ext4 source —
a 40 GB file "syncs" in milliseconds with zero bytes transferred.
(`setvaliddata` requires `SeManageVolumePrivilege`; the agent runs as
SYSTEM.)

**Echo suppression:** every dispatched op opens a suppression window
(`ext4_sync_in_progress`) for its paths. When the resulting MFT writes
arrive, the worker's skip-branches *track* the records (mapping clusters to
ext4) instead of re-materializing, and notify the coordinator, which closes
the window (fallback: ack + 30 s timeout — Windows flushes MFT lazily, so
the agent's ack always precedes the echo; the journal issues a
`flush_volume` after each batch to keep that gap short).

**Consistency gate** (bulk changes, agent outages, scheduled reconcile):

1. barrier — stop dispatch, drain acks
2. offline — guest takes the disk offline (agent op; WinRM fallback)
3. quiesce — drain MFT worker, set `gate_active` (writes → EROFS, reads
   serialize), flush
4. apply — `ntfsfix`; mount the image **file** with ntfs-3g (safe — no
   other NTFS driver can touch it); reconcile every exposed root against
   ext4: create missing (sparse), delete vanished, fix sizes/mtimes;
   unmount; `ntfsfix`
5. refresh — hot-cache reload, `rescan_mft`, re-allocate new sparse files
6. online — bump the journal epoch, guest re-onlines the disk; Windows
   re-reads all metadata from scratch

The guest disk stays offline until a successful gate end, so Windows never
sees a half-applied image. A crash mid-gate is recorded in
`<image>.gate-state.json`.

**Conflict policy:** ext4 wins for share content; NTFS wins for the
overflow dir and Windows-owned trees (System Volume Information,
$RECYCLE.BIN, root-level items). Between gates, Windows-side writes to
unprotected shares propagate to ext4 immediately (existing MFT write
tracking).

## Protocol reference

Endpoint: `http://<control-host>:<control-port>` (default
`192.168.122.1:10810`). Every request carries `X-Bridge-Token` (shared
secret from `<image>.agent-token`, compared with `hmac.compare_digest`).

| Endpoint | Body | Reply |
|---|---|---|
| `POST /v1/hello` | `{agent_version, hostname}` | `{epoch, volume_serial, poll_timeout_s, batch_max}` |
| `POST /v1/poll` | `{cursor}` | `{epoch, ops: [...]}` — long-polls ≤25 s |
| `POST /v1/ack` | `{epoch, results:[{seq,status,code,message}]}` | `{}` (409 on stale epoch) |
| `POST /v1/gate` | `{gate_id, phase}` | phase `await_end` → `{done: bool}` |
| `GET /v1/health` | — | journal/coordinator/gate stats |

Ops (`path`/`dst` are volume-relative, backslashes):
`mkdir`, `rm {recurse}`, `mv {dst}`, `create_sized {size, mtime_ms}`,
`resize {size, mtime_ms}`, `set_mtime {mtime_ms}`, `flush_volume`,
`gate_begin {gate_id}`.

Delivery is **at-least-once** (ops re-sent until acked; all executors are
idempotent). The agent executes in seq order, acks per batch, and persists
its cursor. `epoch` changes on bridge restart and after every gate; on a
mismatch the agent cycles the disk offline/online (dropping stale Windows
caches) and resets its cursor.

## Deployment

Host:

```bash
sudo python3 -m ntfs_bridge.bridge \
    --source /export/bridge-source \
    --image /var/lib/ntfs-bridge/image.raw \
    --mount /mnt/ntfs-bridge \
    --partitioned --lazy --dealloc-timeout 31536000 \
    --roots ShareA,ShareB \
    --two-way \
    --winrm-url 192.168.122.171 --winrm-user user --winrm-password pass
```

- `--roots` = exposure list (which top-level source entries appear on the
  volume). `--protected-roots` = optional read-only subset. For full
  two-way, leave `--protected-roots` unset.
- The agent token is generated at `<image>.agent-token` on first start.
- `--winrm-*` are optional; they enable the gate fallback when the agent
  is unreachable.

Guest (once, elevated):

```powershell
.\install-agent.ps1 -ControlUrl "http://192.168.122.1:10810" -Token "<token>"
```

Operations:

- Manual gate: `kill -USR1 <bridge pid>`.
- Health: `curl -H "X-Bridge-Token: $(cat <image>.agent-token)"
  http://192.168.122.1:10810/v1/health`.
- Agent log: `C:\ProgramData\BridgeAgent\agent.log`.
- Escalation to a gate happens automatically at >500 pending ops (
  `--gate-threshold-ops`) or >10 min unacked (`--gate-threshold-age`).

## Failure modes

| Failure | Behavior |
|---|---|
| Agent offline / VM down | Journal accumulates; age threshold escalates to a gate via WinRM. On reconnect: epoch mismatch → disk cycle → resume |
| Guest op fails | Per-op error ack → path enters the persistent dirty set → repaired by the next gate; >50 dirty paths force one |
| Bridge restart | Journal discarded, epoch bumped; startup populate reconciles; agent's epoch cycle refreshes guest caches |
| VM reboot mid-batch | Unacked ops re-delivered from the agent's persisted cursor; ops idempotent |
| Crash mid-gate | `gate-state.json` records the phase; guest disk stays offline until a successful gate end |
| ext4 changes during a gate | Keep flowing into the (new-epoch) journal; sync via the live path after `gate_end` |

## Status

Verified end-to-end on a real Windows 11 guest (nested KVM under WSL2,
2026-07-12):

- **ext4→NTFS live path — working.** File create, directory create, and
  rename on ext4 propagate to the guest within seconds; content hashes match
  byte-for-byte (proves `createnew`+`setvaliddata` serves real device data,
  not zeros); no echo duplicates appear on ext4.
- **NTFS→ext4 write-through — working.** Files written in the guest to an
  exposed (non-protected) share land on ext4 with matching hashes.
- **Consistency gate — working.** Verified against the live VM: a bulk
  60-file ext4 change grows the `$MFT`, the agent takes the disk offline,
  the bridge applies the delta offline via ntfs-3g, re-derives the grown
  MFT geometry (`Reloaded from image: 151 MFT records, 68 files tracked`),
  re-allocates all sparse files, and the agent brings the disk back online.
  Post-gate: 60/60 files visible in the guest, all sampled content hashes
  match, pre-existing files still read correctly. (The earlier
  "10→3 files" bug — `rescan_mft()` reusing stale MFT geometry — is fixed by
  `reload_from_image()` + a full-image msync before the offline mount.)

Renames are handled correctly: on an ext4-side rename the bridge proactively
remaps its own cluster mappings to the new path (`remap_source_path`), so
reads of the renamed file resolve immediately instead of failing on the gone
old path. Verified live: rename a 300KB file on ext4 → guest shows the new
name at full size, content hash matches, zero read failures.

One minor, non-corrupting issue remains (tracked separately): ntfs-3g leaves
the NTFS dirty bit set, so Windows reports the volume "Scan Needed / Warning"
— reads and writes are unaffected (all content hashes verified in that
state).

## Testing

- **Unit/protocol (no root, no VM):**
  `python -m pytest tests/test_op_journal.py tests/test_control_protocol.py`
- **Gate integration (Linux root, no VM):**
  `sudo python -m pytest tests/test_consistency_gate.py -v`
  (passes in isolation; does NOT exercise concurrent live-path activity —
  that's the untested path where the known gate bug lives)
- **Live VM (nested KVM):** clone a Windows guest, boot with the bridge NBD
  export as a second disk and a slirp `guestfwd` to the control endpoint,
  install the agent, then run the create/rename/delete/write-through checks
  in the runbook below.

## Live VM runbook

Prereqs: bridge running with `--two-way`, VM attached to the NBD volume,
agent installed and polling (check `/v1/health`: `undelivered: 0`).
`SRC` = a share's ext4 path on the host, `V:` = the volume in the guest.

1. **Create:** `head -c 10485760 /dev/urandom > $SRC/runbook-a.bin` — within
   ~5 s the guest shows the file at the right size, and
   `Get-FileHash V:\Share\runbook-a.bin` equals `sha256sum` on ext4.
   Proves createnew + setvaliddata + cluster mapping (not cached zeros).
2. **Rename:** `mv $SRC/runbook-a.bin $SRC/runbook-b.bin` — guest shows the
   rename; **no duplicate appears on ext4** (echo suppression working;
   bridge log shows the exists-branch mapping, not a copy).
3. **Grow/shrink:** `cat >> / truncate -s` an existing file — guest size
   and hash track ext4.
4. **Delete:** `rm $SRC/runbook-b.bin` — gone from the guest; ext4 side
   stays deleted (no resurrection).
5. **Windows→ext4:** create/modify/delete a file under an unprotected share
   in the guest — ext4 content matches by hash within seconds.
6. **Gate under load:** stop the agent task
   (`Stop-ScheduledTask BridgeAgent`), copy 1 000 files into ext4, watch
   escalation in the bridge log → gate runs via WinRM fallback → disk
   cycles offline/online → all files present in the guest; hash-sample 20.
   Restart the agent task.
7. **Reboot resilience:** reboot the VM mid-stream; agent resumes from its
   cursor; no ops lost (compare tree listings).
8. **End-to-end consumer check:** confirm whatever application consumes the
   volume (backup client, indexer) picks up a file created on ext4.
