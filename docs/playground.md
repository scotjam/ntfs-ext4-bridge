# Playground: trying two-way sync by hand, with the real data out of reach

`tools/playground.py` sets up a live two-way bridge between a Windows VM and
a real ext4 tree in a way that **cannot** modify that tree: the tree is
mounted as the read-only lower layer of an overlayfs, the bridge works on
the overlay's merged view, and everything either side writes lands in the
overlay's upper directory - a ledger you can inspect and throw away.

```
sudo python3 tools/playground.py start     # ~30 min the first time
sudo python3 tools/playground.py status
sudo python3 tools/playground.py stop
```

## What you get

| | where |
|---|---|
| Windows | a new drive letter in the VM (printed by `start`), share folder `<Share>\` |
| ext4 workbench | `<BASE>/source/<Share>/` - the merged view of the real tree; change things here |
| ledger | `<BASE>/upper/<Share>/` - every file either side wrote; deletions are whiteouts |
| bridge log | `<BASE>/bridge.log` |
| agent log | `C:\ProgramData\BridgeAgent\agent.log` in the VM |
| health | `curl -s -H "X-Bridge-Token: $(cat <BASE>/image.raw.agent-token)" http://192.168.122.1:10821/v1/health` |

`<BASE>`, `<Share>`, the real tree and the VM name come from the site config
`/root/.bridge-test.json` (git-ignored; `BRIDGE_TEST_CONFIG=<file>` to
override):

```json
{"lower": "/path/to/the/real/tree",
 "base": "/path/on/the/same/disk/bridge-ro",
 "share": "Shows",
 "vm": "libvirt-domain-name"}
```

`lower` is the tree you want to see in Windows (never written); `base` holds
the overlay's upper/work dirs, the image and the logs, and should be on the
same filesystem as `lower`; `share` is the folder name Windows will see.

## Prerequisites (host)

- Linux host with libvirt/QEMU, `nbd-client`, `ntfs-3g`, `python3`,
  `pywinrm` (`pip3 install pywinrm`), run as root.
- The Windows VM defined in libvirt (its domain name goes in the site
  config, see below) on the default NAT network, so
  `192.168.122.1` is the host as the guest sees it. The VM may be off:
  `start` boots it, detaching its production bridge disk from the
  persistent definition first (it cannot connect while the production
  bridge is down) and restoring it on `stop`.
- The production `ntfs-bridge` service **stopped and disabled** for the
  duration. The playground never touches it, but two bridges on one library
  is not a supported state.
- WinRM enabled in the VM (elevated PowerShell once: `winrm quickconfig
  -quiet`; for an HTTP listener on a workgroup machine also
  `Set-Item WSMan:\localhost\Service\AllowUnencrypted $true` and
  `Set-Item WSMan:\localhost\Service\Auth\Basic $true` if NTLM is not
  available) and an admin account for it.
- Credentials in `/root/.bridge-winrm.json`, mode 600:

  ```json
  {"url": "http://192.168.122.<guest>:5985/wsman", "user": "<admin user>", "password": "..."}
  ```

  Find the guest's address with `virsh net-dhcp-leases default`.

## What `start` does

1. Mounts the overlay (`tests/test_real_library_overlay.py setup`) and takes
   a stat manifest of the real tree.
2. Boots the VM if it is off and waits for WinRM.
3. Starts the bridge in `--two-way` on the merged view: NBD on
   `127.0.0.1:10820`, agent endpoint `192.168.122.1:10821`, token at
   `<BASE>/image.raw.agent-token`. First start populates the image (many
   minutes on a large tree; reused afterwards).
4. Hot-plugs the export into the VM as a virtio disk (`virsh
   attach-device`, first free `vdX`), brings it online, finds the drive
   letter by volume serial.
5. Copies `guest_agent/bridge-agent.ps1` + `install-agent.ps1` into the VM
   over WinRM and installs the `BridgeAgent` scheduled task pointed at this
   bridge.
6. Prints the table above and exits, leaving the bridge running
   (`<BASE>/playground.pid`).

## Things to try

- In Windows: create a folder and some files in `<Share>\`, edit one in
  place, append to one, rename, move between folders, delete. Each appears
  in the ext4 workbench within seconds (and in the ledger).
- On ext4: `mkdir`, `cp`, `truncate`, `mv`, `rm` in the workbench. Each
  appears in Windows after the agent applies it - one op every few seconds;
  a burst of files trickles in.
- Rename a small real file from Windows and back; check
  `cmp` against the real tree afterwards.
- Force a consistency gate: `kill -USR1 $(cat <BASE>/playground.pid)`. The
  agent takes the disk offline, the bridge reconciles the image offline,
  the agent brings it back (under a minute); the volume should be
  identical afterwards.
- Watch `tail -f <BASE>/bridge.log` while doing any of this:
  `dispatch ...` (host change sent to the agent), `FILE RENAMED`,
  `DIR DELETED`, `MATERIALIZED`, `RESIZED` (guest changes applied to ext4),
  `Dropping ... guest write` (see below).

## What to expect

- **~2 s window.** After ext4 creates, resizes or moves a file, guest writes
  to *that file* are dropped until the agent has acked the op plus 2 s.
  Other files are unaffected.
- **Cross-share moves on ext4** reach Windows as delete + create.
- **Windows caches file contents.** An in-place, same-size edit on ext4 is
  served by the bridge on the next read from the device, but a program that
  already has the bytes cached keeps showing the old ones until it reopens
  the file. Namespace changes (names, sizes) are always live.
- **Same-offset concurrent edits** have no merge: the last write to reach
  ext4 wins.
- **Guest -> ext4 takes a few seconds** (3-9 s measured): Windows flushes
  the file's record on its own schedule, and the agent prods it every few
  seconds. A small file's content lives inside that record, so it arrives
  with the flush, not with the save.
- **F5 in Explorer** re-reads Windows' view, which the agent keeps updated;
  a host change shows once the agent has applied it (seconds). If inotify
  ever misses one, the 30 s sweep catches it; `grep rescan <BASE>/bridge.log`
  shows when that happened.
- The real tree never changes: `find <LOWER> -newer <BASE>/playground.pid`
  stays empty. The ledger is the complete record of what would have been
  written.

## Stop and clean up

`stop` removes the agent task, takes the playground disk offline in Windows
and unplugs it, stops the bridge, unmounts the overlay (the ledger under
`<BASE>/upper` is kept for inspection - delete it yourself), shuts the VM
down and restores its production disk definition.

## Troubleshooting

- `start` says "already started": a `playground.pid` exists - run `stop`.
- No drive letter in Windows: `Get-Disk` in the VM; bring the newest disk
  online (`Set-Disk -Number N -IsOffline $false`). Windows may keep a disk
  from an earlier session that was never unplugged cleanly (a "ghost",
  offline, same size); a full `virsh shutdown` + `virsh start` clears
  ghosts - a guest reboot does not.
- Agent not connecting: `Get-ScheduledTask BridgeAgent`, the agent log, and
  `C:\ProgramData\BridgeAgent\config.json` must name
  `http://192.168.122.1:10821` and the current token. `Start-ScheduledTask
  BridgeAgent` after fixing.
- WinRM errors "command line too long": the tool already chunks transfers;
  if you script your own, keep each command under ~1 KB of text.
- The first whole-volume walk from Windows (e.g. `dir /s`) is slow on a big
  tree behind a busy disk; subsequent walks are cached.
