# NTFS-Ext4 Bridge

An NBD server that presents ext4 files as an NTFS volume to Windows VMs.

## Overview

This bridge allows a Windows VM (running under KVM/QEMU) to see files stored on a Linux ext4 filesystem as a standard NTFS volume. File data is read directly from ext4 source files while NTFS metadata is served from a template image. Changes made in Windows are written back to the ext4 source files.

## Features

- **Transparent read/write access** — Windows sees a standard NTFS fixed disk
- **Write-through to ext4** — Changes made in Windows are written to Linux source files
- **Automatic image creation** — Generates NTFS image from source directory using ntfs-3g
- **Lazy allocation** — Large files are created sparse; clusters are allocated on first access
- **GPT partition table** — Presents as a real partitioned disk for Windows compatibility
- **Overflow directory** — Windows system files (System Volume Information, etc.) are redirected to a data disk to avoid filling the root filesystem
- **Image reuse** — Restarts reuse the existing image if the size is within 5% tolerance
- **File skip on restart** — Files already on the NTFS mount are skipped during populate, making restarts fast
- **Bidirectional sync** — File/folder creation, deletion, and renames sync between NTFS and ext4

## Architecture

```
Windows VM                    Linux Host (KVM)
┌──────────────┐              ┌──────────────────────────────────────┐
│              │              │                                      │
│  Windows app │              │  NTFSBridge (bridge.py)              │
│      ↓       │              │    ├── ClusterMapper                 │
│  NTFS (F:)   │              │    │     ├── MFT scan + tracking     │
│      ↓       │   NBD/TCP    │    │     ├── cluster_map → ext4      │
│  WNBD driver ├──────────────┤    │     └── lazy allocation        │
│              │              │    ├── PartitionWrapper (GPT)        │
│              │              │    ├── NBDServer                     │
│              │              │    ├── SyncDaemon (ext4 → NTFS)      │
│              │              │    └── FileWatcher (inotify)         │
│              │              │                                      │
│              │              │  Data flow:                          │
│              │              │    Reads:  cluster_map → ext4 files  │
│              │              │    Writes: MFT tracking → ext4 files │
│              │              │    Metadata: NTFS image file         │
└──────────────┘              └──────────────────────────────────────┘
```

## Requirements

### Linux Host
- Python 3.8+
- ntfs-3g (for image creation and mounting)
- nbd-client (for local NBD connection)
- KVM/QEMU with a Windows VM
- `pywinrm` (`pip3 install pywinrm`) — optional, for remote management without VNC

### Windows VM
- [WNBD driver](https://github.com/cloudbase/wnbd) — presents NBD as a local SCSI disk
- Network access to host (e.g. via virtio bridge at 192.168.122.1)
- WinRM enabled — optional, for remote management without VNC

## Usage

### Starting the Bridge

```bash
sudo python3 -m ntfs_bridge.bridge \
    --source /export/bridge-source \
    --image /export/media/bridge-test/image.raw \
    --mount /mnt/ntfs-bridge \
    --port 10809 \
    --partitioned \
    --lazy \
    --dealloc-timeout 86400 \
    --overflow-dir /export/media/bridge-overflow
```

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--source` | Yes | — | Path to ext4 source directory (can contain symlinks) |
| `--image` | Yes | — | Path to NTFS image file (created if missing) |
| `--mount` | Yes | — | Path to mount NTFS via ntfs-3g |
| `--port` | No | 10809 | NBD server port |
| `--size` | No | auto | Minimum image size in MB (auto-calculated from source) |
| `--partitioned` | No | off | Add GPT partition table (required for Windows VM) |
| `--lazy` | No | off | Enable lazy allocation for large files |
| `--dealloc-timeout` | No | 60 | Seconds after last read before deallocating lazy clusters |
| `--overflow-dir` | No | source dir | Directory for Windows-created root items |
| `--virtual` | No | off | Virtual file mode for live ext4→NTFS sync |

### Connecting from Windows

In the Windows VM, use WNBD to connect:

```powershell
C:\wnbd.exe map ntfs-bridge 192.168.122.1 --port 10809
```

Then bring the disk online:

```powershell
# Bring disk online (it may appear offline by default)
Set-Disk -Number 1 -IsOffline $false
Set-Disk -Number 1 -IsReadOnly $false

# Check drive letter assignment
Get-Partition -DiskNumber 1
Get-Volume -DriveLetter F
```

### Disconnecting

```powershell
C:\wnbd.exe unmap ntfs-bridge
```

### Remote Management (no VNC required)

You can start the bridge and connect the Windows VM entirely from the Linux host terminal, without opening a VNC session.

#### One-time setup on the Windows VM

Enable WinRM (run once in an elevated PowerShell session inside the VM):

```powershell
winrm quickconfig -quiet
# Allow NTLM authentication
Set-Item WSMan:\localhost\Service\Auth\Ntlm -Value $true
```

#### One-time setup on the Linux host

```bash
pip3 install pywinrm
```

#### Starting the bridge and connecting Windows

```python
import winrm
import subprocess

# 1. Start the bridge on the Linux host (run as root)
# subprocess.Popen(['python3', '-m', 'ntfs_bridge.bridge', '--source', '...', ...])

# 2. Connect to the Windows VM via WinRM
s = winrm.Session('192.168.122.171', auth=('user', 'password'), transport='ntlm')

# 3. Map the WNBD drive
r = s.run_cmd('C:\\wnbd.exe', ['map', 'ntfs-bridge', '192.168.122.1', '--port', '10809'])

# 4. Bring the disk online
s.run_ps('Set-Disk -Number 1 -IsOffline $false; Set-Disk -Number 1 -IsReadOnly $false')
```

Or as a one-liner from the shell:

```bash
# Check current WNBD mappings
python3 -c "
import winrm
s = winrm.Session('192.168.122.171', auth=('user', 'password'), transport='ntlm')
print(s.run_cmd('C:\\\\wnbd.exe', ['list']).std_out.decode())
"

# Map the drive
python3 -c "
import winrm
s = winrm.Session('192.168.122.171', auth=('user', 'password'), transport='ntlm')
s.run_cmd('C:\\\\wnbd.exe', ['map', 'ntfs-bridge', '192.168.122.1', '--port', '10809'])
s.run_ps('Set-Disk -Number 1 -IsOffline \$false; Set-Disk -Number 1 -IsReadOnly \$false')
print('Done')
"

# Unmap the drive
python3 -c "
import winrm
s = winrm.Session('192.168.122.171', auth=('user', 'password'), transport='ntlm')
s.run_cmd('C:\\\\wnbd.exe', ['unmap', 'ntfs-bridge'])
print('Done')
"
```

### Source Directory Setup

The source directory should contain symlinks to actual data directories:

```
/export/bridge-source/
├── Documents -> /srv/.../documents
├── KidsMovies -> /srv/.../kidsmovies
└── KidsTV -> /srv/.../kidstv
```

This allows the bridge to serve data from multiple physical locations while presenting a unified NTFS volume. The bridge follows symlinks when calculating image size and populating the NTFS image.

### Overflow Directory

When `--overflow-dir` is set, root-level items created by Windows that don't match existing source directory entries are stored in the overflow directory instead. This prevents Windows system files from filling the source directory's filesystem.

Items redirected to overflow:
- `System Volume Information` (Windows creates this automatically)
- User SID folders (e.g. `S-1-5-21-...`)
- Any new root-level file/folder not matching a source entry

Items that stay in source:
- Top-level entries that existed when the bridge started (e.g. `Documents`, `KidsMovies`, `KidsTV`)

## How It Works

1. **Image creation**: ntfs-3g creates an NTFS filesystem with all source files (sparse, no data copied)
2. **MFT scan**: ClusterMapper scans the MFT to map NTFS clusters → ext4 file offsets
3. **NBD serving**: NBDServer handles read/write requests from WNBD
4. **Read path**: Data clusters → read from ext4 file; metadata → read from image
5. **Write path**: MFT changes → tracked and synced to ext4; data writes → written to ext4
6. **Lazy allocation**: Sparse files are allocated on first read access (bitmap + MFT update)

## Known Limitations

- **Resident files** — Files smaller than ~700 bytes are stored inline in NTFS MFT records and cannot be mapped to ext4 source files. Content updates to these files are synced via MFT write tracking.
- **Image recreation** — If the source directory grows beyond 5% of the current image size, the image is recreated on next startup (takes a few minutes for ntfs-3g populate)
- **Single WNBD connection** — Only one Windows VM should connect at a time
- **Two always-failing files** — Files with certain special characters in PAR2 filenames fail to create via ntfs-3g (harmless, logged as warnings)
- **Extreme fragmentation** — If the NTFS image bitmap becomes severely fragmented (e.g. from many alloc/dealloc cycles), very large files may fail to pre-allocate. Recreating the image resolves this.

## Future Work

### Other improvements

- **Automatic WNBD reconnection** — Detect WNBD disconnects and reconnect automatically
- **Multiple VM support** — Allow read-only connections from multiple VMs simultaneously
- **Incremental MFT updates** — Only re-scan changed MFT records on restart instead of full scan
- **Health monitoring** — Periodic MD5 spot-checks to verify data integrity
- **Systemd service** — Package as a systemd service for automatic startup

## Troubleshooting

### Windows shows disk as RAW or won't mount
- Ensure `--partitioned` flag is used (Windows expects a partitioned disk)
- Try `Set-Disk -Number 1 -IsOffline $false` in PowerShell
- Check bridge log for errors during MFT scan

### Files show zero content on Windows
- The MFT scan may have stopped early — check bridge log for "MFT has N records" and "files tracked"
- Restart the bridge to trigger a fresh MFT scan

### Bridge startup is slow
- First startup creates the NTFS image and populates with ntfs-3g (a few minutes for large source trees)
- Subsequent restarts reuse the image and skip existing files (should be fast)
- Pre-allocation of sparse files runs in the background and doesn't block serving

### WNBD disconnects after ~90 requests
- Ensure only one bridge process is running (`ps aux | grep ntfs_bridge`)
- Kill any stale processes before restarting

### Source directory running out of space
- Use `--overflow-dir` pointing to a disk with free space
- Check that `System Volume Information` and SID folders are in the overflow directory

## License

MIT
