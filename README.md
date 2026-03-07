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

### Windows VM
- [WNBD driver](https://github.com/cloudbase/wnbd) — presents NBD as a local SCSI disk
- Network access to host (e.g. via virtio bridge at 192.168.122.1)

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

- **Very large files (40GB+) may fail** — see [Future Work](#future-work) below
- **Resident files** — Files smaller than ~700 bytes are stored inline in NTFS MFT records and cannot be mapped to ext4 source files. Content updates to these files are synced via MFT write tracking.
- **Image recreation** — If the source directory grows beyond 5% of the current image size, the image is recreated on next startup (takes a few minutes for ntfs-3g populate)
- **Single WNBD connection** — Only one Windows VM should connect at a time
- **Two always-failing files** — Files with certain special characters in PAR2 filenames fail to create via ntfs-3g (harmless, logged as warnings)

## Future Work

### Large file support (40GB+)

Files approximately 40GB or larger currently fail during lazy allocation or read. The root cause is likely related to:

- **Data run encoding limits** — NTFS data runs use variable-length encoding. Very large contiguous allocations may produce data run entries that exceed the space available in a single MFT record's $DATA attribute. The current `allocate_file_direct()` finds free clusters and writes data runs into the MFT record, but doesn't handle attribute list overflow or multi-record MFT entries.
- **MFT attribute list** — When a file's attributes (including $DATA with its data runs) exceed the ~1KB MFT record size, NTFS uses an $ATTRIBUTE_LIST to split across multiple MFT records. The bridge does not currently create or manage attribute lists.
- **Bitmap operations at scale** — Finding and marking hundreds of thousands of free clusters (a 40GB file at 64KB clusters = 655,360 clusters) may hit performance or correctness issues in the bitmap scanning code.

Possible approaches:
1. **Fragment large files** — Allocate in smaller chunks (e.g. 4GB segments) to keep data runs compact
2. **Attribute list support** — Implement $ATTRIBUTE_LIST creation for files that need multiple MFT records
3. **Pre-allocation during image creation** — Have ntfs-3g allocate clusters for large files during the initial populate phase (non-sparse), avoiding the need for runtime allocation entirely
4. **Streaming allocation** — Allocate clusters on-demand as reads progress through the file, rather than all at once

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
