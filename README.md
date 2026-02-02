# NTFS-Ext4 Bridge

An NBD (Network Block Device) server that presents ext4 files as an NTFS volume to Windows.

## Overview

This bridge allows Windows to read and write files stored on a Linux ext4 filesystem by synthesizing an NTFS volume on-the-fly. File data is read directly from ext4 source files while NTFS metadata is served from a template.

## Features

- **Transparent read/write access** - Windows sees a standard NTFS volume
- **Write-through to ext4** - Changes made in Windows are written to Linux source files
- **Dynamic template generation** - Automatically creates NTFS template from source directory
- **MFT tracking** - Monitors NTFS Master File Table changes to update cluster mappings
- **New file/folder creation** - Files and folders created in Windows appear in ext4
- **Rename synchronization** - File and folder renames in Windows sync to ext4

## Architecture

```
Windows  ──►  NBD Client  ──►  NBD Server  ──►  Template Synthesizer  ──►  ext4 files
              (wnbd/iSCSI)     (Linux)                  │
                                                        ▼
                                              NTFS Template (metadata)
```

### Connection Options

1. **wnbd (recommended)** - Windows NBD driver, presents as local disk
2. **iSCSI** - Traditional approach using Linux tgt and Windows iSCSI Initiator

## Quick Start

### 1. Create NTFS Template (Linux)

```bash
# Generate template from source directory (100MB volume)
sudo python3 -m ntfs_bridge.auto_template /path/to/source template.raw 100

# Patch hidden sectors for partition alignment (required!)
python3 -c "
import struct
with open('template.raw', 'r+b') as f:
    boot = bytearray(f.read(512))
    struct.pack_into('<I', boot, 0x1C, 2048)  # Set hidden sectors
    f.seek(0)
    f.write(boot)
"
```

### 2. Start NBD Server (Linux)

```bash
python3 -m ntfs_bridge.partitioned_server template.raw /path/to/source -p 10809
```

### 3. Connect from Windows

#### Option A: Using wnbd (Recommended)

```powershell
# Install wnbd driver first (see https://github.com/cloudbase/wnbd)
wnbd-client map mydisk 192.168.1.12 10809
```

#### Option B: Using iSCSI

On Linux:
```bash
# Connect NBD locally
nbd-client localhost 10809 /dev/nbd0 -N ''

# Set up iSCSI target
tgtadm --lld iscsi --op new --mode target --tid 1 -T iqn.2024-01.com.local:ntfs-bridge
tgtadm --lld iscsi --op new --mode logicalunit --tid 1 --lun 1 -b /dev/nbd0
tgtadm --lld iscsi --op bind --mode target --tid 1 -I ALL
tgtadm --lld iscsi --mode logicalunit --op update --tid 1 --lun 1 --params readonly=0
```

On Windows:
1. Open **iSCSI Initiator** (search in Start menu)
2. Go to **Discovery** tab → **Discover Portal** → Enter Linux IP
3. Go to **Targets** tab → Select target → **Connect**
4. Open **Disk Management** → Disk should appear as NTFS volume

## Dynamic Server (Alternative)

All-in-one server that generates template automatically:

```bash
sudo python3 -m ntfs_bridge.dynamic_server /path/to/source -p 10809 -s 100
```

## Requirements

### Linux Server
- Python 3.8+
- ntfs-3g (for template creation)
- nbd-client (for iSCSI method)
- tgt (for iSCSI method)

### Windows Client
- wnbd driver (for direct NBD) OR
- iSCSI Initiator (built into Windows)

## File Structure

- `partitioned_server.py` - NBD server with MBR partition table
- `template_synth.py` - NTFS template synthesizer with cluster mapping and MFT tracking
- `auto_template.py` - Automatic NTFS template generator using ntfs-3g
- `dynamic_server.py` - All-in-one dynamic NBD server

## How It Works

1. **Template Creation**: An NTFS filesystem is created using ntfs-3g with the source files
2. **Cluster Mapping**: The synthesizer maps NTFS clusters to source file offsets
3. **Read Requests**: MBR and metadata come from template; file data comes from ext4
4. **Write Requests**: Writes to mapped clusters go directly to ext4 source files
5. **MFT Monitoring**: Changes to the Master File Table trigger re-mapping and file sync

## Limitations

- Files smaller than ~700 bytes are stored as "resident" data in NTFS MFT and cannot be mapped to ext4 source files
- Template should be regenerated if source directory structure changes significantly
- Large files work best; small files may need the template regenerated after modification

## Troubleshooting

### Windows shows disk as "RAW"
- Ensure hidden sectors field is patched to 2048 (see Quick Start step 1)
- Try disconnecting and reconnecting, then rescan disks in Disk Manager

### NBD client fails to connect
- Check that server is listening: `netstat -tlnp | grep 10809`
- Verify firewall allows port 10809

### Writes not appearing in ext4
- Only files >700 bytes with non-resident data are mapped
- Check server logs for "Written to" messages

## License

MIT
