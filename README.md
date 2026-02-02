# NTFS-Ext4 Bridge

An NBD (Network Block Device) server that presents ext4 files as an NTFS volume to Windows.

## Overview

This bridge allows Windows to read and write files stored on a Linux ext4 filesystem by synthesizing an NTFS volume on-the-fly. File data is read directly from ext4 source files while NTFS metadata is served from a template.

## Features

- **Transparent read/write access** - Windows sees a standard NTFS volume
- **Write-through to ext4** - Changes made in Windows are written to Linux source files
- **Dynamic template generation** - Automatically creates NTFS template from source directory
- **MFT tracking** - Monitors NTFS Master File Table changes to update cluster mappings

## Architecture

```
Windows → iSCSI → NBD Server → Template Synthesizer → ext4 files
                      ↓
              NTFS Template (metadata)
```

## Usage

### Dynamic Server (recommended)

Automatically generates NTFS template and starts NBD server:

```bash
sudo python3 -m ntfs_bridge.dynamic_server /path/to/source -p 10809 -s 100
```

### Manual Template + Server

1. Create template:
```bash
sudo python3 -m ntfs_bridge.auto_template /path/to/source template.raw 100
```

2. Start server:
```bash
python3 -m ntfs_bridge.partitioned_server template.raw /path/to/source -p 10809
```

### Expose to Windows via iSCSI

```bash
# Connect NBD client
nbd-client localhost 10809 /dev/nbd0 -N ''

# Set up iSCSI target
tgtadm --lld iscsi --op new --mode target --tid 1 -T iqn.2024-01.com.local:ntfs-bridge
tgtadm --lld iscsi --op new --mode logicalunit --tid 1 --lun 1 -b /dev/nbd0
tgtadm --lld iscsi --op bind --mode target --tid 1 -I ALL
```

## Requirements

- Python 3.8+
- Linux with ntfs-3g (for template creation)
- nbd-client (for local testing)
- tgt (for iSCSI target)

## File Structure

- `dynamic_server.py` - All-in-one dynamic NBD server
- `partitioned_server.py` - NBD server with MBR partition support
- `template_synth.py` - NTFS template synthesizer with cluster mapping
- `auto_template.py` - Automatic NTFS template generator using ntfs-3g

## Limitations

- Files smaller than ~700 bytes are stored as "resident" data in NTFS MFT and cannot be mapped to ext4 source files
- Template must be regenerated if source directory structure changes significantly

## License

MIT
