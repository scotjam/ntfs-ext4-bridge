#!/bin/bash
# Start NTFS-ext4 bridge with partitioned mode for Windows VM
#
# Usage: sudo ./start_bridge_for_vm.sh /path/to/source [port]
#
# This script starts the bridge with --partitioned flag which adds an MBR
# partition table. Windows expects partitioned disks, not raw filesystems.
#
# After starting, connect from Windows VM using wnbd-client:
#   wnbd-client.exe map ntfs-bridge <host-ip> --port 10809
#
# Or configure QEMU with NBD as virtio-blk:
#   -drive file=nbd:127.0.0.1:10809,format=raw,if=virtio

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
SOURCE_DIR="${1:-/tmp/bridge-test/source}"
PORT="${2:-10809}"
IMAGE_PATH="/tmp/bridge-vm/image.raw"
MOUNT_PATH="/mnt/ntfs-bridge"

echo "========================================"
echo "NTFS-ext4 Bridge (Partitioned VM Mode)"
echo "========================================"
echo ""
echo "Source directory: $SOURCE_DIR"
echo "Image path:       $IMAGE_PATH"
echo "NBD port:         $PORT"
echo "Mount path:       $MOUNT_PATH"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Must run as root (for nbd-client and mount)"
    exit 1
fi

# Check source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory does not exist: $SOURCE_DIR"
    echo ""
    echo "Create it with some test files:"
    echo "  mkdir -p $SOURCE_DIR"
    echo "  echo 'hello world' > $SOURCE_DIR/test.txt"
    exit 1
fi

# Check dependencies
for cmd in python3 nbd-client ntfs-3g; do
    if ! command -v $cmd &> /dev/null; then
        echo "ERROR: $cmd not found. Install with:"
        echo "  apt install nbd-client ntfs-3g python3"
        exit 1
    fi
done

# Create directories
mkdir -p "$(dirname "$IMAGE_PATH")"
mkdir -p "$MOUNT_PATH"

# Remove old image to start fresh
if [ -f "$IMAGE_PATH" ]; then
    echo "Removing old image..."
    rm -f "$IMAGE_PATH"
fi

# Kill any existing bridge processes
pkill -f "ntfs_bridge.bridge" 2>/dev/null || true
sleep 1

# Load nbd kernel module
modprobe nbd max_part=16 2>/dev/null || true

# Start the bridge with partitioned mode
echo ""
echo "Starting bridge..."
echo "  --partitioned: MBR partition table for Windows"
echo "  --lazy: Large files allocated on demand"
echo ""

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Run the bridge
# Note: For VM connection via wnbd, use host's IP, not localhost
python3 -m ntfs_bridge.bridge \
    --source "$SOURCE_DIR" \
    --image "$IMAGE_PATH" \
    --mount "$MOUNT_PATH" \
    --port "$PORT" \
    --partitioned \
    --lazy \
    --dealloc-timeout 300

# Note: The bridge runs in foreground. Use Ctrl+C to stop.
