#!/bin/bash
# Start NTFS-ext4 bridge with partitioned mode for Windows VM (production)
#
# Usage: sudo ./scripts/start_bridge_for_vm.sh [source_dir] [port]
#
# Production defaults:
#   source:  /export/bridge-source
#   image:   /var/lib/ntfs-bridge/image.raw
#   port:    10809
#
# After starting, connect from Windows VM using wnbd-client:
#   wnbd-client.exe map ntfs-bridge <host-ip> --port 10809

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Production defaults
SOURCE_DIR="${1:-/export/bridge-source}"
PORT="${2:-10809}"
IMAGE_PATH="/var/lib/ntfs-bridge/image.raw"
MOUNT_PATH="/mnt/ntfs-bridge"
LOG="/tmp/bridge19.log"

echo "========================================"
echo "NTFS-ext4 Bridge (Partitioned VM Mode)"
echo "========================================"
echo ""
echo "Source directory: $SOURCE_DIR"
echo "Image path:       $IMAGE_PATH"
echo "NBD port:         $PORT"
echo "Mount path:       $MOUNT_PATH"
echo "Log:              $LOG"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Must run as root (for nbd-client and mount)"
    exit 1
fi

# Check source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory does not exist: $SOURCE_DIR"
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

# Load nbd kernel module
modprobe nbd max_part=16 2>/dev/null || true

echo "Starting bridge19 at $(date)" >> "$LOG"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Optional: silent-drop writes for chosen top-level subdirs of the source.
# Operators set PROTECTED_ROOTS=dir1,dir2 in a local wrapper (not committed).
PROTECTED_FLAG=()
if [ -n "${PROTECTED_ROOTS:-}" ]; then
    PROTECTED_FLAG=(--protected-roots "$PROTECTED_ROOTS")
fi

# Optional: send root-level entries Windows creates outside the source tree
# (e.g. System Volume Information, $RECYCLE.BIN, app metadata dirs) to a separate
# overflow directory rather than the source-dir root. Set OVERFLOW_DIR=path
# in a local wrapper (not committed) to point at a partition with room.
OVERFLOW_FLAG=()
if [ -n "${OVERFLOW_DIR:-}" ]; then
    OVERFLOW_FLAG=(--overflow-dir "$OVERFLOW_DIR")
fi

python3 -m ntfs_bridge.bridge \
    --source "$SOURCE_DIR" \
    --image "$IMAGE_PATH" \
    --mount "$MOUNT_PATH" \
    --port "$PORT" \
    --partitioned \
    --lazy \
    --dealloc-timeout 31536000 \
    "${PROTECTED_FLAG[@]}" "${OVERFLOW_FLAG[@]}" >> "$LOG" 2>&1

echo "Bridge exited at $(date) code=$?" >> "$LOG"
