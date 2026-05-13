#!/bin/bash
# NTFS-ext4 Bridge: All-in-one setup and test script for WSL
#
# Usage: sudo ./setup_and_test.sh [/path/to/ext4/source]
#
# If no source path given, creates a test directory with sample files.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
SOURCE_DIR="${1:-/tmp/bridge-test/source}"
IMAGE_PATH="/tmp/bridge-test/image.raw"
MOUNT_POINT="/mnt/ntfs-bridge"
NBD_PORT=10809
BRIDGE_PID=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# =====================================================================
# Step 1: Check prerequisites
# =====================================================================

info "Checking prerequisites..."

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        warn "$1 not found, installing..."
        apt-get install -y "$2" || fail "Failed to install $2"
    fi
}

# Must run as root
if [ "$EUID" -ne 0 ]; then
    fail "Must run as root (use sudo)"
fi

check_cmd nbd-client nbd-client
check_cmd ntfs-3g ntfs-3g
check_cmd mkfs.ntfs ntfs-3g
check_cmd python3 python3

# Load nbd kernel module
if ! lsmod | grep -q '^nbd '; then
    info "Loading nbd kernel module..."
    modprobe nbd max_part=16 || warn "Could not load nbd module (may already be loaded)"
fi

# =====================================================================
# Step 2: Create test source directory if needed
# =====================================================================

if [ ! -d "$SOURCE_DIR" ]; then
    info "Creating test source directory: $SOURCE_DIR"
    mkdir -p "$SOURCE_DIR"
    echo "hello world" > "$SOURCE_DIR/file1.txt"
    echo "test data" > "$SOURCE_DIR/file2.txt"
    mkdir -p "$SOURCE_DIR/subdir"
    echo "nested file" > "$SOURCE_DIR/subdir/file3.txt"
    dd if=/dev/urandom of="$SOURCE_DIR/largefile.bin" bs=1M count=5 2>/dev/null
    info "Created sample files in $SOURCE_DIR"
else
    info "Using existing source directory: $SOURCE_DIR"
fi

# =====================================================================
# Step 3: Clean up any previous state
# =====================================================================

info "Cleaning up previous state..."

# Unmount if mounted
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    umount "$MOUNT_POINT" 2>/dev/null || fusermount -u "$MOUNT_POINT" 2>/dev/null || true
fi

# Disconnect any nbd clients
for dev in /dev/nbd{0..15}; do
    if [ -e "$dev" ]; then
        nbd-client -d "$dev" 2>/dev/null || true
    fi
done

# Remove old image
rm -f "$IMAGE_PATH"

# Create dirs
mkdir -p "$(dirname "$IMAGE_PATH")"
mkdir -p "$MOUNT_POINT"

# =====================================================================
# Step 4: Start the bridge
# =====================================================================

info "Starting NTFS-ext4 bridge..."
info "  Source: $SOURCE_DIR"
info "  Image: $IMAGE_PATH"
info "  Mount: $MOUNT_POINT"
info "  Port: $NBD_PORT"

cd "$PROJECT_DIR"
python3 -m ntfs_bridge.bridge \
    --source "$SOURCE_DIR" \
    --image "$IMAGE_PATH" \
    --mount "$MOUNT_POINT" \
    --port "$NBD_PORT" &
BRIDGE_PID=$!

# Wait for bridge to start
info "Waiting for bridge to start (PID: $BRIDGE_PID)..."
sleep 5

# Check if bridge is still running
if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    fail "Bridge process died"
fi

# Check if mount is up
if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    warn "Mount point not ready, waiting longer..."
    sleep 5
    if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        warn "Bridge is running but mount is not up"
        warn "Check bridge output above for errors"
        warn "You may need to connect manually:"
        warn "  sudo nbd-client -N '' 127.0.0.1 $NBD_PORT /dev/nbd0"
        warn "  sudo mount -t ntfs-3g /dev/nbd0 $MOUNT_POINT"
    fi
fi

# =====================================================================
# Step 5: Run tests
# =====================================================================

info "Running two-way sync tests..."

if [ -f "$SCRIPT_DIR/../tests/test_two_way_sync.sh" ]; then
    bash "$SCRIPT_DIR/../tests/test_two_way_sync.sh" \
        "$SOURCE_DIR" "$MOUNT_POINT"
    TEST_RESULT=$?
else
    warn "Test script not found, skipping tests"
    TEST_RESULT=0
fi

# =====================================================================
# Step 6: Cleanup
# =====================================================================

cleanup() {
    info "Cleaning up..."

    if [ -n "$BRIDGE_PID" ] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill "$BRIDGE_PID" 2>/dev/null
        wait "$BRIDGE_PID" 2>/dev/null || true
    fi

    if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        umount "$MOUNT_POINT" 2>/dev/null || fusermount -u "$MOUNT_POINT" 2>/dev/null || true
    fi

    for dev in /dev/nbd{0..15}; do
        if [ -e "$dev" ]; then
            nbd-client -d "$dev" 2>/dev/null || true
        fi
    done

    info "Cleanup complete"
}

trap cleanup EXIT

# Keep running if tests passed, or exit
if [ $TEST_RESULT -eq 0 ]; then
    info "All tests passed!"
else
    fail "Some tests failed (exit code: $TEST_RESULT)"
fi

exit $TEST_RESULT
