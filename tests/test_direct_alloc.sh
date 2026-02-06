#!/bin/bash
# Test direct allocation: first read should return correct data immediately

set -e

SOURCE="/tmp/bridge-test/source"
MOUNT="/mnt/ntfs-bridge"
LOGFILE="/tmp/bridge.log"

echo "=== Direct Allocation Test ==="
echo ""
echo "This test verifies that with --lazy flag, the first read of a"
echo "large file returns correct data immediately (no stale data)."
echo ""

# Check if bridge is running with lazy mode
if ! pgrep -f "ntfs_bridge.bridge" > /dev/null; then
    echo "ERROR: Bridge not running. Start with:"
    echo "  sudo python3 -m ntfs_bridge.bridge --source $SOURCE --image /tmp/bridge-test/image.raw --mount $MOUNT --lazy"
    exit 1
fi

# Create a unique test file
TESTFILE="direct_alloc_test_$$.bin"
echo "Creating test file: $TESTFILE"
dd if=/dev/urandom of="$SOURCE/$TESTFILE" bs=1M count=2 2>/dev/null
EXT4_HASH=$(md5sum "$SOURCE/$TESTFILE" | cut -d' ' -f1)
echo "ext4 hash: $EXT4_HASH"

# Wait for sync daemon to pick it up
echo "Waiting for sync..."
sleep 3

# Check file exists in NTFS
if [ ! -f "$MOUNT/$TESTFILE" ]; then
    echo "ERROR: File not found in NTFS mount"
    ls -la "$MOUNT/"
    exit 1
fi

# Read the file ONCE and check hash
echo ""
echo "Reading file once (first read test)..."
NTFS_HASH=$(md5sum "$MOUNT/$TESTFILE" | cut -d' ' -f1)
echo "ntfs hash: $NTFS_HASH"

if [ "$EXT4_HASH" = "$NTFS_HASH" ]; then
    echo ""
    echo "PASS: First read returned correct data!"
    echo ""
    echo "Direct allocation is working correctly."
else
    echo ""
    echo "FAIL: First read returned incorrect data"
    echo "  expected: $EXT4_HASH"
    echo "  got: $NTFS_HASH"
    echo ""
    echo "Recent bridge log:"
    tail -30 "$LOGFILE" 2>/dev/null || echo "(no log file)"
    exit 1
fi

# Clean up
rm -f "$SOURCE/$TESTFILE"
echo ""
echo "Test passed, cleaned up test file."
