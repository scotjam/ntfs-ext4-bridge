#!/bin/bash
# Full test of NTFS-ext4 bridge with large file synchronization
# Run on a Linux VM with: sudo bash test_vm_large_sync.sh

set -e

echo "========================================"
echo "NTFS-ext4 Bridge - Large File Sync Test"
echo "========================================"
echo ""

PROJECT_DIR="/opt/ntfs-ext4-bridge"
SOURCE="/tmp/bridge-test/source"
IMAGE="/tmp/bridge-test/image.raw"
MOUNT="/mnt/ntfs-bridge"

# Cleanup any previous state
echo "=== Cleanup ==="
pkill -f "ntfs_bridge.bridge" 2>/dev/null || true
sleep 1
umount "$MOUNT" 2>/dev/null || true
for dev in /dev/nbd*; do
    [ -b "$dev" ] && nbd-client -d "$dev" 2>/dev/null || true
done
rm -f "$IMAGE"
rm -rf "$SOURCE"

# Load nbd module
modprobe nbd max_part=16 2>/dev/null || true

# Create test source directory with files of various sizes
echo ""
echo "=== Creating test data ==="
mkdir -p "$SOURCE/subdir"
echo "hello world" > "$SOURCE/file1.txt"
echo "second file content" > "$SOURCE/file2.txt"
echo "nested file" > "$SOURCE/subdir/file3.txt"

echo "Creating 1MB file..."
dd if=/dev/urandom of="$SOURCE/large1.bin" bs=1M count=1 2>/dev/null
echo "Creating 5MB file..."
dd if=/dev/urandom of="$SOURCE/large5.bin" bs=1M count=5 2>/dev/null
echo "Creating 10MB file..."
dd if=/dev/urandom of="$SOURCE/large10.bin" bs=1M count=10 2>/dev/null
echo "Creating 50MB file..."
dd if=/dev/urandom of="$SOURCE/large50.bin" bs=1M count=50 2>/dev/null

# Record original hashes
HASH_1M=$(md5sum "$SOURCE/large1.bin" | cut -d" " -f1)
HASH_5M=$(md5sum "$SOURCE/large5.bin" | cut -d" " -f1)
HASH_10M=$(md5sum "$SOURCE/large10.bin" | cut -d" " -f1)
HASH_50M=$(md5sum "$SOURCE/large50.bin" | cut -d" " -f1)

echo ""
echo "Source hashes:"
echo "  1MB:  $HASH_1M"
echo "  5MB:  $HASH_5M"
echo "  10MB: $HASH_10M"
echo "  50MB: $HASH_50M"

mkdir -p "$MOUNT"

# Start bridge
echo ""
echo "=== Starting bridge ==="
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

python3 -m ntfs_bridge.bridge \
    --source "$SOURCE" \
    --image "$IMAGE" \
    --mount "$MOUNT" \
    --port 10809 &
BRIDGE_PID=$!

echo "Bridge PID: $BRIDGE_PID"
echo "Waiting for bridge to start..."
sleep 8

if ! kill -0 $BRIDGE_PID 2>/dev/null; then
    echo "FATAL: Bridge process died!"
    exit 1
fi

if ! mountpoint -q "$MOUNT" 2>/dev/null; then
    sleep 5
    if ! mountpoint -q "$MOUNT" 2>/dev/null; then
        echo "FATAL: NTFS mount not available at $MOUNT"
        kill $BRIDGE_PID 2>/dev/null || true
        exit 1
    fi
fi

echo "Bridge running, mount available."
echo ""

PASS=0
FAIL=0

run_test() {
    local name="$1"
    local result="$2"
    if [ "$result" = "PASS" ]; then
        echo "  PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name"
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo "  LARGE FILE SYNC TESTS"
echo "========================================"
echo ""

# Test 1: 1MB read integrity
echo "--- Test 1: 1MB file read integrity ---"
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
NTFS_HASH=$(md5sum "$MOUNT/large1.bin" | cut -d" " -f1)
[ "$HASH_1M" = "$NTFS_HASH" ] && run_test "1MB read integrity" "PASS" || run_test "1MB read integrity (ext4=$HASH_1M ntfs=$NTFS_HASH)" "FAIL"

# Test 2: 5MB read integrity
echo "--- Test 2: 5MB file read integrity ---"
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
NTFS_HASH=$(md5sum "$MOUNT/large5.bin" | cut -d" " -f1)
[ "$HASH_5M" = "$NTFS_HASH" ] && run_test "5MB read integrity" "PASS" || run_test "5MB read integrity (ext4=$HASH_5M ntfs=$NTFS_HASH)" "FAIL"

# Test 3: 10MB read integrity
echo "--- Test 3: 10MB file read integrity ---"
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
NTFS_HASH=$(md5sum "$MOUNT/large10.bin" | cut -d" " -f1)
[ "$HASH_10M" = "$NTFS_HASH" ] && run_test "10MB read integrity" "PASS" || run_test "10MB read integrity (ext4=$HASH_10M ntfs=$NTFS_HASH)" "FAIL"

# Test 4: 50MB read integrity
echo "--- Test 4: 50MB file read integrity ---"
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
NTFS_HASH=$(md5sum "$MOUNT/large50.bin" | cut -d" " -f1)
[ "$HASH_50M" = "$NTFS_HASH" ] && run_test "50MB read integrity" "PASS" || run_test "50MB read integrity (ext4=$HASH_50M ntfs=$NTFS_HASH)" "FAIL"

# Test 5: Write 1MB via NTFS -> ext4
echo "--- Test 5: Write 1MB via NTFS -> ext4 ---"
dd if=/dev/urandom of="$MOUNT/large1.bin" bs=1M count=1 2>/dev/null
sync
sleep 2
NTFS_HASH=$(md5sum "$MOUNT/large1.bin" | cut -d" " -f1)
EXT4_HASH=$(md5sum "$SOURCE/large1.bin" | cut -d" " -f1)
[ "$NTFS_HASH" = "$EXT4_HASH" ] && run_test "Write 1MB NTFS->ext4" "PASS" || run_test "Write 1MB NTFS->ext4 (ntfs=$NTFS_HASH ext4=$EXT4_HASH)" "FAIL"

# Test 6: Modify 5MB in ext4 -> visible in NTFS
echo "--- Test 6: Modify 5MB in ext4 -> NTFS ---"
dd if=/dev/urandom of="$SOURCE/large5.bin" bs=1M count=5 2>/dev/null
NEW_HASH=$(md5sum "$SOURCE/large5.bin" | cut -d" " -f1)
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
sleep 2
NTFS_HASH=$(md5sum "$MOUNT/large5.bin" | cut -d" " -f1)
[ "$NEW_HASH" = "$NTFS_HASH" ] && run_test "Modify 5MB ext4->NTFS" "PASS" || run_test "Modify 5MB ext4->NTFS (ext4=$NEW_HASH ntfs=$NTFS_HASH)" "FAIL"

# Test 7: Create new 2MB in ext4 -> appears in NTFS
echo "--- Test 7: Create 2MB in ext4 -> NTFS ---"
dd if=/dev/urandom of="$SOURCE/new_large.bin" bs=1M count=2 2>/dev/null
NEW_HASH=$(md5sum "$SOURCE/new_large.bin" | cut -d" " -f1)
sleep 5
if [ -f "$MOUNT/new_large.bin" ]; then
    NTFS_HASH=$(md5sum "$MOUNT/new_large.bin" | cut -d" " -f1)
    [ "$NEW_HASH" = "$NTFS_HASH" ] && run_test "Create 2MB ext4->NTFS" "PASS" || run_test "Create 2MB ext4->NTFS (hash mismatch)" "FAIL"
else
    run_test "Create 2MB ext4->NTFS (file not found in NTFS)" "FAIL"
    ls -la "$MOUNT/" | sed "s/^/    /"
fi

# Test 8: Create 2MB in NTFS -> appears in ext4
echo "--- Test 8: Create 2MB in NTFS -> ext4 ---"
dd if=/dev/urandom of="$MOUNT/ntfs_created.bin" bs=1M count=2 2>/dev/null
sync
sleep 3
NTFS_HASH=$(md5sum "$MOUNT/ntfs_created.bin" | cut -d" " -f1)
if [ -f "$SOURCE/ntfs_created.bin" ]; then
    EXT4_HASH=$(md5sum "$SOURCE/ntfs_created.bin" | cut -d" " -f1)
    [ "$NTFS_HASH" = "$EXT4_HASH" ] && run_test "Create 2MB NTFS->ext4" "PASS" || run_test "Create 2MB NTFS->ext4 (hash mismatch)" "FAIL"
else
    run_test "Create 2MB NTFS->ext4 (file not found in ext4)" "FAIL"
    ls -la "$SOURCE/" | sed "s/^/    /"
fi

# Test 9: Partial write to 10MB file
echo "--- Test 9: Partial write to 10MB file ---"
echo "PARTIAL_WRITE_MARKER" | dd of="$MOUNT/large10.bin" bs=1 seek=500000 conv=notrunc 2>/dev/null
sync
sleep 2
if grep -q "PARTIAL_WRITE_MARKER" "$SOURCE/large10.bin"; then
    run_test "Partial write to 10MB -> ext4" "PASS"
else
    run_test "Partial write to 10MB -> ext4" "FAIL"
fi

# Test 10: Small file integrity
echo "--- Test 10: Small file integrity ---"
ORIG=$(cat "$SOURCE/file1.txt")
NTFS=$(cat "$MOUNT/file1.txt")
[ "$ORIG" = "$NTFS" ] && run_test "Small file integrity" "PASS" || run_test "Small file integrity" "FAIL"

# Test 11: Bidirectional small file sync
echo "--- Test 11: NTFS->ext4 small file ---"
echo "ntfs-created-content" > "$MOUNT/fromntfs.txt"
sync
sleep 2
if [ -f "$SOURCE/fromntfs.txt" ] && grep -q "ntfs-created-content" "$SOURCE/fromntfs.txt"; then
    run_test "NTFS->ext4 small file sync" "PASS"
else
    run_test "NTFS->ext4 small file sync" "FAIL"
fi

echo "--- Test 12: ext4->NTFS small file ---"
echo "created-in-ext4" > "$SOURCE/fromext4.txt"
sleep 4
if [ -f "$MOUNT/fromext4.txt" ] && grep -q "created-in-ext4" "$MOUNT/fromext4.txt"; then
    run_test "ext4->NTFS small file sync" "PASS"
else
    run_test "ext4->NTFS small file sync" "FAIL"
fi

# Summary
echo ""
echo "========================================"
echo "  RESULTS: $PASS passed, $FAIL failed out of 12 tests"
echo "========================================"

# Cleanup
echo ""
echo "=== Cleanup ==="
kill $BRIDGE_PID 2>/dev/null || true
sleep 2
umount "$MOUNT" 2>/dev/null || true
for dev in /dev/nbd*; do
    [ -b "$dev" ] && nbd-client -d "$dev" 2>/dev/null || true
done

if [ $FAIL -gt 0 ]; then
    exit 1
fi
echo "All tests passed!"
