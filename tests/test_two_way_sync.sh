#!/bin/bash
# NTFS-ext4 Bridge: Two-way sync test suite
#
# Usage: ./test_two_way_sync.sh <source_dir> <ntfs_mount>
#
# Assumes the bridge is already running with source_dir mapped to ntfs_mount.

set -u

SOURCE_DIR="${1:?Usage: $0 <source_dir> <ntfs_mount>}"
NTFS_MOUNT="${2:?Usage: $0 <source_dir> <ntfs_mount>}"

# Sync delay: time to wait for sync operations
SYNC_DELAY=3

PASSED=0
FAILED=0
TOTAL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass_test() {
    PASSED=$((PASSED + 1))
    TOTAL=$((TOTAL + 1))
    echo -e "  ${GREEN}PASS${NC} Test $TOTAL: $1"
}

fail_test() {
    FAILED=$((FAILED + 1))
    TOTAL=$((TOTAL + 1))
    echo -e "  ${RED}FAIL${NC} Test $TOTAL: $1"
    if [ -n "${2:-}" ]; then
        echo -e "       ${YELLOW}$2${NC}"
    fi
}

drop_caches() {
    # Drop filesystem caches to force re-read
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    sleep 0.5
}

echo ""
echo "=================================================="
echo "  NTFS-ext4 Bridge: Two-Way Sync Test Suite"
echo "=================================================="
echo "  Source:  $SOURCE_DIR"
echo "  Mount:   $NTFS_MOUNT"
echo ""

# Verify mount is accessible
if [ ! -d "$NTFS_MOUNT" ]; then
    echo -e "${RED}ERROR: NTFS mount point does not exist: $NTFS_MOUNT${NC}"
    exit 1
fi

# =====================================================================
# Test 1: Initial files visible in NTFS mount
# =====================================================================

if ls "$NTFS_MOUNT" >/dev/null 2>&1; then
    # Check that at least one file from source is visible
    FOUND=0
    for f in "$SOURCE_DIR"/*; do
        name=$(basename "$f")
        if [ -e "$NTFS_MOUNT/$name" ]; then
            FOUND=$((FOUND + 1))
        fi
    done
    if [ $FOUND -gt 0 ]; then
        pass_test "Initial files visible in NTFS mount ($FOUND files found)"
    else
        fail_test "Initial files visible in NTFS mount" "No source files found in mount"
    fi
else
    fail_test "Initial files visible in NTFS mount" "Cannot list NTFS mount"
fi

# =====================================================================
# Test 2: Read file content matches ext4 source
# =====================================================================

if [ -f "$SOURCE_DIR/file1.txt" ] && [ -f "$NTFS_MOUNT/file1.txt" ]; then
    drop_caches
    SOURCE_CONTENT=$(cat "$SOURCE_DIR/file1.txt")
    NTFS_CONTENT=$(cat "$NTFS_MOUNT/file1.txt")
    if [ "$SOURCE_CONTENT" = "$NTFS_CONTENT" ]; then
        pass_test "Read file content matches ext4 source"
    else
        fail_test "Read file content matches ext4 source" \
            "Expected: '$SOURCE_CONTENT', Got: '$NTFS_CONTENT'"
    fi
else
    fail_test "Read file content matches ext4 source" "file1.txt missing"
fi

# =====================================================================
# Test 3: Write file via NTFS mount -> ext4 updated
# =====================================================================

echo "ntfs write test" > "$NTFS_MOUNT/file1.txt"
sync
sleep $SYNC_DELAY

CONTENT=$(cat "$SOURCE_DIR/file1.txt" 2>/dev/null)
if echo "$CONTENT" | grep -q "ntfs write test"; then
    pass_test "Write file via NTFS mount -> ext4 updated"
else
    fail_test "Write file via NTFS mount -> ext4 updated" \
        "ext4 content: '$CONTENT'"
fi

# =====================================================================
# Test 4: Modify file in ext4 -> content visible in NTFS
# =====================================================================

echo "ext4 modification" > "$SOURCE_DIR/file1.txt"
drop_caches
sleep $SYNC_DELAY

CONTENT=$(cat "$NTFS_MOUNT/file1.txt" 2>/dev/null)
if echo "$CONTENT" | grep -q "ext4 modification"; then
    pass_test "Modify file in ext4 -> content visible in NTFS"
else
    fail_test "Modify file in ext4 -> content visible in NTFS" \
        "NTFS content: '$CONTENT'"
fi

# =====================================================================
# Test 5: Create file in ext4 -> appears in NTFS
# =====================================================================

echo "new ext4 file" > "$SOURCE_DIR/newfile_ext4.txt"
sleep $SYNC_DELAY
drop_caches

if [ -f "$NTFS_MOUNT/newfile_ext4.txt" ]; then
    pass_test "Create file in ext4 -> appears in NTFS"
else
    fail_test "Create file in ext4 -> appears in NTFS" \
        "File not found in NTFS mount"
fi

# =====================================================================
# Test 6: Delete file in ext4 -> disappears from NTFS
# =====================================================================

# First create a file to delete
echo "to be deleted" > "$SOURCE_DIR/delete_me.txt"
sleep $SYNC_DELAY
drop_caches

# Verify it exists in NTFS
if [ -f "$NTFS_MOUNT/delete_me.txt" ]; then
    # Now delete from ext4
    rm "$SOURCE_DIR/delete_me.txt"
    sleep $SYNC_DELAY
    drop_caches

    if [ ! -f "$NTFS_MOUNT/delete_me.txt" ]; then
        pass_test "Delete file in ext4 -> disappears from NTFS"
    else
        fail_test "Delete file in ext4 -> disappears from NTFS" \
            "File still exists in NTFS mount"
    fi
else
    fail_test "Delete file in ext4 -> disappears from NTFS" \
        "Setup: file never appeared in NTFS"
fi

# =====================================================================
# Test 7: Create file in NTFS -> appears in ext4
# =====================================================================

echo "new ntfs file" > "$NTFS_MOUNT/newfile_ntfs.txt"
sync
sleep $SYNC_DELAY

if [ -f "$SOURCE_DIR/newfile_ntfs.txt" ]; then
    pass_test "Create file in NTFS -> appears in ext4"
else
    fail_test "Create file in NTFS -> appears in ext4" \
        "File not found in ext4 source"
fi

# =====================================================================
# Test 8: Delete file in NTFS -> disappears from ext4
# =====================================================================

# Create a file to delete
echo "ntfs delete test" > "$NTFS_MOUNT/ntfs_delete_me.txt"
sync
sleep $SYNC_DELAY

if [ -f "$SOURCE_DIR/ntfs_delete_me.txt" ]; then
    rm "$NTFS_MOUNT/ntfs_delete_me.txt"
    sync
    sleep $SYNC_DELAY

    if [ ! -f "$SOURCE_DIR/ntfs_delete_me.txt" ]; then
        pass_test "Delete file in NTFS -> disappears from ext4"
    else
        fail_test "Delete file in NTFS -> disappears from ext4" \
            "File still exists in ext4 source"
    fi
else
    fail_test "Delete file in NTFS -> disappears from ext4" \
        "Setup: file never appeared in ext4"
fi

# =====================================================================
# Test 9: Rename file in ext4 -> renamed in NTFS
# =====================================================================

echo "rename test" > "$SOURCE_DIR/rename_src.txt"
sleep $SYNC_DELAY
drop_caches

if [ -f "$NTFS_MOUNT/rename_src.txt" ]; then
    mv "$SOURCE_DIR/rename_src.txt" "$SOURCE_DIR/rename_dst.txt"
    sleep $SYNC_DELAY
    drop_caches

    if [ -f "$NTFS_MOUNT/rename_dst.txt" ] && [ ! -f "$NTFS_MOUNT/rename_src.txt" ]; then
        pass_test "Rename file in ext4 -> renamed in NTFS"
    else
        fail_test "Rename file in ext4 -> renamed in NTFS" \
            "rename_dst exists: $([ -f "$NTFS_MOUNT/rename_dst.txt" ] && echo yes || echo no), rename_src exists: $([ -f "$NTFS_MOUNT/rename_src.txt" ] && echo yes || echo no)"
    fi
else
    fail_test "Rename file in ext4 -> renamed in NTFS" \
        "Setup: rename_src.txt never appeared in NTFS"
fi

# =====================================================================
# Test 10: Rename file in NTFS -> renamed in ext4
# =====================================================================

echo "ntfs rename test" > "$NTFS_MOUNT/ntfs_rename_src.txt"
sync
sleep $SYNC_DELAY

if [ -f "$SOURCE_DIR/ntfs_rename_src.txt" ]; then
    mv "$NTFS_MOUNT/ntfs_rename_src.txt" "$NTFS_MOUNT/ntfs_rename_dst.txt"
    sync
    sleep $SYNC_DELAY

    if [ -f "$SOURCE_DIR/ntfs_rename_dst.txt" ] && [ ! -f "$SOURCE_DIR/ntfs_rename_src.txt" ]; then
        pass_test "Rename file in NTFS -> renamed in ext4"
    else
        fail_test "Rename file in NTFS -> renamed in ext4" \
            "ntfs_rename_dst exists in ext4: $([ -f "$SOURCE_DIR/ntfs_rename_dst.txt" ] && echo yes || echo no)"
    fi
else
    fail_test "Rename file in NTFS -> renamed in ext4" \
        "Setup: file never appeared in ext4"
fi

# =====================================================================
# Test 11: Create directory in ext4 -> appears in NTFS
# =====================================================================

mkdir -p "$SOURCE_DIR/newdir_ext4"
sleep $SYNC_DELAY
drop_caches

if [ -d "$NTFS_MOUNT/newdir_ext4" ]; then
    pass_test "Create directory in ext4 -> appears in NTFS"
else
    fail_test "Create directory in ext4 -> appears in NTFS" \
        "Directory not found in NTFS mount"
fi

# =====================================================================
# Test 12: Create nested file (dir + file) in ext4 -> appears in NTFS
# =====================================================================

mkdir -p "$SOURCE_DIR/nested_ext4/inner"
echo "deeply nested" > "$SOURCE_DIR/nested_ext4/inner/deep.txt"
sleep $SYNC_DELAY
drop_caches

if [ -f "$NTFS_MOUNT/nested_ext4/inner/deep.txt" ]; then
    pass_test "Create nested file in ext4 -> appears in NTFS"
else
    # Check partial sync
    if [ -d "$NTFS_MOUNT/nested_ext4" ]; then
        fail_test "Create nested file in ext4 -> appears in NTFS" \
            "Directory exists but file not found"
    else
        fail_test "Create nested file in ext4 -> appears in NTFS" \
            "Neither directory nor file found"
    fi
fi

# =====================================================================
# Summary
# =====================================================================

echo ""
echo "=================================================="
echo "  Results: $PASSED passed, $FAILED failed (out of $TOTAL)"
echo "=================================================="
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED test(s) failed${NC}"
    exit 1
fi
