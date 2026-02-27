#!/bin/bash
# Three-way comparison test: original -> ext4 copy -> NTFS bridge
# Tests that large files (Undercover S1+S2, ~35GB) survive the bridge intact.

set -e

ORIG="/export/media/Undercover (2019)"
SOURCE="/export/media/bridge-test/source"
IMAGE="/export/media/bridge-test/image.raw"
MOUNT="/mnt/ntfs-bridge"
HASHDIR="/export/media/bridge-test"

echo "========================================"
echo "NTFS-ext4 Bridge - Undercover Sync Test"
echo "========================================"
echo ""

# Cleanup
pkill -f "ntfs_bridge.bridge" 2>/dev/null || true
sleep 1
umount "$MOUNT" 2>/dev/null || true
nbd-client -d /dev/nbd0 2>/dev/null || true
nbd-client -d /dev/nbd1 2>/dev/null || true
rm -f "$IMAGE"
modprobe nbd max_part=16 2>/dev/null || true
mkdir -p "$MOUNT"

echo "=== Phase 1: Hash original source files ==="
if [ ! -f "$HASHDIR/hashes_original.txt" ]; then
    cd "$SOURCE"
    find . -type f \( -name "*.mkv" -o -name "*.mp4" \) | sort | while read f; do
        orig="$ORIG/${f#./}"
        [ -f "$orig" ] && md5sum "$orig"
    done > "$HASHDIR/hashes_original.txt"
fi
echo "  $(wc -l < "$HASHDIR/hashes_original.txt") original hashes"

echo ""
echo "=== Phase 2: Hash ext4 copies ==="
if [ ! -f "$HASHDIR/hashes_ext4_copy.txt" ]; then
    cd "$SOURCE"
    find . -type f \( -name "*.mkv" -o -name "*.mp4" \) | sort | while read f; do
        md5sum "$SOURCE/${f#./}"
    done > "$HASHDIR/hashes_ext4_copy.txt"
fi
echo "  $(wc -l < "$HASHDIR/hashes_ext4_copy.txt") ext4 copy hashes"

echo ""
echo "=== Phase 3: Start bridge ==="
cd /opt/ntfs-ext4-bridge
export PYTHONPATH="/opt/ntfs-ext4-bridge:$PYTHONPATH"

python3 -m ntfs_bridge.bridge \
    --source "$SOURCE" \
    --image "$IMAGE" \
    --mount "$MOUNT" \
    --port 10809 \
    --lazy \
    --dealloc-timeout 3600 &
BRIDGE_PID=$!

echo "Bridge PID: $BRIDGE_PID"
echo "Waiting for bridge startup..."
sleep 12

if ! kill -0 $BRIDGE_PID 2>/dev/null; then
    echo "FATAL: Bridge process died!"
    exit 1
fi

# Wait for mount
for i in 1 2 3 4 5; do
    mountpoint -q "$MOUNT" 2>/dev/null && break
    sleep 3
done

if ! mountpoint -q "$MOUNT" 2>/dev/null; then
    echo "FATAL: NTFS mount not available"
    kill $BRIDGE_PID 2>/dev/null || true
    exit 1
fi

echo "Bridge running."
echo ""
echo "NTFS mount contents:"
ls -la "$MOUNT/"
echo ""

echo "=== Phase 4: Hash files via NTFS bridge ==="
cd "$SOURCE"
rm -f "$HASHDIR/hashes_ntfs_bridge.txt"
HASH_ERRORS=0
find . -type f \( -name "*.mkv" -o -name "*.mp4" \) | sort | while read f; do
    ntfs_path="$MOUNT/${f#./}"
    bn=$(basename "$f")
    if [ -f "$ntfs_path" ]; then
        echo -n "  Hashing $bn ... "
        hash=$(md5sum "$ntfs_path" 2>&1)
        if [ $? -eq 0 ]; then
            echo "$hash" >> "$HASHDIR/hashes_ntfs_bridge.txt"
            echo "OK"
        else
            echo "ERROR: $hash" >> "$HASHDIR/hashes_ntfs_bridge.txt"
            echo "FAILED"
        fi
    else
        echo "  MISSING: $bn"
        echo "MISSING $ntfs_path" >> "$HASHDIR/hashes_ntfs_bridge.txt"
    fi
done

echo ""
echo "  $(wc -l < "$HASHDIR/hashes_ntfs_bridge.txt") NTFS bridge hashes"

echo ""
echo "=========================================="
echo "  THREE-WAY COMPARISON"
echo "=========================================="
echo ""

PASS=0
FAIL=0
MISSING=0
ERROR=0

while IFS= read -r line; do
    ext4_hash=$(echo "$line" | cut -d" " -f1)
    ext4_file=$(echo "$line" | sed 's/^[^ ]* *//')
    bn=$(basename "$ext4_file")

    orig_hash=$(grep -F "$bn" "$HASHDIR/hashes_original.txt" | head -1 | cut -d" " -f1)
    ntfs_line=$(grep -F "$bn" "$HASHDIR/hashes_ntfs_bridge.txt" | head -1)
    ntfs_hash=$(echo "$ntfs_line" | cut -d" " -f1)

    if echo "$ntfs_line" | grep -q "MISSING"; then
        echo "  MISSING: $bn"
        MISSING=$((MISSING + 1))
    elif echo "$ntfs_line" | grep -q "ERROR"; then
        echo "  ERROR:   $bn (I/O error reading via NTFS)"
        ERROR=$((ERROR + 1))
    elif [ "$orig_hash" = "$ext4_hash" ] && [ "$ext4_hash" = "$ntfs_hash" ]; then
        echo "  OK:      $bn"
        PASS=$((PASS + 1))
    else
        echo "  FAIL:    $bn"
        echo "           original: $orig_hash"
        echo "           ext4copy: $ext4_hash"
        echo "           ntfs:     $ntfs_hash"
        FAIL=$((FAIL + 1))
    fi
done < "$HASHDIR/hashes_ext4_copy.txt"

TOTAL=$((PASS + FAIL + MISSING + ERROR))
echo ""
echo "=========================================="
echo "  RESULTS: $PASS OK, $FAIL fail, $ERROR error, $MISSING missing (out of $TOTAL)"
echo "=========================================="

# Cleanup
echo ""
echo "=== Cleanup ==="
kill $BRIDGE_PID 2>/dev/null || true
sleep 2
umount "$MOUNT" 2>/dev/null || true
nbd-client -d /dev/nbd0 2>/dev/null || true

if [ $FAIL -gt 0 ] || [ $ERROR -gt 0 ] || [ $MISSING -gt 0 ]; then
    exit 1
fi
echo "All files match across all three sources!"
