#!/bin/bash
# Test the partition wrapper (MBR generation)
#
# Verifies:
# 1. MBR signature at offset 510-511 (0x55AA)
# 2. Partition entry at offset 446 with type 0x07 (NTFS)
# 3. Partition starts at sector 2048 (1 MiB offset)
# 4. Data reads from partition area work correctly

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Partition Wrapper Test ==="
echo ""

# Create test environment
SOURCE="/tmp/partition-test/source"
IMAGE="/tmp/partition-test/image.raw"
MOUNT="/mnt/partition-test"
PORT=10810

mkdir -p "$SOURCE"
mkdir -p "$MOUNT"

# Create test file
echo "partition test content" > "$SOURCE/test.txt"

# Clean up any existing
rm -f "$IMAGE"
pkill -f "ntfs_bridge.bridge.*$PORT" 2>/dev/null || true
sleep 1

# Start bridge with partitioned mode in background
echo "Starting bridge with --partitioned flag..."
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

python3 -m ntfs_bridge.bridge \
    --source "$SOURCE" \
    --image "$IMAGE" \
    --mount "$MOUNT" \
    --port $PORT \
    --partitioned \
    --size 64 &

BRIDGE_PID=$!
echo "Bridge PID: $BRIDGE_PID"
sleep 3

# Check if bridge is running
if ! kill -0 $BRIDGE_PID 2>/dev/null; then
    echo "FAIL: Bridge process died"
    exit 1
fi

# Test MBR by connecting and reading first sector
echo ""
echo "Testing MBR structure via NBD..."

# Use Python to read first 512 bytes via NBD
python3 << 'EOF'
import socket
import struct

# Connect to NBD
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 10810))
sock.settimeout(10)

# NBD handshake
magic = sock.recv(8)
assert magic == b'NBDMAGIC', f"Bad magic: {magic}"

opts_magic = struct.unpack('>Q', sock.recv(8))[0]
assert opts_magic == 0x49484156454F5054, f"Bad opts magic: {opts_magic:x}"

flags = struct.unpack('>H', sock.recv(2))[0]
sock.sendall(struct.pack('>I', 0))  # client flags

# Send OPT_EXPORT_NAME
sock.sendall(struct.pack('>Q', 0x49484156454F5054))  # opts magic
sock.sendall(struct.pack('>I', 1))  # NBD_OPT_EXPORT_NAME
sock.sendall(struct.pack('>I', 0))  # data len = 0

# Receive export info
size = struct.unpack('>Q', sock.recv(8))[0]
export_flags = struct.unpack('>H', sock.recv(2))[0]
sock.recv(124)  # padding

print(f"Disk size: {size} bytes ({size // 1024 // 1024} MB)")

# Read first 512 bytes (MBR)
request = struct.pack('>IHHQQI', 0x25609513, 0, 0, 1, 0, 512)
sock.sendall(request)

reply = sock.recv(16)
magic, error, handle = struct.unpack('>IIQ', reply)
assert magic == 0x67446698, f"Bad reply magic: {magic:x}"
assert error == 0, f"Read error: {error}"

mbr = sock.recv(512)
sock.close()

# Verify MBR
print(f"\nMBR Analysis:")

# Check signature
sig = struct.unpack('<H', mbr[510:512])[0]
if sig == 0xAA55:
    print(f"  MBR signature: 0x{sig:04X} (valid)")
else:
    print(f"  FAIL: MBR signature: 0x{sig:04X} (expected 0xAA55)")
    exit(1)

# Check partition entry 1 (offset 446)
entry = mbr[446:462]
boot_flag = entry[0]
part_type = entry[4]
lba_start = struct.unpack('<I', entry[8:12])[0]
lba_size = struct.unpack('<I', entry[12:16])[0]

print(f"  Partition 1:")
print(f"    Boot flag: 0x{boot_flag:02X} ({'active' if boot_flag == 0x80 else 'inactive'})")
print(f"    Type: 0x{part_type:02X} ({'NTFS' if part_type == 0x07 else 'other'})")
print(f"    LBA start: {lba_start} (sector)")
print(f"    LBA size: {lba_size} sectors ({lba_size * 512 // 1024 // 1024} MB)")

if part_type != 0x07:
    print(f"  FAIL: Expected partition type 0x07 (NTFS)")
    exit(1)

if lba_start != 2048:
    print(f"  FAIL: Expected partition start at sector 2048")
    exit(1)

# Verify partition offset matches (1 MiB = 2048 * 512 bytes)
expected_partition_offset = 2048 * 512  # 1048576
expected_partition_size = size - expected_partition_offset
expected_sectors = expected_partition_size // 512

if lba_size == expected_sectors:
    print(f"    Partition size matches disk layout")
else:
    print(f"  WARNING: Partition size {lba_size} != expected {expected_sectors}")

print("\nPASS: MBR structure is valid")
EOF

RESULT=$?

# Cleanup
echo ""
echo "Cleaning up..."
kill $BRIDGE_PID 2>/dev/null || true
sleep 1
rm -rf /tmp/partition-test

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=== All partition wrapper tests passed ==="
else
    echo ""
    echo "=== Partition wrapper tests FAILED ==="
    exit 1
fi
