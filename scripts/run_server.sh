#!/bin/bash
# Run the NTFS-ext4 bridge NBD server

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
SOURCE_DIR="${1:-/srv/ntfs-bridge-test}"
SOCKET_PATH="${2:-/var/run/ntfs-bridge.sock}"
SOCKET_GROUP="${3:-kvm}"

echo "NTFS-ext4 Bridge NBD Server"
echo "==========================="
echo "Source directory: $SOURCE_DIR"
echo "Socket path: $SOCKET_PATH"
echo "Socket group: $SOCKET_GROUP"
echo ""

# Check source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

# Run the server
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "Starting NBD server..."
python3 -m ntfs_bridge.nbd_server "$SOURCE_DIR" -s "$SOCKET_PATH"
