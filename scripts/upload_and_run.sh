#!/bin/bash
# Upload project to server and run tests

set -e

SERVER="root@192.168.1.12"
REMOTE_DIR="/opt/ntfs-ext4-bridge"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Uploading project to $SERVER:$REMOTE_DIR..."

# Create remote directory
ssh $SERVER "mkdir -p $REMOTE_DIR"

# Upload files
scp -r "$LOCAL_DIR/ntfs_bridge" $SERVER:$REMOTE_DIR/
scp -r "$LOCAL_DIR/tests" $SERVER:$REMOTE_DIR/
scp -r "$LOCAL_DIR/scripts" $SERVER:$REMOTE_DIR/

echo "Running tests on server..."
ssh $SERVER "cd $REMOTE_DIR && python3 tests/test_synthesizer.py"

echo "Done!"
