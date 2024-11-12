#!/bin/bash

# Source environment variables
source .env

# Check if REMOTE_BUCKET is set
if [ -z "$REMOTE_BUCKET" ]; then
    echo "Error: REMOTE_BUCKET environment variable is not set"
    exit 1
fi

# Create remote directory if it doesn't exist
mkdir -p ./remote

# Configure rclone
cat > rclone.conf << EOF
[s3]
type = s3
provider = AWS
env_auth = false
access_key_id = $AWS_ACCESS_KEY_ID
secret_access_key = $AWS_SECRET_ACCESS_KEY
region = us-west-2
endpoint = s3.us-west-2.amazonaws.com
location_constraint = us-west-2
EOF

# Start rclone NFS server in the background
# Using port 12345 (you can change this), enabling full cache mode for write support
rclone serve nfs s3:$REMOTE_BUCKET --addr :12345 --vfs-cache-mode=full --config rclone.conf &
RCLONE_PID=$!

# Wait a moment for the server to start
sleep 2

# Mount the NFS share
mount -t nfs -o port=12345,mountport=12345,tcp localhost:/ ./remote

# Trap script exit to cleanup
cleanup() {
    echo "Cleaning up..."
    umount ./remote
    kill $RCLONE_PID
}
trap cleanup EXIT

echo "NFS mount is ready at ./remote"
echo "Press Ctrl+C to unmount and exit"

# Keep the script running
wait $RCLONE_PID

