#!/bin/bash
# Pull generated videos from RunPod to local machine
# Usage: ./pull-outputs.sh <IP> <PORT>

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./pull-outputs.sh <IP> <PORT>"
    echo "Example: ./pull-outputs.sh 203.57.40.79 10267"
    exit 1
fi

IP=$1
PORT=$2
OUTPUT_DIR=~/Code/MAGIMA-CLOUD-AI/outputs/$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

echo "Pulling videos from RunPod to $OUTPUT_DIR..."
scp -P "$PORT" -o StrictHostKeyChecking=no "root@$IP:/workspace/LTX-Video/*.mp4" "$OUTPUT_DIR/" 2>/dev/null || true
scp -P "$PORT" -o StrictHostKeyChecking=no "root@$IP:/workspace/*.mp4" "$OUTPUT_DIR/" 2>/dev/null || true

echo "Done! Videos saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
