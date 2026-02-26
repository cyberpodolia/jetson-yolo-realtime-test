#!/usr/bin/env bash
set -euo pipefail

HOST_PROJECT="${HOST_PROJECT:-/home/kolins/jetson-yolo}"
CONTAINER_IMG="${CONTAINER_IMG:-nvcr.io/nvidia/l4t-ml:r32.7.1-py3}"

CAMERA="0"
FRAMES="${1:-999999}"
IMGSZ="${2:-320}"
CONF="${3:-0.6}"
IOU="${4:-0.7}"
MAX_DET="${5:-10}"

if [[ ! -d "$HOST_PROJECT" ]]; then
  echo "ERROR: project dir not found: $HOST_PROJECT"
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]]; then
  export DISPLAY=:0
fi

if docker info > /dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

xhost +local:root >/dev/null 2>&1 || true

"${DOCKER[@]}" run --rm --runtime nvidia --network host --ipc host \
  --device=/dev/video0 \
  --group-add video \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOST_PROJECT:/workspace" \
  -w /workspace \
  "$CONTAINER_IMG" \
  python3 /workspace/scripts/realtime_camera_test.py \
  --camera "$CAMERA" \
  --frames "$FRAMES" \
  --imgsz "$IMGSZ" \
  --conf "$CONF" \
  --iou "$IOU" \
  --max-det "$MAX_DET" \
  --show
