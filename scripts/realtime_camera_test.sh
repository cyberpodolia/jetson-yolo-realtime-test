#!/usr/bin/env bash
set -euo pipefail

HOST_PROJECT="${HOST_PROJECT:-/home/kolins/jetson-yolo}"
CONTAINER_IMG="${CONTAINER_IMG:-nvcr.io/nvidia/l4t-ml:r32.7.1-py3}"

CAM_INDEX="${1:-0}"
FRAMES="${2:-300}"
IMGSZ="${3:-320}"
CONF="${4:-0.60}"
IOU="${5:-0.70}"
MODEL_PATH="${6:-/workspace/artifacts/joystick_fp16.engine}"

if [[ ! -d "$HOST_PROJECT" ]]; then
  echo "ERROR: HOST_PROJECT not found: $HOST_PROJECT"
  exit 1
fi

if [[ "$MODEL_PATH" == "/workspace/artifacts/joystick_fp16.engine" && ! -f "$HOST_PROJECT/artifacts/joystick_fp16.engine" ]]; then
  if [[ -f "$HOST_PROJECT/artifacts/joystick.onnx" ]]; then
    MODEL_PATH="/workspace/artifacts/joystick.onnx"
    echo "INFO: FP16 engine not found, fallback to ONNX: $MODEL_PATH"
  else
    echo "ERROR: No model found in $HOST_PROJECT/artifacts"
    exit 1
  fi
fi

run_docker() {
  if docker info > /dev/null 2>&1; then
    docker "$@"
  else
    sudo docker "$@"
  fi
}

mkdir -p "$HOST_PROJECT/outputs"

echo "Starting realtime camera test"
echo "cam=$CAM_INDEX frames=$FRAMES imgsz=$IMGSZ conf=$CONF iou=$IOU model=$MODEL_PATH"

run_docker run --rm   --runtime nvidia   --network host   --ipc host   --device=/dev/video0   --device=/dev/video1   --group-add video   -v "$HOST_PROJECT:/workspace"   -w /workspace   -e CAM_INDEX="$CAM_INDEX"   -e FRAMES="$FRAMES"   -e IMGSZ="$IMGSZ"   -e CONF="$CONF"   -e IOU="$IOU"   -e MODEL_PATH="$MODEL_PATH"   "$CONTAINER_IMG"   bash -lc 'python3 - << "PY2"
import os
import time
import statistics
import cv2
from ultralytics import YOLO

cam = int(os.environ["CAM_INDEX"])
n = int(os.environ["FRAMES"])
imgsz = int(os.environ["IMGSZ"])
conf = float(os.environ["CONF"])
iou = float(os.environ["IOU"])
model_path = os.environ["MODEL_PATH"]

model = YOLO(model_path, task="detect")
cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open /dev/video{cam}")

ok_frames = 0
infer_ms = []
det_frames = 0

t0 = time.time()
for _ in range(n):
    ok, frame = cap.read()
    if not ok:
        continue
    ok_frames += 1

    t1 = time.time()
    res = model.predict(source=frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    infer_ms.append((time.time() - t1) * 1000.0)

    if res and len(res[0].boxes) > 0:
        det_frames += 1

dt = max(time.time() - t0, 1e-6)
cap.release()

if ok_frames == 0:
    raise RuntimeError("Camera opened but no frames read")

fps = ok_frames / dt
p50 = statistics.median(infer_ms)
p95 = sorted(infer_ms)[max(0, int(len(infer_ms) * 0.95) - 1)]
mean = sum(infer_ms) / len(infer_ms)

print("--- realtime_camera_test ---")
print(f"model={model_path}")
print(f"frames={ok_frames}/{n} total_sec={dt:.2f} fps={fps:.2f}")
print(f"infer_ms mean={mean:.2f} p50={p50:.2f} p95={p95:.2f}")
print(f"detection_frames={det_frames}/{ok_frames}")
PY2'
