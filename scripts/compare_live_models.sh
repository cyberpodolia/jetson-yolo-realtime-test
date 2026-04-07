#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASELINE_ENGINE="${1:-$ROOT_DIR/artifacts/joystick_fp16.engine}"
NEW_ENGINE="${2:-$ROOT_DIR/artifacts/train4_fp16.engine}"
CAMERA="${CAMERA:-0}"
FRAMES="${FRAMES:-300}"
IMGSZ="${IMGSZ:-320}"
CONF="${CONF:-0.6}"
IOU="${IOU:-0.7}"
MAX_DET="${MAX_DET:-10}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
TRACKER="${TRACKER:-sort}"
DET_INTERVAL="${DET_INTERVAL:-3}"
TRACK_INTERVAL="${TRACK_INTERVAL:-2}"
VIDEO_SKIP="${VIDEO_SKIP:-2}"
VIDEO_CODEC="${VIDEO_CODEC:-mp4v}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/live_compare}"
TRTEXEC_BIN="${TRTEXEC_BIN:-/usr/src/tensorrt/bin/trtexec}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$BASELINE_ENGINE" ]]; then
  echo "ERROR: baseline engine not found: $BASELINE_ENGINE"
  exit 1
fi

if [[ ! -f "$NEW_ENGINE" ]]; then
  echo "ERROR: new engine not found: $NEW_ENGINE"
  exit 1
fi

run_model() {
  local label="$1"
  local engine_path="$2"
  local json_path="$OUT_DIR/${label}.json"
  local video_path="$OUT_DIR/${label}.mp4"
  local trtexec_path="$OUT_DIR/${label}_trtexec.txt"

  echo "=== ${label} ==="
  echo "engine=${engine_path}"

  if [[ -x "$TRTEXEC_BIN" ]]; then
    "$TRTEXEC_BIN" \
      --loadEngine="$engine_path" \
      --warmUp=1000 \
      --duration=10 \
      --streams=1 >"$trtexec_path" 2>&1 || true
  fi

  python3 -m src.app \
    --engine "$engine_path" \
    --camera "$CAMERA" \
    --frames "$FRAMES" \
    --imgsz "$IMGSZ" \
    --conf "$CONF" \
    --iou "$IOU" \
    --max-det "$MAX_DET" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fps "$FPS" \
    --tracker "$TRACKER" \
    --det-interval "$DET_INTERVAL" \
    --track-interval "$TRACK_INTERVAL" \
    --video-skip "$VIDEO_SKIP" \
    --video-codec "$VIDEO_CODEC" \
    --save-json "$json_path" \
    --save-video "$video_path"
}

run_model "baseline_live" "$BASELINE_ENGINE"
run_model "train4_live" "$NEW_ENGINE"

echo "Outputs saved in: $OUT_DIR"
