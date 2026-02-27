#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_PATH="${1:-$ROOT_DIR/artifacts/joystick.onnx}"
ENGINE_PATH="${2:-$ROOT_DIR/artifacts/joystick_fp16.engine}"
WORKSPACE_MB="${WORKSPACE_MB:-1024}"

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "ERROR: ONNX model not found: $ONNX_PATH"
  exit 1
fi

mkdir -p "$(dirname "$ENGINE_PATH")"

TRTEXEC_BIN="${TRTEXEC_BIN:-/usr/src/tensorrt/bin/trtexec}"
if [[ ! -x "$TRTEXEC_BIN" ]]; then
  echo "ERROR: trtexec not found or not executable: $TRTEXEC_BIN"
  echo "Set TRTEXEC_BIN env var if trtexec path is different."
  exit 1
fi

echo "Building TensorRT engine..."
echo "  onnx:   $ONNX_PATH"
echo "  engine: $ENGINE_PATH"
echo "  fp16:   enabled"
echo "  ws_mb:  $WORKSPACE_MB"

"$TRTEXEC_BIN" \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --fp16 \
  --workspace="$WORKSPACE_MB" \
  --minTiming=1 \
  --avgTiming=1

echo "Done: $ENGINE_PATH"
