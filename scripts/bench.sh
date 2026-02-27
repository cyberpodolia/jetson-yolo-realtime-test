#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENGINE_PATH="${ENGINE_PATH:-$ROOT_DIR/artifacts/joystick_fp16.engine}"
CAMERA="${CAMERA:-0}"
FRAMES="${FRAMES:-300}"
IMGSZ="${IMGSZ:-320}"
CONF="${CONF:-0.6}"
IOU="${IOU:-0.7}"
MAX_DET="${MAX_DET:-10}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
DETECT_INTERVALS="${DETECT_INTERVALS:-1 3 5}"
TEGR_INTERVAL_MS="${TEGR_INTERVAL_MS:-1000}"

OUT_DIR="$ROOT_DIR/outputs"
mkdir -p "$OUT_DIR"

if [[ ! -f "$ENGINE_PATH" ]]; then
  echo "ERROR: engine file not found: $ENGINE_PATH"
  echo "Run scripts/build_engine.sh first."
  exit 1
fi

if ! command -v tegrastats >/dev/null 2>&1; then
  echo "ERROR: tegrastats command not found."
  exit 1
fi

echo "Starting tegrastats logger..."
TEGRA_LOG="$OUT_DIR/tegrastats.log"
tegrastats --interval "$TEGR_INTERVAL_MS" >"$TEGRA_LOG" &
TEGR_PID=$!

cleanup() {
  if kill -0 "$TEGR_PID" >/dev/null 2>&1; then
    kill "$TEGR_PID" >/dev/null 2>&1 || true
    wait "$TEGR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Running benchmark matrix (det_interval in: $DETECT_INTERVALS)"
for det_n in $DETECT_INTERVALS; do
  out_json="$OUT_DIR/metrics_det${det_n}.json"
  echo "  det_interval=${det_n} -> $out_json"
  python3 -m src.app \
    --engine "$ENGINE_PATH" \
    --camera "$CAMERA" \
    --frames "$FRAMES" \
    --imgsz "$IMGSZ" \
    --conf "$CONF" \
    --iou "$IOU" \
    --max-det "$MAX_DET" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fps "$FPS" \
    --tracker csrt \
    --det-interval "$det_n" \
    --save-json "$out_json"
done

echo "Building aggregate metrics: $OUT_DIR/metrics.json"
python3 - <<'PY'
import glob
import json
from pathlib import Path

root = Path("outputs")
items = []
for p in sorted(glob.glob(str(root / "metrics_det*.json"))):
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    items.append(data)

summary = {
    "runs": items,
    "det_intervals": [r.get("det_interval") for r in items],
}
if items:
    summary["best_fps_run"] = max(items, key=lambda x: float(x.get("fps", 0.0)))

out = root / "metrics.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(out)
PY

echo "Done."
echo "  tegrastats: $TEGRA_LOG"
echo "  metrics:    $OUT_DIR/metrics.json"
