#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_PATH="${1:-$ROOT_DIR/outputs/tegrastats.log}"
INTERVAL_MS="${2:-1000}"

if ! command -v tegrastats >/dev/null 2>&1; then
  echo "ERROR: tegrastats command not found."
  exit 1
fi

mkdir -p "$(dirname "$OUT_PATH")"
echo "Logging tegrastats to: $OUT_PATH (interval=${INTERVAL_MS}ms)"
tegrastats --interval "$INTERVAL_MS" >"$OUT_PATH"
