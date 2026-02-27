# REPORT

## Goal
Make the repository portfolio-ready for Jetson Nano joystick realtime detection:
- modular runtime app
- tracking-by-detection support
- reproducible build/benchmark scripts
- documented runbook and metrics workflow

## What changed
- Implemented modular runtime package:
  - `src/app.py`
  - `src/gst.py`
  - `src/trt_infer.py`
  - `src/postprocess.py`
  - `src/overlay.py`
  - `src/metrics.py`
  - `src/tracker_csrt.py`
  - `src/config.py`
- Added tracking-by-detection path in `src.app`:
  - `--tracker csrt|off`
  - `--det-interval N`
  - detector re-acquire when tracker update fails
- Added reproducible Jetson scripts:
  - `scripts/build_engine.sh`
  - `scripts/tegrastats_log.sh`
  - `scripts/bench.sh`
- Updated docs:
  - `README.md`
  - `run.txt`
- Added dependency lock snapshot:
  - `requirements-lock.txt`

## AC status
- AC1 (modular `src/app.py` + `--help` smoke): PASS
- AC2 (`scripts/build_engine.sh` builds engine from ONNX): PASS
- AC3 (`scripts/bench.sh` produces tegrastats + metrics for det_interval 1/3/5): FAIL
  - Reason: script implemented but full matrix was not executed on target Jetson from this run.
- AC4 (CSRT tracking-by-detection with reacquire): PASS
- AC5 (no dataset/engine/outputs committed): PASS

## Runtime metrics (current evidence)

Source files:
- `outputs/realtime_test.json` (copied from Jetson)
- `outputs/trtexec_live.txt` (copied from Jetson)
- `outputs/tegrastats_live.log` (copied from Jetson)

Realtime camera sample (`outputs/realtime_test.json`):
- frames: 30 / 30
- FPS: 20.185
- infer p50: 26.390 ms
- infer p95: 26.745 ms
- camera mode: 640x480 @ requested 30 FPS

Engine-only sample (`outputs/trtexec_live.txt`):
- Throughput: 57.57 qps
- Latency mean: 17.06 ms
- Latency p99: 34.65 ms

Tegrastats summary (`outputs/tegrastats_live.log`, parsed):
- samples: 11853
- RAM used min/max/avg: 1452 / 3616 / 1586 MB (total 3956 MB)
- GR3D freq min/max/avg: 0 / 99 / 7.8 %
- CPU average load min/max/mean: 0.5 / 74.8 / 7.5 %

## Gates
- format: PASS (`python -m ruff format src`)
- lint: PASS (`python -m ruff check src`)
- unit: FAIL (no unit tests executed in this run)
- smoke: PASS
  - `python -m compileall src`
  - `python -c "import src; print('import_ok')"`
  - `python -m src.app --help`
  - `python -m src.app --dry-run --tracker csrt --det-interval 3`
- review: CHANGES_REQUIRED (formal reviewer stage not yet executed)

## Evidence
- steps.log: not persisted in KB run folder from this terminal run
- gates.json: not persisted in KB run folder from this terminal run
- CI URL (optional): n/a

## Risks / debt
- Need one real Jetson benchmark matrix run via `scripts/bench.sh` to complete AC3 and report det_interval 1/3/5 in one consistent artifact set.
- `src/scripts/` still contains local raw media folders from earlier sessions (ignored by git, but noisy).
- `steps.log` and `gates.json` should be written under KB run folder for full protocol evidence compliance.

## Next
- Run on Jetson:
  - `bash scripts/build_engine.sh`
  - `bash scripts/bench.sh`
- Update `REPORT.md` AC3 from FAIL to PASS with `outputs/metrics.json` values for det_interval `{1,3,5}`.
