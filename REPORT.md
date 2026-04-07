# REPORT

## Goal
Make the repository portfolio-ready for Jetson Nano joystick realtime detection:
- modular runtime app
- tracking-by-detection support
- reproducible build/benchmark scripts
- documented runbook and metrics workflow

## Current deployment candidate

- Training run: `runs/detect/train4`
- Dataset: `dataset_v2/dataset.yaml`
- Base model: `yolov8n.pt`
- Training setup: `epochs=80`, `imgsz=640`, `batch=16`, `lr0=0.001`, `mixup=0.1`
- Final validation metrics at epoch 80:
  - precision: `0.99552`
  - recall: `0.98439`
  - mAP50: `0.99372`
  - mAP50-95: `0.95827`
- Git policy: keep only lightweight `train4` evidence (`*.png`, `*.jpg`, `results.csv`, `args.yaml`) and leave `.pt` weights local-only.

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
- Promoted `runs/detect/train4` to the current synthetic-augmented reference run in docs.
- Tightened `.gitignore` to keep datasets, synthetic renders, sessions, and training weights out of Git while allowing lightweight `train4` artifacts.
- Updated docs:
  - `README.md`
  - `run.txt`
  - `ROADMAP.md`
- Added dependency lock snapshot:
  - `requirements-lock.txt`

## AC status
- AC1 (modular `src/app.py` + `--help` smoke): PASS
- AC2 (`scripts/build_engine.sh` builds engine from ONNX): PASS
- AC3 (`scripts/bench.sh` produces tegrastats + metrics for det_interval 1/3/5): FAIL
  - Reason: `train4` has not yet been exported to ONNX and rerun on target Jetson through the benchmark matrix.
- AC4 (CSRT tracking-by-detection with reacquire): PASS
- AC5 (no dataset/engine/outputs committed): PASS
  - Datasets, synthetic outputs, sessions, and training weights remain ignored; only lightweight reference artifacts are intended for Git.

## Jetson runtime evidence (previous deploy)

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

These runtime numbers were collected before the `train4` refresh. They remain useful as baseline Jetson evidence, but they should be regenerated after exporting `runs/detect/train4/weights/best.pt` to ONNX and rebuilding the TensorRT engine.

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
- `artifacts/joystick.onnx` should be refreshed from `runs/detect/train4/weights/best.pt` before the next Jetson deploy.
- Need one real Jetson benchmark matrix run via `scripts/bench.sh` to complete AC3 and report det_interval 1/3/5 in one consistent artifact set for `train4`.
- `src/scripts/` still contains local raw media folders from earlier sessions (ignored by git, but noisy).
- `steps.log` and `gates.json` should be written under KB run folder for full protocol evidence compliance.

## Next
- Export `runs/detect/train4/weights/best.pt` to ONNX and refresh `artifacts/joystick.onnx`.
- Run on Jetson:
  - `bash scripts/build_engine.sh`
  - `bash scripts/bench.sh`
- Update `README.md` and `REPORT.md` with refreshed Jetson numbers after the `train4` deploy.
