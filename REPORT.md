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
- Fair Jetson A/B export: `artifacts/train4_320_fp16.engine`
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
- Added prerecorded benchmark input path in `src.app`:
  - `--video path/to/input.mp4`
  - same TensorRT + tracker runtime can now be replayed on saved Jetson sessions
- Added reproducible Jetson scripts:
  - `scripts/build_engine.sh`
  - `scripts/tegrastats_log.sh`
  - `scripts/bench.sh`
- Promoted `runs/detect/train4` to the current synthetic-augmented reference run in docs.
- Added fresh Jetson fast-path A/B evidence for `train4` vs baseline under `docs/fastpath_compare/`.
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
  - Reason: `train4` has already been exported and benchmarked on Jetson, but not yet through one fresh full `scripts/bench.sh` det_interval `{1,3,5}` matrix artifact set.
- AC4 (CSRT tracking-by-detection with reacquire): PASS
- AC5 (no dataset/engine/outputs committed): PASS
  - Datasets, synthetic outputs, sessions, and training weights remain ignored; only lightweight reference artifacts are intended for Git.

## Jetson runtime evidence

Source files:
- `outputs/realtime_test.json` (copied from Jetson)
- `outputs/trtexec_live.txt` (copied from Jetson)
- `outputs/tegrastats_live.log` (copied from Jetson)
- `docs/fastpath_compare/*.json`
- `docs/fastpath_compare/*.mp4`

Historical camera sample (`outputs/realtime_test.json`):
- frames: 30 / 30
- FPS: 20.185
- infer p50: 26.390 ms
- infer p95: 26.745 ms
- camera mode: 640x480 @ requested 30 FPS

Fresh prerecorded fast-path A/B (`src.app --video`, `sort`, `det=3`, `track=2`, `video-skip=2`):
- `session_0002_30s`
  - baseline: `32.959 FPS`, `infer p50 13.700 ms`, `det_ratio 0.9011`
  - `train4`: `34.985 FPS`, `infer p50 13.229 ms`, `det_ratio 0.9989`
- `session_0005_60s_30s`
  - baseline: `24.777 FPS`, `infer p50 13.745 ms`, `det_ratio 0.3289`
  - `train4`: `35.737 FPS`, `infer p50 13.265 ms`, `det_ratio 1.0000`

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
- Need one fresh Jetson benchmark matrix run via `scripts/bench.sh` to complete AC3 and report det_interval 1/3/5 in one consistent artifact set for `train4`.
- Need one live-camera `train4` demo capture to complement the prerecorded fast-path comparisons.
- `src/scripts/` still contains local raw media folders from earlier sessions (ignored by git, but noisy).
- `steps.log` and `gates.json` should be written under KB run folder for full protocol evidence compliance.

## Next
- Run the live-camera `train4` demo path on Jetson and record the final showcase clip.
- Run on Jetson:
  - `bash scripts/build_engine.sh artifacts/train4_320.onnx artifacts/train4_320_fp16.engine`
  - `bash scripts/bench.sh`
- Keep `README.md` and `REPORT.md` aligned with both live and prerecorded Jetson evidence.
