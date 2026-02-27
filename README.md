# Jetson YOLO Realtime (Joystick)

Realtime joystick detection for Jetson Nano 4GB + PS3 Eye camera.

Pipeline:
- YOLOv8n training on PC (`imgsz=320`)
- ONNX export (`artifacts/joystick.onnx`)
- TensorRT FP16 engine build on Jetson (`artifacts/joystick_fp16.engine`, git-ignored)
- Modular runtime app with detector + optional CSRT tracking-by-detection

## Implemented

- Training:
  - `src/train/train_yolo.py`
- Modular runtime:
  - `src/app.py`
  - `src/gst.py`
  - `src/trt_infer.py`
  - `src/postprocess.py`
  - `src/overlay.py`
  - `src/metrics.py`
  - `src/tracker_csrt.py`
- Jetson scripts:
  - `scripts/build_engine.sh`
  - `scripts/tegrastats_log.sh`
  - `scripts/bench.sh`
  - `scripts/live_preview.sh`
  - `scripts/realtime_camera_test.py`
- Baseline train artifacts:
  - `runs/detect/joystick_320_neg_v4/`

## Runtime Modes

Detector only:
```bash
python -m src.app --engine artifacts/joystick_fp16.engine --tracker off
```

Tracking-by-detection (CSRT):
```bash
python -m src.app --engine artifacts/joystick_fp16.engine --tracker csrt --det-interval 3
```

Dry run (no camera/model load):
```bash
python -m src.app --dry-run --tracker csrt --det-interval 5
```

## Quick Start

### 1) Train on PC

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/train/train_yolo.py \
  --data data/dataset.yaml \
  --model yolov8n.pt \
  --imgsz 320 \
  --epochs 150 \
  --batch 32 \
  --device 0 \
  --name joystick_320
```

### 2) Export ONNX

```bash
yolo export model=runs/detect/joystick_320_neg_v4/weights/best.pt \
  format=onnx opset=12 imgsz=320 simplify=True dynamic=False
```

Copy result to:
- `artifacts/joystick.onnx`

### 3) Build TensorRT engine on Jetson

```bash
bash scripts/build_engine.sh
```

### 4) Run benchmark matrix on Jetson (`det_interval=1,3,5`)

```bash
bash scripts/bench.sh
```

Expected outputs:
- `outputs/tegrastats.log`
- `outputs/metrics_det1.json`
- `outputs/metrics_det3.json`
- `outputs/metrics_det5.json`
- `outputs/metrics.json`

## Dependency Files

- `requirements.txt`: top-level, human-maintained dependencies.
- `requirements-lock.txt`: exact local `.venv` snapshot (`pip freeze`), useful for strict reproducibility.

## Notes

- `data/`, `outputs/`, `*.engine`, and helper/dev artifacts are git-ignored.
- Do not commit TensorRT engine binaries; build them on target Jetson.
- See `run.txt` for command runbook and `REPORT.md` for current measured status.
