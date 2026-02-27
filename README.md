# Jetson YOLO Realtime (Joystick)

Realtime joystick detection on Jetson Nano 4GB with a PS3 Eye camera.

## Project Pipeline

1. Train YOLOv8n on PC (`imgsz=320`).
2. Export ONNX (`artifacts/joystick.onnx`).
3. Build TensorRT FP16 engine on Jetson (`artifacts/joystick_fp16.engine`, ignored by git).
4. Run modular realtime app with detector-only or tracking-by-detection mode.

## Implemented Components

- Training
  - `src/train/train_yolo.py`
- Realtime runtime
  - `src/app.py`
  - `src/gst.py`
  - `src/trt_infer.py`
  - `src/postprocess.py`
  - `src/overlay.py`
  - `src/metrics.py`
  - `src/tracker_csrt.py`
- Jetson scripts
  - `scripts/build_engine.sh`
  - `scripts/tegrastats_log.sh`
  - `scripts/bench.sh`
  - `scripts/live_preview.sh`
  - `scripts/realtime_camera_test.py`
- Baseline training run artifacts
  - `runs/detect/joystick_320_neg_v4/`

## Training Snapshots

Training batch sample:

![Train Batch 0](runs/detect/joystick_320_neg_v4/train_batch0.jpg)

Training metrics summary:

![Training Results](runs/detect/joystick_320_neg_v4/results.png)

## Runtime Modes

Detector only:

```bash
python3 -m src.app --engine artifacts/joystick_fp16.engine --tracker off
```

Tracking-by-detection:

```bash
python3 -m src.app \
  --engine artifacts/joystick_fp16.engine \
  --tracker csrt --det-interval 3 --track-interval 2
```

Dry run (no camera/model startup):

```bash
python3 -m src.app --dry-run --tracker csrt --det-interval 3 --track-interval 2
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

Copy output to `artifacts/joystick.onnx`.

### 3) Build TensorRT engine on Jetson

```bash
bash scripts/build_engine.sh
```

### 4) Benchmark on Jetson (`det_interval=1,3,5`)

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

- `requirements.txt`: curated dependencies.
- `requirements-lock.txt`: exact local environment snapshot (`pip freeze`).

## Notes

- `data/`, `outputs/`, `*.engine`, and helper/dev artifacts are ignored.
- TensorRT engine binaries should be built on target Jetson, not committed.
- See `run.txt` for run commands.
