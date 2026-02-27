# Jetson YOLO Realtime (Joystick)

Real-time joystick detection on Jetson Nano 4GB (PS3 Eye camera), with TensorRT FP16 and tracking-by-detection.

## What This Project Shows

- End-to-end deployment pipeline: YOLOv8 -> ONNX -> TensorRT engine on Jetson.
- Realtime app with two modes:
  - detector only (`--tracker off`)
  - tracking-by-detection (`--tracker csrt`)
- Practical runtime controls for FPS stability:
  - `--det-interval` (how often to run detector)
  - `--track-interval` (how often to update tracker)
  - `--preview-skip` (how often to render GUI frame)

## Tracking-by-Detection

Runtime logic:

1. Run detector every `N` frames (`--det-interval`).
2. Between detector frames, run CSRT tracker.
3. If tracker update fails, force reacquire with detector.
4. If no joystick is detected, overlay status switches to red: `NO_JOYSTICK`.

Recommended realtime preset:

```bash
python3 -m src.app \
  --engine artifacts/joystick_fp16.engine \
  --tracker csrt --det-interval 3 --track-interval 2 \
  --overlay-alpha 0.15 --preview-skip 2
```

## Training Screenshots

Train batch sample:

![Train Batch 0](runs/detect/joystick_320_neg_v4/train_batch0.jpg)

Training metrics curve:

![Training Results](runs/detect/joystick_320_neg_v4/results.png)

## Metrics Snapshot

Current measured snapshot from `REPORT.md`:

| Metric | Value |
|---|---|
| Realtime FPS (sample) | 20.185 |
| Inference p50 | 26.390 ms |
| Inference p95 | 26.745 ms |
| `trtexec` throughput | 57.57 qps |
| `trtexec` latency mean | 17.06 ms |
| `trtexec` latency p99 | 34.65 ms |
| RAM used (min/max/avg) | 1452 / 3616 / 1586 MB |
| CPU average load mean | 7.5% |
| GR3D average | 7.8% |

Full details: `REPORT.md` and `run.txt`.

## Minimal Run Commands

Build TensorRT engine on Jetson:

```bash
bash scripts/build_engine.sh
```

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

Benchmark matrix (`det_interval=1,3,5`):

```bash
bash scripts/bench.sh
```

Expected outputs:

- `outputs/tegrastats.log`
- `outputs/metrics_det1.json`
- `outputs/metrics_det3.json`
- `outputs/metrics_det5.json`
- `outputs/metrics.json`

## Notes

- `data/`, `outputs/`, `*.engine`, and helper/dev artifacts are ignored.
- TensorRT `.engine` files are built on target Jetson and are not committed.
