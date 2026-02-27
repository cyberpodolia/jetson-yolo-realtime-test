# Jetson YOLO Realtime (Joystick)

Real-time joystick detection on Jetson Nano 4GB (PS3 Eye camera), with TensorRT FP16 and tracking-by-detection.

## Highlights

- End-to-end deployment: YOLOv8 -> ONNX -> TensorRT engine on Jetson.
- Realtime runtime with:
  - detector-only mode (`--tracker off`)
  - tracking-by-detection (`--tracker csrt` or host-safe `--tracker sort`)
- Runtime tuning knobs:
  - `--det-interval`
  - `--track-interval`
  - `--preview-skip`
  - `--video-skip`

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
  --tracker sort --det-interval 3 --track-interval 2 \
  --overlay-alpha 0.15 --preview-skip 2 \
  --video-skip 2 --video-codec mp4v
```

## Screenshots

Train batch sample:

![Train Batch 0](runs/detect/joystick_320_neg_v4/train_batch0.jpg)

Training metrics curve:

![Training Results](runs/detect/joystick_320_neg_v4/results.png)

## Latest Benchmarks (Jetson Host)

Source files are in `outputs/` (local artifacts copied from Jetson):
- `prof_off_noviz.json`
- `prof_sort_noviz.json`
- `perf_vs1_mp4v.json`
- `perf_vs2_mp4v.json`

| Mode | FPS | Infer p50 | Notes |
|---|---:|---:|---|
| `off`, no video | 22.04 | 20.58 ms | detector every frame |
| `sort`, no video | 22.62 | 20.51 ms | `det=3`, `track=2` |
| `sort`, video `mp4v`, `video-skip=1` | 16.30 | 15.79 ms | writer becomes bottleneck |
| `sort`, video `mp4v`, `video-skip=2` | 28.73 | 15.73 ms | highest throughput in tests |

Main bottleneck during recording: video writer (`stage_ms.writer`).

## Demo Outputs

Recorded runtime videos (local, not committed):
- `outputs/host_sort.mp4`
- `outputs/host_sort2.mp4`
- `outputs/prof_sort_video.mp4`
- `outputs/perf_vs1_mp4v.avi`
- `outputs/perf_vs2_mp4v.avi`

Metrics JSON for montage overlays:
- `outputs/metrics_host_sort.json`
- `outputs/perf_vs1_mp4v.json`
- `outputs/perf_vs2_mp4v.json`

## Minimal Run Commands

Build TensorRT engine on Jetson:

```bash
bash scripts/build_engine.sh
```

Detector only:

```bash
python3 -m src.app --engine artifacts/joystick_fp16.engine --tracker off
```

Tracking-by-detection (host-safe):

```bash
python3 -m src.app \
  --engine artifacts/joystick_fp16.engine \
  --tracker sort --det-interval 3 --track-interval 2 \
  --preview-skip 2 --video-skip 2 --video-codec mp4v
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
- `CSRT` may be unavailable in host OpenCV builds; runtime falls back to `sort`.
