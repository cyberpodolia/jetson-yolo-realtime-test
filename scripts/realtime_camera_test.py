#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt


def default_model_path(project_root: Path) -> Path:
    engine = project_root / "artifacts" / "joystick_fp16.engine"
    if engine.exists():
        return engine
    raise FileNotFoundError("No model found. Expected artifacts/joystick_fp16.engine")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Realtime camera benchmark for Jetson YOLO model")
    parser.add_argument("--model", type=Path, default=default_model_path(project_root), help="Path to TensorRT .engine model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (/dev/videoX)")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to process")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.60, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.70, help="IoU threshold")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames before measuring")
    parser.add_argument("--max-det", type=int, default=10, help="Max detections per frame after NMS")
    parser.add_argument("--show", action="store_true", help="Show realtime preview window")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save benchmark JSON")
    parser.add_argument("--save-video", type=Path, default=None, help="Optional path to save raw camera MP4")
    return parser.parse_args()


def make_writer(path: Path, width: int, height: int, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1), (width, height))
    return writer if writer.isOpened() else None


def percentile(values, p):
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, int(len(values) * p) - 1))
    return sorted(values)[idx]


def load_engine(engine_path: Path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")
    return engine, context


def allocate_buffers(engine, context):
    stream = cuda.Stream()
    bindings = [None] * engine.num_bindings
    inputs = []
    outputs = []

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        shape = list(context.get_binding_shape(i))

        if any(dim < 0 for dim in shape):
            profile_shape = engine.get_profile_shape(0, i)
            shape = list(profile_shape[2])
            context.set_binding_shape(i, tuple(shape))

        size = int(np.prod(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        dev_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings[i] = int(dev_mem)

        binding = {
            "index": i,
            "name": name,
            "shape": tuple(shape),
            "host": host_mem,
            "device": dev_mem,
        }
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)

    if len(inputs) != 1:
        raise RuntimeError(f"Expected single input binding, got {len(inputs)}")
    return inputs[0], outputs, bindings, stream


def preprocess(frame: np.ndarray, in_h: int, in_w: int) -> np.ndarray:
    resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0)


def decode_detections(output: np.ndarray, in_w: int, in_h: int, frame_w: int, frame_h: int, conf_thres: float, max_det: int):
    pred = output
    if pred.ndim == 3:
        pred = pred[0]

    if pred.ndim != 2:
        return []

    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes_xywh = []
    scores = []

    sx = frame_w / float(in_w)
    sy = frame_h / float(in_h)

    for row in pred:
        if row.shape[0] < 5:
            continue

        cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        obj = float(row[4])

        if row.shape[0] > 5:
            cls_prob = float(np.max(row[5:]))
            score = obj * cls_prob
        else:
            score = obj

        if score < conf_thres:
            continue

        x = (cx - w / 2.0) * sx
        y = (cy - h / 2.0) * sy
        bw = w * sx
        bh = h * sy

        x = max(0.0, min(x, frame_w - 1.0))
        y = max(0.0, min(y, frame_h - 1.0))
        bw = max(1.0, min(bw, frame_w - x))
        bh = max(1.0, min(bh, frame_h - y))

        boxes_xywh.append([int(x), int(y), int(bw), int(bh)])
        scores.append(float(score))

    if not boxes_xywh:
        return []

    nms = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_thres, 0.5)
    if nms is None or len(nms) == 0:
        return []

    keep = []
    for idx in np.array(nms).reshape(-1)[:max_det]:
        x, y, bw, bh = boxes_xywh[int(idx)]
        keep.append((x, y, x + bw, y + bh, scores[int(idx)]))
    return keep


def draw_detections(frame: np.ndarray, detections):
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
        label = f"joystick {score:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 40), 2)


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if model_path.suffix.lower() != ".engine":
        raise ValueError("This script supports TensorRT .engine only")

    print("Starting realtime camera test")
    print(
        f"model={model_path} cam={args.camera} frames={args.frames} "
        f"imgsz={args.imgsz} conf={args.conf} iou={args.iou}"
    )

    engine, context = load_engine(model_path)
    input_binding, output_bindings, bindings, stream = allocate_buffers(engine, context)

    in_shape = input_binding["shape"]
    if len(in_shape) != 4:
        raise RuntimeError(f"Expected NCHW input shape, got {in_shape}")
    _, _, in_h, in_w = in_shape

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{args.camera}")

    writer = make_writer(args.save_video, args.width, args.height, args.fps) if args.save_video else None
    can_show = bool(args.show)
    if can_show:
        try:
            cv2.namedWindow("jetson-yolo", cv2.WINDOW_NORMAL)
        except cv2.error:
            can_show = False
            print("WARN: cannot open display window, continuing without preview")

    infer_ms = []
    ok_frames = 0
    det_frames = 0

    for _ in range(max(args.warmup, 0)):
        ok, frame = cap.read()
        if not ok:
            continue
        blob = preprocess(frame, in_h, in_w)
        np.copyto(input_binding["host"], blob.ravel())
        cuda.memcpy_htod_async(input_binding["device"], input_binding["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for out in output_bindings:
            cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
        stream.synchronize()

    t0 = time.time()
    while ok_frames < args.frames:
        ok, frame = cap.read()
        if not ok:
            continue

        t1 = time.time()
        blob = preprocess(frame, in_h, in_w)
        np.copyto(input_binding["host"], blob.ravel())

        infer_ms.append((time.time() - t1) * 1000.0)
        cuda.memcpy_htod_async(input_binding["device"], input_binding["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for out in output_bindings:
            cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
        stream.synchronize()
        infer_ms[-1] = (time.time() - t1) * 1000.0
        ok_frames += 1

        output0 = np.array(output_bindings[0]["host"], copy=False).reshape(output_bindings[0]["shape"])
        detections = decode_detections(
            output0,
            in_w=in_w,
            in_h=in_h,
            frame_w=frame.shape[1],
            frame_h=frame.shape[0],
            conf_thres=args.conf,
            max_det=args.max_det,
        )
        if detections:
            det_frames += 1

        vis = frame
        if can_show or writer is not None:
            vis = frame.copy()
            draw_detections(vis, detections)
            fps_now = ok_frames / max(time.time() - t0, 1e-6)
            cv2.putText(vis, f"FPS {fps_now:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

        if writer is not None:
            writer.write(vis)

        if can_show:
            cv2.imshow("jetson-yolo", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        if ok_frames % 30 == 0:
            elapsed = max(time.time() - t0, 1e-6)
            print(f"progress: {ok_frames}/{args.frames} fps={ok_frames / elapsed:.2f}")

    dt = max(time.time() - t0, 1e-6)
    cap.release()
    if can_show:
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    if ok_frames == 0:
        raise RuntimeError("Camera opened but no frames read")

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": str(model_path),
        "camera": args.camera,
        "frames": ok_frames,
        "requested_frames": args.frames,
        "fps": round(ok_frames / dt, 3),
        "total_sec": round(dt, 3),
        "infer_ms_mean": round(sum(infer_ms) / len(infer_ms), 3),
        "infer_ms_p50": round(statistics.median(infer_ms), 3),
        "infer_ms_p95": round(percentile(infer_ms, 0.95), 3),
        "det_frames": det_frames,
        "det_ratio": round(det_frames / ok_frames, 4),
        "input_shape": [int(x) for x in in_shape],
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "width": args.width,
        "height": args.height,
        "camera_fps_requested": args.fps,
    }

    print("--- realtime_camera_test ---")
    print(f"frames={report['frames']}/{report['requested_frames']} total_sec={report['total_sec']:.2f} fps={report['fps']:.2f}")
    print(
        "infer_ms "
        f"mean={report['infer_ms_mean']:.2f} "
        f"p50={report['infer_ms_p50']:.2f} "
        f"p95={report['infer_ms_p95']:.2f}"
    )
    print(f"input_shape={report['input_shape']}")
    print(f"detection_frames={report['det_frames']}/{report['frames']}")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report_saved={args.save_json}")

    if args.save_video:
        print(f"video_saved={args.save_video}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
