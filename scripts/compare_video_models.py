#!/usr/bin/env python3
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import RuntimeMetrics
from src.overlay import draw_detections, draw_fps, draw_status
from src.postprocess import decode_and_filter
from src.trt_infer import TrtConfig, TrtInferencer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two TensorRT engines on the same prerecorded video."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument("--baseline-engine", type=Path, required=True, help="Baseline TensorRT .engine.")
    parser.add_argument("--candidate-engine", type=Path, required=True, help="Candidate TensorRT .engine.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/video_compare"),
        help="Directory for rendered videos and JSON reports.",
    )
    parser.add_argument("--conf", type=float, default=0.60, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="IoU threshold.")
    parser.add_argument("--max-det", type=int, default=10, help="Max detections after NMS.")
    parser.add_argument("--overlay-alpha", type=float, default=0.20, help="Overlay fill opacity.")
    parser.add_argument("--video-codec", default="mp4v", help="4-char output codec.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video.")
    return parser.parse_args()


def resolve_path(raw_path: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def make_writer(path: Path, width: int, height: int, fps: float, codec: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, max(float(fps), 1.0), (width, height))
    return writer if writer.isOpened() else None


def process_video(label, engine_path, video_path, out_dir, args):
    inferencer = TrtInferencer(TrtConfig(engine_path=engine_path))
    inferencer.load()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{label}.mp4"
    out_json = out_dir / f"{label}.json"

    writer = make_writer(out_video, width, height, fps, args.video_codec)
    if writer is None:
        raise RuntimeError(f"Cannot open output writer: {out_video}")

    metrics = RuntimeMetrics()
    metrics.start()
    total_detections = 0
    best_score = 0.0

    frame_idx = 0
    while True:
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        t_read = time.perf_counter()
        ok, frame = cap.read()
        metrics.add_stage_ms("capture_read", (time.perf_counter() - t_read) * 1000.0)
        if not ok:
            break

        t_infer_total = time.perf_counter()
        raw_output, infer_ms = inferencer.infer(frame)
        metrics.add_stage_ms("infer_total", (time.perf_counter() - t_infer_total) * 1000.0)
        metrics.add_stage_ms("infer_reported", infer_ms)

        _, _, in_h, in_w = inferencer.input_shape
        t_decode = time.perf_counter()
        detections = decode_and_filter(
            raw_output=raw_output,
            in_w=in_w,
            in_h=in_h,
            frame_w=frame.shape[1],
            frame_h=frame.shape[0],
            conf_thres=args.conf,
            iou_thres=args.iou,
            max_det=args.max_det,
        )
        metrics.add_stage_ms("decode_nms", (time.perf_counter() - t_decode) * 1000.0)
        metrics.add_frame(infer_ms=infer_ms, has_detection=bool(detections))

        total_detections += len(detections)
        if detections:
            best_score = max(best_score, max(det.score for det in detections))

        vis = frame.copy()
        draw_detections(vis, detections, source="detect", alpha=args.overlay_alpha)
        draw_fps(vis, metrics.fps_now())
        draw_status(
            vis,
            f"{label} det_frames={metrics.det_frames}/{max(metrics.frames, 1)} dets={total_detections}",
        )
        writer.write(vis)
        frame_idx += 1

        if frame_idx % 60 == 0:
            snap = metrics.snapshot()
            print(
                f"{label}: frames={int(snap['frames'])} fps={snap['fps']:.2f} "
                f"infer_p50={snap['infer_ms_p50']:.2f}ms det_ratio={snap['det_ratio']:.3f}"
            )

    cap.release()
    writer.release()

    snap = metrics.snapshot()
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "label": label,
        "video": str(video_path),
        "model": str(engine_path),
        "frames": int(snap["frames"]),
        "fps": snap["fps"],
        "total_sec": snap["total_sec"],
        "infer_ms_mean": snap["infer_ms_mean"],
        "infer_ms_p50": snap["infer_ms_p50"],
        "infer_ms_p95": snap["infer_ms_p95"],
        "infer_samples": int(snap["infer_samples"]),
        "det_frames": int(snap["det_frames"]),
        "det_ratio": snap["det_ratio"],
        "total_detections": int(total_detections),
        "avg_detections_per_frame": round(total_detections / max(int(snap["frames"]), 1), 4),
        "best_score": round(best_score, 4),
        "input_shape": [int(x) for x in inferencer.input_shape],
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "stage_ms": snap.get("stage_ms", {}),
        "rendered_video": str(out_video),
    }
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{label}: report_saved={out_json}")
    print(f"{label}: video_saved={out_video}")
    return report


def main():
    args = parse_args()
    video_path = resolve_path(args.video)
    baseline_engine = resolve_path(args.baseline_engine)
    candidate_engine = resolve_path(args.candidate_engine)
    out_dir = resolve_path(args.out_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not baseline_engine.exists():
        raise FileNotFoundError(f"Baseline engine not found: {baseline_engine}")
    if not candidate_engine.exists():
        raise FileNotFoundError(f"Candidate engine not found: {candidate_engine}")

    baseline = process_video("baseline", baseline_engine, video_path, out_dir, args)
    candidate = process_video("candidate", candidate_engine, video_path, out_dir, args)

    comparison = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "video": str(video_path),
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "fps": round(candidate["fps"] - baseline["fps"], 3),
            "infer_ms_mean": round(candidate["infer_ms_mean"] - baseline["infer_ms_mean"], 3),
            "infer_ms_p50": round(candidate["infer_ms_p50"] - baseline["infer_ms_p50"], 3),
            "infer_ms_p95": round(candidate["infer_ms_p95"] - baseline["infer_ms_p95"], 3),
            "det_ratio": round(candidate["det_ratio"] - baseline["det_ratio"], 4),
            "total_detections": int(candidate["total_detections"] - baseline["total_detections"]),
            "best_score": round(candidate["best_score"] - baseline["best_score"], 4),
        },
    }
    comparison_path = out_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"comparison_saved={comparison_path}")


if __name__ == "__main__":
    raise SystemExit(main())
