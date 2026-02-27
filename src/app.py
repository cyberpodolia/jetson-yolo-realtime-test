import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from .config import DEFAULTS


def resolve_path(repo_root, raw_path):
    """Resolve relative paths against repo root."""
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def parse_args():
    """Parse CLI arguments for modular runtime."""
    parser = argparse.ArgumentParser(description="Jetson YOLO modular runtime app.")
    parser.add_argument(
        "--engine",
        "--model",
        dest="engine",
        default=DEFAULTS.engine,
        help="Path to TensorRT .engine model.",
    )
    parser.add_argument("--camera", type=int, default=DEFAULTS.camera, help="Camera index (/dev/videoX).")
    parser.add_argument("--frames", type=int, default=DEFAULTS.frames, help="Number of frames to process.")
    parser.add_argument("--imgsz", type=int, default=DEFAULTS.imgsz, help="Inference image size target.")
    parser.add_argument("--conf", type=float, default=DEFAULTS.conf, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=DEFAULTS.iou, help="IoU threshold for NMS.")
    parser.add_argument("--width", type=int, default=DEFAULTS.width, help="Camera width.")
    parser.add_argument("--height", type=int, default=DEFAULTS.height, help="Camera height.")
    parser.add_argument("--fps", type=int, default=DEFAULTS.fps, help="Requested camera FPS.")
    parser.add_argument("--warmup", type=int, default=DEFAULTS.warmup, help="Warmup frames before measuring.")
    parser.add_argument("--max-det", type=int, default=DEFAULTS.max_det, help="Max detections after NMS.")
    parser.add_argument("--show", action="store_true", help="Show realtime preview window.")
    parser.add_argument("--use-gst", action="store_true", help="Try GStreamer capture path.")
    parser.add_argument("--gst", default="", help="Custom GStreamer pipeline string.")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional metrics JSON path.")
    parser.add_argument("--save-video", type=Path, default=None, help="Optional MP4 output path.")
    parser.add_argument(
        "--video-codec",
        default=DEFAULTS.video_codec,
        help="4-char video codec for OpenCV writer (e.g. mp4v, MJPG, XVID).",
    )
    parser.add_argument(
        "--video-skip",
        type=int,
        default=DEFAULTS.video_skip,
        help="Write every Nth frame to output video to reduce encoder overhead.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.2,
        help="BBox fill opacity in preview/video (0.0 disables fill, 1.0 solid).",
    )
    parser.add_argument(
        "--preview-skip",
        type=int,
        default=1,
        help="Render/show every Nth frame to reduce GUI overhead.",
    )
    parser.add_argument(
        "--det-interval",
        type=int,
        default=DEFAULTS.det_interval,
        help="Detect every N frames (tracking-by-detection stage follows in step 00030).",
    )
    parser.add_argument(
        "--track-interval",
        type=int,
        default=DEFAULTS.track_interval,
        help="Run tracker update every N frames; intermediate frames reuse last tracked bbox.",
    )
    parser.add_argument(
        "--tracker",
        choices=("off", "csrt", "sort"),
        default=DEFAULTS.tracker,
        help="Tracker mode: off, csrt, or sort.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and paths only, without camera/model startup.",
    )
    return parser.parse_args()


def _normalize_codec(raw_codec):
    codec = str(raw_codec).strip()
    if len(codec) != 4:
        raise ValueError("--video-codec must be exactly 4 characters")
    return codec


def _make_writer(cv2_module, path, width, height, fps, codec):
    """Create optional MP4 writer."""
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2_module.VideoWriter_fourcc(*codec)
    writer = cv2_module.VideoWriter(str(path), fourcc, max(fps, 1), (width, height))
    return writer if writer.isOpened() else None


def main():
    """Run realtime pipeline and print summary metrics."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    engine_path = resolve_path(repo_root, args.engine)
    save_json_path = resolve_path(repo_root, args.save_json) if args.save_json else None
    save_video_path = resolve_path(repo_root, args.save_video) if args.save_video else None

    if args.det_interval < 1:
        raise ValueError("--det-interval must be >= 1")
    if args.track_interval < 1:
        raise ValueError("--track-interval must be >= 1")
    if args.video_skip < 1:
        raise ValueError("--video-skip must be >= 1")
    if args.preview_skip < 1:
        raise ValueError("--preview-skip must be >= 1")
    if args.overlay_alpha < 0.0 or args.overlay_alpha > 1.0:
        raise ValueError("--overlay-alpha must be within [0.0, 1.0]")
    video_codec = _normalize_codec(args.video_codec)

    if args.dry_run:
        print(
            "dry_run_ok "
            "engine={} camera={} imgsz={} det_interval={} track_interval={} tracker={} overlay_alpha={} preview_skip={} video_skip={} video_codec={}".format(
                engine_path,
                args.camera,
                args.imgsz,
                args.det_interval,
                args.track_interval,
                args.tracker,
                args.overlay_alpha,
                args.preview_skip,
                args.video_skip,
                video_codec,
            )
        )
        return 0

    if not engine_path.exists():
        raise FileNotFoundError("TensorRT engine not found: {}".format(engine_path))
    if engine_path.suffix.lower() != ".engine":
        raise ValueError("Runtime expects a TensorRT .engine file")

    import cv2

    from .gst import open_camera
    from .metrics import RuntimeMetrics
    from .overlay import draw_detections, draw_fps, draw_status
    from .postprocess import decode_and_filter
    from .tracker_csrt import CsrtTracker, csrt_available
    from .tracker_sort import SortTracker, sort_available
    from .trt_infer import TrtConfig, TrtInferencer

    inferencer = TrtInferencer(TrtConfig(engine_path=engine_path, input_size=args.imgsz))
    inferencer.load()
    in_shape = inferencer.input_shape
    if len(in_shape) != 4:
        raise RuntimeError("Expected NCHW input shape, got: {}".format(in_shape))
    _, _, in_h, in_w = in_shape

    cap = open_camera(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_gstreamer=args.use_gst or bool(args.gst),
        gst_pipeline=args.gst or None,
    )
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera index {}".format(args.camera))

    writer = _make_writer(cv2, save_video_path, args.width, args.height, args.fps, video_codec)
    if save_video_path is not None and writer is None:
        print(
            "WARN: cannot open video writer with codec '{}'; disabling --save-video.".format(
                video_codec
            )
        )
    can_show = bool(args.show)
    if can_show:
        try:
            cv2.namedWindow("jetson-yolo", cv2.WINDOW_NORMAL)
        except cv2.error:
            can_show = False
            print("WARN: cannot open display window, continuing without preview.")

    for _ in range(max(args.warmup, 0)):
        ok, frame = cap.read()
        if not ok:
            continue
        inferencer.infer(frame)

    tracker_mode = args.tracker
    tracker = None
    if tracker_mode == "csrt":
        if csrt_available():
            tracker = CsrtTracker()
        elif sort_available():
            tracker_mode = "sort"
            tracker = SortTracker()
            print(
                "WARN: CSRT tracker is not available in this OpenCV build; fallback to --tracker sort."
            )
        else:
            tracker_mode = "off"
            print(
                "WARN: no compatible tracker backend available; fallback to --tracker off."
            )
    elif tracker_mode == "sort":
        if sort_available():
            tracker = SortTracker()
        else:
            tracker_mode = "off"
            print("WARN: SORT tracker backend is unavailable; fallback to --tracker off.")

    if tracker_mode == "off" and args.det_interval != 1:
        print(
            "WARN: --det-interval is active only with tracker enabled; running detector each frame."
        )
    if tracker_mode == "off" and args.track_interval != 1:
        print("WARN: --track-interval is active only with tracker enabled.")

    metrics = RuntimeMetrics()
    metrics.start()
    frame_id = 0
    last_track_detection = None

    while metrics.frames < args.frames:
        t_cap = time.perf_counter()
        ok, frame = cap.read()
        metrics.add_stage_ms("capture_read", (time.perf_counter() - t_cap) * 1000.0)
        if not ok:
            continue
        frame_id += 1

        detections = []
        infer_ms = None
        source = "detect"

        need_detect = True
        if tracker_mode in ("csrt", "sort") and tracker is not None:
            need_detect = (not tracker.is_active) or (frame_id % args.det_interval == 0)
            if not need_detect and tracker.is_active:
                run_track_update = (frame_id % args.track_interval == 0) or (
                    last_track_detection is None
                )
                if run_track_update:
                    t_track = time.perf_counter()
                    tracked = tracker.update(frame)
                    metrics.add_stage_ms(
                        "track_update", (time.perf_counter() - t_track) * 1000.0
                    )
                    if tracked.ok and tracked.detection is not None:
                        last_track_detection = tracked.detection
                        detections = [tracked.detection]
                        source = "track"
                        need_detect = False
                    else:
                        last_track_detection = None
                        need_detect = True
                else:
                    detections = [last_track_detection]
                    source = "track_hold"
                    need_detect = False

        if need_detect:
            t_infer_total = time.perf_counter()
            raw_output, infer_ms = inferencer.infer(frame)
            metrics.add_stage_ms(
                "infer_total", (time.perf_counter() - t_infer_total) * 1000.0
            )
            if infer_ms is not None:
                metrics.add_stage_ms("infer_reported", infer_ms)
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
            source = "detect"

            if tracker_mode in ("csrt", "sort") and tracker is not None:
                if detections:
                    best_det = max(detections, key=lambda d: d.score)
                    t_track_init = time.perf_counter()
                    tracker.init_from_detection(frame, best_det)
                    metrics.add_stage_ms(
                        "track_init", (time.perf_counter() - t_track_init) * 1000.0
                    )
                    last_track_detection = best_det
                elif not tracker.is_active:
                    tracker.reset()
                    last_track_detection = None

        metrics.add_frame(infer_ms=infer_ms, has_detection=bool(detections))

        vis = frame
        write_now = writer is not None and (frame_id % args.video_skip == 0)
        render_now = write_now or (can_show and (frame_id % args.preview_skip == 0))
        if render_now:
            t_render = time.perf_counter()
            vis = frame.copy()
            draw_detections(vis, detections, source=source, alpha=args.overlay_alpha)
            draw_fps(vis, metrics.fps_now())
            has_joystick = bool(detections)
            status_color = (255, 220, 80) if has_joystick else (0, 0, 255)
            status_state = "OK" if has_joystick else "NO_JOYSTICK"
            draw_status(
                vis,
                "mode={} src={} detN={} {}".format(
                    tracker_mode, source, args.det_interval, status_state
                ),
                color=status_color,
            )
            metrics.add_stage_ms("render_overlay", (time.perf_counter() - t_render) * 1000.0)

        if write_now:
            t_writer = time.perf_counter()
            writer.write(vis)
            metrics.add_stage_ms("writer", (time.perf_counter() - t_writer) * 1000.0)

        if can_show:
            if render_now:
                t_imshow = time.perf_counter()
                cv2.imshow("jetson-yolo", vis)
                metrics.add_stage_ms("imshow", (time.perf_counter() - t_imshow) * 1000.0)
            t_wait = time.perf_counter()
            key = cv2.waitKey(1) & 0xFF
            metrics.add_stage_ms("waitkey", (time.perf_counter() - t_wait) * 1000.0)
            if key in (27, ord("q")):
                break

        if metrics.frames % 30 == 0:
            snap = metrics.snapshot()
            print(
                "progress: {}/{} fps={:.2f} infer_p50={:.2f}ms infer_samples={}".format(
                    int(snap["frames"]),
                    args.frames,
                    snap["fps"],
                    snap["infer_ms_p50"],
                    int(snap["infer_samples"]),
                )
            )

    cap.release()
    if writer is not None:
        writer.release()
    if can_show:
        cv2.destroyAllWindows()

    snap = metrics.snapshot()
    if int(snap["frames"]) == 0:
        raise RuntimeError("Camera opened but no frames were processed")

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": str(engine_path),
        "camera": args.camera,
        "frames": int(snap["frames"]),
        "requested_frames": args.frames,
        "fps": snap["fps"],
        "total_sec": snap["total_sec"],
        "infer_ms_mean": snap["infer_ms_mean"],
        "infer_ms_p50": snap["infer_ms_p50"],
        "infer_ms_p95": snap["infer_ms_p95"],
        "infer_samples": int(snap["infer_samples"]),
        "det_frames": int(snap["det_frames"]),
        "det_ratio": snap["det_ratio"],
        "input_shape": [int(x) for x in in_shape],
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "width": args.width,
        "height": args.height,
        "camera_fps_requested": args.fps,
        "det_interval": args.det_interval,
        "track_interval": args.track_interval,
        "tracker": tracker_mode,
        "video_skip": args.video_skip,
        "video_codec": video_codec,
        "stage_ms": snap.get("stage_ms", {}),
    }

    print("--- src.app runtime ---")
    print(
        "frames={}/{} total_sec={:.2f} fps={:.2f}".format(
            report["frames"], report["requested_frames"], report["total_sec"], report["fps"]
        )
    )
    print(
        "infer_ms mean={:.2f} p50={:.2f} p95={:.2f}".format(
            report["infer_ms_mean"], report["infer_ms_p50"], report["infer_ms_p95"]
        )
    )
    print("input_shape={}".format(report["input_shape"]))
    print("detection_frames={}/{}".format(report["det_frames"], report["frames"]))
    if report["stage_ms"]:
        sorted_stages = sorted(
            report["stage_ms"].items(), key=lambda kv: kv[1].get("mean", 0.0), reverse=True
        )
        print("stage_top_ms={}".format(", ".join(
            [
                "{}:{:.2f}".format(name, stats.get("mean", 0.0))
                for name, stats in sorted_stages[:5]
            ]
        )))

    if save_json_path:
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("report_saved={}".format(save_json_path))

    if save_video_path:
        print("video_saved={}".format(save_video_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
