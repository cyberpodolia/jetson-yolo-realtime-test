from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


DEFAULT_WEIGHTS = r"H:\work\jetson-yolo\runs\detect\joystick_320_neg_v4\weights\best.pt"


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Realtime webcam inference with YOLO.")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to .pt model file")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index")
    ap.add_argument("--imgsz", type=int, default=320, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.84, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu")
    ap.add_argument("--window", default="YOLO Realtime", help="OpenCV window title")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    weights_path = resolve_path(repo_root, args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: python -m pip install ultralytics"
        ) from exc

    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    print("Press 'q' or ESC to exit.")
    print(
        "Running with "
        f"weights={weights_path}, imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}, device={args.device}"
    )

    prev_time = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame read failed. Exiting.")
            break

        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        plotted = results[0].plot()

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0

        cv2.putText(
            plotted,
            f"FPS: {fps:.1f}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (40, 240, 40),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(args.window, plotted)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
