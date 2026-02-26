from __future__ import annotations

import argparse
from pathlib import Path


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train YOLO model for joystick detection.")
    ap.add_argument("--data", default="data/dataset.yaml", help="Path to dataset.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO model checkpoint")
    ap.add_argument("--imgsz", type=int, default=320, help="Training image size")
    ap.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    ap.add_argument("--batch", type=int, default=32, help="Batch size")
    ap.add_argument("--device", default="0", help="Training device, e.g. 0 or cpu")
    ap.add_argument("--workers", type=int, default=8, help="Data loader workers")
    ap.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--project", default="runs/detect", help="Output project directory")
    ap.add_argument("--name", default="joystick_320", help="Run name")
    ap.add_argument("--cache", action="store_true", help="Cache images for faster training")
    ap.add_argument("--exist_ok", action="store_true", help="Allow existing run name")
    ap.add_argument("--resume", action="store_true", help="Resume latest checkpoint")
    ap.add_argument("--fliplr", type=float, default=0.3, help="Horizontal flip probability")
    ap.add_argument("--hsv_v", type=float, default=0.15, help="Brightness augmentation strength")
    ap.add_argument("--degrees", type=float, default=5.0, help="Max random rotation in degrees")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    data_path = resolve_path(repo_root, args.data)
    project_path = resolve_path(repo_root, args.project)
    if not data_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: python -m pip install ultralytics"
        ) from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=str(project_path),
        name=args.name,
        cache=args.cache,
        exist_ok=args.exist_ok,
        resume=args.resume,
        fliplr=args.fliplr,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"Training finished. Artifacts: {save_dir}")


if __name__ == "__main__":
    main()
