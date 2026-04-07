import argparse
import random
import shutil
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build YOLO dataset from synthetic RGB+mask pairs.")
    ap.add_argument("--rgb-dir", default="synthetic_out/rgb", help="Directory with rgb_*.png")
    ap.add_argument("--mask-dir", default="synthetic_out/mask", help="Directory with mask_*.png")
    ap.add_argument("--out-root", default="data/synth_only_1000", help="Output dataset root")
    ap.add_argument("--class-id", type=int, default=0, help="YOLO class id")
    ap.add_argument("--class-name", default="joystick", help="Class name for dataset.yaml")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-area", type=int, default=16, help="Min bbox area in pixels")
    return ap.parse_args()


def resolve(repo_root: Path, raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def mask_path_for(rgb_path: Path, mask_dir: Path) -> Path:
    name = rgb_path.name
    if name.startswith("rgb_"):
        name = "mask_" + name[len("rgb_") :]
    return mask_dir / name


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def yolo_label_from_mask(mask_path: Path, class_id: int, min_area: int) -> str:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")

    h, w = mask.shape[:2]
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return ""

    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    if bw * bh < int(min_area):
        return ""

    xc = (x + bw * 0.5) / float(w)
    yc = (y + bh * 0.5) / float(h)
    wn = bw / float(w)
    hn = bh / float(h)
    return f"{int(class_id)} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def write_yaml(out_root: Path, class_name: str) -> Path:
    out = out_root / "dataset.yaml"
    with out.open("w", encoding="utf-8") as f:
        f.write(f"path: {out_root.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n")
        f.write(f"  0: {class_name}\n")
    return out


def split_items(items, train_ratio: float, val_ratio: float):
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return (
        items[:n_train],
        items[n_train : n_train + n_val],
        items[n_train + n_val : n_train + n_val + n_test],
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    rgb_dir = resolve(repo_root, args.rgb_dir)
    mask_dir = resolve(repo_root, args.mask_dir)
    out_root = resolve(repo_root, args.out_root)

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB dir not found: {rgb_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")

    rgb_files = sorted(rgb_dir.glob("*.png"))
    if not rgb_files:
        raise RuntimeError(f"No PNG files found in {rgb_dir}")

    pairs = []
    for rgb in rgb_files:
        mask = mask_path_for(rgb, mask_dir)
        if mask.exists():
            pairs.append((rgb, mask))

    if not pairs:
        raise RuntimeError("No RGB/mask pairs found")

    random.seed(int(args.seed))
    random.shuffle(pairs)

    train_items, val_items, test_items = split_items(pairs, float(args.train_ratio), float(args.val_ratio))
    splits = [("train", train_items), ("val", val_items), ("test", test_items)]

    if out_root.exists():
        shutil.rmtree(out_root)
    for split_name, _ in splits:
        (out_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    total_pos = 0
    total_neg = 0
    for split_name, items in splits:
        for rgb, mask in items:
            dst_img = out_root / "images" / split_name / rgb.name
            dst_lbl = out_root / "labels" / split_name / (rgb.stem + ".txt")
            link_or_copy(rgb, dst_img)
            line = yolo_label_from_mask(mask, args.class_id, args.min_area)
            with dst_lbl.open("w", encoding="utf-8") as f:
                if line:
                    f.write(line + "\n")
                    total_pos += 1
                else:
                    total_neg += 1

    yaml_path = write_yaml(out_root, args.class_name)
    print(f"Dataset built: {out_root}")
    print(f"YAML: {yaml_path}")
    print(f"Total: {len(pairs)} | positives: {total_pos} | negatives: {total_neg}")
    print(f"Split sizes: train={len(train_items)} val={len(val_items)} test={len(test_items)}")


if __name__ == "__main__":
    main()
