#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path

import cv2

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_leaf_folders(root: Path):
    """
    返回形如 root/category/session 的二级子目录，
    只要里面有图片就认为是要处理的“叶子目录”。
    """
    folders = []
    for cat in root.iterdir():
        if not cat.is_dir():
            continue
        for sub in cat.iterdir():
            if not sub.is_dir():
                continue
            imgs = []
            for ext in IMG_EXTS:
                imgs.extend(glob.glob(str(sub / f"*{ext}")))
            if imgs:
                folders.append(sub)
    return sorted(folders)


def images_to_video(
    folder: Path,
    out_root: Path,
    fps: int = 30,
) -> bool:
    """
    将单个子文件夹中的图片合成为视频。
    输出路径：out_root / category / (folder.name + ".mp4")
    """
    images = []
    for ext in IMG_EXTS:
        images.extend(glob.glob(str(folder / f"*{ext}")))
    images = sorted(images)

    if not images:
        print(f"[SKIP] No images in {folder}")
        return False

    first = cv2.imread(images[0])
    if first is None:
        print(f"[ERROR] Failed to read first image in {folder}")
        return False
    h, w = first.shape[:2]

    # 例如：root=ManiTIL, folder=ManiTIL/cabinet/2026-0209-13-16-07
    # rel = cabinet/2026-0209-13-16-07
    # out_dir = out_root / "cabinet"
    # out_name = "2026-0209-13-16-07.mp4"
    rel = folder.relative_to(folder.parents[2])  # folder.parents[2] == root
    cat_dir = rel.parent  # cabinet
    out_dir = out_root / cat_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{folder.name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    count = 0
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [WARN] Failed to read {img_path}, skip.")
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        count += 1

    writer.release()

    if count == 0:
        print(f"[WARN] No valid frames written for {folder}")
        if out_path.exists():
            out_path.unlink()
        return False

    print(f"[OK] {folder} -> {out_path} ({count} frames @ {fps} fps)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert ManiTIL image folders into 30fps videos."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="/data/EgoLoc/ManiTIL",
        help="Input ManiTIL root directory.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/data/EgoLoc/ManiTIL_videos",
        help="Output root directory for videos.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video fps (default: 30).",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        print(f"ERROR: input_root does not exist: {input_root}")
        return 1

    folders = list_leaf_folders(input_root)
    print(f"Found {len(folders)} leaf folders under {input_root}")

    ok, fail = 0, 0
    for idx, folder in enumerate(folders, 1):
        print(f"[{idx}/{len(folders)}] {folder}")
        try:
            if images_to_video(folder, output_root, fps=args.fps):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            fail += 1

    print("=" * 40)
    print(f"Done. Success: {ok}, Failed: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())