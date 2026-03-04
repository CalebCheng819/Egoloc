#!/usr/bin/env python3
"""将视频转换为帧图片序列"""
import argparse
from pathlib import Path

import cv2


def video_to_frames(video_path: str, out_dir: str = None, fmt: str = "png"):
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_dir = Path(out_dir) if out_dir else video_path.parent / (video_path.stem + "_frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ext = "." + fmt
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        out_path = out_dir / f"{idx:06d}{ext}"
        cv2.imwrite(str(out_path), frame)
        idx += 1
    cap.release()
    print(f"[OK] {video_path.name} -> {out_dir}/ ({idx} frames)")
    return out_dir, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="Video file(s)")
    ap.add_argument("--out", type=str, default=None, help="Output dir (default: <video>_frames)")
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "jpg"])
    args = ap.parse_args()

    for v in args.videos:
        try:
            video_to_frames(v, args.out, args.fmt)
        except Exception as e:
            print(f"[ERR] {v}: {e}")


if __name__ == "__main__":
    main()
