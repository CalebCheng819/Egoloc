#!/usr/bin/env python3
"""
重跑缺失 Ego4D 速度（多 GPU）

用途：
  - 只针对“缺失 *_with_speed_twohands.json 或文件过小”的 Ego4D 视频片段重跑速度
  - 通过创建一个临时 video_folder（软链接到原视频）来复用 produce_velocity_multigpu.py

默认路径（均为 egoloc 容器内路径）：
  - 原始视频：/data/EgoLoc/Ego4D/*.mp4
  - 输出根目录：/home/EgoLoc/hand_data_drawer/Ego4d/<stem>/<stem>_with_speed_twohands.json
  - 临时子集目录：/home/EgoLoc/hand_data_drawer/Ego4d_missing_tmp

运行（推荐在 egoloc 容器内）：
  python rerun_missing_ego4d_speed_multigpu.py --gpus 0 1 2 3
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


DEFAULT_VIDEO_ROOT = Path("/data/EgoLoc/Ego4D")
DEFAULT_OUTPUT_ROOT = Path("/home/EgoLoc/hand_data_drawer/Ego4d")
DEFAULT_TMP_FOLDER = Path("/home/EgoLoc/hand_data_drawer/Ego4d_missing_tmp")
DEFAULT_PRODUCER = Path("/home/EgoLoc/produce_velocity_multigpu.py")


def _expected_speed_json(output_root: Path, stem: str) -> Path:
    return output_root / stem / f"{stem}_with_speed_twohands.json"


def find_missing(video_root: Path, output_root: Path, *, min_bytes: int = 10) -> Tuple[List[Path], List[str]]:
    videos = sorted(video_root.glob("*.mp4"))
    missing: List[str] = []
    for v in videos:
        stem = v.stem
        speed_json = _expected_speed_json(output_root, stem)
        if (not speed_json.exists()) or speed_json.stat().st_size <= min_bytes:
            missing.append(stem)
    return videos, missing


def prepare_tmp_folder(tmp_folder: Path, video_root: Path, stems: List[str]) -> None:
    tmp_folder.mkdir(parents=True, exist_ok=True)

    # clean old contents
    for p in tmp_folder.iterdir():
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

    for stem in stems:
        src = video_root / f"{stem}.mp4"
        if not src.exists():
            print(f"[WARN] missing source video, skip: {src}")
            continue
        dst = tmp_folder / src.name
        os.symlink(src, dst)


def run_multigpu(
    producer: Path,
    *,
    video_folder: Path,
    output_root: Path,
    encoder: str,
    gpus: List[int] | None,
    dino_batch: int,
    no_skip: bool,
    log_path: Path | None,
) -> int:
    if not producer.exists():
        raise FileNotFoundError(f"producer script not found: {producer}")

    cmd = [
        "python",
        str(producer),
        "--video_folder",
        str(video_folder),
        "--output_root",
        str(output_root),
        "--encoder",
        encoder,
        "--dino_batch",
        str(dino_batch),
    ]
    if gpus:
        cmd += ["--gpus", *[str(g) for g in gpus]]
    if no_skip:
        cmd += ["--no_skip"]

    print("[RUN]", " ".join(cmd))

    if log_path is None:
        return subprocess.run(cmd).returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        return p.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", type=str, default=str(DEFAULT_VIDEO_ROOT))
    ap.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tmp_folder", type=str, default=str(DEFAULT_TMP_FOLDER))
    ap.add_argument("--producer", type=str, default=str(DEFAULT_PRODUCER))
    ap.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitl"])
    ap.add_argument("--dino_batch", type=int, default=4)
    ap.add_argument("--min_bytes", type=int, default=10, help="treat <= this size as missing")
    ap.add_argument("--no_skip", action="store_true", help="pass --no_skip to producer")
    ap.add_argument("--gpus", type=int, nargs="*", default=None, help="GPU ids, e.g. --gpus 0 1 2 3 (default: all)")
    ap.add_argument("--log", type=str, default="rerun_ego4d_missing_multigpu.log")
    args = ap.parse_args()

    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    tmp_folder = Path(args.tmp_folder)
    producer = Path(args.producer)
    log_path = Path(args.log) if args.log else None

    videos, missing = find_missing(video_root, output_root, min_bytes=args.min_bytes)
    print(f"[INFO] Total videos: {len(videos)}")
    print(f"[INFO] Missing/tiny speed JSON: {len(missing)}")
    if missing:
        for s in missing:
            print(f"  - {s}")
    else:
        print("[INFO] Nothing to rerun.")
        return

    print(f"[STEP] Preparing tmp folder: {tmp_folder}")
    prepare_tmp_folder(tmp_folder, video_root, missing)

    print("[STEP] Running multi-GPU producer...")
    rc = run_multigpu(
        producer,
        video_folder=tmp_folder,
        output_root=output_root,
        encoder=args.encoder,
        gpus=args.gpus,
        dino_batch=args.dino_batch,
        no_skip=args.no_skip,
        log_path=log_path,
    )
    print(f"[DONE] producer exit_code={rc}")

    # verify
    _, missing_after = find_missing(video_root, output_root, min_bytes=args.min_bytes)
    print(f"[VERIFY] Missing after rerun: {len(missing_after)}")
    if missing_after:
        for s in missing_after:
            print(f"  - {s}")


if __name__ == "__main__":
    main()

