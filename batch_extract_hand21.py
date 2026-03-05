#!/usr/bin/env python3
"""
批量为多个视频生成 ViTPose 21 点手部关键点坐标（HAMER 格式）。
基于 extract_hand21_and_visualize.py，从视频文件夹中遍历所有 .mp4，逐视频输出 hand21 JSON。
视频无深度时使用全零 depth，仅走 Detectron2+ViTPose 路径。
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

EGOLOC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EGOLOC_ROOT))

from egoloc_hand_utils import _get_body_detector, _get_vitpose_model, _load_depth
from extract_hand21_and_visualize import (
    _get_hand21_from_frame,
    draw_hand_points,
    fix_left_right_identity_hand21,
    kpts_to_json_serializable,
)


def collect_videos(video_folder: Path, recursive: bool = False) -> List[Path]:
    """收集视频路径：默认仅当前目录下 .mp4，recursive 时包含子目录。"""
    video_folder = video_folder.resolve()
    if not video_folder.is_dir():
        return []
    if recursive:
        return sorted(video_folder.rglob("*.mp4"))
    return sorted(video_folder.glob("*.mp4"))


def process_one_video(
    video_path: Path,
    cpm,
    detector,
    *,
    out_dir: Path,
    depth_dir: Optional[Path] = None,
    device: str = "cuda",
    kp_conf: float = 0.25,
    save_vis: bool = False,
    max_preload_frames: int = 2000,
) -> Optional[Path]:
    """
    处理单个视频，写出 hand21 JSON；可选写出可视化 mp4。
    depth_dir 为 None 时使用全零 depth（仅 Detectron2+ViTPose）。
    返回写入的 JSON 路径，失败返回 None。
    """
    video_path = video_path.resolve()
    video_stem = video_path.stem
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / f"{video_stem}_hand21.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] cannot open: {video_path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 预加载深度（若有）
    use_depth = depth_dir is not None and depth_dir.exists()
    if use_depth and total_frames <= max_preload_frames:
        depths = [_load_depth(depth_dir, i) for i in range(total_frames)]
    else:
        depths = None

    hand21_left: Dict[str, List[List[float]]] = {}
    hand21_right: Dict[str, List[List[float]]] = {}
    prev_wrist_L: Optional[Tuple[float, float]] = None
    prev_wrist_R: Optional[Tuple[float, float]] = None

    writer = None
    for fidx in range(total_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if use_depth and depths is not None:
            depth = depths[fidx]
        elif use_depth:
            depth = _load_depth(depth_dir, fidx)
        else:
            depth = None

        if depth is None:
            gray = np.zeros((H, W), dtype=np.float32)
        else:
            gray = depth.astype(np.float32)

        hands = _get_hand21_from_frame(
            frame,
            gray,
            cpm,
            detector=detector,
            kp_conf_thr=kp_conf,
            verbose=(fidx % 100 == 0 and fidx > 0),
        )
        hands = fix_left_right_identity_hand21(hands, prev_wrist_L, prev_wrist_R)

        frame_key = str(fidx + 1)
        if hands["left"] is not None:
            hand21_left[frame_key] = kpts_to_json_serializable(hands["left"])
            prev_wrist_L = (float(hands["left"][0, 0]), float(hands["left"][0, 1]))
        if hands["right"] is not None:
            hand21_right[frame_key] = kpts_to_json_serializable(hands["right"])
            prev_wrist_R = (float(hands["right"][0, 0]), float(hands["right"][0, 1]))

        if save_vis:
            vis = draw_hand_points(
                frame, hands["left"], hands["right"], kp_conf_thr=kp_conf
            )
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                mp4_out = out_dir / f"{video_stem}_hamer21.mp4"
                writer = cv2.VideoWriter(str(mp4_out), fourcc, float(fps), (W, H))
            writer.write(vis)

        if (fidx + 1) % 100 == 0 or fidx == 0:
            print(f"  [{video_stem}] frame {fidx + 1}/{total_frames}")

    cap.release()
    if writer is not None:
        writer.release()

    result = {"left": hand21_left, "right": hand21_right}
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  [SAVE] {json_out}")
    return json_out


def main():
    ap = argparse.ArgumentParser(
        description="批量为视频生成 21 点 HAMER 手部关键点 JSON"
    )
    ap.add_argument(
        "--video_folder",
        type=str,
        default="/data/EgoLoc/EgoDex_10fps/long",
        help="视频所在文件夹（默认: /data/EgoLoc/EgoDex_10fps/long）",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录，默认: <EgoLoc>/output/hand21_<video_folder_name>",
    )
    ap.add_argument(
        "--depth_base",
        type=str,
        default=None,
        help="可选：深度根目录，每视频对应 <depth_base>/<video_stem>/depth 与 pred_depth_*.npy",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    ap.add_argument(
        "--kp_conf",
        type=float,
        default=0.25,
        help="关键点置信度阈值",
    )
    ap.add_argument(
        "--no_vis",
        action="store_true",
        help="不生成可视化 mp4，只写 JSON（更快）",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索子目录中的 .mp4",
    )
    ap.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="逗号分隔的 GPU ID，如 0,1,2,3；多卡时自动分配视频到各卡并行",
    )
    ap.add_argument(
        "--worker_tasks",
        type=str,
        default=None,
        help="[内部] worker 模式：本进程要处理的视频路径列表 JSON 文件",
    )
    ap.add_argument(
        "--worker_output_dir",
        type=str,
        default=None,
        help="[内部] worker 模式：本进程输出目录",
    )
    args = ap.parse_args()

    # ---------- Worker 模式：只处理 tasks 中的视频并写入 worker_output_dir ----------
    if args.worker_tasks and args.worker_output_dir:
        with open(args.worker_tasks, "r", encoding="utf-8") as f:
            task_paths = json.load(f)
        worker_out = Path(args.worker_output_dir)
        worker_out.mkdir(parents=True, exist_ok=True)
        device = "cuda"  # 主进程已设置 CUDA_VISIBLE_DEVICES
        depth_base = Path(args.depth_base) if args.depth_base else None
        cpm = _get_vitpose_model(device)
        detector = _get_body_detector(device)
        if cpm is None or detector is None:
            print("[ERROR] worker: failed to load ViTPose or Detectron2")
            sys.exit(1)
        success = 0
        for i, p in enumerate(task_paths):
            vpath = Path(p)
            depth_dir = (depth_base / vpath.stem / "depth") if depth_base else None
            print(f"[worker] [{i+1}/{len(task_paths)}] {vpath.name}")
            out_path = process_one_video(
                vpath,
                cpm,
                detector,
                out_dir=worker_out,
                depth_dir=depth_dir,
                device=device,
                kp_conf=args.kp_conf,
                save_vis=not args.no_vis,
            )
            if out_path is not None:
                success += 1
        print(f"[worker] DONE {success}/{len(task_paths)}")
        sys.exit(0)

    # ---------- 主进程 ----------
    video_folder = Path(args.video_folder)
    if not video_folder.exists():
        print(f"[ERROR] video_folder not found: {video_folder}")
        sys.exit(1)

    videos = collect_videos(video_folder, recursive=args.recursive)
    if not videos:
        print(f"[ERROR] no .mp4 found under {video_folder}")
        sys.exit(1)
    print(f"[INFO] found {len(videos)} video(s) in {video_folder}")

    out_dir = Path(args.out_dir) if args.out_dir else (EGOLOC_ROOT / "output" / f"hand21_{video_folder.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] output dir: {out_dir}")

    gpu_list = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if len(gpu_list) > 1:
        # 多卡：按卡数切分视频，起子进程，再合并 JSON 到 out_dir
        n = len(videos)
        n_workers = len(gpu_list)
        chunk_size = (n + n_workers - 1) // n_workers
        procs = []
        worker_dirs = []
        script_path = os.path.abspath(__file__)
        for i, gpu_id in enumerate(gpu_list):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            if start >= end:
                continue
            chunk = [str(v.resolve()) for v in videos[start:end]]
            tasks_file = out_dir / f"worker_{i}_tasks.json"
            worker_out = out_dir / f"worker_{i}"
            worker_out.mkdir(parents=True, exist_ok=True)
            worker_dirs.append(worker_out)
            with open(tasks_file, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False)
            cmd = [
                sys.executable,
                script_path,
                "--worker_tasks", str(tasks_file),
                "--worker_output_dir", str(worker_out),
                "--kp_conf", str(args.kp_conf),
            ]
            if args.no_vis:
                cmd.append("--no_vis")
            if args.depth_base:
                cmd.extend(["--depth_base", args.depth_base])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            procs.append(subprocess.Popen(cmd, env=env, cwd=os.getcwd()))
        for p in procs:
            p.wait()
        # 将各 worker 目录下的 *_hand21.json 移到 out_dir
        for wd in worker_dirs:
            for jf in wd.glob("*_hand21.json"):
                dst = out_dir / jf.name
                shutil.move(str(jf), str(dst))
            for mp4 in wd.glob("*_hamer21.mp4"):
                dst = out_dir / mp4.name
                shutil.move(str(mp4), str(dst))
            try:
                wd.rmdir()
            except OSError:
                pass
        print(f"[DONE] multi-GPU: results merged to {out_dir}")
        return

    # 单卡
    depth_base = Path(args.depth_base) if args.depth_base else None
    cpm = _get_vitpose_model(args.device)
    detector = _get_body_detector(args.device)
    if cpm is None or detector is None:
        print("[ERROR] failed to load ViTPose or Detectron2")
        sys.exit(1)

    success = 0
    for i, vpath in enumerate(videos):
        depth_dir = None
        if depth_base is not None:
            depth_dir = depth_base / vpath.stem / "depth"
        print(f"[{i+1}/{len(videos)}] {vpath.name}")
        out_path = process_one_video(
            vpath,
            cpm,
            detector,
            out_dir=out_dir,
            depth_dir=depth_dir,
            device=args.device,
            kp_conf=args.kp_conf,
            save_vis=not args.no_vis,
        )
        if out_path is not None:
            success += 1

    print(f"[DONE] {success}/{len(videos)} videos wrote hand21 JSON to {out_dir}")


if __name__ == "__main__":
    main()
