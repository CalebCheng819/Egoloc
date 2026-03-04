"""
produce_velocity_recursive_multigpu.py
======================================
在原始 `produce_velocity_multigpu.py` 基础上封装的递归版本：

- 支持从一个**根目录**递归地查找所有视频文件（多级子目录），
  例如：/data/EgoLoc/ManiTIL 下的 cabinet/xxx/*.mp4 等；
- 其他行为（多 GPU 并行、断点续跑、输出结构）与原脚本保持一致。

典型用法:

    # 递归处理 /data/EgoLoc/ManiTIL 下所有视频
    python produce_velocity_recursive_multigpu.py \
        --video_root /data/EgoLoc/ManiTIL \
        --output_root /home/EgoLoc/hand_data_drawer/ManiTIL_speed \
        --gpus 0 1 2 3 \
        --dino_batch 8
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List

import multiprocessing as mp

from produce_velocity_multigpu import (
    _worker,
    _is_done,
    VIDEO_EXTS,
    log,
)


def _list_videos_recursive(root: Path) -> List[Path]:
    """
    递归遍历 root，收集所有后缀在 VIDEO_EXTS 里的文件。
    """
    videos: List[Path] = []
    if not root.exists():
        log.error("Video root does not exist: %s", root)
        return videos

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)

    return sorted(videos, key=lambda p: p.name)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU recursive velocity generation (recursive video root)."
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="/data/EgoLoc/ManiTIL",
        help="Root directory to recursively search videos (default: /data/EgoLoc/ManiTIL)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/EgoLoc/hand_data_drawer/ManiTIL_speed",
        help="Output root directory",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs to use (default: all available GPUs)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=["vits", "vitl"],
        help="Depth model encoder (default: vits)",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Disable skip-if-done (reprocess everything)",
    )
    parser.add_argument(
        "--dino_batch",
        type=int,
        default=4,
        help="DINO batch size for hand detection (default: 4)",
    )
    args = parser.parse_args()

    # ── 检测可用 GPU ──
    if args.gpus is not None:
        gpu_ids = args.gpus
    else:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        gpu_ids = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]

    if not gpu_ids:
        log.error("No GPUs detected. Exiting.")
        sys.exit(1)
    log.info("Using %d GPUs: %s", len(gpu_ids), gpu_ids)

    # ── 扫描视频（递归） ──
    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_videos = _list_videos_recursive(video_root)
    if not all_videos:
        log.error("No video files found under %s (recursive).", video_root)
        sys.exit(1)

    log.info("Found %d videos under %s (recursive).", len(all_videos), video_root)

    # ── 断点续跑：过滤已完成的 ──
    if args.no_skip:
        todo_videos = [str(p) for p in all_videos]
    else:
        todo_videos = [str(p) for p in all_videos if not _is_done(str(p), output_root)]
        skipped = len(all_videos) - len(todo_videos)
        if skipped > 0:
            log.info("Skipped %d already-done videos", skipped)

    if not todo_videos:
        log.info("All %d videos already processed. Nothing to do.", len(all_videos))
        return

    log.info(
        "Total: %d videos, to process: %d, GPUs: %d",
        len(all_videos),
        len(todo_videos),
        len(gpu_ids),
    )

    # ── 轮询分配视频到各 GPU ──
    gpu_tasks: dict[int, List[str]] = {g: [] for g in gpu_ids}
    for i, vp in enumerate(todo_videos):
        gpu = gpu_ids[i % len(gpu_ids)]
        gpu_tasks[gpu].append(vp)

    for g in gpu_ids:
        log.info("GPU %d: %d videos assigned", g, len(gpu_tasks[g]))

    # ── 启动子进程（spawn 模式，避免 fork 后 CUDA 状态冲突） ──
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    t_start = time.time()

    for gpu_id in gpu_ids:
        if not gpu_tasks[gpu_id]:
            continue
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, gpu_tasks[gpu_id], str(output_root), args.encoder, result_queue, args.dino_batch),
            name=f"GPU-{gpu_id}",
        )
        p.start()
        processes.append(p)
        log.info("Started worker process PID=%d on GPU %d", p.pid, gpu_id)

    # ── 等待所有子进程完成 ──
    for p in processes:
        p.join()

    total_elapsed = time.time() - t_start

    # ── 汇总结果 ──
    all_success = []
    all_failed = []
    while not result_queue.empty():
        r = result_queue.get_nowait()
        all_success.extend(r["success"])
        all_failed.extend(r["failed"])

    log.info("=" * 60)
    log.info("ALL DONE in %.1f s (%.1f min)", total_elapsed, total_elapsed / 60)
    log.info("Success: %d", len(all_success))
    if all_success:
        for s in all_success:
            log.info("  OK  %s", s)
    log.info("Failed:  %d", len(all_failed))
    if all_failed:
        for name, err in all_failed:
            log.info("  FAIL %s: %s", name, err)
    log.info("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

