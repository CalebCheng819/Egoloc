"""
produce_velocity_multigpu.py
============================
多 GPU 并行版本的速度生成脚本。

用法:
    python produce_velocity_multigpu.py                        # 使用全部 GPU
    python produce_velocity_multigpu.py --gpus 0 1 2           # 指定 GPU
    python produce_velocity_multigpu.py --gpus 0 1 --encoder vitl

设计思路:
    1. 主进程扫描视频目录，按文件名排序，轮询分配给 N 个 GPU worker
    2. 每个 worker 是一个独立子进程（spawn 方式），通过 CUDA_VISIBLE_DEVICES
       环境变量绑定到指定 GPU
    3. 子进程内部 import egoloc_speed_twohands，模块级加载的模型会自动落在
       该进程可见的唯一 GPU 上
    4. 支持断点续跑：已有 *_with_speed_twohands.json 的视频自动跳过
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path
from typing import List, Tuple

import multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("multigpu")

# ── 支持的视频格式 ──
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# ---------------------------------------------------------------------------
# Worker: 运行在子进程中，绑定到指定 GPU
# ---------------------------------------------------------------------------
def _worker(
    gpu_id: int,
    video_list: List[str],
    output_root: str,
    encoder: str,
    result_queue: mp.Queue,
    dino_batch_size: int = 4,
):
    """
    每个 worker 独占一块 GPU，串行处理分配给自己的视频列表。
    通过 CUDA_VISIBLE_DEVICES 让当前进程只能看到一块 GPU，
    这样 egoloc_speed_twohands 里的全局模型会自动加载到这块卡上。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 延迟 import —— 必须在设好 CUDA_VISIBLE_DEVICES 之后才 import，
    # 否则模块级的 load_model / torch 会绑到错误的 GPU
    sys.path.insert(0, "/home/Egoloc/Egolocx")
    sys.path.insert(0, "/home/Egoloc")
    sys.path.insert(0, "/home/EgoLoc/Grounded-Segment-Anything/GroundingDINO")

    from egoloc_speed_twohands import extract_3d_speed_and_visualize

    output_root = Path(output_root)
    success = []
    failed = []

    for i, video_path in enumerate(video_list, 1):
        video_name = Path(video_path).stem
        outdir = output_root / video_name

        log.info("[GPU %d] (%d/%d) Processing: %s", gpu_id, i, len(video_list), video_name)
        t0 = time.time()

        try:
            pairs, speed_json, speed_png, depth_vis = extract_3d_speed_and_visualize(
                video_path=str(video_path),
                output_dir=str(outdir),
                device="cuda",      # 子进程只能看到 1 张卡，cuda == cuda:0
                encoder=encoder,
                dino_batch_size=dino_batch_size,
            )
            elapsed = time.time() - t0
            # 简要校验：数量 + 有效帧统计
            n_pairs = len(pairs)
            valid_L = sum(1 for p in pairs if len(p) >= 2 and p[1] > 0)
            valid_R = sum(1 for p in pairs if len(p) >= 3 and p[2] > 0)
            ok_check = n_pairs > 0 and (valid_L > 0 or valid_R > 0)
            check_tag = "OK" if ok_check else "WARN(all-zero?)"
            log.info(
                "[GPU %d] Done: %s (%.1fs) [%s] pairs=%d valid_L=%d valid_R=%d\n"
                "   ├─ speed JSON: %s\n"
                "   ├─ speed plot: %s\n"
                "   └─ depth vis:  %s",
                gpu_id, video_name, elapsed, check_tag, n_pairs, valid_L, valid_R,
                speed_json, speed_png, depth_vis,
            )
            success.append(video_name)

        except Exception as e:
            elapsed = time.time() - t0
            log.error("[GPU %d] FAILED: %s (%.1fs) error=%s", gpu_id, video_name, elapsed, e)
            failed.append((video_name, str(e)))

    result_queue.put({"gpu": gpu_id, "success": success, "failed": failed})


# ---------------------------------------------------------------------------
# 判断某个视频是否已经处理完成（断点续跑）
# ---------------------------------------------------------------------------
def _is_done(video_path: str, output_root: Path) -> bool:
    video_name = Path(video_path).stem
    outdir = output_root / video_name
    speed_json = outdir / f"{video_name}_with_speed_twohands.json"
    if speed_json.exists() and speed_json.stat().st_size > 10:
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-GPU parallel velocity generation")
    parser.add_argument(
        "--video_folder",
        type=str,
        default="/home/EgoLoc/hand_data_drawer/ego4dvideo",
        help="Input video directory",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/EgoLoc/hand_data_drawer/EgoDex_10",
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
            capture_output=True, text=True,
        )
        gpu_ids = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]

    if not gpu_ids:
        log.error("No GPUs detected. Exiting.")
        sys.exit(1)
    log.info("Using %d GPUs: %s", len(gpu_ids), gpu_ids)

    # ── 扫描视频 ──
    video_folder = Path(args.video_folder)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_videos = sorted(
        [p for p in video_folder.iterdir() if p.suffix.lower() in VIDEO_EXTS],
        key=lambda p: p.name,
    )
    if not all_videos:
        log.error("No video files found in %s", video_folder)
        sys.exit(1)

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

    log.info("Total: %d videos, to process: %d, GPUs: %d", len(all_videos), len(todo_videos), len(gpu_ids))

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
