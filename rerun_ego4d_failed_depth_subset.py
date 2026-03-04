#!/usr/bin/env python3
"""
仅重跑 Ego4D 中 depth/VDA 失败的那 9 个视频片段。

失败列表来自 rerun_ego4d_missing_multigpu.log 末尾：
  - 41b8254c-ca1e-464c-9323-55301fb5f0e8_213_282
  - 4aa9dde3-5a49-43f9-83a7-598e92318951_2378_2407
  - 4d32b6c2-e922-4f62-9228-4600f876b5c1_268_287
  - 6e7ca65b-f2b3-4d34-8b6f-f1a260759f2b_2780_2845
  - 90eae323-a044-46db-ac25-d99e7fbbd49e_44_83
  - a57bdef4-0f94-4dc4-a5fe-168a8dcc6a5b_2704_2732
  - 4988de48-c77e-406d-8367-68fe2266a8aa_166_188
  - 6e7ca65b-f2b3-4d34-8b6f-f1a260759f2b_2076_2152
  - 90eae323-a044-46db-ac25-d99e7fbbd49e_126_153

策略：
  1. 在 /data/EgoLoc/Ego4D 下找到上述 stem 对应的 mp4
  2. 在 /home/EgoLoc/hand_data_drawer/Ego4d_failed_tmp 下为这些视频建立软链接
  3. 调用 /home/EgoLoc/produce_velocity_multigpu.py 只处理这个临时目录

注意：本脚本预期在 egoloc 容器内运行。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List


VIDEO_ROOT = Path("/data/EgoLoc/Ego4D")
OUTPUT_ROOT = Path("/home/EgoLoc/hand_data_drawer/Ego4d")
TMP_FOLDER = Path("/home/EgoLoc/hand_data_drawer/Ego4d_failed_tmp")
PRODUCER = Path("/home/EgoLoc/produce_velocity_multigpu.py")

# 这 9 个是在 VDA 深度阶段失败的片段
FAILED_STEMS: List[str] = [
    "41b8254c-ca1e-464c-9323-55301fb5f0e8_213_282",
    "4aa9dde3-5a49-43f9-83a7-598e92318951_2378_2407",
    "4d32b6c2-e922-4f62-9228-4600f876b5c1_268_287",
    "6e7ca65b-f2b3-4d34-8b6f-f1a260759f2b_2780_2845",
    "90eae323-a044-46db-ac25-d99e7fbbd49e_44_83",
    "a57bdef4-0f94-4dc4-a5fe-168a8dcc6a5b_2704_2732",
    "4988de48-c77e-406d-8367-68fe2266a8aa_166_188",
    "6e7ca65b-f2b3-4d34-8b6f-f1a260759f2b_2076_2152",
    "90eae323-a044-46db-ac25-d99e7fbbd49e_126_153",
]


def prepare_tmp_folder() -> None:
    TMP_FOLDER.mkdir(parents=True, exist_ok=True)

    # 清空旧内容
    for p in TMP_FOLDER.iterdir():
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

    for stem in FAILED_STEMS:
        src = VIDEO_ROOT / f"{stem}.mp4"
        if not src.exists():
            print(f"[WARN] 源视频不存在，跳过: {src}")
            continue
        dst = TMP_FOLDER / src.name
        try:
            os.symlink(src, dst)
            print(f"[LINK] {dst} -> {src}")
        except FileExistsError:
            print(f"[SKIP] 已存在: {dst}")


def run_multigpu(gpus: List[int] | None, encoder: str, dino_batch: int, no_skip: bool, log_path: Path | None) -> int:
    if not PRODUCER.exists():
        raise FileNotFoundError(f"producer script not found: {PRODUCER}")

    cmd = [
        "python",
        str(PRODUCER),
        "--video_folder",
        str(TMP_FOLDER),
        "--output_root",
        str(OUTPUT_ROOT),
        "--encoder",
        encoder,
        "--dino_batch",
        str(dino_batch),
    ]
    if gpus:
        cmd += ["--gpus", *[str(g) for g in gpus]]
    if no_skip:
        cmd += ["--no_skip"]

    print("\n[RUN]", " ".join(cmd))

    if log_path is None:
        return subprocess.run(cmd).returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        return p.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitl"])
    ap.add_argument("--dino_batch", type=int, default=4)
    ap.add_argument("--no_skip", action="store_true", help="传给 produce_velocity_multigpu.py 的 --no_skip")
    ap.add_argument("--gpus", type=int, nargs="*", default=None, help="GPU ids，如: --gpus 0 1 2 3 (默认: 全部)")
    ap.add_argument("--log", type=str, default="/home/EgoLoc/rerun_ego4d_failed_depth_subset.log")
    args = ap.parse_args()

    print(f"[INFO] 失败视频数量: {len(FAILED_STEMS)}")
    for s in FAILED_STEMS:
        print(f"  - {s}")

    print(f"\n[STEP] 准备临时目录: {TMP_FOLDER}")
    prepare_tmp_folder()

    log_path = Path(args.log) if args.log else None
    print("\n[STEP] 调用 produce_velocity_multigpu.py 仅重跑上述 9 个片段...")
    rc = run_multigpu(
        gpus=args.gpus,
        encoder=args.encoder,
        dino_batch=args.dino_batch,
        no_skip=args.no_skip,
        log_path=log_path,
    )
    print(f"\n[DONE] 退出码: {rc}")


if __name__ == "__main__":
    main()

