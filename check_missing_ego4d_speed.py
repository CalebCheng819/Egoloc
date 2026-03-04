#!/usr/bin/env python3
"""
读取 /data/EgoLoc/Ego4D 与 /home/EgoLoc/hand_data_drawer/Ego4d 的对应关系，
找出缺失 *_with_speed_twohands.json 的视频，
为这些视频建立一个临时子集目录，然后调用 produce_velocity_multigpu.py 重跑。
"""

from __future__ import annotations
from pathlib import Path
import os
import shutil
import subprocess
from typing import List, Tuple

VIDEO_ROOT = Path("/data/EgoLoc/Ego4D")
OUT_ROOT = Path("/home/EgoLoc/hand_data_drawer/Ego4d")
TMP_FOLDER = Path("/home/EgoLoc/hand_data_drawer/Ego4d_missing_tmp")
PRODUCE_SCRIPT = Path("/home/chengjuntao/data0/EgoLoc/produce_velocity_multigpu.py")


def find_missing() -> Tuple[List[Path], List[str]]:
    """返回 (所有视频列表, 缺失速度的 stem 列表)"""
    videos = sorted(VIDEO_ROOT.glob("*.mp4"))
    missing_stems: List[str] = []

    for v in videos:
        stem = v.stem
        speed_json = OUT_ROOT / stem / f"{stem}_with_speed_twohands.json"
        if not speed_json.exists() or speed_json.stat().st_size <= 10:
            missing_stems.append(stem)

    return videos, missing_stems


def prepare_tmp_folder(missing_stems: List[str]) -> None:
    """在 TMP_FOLDER 下为缺失的视频创建软链接子集"""
    if TMP_FOLDER.exists():
        # 清空旧内容
        for p in TMP_FOLDER.iterdir():
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    else:
        TMP_FOLDER.mkdir(parents=True, exist_ok=True)

    for stem in missing_stems:
        src = VIDEO_ROOT / f"{stem}.mp4"
        if not src.exists():
            print(f"[WARN] 视频不存在，跳过: {src}")
            continue
        dst = TMP_FOLDER / src.name
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass
        print(f"[LINK] {dst} -> {src}")


def run_multigpu(gpus: str = "0 1 2 3", encoder: str = "vits") -> int:
    """调用 produce_velocity_multigpu.py，只处理 TMP_FOLDER 里的视频"""
    if not PRODUCE_SCRIPT.exists():
        raise FileNotFoundError(f"produce_velocity_multigpu.py not found at {PRODUCE_SCRIPT}")

    cmd = [
        "python",
        str(PRODUCE_SCRIPT),
        "--video_folder",
        str(TMP_FOLDER),
        "--output_root",
        str(OUT_ROOT),
        "--encoder",
        encoder,
        "--gpus",
    ] + gpus.split()

    print("\n[RUN] 命令：", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    videos, missing_stems = find_missing()
    total = len(videos)
    miss_n = len(missing_stems)

    print(f"Total videos under {VIDEO_ROOT}: {total}")
    print(f"Missing or tiny speed JSONs under {OUT_ROOT}: {miss_n}")
    if not missing_stems:
        print("✅ 没有缺失，无需重跑。")
        return

    print("\n将要重跑的 stem 列表：")
    for s in missing_stems:
        print(f"- {s}")

    # 1) 准备临时子集目录
    print(f"\n[STEP] 准备临时目录: {TMP_FOLDER}")
    prepare_tmp_folder(missing_stems)

    # 2) 调用多卡脚本
    print("\n[STEP] 调用 produce_velocity_multigpu.py 重新跑缺失视频...")
    ret = run_multigpu(gpus="0 1 2 3", encoder="vits")
    if ret != 0:
        print(f"[ERROR] produce_velocity_multigpu.py 退出码 {ret}，请检查日志。")
        return

    # 3) 重新检查缺失情况
    print("\n[STEP] 重新检查缺失速度文件……")
    _, missing_after = find_missing()
    if not missing_after:
        print("✅ 所有 Ego4D 视频现在都有速度 JSON 了。")
    else:
        print("⚠️ 仍有以下视频缺失速度，请手动检查：")
        for s in missing_after:
            print(f"- {s}")


if __name__ == "__main__":
    main()