from __future__ import annotations
import os, sys, json, math, subprocess, time, warnings, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import os, json, math, logging
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境也能画
import matplotlib.pyplot as plt
import cv2

#源文件的import
import argparse
import base64
import json
import math
import os
import subprocess
import time
import warnings
from pathlib import Path
import open3d as o3d

import cv2
import dotenv  # read .env creds for GPT-4o
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage  # connected‑component helper
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Tuple, Optional, Any
import sys
sys.path.append("/home/EgoLoc/hamer")

# ---- 放在脚本最顶部：在所有 import 之前 ----
import os, sys

# 注意：用直引号，不要用中文/弯引号
MMPOSE_ROOT = "/home/EgoLoc/mmpose/mmpose-0.x"   # 这里应该是包含 mmpose/ 包的仓库根目录
HAMER_ROOT  = "/home/EgoLoc/hamer"               # 这里是 hamer 项目根目录

# 1) 设置 PYTHONPATH（等价于 export PYTHONPATH="...:$PYTHONPATH"）
paths = [MMPOSE_ROOT, HAMER_ROOT]
old = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = ":".join([p for p in paths if p]) + (":" + old if old else "")

# 2) 同步加到 sys.path，确保本进程内的 import 立刻可用
for p in paths:
    if p and p not in sys.path:
        sys.path.insert(0, p)





try:
    import torch
except ImportError:  # Allow import on machines without torch
    torch = None

# ---------------------------------------------------------------------------
# External repos that *must* exist inside EgoLoc root
# ---------------------------------------------------------------------------
try:
    import hamer  # type: ignore
except ImportError as e:
    raise ImportError(
        "HaMeR repo not found.  Make sure you cloned it inside the EgoLoc root."
    ) from e

try:
    from vitpose_model import ViTPoseModel  # type: ignore
except ImportError as e:
    raise ImportError(
        "vitpose_model.py not found.  It ships with HaMeR; "
        "confirm your PYTHONPATH contains that folder."
    ) from e

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.switch_backend("Agg")  # headless plotting


log = logging.getLogger(__name__)

# ---- 工具：原子写 JSON，避免空文件/半文件 ----
def _atomic_json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import savgol_filter
import open3d as o3d

# ====================== 基础环境 ======================
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
VDA_DIR = REPO_ROOT / "Video-Depth-Anything"

DEPTH_SCALE_M = 3.0   # inverse-depth → meters
MAX_REPAIR = 5

# ====================== 工具函数 ======================
def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _atomic_json_dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

# ====================== VDA ======================
def generate_depth_video_vda(video_path, depth_out):
    cmd = [
        "python", "run.py",
        "--input_video", str(video_path),
        "--output_dir", str(depth_out),
        "--encoder", "vits",
        "--save_npz"
    ]
    subprocess.run(cmd, cwd=VDA_DIR, check=True)

def _unpack_depth_npz(depth_dir: Path):
    npzs = list(depth_dir.glob("*_depths.npz"))
    if not npzs:
        return
    arr = np.load(npzs[0])["depths"]
    for i, d in enumerate(arr):
        f = depth_dir / f"pred_depth_{i:06d}.npy"
        if not f.exists():
            np.save(f, d.astype(np.float32))

def _is_invalid_inv(inv):
    return (not np.isfinite(inv).any()) or np.nanstd(inv) < 1e-4

def _invalid_depth_indices(depth_dir):
    bad = []
    for f in depth_dir.glob("pred_depth_*.npy"):
        idx = int(f.stem.split("_")[-1])
        if _is_invalid_inv(np.load(f)):
            bad.append(idx)
    return bad

def _remove_depth_tensors(depth_dir, idxs):
    for i in idxs:
        f = depth_dir / f"pred_depth_{i:06d}.npy"
        if f.exists():
            f.unlink()

def _load_depth(depth_dir, idx):
    f = depth_dir / f"pred_depth_{idx:06d}.npy"
    if not f.exists():
        return None
    inv = np.load(f).astype(np.float32)
    if _is_invalid_inv(inv):
        return None
    return DEPTH_SCALE_M / (inv + 1e-6)

# ====================== ViTPose / HaMeR ======================
sys.path.append(str(REPO_ROOT / "hamer"))
from vitpose_model import ViTPoseModel
_HAMER = None

def _get_vitpose_model(device="cuda"):
    global _HAMER
    if _HAMER is None:
        _HAMER = ViTPoseModel(device)
    return _HAMER

# ====================== wrist 检测（单手） ======================
def _wrist_from_frame(frame_bgr, depth, cpm, hand="right", kp_thr=0.3):
    nearest = depth < np.percentile(depth, 20)
    labels, n = ndimage.label(nearest)
    if n == 0:
        return None

    sizes = ndimage.sum(nearest, labels, range(1, n + 1))
    order = np.argsort(sizes)[::-1]

    for idx in order[:3]:
        mask = labels == (idx + 1)
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        pad = 10
        y0 = max(0, y0 - pad); y1 = min(frame_bgr.shape[0] - 1, y1 + pad)
        x0 = max(0, x0 - pad); x1 = min(frame_bgr.shape[1] - 1, x1 + pad)

        roi = frame_bgr[y0:y1+1, x0:x1+1]
        bbox = np.array([[0, 0, roi.shape[1]-1, roi.shape[0]-1, 1.0]])
        pose = cpm.predict_pose(roi[:, :, ::-1], [bbox])[0]
        kpts = pose["keypoints"]

        if hand == "left":
            hk = kpts[-42:-21]
        else:
            hk = kpts[-21:]

        valid = hk[:, 2] > kp_thr
        if valid.sum() < 4:
            continue

        xs = hk[valid, 0]; ys = hk[valid, 1]; cs = hk[valid, 2]
        cx = float((xs * cs).sum() / cs.sum())
        cy = float((ys * cs).sum() / cs.sum())
        return (x0 + cx, y0 + cy)

    return None

# ====================== 深度鲁棒 ======================
def robust_depth_at(depth, u, v, prev_z=None, win=3, max_jump=0.1):
    H, W = depth.shape
    ui, vi = int(round(u)), int(round(v))
    x0 = max(0, ui-win); x1 = min(W, ui+win+1)
    y0 = max(0, vi-win); y1 = min(H, vi+win+1)
    patch = depth[y0:y1, x0:x1].flatten()
    patch = patch[np.isfinite(patch)]
    patch = patch[patch > 1e-3]
    if patch.size == 0:
        return prev_z
    z = float(np.median(patch))
    if prev_z is not None and abs(z - prev_z) > max_jump:
        return prev_z
    return z

def _pixel_to_camera(u, v, z_mm, W, H):
    fx = fy = max(W, H)
    cx, cy = W / 2, H / 2
    X = (u - cx) * z_mm / fx
    Y = (v - cy) * z_mm / fy
    return X, Y, z_mm

# ====================== 主函数 ======================
def extract_3d_speed_onehand(video_path, output_dir, hand="right"):
    assert hand in ("left", "right")

    output_dir = Path(output_dir)
    _ensure_dir(output_dir)
    video_name = Path(video_path).stem

    depth_dir = output_dir / "depth"
    _ensure_dir(depth_dir)

    if not list(depth_dir.glob("pred_depth_*.npy")):
        generate_depth_video_vda(video_path, depth_dir)
        _unpack_depth_npz(depth_dir)

    for _ in range(MAX_REPAIR):
        bad = _invalid_depth_indices(depth_dir)
        if not bad:
            break
        _remove_depth_tensors(depth_dir, bad)
        generate_depth_video_vda(video_path, depth_dir)
        _unpack_depth_npz(depth_dir)

    cpm = _get_vitpose_model()
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cam_track = {}
    speed_pairs = []
    prev_xyz = None
    prev_z = None

    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            speed_pairs.append([idx, 0.0])
            prev_xyz = None
            continue

        depth = _load_depth(depth_dir, idx)
        if depth is None:
            speed_pairs.append([idx, 0.0])
            prev_xyz = None
            continue

        wrist = _wrist_from_frame(frame, depth, cpm, hand=hand)
        if wrist is None:
            speed_pairs.append([idx, 0.0])
            prev_xyz = None
            continue

        u, v = wrist
        z = robust_depth_at(depth, u, v, prev_z)
        prev_z = z
        H, W = depth.shape
        X, Y, Z = _pixel_to_camera(u, v, z * 1000, W, H)

        cam_track[str(idx+1)] = [X, Y, Z]

        if prev_xyz is None:
            vmag = 0.0
        else:
            vmag = float(np.linalg.norm(np.array([X, Y, Z]) - np.array(prev_xyz)))
        prev_xyz = (X, Y, Z)
        speed_pairs.append([idx, vmag])

    cap.release()

    out_json = output_dir / f"{video_name}_with_speed_onehand_{hand}.json"
    _atomic_json_dump(out_json, speed_pairs)

    plt.figure(figsize=(12,4))
    plt.plot([x[0] for x in speed_pairs], [x[1] for x in speed_pairs])
    plt.xlabel("Frame"); plt.ylabel("Speed")
    plt.tight_layout()
    out_png = output_dir / f"{video_name}_speed_onehand_{hand}.png"
    plt.savefig(out_png); plt.close()

    print(f"[DONE] {video_name} ({hand})")
    print(f"  JSON: {out_json}")
    print(f"  PNG : {out_png}")

def batch_process_videos_onehand(video_folder: str, output_root: str, hand="right"):
    video_folder = Path(video_folder)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    videos = [p for p in video_folder.iterdir() if p.is_file() and p.suffix.lower() in exts]

    if not videos:
        print(f"[WARN] 未找到视频文件: {video_folder}")
        return

    print(f"[INFO] 发现 {len(videos)} 个视频，开始处理（hand={hand}）...\n")

    success, failed = [], []
    for vp in sorted(videos):
        video_name = vp.stem
        outdir = output_root / video_name
        print(f"[RUN] {video_name}")

        try:
            extract_3d_speed_onehand(
                video_path=str(vp),
                output_dir=str(outdir),
                hand=hand
            )
            success.append(video_name)
        except Exception as e:
            print(f"[FAIL] {video_name}: {e}")
            failed.append((video_name, str(e)))

    print("\n====== 处理结果汇总 ======")
    print(f"成功 {len(success)} 个: {success}")
    print(f"失败 {len(failed)} 个:")
    for n, err in failed:
        print(f"  - {n}: {err}")


# ====================== CLI ======================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="单个视频路径（可选）")
    ap.add_argument("--video_folder", default=None, help="视频文件夹路径（批处理）")
    ap.add_argument("--out", required=True, help="输出根目录")
    ap.add_argument("--hand", default="right", choices=["left", "right"])
    args = ap.parse_args()

    if args.video_folder is not None:
        batch_process_videos_onehand(
            video_folder=args.video_folder,
            output_root=args.out,
            hand=args.hand
        )
    elif args.video is not None:
        extract_3d_speed_onehand(
            video_path=args.video,
            output_dir=args.out,
            hand=args.hand
        )
    else:
        raise ValueError("必须提供 --video 或 --video_folder 其中之一")
