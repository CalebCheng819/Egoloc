# egoloc_speed.py
from __future__ import annotations
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

# （可选）调试输出，确认生效
# print("PYTHONPATH =", os.environ["PYTHONPATH"])
# print("sys.path head =", sys.path[:5])


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

# ---- 约定 hooks 的键名（必须全部提供）----
REQUIRED_HOOKS = [
    "_get_vitpose_model",
    "generate_depth_video_vda",
    "_invalid_depth_indices",
    "_remove_depth_tensors",
    "_load_depth",
    "_wrist_from_frame",
    "_pixel_to_camera",
    "_generate_pointclouds",
    "register_hand_positions",
]

Hooks = Dict[str, Callable[..., object]]

def _need(hooks: Hooks, name: str):
    if name not in hooks or not callable(hooks[name]):
        raise RuntimeError(
            f"缺少必要 hook: '{name}'。请在调用时通过 hooks={{...}} 提供对应函数。"
        )
    return hooks[name]

# ---------------------------------------------------------------------------
# HaMeR / ViTPose – created once, reused
# ---------------------------------------------------------------------------
_HAMER_CACHE: Dict[str, ViTPoseModel] = {}

def _get_vitpose_model(device: str = "cuda") -> ViTPoseModel:
    """Return a cached ViTPoseModel (no Detectron2 dependency)."""
    if "cpm" in _HAMER_CACHE:
        return _HAMER_CACHE["cpm"]

    import hamer.vitpose_model as _vpm  # local import avoids side‑effects if unused

    _vpm.ROOT_DIR = "./hamer"
    _vpm.VIT_DIR =  "./hamer/third-party/ViTPose"

    cfg_rel = Path(
        "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/"
        "ViTPose_huge_wholebody_256x192.py"
    )
    ckpt_rel = Path("_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth")

    for _name, _dic in _vpm.ViTPoseModel.MODEL_DICT.items():
        _dic["config"] = str(Path(_vpm.VIT_DIR) / cfg_rel)
        _dic["model"] = str(Path(_vpm.ROOT_DIR) / ckpt_rel)

    _HAMER_CACHE["cpm"] = ViTPoseModel(device)
    return _HAMER_CACHE["cpm"]


# ---------------------------------------------------------------------------
# Depth video generation via Video‑Depth‑Anything
# ---------------------------------------------------------------------------
def generate_depth_video_vda(video_path: str, depth_out_path: str, *, device: str = "cuda", encoder: str = "vits") -> Path:
    """Run Video‑Depth‑Anything once and save the raw depth video."""
    video_path = Path(video_path).resolve()
    depth_out_path = Path(depth_out_path).resolve()
    depth_out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "run.py",
        "--input_video",
        str(video_path),
        "--output_dir",
        str(depth_out_path),
        "--encoder",
        encoder,  # or "vitl" if you want the large model
        "--save_npz",  # save per‑frame tensors
    ]
    if device == "cpu":
        cmd.append("--fp32")  # avoids half‑precision on CPU

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{VDA_DIR}:{env.get('PYTHONPATH', '')}"

    print("[VDA] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=VDA_DIR, env=env)

    # Unpack & sanity‑check
    _unpack_depth_npz(depth_out_path)
    if not any(depth_out_path.glob("pred_depth_*.npy")):
        raise RuntimeError(
            "[VDA] No pred_depth_*.npy tensors were produced – check the VDA output above."
        )

    return depth_out_path


# ---------------------------------------------------------------------------
#                     DEPTH-TENSOR REPAIR HELPERS
# ---------------------------------------------------------------------------
def _invalid_depth_indices(depth_dir: Path) -> List[int]:
    """Return a list of frame indices whose depth tensors are unusable."""
    bad_idx = []
    for f in depth_dir.glob("pred_depth_*.npy"):
        idx = int(f.stem.split("_")[-1])
        inv = np.load(f, mmap_mode="r")
        if _is_invalid_inv(inv):
            bad_idx.append(idx)
    return bad_idx

def _remove_depth_tensors(depth_dir: Path, indices: List[int]) -> None:
    """Delete pred_depth_XXXXXX.npy for the given indices (if they exist)."""
    for idx in indices:
        f = depth_dir / f"pred_depth_{idx:06d}.npy"
        if f.exists():
            f.unlink()


# -------------------------------------------------------------------------
# Depth loading helper
# ---------------------------------------------------------------------------
def _load_depth(depth_dir: Path, idx: int) -> Optional[np.ndarray]:
    """
    VDA stores **inverse depth** (bigger = nearer).
    Convert to metric depth in metres and keep a useful range.
    """
    f = depth_dir / f"pred_depth_{idx:06d}.npy"
    if not f.exists():
        return None
    inv = np.load(f).astype(np.float32)          # (H, W)
    if _is_invalid_inv(inv):                     # ← early-reject unusable tensor
        return None
    depth = DEPTH_SCALE_M / (inv + 1e-6)         # invert once, not twice
    return depth


# ---------------------------------------------------------------------------
# Simple camera projection helper
# ---------------------------------------------------------------------------
def _pixel_to_camera(u: float, v: float, z: float, W: int, H: int):
    fx = fy = max(W, H)
    cx, cy = W / 2.0, H / 2.0
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return X, Y, z


# ---------------------------------------------------------------------------
# Wrist detection helper (depth‑guided + ViTPose)
# ---------------------------------------------------------------------------
def _wrist_from_frame(frame_bgr: np.ndarray, gray_depth: np.ndarray, cpm: ViTPoseModel):
    # 1) Depth‑based hand ROI – nearest object in view
    nearest = gray_depth < np.percentile(gray_depth, 15)  # closest 25 %,原来是10
    labels, n_lbl = ndimage.label(nearest)
    if n_lbl == 0:
        return None

    # largest blob → hand / forearm
    sizes = ndimage.sum(nearest, labels, range(1, n_lbl + 1))
    hand_lbl = 1 + int(np.argmax(sizes))
    mask = labels == hand_lbl
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    roi_bgr = frame_bgr[y0 : y1 + 1, x0 : x1 + 1]

    # 2) ViTPose inside ROI
    bbox = np.array(
        [[0, 0, roi_bgr.shape[1] - 1, roi_bgr.shape[0] - 1, 1.0]], dtype=np.float32
    )
    pose = cpm.predict_pose(roi_bgr[:, :, ::-1], [bbox])[0]

    hand_kpts = pose["keypoints"][-21:]  # right‑hand keypoints
    valid = hand_kpts[:, 2] > 0.3#原来是0.35
    if valid.sum() <= 3:
        return None

    wrist_u = x0 + hand_kpts[0, 0]
    wrist_v = y0 + hand_kpts[0, 1]
    return float(wrist_u), float(wrist_v)


# ---------------------------------------------------------------------------
# Point Cloud Generation
# ---------------------------------------------------------------------------
def _generate_pointclouds(depth_dir: Path,  video_path: str, pcd_out_dir: Path, intrinsics: Optional[Tuple[float,float,float,float]] = None):
    """
    Create a coloured .ply point cloud for every frame whose depth tensor exists.
      • depth_dir   : folder with pred_depth_000000.npy … (metres)
      • video_path  : original RGB video (to grab colours)
      • pcd_out_dir : output <idx>.ply files
    """
    pcd_out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if intrinsics is None:
        fx = fy = max(H, W)          # <-- quick default; replace with calibrated fx,fy
        cx, cy = W / 2.0, H / 2.0
    else:
        fx, fy, cx, cy = intrinsics

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(W, H, fx, fy, cx, cy)

    depth_files = sorted(depth_dir.glob("pred_depth_*.npy"))
    for dfile in depth_files:
        idx = int(dfile.stem.split("_")[-1])
        # grab colour frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        depth_m = _load_depth(depth_dir, idx)      # **same     depth   maths** everywhere
        if depth_m is None:
            continue

        # Open3D expects depth in millimetres by default (depth_scale=1000)
        depth_o3d = o3d.geometry.Image((depth_m * 1000).astype(np.uint16))
        color_o3d = o3d.geometry.Image(frame_rgb)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1000.0,
            depth_trunc=4.0, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        # flip to keep +Z forward, +Y up (matches your earlier flip in register func)
#       pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) # < --------------------------
        o3d.io.write_point_cloud(str(pcd_out_dir / f"{idx}.ply"), pcd, write_ascii=False)
    cap.release()
    print(f"[PCD] Generated {len(depth_files)} point clouds in {pcd_out_dir}")


# ---------------------------------------------------------------------------
# Hand Position Registration With ICP and Frame 0 Alignment Helper
# ---------------------------------------------------------------------------
def register_hand_positions(pcd_root, hand3d_root, save_reg_root, threshold=0.03):
    """
    Align per-frame 3-D hand positions to the first frame using point-to-point
    ICP.

    Args:
        pcd_root (str): Directory containing colored point clouds organised as
            `<video>/frame.ply.
        hand3d_root (str): Directory with camera-coordinate 3-D hand JSON files.
        save_reg_root (str): Output directory where globally registered hand
            trajectories will be saved.
        threshold (float, optional): ICP correspondence distance threshold in
            metres. Defaults to 0.03.

    Returns:
        None
    """
    os.makedirs(save_reg_root, exist_ok=True)                              # ensure output dir exists
    for video in sorted(os.listdir(pcd_root)):                             # iterate each video folder
        pcd_dir = os.path.join(pcd_root, video)                            # path to this video’s PLYs
        hand3d_path = os.path.join(hand3d_root, f"{video}.json")           # path to camera-frame hand JSON
        if not os.path.isdir(pcd_dir) or not os.path.isfile(hand3d_path):  # skip if either missing
            continue
        hand3d = json.load(open(hand3d_path))                              # load camera-frame keypoints
        plys = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.ply')],
                      key=lambda x: int(os.path.splitext(x)[0]))           # list PLYs in order

        first_pcd = None                                                   # reference cloud (frame 1)
        odoms = []                                                         # unused – kept from orig code
        reg_hand_dict = {}                                                 # output dict: frameID → (x,y,z)
        prev_pcd = None                                                    # helps store first_pcd

        for i, ply in enumerate(plys):                                     # walk over every cloud
            frame_id = str(i + 1)                                          # JSON is 1-based indexing
            pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, ply))      # load current cloud
            pts = np.asarray(pcd.points)                                   # numpy view of XYZ
            pcd.points = o3d.utility.Vector3dVector(pts)                   # write back to Open3D cloud

            if i == 0:                                                     # first frame: no registration
                odoms.append(np.eye(4))                                    # placeholder (unused)
                if frame_id in hand3d:                                     # store original hand if exists
                    h0 = np.array(hand3d[frame_id])                        # camera-frame wrist point
                    reg_hand_dict[frame_id] = h0.tolist()                  # frame 1 becomes origin
            else:
                if first_pcd is None:                                      # cache reference once
                    first_pcd = prev_pcd

                reg = o3d.pipelines.registration.registration_icp(         # ICP: current → first
                    pcd,                                                   # source cloud
                    first_pcd,                                             # target cloud
                    threshold,                                             # correspondence distance (m)
                    np.eye(4),                                             # initial guess = identity
                    o3d.pipelines.registration.
                        TransformationEstimationPointToPoint()
                )
                T = reg.transformation                                     # 4×4 rigid transform

                h = np.array(hand3d.get(str(i + 1), [np.nan, np.nan, np.nan]))  # wrist in camera frame
                h4 = np.append(h, 1)                                       # homogeneous coordinate
                h_reg = (T @ h4)[:3].tolist()                              # apply ICP transform
                if frame_id in hand3d:                                     # store if key exists
                    reg_hand_dict[frame_id] = h_reg

            prev_pcd = pcd                                                 # keep for first_pcd assignment

        # save globally registered trajectory for this video
        with open(os.path.join(save_reg_root, f"{video}.json"), 'w') as f:
            json.dump(reg_hand_dict, f, indent=4)
        print(f"Computed globally registered 3-D hand trajectory for {video}")

#修补处理过程遇到的问题
def _ensure_dir(p) -> None:
    """Create directory p if not exists (accepts str or Path)."""
    Path(p).mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    p
    for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents)
    if (p / "Video-Depth-Anything").exists()
)
VDA_DIR = REPO_ROOT / "Video-Depth-Anything"

DEPTH_SCALE_M = 3.0  # pixel value 255 ↔ 3 m (linear scaling)
MAX_FEEDBACKS = 1


# ---------------------------------------------------------------------------
# Helper – unpack Video‑Depth‑Anything *_depths.npz → per‑frame .npy
# ---------------------------------------------------------------------------
def _unpack_depth_npz(depth_dir: Path) -> None:
    """Convert VDA’s *_depths.npz to individual .npy, skipping ones that exist."""
    npz_files = list(depth_dir.glob("*_depths.npz"))
    if not npz_files:
        return  # nothing to unpack

    arr = np.load(npz_files[0])["depths"]  # (N, H, W)
    created = 0
    for i, depth in enumerate(arr):
        out_f = depth_dir / f"pred_depth_{i:06d}.npy"
        if out_f.exists():          # keep the good tensor we already have
            continue
        np.save(out_f, depth.astype(np.float32))
        created += 1
    if created:
        print(f"[VDA] Unpacked {created} new tensors")


# ---------------------------------------------------------------------------
# Depth-quality helpers
# ---------------------------------------------------------------------------
def _is_invalid_inv(inv: np.ndarray) -> bool:
    """
    Return True when the inverse-depth tensor is all-NaN/Inf or nearly flat.
    """
    return (not np.isfinite(inv).any()) or np.nanstd(inv) < 1e-4

# def extract_3d_speed_and_visualize(
#     video_path: str,
#     output_dir: str,
#     *,
#     device: str = "cuda",
#     encoder: str = "vits",
#     hooks: Hooks,
# ) -> Tuple[List[List[float]], str, str, str]:
#     """
#     计算 3D 手速并产出:
#       - speed_pairs: [[frame, speed], ...]   （与 *_with_speed.json 的格式一致）
#       - speed_json_path: <output_dir>/<video_name>_with_speed.json
#       - speed_vis_path:  <output_dir>/<video_name>_speed_vis.png
#       - depth_vis_path:  <output_dir>/depth/depth_vis.mp4
#
#     依赖通过 hooks 提供，必须包含：
#       _get_vitpose_model, generate_depth_video_vda, _invalid_depth_indices,
#       _remove_depth_tensors, _load_depth, _wrist_from_frame, _pixel_to_camera,
#       _generate_pointclouds, register_hand_positions
#     """
#     # 取出 hook
#     _get_vitpose_model   = _need(hooks, "_get_vitpose_model")
#     generate_depth_video_vda = _need(hooks, "generate_depth_video_vda")
#     _invalid_depth_indices   = _need(hooks, "_invalid_depth_indices")
#     _remove_depth_tensors    = _need(hooks, "_remove_depth_tensors")
#     _load_depth              = _need(hooks, "_load_depth")
#     _wrist_from_frame        = _need(hooks, "_wrist_from_frame")
#     _pixel_to_camera         = _need(hooks, "_pixel_to_camera")
#     _generate_pointclouds    = _need(hooks, "_generate_pointclouds")
#     register_hand_positions  = _need(hooks, "register_hand_positions")
#
#     cpm = _get_vitpose_model(device)
#
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     video_name = Path(video_path).stem
#
#     depth_dir = output_dir / "depth"
#     depth_vis_path = depth_dir / "depth_vis.mp4"
#     undet_dir = output_dir / "undetected_frames"
#     undet_dir.mkdir(parents=True, exist_ok=True)
#
#     vda_ready = (depth_dir / "pred_depth_000000.npy").exists()
#     if (not depth_vis_path.exists()) or (not vda_ready):
#         log.info("[3D] Generating depth (tensors + video)…")
#         depth_dir.mkdir(parents=True, exist_ok=True)
#         generate_depth_video_vda(video_path, depth_dir, device=device, encoder=encoder)
#     else:
#         log.info("[3D] Reusing cached depth in %s", depth_dir)
#
#     # ---- 深度质量检查 + 自动修复 ----
#     max_repairs = 5
#     for attempt in range(max_repairs):
#         bad_idx = _invalid_depth_indices(depth_dir)
#         if not bad_idx:
#             break
#         total = len(list(depth_dir.glob("pred_depth_*.npy"))) or 1
#         pct = len(bad_idx) / total * 100
#         log.warning("[depth] %d tensors invalid (%.1f%%) – repairing (%d/%d)",
#                     len(bad_idx), pct, attempt + 1, max_repairs)
#         _remove_depth_tensors(depth_dir, bad_idx)
#         generate_depth_video_vda(video_path, depth_dir, device=device, encoder=encoder)
#     else:
#         raise RuntimeError(f"Depth repair failed after {max_repairs} attempts")
#
#     # ---- 点云构建 ----
#     pcd_dir = output_dir / "pointclouds" / video_name
#     pcd_dir.mkdir(parents=True, exist_ok=True)
#     if not (pcd_dir / "0.ply").exists():
#         log.info("[PCD] Building point clouds…")
#         _generate_pointclouds(depth_dir, video_path, pcd_dir)
#     else:
#         log.info("[PCD] Reusing cached point clouds in %s", pcd_dir)
#
#     # ---- 相机系手腕坐标 & 注册到世界系 ----
#     cam_hand_dir = output_dir / "hand3d_cam"
#     cam_hand_dir.mkdir(parents=True, exist_ok=True)
#     cam_hand_json = cam_hand_dir / f"{video_name}.json"
#
#     cap_rgb = cv2.VideoCapture(video_path)
#     if not cap_rgb.isOpened():
#         raise RuntimeError(f"Could not open RGB video: {video_path}")
#
#     total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
#     cam_hand: Dict[str, List[float]] = {}
#     prev_xyz_cam = None
#
#     for idx in range(total_frames):
#         ok_rgb, frame_bgr = cap_rgb.read()
#         if not ok_rgb:
#             prev_xyz_cam = None
#             continue
#
#         gray_depth = _load_depth(depth_dir, idx)
#         if gray_depth is None:
#             prev_xyz_cam = None
#             # 可选：cv2.imwrite(str(undet_dir / f"{video_name}_DEPTHMISS_{idx:06d}.png"), frame_bgr)
#             continue
#
#         H, W = gray_depth.shape
#         wrist = hooks["_wrist_from_frame"](frame_bgr, gray_depth, cpm)
#         if wrist is None:
#             prev_xyz_cam = None
#             # 可选：cv2.imwrite(str(undet_dir / f"{video_name}_NOWRIST_{idx:06d}.png"), frame_bgr)
#             continue
#
#         u, v = wrist
#         u_i, v_i = min(max(int(u), 0), W - 1), min(max(int(v), 0), H - 1)
#         z = float(gray_depth[v_i, u_i]) * 1000.0  # m->mm（按你原意）
#         X, Y, Z = _pixel_to_camera(u, v, z, W, H)
#         cam_hand[str(idx + 1)] = [float(X), float(Y), float(Z)]
#
#         if (idx + 1) % 100 == 0 or idx == total_frames - 1:
#             log.info("[3D] Processed %d/%d frames", idx + 1, total_frames)
#
#     cap_rgb.release()
#     _atomic_json_dump(cam_hand_json, cam_hand)
#
#     reg_out_dir = output_dir / "registered_hands"
#     reg_out_dir.mkdir(parents=True, exist_ok=True)
#     register_hand_positions(str(pcd_dir.parent), str(cam_hand_dir), str(reg_out_dir))
#
#     reg_json = reg_out_dir / f"{video_name}.json"
#     if not reg_json.exists():
#         raise RuntimeError(f"Registration output missing: {reg_json}")
#     reg_hand = json.loads(reg_json.read_text(encoding="utf-8"))
#
#     # ---- 世界系速度（与下游对齐：[[frame, speed], ...] & *_with_speed.json）----
#     speed_pairs: List[List[float]] = []
#     prev_xyz_world = None
#     for idx in range(total_frames):
#         xyz = reg_hand.get(str(idx + 1), reg_hand.get(idx + 1, None))
#         if xyz is None:
#             speed = 0.0
#             prev_xyz_world = None
#         else:
#             if prev_xyz_world is None:
#                 speed = 0.0
#             else:
#                 dx, dy, dz = np.array(xyz, dtype=float) - np.array(prev_xyz_world, dtype=float)
#                 speed = float(np.linalg.norm([dx, dy, dz]))
#             prev_xyz_world = xyz
#         speed_pairs.append([int(idx), float(speed)])
#
#     speed_json_path = output_dir / f"{video_name}_with_speed.json"
#     _atomic_json_dump(speed_json_path, speed_pairs)
#
#     # 可视化
#     xs = [p[0] for p in speed_pairs]; ys = [p[1] for p in speed_pairs]
#     plt.figure(figsize=(12, 4))
#     plt.plot(xs, ys, label="3-D Hand Speed (world)")
#     plt.xlabel("Frame"); plt.ylabel("Speed (relative)")
#     plt.tight_layout()
#     speed_vis_path = output_dir / f"{video_name}_speed_vis.png"
#     plt.savefig(speed_vis_path); plt.close()
#
#     return speed_pairs, str(speed_json_path), str(speed_vis_path), str(depth_vis_path)
def extract_3d_speed_and_visualize(video_path: str, output_dir: str, *, device: str = "cuda", encoder: str = "vits") -> Tuple[List[List[float]], str, str, str]:
    """
    返回：
      - speed_pairs: [[frame, speed], ...]
      - speed_json_path: <output_dir>/<video>_with_speed.json
      - speed_vis_path:  <output_dir>/<video>_speed_vis.png
      - depth_vis_path:  <output_dir>/depth/depth_vis.mp4
    """
    cpm = _get_vitpose_model(device)#必须选择·导入
    output_dir = Path(output_dir); _ensure_dir(output_dir)
    video_name = Path(video_path).stem

    depth_dir = output_dir / "depth"
    depth_vis_path = depth_dir / "depth_vis.mp4"
    undet_dir = output_dir / "undetected_frames"; _ensure_dir(undet_dir)

    # 1) 深度
    vda_ready = (depth_dir / "pred_depth_000000.npy").exists()
    if (not depth_vis_path.exists()) or (not vda_ready):
        log.info("[3D] Generating/Checking depth …")
        generate_depth_video_vda(video_path, depth_dir, device=device, encoder=encoder)
    else:
        log.info("[3D] Reusing cached depth in %s", depth_dir)

    # 1.5) 自检 + 修复
    max_repairs = 5
    for attempt in range(max_repairs):
        bad = _invalid_depth_indices(depth_dir)
        if not bad:
            break
        total = len(_list_depth_tensors(depth_dir)) or 1
        log.warning("[depth] %d invalid (%.1f%%) – repairing (%d/%d)", len(bad), len(bad)/total*100, attempt+1, max_repairs)
        _remove_depth_tensors(depth_dir, bad)
        generate_depth_video_vda(video_path, depth_dir, device=device, encoder=encoder)
    else:
        raise RuntimeError(f"Depth repair failed after {max_repairs} attempts")

    # 2) 点云
    pcd_dir = output_dir / "pointclouds" / video_name
    if not (pcd_dir / "0.ply").exists():
        log.info("[PCD] Building …")
        _generate_pointclouds(depth_dir, video_path, pcd_dir)
    else:
        log.info("[PCD] Reusing %s", pcd_dir)

    # 3) 相机系手腕
    cam_hand_dir = output_dir / "hand3d_cam"; _ensure_dir(cam_hand_dir)
    cam_hand_json = cam_hand_dir / f"{video_name}.json"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cam_track: Dict[str, List[float]] = {}
    prev_cam = None
    speed_cam: Dict[int, float] = {}

    for idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            speed_cam[idx] = 0.0
            prev_cam = None
            continue
        depth = _load_depth(depth_dir, idx)
        if depth is None:
            speed_cam[idx] = 0.0
            prev_cam = None
            continue

        H, W = depth.shape
        wrist = _wrist_from_frame(frame, depth, cpm)
        if wrist is None:
            speed_cam[idx] = 0.0
            prev_cam = None
            continue

        u, v = wrist
        ui = min(max(int(u), 0), W-1)
        vi = min(max(int(v), 0), H-1)
        z_mm = float(depth[vi, ui]) * 1000.0
        X, Y, Z = _pixel_to_camera(u, v, z_mm, W, H)
        cam_track[str(idx+1)] = [float(X), float(Y), float(Z)]

        if prev_cam is None:
            sp = 0.0
        else:
            dX, dY, dZ = X - prev_cam[0], Y - prev_cam[1], Z - prev_cam[2]
            sp = math.sqrt(dX*dX + dY*dY + dZ*dZ)
        speed_cam[idx] = float(sp)
        prev_cam = (X, Y, Z)

        if (idx+1) % 100 == 0 or idx == total_frames-1:
            log.info("[3D] Frames %d/%d", idx+1, total_frames)

    cap.release()
    _atomic_json_dump(cam_hand_json, cam_track)

    # 4) 注册到世界坐标
    reg_dir = output_dir / "registered_hands"; _ensure_dir(reg_dir)
    register_hand_positions(str(pcd_dir), str(cam_hand_dir), str(reg_dir))
    reg_json = reg_dir / f"{video_name}.json"
    if not reg_json.exists():
        log.warning("[REG] Registration output missing; using camera-frame track.")
        reg_track = cam_track
    else:
        reg_track = json.loads(reg_json.read_text(encoding="utf-8"))

    # 5) 世界系速度
    speed_pairs: List[List[float]] = []
    prev_w = None
    for idx in range(total_frames):
        xyz = reg_track.get(str(idx+1))
        if xyz is None:
            speed = 0.0
            prev_w = None
        else:
            if prev_w is None:
                speed = 0.0
            else:
                dx, dy, dz = np.array(xyz) - np.array(prev_w)
                speed = float(np.linalg.norm([dx,dy,dz]))
            prev_w = xyz
        speed_pairs.append([idx, speed])

    # 6) 输出
    speed_json_path = output_dir / f"{video_name}_with_speed.json"
    _atomic_json_dump(speed_json_path, speed_pairs)

    plt.figure(figsize=(12, 4))
    xs = [p[0] for p in speed_pairs]
    ys = [p[1] for p in speed_pairs]
    plt.plot(xs, ys, label="3D Hand Speed (world)")
    plt.xlabel("Frame"); plt.ylabel("Speed (relative)")
    plt.tight_layout()
    speed_vis_path = output_dir / f"{video_name}_speed_vis.png"
    plt.savefig(speed_vis_path); plt.close()

    return speed_pairs, str(speed_json_path), str(speed_vis_path), str(depth_vis_path)

def batch_process_videos(video_folder: str, output_root: str, device="cuda", encoder="vits"):

    video_folder = Path(video_folder)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 支持的格式
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [p for p in video_folder.iterdir() if p.suffix.lower() in exts]

    if not videos:
        print(f" 未找到视频文件: {video_folder}")
        return

    print(f" 发现 {len(videos)} 个视频，开始处理...\n")

    success, failed = [], []

    for video_path in videos:
        video_name = video_path.stem
        outdir = output_root / video_name
        print(f" 处理视频: {video_name}")

        try:
            pairs, speed_json, speed_png, depth_vis = extract_3d_speed_and_visualize(
                video_path=str(video_path),
                output_dir=str(outdir),
                device=device,
                encoder=encoder
            )
            print(f" 完成: {video_name}")
            print(f"   ├─ 速度JSON: {speed_json}")
            print(f"   ├─ 速度图:   {speed_png}")
            print(f"   └─ 深度视频: {depth_vis}\n")
            success.append(video_name)

        except Exception as e:
            print(f" {video_name} 失败: {e}\n")
            failed.append(video_name)

    print("\n======  处理结果汇总 ======")
    print(f" 成功 {len(success)} 个: {success}")
    print(f" 失败 {len(failed)} 个: {failed}")

