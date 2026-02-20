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
_HAMER_CACHE: Dict[str, Any] = {}

def _get_vitpose_model(device: str = "cuda") -> ViTPoseModel:
    """Return a cached ViTPoseModel."""
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


def _get_body_detector(device: str = "cuda"):
    """Return a cached Detectron2 body detector (HaMeR style)."""
    if "detector" in _HAMER_CACHE:
        return _HAMER_CACHE["detector"]
    
    try:
        # 确保 hamer 包在 sys.path 中
        import sys
        import importlib.util
        
        # 尝试多种方式导入
        try:
            # 方式1: 标准导入（如果 hamer 已安装）
            from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        except ImportError:
            # 方式2: 直接导入文件
            utils_detectron2_path = Path(HAMER_ROOT) / "hamer" / "utils" / "utils_detectron2.py"
            if utils_detectron2_path.exists():
                spec = importlib.util.spec_from_file_location("hamer.utils.utils_detectron2", utils_detectron2_path)
                utils_detectron2_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(utils_detectron2_module)
                DefaultPredictor_Lazy = utils_detectron2_module.DefaultPredictor_Lazy
            else:
                raise ImportError(f"Cannot find utils_detectron2.py at {utils_detectron2_path}")
        
        from detectron2.config import LazyConfig
        import hamer
        
        # 使用 regnety 检测器（更快，内存占用更少）
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            detectron2_cfg = model_zoo.get_config(
                'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', 
                trained=True
            )
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            detector = DefaultPredictor_Lazy(detectron2_cfg)
            log.info("[DET] Using RegNetY detector (faster)")
        except Exception as e:
            # 回退到 vitdet
            log.info("[DET] RegNetY not available, using ViTDet")
            cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            # 尝试使用本地模型路径
            local_model = "/home/hamer/models/model_final_f05665.pkl"
            if os.path.exists(local_model):
                detectron2_cfg.train.init_checkpoint = local_model
            else:
                detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            detector = DefaultPredictor_Lazy(detectron2_cfg)
        
        _HAMER_CACHE["detector"] = detector
        return detector
    except Exception as e:
        log.warning(f"[DET] Failed to load Detectron2 detector: {e}")
        log.warning("[DET] Falling back to depth-guided method")
        return None


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
        depth = np.load(f, mmap_mode="r")
        # 使用 metric depth 检查函数
        if _is_invalid_depth(depth):
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
    加载 metric depth（单位：米）。
    
    NOTE: MoGe2 输出的是 metric depth，直接返回，不做逆深度转换。
    如果将来需要使用 VDA 的逆深度，需要修改此函数。
    """
    f = depth_dir / f"pred_depth_{idx:06d}.npy"
    if not f.exists():
        return None
    depth = np.load(f).astype(np.float32)        # (H, W) metric depth in metres
    if _is_invalid_depth(depth):                 # ← early-reject unusable tensor
        return None
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
def _wrist_from_frame(frame_bgr: np.ndarray, gray_depth: np.ndarray, cpm: ViTPoseModel, detector=None):
    """
    HaMeR 风格的手腕检测：
    1. 使用 Detectron2 检测人体
    2. 在全图使用 ViTPose 检测人体关键点
    3. 从人体关键点中提取手部关键点
    4. 返回手腕位置
    """
    # 如果没有提供检测器，尝试获取
    if detector is None:
        detector = _get_body_detector()
    
    # 如果检测器不可用，回退到深度引导方法
    if detector is None:
        return _wrist_from_frame_depth_guided(frame_bgr, gray_depth, cpm)
    
    try:
        # 1) 使用 Detectron2 检测人体
        det_out = detector(frame_bgr)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        
        if len(pred_bboxes) == 0:
            return None
        
        # 2) 在全图使用 ViTPose 检测人体关键点
        img_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
        vitposes_out = cpm.predict_pose(
            img_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        
        if len(vitposes_out) == 0:
            return None
        
        # 3) 从人体关键点中提取手部关键点（使用第一个检测到的人）
        vitposes = vitposes_out[0]
        right_hand_keyp = vitposes['keypoints'][-21:]  # 右手21个关键点
        
        # 4) 检查手部关键点置信度
        valid = right_hand_keyp[:, 2] > 0.5  # 置信度阈值 0.5（HaMeR 使用 0.5）
        if valid.sum() <= 3:
            return None
        
        # 5) 返回手腕位置（第0个关键点）
        wrist_u = float(right_hand_keyp[0, 0])
        wrist_v = float(right_hand_keyp[0, 1])
        return wrist_u, wrist_v
        
    except Exception as e:
        log.warning(f"[WRIST] HaMeR detection failed: {e}, falling back to depth-guided")
        return _wrist_from_frame_depth_guided(frame_bgr, gray_depth, cpm)


def _wrist_from_frame_depth_guided(frame_bgr: np.ndarray, gray_depth: np.ndarray, cpm: ViTPoseModel):
    """
    原始的深度引导方法（作为回退方案）
    """
    # 1) Depth‑based hand ROI – nearest object in view
    nearest = gray_depth < np.percentile(gray_depth, 15)  # closest 15%
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
    valid = hand_kpts[:, 2] > 0.3
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
    NOTE: 现在用于 metric depth，检查逻辑仍然适用（NaN/Inf/平坦检查）。
    """
    return (not np.isfinite(inv).any()) or np.nanstd(inv) < 1e-4

def _is_invalid_depth(depth: np.ndarray) -> bool:
    """
    Return True when the metric depth tensor is all-NaN/Inf or nearly flat.
    用于检查 metric depth（单位：米）的有效性。
    """
    if not np.isfinite(depth).any():
        return True
    # 对于 metric depth，检查标准差是否太小（几乎平坦）
    # 也检查是否有合理的深度范围（例如 0.01m 到 10m）
    valid_mask = np.isfinite(depth) & (depth > 0.01) & (depth < 10.0)
    if valid_mask.sum() == 0:
        return True
    std_val = np.nanstd(depth[valid_mask])
    return std_val < 1e-4

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

#更鲁棒深度获取函数
def robust_depth_at(
        depth, u, v,
        prev_z=None,
        win=3,                  # 半径 3 -> 7x7 窗口
        drop_extreme_ratio=0.1, # 剔除前10%最大 + 后10%最小
        max_jump=None,          # 限制与上一帧的最大跳变
        debug=False,
        frame_idx=None,
        hand_side="?"
):
    """
    depth: (H, W) float32 深度（单位：米）
    u, v: 像素坐标（可为浮点）
    prev_z: 上一帧深度（米）
    win: 窗口半径
    drop_extreme_ratio: 丢弃窗口里最小和最大各10%的值
    max_jump: 若超过此跳变则使用 prev_z
    debug: 若 True 打印窗口信息
    """
    H, W = depth.shape
    ui = int(round(u))
    vi = int(round(v))

    # 取窗口
    x0 = max(0, ui - win)
    x1 = min(W, ui + win + 1)
    y0 = max(0, vi - win)
    y1 = min(H, vi + win + 1)

    patch = depth[y0:y1, x0:x1].astype(float)

    # 去掉 NaN / Inf / 非法值
    raw = patch.flatten()
    raw = raw[np.isfinite(raw)]
    raw = raw[raw > 1e-3]  # 丢弃 0 或很小的无意义深度

    if raw.size == 0:
        if debug:
            print(f"[depth][F{frame_idx} {hand_side}] patch empty at ({ui},{vi})")
        return None

    # 按值排序
    sorted_vals = np.sort(raw)

    # 丢掉最大/最小若干比例的值
    k = int(len(sorted_vals) * drop_extreme_ratio)
    if k > 0:
        trimmed = sorted_vals[k:-k]
    else:
        trimmed = sorted_vals

    if trimmed.size == 0:
        trimmed = sorted_vals  # 避免被剔除光

    # 中值
    z = float(np.median(trimmed))

    # 限制过大跳变
    if prev_z is not None and max_jump is not None:
        if abs(z - prev_z) > max_jump:
            if debug:
                print(f"[depth][F{frame_idx} {hand_side}] JUMP! prev={prev_z:.3f} -> z={z:.3f}, use prev")
            return prev_z

    if debug:
        print(f"\n[depth][F{frame_idx} {hand_side}]")
        print(f"  u,v=({u:.1f},{v:.1f}), window=({x0}:{x1}, {y0}:{y1})")
        print(f"  raw size={len(raw)}, raw min/max={raw.min():.3f}/{raw.max():.3f}")
        print(f"  trimmed range=({trimmed.min():.3f}, {trimmed.max():.3f}), size={len(trimmed)}")
        print(f"  median z = {z:.3f}")

    return z
def extract_3d_speed_and_visualize(video_path: str, output_dir: str, *, device: str = "cuda", encoder: str = "vits", depth_root: Optional[str] = None, speed_only: bool = False, intermediate_dir: Optional[str] = None) -> Tuple[List[List[float]], str, str, str]:
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
    prev_z = None  # 上一帧左手的深度（米）

    # 确定深度文件目录和中间文件目录
    # 如果指定了 intermediate_dir，使用它；否则根据 speed_only 决定
    if intermediate_dir is not None:
        intermediate_dir = Path(intermediate_dir)
    elif speed_only and depth_root is not None:
        intermediate_dir = Path(depth_root) / video_name
    else:
        intermediate_dir = output_dir
    
    # 确定深度文件目录：优先使用预转换的深度，否则使用输出目录
    if depth_root is not None:
        # 使用预转换的深度目录
        pre_depth_dir = Path(depth_root) / video_name / "depth"
        if pre_depth_dir.exists() and list(pre_depth_dir.glob("pred_depth_*.npy")):
            depth_dir = pre_depth_dir
            log.info("[3D] Using pre-converted depth from: %s", depth_dir)
        else:
            depth_dir = intermediate_dir / "depth"
            log.info("[3D] Pre-converted depth not found at %s, using intermediate dir", pre_depth_dir)
    else:
        depth_dir = intermediate_dir / "depth"
    
    depth_vis_path = depth_dir / "depth_vis.mp4"
    undet_dir = intermediate_dir / "undetected_frames"; _ensure_dir(undet_dir)

    # 1) 深度
    # 检查是否已有深度文件（来自 MoGe2 转换）
    depth_files_exist = list(depth_dir.glob("pred_depth_*.npy"))
    vda_ready = len(depth_files_exist) > 0
    
    if not vda_ready:
        # 如果没有深度文件，使用 VDA 生成
        log.info("[3D] No depth files found, generating with VDA …")
        generate_depth_video_vda(video_path, depth_dir, device=device, encoder=encoder)
    else:
        log.info("[3D] Found %d existing depth files in %s (from MoGe2 conversion)", 
                 len(depth_files_exist), depth_dir)

    # 1.5) 自检（但不修复，因为 MoGe2 深度已经转换好了）
    # 如果深度文件来自 MoGe2，只检查不修复
    bad = _invalid_depth_indices(depth_dir)
    if bad:
        total = len(list(depth_dir.glob("pred_depth_*.npy"))) or 1
        log.warning("[depth] %d invalid depth files (%.1f%%) found, but skipping repair "
                   "since using MoGe2 metric depth", len(bad), len(bad)/total*100)
        # 注意：不删除或重新生成，因为 MoGe2 深度已经转换好了

    # 2) 点云
    pcd_dir = intermediate_dir / "pointclouds" / video_name
    if not (pcd_dir / "0.ply").exists():
        log.info("[PCD] Building …")
        _generate_pointclouds(depth_dir, video_path, pcd_dir)
    else:
        log.info("[PCD] Reusing %s", pcd_dir)

    # 3) 相机系手腕
    cam_hand_dir = intermediate_dir / "hand3d_cam"; _ensure_dir(cam_hand_dir)
    cam_hand_json = cam_hand_dir / f"{video_name}.json"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cam_track: Dict[str, List[float]] = {}
    prev_cam = None
    speed_cam: Dict[int, float] = {}
    
    # 诊断统计
    stats = {
        "total_frames": total_frames,
        "frame_read_failed": 0,
        "depth_missing": 0,
        "wrist_detection_failed": 0,
        "depth_extraction_failed": 0,
        "success": 0
    }

    for idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            stats["frame_read_failed"] += 1
            speed_cam[idx] = 0.0
            prev_cam = None
            continue
        depth = _load_depth(depth_dir, idx)
        if depth is None:
            stats["depth_missing"] += 1
            speed_cam[idx] = 0.0
            prev_cam = None
            continue

        H, W = depth.shape
        # 使用 HaMeR 方法检测手腕（不依赖深度图）
        detector = _get_body_detector()
        wrist = _wrist_from_frame(frame, depth, cpm, detector=detector)
        if wrist is None:
            stats["wrist_detection_failed"] += 1
            speed_cam[idx] = 0.0
            prev_cam = None
            continue

        u, v = wrist
        ui = min(max(int(u), 0), W-1)
        vi = min(max(int(v), 0), H-1)
        z_m = robust_depth_at(
            depth, ui, vi,
            prev_z=prev_z,
            win=3,  # 窗口半径3 -> 7x7
            drop_extreme_ratio=0.1,  # 剔除前10%最大+最小
            max_jump=1.0,  # 每帧深度最多跳 1.0 m
            debug=False,  # 关闭调试输出以减少日志
            frame_idx=idx,
            hand_side="L"
        )
        
        # 如果深度获取失败，跳过该帧
        if z_m is None:
            stats["depth_extraction_failed"] += 1
            speed_cam[idx] = 0.0
            prev_cam = None
            continue
        
        stats["success"] += 1
        prev_z = z_m  # ★ 更新上一帧深度
        z_mm = z_m * 1000.0
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
    
    # 输出详细统计
    log.info("[DIAG] Frame processing statistics:")
    log.info("  Total frames: %d", stats["total_frames"])
    log.info("  Frame read failed: %d (%.1f%%)", 
             stats["frame_read_failed"], stats["frame_read_failed"]/stats["total_frames"]*100)
    log.info("  Depth missing: %d (%.1f%%)", 
             stats["depth_missing"], stats["depth_missing"]/stats["total_frames"]*100)
    log.info("  Wrist detection failed: %d (%.1f%%)", 
             stats["wrist_detection_failed"], stats["wrist_detection_failed"]/stats["total_frames"]*100)
    log.info("  Depth extraction failed: %d (%.1f%%)", 
             stats["depth_extraction_failed"], stats["depth_extraction_failed"]/stats["total_frames"]*100)
    log.info("  Successfully processed: %d (%.1f%%)", 
             stats["success"], stats["success"]/stats["total_frames"]*100)

    cap.release()
    _atomic_json_dump(cam_hand_json, cam_track)
    
    # 统计信息
    detected_frames = len(cam_track)
    log.info("[CAM] Camera-frame tracking: %d/%d frames (%.1f%%)", 
             detected_frames, total_frames, detected_frames/total_frames*100 if total_frames > 0 else 0)
    
    # 输出前10帧的检测状态作为示例
    if detected_frames > 0:
        sample_frames = sorted(cam_track.keys(), key=int)[:min(10, len(cam_track))]
        log.info("[CAM] Sample camera coordinates (first 10 detected frames):")
        for frame_id in sample_frames:
            xyz = cam_track[frame_id]
            log.info("  Frame %s: [%.3f, %.3f, %.3f]", frame_id, xyz[0], xyz[1], xyz[2])

    # 4) 注册到世界坐标
    reg_dir = intermediate_dir / "registered_hands"; _ensure_dir(reg_dir)
    # register_hand_positions 期望 pcd_root 包含视频子目录，所以传入 pointclouds 目录
    pcd_root = pcd_dir.parent  # pointclouds 目录
    register_hand_positions(str(pcd_root), str(cam_hand_dir), str(reg_dir))
    reg_json = reg_dir / f"{video_name}.json"
    if not reg_json.exists():
        log.warning("[REG] Registration output missing; using camera-frame track.")
        reg_track = cam_track
    else:
        reg_track = json.loads(reg_json.read_text(encoding="utf-8"))
    
    registered_frames = len(reg_track)
    log.info("[REG] Registered %d/%d frames (%.1f%%)", 
             registered_frames, total_frames, registered_frames/total_frames*100 if total_frames > 0 else 0)
    
    # 输出配准后的示例坐标
    if registered_frames > 0:
        sample_frames = sorted(reg_track.keys(), key=int)[:min(10, len(reg_track))]
        log.info("[REG] Sample registered coordinates (first 10 frames):")
        for frame_id in sample_frames:
            xyz = reg_track[frame_id]
            log.info("  Frame %s: [%.3f, %.3f, %.3f]", frame_id, xyz[0], xyz[1], xyz[2])

    # 5) 对注册后的坐标进行插值（填充缺失帧）
    def interpolate_hand_positions(hand_dict: Dict[str, List[float]], total_frames: int) -> Dict[int, List[float]]:
        """
        对缺失的手部位置进行线性插值。
        
        Args:
            hand_dict: dict mapping frame index (str) to [x, y, z]
            total_frames: 总帧数
        
        Returns:
            dict mapping frame index (int) to [x, y, z]，所有帧都有值（通过插值填充）
        """
        if not hand_dict:
            return {}
        
        # 提取有效帧和坐标
        valid_frames = []
        valid_coords = []
        for frame_str, coords in hand_dict.items():
            try:
                frame_idx = int(frame_str)
                if 0 <= frame_idx <= total_frames and isinstance(coords, (list, tuple)) and len(coords) >= 3:
                    valid_frames.append(frame_idx)
                    valid_coords.append([float(coords[0]), float(coords[1]), float(coords[2])])
            except (ValueError, TypeError, IndexError):
                continue
        
        if len(valid_frames) < 2:
            # 如果有效帧少于2个，无法插值，返回原始数据
            return {int(k): v for k, v in hand_dict.items() if isinstance(v, (list, tuple)) and len(v) >= 3}
        
        valid_frames = np.array(valid_frames, dtype=int)
        valid_coords = np.array(valid_coords, dtype=float)  # (N, 3)
        
        # 构建完整的帧序列
        min_f, max_f = int(valid_frames.min()), int(valid_frames.max())
        all_frames = np.arange(min_f, max_f + 1, dtype=int)
        
        # 构建位置数组，缺失帧用 NaN 填充
        pos = np.full((len(all_frames), 3), np.nan, dtype=float)
        frame_to_idx = {f: i for i, f in enumerate(all_frames)}
        
        for f, coords in zip(valid_frames, valid_coords):
            if f in frame_to_idx:
                pos[frame_to_idx[f]] = coords
        
        # 对每个坐标轴做线性插值填充 NaN
        for d in range(3):
            arr = pos[:, d]
            nans = np.isnan(arr)
            if np.all(nans):
                continue
            valid_idx = np.where(~nans)[0]
            if len(valid_idx) < 2:
                continue
            valid_vals = arr[valid_idx]
            # 使用 np.interp 进行线性插值
            interp_vals = np.interp(np.arange(len(arr)), valid_idx, valid_vals)
            arr[nans] = interp_vals[nans]
            pos[:, d] = arr
        
        # 构建结果字典（只包含插值后的有效值）
        interpolated = {}
        for i, frame_idx in enumerate(all_frames):
            if np.all(np.isfinite(pos[i])):
                interpolated[frame_idx] = [float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])]
        
        return interpolated
    
    # 对注册后的坐标进行插值
    reg_track_interpolated = interpolate_hand_positions(reg_track, total_frames)
    # 计算插值帧数：插值后有的帧 - 原始有的帧
    original_frame_set = {int(k) for k in reg_track.keys() if k.isdigit() or isinstance(k, int)}
    interpolated_frame_set = set(reg_track_interpolated.keys())
    interpolated_count = len(interpolated_frame_set - original_frame_set)
    if interpolated_count > 0:
        log.info("[INTERP] Interpolated %d missing frames (from %d to %d frames)", 
                 interpolated_count, len(original_frame_set), len(interpolated_frame_set))
    
    # 5) 世界系速度（基于插值后的坐标）
    speed_pairs: List[List[float]] = []
    prev_w = None
    zero_speed_count = 0
    speed_stats = {
        "missing_in_reg": 0,  # 配准后缺失的帧（插值前）
        "missing_after_interp": 0,  # 插值后仍然缺失的帧
        "first_frame": 0,     # 第一帧或前一帧缺失
        "valid_speed": 0      # 有效速度
    }
    
    for idx in range(total_frames):
        # 优先使用插值后的坐标，如果没有则使用原始坐标
        xyz = reg_track_interpolated.get(idx+1) or reg_track.get(str(idx+1))
        if xyz is None:
            speed = 0.0
            prev_w = None
            zero_speed_count += 1
            if str(idx+1) not in reg_track:
                speed_stats["missing_in_reg"] += 1
            else:
                speed_stats["missing_after_interp"] += 1
        else:
            if prev_w is None:
                speed = 0.0
                zero_speed_count += 1
                speed_stats["first_frame"] += 1
            else:
                dx, dy, dz = np.array(xyz) - np.array(prev_w)
                speed = float(np.linalg.norm([dx,dy,dz]))
                speed_stats["valid_speed"] += 1
            prev_w = xyz
        speed_pairs.append([idx, speed])
    
    log.info("[SPEED] Speed calculation statistics:")
    log.info("  Missing in registration (before interpolation): %d (%.1f%%)", 
             speed_stats["missing_in_reg"], speed_stats["missing_in_reg"]/total_frames*100)
    log.info("  Missing after interpolation: %d (%.1f%%)", 
             speed_stats["missing_after_interp"], speed_stats["missing_after_interp"]/total_frames*100)
    log.info("  First frame or gap: %d (%.1f%%)", 
             speed_stats["first_frame"], speed_stats["first_frame"]/total_frames*100)
    log.info("  Valid speed: %d (%.1f%%)", 
             speed_stats["valid_speed"], speed_stats["valid_speed"]/total_frames*100)
    log.info("  Total zero speed: %d/%d frames (%.1f%%)", 
             zero_speed_count, total_frames, 
             zero_speed_count/total_frames*100 if total_frames > 0 else 0)

    # 6) 输出
    # output_dir 已经是正确的输出目录（调用时已设置为 speed_output_dir）
    speed_json_path = output_dir / f"{video_name}_with_speed.json"
    _atomic_json_dump(speed_json_path, speed_pairs)

    # 速度可视化图（如果不需要可以跳过）
    if speed_only:
        speed_vis_path = ""  # 不生成可视化图
        depth_vis_path_str = ""  # 不生成深度可视化
    else:
        plt.figure(figsize=(12, 4))
        xs = [p[0] for p in speed_pairs]
        ys = [p[1] for p in speed_pairs]
        plt.plot(xs, ys, label="3D Hand Speed (world)")
        plt.xlabel("Frame"); plt.ylabel("Speed (relative)")
        plt.tight_layout()
        speed_vis_path = output_dir / f"{video_name}_speed_vis.png"
        plt.savefig(speed_vis_path); plt.close()
        # depth_vis_path 在前面已经定义过了
        depth_vis_path_str = str(depth_vis_path) if depth_vis_path.exists() else ""

    return speed_pairs, str(speed_json_path), str(speed_vis_path), depth_vis_path_str

def batch_process_videos(video_folder: str, output_root: str, device="cuda", encoder="vits", depth_root: Optional[str] = None, speed_only: bool = False):

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
        # 如果 speed_only，中间文件用临时目录，速度 JSON 直接保存在 output_root
        if speed_only:
            # 中间文件保存到 depth_root 或临时目录
            if depth_root is not None:
                intermediate_dir = Path(depth_root) / video_name
            else:
                intermediate_dir = output_root / video_name
            # 速度 JSON 保存在 output_root（不创建子目录）
            speed_output_dir = output_root
        else:
            intermediate_dir = output_root / video_name
            speed_output_dir = intermediate_dir
        
        print(f" 处理视频: {video_name}")

        try:
            pairs, speed_json, speed_png, depth_vis = extract_3d_speed_and_visualize(
                video_path=str(video_path),
                output_dir=str(speed_output_dir),  # 速度 JSON 的输出目录
                intermediate_dir=str(intermediate_dir),  # 中间文件的目录
                device=device,
                encoder=encoder,
                depth_root=depth_root,
                speed_only=speed_only
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


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="基于 MoGe2 metric depth 计算 3D 手部速度"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="单个视频路径（如 /path/to/video.mp4）"
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help="视频文件夹路径（批量处理）"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="输出根目录（如 /home/EgoLoc/output_egoloc）"
    )
    parser.add_argument(
        "--depth_root",
        type=str,
        default=None,
        help="预转换的深度文件根目录（如 /home/EgoLoc/output_egoloc）。如果不指定，会在 output_root 中查找"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备（默认: cuda）"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=["vits", "vitl"],
        help="VDA encoder（默认: vits，仅在需要生成深度时使用）"
    )
    parser.add_argument(
        "--speed_only",
        action="store_true",
        help="只输出速度 JSON 文件，中间文件（点云、可视化等）保存到其他位置"
    )
    
    args = parser.parse_args()
    
    if args.video_folder is not None:
        # 批量处理
        batch_process_videos(
            video_folder=args.video_folder,
            output_root=args.output_root,
            device=args.device,
            encoder=args.encoder,
            depth_root=args.depth_root,
            speed_only=args.speed_only
        )
    elif args.video is not None:
        # 单个视频处理
        video_path = Path(args.video)
        video_name = video_path.stem
        
        # 如果 speed_only，中间文件用临时目录，速度 JSON 直接保存在 output_root
        if args.speed_only:
            if args.depth_root is not None:
                intermediate_dir = Path(args.depth_root) / video_name
            else:
                intermediate_dir = Path(args.output_root) / video_name
            speed_output_dir = Path(args.output_root)  # 速度 JSON 直接保存在根目录
        else:
            intermediate_dir = Path(args.output_root) / video_name
            speed_output_dir = intermediate_dir
        
        print(f"处理视频: {video_name}")
        print(f"输出目录: {speed_output_dir}")
        
        try:
            pairs, speed_json, speed_png, depth_vis = extract_3d_speed_and_visualize(
                video_path=str(video_path),
                output_dir=str(speed_output_dir),
                intermediate_dir=str(intermediate_dir),
                device=args.device,
                encoder=args.encoder,
                depth_root=args.depth_root,
                speed_only=args.speed_only
            )
            print(f"\n完成: {video_name}")
            print(f"  ├─ 速度JSON: {speed_json}")
            print(f"  ├─ 速度图:   {speed_png}")
            print(f"  └─ 深度视频: {depth_vis}")
        except Exception as e:
            print(f"\n失败: {e}")
            raise
    else:
        parser.error("必须提供 --video 或 --video_folder 其中之一")


if __name__ == "__main__":
    main()
