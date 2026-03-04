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

from groundingdino.util.inference import load_model, load_image, predict
_model = load_model(
    "/home/EgoLoc/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "/home/EgoLoc/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"  # 直接放在weights目录外
)


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
def _load_depth(depth_dir: Path, idx: int, mmap: bool = True) -> Optional[np.ndarray]:
    """
    VDA stores **inverse depth** (bigger = nearer).
    Convert to metric depth in metres and keep a useful range.
    mmap=True: use memory-mapped load to reduce copy overhead.
    """
    f = depth_dir / f"pred_depth_{idx:06d}.npy"
    if not f.exists():
        return None
    inv = np.load(f, mmap_mode="r" if mmap else None)
    if inv.dtype != np.float32:
        inv = np.asarray(inv, dtype=np.float32)
    if _is_invalid_inv(inv):
        return None
    depth = DEPTH_SCALE_M / (inv + 1e-6)
    return np.asarray(depth, dtype=np.float32)

# 对每一帧你有两个 wrist 候选 a, b（来自ViTPose的 left/right）。
# 设上一帧稳定的 prev_L, prev_R。
#
# 如果出现：
#
# dist(a, prev_L) + dist(b, prev_R)
# 明显大于
# dist(a, prev_R) + dist(b, prev_L)
#
# 那基本就是当前帧左右手被交换了。
def is_identity_flip(a, b, prev_L, prev_R, margin=0.02):
    # margin: 允许的小差值，避免噪声误判
    if prev_L is None or prev_R is None or a is None or b is None:
        return False
    d_same = np.linalg.norm(np.array(a)-np.array(prev_L)) + np.linalg.norm(np.array(b)-np.array(prev_R))
    d_swap = np.linalg.norm(np.array(a)-np.array(prev_R)) + np.linalg.norm(np.array(b)-np.array(prev_L))
    return d_swap + margin < d_same

# ---------------------------------------------------------------------------
# Simple camera projection helper
# ---------------------------------------------------------------------------
def _pixel_to_camera(u: float, v: float, z: float, W: int, H: int):
    fx = fy = max(W, H)
    cx, cy = W / 2.0, H / 2.0
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return X, Y, z


def _wrist_to_cam3d_if_valid(
    wrist_uv: Optional[Tuple[float, float]],
    z_m: Optional[float],
    W: int, H: int
) -> Optional[Tuple[float, float, float]]:
    """
    2D 手腕 + 深度 → 3D 相机坐标（毫米）。
    wrist 或 z_m 任一无效则返回 None，不产生 3D 点、不更新 tracking。
    """
    if wrist_uv is None or z_m is None:
        return None
    u, v = wrist_uv
    z_mm = float(z_m) * 1000.0
    X, Y, Z = _pixel_to_camera(u, v, z_mm, W, H)
    return (float(X), float(Y), float(Z))


import numpy as np
from scipy import ndimage

def _wrist_from_frame(frame_bgr: np.ndarray,
                      gray_depth: np.ndarray,
                      cpm: "ViTPoseModel",
                      *,
                      detector=None,
                      # --- Detectron2 params ---
                      det_score_thr: float = 0.5,
                      det_max_person: int = 2,
                      # --- ViTPose hand params ---
                      kp_conf_thr: float = 0.3,
                      # --- depth-guided params ---
                      max_blobs: int = 3,
                      dup_px_thresh: float = 30.0,
                      depth_percentile: float = 20.0,
                      pad: int = 8,
                      verbose: bool = True):
    """
    Detectron2 优先：检测人框 -> ViTPose -> 左右手中心点
    失败则回退 depth-guided 多 blob：最近深度前景 -> 多 ROI -> ViTPose -> 合并去重

    Returns:
      dict: {'left': (u,v) or None, 'right': (u,v) or None}
    """

    def _log(msg: str):
        if verbose:
            print(msg)

    # helper：把一堆候选点去重合并，再按水平位置/side输出
    def _merge_and_assign(cand_uv):
        if not cand_uv:
            return {"left": None, "right": None}

        merged = []
        used = [False] * len(cand_uv)
        for i, ci in enumerate(cand_uv):
            if used[i]:
                continue
            ux, vy, sc, side = ci
            group = [i]
            for j in range(i + 1, len(cand_uv)):
                if used[j]:
                    continue
                ux2, vy2, sc2, side2 = cand_uv[j]
                d = float(np.hypot(ux - ux2, vy - vy2))
                if d < dup_px_thresh:
                    group.append(j)
            best = max(group, key=lambda k: cand_uv[k][2])
            merged.append(cand_uv[best])
            for k in group:
                used[k] = True

        merged = sorted(merged, key=lambda x: x[2], reverse=True)[:2]

        if len(merged) == 1:
            u, v, s, guessed_side = merged[0]
            # 单候选先放 left（外层可用 prev 修正身份）
            return {"left": (u, v), "right": None}

        a, b = merged[0], merged[1]
        # 按 x 排左右（更稳；side 作为参考，不强绑定）
        if a[0] <= b[0]:
            return {"left": (a[0], a[1]), "right": (b[0], b[1])}
        else:
            return {"left": (b[0], b[1]), "right": (a[0], a[1])}

    # helper：从 hand kpts 计算带权中心 + 平均置信度
    def _center_and_conf(hand_kpts, x0=0.0, y0=0.0):
        valid = hand_kpts[:, 2] > kp_conf_thr
        if valid.sum() <= 3:
            return None
        xs = hand_kpts[valid, 0]
        ys = hand_kpts[valid, 1]
        conf = hand_kpts[valid, 2]
        cx = float((xs * conf).sum() / (conf.sum() + 1e-8))
        cy = float((ys * conf).sum() / (conf.sum() + 1e-8))
        score = float(conf.mean())
        return (x0 + cx, y0 + cy, score)

    # ============================================================
    # 1) Detectron2 路径：人框 -> ViTPose -> 左/右手中心点
    # ============================================================
    if detector is not None:
        try:
            det_out = detector(frame_bgr)
            inst = det_out["instances"]

            # person class=0
            valid = (inst.pred_classes == 0) & (inst.scores >= det_score_thr)
            if valid.sum() == 0:
                _log(f"[WRIST] DETECTRON2 found 0 person >= {det_score_thr:.2f}, fallback.")
            else:
                # 取分数最高的前 det_max_person 个
                boxes = inst.pred_boxes.tensor[valid].detach().cpu().numpy()
                scores = inst.scores[valid].detach().cpu().numpy()
                order = np.argsort(scores)[::-1][:det_max_person]
                boxes = boxes[order]
                scores = scores[order]

                # ViTPose 要 RGB
                img_rgb = frame_bgr[:, :, ::-1]
                bboxes_with_score = np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32)

                poses = cpm.predict_pose(img_rgb, [bboxes_with_score])
                # poses 的结构常见是：list，每个人一个 dict
                # 你之前的代码是 vitposes_out[0]，这里更通用：遍历每个人
                cand_uv = []
                for pi, pose in enumerate(poses):
                    kpts = pose["keypoints"]  # (K,3) wholebody
                    left_kpts = kpts[-42:-21]
                    right_kpts = kpts[-21:]

                    lc = _center_and_conf(left_kpts, 0.0, 0.0)
                    rc = _center_and_conf(right_kpts, 0.0, 0.0)

                    if lc is not None:
                        cand_uv.append((lc[0], lc[1], lc[2], "left"))
                    if rc is not None:
                        cand_uv.append((rc[0], rc[1], rc[2], "right"))

                out = _merge_and_assign(cand_uv)
                if out["left"] is not None or out["right"] is not None:
                    _log("[WRIST] Using DETECTRON2+ViTPose.")
                    return out
                else:
                    _log("[WRIST] DETECTRON2+ViTPose produced no valid hand kpts, fallback.")

        except Exception as e:
            _log(f"[WRIST] DETECTRON2 path failed ({type(e).__name__}: {e}), fallback.")

    # ============================================================
    # 2) Depth-guided 多 blob 兜底（你原来的逻辑）
    # ============================================================
    _log("[WRIST] Using DEPTH_GUIDED multi-blob+ViTPose.")

    nearest = gray_depth < np.percentile(gray_depth, depth_percentile)
    labels, n_lbl = ndimage.label(nearest)
    if n_lbl == 0:
        return {"left": None, "right": None}

    sizes = ndimage.sum(nearest, labels, range(1, n_lbl + 1))
    order = np.argsort(sizes)[::-1]

    rois = []
    H, W = frame_bgr.shape[:2]
    for idx in order[:max_blobs]:
        lbl = idx + 1
        mask = labels == lbl
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(W - 1, x1 + pad); y1 = min(H - 1, y1 + pad)
        rois.append((x0, y0, x1, y1))

    cand_uv = []
    for (x0, y0, x1, y1) in rois:
        roi_bgr = frame_bgr[y0:y1+1, x0:x1+1]
        if roi_bgr.size == 0:
            continue

        bbox = np.array([[0, 0, roi_bgr.shape[1] - 1, roi_bgr.shape[0] - 1, 1.0]], dtype=np.float32)
        pose = cpm.predict_pose(roi_bgr[:, :, ::-1], [bbox])[0]
        kpts = pose["keypoints"]

        left_kpts  = kpts[-42:-21]
        right_kpts = kpts[-21:]

        lc = _center_and_conf(left_kpts, float(x0), float(y0))
        rc = _center_and_conf(right_kpts, float(x0), float(y0))

        if lc is not None:
            cand_uv.append((lc[0], lc[1], lc[2], "left"))
        if rc is not None:
            cand_uv.append((rc[0], rc[1], rc[2], "right"))

    return _merge_and_assign(cand_uv)



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
        # === FIX: depth 与 RGB 必须同尺寸，否则 Open3D 会报错 ===
        Hc, Wc = frame_rgb.shape[:2]
        Hd, Wd = depth_m.shape[:2]
        if (Hd != Hc) or (Wd != Wc):
            print(f"[PCD] Resize depth {Hd}x{Wd} -> {Hc}x{Wc} (frame={idx})")
            depth_m = cv2.resize(depth_m, (Wc, Hc), interpolation=cv2.INTER_LINEAR)

        # Open3D expects uint16 depth in millimetres
        depth_mm_u16 = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        depth_o3d = o3d.geometry.Image(depth_mm_u16)

        # # Open3D expects depth in millimetres by default (depth_scale=1000)
        # depth_o3d = o3d.geometry.Image((depth_m * 1000).astype(np.uint16))
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
                fitness = reg.fitness
                rmse = reg.inlier_rmse

                print(f"[ICP] video={video} frame={frame_id} fitness={fitness:.4f} rmse={rmse:.4f}")
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
def _sample_roi_depth(
    depth: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
    min_valid: int,
    drop_extreme_ratio: float = 0.10,
) -> Optional[float]:
    """在给定 ROI 内采样深度（米），valid 不足则返回 None。"""
    roi = depth[y0:y1, x0:x1].astype(np.float32)
    valid = roi[np.isfinite(roi) & (roi > 0)]
    if valid.size < min_valid:
        return None
    valid.sort()
    n = valid.size
    k = int(drop_extreme_ratio * n)
    valid2 = valid[k:n - k] if 2 * k < n else valid
    return float(np.median(valid2))


def depth_from_hand_roi_meters(
    depth: np.ndarray,
    box_xyxy: list,
    *,
    prev_z: float = None,          # meters
    drop_extreme_ratio: float = 0.10,
    max_jump: float = 0.10,        # meters/frame
    ema_alpha: float = 0.60,
    min_valid: int = 20,
    min_roi_side: int = 10,
    frame_H: Optional[int] = None,  # box 对应的 frame 尺寸（与 depth 不同时做缩放）
    frame_W: Optional[int] = None,
):
    """
    从 hand box ROI 内采样深度（米）。
    P0: ROI 严格 clip + min_roi_side 面积检查。
    P1: valid 不足时依次尝试全框 → 中心 40%×40% → 中心 20%×20%。
    P2: frame_H/frame_W 与 depth 尺寸不同时，将 box 从 frame 坐标系缩放到 depth 坐标系。
    """
    if depth is None or box_xyxy is None:
        return prev_z

    H_d, W_d = depth.shape[:2]
    bx0, by0, bx1, by1 = map(int, box_xyxy)

    # P2: box 从 frame 坐标缩放到 depth 坐标（坑 A：x0 floor, x1 ceil 避免 ROI 变空）
    if frame_H is not None and frame_W is not None and (H_d, W_d) != (frame_H, frame_W):
        scale_x = W_d / frame_W
        scale_y = H_d / frame_H
        bx0 = int(math.floor(bx0 * scale_x))
        bx1 = int(math.ceil(bx1 * scale_x))
        by0 = int(math.floor(by0 * scale_y))
        by1 = int(math.ceil(by1 * scale_y))

    # 坑 A：缩放后若 ROI 变空，直接 return prev_z
    if bx1 <= bx0 or by1 <= by0:
        return prev_z

    def clip_and_check(x0, y0, x1, y1):
        # 坑 A：上界用 W_d/H_d（允许等于），下界用 W_d-1/H_d-1
        x0 = max(0, min(x0, W_d - 1))
        x1 = max(0, min(x1, W_d))
        y0 = max(0, min(y0, H_d - 1))
        y1 = max(0, min(y1, H_d))
        if x1 <= x0 or y1 <= y0:
            return None
        if (x1 - x0) < min_roi_side or (y1 - y0) < min_roi_side:
            return None
        return (x0, y0, x1, y1)

    # P1: 依次尝试 全框 → 中心 40% → 中心 20%
    for frac in [1.0, 0.4, 0.2]:
        cx = (bx0 + bx1) / 2
        cy = (by0 + by1) / 2
        hw = (bx1 - bx0) * frac / 2
        hh = (by1 - by0) * frac / 2
        sx0 = int(math.floor(cx - hw))
        sx1 = int(math.ceil(cx + hw))
        sy0 = int(math.floor(cy - hh))
        sy1 = int(math.ceil(cy + hh))
        clipped = clip_and_check(sx0, sy0, sx1, sy1)
        if clipped is None:
            continue
        x0, y0, x1, y1 = clipped
        z_now = _sample_roi_depth(depth, x0, y0, x1, y1, min_valid, drop_extreme_ratio)
        if z_now is not None:
            if prev_z is not None and np.isfinite(prev_z):
                dz = z_now - float(prev_z)
                if abs(dz) > max_jump:
                    z_now = float(prev_z) + float(np.clip(dz, -max_jump, max_jump))
                z_now = float(ema_alpha * z_now + (1.0 - ema_alpha) * float(prev_z))
            return z_now
    return prev_z
def get_twohand_boxes_groundingdino(
    frame_bgr: np.ndarray,
    *,
    model,
    load_image_fn,
    predict_fn,
    box_thresh=0.30,
    text_thresh=0.25,
    expand_ratio=0.20,
    prev_box_L=None,
    prev_box_R=None,
):
    H, W = frame_bgr.shape[:2]

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, frame_bgr)
        pil_img, tensor_img = load_image_fn(tmp.name)

    boxes_norm, logits, phrases = predict_fn(
        model=model,
        image=tensor_img,
        caption="hand",
        box_threshold=box_thresh,
        text_threshold=text_thresh
    )
    if boxes_norm is None or len(boxes_norm) == 0:
        return None, None, prev_box_L, prev_box_R

    def xyxy_from_cxcywh_norm(b):
        cx, cy, bw, bh = b
        x0 = int((cx - bw/2) * W); y0 = int((cy - bh/2) * H)
        x1 = int((cx + bw/2) * W); y1 = int((cy + bh/2) * H)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    def box_center_x(box):
        return (box[0] + box[2]) / 2.0

    def iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0,bx0), max(ay0,by0)
        ix1, iy1 = min(ax1,bx1), min(ay1,by1)
        iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
        inter = iw*ih
        area_a = (ax1-ax0)*(ay1-ay0)
        area_b = (bx1-bx0)*(by1-by0)
        return inter / (area_a + area_b - inter + 1e-6)

    def expand(box):
        x0,y0,x1,y1 = box
        w = x1-x0; h = y1-y0
        ex = int(w * expand_ratio); ey = int(h * expand_ratio)
        x0 = max(0, x0-ex); y0 = max(0, y0-ey)
        x1 = min(W, x1+ex); y1 = min(H, y1+ey)
        return [x0,y0,x1,y1]

    candidates = []
    for k, b in enumerate(boxes_norm):
        box = xyxy_from_cxcywh_norm(b)
        if box is None:
            continue
        score = float(logits[k]) if logits is not None and len(logits) > k else 0.0
        candidates.append((box, score))

    if not candidates:
        return None, None, prev_box_L, prev_box_R

    # 有 prev_box：优先 IoU 匹配，避免左右互换
    used = set()
    boxL = None; boxR = None

    if prev_box_L is not None:
        ranked = sorted([(i, iou(candidates[i][0], prev_box_L), candidates[i][1]) for i in range(len(candidates))],
                        key=lambda x: (x[1], x[2]), reverse=True)
        for i, _, _ in ranked:
            if i not in used:
                boxL = candidates[i][0]; used.add(i); break

    if prev_box_R is not None:
        ranked = sorted([(i, iou(candidates[i][0], prev_box_R), candidates[i][1]) for i in range(len(candidates))],
                        key=lambda x: (x[1], x[2]), reverse=True)
        for i, _, _ in ranked:
            if i not in used:
                boxR = candidates[i][0]; used.add(i); break

    # 缺失就按 x 排序补齐
    remaining = [candidates[i] for i in range(len(candidates)) if i not in used]
    if boxL is None and remaining:
        boxL = sorted(remaining, key=lambda bs: box_center_x(bs[0]))[0][0]
    if boxR is None and remaining:
        boxR = sorted(remaining, key=lambda bs: box_center_x(bs[0]))[-1][0]

    if boxL is not None: boxL = expand(boxL)
    if boxR is not None: boxR = expand(boxR)

    new_prev_L = boxL if boxL is not None else prev_box_L
    new_prev_R = boxR if boxR is not None else prev_box_R
    return boxL, boxR, new_prev_L, new_prev_R


def _predict_dino_batch(model, tensors: list, caption: str, box_thresh: float, text_thresh: float, device: str = "cuda"):
    """
    Batch DINO inference. tensors: list of (C,H,W) tensors.
    Returns list of (boxes_norm, logits, phrases) per image.
    """
    if not tensors:
        return []
    import torch
    from groundingdino.util.inference import preprocess_caption
    from groundingdino.util.utils import get_phrases_from_posmap

    caption = preprocess_caption(caption)
    model = model.to(device)

    # Stack or pad to same size
    shapes = [t.shape for t in tensors]
    if len(set(shapes)) == 1:
        batch = torch.stack(tensors, dim=0).to(device)
    else:
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        padded = []
        for t in tensors:
            c, h, w = t.shape
            if h < max_h or w < max_w:
                pad = torch.zeros(1, c, max_h, max_w, dtype=t.dtype, device=t.device)
                pad[0, :, :h, :w] = t.unsqueeze(0)
                padded.append(pad)
            else:
                padded.append(t.unsqueeze(0))
        batch = torch.cat(padded, dim=0).to(device)

    with torch.no_grad():
        outputs = model(batch, captions=[caption] * len(tensors))

    pred_logits = outputs["pred_logits"].cpu().sigmoid()
    pred_boxes = outputs["pred_boxes"].cpu()
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    results = []
    for b in range(len(tensors)):
        mask = pred_logits[b].max(dim=1)[0] > box_thresh
        logits_b = pred_logits[b][mask]
        boxes_b = pred_boxes[b][mask]
        phrases_b = [
            get_phrases_from_posmap(lg > text_thresh, tokenized, tokenizer).replace(".", "")
            for lg in logits_b
        ]
        logits_1d = logits_b.max(dim=1)[0] if logits_b.numel() > 0 else pred_logits[b].max(dim=1)[0][:0]
        results.append((boxes_b, logits_1d, phrases_b))
    return results


def get_twohand_boxes_groundingdino_batch(
    frames: List[np.ndarray],
    *,
    model,
    load_image_fn,
    box_thresh=0.30,
    text_thresh=0.25,
    expand_ratio=0.20,
    prev_box_L=None,
    prev_box_R=None,
    batch_size=4,
) -> List[Tuple]:
    """
    Batch version: process multiple frames in one DINO forward.
    Returns list of (boxL, boxR, new_prev_L, new_prev_R) for each frame.
    """
    import tempfile
    import os

    if not frames:
        return []

    H, W = frames[0].shape[:2]
    tensors = []
    tmp_files = []
    for f in frames:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_files.append(tmp.name)
        cv2.imwrite(tmp.name, f)
        _, tensor_img = load_image_fn(tmp.name)
        tensors.append(tensor_img)
    for t in tmp_files:
        try:
            os.unlink(t)
        except OSError:
            pass

    batch_results = _predict_dino_batch(model, tensors, "hand", box_thresh, text_thresh)

    def xyxy_from_cxcywh_norm(b, W, H):
        if hasattr(b, "cpu"):
            b = b.cpu().numpy()
        b = np.asarray(b).ravel()
        if len(b) < 4:
            return None
        cx, cy, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        x0 = int((cx - bw / 2) * W)
        y0 = int((cy - bh / 2) * H)
        x1 = int((cx + bw / 2) * W)
        y1 = int((cy + bh / 2) * H)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    def box_center_x(box):
        return (box[0] + box[2]) / 2.0

    def iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        area_a = (ax1 - ax0) * (ay1 - ay0)
        area_b = (bx1 - bx0) * (by1 - by0)
        return inter / (area_a + area_b - inter + 1e-6)

    def expand(box, W, H):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        ex, ey = int(w * expand_ratio), int(h * expand_ratio)
        return [max(0, x0 - ex), max(0, y0 - ey), min(W, x1 + ex), min(H, y1 + ey)]

    out = []
    pL, pR = prev_box_L, prev_box_R
    for b, (boxes_norm, logits, phrases) in enumerate(batch_results):
        if boxes_norm is None or (hasattr(boxes_norm, "numel") and boxes_norm.numel() == 0):
            out.append((None, None, pL, pR))
            continue

        candidates = []
        nbox = len(boxes_norm)
        for k in range(nbox):
            bk = boxes_norm[k]
            if hasattr(bk, "cpu"):
                bk = bk.cpu().numpy()
            box = xyxy_from_cxcywh_norm(bk, W, H)
            if box is None:
                continue
            sc = float(logits[k]) if logits is not None and k < len(logits) else 0.0
            candidates.append((box, sc))

        if not candidates:
            out.append((None, None, pL, pR))
            continue

        used = set()
        boxL = boxR = None
        if pL is not None:
            ranked = sorted(
                [(i, iou(candidates[i][0], pL), candidates[i][1]) for i in range(len(candidates))],
                key=lambda x: (x[1], x[2]),
                reverse=True,
            )
            for i, _, _ in ranked:
                if i not in used:
                    boxL = candidates[i][0]
                    used.add(i)
                    break
        if pR is not None:
            ranked = sorted(
                [(i, iou(candidates[i][0], pR), candidates[i][1]) for i in range(len(candidates))],
                key=lambda x: (x[1], x[2]),
                reverse=True,
            )
            for i, _, _ in ranked:
                if i not in used:
                    boxR = candidates[i][0]
                    used.add(i)
                    break

        remaining = [candidates[i] for i in range(len(candidates)) if i not in used]
        if boxL is None and remaining:
            boxL = sorted(remaining, key=lambda bs: box_center_x(bs[0]))[0][0]
        if boxR is None and remaining:
            boxR = sorted(remaining, key=lambda bs: box_center_x(bs[0]))[-1][0]

        if boxL is not None:
            boxL = expand(boxL, W, H)
        if boxR is not None:
            boxR = expand(boxR, W, H)
        pL = boxL if boxL is not None else pL
        pR = boxR if boxR is not None else pR
        out.append((boxL, boxR, pL, pR))

    return out


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


def fix_left_right_identity(wrists, prev_L, prev_R):
    """
    wrists: {"left": (u,v) or None, "right": (u,v) or None}
    prev_L/prev_R: previous assigned (u,v) or None
    return: corrected wrists dict
    """
    a = wrists.get("left")
    b = wrists.get("right")

    # 情况1：只有一个手
    if a is None and b is None:
        return {"left": None, "right": None}
    if a is None:
        # 只有 b，分给离 prev 更近的那侧
        if prev_L is None and prev_R is None:
            return {"left": None, "right": b}
        if prev_R is None:
            return {"left": b, "right": None}
        if prev_L is None:
            return {"left": None, "right": b}
        # 两个 prev 都有时，谁近给谁
        dL = np.linalg.norm(np.array(b)-np.array(prev_L))
        dR = np.linalg.norm(np.array(b)-np.array(prev_R))
        return {"left": b, "right": None} if dL < dR else {"left": None, "right": b}

    if b is None:
        if prev_L is None and prev_R is None:
            return {"left": a, "right": None}
        if prev_L is None:
            return {"left": None, "right": a}
        if prev_R is None:
            return {"left": a, "right": None}
        dL = np.linalg.norm(np.array(a)-np.array(prev_L))
        dR = np.linalg.norm(np.array(a)-np.array(prev_R))
        return {"left": a, "right": None} if dL < dR else {"left": None, "right": a}

    # 情况2：两只手都检测到 → 选择“是否交换”使得和上一帧最一致
    if prev_L is None or prev_R is None:
        # 如果上一帧没法提供双手约束，就用水平位置做启发式
        # 更靠左(小u)当 left
        if a[0] <= b[0]:
            return {"left": a, "right": b}
        else:
            return {"left": b, "right": a}

    d_same = np.linalg.norm(np.array(a)-np.array(prev_L)) + np.linalg.norm(np.array(b)-np.array(prev_R))
    d_swap = np.linalg.norm(np.array(a)-np.array(prev_R)) + np.linalg.norm(np.array(b)-np.array(prev_L))

    if d_swap < d_same:
        # 交换
        return {"left": b, "right": a}
    else:
        return {"left": a, "right": b}
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
def compute_central_speed(pos_dict, total_frames):
    """
    使用 central difference:
    v_t = ||p_{t+1} - p_{t-1}|| / 2
    """
    speed = {}
    for t in range(total_frames):
        if (t - 1) in pos_dict and (t + 1) in pos_dict:
            p_prev = pos_dict[t - 1]
            p_next = pos_dict[t + 1]
            v = np.linalg.norm(p_next - p_prev) / 2.0
            speed[t] = float(v)
        else:
            speed[t] = 0.0
    return speed


def _check_and_log_speed_stats(video_name: str, speed_pairs: List[List[float]], total_frames: int) -> None:
    """
    自检速度输出：校验数量、统计有效帧、检测异常，并打印简要统计。
    """
    n = len(speed_pairs)
    if n == 0:
        log.warning("[speed] %s: empty speed_pairs", video_name)
        return
    if n != total_frames:
        log.warning("[speed] %s: pairs=%d vs total_frames=%d (mismatch)", video_name, n, total_frames)

    vL_list = [p[1] for p in speed_pairs if len(p) >= 2]
    vR_list = [p[2] for p in speed_pairs if len(p) >= 3]
    valid_L = sum(1 for v in vL_list if v > 0 and np.isfinite(v))
    valid_R = sum(1 for v in vR_list if v > 0 and np.isfinite(v))
    valid_any = sum(1 for p in speed_pairs if len(p) >= 3 and ((p[1] > 0 and np.isfinite(p[1])) or (p[2] > 0 and np.isfinite(p[2]))))

    max_L = max(vL_list) if vL_list else 0.0
    max_R = max(vR_list) if vR_list else 0.0
    mean_L = float(np.mean(vL_list)) if vL_list else 0.0
    mean_R = float(np.mean(vR_list)) if vR_list else 0.0

    # 异常检测
    all_zero = valid_any == 0
    very_few = valid_any < max(5, total_frames * 0.05)
    extreme = (max_L > 5.0 or max_R > 5.0)  # 世界系速度 > 5 单位/帧 视为异常

    log.info(
        "[speed] %s: frames=%d pairs=%d | L: valid=%d max=%.4f mean=%.4f | R: valid=%d max=%.4f mean=%.4f",
        video_name, total_frames, n, valid_L, max_L, mean_L, valid_R, max_R, mean_R,
    )
    if all_zero:
        log.warning("[speed] %s: all frames zero speed (possible failure)", video_name)
    elif very_few:
        log.warning("[speed] %s: only %d frames with non-zero speed (%.1f%%)", video_name, valid_any, valid_any / max(1, n) * 100)
    if extreme:
        log.warning("[speed] %s: extreme speed values (L=%.2f R=%.2f)", video_name, max_L, max_R)


def extract_3d_speed_and_visualize(video_path: str, output_dir: str, *, device: str = "cuda", encoder: str = "vits", dino_batch_size: int = 4) -> Tuple[List[List[float]], str, str, str]:
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
    # cam_hand_dir = output_dir / "hand3d_cam"; _ensure_dir(cam_hand_dir)
    # cam_hand_json = cam_hand_dir / f"{video_name}.json"
    cam_hand_dir_L = output_dir / "hand3d_cam_left";
    _ensure_dir(cam_hand_dir_L)
    cam_hand_dir_R = output_dir / "hand3d_cam_right";
    _ensure_dir(cam_hand_dir_R)

    cam_hand_json_L = cam_hand_dir_L / f"{video_name}.json"
    cam_hand_json_R = cam_hand_dir_R / f"{video_name}.json"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # cam_track: Dict[str, List[float]] = {}
    # prev_cam = None
    # speed_cam: Dict[int, float] = {}
    cam_track_L: Dict[str, List[float]] = {}
    cam_track_R: Dict[str, List[float]] = {}
    prev_cam_L = None
    prev_cam_R = None
    speed_cam_L: Dict[int, float] = {}
    speed_cam_R: Dict[int, float] = {}
    prev_uL = None  # 上一帧左手2D wrist
    prev_uR = None  # 上一帧右手2D wrist
    wrist2d_L: Dict[str, List[float]] = {}
    wrist2d_R: Dict[str, List[float]] = {}
    prev_zL = None  # 上一帧左手的深度（米）
    prev_zR = None  # 上一帧右手的深度（米）
    cam_positions_L = {}
    cam_positions_R = {}
    prev_box_L = None
    prev_box_R = None

    # 深度预加载：避免主循环内频繁读磁盘（长视频限制 2000 帧防 OOM）
    max_preload = 2000
    if total_frames <= max_preload:
        depths = [_load_depth(depth_dir, i) for i in range(total_frames)]
    else:
        depths = None

    detector = _get_body_detector()
    batch_size = max(1, min(dino_batch_size, 8))

    idx = 0
    while idx < total_frames:
        batch_frames = []
        batch_indices = []
        for _ in range(batch_size):
            if idx >= total_frames:
                break
            ok, frame = cap.read()
            if not ok:
                speed_cam_L[idx] = 0.0
                speed_cam_R[idx] = 0.0
                prev_cam_L = None
                prev_cam_R = None
                idx += 1
                continue
            batch_frames.append(frame)
            batch_indices.append(idx)
            idx += 1

        if not batch_frames:
            continue

        # --- DINO 批量或逐帧 ---
        if len(batch_frames) > 1 and batch_size > 1:
            batch_results = get_twohand_boxes_groundingdino_batch(
                batch_frames,
                model=_model,
                load_image_fn=load_image,
                box_thresh=0.30,
                text_thresh=0.25,
                expand_ratio=0.20,
                prev_box_L=prev_box_L,
                prev_box_R=prev_box_R,
            )
        else:
            boxL, boxR, prev_box_L, prev_box_R = get_twohand_boxes_groundingdino(
                batch_frames[0],
                model=_model,
                load_image_fn=load_image,
                predict_fn=predict,
                box_thresh=0.30,
                text_thresh=0.25,
                expand_ratio=0.20,
                prev_box_L=prev_box_L,
                prev_box_R=prev_box_R,
            )
            batch_results = [(boxL, boxR, prev_box_L, prev_box_R)]

        for bi, frame in enumerate(batch_frames):
            fidx = batch_indices[bi]
            boxL, boxR, prev_box_L, prev_box_R = batch_results[bi]
            depth = depths[fidx] if depths is not None else _load_depth(depth_dir, fidx)

            # --- ROI 鲁棒深度（单位：m） ---
            fh, fw = frame.shape[:2]
            zL_m = depth_from_hand_roi_meters(depth, boxL, prev_z=prev_zL, frame_H=fh, frame_W=fw) if boxL is not None else None
            zR_m = depth_from_hand_roi_meters(depth, boxR, prev_z=prev_zR, frame_H=fh, frame_W=fw) if boxR is not None else None

            # 更新上一帧深度（m）
            prev_zL = zL_m if zL_m is not None else prev_zL
            prev_zR = zR_m if zR_m is not None else prev_zR

            if depth is None:
                speed_cam_L[fidx] = 0.0
                speed_cam_R[fidx] = 0.0
                prev_cam_L = None
                prev_cam_R = None
                continue

            H, W = depth.shape
            wrists = _wrist_from_frame(frame, depth, cpm, detector=detector, verbose=False)
            wrists = fix_left_right_identity(wrists, prev_uL, prev_uR)
            lw = wrists["left"]
            rw = wrists["right"]
            # ==== 在这里多存 2D 像素坐标 ====
            frame_key = str(fidx + 1)  # 和你 JSON 里其它地方一样 1-based
            if lw is not None:
                uL, vL = lw
                wrist2d_L[frame_key] = [float(uL), float(vL)]
            if rw is not None:
                uR, vR = rw
                wrist2d_R[frame_key] = [float(uR), float(vR)]

            # 更新 prev_uL/prev_uR
            prev_uL = lw if lw is not None else prev_uL
            prev_uR = rw if rw is not None else prev_uR
            if wrists is None:
                speed_cam_L[fidx] = 0.0
                speed_cam_R[fidx] = 0.0
                prev_cam_L = None
                prev_cam_R = None
                continue

            H, W = depth.shape

            # ------- 左手：depth 有效才算 3D，否则 reset tracking -------
            cam3d_L = _wrist_to_cam3d_if_valid(lw, zL_m, W, H)
            if cam3d_L is not None:
                cam_track_L[str(fidx + 1)] = list(cam3d_L)
                cam_positions_L[fidx] = np.array(cam3d_L, dtype=float)
                prev_cam_L = cam3d_L
            else:
                prev_cam_L = None

            # ------- 右手：同上 -------
            cam3d_R = _wrist_to_cam3d_if_valid(rw, zR_m, W, H)
            if cam3d_R is not None:
                cam_track_R[str(fidx + 1)] = list(cam3d_R)
                cam_positions_R[fidx] = np.array(cam3d_R, dtype=float)
                prev_cam_R = cam3d_R
            else:
                prev_cam_R = None

            if (fidx + 1) % 100 == 0 or fidx == total_frames - 1:
                log.info("[3D] Frames %d/%d", fidx + 1, total_frames)

    cap.release()
    speed_cam_L = compute_central_speed(cam_positions_L, total_frames)
    speed_cam_R = compute_central_speed(cam_positions_R, total_frames)

    #_atomic_json_dump(cam_hand_json, cam_track)
    _atomic_json_dump(cam_hand_json_L, cam_track_L)
    _atomic_json_dump(cam_hand_json_R, cam_track_R)
    #保存2d手腕坐标，在深度图可视化
    # 保存 2D 像素坐标
    wrist2d_dir = output_dir / "hand2d_px"
    _ensure_dir(wrist2d_dir)
    wrist2d_L_json = wrist2d_dir / f"{video_name}_left.json"
    wrist2d_R_json = wrist2d_dir / f"{video_name}_right.json"
    _atomic_json_dump(wrist2d_L_json, wrist2d_L)
    _atomic_json_dump(wrist2d_R_json, wrist2d_R)
    # 4) 注册到世界坐标

    reg_dir_L = output_dir / "registered_hands_left";
    _ensure_dir(reg_dir_L)
    reg_dir_R = output_dir / "registered_hands_right";
    _ensure_dir(reg_dir_R)
    pcd_root = str((output_dir / "pointclouds"))#保证能有register_hand_positions输出

    register_hand_positions(str(pcd_root), str(cam_hand_dir_L), str(reg_dir_L))
    register_hand_positions(str(pcd_root), str(cam_hand_dir_R), str(reg_dir_R))

    reg_json_L = reg_dir_L / f"{video_name}.json"
    reg_json_R = reg_dir_R / f"{video_name}.json"

    reg_track_L = json.loads(reg_json_L.read_text(encoding="utf-8")) if reg_json_L.exists() else cam_track_L
    reg_track_R = json.loads(reg_json_R.read_text(encoding="utf-8")) if reg_json_R.exists() else cam_track_R
    # 5) 世界系速度

    speed_pairs: List[List[float]] = []  # [[frame, v_left, v_right], ...]
    prev_w_L = None
    prev_w_R = None

    for idx in range(total_frames):
        # 左手
        xyzL = reg_track_L.get(str(idx + 1))
        if xyzL is None:
            vL = 0.0
            prev_w_L = None
        else:
            if prev_w_L is None:
                vL = 0.0
            else:
                dx, dy, dz = np.array(xyzL) - np.array(prev_w_L)
                vL = float(np.linalg.norm([dx, dy, dz]))
            prev_w_L = xyzL

        # 右手
        xyzR = reg_track_R.get(str(idx + 1))
        if xyzR is None:
            vR = 0.0
            prev_w_R = None
        else:
            if prev_w_R is None:
                vR = 0.0
            else:
                dx, dy, dz = np.array(xyzR) - np.array(prev_w_R)
                vR = float(np.linalg.norm([dx, dy, dz]))
            prev_w_R = xyzR

        speed_pairs.append([idx, vL, vR])

    # 6) 输出
    #speed_json_path = output_dir / f"{video_name}_with_speed.json"
    speed_json_path = output_dir / f"{video_name}_with_speed_twohands.json"
    _atomic_json_dump(speed_json_path, speed_pairs)

    # 6.5) 速度自检与统计
    _check_and_log_speed_stats(video_name, speed_pairs, total_frames)

    plt.figure(figsize=(12, 4))
    xs = [p[0] for p in speed_pairs]
    #ys = [p[1] for p in speed_pairs]
    ysL = [p[1] for p in speed_pairs]
    ysR = [p[2] for p in speed_pairs]
    #plt.plot(xs, ys, label="3D Hand Speed (world)")
    plt.plot(xs, ysL, label="Left Hand Speed (world)")
    plt.plot(xs, ysR, label="Right Hand Speed (world)")
    plt.xlabel("Frame"); plt.ylabel("Speed (relative)")
    plt.tight_layout()
    
    speed_vis_path = output_dir / f"{video_name}_speed_vis_twohands.png"
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

