"""
Minimal hand detection utils for extract_hand21_and_visualize.
No open3d, no groundingdino - just Detectron2 + ViTPose.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# 路径配置：优先环境变量，否则基于脚本位置推导
_EGOLOC = Path(os.environ.get("EGOLOC_ROOT", Path(__file__).resolve().parent))
_HAMER = Path(os.environ.get("HAMER_ROOT", _EGOLOC / "hamer"))
_MMPOSE = "/home/EgoLoc/mmpose/mmpose-0.x"

paths = [str(_MMPOSE), str(_HAMER)]
for p in paths:
    if p and p not in sys.path:
        sys.path.insert(0, p)
old = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = ":".join([p for p in paths if p]) + (":" + old if old else "")

DEPTH_SCALE_M = 3.0


def _is_invalid_inv(inv: np.ndarray) -> bool:
    return bool((not np.isfinite(inv).any()) or np.nanstd(inv) < 1e-4)


def _load_depth(depth_dir: Path, idx: int, mmap: bool = True) -> Optional[np.ndarray]:
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


def fix_left_right_identity(wrists, prev_L, prev_R):
    a, b = wrists.get("left"), wrists.get("right")
    if a is None and b is None:
        return {"left": None, "right": None}
    if a is None:
        if prev_L is None and prev_R is None:
            return {"left": None, "right": b}
        if prev_R is None:
            return {"left": b, "right": None}
        if prev_L is None:
            return {"left": None, "right": b}
        dL = np.linalg.norm(np.array(b) - np.array(prev_L))
        dR = np.linalg.norm(np.array(b) - np.array(prev_R))
        return {"left": b, "right": None} if dL < dR else {"left": None, "right": b}
    if b is None:
        if prev_L is None and prev_R is None:
            return {"left": a, "right": None}
        if prev_L is None:
            return {"left": None, "right": a}
        if prev_R is None:
            return {"left": a, "right": None}
        dL = np.linalg.norm(np.array(a) - np.array(prev_L))
        dR = np.linalg.norm(np.array(a) - np.array(prev_R))
        return {"left": a, "right": None} if dL < dR else {"left": None, "right": a}
    if prev_L is None or prev_R is None:
        return {"left": (a if a[0] <= b[0] else b), "right": (b if a[0] <= b[0] else a)}
    d_same = np.linalg.norm(np.array(a) - np.array(prev_L)) + np.linalg.norm(np.array(b) - np.array(prev_R))
    d_swap = np.linalg.norm(np.array(a) - np.array(prev_R)) + np.linalg.norm(np.array(b) - np.array(prev_L))
    if d_swap < d_same:
        return {"left": b, "right": a}
    return {"left": a, "right": b}


_CACHE: Dict[str, object] = {}


def _get_body_detector(device: str = "cuda"):
    if "detector" in _CACHE:
        return _CACHE["detector"]
    try:
        import importlib.util
        utils_path = _HAMER / "hamer" / "utils" / "utils_detectron2.py"
        if utils_path.exists():
            spec = importlib.util.spec_from_file_location("utils_detectron2", utils_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                DefaultPredictor_Lazy = mod.DefaultPredictor_Lazy
            else:
                from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        else:
            from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            detector = DefaultPredictor_Lazy(cfg)
        except Exception:
            import hamer
            from detectron2.config import LazyConfig
            cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
            cfg = LazyConfig.load(str(cfg_path))
            for i in range(3):
                cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            detector = DefaultPredictor_Lazy(cfg)
        _CACHE["detector"] = detector
        return detector
    except Exception as e:
        print(f"[egoloc_hand_utils] Detectron2 failed: {e}")
        return None


def _get_vitpose_model(device: str = "cuda"):
    if "cpm" in _CACHE:
        return _CACHE["cpm"]
    if str(_HAMER) not in sys.path:
        sys.path.insert(0, str(_HAMER))
    vit_dir = _HAMER / "third-party" / "ViTPose"
    cfg_path = vit_dir / "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py"
    ckpt_path = _HAMER / "_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth"
    import vitpose_model as _vpm
    _vpm.ROOT_DIR = str(_HAMER) + "/"
    _vpm.VIT_DIR = str(vit_dir)
    for _dic in _vpm.ViTPoseModel.MODEL_DICT.values():
        _dic["config"] = str(cfg_path)
        _dic["model"] = str(ckpt_path)
    from vitpose_model import ViTPoseModel
    _CACHE["cpm"] = ViTPoseModel(device)
    return _CACHE["cpm"]
