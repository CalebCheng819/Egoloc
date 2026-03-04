#!/usr/bin/env python3
"""
提取 ViTPose COCO-WholeBody 21 点手部关键点，并在原视频上绘制 21 个点（不画连线）。
复用 egoloc_speed_twohands 的 Detectron2+ViTPose 流程。
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

# Use minimal hand utils (no open3d, no groundingdino)
EGOLOC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EGOLOC_ROOT))

from egoloc_hand_utils import (
    _get_body_detector,
    _get_vitpose_model,
    _load_depth,
    fix_left_right_identity,
)

def _get_hand21_from_frame(
    frame_bgr: np.ndarray,
    gray_depth: np.ndarray,
    cpm: "ViTPoseModel",
    *,
    detector=None,
    det_score_thr: float = 0.5,
    det_max_person: int = 2,
    kp_conf_thr: float = 0.25,
    max_blobs: int = 3,
    dup_px_thresh: float = 30.0,
    depth_percentile: float = 20.0,
    pad: int = 8,
    verbose: bool = False,
) -> Dict[str, Optional[np.ndarray]]:
    """
    与 _wrist_from_frame 相同流程，但返回完整 21 点手部关键点。
    Returns: {"left": (21,3) np.array or None, "right": (21,3) or None}
    关键点格式: (x, y, confidence)，坐标为全图像素坐标。
    """
    def _log(msg: str):
        if verbose:
            print(msg)

    def _center_and_conf(hand_kpts: np.ndarray, x0: float = 0.0, y0: float = 0.0):
        """返回 (cx, cy, score) 或 None"""
        valid = hand_kpts[:, 2] > kp_conf_thr
        if valid.sum() <= 3:
            return None
        xs, ys = hand_kpts[valid, 0], hand_kpts[valid, 1]
        conf = hand_kpts[valid, 2]
        cx = float((xs * conf).sum() / (conf.sum() + 1e-8)) + x0
        cy = float((ys * conf).sum() / (conf.sum() + 1e-8)) + y0
        return (cx, cy, float(conf.mean()))

    def _kpts_to_full_image(hand_kpts: np.ndarray, x0: float, y0: float) -> np.ndarray:
        """将 crop 坐标系下的关键点转换为全图坐标"""
        out = hand_kpts.copy()
        out[:, 0] += x0
        out[:, 1] += y0
        return out

    def _merge_and_assign_with_kpts(
        cand: List[Tuple[float, float, float, str, np.ndarray]]
    ) -> Dict[str, Optional[np.ndarray]]:
        """cand: [(ux, vy, score, side, kpts), ...], each entry = one hand's kpts"""
        if not cand:
            return {"left": None, "right": None}

        merged = []
        used = [False] * len(cand)
        for i, ci in enumerate(cand):
            if used[i]:
                continue
            ux, vy, sc = ci[0], ci[1], ci[2]
            group = [i]
            for j in range(i + 1, len(cand)):
                if used[j]:
                    continue
                ux2, vy2 = cand[j][0], cand[j][1]
                if np.hypot(ux - ux2, vy - vy2) < dup_px_thresh:
                    group.append(j)
            best_idx = max(group, key=lambda k: cand[k][2])
            merged.append(cand[best_idx])
            for k in group:
                used[k] = True

        merged = sorted(merged, key=lambda x: x[2], reverse=True)[:2]
        out = {"left": None, "right": None}
        if len(merged) == 1:
            out["left"] = merged[0][4]  # 单候选默认放 left
            return out

        a, b = merged[0], merged[1]
        kA, kB = a[4], b[4]
        if a[0] <= b[0]:  # a 更靠左 -> a=left, b=right
            out["left"] = kA if a[3] == "left" else kB
            out["right"] = kB if a[3] == "left" else kA
        else:
            out["left"] = kB if b[3] == "left" else kA
            out["right"] = kA if b[3] == "left" else kB
        return out

    # ---- 1) Detectron2 路径 ----
    if detector is not None:
        try:
            det_out = detector(frame_bgr)
            inst = det_out["instances"]
            valid = (inst.pred_classes == 0) & (inst.scores >= det_score_thr)
            if valid.sum() == 0:
                _log("[HAND21] Detectron2: 0 person, fallback.")
            else:
                boxes = inst.pred_boxes.tensor[valid].detach().cpu().numpy()
                scores = inst.scores[valid].detach().cpu().numpy()
                order = np.argsort(scores)[::-1][:det_max_person]
                boxes, scores = boxes[order], scores[order]

                img_rgb = frame_bgr[:, :, ::-1]
                bboxes_with_score = np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32)
                poses = cpm.predict_pose(img_rgb, [bboxes_with_score])

                cand = []
                for pi, pose in enumerate(poses):
                    kpts = pose["keypoints"]
                    # mmpose top-down 传入原图+bbox 时，predict_pose 返回原图坐标
                    left_kpts = np.asarray(kpts[-42:-21], dtype=np.float32).copy()
                    right_kpts = np.asarray(kpts[-21:], dtype=np.float32).copy()

                    lc = _center_and_conf(left_kpts, 0.0, 0.0)
                    rc = _center_and_conf(right_kpts, 0.0, 0.0)
                    if lc:
                        cand.append((lc[0], lc[1], lc[2], "left", left_kpts))
                    if rc:
                        cand.append((rc[0], rc[1], rc[2], "right", right_kpts))

                if cand:
                    out = _merge_and_assign_with_kpts(cand)
                    if out["left"] is not None or out["right"] is not None:
                        _log("[HAND21] Detectron2+ViTPose OK")
                        return out
        except Exception as e:
            _log(f"[HAND21] Detectron2 failed: {e}")

    # ---- 2) Depth-guided 兜底 ----
    _log("[HAND21] Using depth-guided fallback.")
    nearest = gray_depth < np.percentile(gray_depth, depth_percentile)
    labels, n_lbl = ndimage.label(nearest)
    if n_lbl == 0:
        return {"left": None, "right": None}

    sizes = ndimage.sum(nearest, labels, range(1, n_lbl + 1))
    order = np.argsort(sizes)[::-1]
    H, W = frame_bgr.shape[:2]
    rois = []
    for idx in order[:max_blobs]:
        lbl = idx + 1
        mask = labels == lbl
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(H - 1, int(ys.max()) + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(W - 1, int(xs.max()) + pad)
        rois.append((x0, y0, x1, y1))

    cand = []
    for (x0, y0, x1, y1) in rois:
        roi_bgr = frame_bgr[y0:y1 + 1, x0:x1 + 1]
        if roi_bgr.size == 0:
            continue
        bbox = np.array([[0, 0, roi_bgr.shape[1] - 1, roi_bgr.shape[0] - 1, 1.0]], dtype=np.float32)
        pose = cpm.predict_pose(roi_bgr[:, :, ::-1], [bbox])[0]
        kpts = pose["keypoints"]
        left_kpts = _kpts_to_full_image(np.asarray(kpts[-42:-21], dtype=np.float32), float(x0), float(y0))
        right_kpts = _kpts_to_full_image(np.asarray(kpts[-21:], dtype=np.float32), float(x0), float(y0))

        lc = _center_and_conf(left_kpts)
        rc = _center_and_conf(right_kpts)
        if lc:
            cand.append((lc[0], lc[1], lc[2], "left", left_kpts))
        if rc:
            cand.append((rc[0], rc[1], rc[2], "right", right_kpts))

    return _merge_and_assign_with_kpts(cand) if cand else {"left": None, "right": None}


def fix_left_right_identity_hand21(
    hands: Dict[str, Optional[np.ndarray]],
    prev_L: Optional[Tuple[float, float]],
    prev_R: Optional[Tuple[float, float]],
) -> Dict[str, Optional[np.ndarray]]:
    """根据 wrist (kpt 0) 修正左右手身份"""
    kL, kR = hands.get("left"), hands.get("right")
    if kL is None and kR is None:
        return {"left": None, "right": None}
    if kL is None:
        w = (float(kR[0, 0]), float(kR[0, 1])) if kR is not None and kR.size >= 3 else None
        wrists = {"left": None, "right": w}
        fixed = fix_left_right_identity(wrists, prev_L, prev_R)
        if fixed["left"] is not None:
            return {"left": kR, "right": None}
        return {"left": None, "right": kR}
    if kR is None:
        w = (float(kL[0, 0]), float(kL[0, 1]))
        wrists = {"left": w, "right": None}
        fixed = fix_left_right_identity(wrists, prev_L, prev_R)
        if fixed["right"] is not None:
            return {"left": None, "right": kL}
        return {"left": kL, "right": None}

    wL = (float(kL[0, 0]), float(kL[0, 1]))
    wR = (float(kR[0, 0]), float(kR[0, 1]))
    fixed = fix_left_right_identity({"left": wL, "right": wR}, prev_L, prev_R)
    if fixed["left"] == wL and fixed["right"] == wR:
        return {"left": kL, "right": kR}
    return {"left": kR, "right": kL}


def draw_hand_points(
    frame: np.ndarray,
    left_kpts: Optional[np.ndarray],
    right_kpts: Optional[np.ndarray],
    *,
    kp_conf_thr: float = 0.25,
    point_radius: int = 3,
    color_left: Tuple[int, int, int] = (255, 100, 100),   # BGR
    color_right: Tuple[int, int, int] = (100, 255, 100),
) -> np.ndarray:
    """只绘制 21 个关键点，不画连线"""
    out = frame.copy()
    h, w = out.shape[:2]

    def draw_points(kpts: np.ndarray, color: Tuple[int, int, int]):
        if kpts is None or kpts.shape[0] < 21:
            return
        for i in range(21):
            if kpts[i, 2] < kp_conf_thr:
                continue
            pt = (int(np.clip(kpts[i, 0], 0, w - 1)), int(np.clip(kpts[i, 1], 0, h - 1)))
            cv2.circle(out, pt, point_radius, color, -1)

    draw_points(left_kpts, color_left)
    draw_points(right_kpts, color_right)
    return out


def kpts_to_json_serializable(kpts: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    if kpts is None or kpts.size == 0:
        return None
    return [[float(x), float(y), float(c)] for x, y, c in kpts.tolist()]


def main():
    ap = argparse.ArgumentParser(description="Extract ViTPose 21-point hand keypoints and render points only (no lines)")
    ap.add_argument("--root", type=str, required=True, help="Video output root, e.g. .../video32")
    ap.add_argument("--video_name", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None, help="default: <root>/vis_hamer_full")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--kp_conf", type=float, default=0.25)
    args = ap.parse_args()

    root = Path(args.root)
    video_name = args.video_name or root.name
    out_dir = Path(args.out_dir) if args.out_dir else (root / "vis_hamer_full")
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_dir = root / "depth"
    if not depth_dir.exists():
        raise FileNotFoundError(f"depth dir not found: {depth_dir}")

    video_path = depth_dir / f"{video_name}_src.mp4"
    if not video_path.exists():
        mp4s = list(depth_dir.glob("*.mp4"))
        video_path = mp4s[0] if mp4s else None
    if not video_path or not Path(video_path).exists():
        raise FileNotFoundError(f"no source video in {depth_dir}")

    cpm = _get_vitpose_model(args.device)
    detector = _get_body_detector(args.device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    hand21_left: Dict[str, List[List[float]]] = {}
    hand21_right: Dict[str, List[List[float]]] = {}
    prev_wrist_L = None
    prev_wrist_R = None

    json_out = out_dir / f"{video_name}_hand21.json"
    mp4_out = out_dir / f"{video_name}_hamer21.mp4"

    max_preload = 2000
    depths = [_load_depth(depth_dir, i) for i in range(min(total_frames, max_preload))] if total_frames <= max_preload else None

    writer = None
    for fidx in range(total_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        depth = depths[fidx] if depths else _load_depth(depth_dir, fidx)
        if depth is None:
            gray = np.zeros((H, W), dtype=np.float32)
        else:
            gray = depth.astype(np.float32)

        hands = _get_hand21_from_frame(
            frame, gray, cpm, detector=detector,
            kp_conf_thr=args.kp_conf, verbose=(fidx % 50 == 0)
        )
        hands = fix_left_right_identity_hand21(hands, prev_wrist_L, prev_wrist_R)

        frame_key = str(fidx + 1)
        if hands["left"] is not None:
            hand21_left[frame_key] = kpts_to_json_serializable(hands["left"])
            prev_wrist_L = (float(hands["left"][0, 0]), float(hands["left"][0, 1]))
        else:
            prev_wrist_L = prev_wrist_L
        if hands["right"] is not None:
            hand21_right[frame_key] = kpts_to_json_serializable(hands["right"])
            prev_wrist_R = (float(hands["right"][0, 0]), float(hands["right"][0, 1]))
        else:
            prev_wrist_R = prev_wrist_R

        vis = draw_hand_points(frame, hands["left"], hands["right"], kp_conf_thr=args.kp_conf)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(mp4_out), fourcc, float(fps), (W, H))
        writer.write(vis)

        if (fidx + 1) % 50 == 0 or fidx == 0:
            print(f"[HAND21] frame {fidx + 1}/{total_frames}")

    cap.release()
    if writer:
        writer.release()

    result = {"left": hand21_left, "right": hand21_right}
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] hand21 JSON: {json_out}")
    print(f"[SAVE] skeleton video: {mp4_out}")


if __name__ == "__main__":
    main()
