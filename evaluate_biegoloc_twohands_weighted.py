#!/usr/bin/env python3
"""
评估 BiEgoLoc 左右手预测结果（参考 EgoLoc_long_twohands.py 的评估方式）：
- 标注从 JSON 读取（short.json）
- 加权按左右手接触对数量：把左右手全部预测与标注放在一起，每个 (GT, 预测对) 算一个样本，
  在全体样本上求平均，等价于按左右手贡献的样本数自然加权
- 左右手预测都为空时，该视频不纳入计算（不产生任何样本）
"""

import json
import os
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


# 与 compute_metric.evaluate_single_video 一致的指标计算
def evaluate_single_video(
    ground_truth: Tuple[int, int],
    prediction: Tuple[int, int],
    total_frame: int,
) -> Optional[Dict[str, float]]:
    gt_contact, gt_separation = ground_truth
    pred_contact, pred_separation = prediction

    if gt_contact == 0 and gt_separation == 0:
        return None
    if pred_contact == 0 and pred_separation == 0:
        return None

    if gt_contact == 0:
        contact_error = None
        contact_correct1 = contact_correct2 = contact_correct3 = None
    else:
        contact_error = abs(gt_contact - pred_contact)
        contact_correct1 = contact_error <= 1
        contact_correct2 = contact_error <= 3
        contact_correct3 = contact_error <= 5

    if gt_separation == 0:
        separation_error = None
        separation_correct1 = separation_correct2 = separation_correct3 = None
    else:
        separation_error = abs(gt_separation - pred_separation)
        separation_correct1 = separation_error <= 1
        separation_correct2 = separation_error <= 3
        separation_correct3 = separation_error <= 5

    total_checks = 0
    acc1 = acc2 = acc3 = 0
    if contact_error is not None:
        acc1 += int(contact_correct1)
        acc2 += int(contact_correct2)
        acc3 += int(contact_correct3)
        total_checks += 1
    if separation_error is not None:
        acc1 += int(separation_correct1)
        acc2 += int(separation_correct2)
        acc3 += int(separation_correct3)
        total_checks += 1
    accuracy1 = acc1 / total_checks if total_checks else 0.0
    accuracy2 = acc2 / total_checks if total_checks else 0.0
    accuracy3 = acc3 / total_checks if total_checks else 0.0

    errors = [e for e in (contact_error, separation_error) if e is not None]
    temporal_error = float(np.mean(errors)) if errors else 0.0
    temporal_similarity = 1.0 / (1 + temporal_error) if errors else np.nan

    MoF = IoU = None
    if gt_contact and gt_separation:
        pred_has_event = pred_contact > 0 and pred_separation > 0
        correct = 0
        for f in range(total_frame):
            in_gt = gt_contact <= f <= gt_separation
            in_pred = pred_has_event and (pred_contact <= f <= pred_separation)
            if in_gt == in_pred:
                correct += 1
        MoF = correct / total_frame
        gt_set = set(range(gt_contact, gt_separation + 1))
        pred_set = set(range(pred_contact, pred_separation + 1)) if pred_has_event else set()
        union = gt_set | pred_set
        IoU = len(gt_set & pred_set) / len(union) if union else 0.0

    return {
        "Accuracy_1": accuracy1,
        "Accuracy_2": accuracy2,
        "Accuracy_3": accuracy3,
        "Temporal Error (MAE)": temporal_error,
        "Temporal Similarity": temporal_similarity,
        "MoF": MoF,
        "IoU": IoU,
    }


def _to_video_key(name: str) -> str:
    name = (name or "").strip()
    name = os.path.basename(name)
    if not name.endswith(".mp4"):
        name = name + ".mp4"
    return name


def _normalize_pairs(pairs_any: Any) -> List[Tuple[int, int]]:
    """将 GT 中的 left/right 转为 [(c,s), ...]"""
    out: List[Tuple[int, int]] = []
    if pairs_any is None:
        return out
    if isinstance(pairs_any, (list, tuple)):
        for p in pairs_any:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((int(p[0]), int(p[1])))
            elif isinstance(p, dict):
                c = p.get("contact", p.get("c"))
                s = p.get("separation", p.get("s"))
                if c is not None and s is not None:
                    out.append((int(c), int(s)))
    return out


def load_gt_from_short_json(gt_path: str) -> Dict[str, Dict[str, Any]]:
    """
    从 short.json 加载 GT。
    支持格式：
    A) 单对/视频: {"video1.mp4": {"total_frames": n, "contact": c, "separation": s}, ...}
       或 list of dict 含 video_name, total_frames, contact, separation
    B) 左右手分开（与 EgoLoc long_metric 一致）:
       {"video1": {"total_frames": n, "left": [[c,s],...], "right": [[c,s],...]}, ...}
    返回: { "video1.mp4": {"total_frames": int, "contact": int, "separation": int,
            "left": [(c,s),...] or None, "right": [(c,s),...] or None}, ... }
    若为 B 则 contact/separation 取自 left 第一对（兼容下游），left/right 为列表。
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, Dict[str, Any]] = {}

    def put(key: str, total: int, c: int, s: int, left: Optional[List[Tuple[int, int]]], right: Optional[List[Tuple[int, int]]]):
        out[key] = {
            "total_frames": int(total),
            "contact": int(c),
            "separation": int(s),
            "left": left,
            "right": right,
        }

    if isinstance(data, dict):
        for k, v in data.items():
            key = _to_video_key(k)
            if not isinstance(v, dict):
                continue
            total = v.get("total_frames", 0)
            left_raw = v.get("left")
            right_raw = v.get("right")
            left_pairs = _normalize_pairs(left_raw) if left_raw is not None else None
            right_pairs = _normalize_pairs(right_raw) if right_raw is not None else None

            if left_pairs is not None or right_pairs is not None:
                # 有 left/right 则用第一对作为 contact/separation 兜底
                c, s = 0, 0
                if left_pairs:
                    c, s = left_pairs[0]
                elif right_pairs:
                    c, s = right_pairs[0]
                put(key, total, c, s, left_pairs, right_pairs)
            else:
                c = v.get("contact", v.get("c", v.get("contact_frame", 0)))
                s = v.get("separation", v.get("s", v.get("separation_frame", 0)))
                put(key, total, int(c), int(s), None, None)
        return out

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = item.get("video_name", item.get("video", item.get("name", "")))
                total = item.get("total_frames", 0)
                left_raw = item.get("left")
                right_raw = item.get("right")
                left_pairs = _normalize_pairs(left_raw) if left_raw is not None else None
                right_pairs = _normalize_pairs(right_raw) if right_raw is not None else None
                key = _to_video_key(name)
                if left_pairs is not None or right_pairs is not None:
                    c, s = (left_pairs[0] if left_pairs else right_pairs[0])
                    put(key, total, c, s, left_pairs, right_pairs)
                else:
                    c = item.get("contact", item.get("c", 0))
                    s = item.get("separation", item.get("s", 0))
                    put(key, total, int(c), int(s), None, None)
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                name, total, c, s = item[0], item[1], item[2], item[3]
                key = _to_video_key(name)
                put(key, int(total), int(c), int(s), None, None)
        return out

    raise ValueError(f"Unsupported GT JSON structure: {type(data)}")


def load_predictions_json(json_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    预测 JSON 格式: [["video1.mp4", [[c,s], ...]], ...]
    返回: {"video1.mp4": [(c,s), ...], ...}，空列表表示该视频无预测。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[str, List[Tuple[int, int]]] = {}
    for name, lst in raw:
        key = _to_video_key(name)
        if not lst:
            out[key] = []
            continue
        pairs = []
        for p in lst:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pairs.append((int(p[0]), int(p[1])))
        out[key] = pairs
    return out


def get_first_pair(pairs: List[Tuple[int, int]]) -> Tuple[int, int]:
    if not pairs:
        return (0, 0)
    return pairs[0]


def evaluate_twohands_pooled(
    gt_path: str,
    pred_left_path: str,
    pred_right_path: str,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    左右手全部预测与标注放在一起计算：
    - 每个视频若左手有预测则产生一个样本 (GT, 左手第一对, total_frames)；
      若右手有预测则产生一个样本 (GT, 右手第一对, total_frames)。
    - 若该视频左右手都为空，则不产生任何样本（该视频不纳入计算）。
    - 对所有样本逐条算指标，再在全体样本上求平均，等价于按左右手接触对数自然加权。
    """
    gt = load_gt_from_short_json(gt_path)
    pred_left = load_predictions_json(pred_left_path)
    pred_right = load_predictions_json(pred_right_path)

    metric_keys = [
        "Accuracy_1", "Accuracy_2", "Accuracy_3",
        "Temporal Error (MAE)", "Temporal Similarity", "MoF", "IoU",
    ]

    # 样本池：每个元素是 (gt_pair, pred_pair, total_frame, hand_label)
    samples: List[Tuple[Tuple[int, int], Tuple[int, int], int, str]] = []
    skipped_both_empty = 0
    n_left_samples = 0
    n_right_samples = 0

    for key in sorted(gt.keys()):
        info = gt[key]
        total_frame = info["total_frames"]
        # 优先用 per-hand GT（left/right 第一对），否则用统一的 contact/separation
        gt_left = get_first_pair(info.get("left") or [])
        gt_right = get_first_pair(info.get("right") or [])
        gt_fallback = (info["contact"], info["separation"])
        gt_pair_left = gt_left if gt_left != (0, 0) else gt_fallback
        gt_pair_right = gt_right if gt_right != (0, 0) else gt_fallback

        left_pairs = pred_left.get(key, [])
        right_pairs = pred_right.get(key, [])

        pred_l = get_first_pair(left_pairs)
        pred_r = get_first_pair(right_pairs)

        left_empty = pred_l == (0, 0) or not left_pairs
        right_empty = pred_r == (0, 0) or not right_pairs

        if left_empty and right_empty:
            skipped_both_empty += 1
            continue

        if not left_empty:
            samples.append((gt_pair_left, pred_l, total_frame, "left"))
            n_left_samples += 1
        if not right_empty:
            samples.append((gt_pair_right, pred_r, total_frame, "right"))
            n_right_samples += 1

    # 对每个样本算指标，再在全体样本上求平均
    all_metrics: List[Dict[str, float]] = []
    for gt_pair, pred_pair, total_frame, _ in samples:
        m = evaluate_single_video(gt_pair, pred_pair, total_frame)
        if m is not None:
            all_metrics.append(m)

    avg = {}
    if all_metrics:
        for k in metric_keys:
            vals = [m[k] for m in all_metrics if k in m and m[k] is not None]
            avg[k] = float(np.nanmean(vals)) if vals else np.nan
    else:
        for k in metric_keys:
            avg[k] = np.nan

    info = {
        "videos_in_gt": len(gt),
        "total_samples": len(samples),
        "n_left_samples": n_left_samples,
        "n_right_samples": n_right_samples,
        "skipped_both_empty": skipped_both_empty,
    }
    return avg, info


def main():
    parser = argparse.ArgumentParser(
        description="BiEgoLoc 左右手评估：左右手全部预测与标注放在一起计算（按接触对数自然加权），排除双手都为空视频"
    )
    parser.add_argument(
        "--gt",
        default="/data/EgoLoc/EgoDex/short/short.json",
        help="标注 JSON 路径（short.json）",
    )
    parser.add_argument(
        "--pred_left",
        default="/home/chengjuntao/data0/EgoLoc/predictions_BiEgoLoc_left.json",
        help="左手预测 JSON",
    )
    parser.add_argument(
        "--pred_right",
        default="/home/chengjuntao/data0/EgoLoc/predictions_BiEgoLoc_right.json",
        help="右手预测 JSON",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.gt):
        print(f"GT 文件不存在: {args.gt}")
        return
    if not os.path.isfile(args.pred_left):
        print(f"左手预测文件不存在: {args.pred_left}")
        return
    if not os.path.isfile(args.pred_right):
        print(f"右手预测文件不存在: {args.pred_right}")
        return

    avg, info = evaluate_twohands_pooled(
        args.gt,
        args.pred_left,
        args.pred_right,
    )

    print("--- BiEgoLoc 左右手评估（按接触对数合并计算）---")
    print(f"GT: {args.gt}")
    print(f"Pred Left:  {args.pred_left}")
    print(f"Pred Right: {args.pred_right}")
    print(f"GT 视频数: {info['videos_in_gt']}")
    print(f"总样本数（接触对）: {info['total_samples']}（左手 {info['n_left_samples']}，右手 {info['n_right_samples']}）")
    print(f"排除双手都为空视频数: {info['skipped_both_empty']}")
    print("\n--- 全体样本平均指标 ---")
    for k, v in avg.items():
        if isinstance(v, float) and np.isfinite(v):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("---")


if __name__ == "__main__":
    main()
