import json
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import os
from typing import Dict, List, Tuple, Any
import os
import json
from typing import Any, Dict, List, Optional

def _as_int_list(x: Any) -> List[int]:
    """把各种可能的 minima 表达转成 int list。"""
    out: List[int] = []
    if x is None:
        return out

    # 单个数字
    if isinstance(x, (int, float)):
        try:
            xi = int(x)
            if xi >= 0:
                out.append(xi)
        except Exception:
            pass
        return out

    # 字符串：可能是 "123" 或 "1,2,3"
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return out
        # "1,2,3"
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            for p in parts:
                if p.isdigit():
                    out.append(int(p))
            return out
        # "123"
        if s.isdigit():
            out.append(int(s))
        return out

    # list/tuple
    if isinstance(x, (list, tuple)):
        for item in x:
            out.extend(_as_int_list(item))
        return out

    # dict：不直接当 minima 列表
    return out


def load_minima_json(minima_json_path: str) -> Dict[str, List[int]]:
    """
    读取 minima proposals JSON，并统一成：
      { "videoX.mp4": [minima_frame_int, ...], ... }

    兼容结构：
      A) dict: {video: [minima...]}
      B) dict: {video: {"minima":[...]} } / {"frames":[...]} / {"minima_frames":[...]} / {"indices":[...]}
      C) list of dict: [{"video":..., "minima":[...]}, ...] (video字段名可变)
      D) list of [video, minima]
    """
    with open(minima_json_path, "r") as f:
        data = json.load(f)

    minima_by_video: Dict[str, List[int]] = {}

    def _put(video_key_raw: Any, minima_raw: Any):
        video_key = _to_video_key(video_key_raw)
        if video_key is None:
            return
        mins = _as_int_list(minima_raw)
        # 去重、排序、过滤负数
        mins = sorted({int(m) for m in mins if isinstance(m, int) and m >= 0})
        minima_by_video[video_key] = mins

    # -------- A/B: dict keyed by video --------
    if isinstance(data, dict):
        for k, v in data.items():
            # A: v 就是 list
            if isinstance(v, (list, tuple, int, float, str)):
                _put(k, v)
                continue

            # B: v 是 dict，minima 在字段里
            if isinstance(v, dict):
                # 常见字段名兜底
                minima_raw = (
                    v.get("minima", None)
                    or v.get("minima_frames", None)
                    or v.get("frames", None)
                    or v.get("indices", None)
                    or v.get("minima_indices", None)
                )
                # 有些人会写 {"left":[...], "right":[...]} ——这种你也可以按需扩展
                _put(k, minima_raw)
                continue

            # 其他类型忽略
        return minima_by_video

    # -------- C/D: list --------
    if isinstance(data, list):
        for item in data:
            # D: ["video1.mp4", [..]]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                _put(item[0], item[1])
                continue

            # C: {"video":..., "minima":[...]}
            if isinstance(item, dict):
                video_key_raw = (
                    item.get("video", None)
                    or item.get("video_name", None)
                    or item.get("name", None)
                    or item.get("path", None)
                    or item.get("video_path", None)
                )
                minima_raw = (
                    item.get("minima", None)
                    or item.get("minima_frames", None)
                    or item.get("frames", None)
                    or item.get("indices", None)
                    or item.get("minima_indices", None)
                )
                _put(video_key_raw, minima_raw)
                continue

        return minima_by_video

    raise ValueError(f"Unrecognized minima json structure: {type(data)}")

def _to_video_key(k: str) -> str:
    """统一 video key 格式：保证以 .mp4 结尾"""
    if k is None:
        return None
    k = str(k).strip()
    # 有的 GT 可能给 "video1" 或 "/path/video1.mp4"
    k = os.path.basename(k)
    if not k.endswith(".mp4"):
        k = k + ".mp4"
    return k

def _normalize_pairs(pairs: Any) -> List[Tuple[int, int]]:
    """
    把 pairs 规范成 [(c,s),...] 的 int tuple list。
    允许 pairs = [[c,s], ...] 或 [{"contact":c,"separation":s}, ...]
    """
    out = []
    if pairs is None:
        return out

    # case1: list
    if isinstance(pairs, list):
        for item in pairs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                c, s = item[0], item[1]
            elif isinstance(item, dict):
                # 兼容 dict 格式
                c = item.get("contact", item.get("c", None))
                s = item.get("separation", item.get("s", None))
            else:
                continue

            try:
                c = int(c); s = int(s)
            except Exception:
                continue

            # 过滤明显非法：负数/倒序（你也可以保留倒序让后续处理）
            if c < 0 or s < 0:
                continue
            out.append((c, s))

    # 排序 + 去重
    out = sorted(set(out), key=lambda x: (x[0], x[1]))
    return out

def load_gt_pairs_from_your_gt_json(gt_json_path: str, hand: str = "left") -> Dict[str, List[Tuple[int, int]]]:
    """
    把你 GT json 解析成:
      { "videoX.mp4": [(c,s), ...], ... }

    hand: "left" or "right"
    """
    with open(gt_json_path, "r") as f:
        gt = json.load(f)

    hand = str(hand).lower()
    gt_by_video: Dict[str, List[Tuple[int, int]]] = {}

    # ----------- 结构 B / C：dict keyed by video -----------
    if isinstance(gt, dict):
        # B: gt["video1.mp4"] = {"left": [...], "right": [...]}
        # C: gt["video1.mp4"] = [[c,s], ...]
        for k, v in gt.items():
            video_key = _to_video_key(k)
            if video_key is None:
                continue

            if isinstance(v, dict):
                # B
                pairs_raw = v.get(hand, v.get(hand.capitalize(), None))
            else:
                # C
                pairs_raw = v

            pairs = _normalize_pairs(pairs_raw)
            gt_by_video[video_key] = pairs

        return gt_by_video

    # ----------- 结构 A：list of dict -----------
    if isinstance(gt, list):
        for item in gt:
            if not isinstance(item, dict):
                continue

            # 常见字段名：video / video_name / name / path
            video_key = _to_video_key(
                item.get("video", None) or item.get("video_name", None) or item.get("name", None) or item.get("path", None)
            )
            if video_key is None:
                continue

            # pairs 可能就在 item[hand]，也可能在 item["gt"][hand] 等
            pairs_raw = None

            if hand in item:
                pairs_raw = item[hand]
            elif "gt" in item and isinstance(item["gt"], dict) and hand in item["gt"]:
                pairs_raw = item["gt"][hand]
            else:
                # 兜底：如果没有按手分，试试 item["pairs"]
                pairs_raw = item.get("pairs", None)

            pairs = _normalize_pairs(pairs_raw)
            gt_by_video[video_key] = pairs

        return gt_by_video

    raise ValueError(f"Unrecognized GT json structure: {type(gt)}")

def load_ground_truth_json_twohands(gt_json_path, hand="right"):
    """
    支持你的 GT JSON 格式：
    {
      "video1": {
        "total_frames": 339,
        "left": [...],
        "right": [...]
      }
    }
    返回格式统一为：
    {
      "video1.mp4": {
          "total_frames": int,
          "pairs": [(c,s), ...]
      }
    }
    """
    assert hand in ["left", "right"]

    with open(gt_json_path, "r") as f:
        data = json.load(f)

    gt_dict = {}
    for vkey, info in data.items():
        total = int(info["total_frames"])
        pairs = [tuple(p) for p in info.get(hand, [])]

        # 对齐 prediction key
        json_key = vkey if vkey.endswith(".mp4") else f"{vkey}.mp4"

        gt_dict[json_key] = {
            "total_frames": total,
            "pairs": pairs
        }
    return gt_dict

def load_predictions(json_path):
    """
    从 JSON 文件读取预测结果。
    支持两种格式：
    1. {"video1.mp4": [[c1,s1],[c2,s2],…], ...}
    2. [["video1.mp4", [[c1,s1],[c2,s2],…]], ...]
    返回一个 dict： key 是视频名，value 是 [(c,s),…] 列表。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            # 转成 dict
            return {k: v for k, v in data}
        return data

def load_ground_truth_rowwise(gt_excel_path, sheet_name="Sheet9"):
    """
    从 Excel 按行读取 GT。要求：
      - 表头（第 1 行）包含列名： video_name, total_frames, c1, s1, c2, s2, …
      - 从第 2 行开始，video_name 列里是数字 ID：1,2,3,…
      - total_frames 在第 2 列；c1,s1,c2,s2…从第 3 列开始，每两个一组。
    返回：
      gt_dict: {
        '1': {'total_frames': int, 'pairs': [(c1,s1),(c2,s2),…]},
        '2': {…}, …
      }
    注意这里把 video_name 列读成 str，以便后面拼接成 JSON key。
    """
    df = pd.read_excel(
        gt_excel_path,
        sheet_name=sheet_name,
        dtype={'video_name': str}    # 把第一列强制读成字符串
    )
    gt_dict = {}
    for _, row in df.iterrows():
        vid_id = row['video_name']           # e.g. "1", "2", …
        total = int(row['total_frames'])     
        # 从第 3 列开始读所有非空值，并转成 int 列表
        vals = row.iloc[2:].dropna().astype(int).tolist()
        # 每两个值打包成 (contact_frame, separation_frame)
        pairs = list(zip(vals[0::2], vals[1::2]))
        gt_dict[vid_id] = {
            'total_frames': total,
            'pairs': pairs
        }
    return gt_dict

def match_pairs(gt_pairs, pred_pairs, matching_tolerance):
    """
    按 pair-level 进行匹配：如果 pred_pair 和 某个 gt_pair
    在 contact 和 separation 两点上都在 tolerance 内，就认为匹配上。
    返回列表 of (pred_index, gt_index)。
    """
    matched_gt = set()
    matched_pred = set()
    matches = []
    for i, pred_pair in enumerate(pred_pairs):
        for j, gt_pair in enumerate(gt_pairs):
            if j in matched_gt:
                continue
            if (abs(pred_pair[0] - gt_pair[0]) <= matching_tolerance and
                abs(pred_pair[1] - gt_pair[1]) <= matching_tolerance):
                matches.append((i, j))
                matched_gt.add(j)
                matched_pred.add(i)
                break
    return matches

def compute_mof(gt_pairs, pred_pairs, total_frames):
    """
    计算 Mean over Frames (MoF):
    MoF = 正确预测为动作帧数 / 总帧数
    只要预测区间和GT区间有重叠的帧都算正确。
    """
    gt_mask = np.zeros(total_frames, dtype=bool)
    for c, s in gt_pairs:
        gt_mask[c:s+1] = True
    pred_mask = np.zeros(total_frames, dtype=bool)
    for c, s in pred_pairs:
        pred_mask[c:s+1] = True
    correct = np.logical_and(gt_mask, pred_mask).sum()
    mof = correct / total_frames if total_frames > 0 else None
    return mof
import json
from collections import defaultdict

def _flatten_gt_events(gt_pairs):
    """gt_pairs: [(c,s), ...] -> contacts[], separations[]"""
    contacts = []
    separations = []
    for c, s in gt_pairs:
        if c is not None:
            contacts.append(int(c))
        if s is not None:
            separations.append(int(s))
    return contacts, separations

def _hit_any(target_frame, proposal_frames, tol):
    tf = int(target_frame)
    for p in proposal_frames:
        if abs(int(p) - tf) <= tol:
            return True
    return False

def evaluate_minima_json(
    gt_by_video,         # dict: video.mp4 -> [(c,s), ...]
    minima_by_video,     # dict: video.mp4 -> [minima frames]
    tol=5,
):
    """
    Stage-1 Proposal Recall:
    - 对 GT 的每个 contact/separation 帧，看 minima 是否在 ±tol 覆盖
    """

    total_contact = total_sep = 0
    hit_contact = hit_sep = 0

    per_video = {}

    for vid, gt_pairs in gt_by_video.items():
        minima = minima_by_video.get(vid, [])
        minima = sorted(set(int(x) for x in minima if isinstance(x, int) or str(x).isdigit()))

        gt_contacts, gt_seps = _flatten_gt_events(gt_pairs)

        vc_total_c = len(gt_contacts)
        vc_total_s = len(gt_seps)
        vc_hit_c = sum(_hit_any(f, minima, tol) for f in gt_contacts)
        vc_hit_s = sum(_hit_any(f, minima, tol) for f in gt_seps)

        total_contact += vc_total_c
        total_sep += vc_total_s
        hit_contact += vc_hit_c
        hit_sep += vc_hit_s

        per_video[vid] = {
            "minima_count": len(minima),
            "gt_contact": vc_total_c,
            "gt_sep": vc_total_s,
            "hit_contact": vc_hit_c,
            "hit_sep": vc_hit_s,
            "recall_contact": (vc_hit_c / vc_total_c) if vc_total_c > 0 else None,
            "recall_sep": (vc_hit_s / vc_total_s) if vc_total_s > 0 else None,
        }

    recall_contact = (hit_contact / total_contact) if total_contact > 0 else None
    recall_sep = (hit_sep / total_sep) if total_sep > 0 else None

    total_events = total_contact + total_sep
    hit_events = hit_contact + hit_sep
    recall_all = (hit_events / total_events) if total_events > 0 else None

    return {
        "tol": tol,
        "recall_all": recall_all,
        "recall_contact": recall_contact,
        "recall_separation": recall_sep,
        "total_contact": total_contact,
        "total_separation": total_sep,
        "per_video": per_video,
    }

def compute_iou(gt_pairs, pred_pairs, total_frames):
    """
    计算 IoU (Intersection over Union) for action frames.
    """
    gt_mask = np.zeros(total_frames, dtype=bool)
    for c, s in gt_pairs:
        gt_mask[c:s+1] = True
    pred_mask = np.zeros(total_frames, dtype=bool)
    for c, s in pred_pairs:
        pred_mask[c:s+1] = True
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou = intersection / union if union > 0 else None
    return iou

def evaluate_video(
    gt_pairs, pred_pairs, total_frames,
    sr_tolerances=(1,3,5),    # 用于SR@1/3/5
    psr_tolerance=6           # 用于PSR
):
    """
    评估单个视频：
      - SR@1, SR@3, SR@5: 以预测为主，预测的每个接触/分离点分别在GT同类点中是否有命中（在tol内），分母为预测点总数
      - PSR: pair级recall（GT每一对在matching_tolerance下是否被预测对命中）
      - mae: 以预测为主，预测每个接触/分离点到最近GT同类点的平均误差，分母为预测点总数
      - MoF: Mean over Frames
      - IoU: Intersection over Union
    """
    # 拆分预测和GT的接触/分离点
    pred_contacts = [c for c, s in pred_pairs]
    pred_separations = [s for c, s in pred_pairs]
    gt_contacts = [c for c, s in gt_pairs]
    gt_separations = [s for c, s in gt_pairs]

    # SR@tol（point级）
    sr_results = {}
    for tol in sr_tolerances:
        contact_hits = 0
        for pred_c in pred_contacts:
            if gt_contacts and any(abs(gt_c - pred_c) <= tol for gt_c in gt_contacts):
                contact_hits += 1
        separation_hits = 0
        for pred_s in pred_separations:
            if gt_separations and any(abs(gt_s - pred_s) <= tol for gt_s in gt_separations):
                separation_hits += 1
        total_pred_points = len(pred_contacts) + len(pred_separations)
        total_hits = contact_hits + separation_hits
        sr_results[f"SR@{tol}"] = total_hits / total_pred_points if total_pred_points else None

    # PSR (pair segment recall, 以GT为主)
    matched_gt = set()
    for i, gt_pair in enumerate(gt_pairs):
        for pred_pair in pred_pairs:
            if (abs(pred_pair[0] - gt_pair[0]) <= psr_tolerance and
                abs(pred_pair[1] - gt_pair[1]) <= psr_tolerance):
                matched_gt.add(i)
                break
    psr = len(matched_gt) / len(gt_pairs) if gt_pairs else None

    # MAE（point级）
    mae_sum = 0
    for pred_c in pred_contacts:
        if gt_contacts:
            closest_gt = min(gt_contacts, key=lambda gt_c: abs(gt_c - pred_c))
            mae_sum += abs(closest_gt - pred_c)
    for pred_s in pred_separations:
        if gt_separations:
            closest_gt = min(gt_separations, key=lambda gt_s: abs(gt_s - pred_s))
            mae_sum += abs(closest_gt - pred_s)
    total_pred_points = len(pred_contacts) + len(pred_separations)
    mae = mae_sum / total_pred_points if total_pred_points else None

    # MoF & IoU
    mof = compute_mof(gt_pairs, pred_pairs, total_frames)
    iou = compute_iou(gt_pairs, pred_pairs, total_frames)

    return {
        **sr_results,
        "PSR": psr,
        "mae": mae,
        "MoF": mof,
        "IoU": iou,
    }
def evaluate_all(
    pred_json,
    gt_json,
    hand="right",
    sr_tolerances=(1,3,5),
    psr_tolerance=6,
    videos_to_exclude=None,
):
    """
    videos_to_exclude: 不纳入计算的视频 key 集合（如 GT 左右手都为空的视频）。
    """
    preds = load_predictions(pred_json)
    gts = load_ground_truth_json_twohands(gt_json, hand=hand)
    if videos_to_exclude is None:
        videos_to_exclude = set()

    all_metrics = []
    for video_key, gt_info in gts.items():
        if video_key in videos_to_exclude:
            continue
        pred_pairs = preds.get(video_key, [])
        gt_pairs = gt_info["pairs"]
        total_frames = gt_info["total_frames"]

        metrics = evaluate_video(
            gt_pairs, pred_pairs, total_frames,
            sr_tolerances=sr_tolerances,
            psr_tolerance=psr_tolerance
        )
        all_metrics.append(metrics)

    if not all_metrics:
        return {k: None for k in ["SR@1", "SR@3", "SR@5", "PSR", "mae", "MoF", "IoU"]}
    final = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics if m[key] is not None]
        final[key] = np.mean(vals) if vals else None
    return final


def evaluate_all_stages_pooled_twohands(
    pred_left_path,
    pred_right_path,
    gt_json_path,
    sr_tolerances=(1, 3, 5),
    psr_tolerance=10,
):
    """
    左右手一起评估（按接触对合并）：不采用 (left_metric + right_metric) / 2，
    而是把每个 (video, hand) 当作一个样本，逐样本算 stage3 指标后对全体样本求平均，
    等价于按左右手贡献的接触对数量自然加权。
    - 仅对「该手标注非空」的 (video, hand) 纳入：若某视频左手/右手标注为空（如 "left": []），则该手不纳入计算，预测也不计入。
    返回: {"hand": "both", "stage3": {SR@1, SR@3, SR@5, PSR, mae, MoF, IoU}}
    """
    preds_left = load_predictions(pred_left_path)
    preds_right = load_predictions(pred_right_path)
    gts_left = load_ground_truth_json_twohands(gt_json_path, hand="left")
    gts_right = load_ground_truth_json_twohands(gt_json_path, hand="right")

    all_videos = set(gts_left.keys()) | set(gts_right.keys()) | set(preds_left.keys()) | set(preds_right.keys())
    samples = []  # list of (gt_pairs, pred_pairs, total_frames)

    for video in all_videos:
        total = None
        if video in gts_left:
            total = gts_left[video].get("total_frames")
        if total is None and video in gts_right:
            total = gts_right[video].get("total_frames")
        if total is None:
            continue

        gt_l = gts_left.get(video, {}).get("pairs", [])
        pred_l = preds_left.get(video, [])
        gt_r = gts_right.get(video, {}).get("pairs", [])
        pred_r = preds_right.get(video, [])

        # 仅当该手标注非空时才加入样本（标注为空则该手不纳入，包括预测也不计入）
        if gt_l:
            samples.append((gt_l, pred_l, total))
        if gt_r:
            samples.append((gt_r, pred_r, total))

    if not samples:
        keys = ["SR@1", "SR@3", "SR@5", "PSR", "mae", "MoF", "IoU"]
        return {"hand": "both", "stage3": {k: None for k in keys}}

    all_metrics = []
    for gt_pairs, pred_pairs, total_frames in samples:
        m = evaluate_video(
            gt_pairs, pred_pairs, total_frames,
            sr_tolerances=sr_tolerances,
            psr_tolerance=psr_tolerance,
        )
        all_metrics.append(m)

    final = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics if m[key] is not None]
        final[key] = float(np.mean(vals)) if vals else None
    return {"hand": "both", "stage3": final}


def evaluate_all_stages(
    pred_json_path,
    gt_json_path,
    hand,
    minima_json_path=None,
    tol=5,
    videos_to_exclude=None,
):
    # Stage-3：保持你原 evaluate_all 的调用方式；可排除 GT 左右手都为空的视频
    stage3 = evaluate_all(
        pred_json=pred_json_path,
        gt_json=gt_json_path,
        hand=hand,
        sr_tolerances=(1,3,5),
        psr_tolerance=10,
        videos_to_exclude=videos_to_exclude,
    )

    out = {"hand": hand, "stage3": stage3}

    # Stage-1：需要把 GT 转成 gt_by_video（video -> pairs）
    if minima_json_path is not None:
        minima_by_video = load_minima_json(minima_json_path)

        # 这一步你要写一个解析器，把 gt_json 转成 {video.mp4: [(c,s),...]}
        gt_by_video = load_gt_pairs_from_your_gt_json(gt_json_path, hand=hand)

        stage1 = evaluate_minima_json(gt_by_video, minima_by_video, tol=tol)
        out["stage1"] = stage1

    return out

# def evaluate_all(pred_json, gt_xlsx, sheet_name="Sheet9", sr_tolerances=(1,3,5), psr_tolerance=6):
#     """
#     整体评估：按 video_id 拼接 JSON key，然后对每个视频调用 evaluate_video，
#     最后对各项指标做平均。
#     返回 dict: {point_accuracy:…, mae:…, SR@1:…, SR@3:…, SR@5:…, MoF:…, IoU:…}
#     """
#     # 1. 读预测
#     preds = load_predictions(pred_json)
#
#     # 2. 读 GT
#     gts = load_ground_truth_rowwise(gt_xlsx, sheet_name)
#
#     # 3. 逐视频评估
#     all_metrics = []
#     for vid_id, gt_info in gts.items():
#         json_key = f"video{vid_id}.mp4"
#         pred_pairs = preds.get(json_key, [])
#         gt_pairs = gt_info['pairs']
#         total_frames = gt_info['total_frames']
#         metrics = evaluate_video(
#             gt_pairs, pred_pairs, total_frames,
#             sr_tolerances=sr_tolerances,
#             psr_tolerance=psr_tolerance
#         )
#         all_metrics.append(metrics)
#
#     # 4. 对每个指标做平均
#     final = {}
#     for key in all_metrics[0].keys():
#         vals = [m[key] for m in all_metrics if m[key] is not None]
#         final[key] = np.mean(vals) if vals else None
#     return final

# if __name__ == "__main__":
#     results = evaluate_all(
#         pred_json="/home/VLM-Video-Action-Localization-main/VLM-Video-Action-Localization-main/result/EgoLoc_long_video2.json",
#         gt_xlsx="/home/EgoLoc/ground_truth/KitchenCounter1.xlsx",
#         matching_tolerance=8,        # 用于 pair 匹配
#         evaluation_tolerance=5,      # 用于 accuracy 和 MAE 精度计算
#         sheet_name="Sheet10"
#     )






