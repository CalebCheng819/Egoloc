#!/usr/bin/env python3
"""
仅做 long 评测、不跑预测：读取已有预测 JSON 与标注 JSON，调用 long_metric 输出 left / right / both-pooled 的 stage3 指标。
适用于：已有 predictions_left.json、predictions_right.json 和 result.json 格式的 GT，只想单独评测。
"""

import os
import sys
import argparse

# 保证能导入 EgoLocx（按项目根或容器路径）
_EGOLOC = os.path.dirname(os.path.abspath(__file__))
if _EGOLOC not in sys.path:
    sys.path.insert(0, _EGOLOC)

from EgoLocx.script.long_metric import (
    evaluate_all_stages,
    evaluate_all_stages_pooled_twohands,
    load_ground_truth_json_twohands,
)


def main():
    parser = argparse.ArgumentParser(
        description="Long 评测：仅用已有预测与标注，输出 left / right / both-pooled 的 stage3"
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="标注 JSON 路径（result.json 格式：video_id -> total_frames, left, right）",
    )
    parser.add_argument(
        "--pred_left",
        required=True,
        help="左手预测 JSON 路径",
    )
    parser.add_argument(
        "--pred_right",
        required=True,
        help="右手预测 JSON 路径",
    )
    parser.add_argument(
        "--minima_left",
        default=None,
        help="左手 minima JSON（可选，用于 stage1）",
    )
    parser.add_argument(
        "--minima_right",
        default=None,
        help="右手 minima JSON（可选，用于 stage1）",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.gt):
        print(f"GT 文件不存在: {args.gt}")
        sys.exit(1)
    if not os.path.isfile(args.pred_left):
        print(f"左手预测文件不存在: {args.pred_left}")
        sys.exit(1)
    if not os.path.isfile(args.pred_right):
        print(f"右手预测文件不存在: {args.pred_right}")
        sys.exit(1)

    print("--- Long 评测（仅评测，不跑预测）---")
    print(f"GT:         {args.gt}")
    print(f"Pred Left:  {args.pred_left}")
    print(f"Pred Right: {args.pred_right}")
    print()

    # 加载标注；仅对「该手存在标注」的 (视频, 手) 计算，标注为空（如 "left": []）的该手不纳入，预测也不计入
    gts_left = load_ground_truth_json_twohands(args.gt, hand="left")
    gts_right = load_ground_truth_json_twohands(args.gt, hand="right")
    all_gt_videos = set(gts_left.keys()) | set(gts_right.keys())
    # 左手标注为空的视频不参与左手计算
    videos_to_exclude_left = {v for v in all_gt_videos if not gts_left.get(v, {}).get("pairs", [])}
    # 右手标注为空的视频不参与右手计算
    videos_to_exclude_right = {v for v in all_gt_videos if not gts_right.get(v, {}).get("pairs", [])}
    if videos_to_exclude_left:
        print(f"左手标注为空，不纳入左手计算: {len(videos_to_exclude_left)} 个视频 {sorted(videos_to_exclude_left)[:10]}{'...' if len(videos_to_exclude_left) > 10 else ''}")
    if videos_to_exclude_right:
        print(f"右手标注为空，不纳入右手计算: {len(videos_to_exclude_right)} 个视频 {sorted(videos_to_exclude_right)[:10]}{'...' if len(videos_to_exclude_right) > 10 else ''}")
    print(f"参与左手计算的视频数: {len(all_gt_videos) - len(videos_to_exclude_left)}；参与右手计算的视频数: {len(all_gt_videos) - len(videos_to_exclude_right)}")
    print()

    # 左手（仅计算左手标注非空的视频）
    r_left = evaluate_all_stages(
        pred_json_path=args.pred_left,
        gt_json_path=args.gt,
        hand="left",
        minima_json_path=args.minima_left,
        videos_to_exclude=videos_to_exclude_left,
    )
    print("Evaluation (left):")
    if "stage1" in r_left and r_left["stage1"]:
        for k, v in r_left["stage1"].items():
            if k != "per_video" and v is not None:
                print(f"  stage1.{k}: {v}")
    for k, v in (r_left.get("stage3") or {}).items():
        if v is not None:
            print(f"  stage3.{k}: {v:.4f}")
        else:
            print(f"  stage3.{k}: None")
    print()

    # 右手（仅计算右手标注非空的视频）
    r_right = evaluate_all_stages(
        pred_json_path=args.pred_right,
        gt_json_path=args.gt,
        hand="right",
        minima_json_path=args.minima_right,
        videos_to_exclude=videos_to_exclude_right,
    )
    print("Evaluation (right):")
    if "stage1" in r_right and r_right["stage1"]:
        for k, v in r_right["stage1"].items():
            if k != "per_video" and v is not None:
                print(f"  stage1.{k}: {v}")
    for k, v in (r_right.get("stage3") or {}).items():
        if v is not None:
            print(f"  stage3.{k}: {v:.4f}")
        else:
            print(f"  stage3.{k}: None")
    print()

    # 左右手合并（仅 stage3）
    r_both = evaluate_all_stages_pooled_twohands(
        pred_left_path=args.pred_left,
        pred_right_path=args.pred_right,
        gt_json_path=args.gt,
        sr_tolerances=(1, 3, 5),
        psr_tolerance=10,
    )
    print("Evaluation (both-pooled):  # 左右手全部 (video,hand) 样本一起算 stage3 再平均")
    for k, v in (r_both.get("stage3") or {}).items():
        if v is not None:
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: None")
    print("---")


if __name__ == "__main__":
    main()
