"""
Debug visualization: replay the full detection pipeline for a single video,
annotate each frame with GroundingDINO boxes + ViTPose wrists + depth info,
and output a diagnostic video, CSV, and key-frame grid.

Usage (inside Docker):
    python3 debug_pipeline.py --video video12
    python3 debug_pipeline.py --video video12 --dataset EgoDex_long
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import UnivariateSpline
import re
from pathlib import Path

sys.path.append('/home/Egoloc/Egolocx')  # 将 /home 路径添加到模块搜索路径中
sys.path.append('/home/Egoloc')
sys.path.append('/home/EgoLoc/Grounded-Segment-Anything/GroundingDINO')  # 必需
from groundingdino.util.inference import load_model, load_image, predict
from EgoLocx.script.long_metric import evaluate_all
from EgoLocx.script.compute_metric import evaluate_predictions
import tempfile
from egoloc_speed_twohands import extract_3d_speed_and_visualize  # 新封装的生成速度文件的函数
from egoloc_speed_twohands import batch_process_videos  # 对文件夹内的所有视频执行extract_3d_speed_and_visualize
from typing import List, Optional   # ← 新增这一行
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

_model = load_model(
    "/home/EgoLoc/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "/home/EgoLoc/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"  # 直接放在weights目录外
)
from egoloc_speed_twohands import (
    _model,
    load_image,
    predict,
    _get_vitpose_model,
    _get_body_detector,
    _load_depth,
    _pixel_to_camera,
    depth_from_hand_roi_meters,
    fix_left_right_identity,
    get_twohand_boxes_groundingdino,
    _wrist_from_frame,
)

COLORS = {
    "boxL": (0, 255, 0),       # green
    "boxR": (0, 0, 255),       # red
    "wristL": (255, 180, 0),   # cyan-ish
    "wristR": (255, 0, 255),   # magenta
    "ok": (0, 180, 0),
    "partial": (0, 200, 255),  # yellow-ish (BGR)
    "crash": (0, 0, 255),
    "text": (255, 255, 255),
}


def draw_box(frame, box, color, label=""):
    if box is None:
        return
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    if label:
        cv2.putText(frame, label, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                     0.5, color, 1, cv2.LINE_AA)


def draw_wrist(frame, pt, color, label=""):
    if pt is None:
        return
    u, v = int(pt[0]), int(pt[1])
    cv2.circle(frame, (u, v), 8, color, -1)
    cv2.circle(frame, (u, v), 10, color, 2)
    if label:
        cv2.putText(frame, label, (u + 12, v + 4), cv2.FONT_HERSHEY_SIMPLEX,
                     0.45, color, 1, cv2.LINE_AA)


def draw_status_bar(frame, text, bar_color):
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (W, 32), bar_color, -1)
    cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def classify_frame(boxL, boxR, wristL, wristR, zL_m, zR_m, prev_zL, prev_zR):
    """Return a status string for this frame."""
    dino_L = boxL is not None
    dino_R = boxR is not None
    has_wL = wristL is not None
    has_wR = wristR is not None

    if has_wL and zL_m is None and prev_zL is None:
        return "CRASH_L"
    if has_wR and zR_m is None and prev_zR is None:
        return "CRASH_R"

    if not dino_L and not dino_R:
        if has_wL or has_wR:
            return "wrist_only"
        return "dino_miss"

    if (dino_L or not has_wL) and (dino_R or not has_wR) and (has_wL or has_wR):
        return "ok"

    return "partial"


def status_color(status):
    if status.startswith("CRASH"):
        return COLORS["crash"]
    if status == "ok":
        return COLORS["ok"]
    return COLORS["partial"]


def main():
    parser = argparse.ArgumentParser(description="Debug pipeline visualization")
    parser.add_argument("--video", default="video12", help="Video name (e.g. video12)")
    parser.add_argument("--dataset", default="EgoDex_short", help="Dataset folder name")
    parser.add_argument("--base", default="/home/EgoLoc/hand_data_drawer",
                        help="Base data dir")
    parser.add_argument("--out", default=None, help="Output dir (default: base/debug_grids/VIDEO)")
    args = parser.parse_args()

    vname = args.video
    base = Path(args.base)
    depth_dir = base / args.dataset / vname / "depth"

    video_path = None
    for candidate in [
        
        base / args.dataset / f"{vname}.mp4",
        Path(f"/data/EgoLoc/EgoDex/short/{vname}.mp4"),
    ]:
        if candidate.exists():
            video_path = candidate
            break
    if video_path is None:
        sys.exit(f"Cannot find video file for {vname}")

    out_dir = Path(args.out) if args.out else base / "debug_grids" / vname
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] Video:  {video_path}")
    print(f"[DEBUG] Depth:  {depth_dir}")
    print(f"[DEBUG] Output: {out_dir}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[DEBUG] Frames: {total_frames}, FPS: {fps:.1f}, Size: {W_vid}x{H_vid}")

    cpm = _get_vitpose_model("cuda")
    detector = _get_body_detector()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_out = cv2.VideoWriter(str(out_dir / f"debug_{vname}.mp4"),
                               fourcc, fps, (W_vid, H_vid))

    csv_path = out_dir / f"debug_{vname}.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow([
        "frame", "dino_detections", "boxL", "boxR",
        "wristL", "wristR", "zL_m", "zR_m",
        "prev_zL", "prev_zR", "status"
    ])

    prev_box_L = None
    prev_box_R = None
    prev_zL = None
    prev_zR = None
    prev_uL = None
    prev_uR = None

    key_frames = {}
    had_ok = False
    counters = {"ok": 0, "partial": 0, "dino_miss": 0, "wrist_only": 0,
                "CRASH_L": 0, "CRASH_R": 0}

    def maybe_store(tag, idx, frame_vis):
        if tag not in key_frames:
            key_frames[tag] = (idx, frame_vis.copy())

    for idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break

        depth = _load_depth(depth_dir, idx)

        boxL, boxR, prev_box_L, prev_box_R = get_twohand_boxes_groundingdino(
            frame, model=_model, load_image_fn=load_image, predict_fn=predict,
            box_thresh=0.30, text_thresh=0.25, expand_ratio=0.20,
            prev_box_L=prev_box_L, prev_box_R=prev_box_R,
        )

        dino_count = (1 if boxL is not None else 0) + (1 if boxR is not None else 0)

        zL_m = depth_from_hand_roi_meters(depth, boxL, prev_z=prev_zL) if boxL is not None else None
        zR_m = depth_from_hand_roi_meters(depth, boxR, prev_z=prev_zR) if boxR is not None else None

        wrists = _wrist_from_frame(frame, depth, cpm, detector=detector, verbose=False)
        wrists = fix_left_right_identity(wrists, prev_uL, prev_uR)
        wristL = wrists["left"]
        wristR = wrists["right"]

        status = classify_frame(boxL, boxR, wristL, wristR, zL_m, zR_m, prev_zL, prev_zR)
        counters[status] = counters.get(status, 0) + 1

        prev_zL = zL_m if zL_m is not None else prev_zL
        prev_zR = zR_m if zR_m is not None else prev_zR
        prev_uL = wristL if wristL is not None else prev_uL
        prev_uR = wristR if wristR is not None else prev_uR

        vis = frame.copy()
        draw_box(vis, boxL, COLORS["boxL"], f"L z={zL_m:.3f}" if zL_m else "L z=None")
        draw_box(vis, boxR, COLORS["boxR"], f"R z={zR_m:.3f}" if zR_m else "R z=None")
        draw_wrist(vis, wristL, COLORS["wristL"], "wL")
        draw_wrist(vis, wristR, COLORS["wristR"], "wR")

        info = f"#{idx}  DINO:{dino_count}  status:{status}"
        depth_info = f"  zL={'%.3f'%zL_m if zL_m else 'None'}  zR={'%.3f'%zR_m if zR_m else 'None'}"
        draw_status_bar(vis, info + depth_info, status_color(status))

        cv2.putText(vis, f"F{idx}/{total_frames}", (W_vid - 140, H_vid - 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1, cv2.LINE_AA)

        vid_out.write(vis)

        def fmt_pt(pt):
            return f"({pt[0]:.1f},{pt[1]:.1f})" if pt else ""

        def fmt_box(b):
            return f"[{b[0]},{b[1]},{b[2]},{b[3]}]" if b else ""

        writer.writerow([
            idx, dino_count, fmt_box(boxL), fmt_box(boxR),
            fmt_pt(wristL), fmt_pt(wristR),
            f"{zL_m:.4f}" if zL_m else "",
            f"{zR_m:.4f}" if zR_m else "",
            f"{prev_zL:.4f}" if prev_zL else "",
            f"{prev_zR:.4f}" if prev_zR else "",
            status,
        ])

        if idx == 0:
            maybe_store("first_frame", idx, vis)
        if status == "ok" and not had_ok:
            had_ok = True
            maybe_store("first_ok", idx, vis)
        if dino_count == 0:
            maybe_store("first_dino_miss", idx, vis)
            if had_ok:
                maybe_store("first_dino_miss_after_ok", idx, vis)
        if (wristL is not None or wristR is not None) and dino_count == 0:
            maybe_store("first_wrist_only", idx, vis)
        if status.startswith("CRASH"):
            maybe_store("first_crash", idx, vis)
        key_frames["last_frame"] = (idx, vis.copy())

        if (idx + 1) % 20 == 0 or idx == total_frames - 1:
            print(f"  [{idx+1}/{total_frames}] status={status} dino={dino_count} "
                  f"wL={'Y' if wristL else 'N'} wR={'Y' if wristR else 'N'} "
                  f"zL={zL_m} zR={zR_m}")

    cap.release()
    vid_out.release()
    csv_f.close()

    print(f"\n{'='*60}")
    print(f"SUMMARY for {vname} ({total_frames} frames)")
    print(f"{'='*60}")
    for k, v in sorted(counters.items()):
        if v > 0:
            print(f"  {k:>15s}: {v:4d}  ({100*v/total_frames:.1f}%)")

    print(f"\nKey frames captured:")
    for tag, (fidx, _) in sorted(key_frames.items(), key=lambda x: x[1][0]):
        print(f"  {tag:>30s}: frame {fidx}")

    tags_order = ["first_frame", "first_ok", "first_dino_miss",
                  "first_dino_miss_after_ok", "first_wrist_only",
                  "first_crash", "last_frame"]
    panels = [(tag, key_frames[tag]) for tag in tags_order if tag in key_frames]

    if panels:
        n = len(panels)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        thumb_w, thumb_h = 480, int(480 * H_vid / W_vid)
        grid_w = cols * thumb_w
        grid_h = rows * (thumb_h + 28)
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i, (tag, (fidx, img)) in enumerate(panels):
            r, c = divmod(i, cols)
            thumb = cv2.resize(img, (thumb_w, thumb_h))
            y0 = r * (thumb_h + 28)
            grid[y0:y0+thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb
            label = f"{tag} (f{fidx})"
            cv2.putText(grid, label, (c * thumb_w + 4, y0 + thumb_h + 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        grid_path = out_dir / f"debug_{vname}_grid.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"\nGrid saved: {grid_path}")

    print(f"Video saved: {out_dir / f'debug_{vname}.mp4'}")
    print(f"CSV saved:   {csv_path}")


if __name__ == "__main__":
    main()
