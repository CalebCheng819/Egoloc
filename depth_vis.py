import json
import numpy as np

import cv2
from pathlib import Path
import re

def images_to_video(image_dir: Path,
                    out_path: Path,
                    fps: int = 10,
                    pattern: str = r"depth_wrist_(\d+)\.png"):
    """
    将 image_dir 中的 depth_wrist_XXXXXX.png 合成为一个视频

    pattern 用正则表达式提取帧号，确保按帧顺序排序
    """
    image_dir = Path(image_dir)
    out_path = Path(out_path)

    # 找到所有符合 pattern 的 PNG
    png_list = []
    for p in image_dir.glob("*.png"):
        m = re.match(pattern, p.name)
        if m:
            frame_idx = int(m.group(1))
            png_list.append((frame_idx, p))

    if not png_list:
        print(f"[WARN] No images matched pattern '{pattern}' in {image_dir}")
        return

    # 按帧号排序
    png_list.sort(key=lambda x: x[0])

    # 读取第一帧确定分辨率
    _, first_img_path = png_list[0]
    frame0 = cv2.imread(str(first_img_path))
    if frame0 is None:
        print(f"[ERR] Cannot read first image {first_img_path}")
        return

    h, w, _ = frame0.shape

    # 创建 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # 写入帧
    for idx, img_path in png_list:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Skip unreadable image {img_path}")
            continue

        # 防御性检查：如果尺寸不一致，统一 resize
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)

    writer.release()
    print(f"[SAVE VIDEO] {out_path}")


def depth_to_vis(depth: np.ndarray,
                 *,
                 near_percentile: float = 1.0,
                 far_percentile: float = 99.0,
                 blur_ksize: int = 3) -> np.ndarray:
    """
    把 float32 depth 转成伪彩色 BGR 图，但：
      - 用分位数裁剪，避免极远/极近点拉低对比度
      - 轻微平滑，轮廓更干净一点

    depth: (H, W) float32, 单位 m
    """
    d = depth.copy()

    # 1) 用分位数定义显示范围，而不是直接用 min/max
    #    比如 1%~99% 的范围映射到 0~255
    vmin = np.percentile(d[np.isfinite(d)], near_percentile)
    vmax = np.percentile(d[np.isfinite(d)], far_percentile)
    if vmax <= vmin:
        vmax = vmin + 1e-3

    d = np.clip(d, vmin, vmax)
    d_norm = (d - vmin) / (vmax - vmin)

    d_8u = (d_norm * 255).astype(np.uint8)

    # 2) 轻微高斯平滑一下，避免太噪
    if blur_ksize > 1:
        d_8u = cv2.GaussianBlur(d_8u, (blur_ksize, blur_ksize), 0)

    # 3) 伪彩
    vis = cv2.applyColorMap(d_8u, cv2.COLORMAP_JET)

    return vis


def crop_around_point(img: np.ndarray,
                      u: float,
                      v: float,
                      crop_size: int = 160) -> np.ndarray:
    """
    以 (u, v) 为中心，从 img (H, W, 3) 裁剪一个正方形区域。
    crop_size: 裁剪窗口边长（像素）
    """
    H, W = img.shape[:2]
    half = crop_size // 2

    cx = int(round(u))
    cy = int(round(v))

    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(W, cx + half)
    y1 = min(H, cy + half)

    patch = img[y0:y1, x0:x1].copy()
    return patch

def visualize_wrist_on_depth(
    depth_dir: Path,
    wrist2d_L_path: Path,
    wrist2d_R_path: Path,
    out_dir: Path,
    radius: int = 5,
    crop_size: int = 160,
):
    """
    生成两类可视化：
      1) 全局深度图 + 左右手 wrist 圆点
      2) 对每一帧（如果有 wrist），输出局部放大图：
         depth_wrist_L_zoom_xxxxxx.png / depth_wrist_R_zoom_xxxxxx.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    wrist2d_L = json.loads(wrist2d_L_path.read_text(encoding="utf-8"))
    wrist2d_R = json.loads(wrist2d_R_path.read_text(encoding="utf-8"))

    depth_files = sorted(depth_dir.glob("pred_depth_*.npy"))
    print(f"[INFO] found {len(depth_files)} depth frames in {depth_dir}")

    for f in depth_files:
        idx = int(f.stem.split("_")[-1])        # e.g. pred_depth_000123.npy -> 123
        frame_key = str(idx + 1)                # 1-based

        depth = np.load(f).astype(np.float32)
        vis_full = depth_to_vis(depth)

        H, W = depth.shape

        # ===== 1) 全图上画 wrist 点 =====
        has_any = False

        if frame_key in wrist2d_L:
            uL, vL = wrist2d_L[frame_key]
            uL_i = int(np.clip(uL, 0, W - 1))
            vL_i = int(np.clip(vL, 0, H - 1))
            cv2.circle(vis_full, (uL_i, vL_i), radius, (0, 0, 255), thickness=-1)  # 左手：红
            has_any = True

        if frame_key in wrist2d_R:
            uR, vR = wrist2d_R[frame_key]
            uR_i = int(np.clip(uR, 0, W - 1))
            vR_i = int(np.clip(vR, 0, H - 1))
            cv2.circle(vis_full, (uR_i, vR_i), radius, (0, 255, 0), thickness=-1)  # 右手：绿
            has_any = True

        out_path_full = out_dir / f"depth_wrist_{idx:06d}.png"
        cv2.imwrite(str(out_path_full), vis_full)
        if has_any:
            print("[SAVE FULL]", out_path_full)

        # ===== 2) 局部放大图：左手 =====
        if frame_key in wrist2d_L:
            uL, vL = wrist2d_L[frame_key]
            patchL = crop_around_point(vis_full, uL, vL, crop_size=crop_size)

            # 再放大一倍，看得更清楚
            patchL_zoom = cv2.resize(
                patchL,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_NEAREST,
            )

            # 在放大图中心再画一个大一点的圆圈+十字
            hL, wL = patchL_zoom.shape[:2]
            cxL, cyL = wL // 2, hL // 2
            cv2.circle(patchL_zoom, (cxL, cyL), 8, (0, 0, 255), thickness=2)
            cv2.line(patchL_zoom, (cxL - 10, cyL), (cxL + 10, cyL), (255, 255, 255), 1)
            cv2.line(patchL_zoom, (cxL, cyL - 10), (cxL, cyL + 10), (255, 255, 255), 1)

            out_patch_L = out_dir / f"depth_wrist_L_zoom_{idx:06d}.png"
            cv2.imwrite(str(out_patch_L), patchL_zoom)
            print("[SAVE L-ZOOM]", out_patch_L)

        # ===== 3) 局部放大图：右手 =====
        if frame_key in wrist2d_R:
            uR, vR = wrist2d_R[frame_key]
            patchR = crop_around_point(vis_full, uR, vR, crop_size=crop_size)

            patchR_zoom = cv2.resize(
                patchR,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_NEAREST,
            )

            hR, wR = patchR_zoom.shape[:2]
            cxR, cyR = wR // 2, hR // 2
            cv2.circle(patchR_zoom, (cxR, cyR), 8, (0, 255, 0), thickness=2)
            cv2.line(patchR_zoom, (cxR - 10, cyR), (cxR + 10, cyR), (255, 255, 255), 1)
            cv2.line(patchR_zoom, (cxR, cyR - 10), (cxR, cyR + 10), (255, 255, 255), 1)

            out_patch_R = out_dir / f"depth_wrist_R_zoom_{idx:06d}.png"
            cv2.imwrite(str(out_patch_R), patchR_zoom)
            print("[SAVE R-ZOOM]", out_patch_R)

if __name__ == "__main__":
    video_name = "video1"
    root = Path("/home/EgoLoc/hand_data_drawer/twohands_test_out14") / video_name

    depth_dir = root / "depth"
    wrist2d_dir = root / "hand2d_px"   # 按你保存 2D JSON 的目录改
    out_dir = root / "depth_wrist_vis" # 输出可视化目录

    visualize_wrist_on_depth(
        depth_dir=depth_dir,
        wrist2d_L_path=wrist2d_dir / f"{video_name}_left.json",
        wrist2d_R_path=wrist2d_dir / f"{video_name}_right.json",
        out_dir=out_dir,
        radius=4,
        crop_size=160,
    )
    image_dir = root / "depth_wrist_vis"
    out_video = root / "depth_wrist_vis.mp4"

    images_to_video(image_dir, out_video, fps=10)
