import json
import numpy as np
from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt
import cv2  # 用于把图片合成视频

# ==== 配置路径（按你的结构改一下 video_name / out_root）====
video_name = "video2"  # 举例
out_root = Path("/home/EgoLoc/hand_data_drawer/twohands_test_out10")  # 改成你的根目录

root = out_root / video_name
pcd_dir = root / "pointclouds" / video_name

cam_L_json = root / "registered_hands_left" / f"{video_name}.json"
cam_R_json = root / "registered_hands_right" / f"{video_name}.json"

out_vis_dir = root / "diagnostics_registered"
out_vis_dir.mkdir(parents=True, exist_ok=True)

# ==== 1. 读取相机系手腕轨迹 ====
cam_L = json.loads(cam_L_json.read_text(encoding="utf-8")) if cam_L_json.exists() else {}
cam_R = json.loads(cam_R_json.read_text(encoding="utf-8")) if cam_R_json.exists() else {}

def load_pcd_for_frame(pcd_dir: Path, frame_idx: int):
    """载入某一帧的点云：frame_idx 是 0-based，这里 PLY 文件名就是 idx.ply。"""
    ply_path = pcd_dir / f"{frame_idx}.ply"
    if not ply_path.exists():
        print(f"[WARN] no pcd for frame {frame_idx}: {ply_path}")
        return None
    pcd = o3d.io.read_point_cloud(str(ply_path))
    return pcd

def get_hand_xyz_for_frame(frame_idx: int):
    """
    从 hand3d_cam_{left,right} 里取这一帧的手腕坐标。
    JSON 用的是 1-based 索引，所以这里 +1。
    返回 (xyz_L or None, xyz_R or None)
    """
    key = str(frame_idx + 1)  # 1-based
    xyzL = cam_L.get(key, None)
    xyzR = cam_R.get(key, None)
    return xyzL, xyzR

def visualize_frame_cam_space(frame_idx: int):
    """
    在相机坐标系下，把这一帧的点云 + 左右手 3D 坐标画在一起，存成一张图。
    左手红色，右手绿色。
    """
    pcd = load_pcd_for_frame(pcd_dir, frame_idx)
    if pcd is None:
        return

    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        print(f"[WARN] empty point cloud for frame {frame_idx}")
        return

    xyzL, xyzR = get_hand_xyz_for_frame(frame_idx)

    fig = plt.figure(figsize=(12, 5))

    # ========= 子图 1：XZ 投影 =========
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(pts[:, 0], pts[:, 2], s=0.5, alpha=0.3, label="Point cloud")

    if xyzL is not None:
        xyzL = np.array(xyzL, dtype=float) / 1000.0  # ← 从毫米转成米
        ax1.scatter(xyzL[0], xyzL[2], c="red", s=50, label="Left wrist")

    if xyzR is not None:
        xyzR = np.array(xyzR, dtype=float) / 1000.0  # ← 同上
        ax1.scatter(xyzR[0], xyzR[2], c="green", s=50, label="Right wrist")

    ax1.set_title(f"Frame {frame_idx} – XZ projection (camera coord)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.axis("equal")
    ax1.legend()

    # ========= 子图 2：3D 散点 =========
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # 下采样一下点云，避免太密
    if pts.shape[0] > 40000:
        idx = np.random.choice(pts.shape[0], size=40000, replace=False)
        pts_small = pts[idx]
    else:
        pts_small = pts

    ax2.scatter(pts_small[:, 0], pts_small[:, 1], pts_small[:, 2],
                s=0.5, alpha=0.2, label="Point cloud")

    if xyzL is not None:
        ax2.scatter(xyzL[0], xyzL[1], xyzL[2],
                    c="red", s=50, label="Left wrist")
    if xyzR is not None:
        ax2.scatter(xyzR[0], xyzR[1], xyzR[2],
                    c="green", s=50, label="Right wrist")

    ax2.set_title("3D point cloud + both wrists (camera coord)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()

    plt.tight_layout()
    out_path = out_vis_dir / f"frame_cam_{frame_idx:04d}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[SAVE] {out_path}")
def get_all_frame_indices_from_pcd(pcd_dir: Path):
    """
    根据 pointclouds 目录下的 *.ply 文件名获取所有帧号（0-based）。
    假设文件名是 0.ply, 1.ply, ...
    """
    idxs = []
    for p in pcd_dir.glob("*.ply"):
        try:
            idx = int(p.stem)
            idxs.append(idx)
        except ValueError:
            continue
    idxs = sorted(set(idxs))
    return idxs

def visualize_all_frames():
    frame_indices = get_all_frame_indices_from_pcd(pcd_dir)
    print(f"[INFO] found {len(frame_indices)} frames in {pcd_dir}")
    for i in frame_indices:
        visualize_frame_cam_space(i)
def images_to_video(image_dir: Path, video_out_path: Path, fps: int = 10):
    """
    把 image_dir 下面 frame_cam_XXXX.png 按序合成一个 mp4 视频。
    """
    png_list = sorted(image_dir.glob("frame_cam_*.png"))
    if not png_list:
        print(f"[WARN] no images in {image_dir}")
        return

    # 读取第一张图片确定分辨率
    frame0 = cv2.imread(str(png_list[0]))
    if frame0 is None:
        print(f"[ERR] cannot read {png_list[0]}")
        return
    h, w, _ = frame0.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (w, h))

    for img_path in png_list:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] skip unreadable image {img_path}")
            continue
        # 保证大小一致（一般不需要，防御性写法）
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)

    writer.release()
    print(f"[SAVE VIDEO] {video_out_path}")
if __name__ == "__main__":
    # 1) 把所有帧的点云 + 左右手腕都画出来
    visualize_all_frames()

    # 2) 把这些 PNG 合成一个视频
    video_out = root / "diagnostics_registered_vis.mp4"
    images_to_video(out_vis_dir, video_out, fps=10)
