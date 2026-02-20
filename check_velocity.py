#这个脚本用于检测为何速度峰值过高的问题
import os
import json, numpy as np
from pathlib import Path
import matplotlib,cv2
import matplotlib.pyplot as plt
import open3d as o3d
matplotlib.use("Agg")          # 服务器无显示环境，固定用 Agg 后端

video_name = "video1"
out_root = Path("/home/EgoLoc/hand_data_drawer/twohands_test_out14")  # 改成你现在用的根目录

# # 相机系
# cam_json = out_root / video_name / "hand3d_cam" / f"{video_name}.json"
# cam_track = json.loads(cam_json.read_text())
#
# # 世界系
# reg_json = out_root / video_name / "registered_hands" / f"{video_name}.json"
# reg_track = json.loads(reg_json.read_text())

# 速度
speed_json = out_root / video_name / f"{video_name}_with_speed_twohands.json"
speed_pairs = json.loads(speed_json.read_text())  # [[frame, vL, vR], ...] 或 [[frame, v], ...]
data=json.loads(speed_json.read_text())

frames = [d[0] for d in speed_pairs]
vL = np.array([d[1] for d in speed_pairs], float)
vR = np.array([d[2] for d in speed_pairs], float)

def check_stats(name, v):
    v_pos = v[v > 0]
    print(f"=== {name} ===")
    print("min / max:", v_pos.min(), v_pos.max())
    print("mean / median:", v_pos.mean(), np.median(v_pos))
    print("95th percentile:", np.percentile(v_pos, 95))
    print("99th percentile:", np.percentile(v_pos, 99))

    plt.figure(figsize=(10,4))
    plt.plot(frames, v, label=name)
    plt.ylim(0, np.percentile(v_pos, 99)*1.2)
    plt.legend(); plt.xlabel("frame"); plt.ylabel("speed")
    plt.show()

check_stats("left", vL)
check_stats("right", vR)

def plot_twohand_speed(json_path, save_dir, img_name="speed_curve.png"):
    os.makedirs(save_dir, exist_ok=True)

    # 读取速度数据：[frame, vL, vR]
    import json
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = np.array([d[0] for d in data])
    vL     = np.array([d[1] for d in data])
    vR     = np.array([d[2] for d in data])

    # 画图 & 截断极大值，避免尾部尖峰把 y 轴拉满
    vmax = np.percentile(np.concatenate((vL, vR)), 99) * 1.3

    plt.figure(figsize=(12, 4))
    plt.plot(frames, vL, label="left")
    plt.plot(frames, vR, label="right")
    plt.ylim(0, vmax)
    plt.xlabel("Frame")
    plt.ylabel("Speed (relative)")
    plt.title("Left / Right Hand Speed")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, img_name)
    plt.savefig(out_path, dpi=200)   # ✅ 保存为图片
    plt.close()                      # 及时关闭，防止内存累积
    print("速度曲线已保存到:", out_path)

output_dir=out_root / video_name
plot_twohand_speed(speed_json,out_root / video_name)

data = json.loads(speed_json.read_text(encoding="utf-8"))
data = np.array(data, dtype=float)   # (N, 3): [frame, vL, vR]
frames = data[:, 0].astype(int)
vL = data[:, 1]
vR = data[:, 2]

def get_spike_indices(v, frames, name, p=99):
    """
    v: 速度数组 (N,)
    frames: 对应帧号 (N,)
    name: "Left hand" / "Right hand"
    p: 分位数（例如 96.5、99）
    """
    # 分位数作为临界值
    thr = np.percentile(v, p)

    # spike index
    spike_mask = v > thr
    spike_idx = np.where(spike_mask)[0]

    # 未超过 thr 的最大速度
    if np.any(~spike_mask):
        max_below = v[~spike_mask].max()
    else:
        max_below = None

    print(f"\n===== {name} =====")
    print(f"阈值 (percentile={p}): {thr:.4f}")
    print(f"Spike 个数: {len(spike_idx)}")
    print(f"未超过阈值的最大速度: {max_below:.4f}")
    print(f"Spike 帧号: {frames[spike_idx].tolist()}")

    return frames[spike_idx]
spikes_L = get_spike_indices(vL, frames, "Left hand", p=99)
spikes_R = get_spike_indices(vR, frames, "Right hand", p=99)
def expand_spikes(spikes, neighbor=3, max_frame=None):
    """
    返回 0-based 帧号（即 f-1）
    """
    expanded = set()

    # spikes 输入本来就是 frames[] 的值，即 0-based or 1-based?
    # 你当前 get_spike_indices 返回的是 frames[idx]，frames 是 0-based
    # 所以这里 spikes 是 0-based 不需要再 -1
    # 但你现在要求最终帧数“-1”，我在输出端统一处理

    if max_frame is None:
        max_frame = max(spikes) + neighbor + 1

    for f in spikes:
        start = max(0, f - neighbor)
        end   = min(max_frame, f + neighbor)
        for k in range(start, end + 1):
            expanded.add(k)

    expanded_sorted = sorted(list(expanded))

    #  核心：最终输出帧号 -1（但确保不会出现 -1 的情况）
    expanded_shifted = [max(0, f - 1) for f in expanded_sorted]

    return expanded_shifted


expanded_L = expand_spikes(spikes_L, neighbor=5, max_frame=int(frames.max()))
expanded_R = expand_spikes(spikes_R, neighbor=5, max_frame=int(frames.max()))

print("Left expanded:", expanded_L)
print("Right expanded:", expanded_R)

#点云和ICP结果检查
root = out_root / video_name

pcd_dir = root / "pointclouds"/video_name
reg_L_json = root / "registered_hands_left" / f"{video_name}.json"
reg_R_json = root / "registered_hands_right" / f"{video_name}.json"

out_vis_dir = root / "diagnostics"
out_vis_dir.mkdir(parents=True, exist_ok=True)

# 1) 读取注册后的 3D 轨迹
reg_L = json.loads(reg_L_json.read_text(encoding="utf-8"))
reg_R = json.loads(reg_R_json.read_text(encoding="utf-8"))

def dict_to_array(track_dict):
    """track_dict: { '1':[x,y,z], '2':[x,y,z], ... } -> (N,3), frame_idx list"""
    items = sorted(((int(k), v) for k, v in track_dict.items()), key=lambda x: x[0])
    frames = np.array([k for k, _ in items], dtype=int)
    xyz = np.array([v for _, v in items], dtype=float)
    return frames, xyz

frames_L, xyz_L = dict_to_array(reg_L)
frames_R, xyz_R = dict_to_array(reg_R)

# 2) 指定要检查的帧（你可以换成 spikes_L / spikes_R 的并集）
frames_to_check = sorted(set(expanded_L + expanded_R))

def load_pcd_for_frame(pcd_dir, frame_idx):
    ply_path = pcd_dir / f"{frame_idx}.ply"
    if not ply_path.exists():
        print(f"[WARN] {ply_path} not found")
        return None
    return o3d.io.read_point_cloud(str(ply_path))

def visualize_frame(frame_idx: int):
    # 点云
    pcd = load_pcd_for_frame(pcd_dir, frame_idx)
    if pcd is None:
        return

    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        print(f"[WARN] empty point cloud for frame {frame_idx}")
        return

    # 左右手轨迹中最近的 index
    def nearest_idx(frames, target):
        return int(np.argmin(np.abs(frames - target)))

    iL = nearest_idx(frames_L, frame_idx)
    iR = nearest_idx(frames_R, frame_idx)

    # 画图
    fig = plt.figure(figsize=(12, 5))

    # 2D 投影：XZ
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(pts[:, 0], pts[:, 2], s=1, alpha=0.5)
    ax1.set_title(f"Point cloud XZ @ frame {frame_idx}")
    ax1.set_xlabel("X"); ax1.set_ylabel("Z")
    ax1.axis("equal")

    # 轨迹：世界系 3D 轨迹的 XZ 平面投影
    ax2 = fig.add_subplot(1, 2, 2)
    # 左手完整轨迹
    ax2.plot(xyz_L[:, 0], xyz_L[:, 2], label="Left traj", alpha=0.5)
    ax2.scatter(xyz_L[iL, 0], xyz_L[iL, 2], c="red", s=50, label=f"L frame≈{frames_L[iL]}")
    # 右手完整轨迹
    ax2.plot(xyz_R[:, 0], xyz_R[:, 2], label="Right traj", alpha=0.5)
    ax2.scatter(xyz_R[iR, 0], xyz_R[iR, 2], c="green", s=50, label=f"R frame≈{frames_R[iR]}")

    ax2.set_title("Registered hand trajectories (XZ projection)")
    ax2.set_xlabel("X"); ax2.set_ylabel("Z")
    ax2.axis("equal")
    ax2.legend()

    plt.tight_layout()
    out_path = out_vis_dir / f"frame_{frame_idx:04d}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[SAVE] {out_path}")

# # === 实际调用：你可以把 frames_to_check 换成 spikes_L / spikes_R 的 union ===
# for f in frames_to_check:
#     visualize_frame(f)
# ---------- 合成视频 ----------
def pngs_to_video(png_dir: Path, out_mp4: Path, fps=10):
    images = sorted(png_dir.glob("frame_*.png"))
    if not images:
        print("No PNGs to make video.")
        return
    first = cv2.imread(str(images[0]))
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for im in images:
        frame = cv2.imread(str(im))
        vw.write(frame)
    vw.release()
    print("Saved video:", out_mp4)

pngs_to_video(out_vis_dir, out_vis_dir / "icp_overlap_check.mp4", fps=10)

