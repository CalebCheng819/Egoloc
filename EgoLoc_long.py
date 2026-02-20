# export HF_ENDPOINT=https://hf-mirror.com
# python ./EgoLoc_long.py --credentials auth.env  --grid_size 4  --video_type long

import numpy as np
import cv2
import base64
from openai import OpenAI, AzureOpenAI
import os
from PIL import Image
import math
import json
import dotenv
import time
import argparse
import openai
import pandas as pd
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
from egoloc_speed import extract_3d_speed_and_visualize  # 新封装的生成速度文件的函数
from egoloc_speed import batch_process_videos  # 对文件夹内的所有视频执行extract_3d_speed_and_visualize
from typing import List, Optional   # ← 新增这一行
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

_model = load_model(
    "/home/EgoLoc/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "/home/EgoLoc/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"  # 直接放在weights目录外
)


def run_groundingdino_and_crop(
        image: np.ndarray,
        text_prompt: str = "hand",
        box_thresh: float = 0.35,
        text_thresh: float = 0.25,
        expand_pixels: int = 10
) -> np.ndarray:
    """
    - 输入：OpenCV 读出的 BGR np.ndarray (H x W x C)
    - 输出：裁剪并 resize 回原始 HxW 的 BGR np.ndarray
    - expand_pixels: 裁剪框扩大的像素值
    """
    H, W = image.shape[:2]

    # 1) 将 np.ndarray → PIL, 再用 load_image + predict
    #    （GroundingDINO 只接受文件或 PIL，且内部会转 Tensor 到 GPU）
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        pil_img, tensor_img = load_image(tmp.name)

    # 2) predict 得到归一化的 [cx,cy,w,h]
    boxes_norm, logits, phrases = predict(
        model=_model,
        image=tensor_img,
        caption=text_prompt,
        box_threshold=box_thresh,
        text_threshold=text_thresh
    )
    # 如果没检测到
    if len(boxes_norm) == 0:
        return image

    # 3) 转回像素坐标并筛选合法框
    boxes_px = []
    for cx, cy, bw, bh in boxes_norm:
        x0 = (cx - bw / 2) * W
        y0 = (cy - bh / 2) * H
        x1 = (cx + bw / 2) * W
        y1 = (cy + bh / 2) * H
        # 强制转换为 int，并做边界裁剪
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 > x0 and y1 > y0:
            boxes_px.append([x0, y0, x1, y1])

    if not boxes_px:
        # 没有合法框，直接返回原图
        return image

    # 4) 选择面积最小的框
    areas = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes_px]
    idx_min = int(np.argmin(areas))
    x0, y0, x1, y1 = boxes_px[idx_min]

    # 5) 扩大裁剪框（上下左右各扩大expand_pixels像素）
    x0_expanded = max(0, x0 - expand_pixels)
    y0_expanded = max(0, y0 - expand_pixels)
    x1_expanded = min(W, x1 + expand_pixels)
    y1_expanded = min(H, y1 + expand_pixels)

    # 6) 裁剪 + resize 回原图
    crop = image[y0_expanded:y1_expanded, x0_expanded:x1_expanded]
    if crop.size == 0:
        return image
    resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
    return resized


def is_positive_feedback(feedback_result):
    if feedback_result is None:
        return False
    s = str(feedback_result).strip().lower()
    # 只要不是明确否定
    return not (s in ["0", "no", "None", "separation", "neither", "false"])


def save_predictions(all_predictions, file_path):
    with open(file_path, "w") as f:
        json.dump(all_predictions, f)


def load_predictions(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        with open(file_path, "w") as f:
            json.dump([], f)
        return []


def extract_event_info(response):
    # Define regex patterns for event type and frame number
    event_match = re.search(r"Event:\s*(Contact|Separation|Neither|None)", response, re.IGNORECASE)
    # print("event match:",event_match)
    if event_match:
        event_type = event_match.group(1).capitalize()  # Ensure proper capitalization
        return event_type
    else:
        return None


def extract_frame_info(response):
    frame_match = re.search(r"Frame:\s*(-?\d+)", response)
    if not frame_match:
        return -1
    try:
        return int(frame_match.group(1))
    except ValueError:
        # 如果转换失败也返回-1
        return -1


def enforce_min_distance(indices, min_dist, reference_values):
    if len(indices) == 0:
        return np.array([], dtype=int)

    # 按显著性排序（速度值越小越优先）
    sorted_indices = sorted(indices, key=lambda x: reference_values[x])

    filtered = []
    for idx in sorted_indices:
        # 检查与已选点的最小距离
        if all(abs(idx - exist) >= min_dist for exist in filtered):
            filtered.append(idx)

    # 按原始顺序返回
    return np.sort(np.array(filtered, dtype=int))


def extract_local_minima_frames(video_path, folder_path="/home/bathroomCabinet/3D_speed_VDA"):
    thresh = 0.08  # 异常值阈值
    savgol_polyorder = 2  # 提高多项式阶数以更好拟合曲线
    spline_s = 1e-4  # 增大样条平滑因子抑制噪声
    min_peak_prominence = 0.2  # 极小值的最小突出度（需根据数据调整）
    min_peak_distance = 5  # 极小值间最小间隔（单位：帧）
    savgol_window_ratio = 0.2

    filename = f"{video_path}_with_speed.json"
    file_path = os.path.join(folder_path, filename)
    if not os.path.isfile(file_path):
        print(f"❗ 文件未找到: {file_path}")
        return []
    match = re.search(r'\d+', video_path)
    video_id = match.group(0) if match else video_path

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❗ JSON 读取失败: {file_path}, 错误: {e}")
        return []
    if not isinstance(data, list) or len(data) == 0:
        print(f"❗ 数据为空: {file_path}")
        return []

    # 过滤异常值
    filtered_data = [
        (frame, speed) for frame, speed in data
        if isinstance(speed, (int, float)) and 0 < speed < thresh
    ]
    if len(filtered_data) < 4:
        print(f"⚠️ 视频 {video_id} 数据点太少（{len(filtered_data)}），跳过！")

    frames, speeds = zip(*filtered_data)

    # =================== Savitzky-Golay 动态窗口 ===================
    data_len = len(speeds)
    window_length = min(
        max(int(data_len * savgol_window_ratio), 3),
        data_len
    )
    window_length = window_length + 1 if window_length % 2 == 0 else window_length
    window_length = max(window_length, 3)

    try:
        speeds_smooth_sg = savgol_filter(
            speeds,
            window_length=window_length,
            polyorder=min(savgol_polyorder, window_length - 1),
            mode='nearest'
        )
    except Exception as e:
        print(f"⚠️ Savitzky-Golay 失败: {filename}, 错误: {e}")

    # =================== 样条插值平滑 ===================
    try:
        spl = UnivariateSpline(frames, speeds_smooth_sg, s=spline_s)
        frames_smooth = np.linspace(min(frames), max(frames), 300)
        speeds_smooth = spl(frames_smooth)
    except Exception as e:
        print(f"⚠️ 样条插值失败: {filename}, 错误: {e}")

    # =================== 极值检测 ===================
    try:
        # 方法1：直接找速度极小值
        minima, properties = find_peaks(
            -speeds_smooth,
            prominence=min_peak_prominence,
            distance=min_peak_distance
        )
        valid_minima = minima[properties['prominences'] > min_peak_prominence]

        # 方法2：通过加速度过零点找极小值
        velocity_derivative = np.gradient(speeds_smooth, frames_smooth)
        zero_crossings = np.where(np.diff(np.sign(velocity_derivative)))[0]
        minima_candidates = zero_crossings[np.diff(np.sign(velocity_derivative))[zero_crossings] > 0]

        # 合并候选点并强制间距限制
        all_candidates = np.union1d(valid_minima, minima_candidates).astype(int)
        all_minima = enforce_min_distance(all_candidates, min_peak_distance, speeds_smooth)

        # 映射到实际帧号（四舍五入+去重）
        extrema_frames = frames_smooth[all_minima]
        extrema_frames_int = np.unique(np.rint(extrema_frames)).astype(int)  # 关键步骤！
    except Exception as e:
        print(f"⚠️ 极值检测失败: {filename}, 错误: {e}")
    return extrema_frames_int.tolist()


def extract_local_minima_frames(video_path, folder_path="/home/bathroomCabinet/3D_hand_speed"):
    # 返回速度从慢到快的极小值点列表
    thresh = 1  # 异常值阈值
    savgol_polyorder = 2  # Savitzky-Golay 多项式阶数
    spline_s = 1e-4  # 样条插值平滑因子
    savgol_mode = 'nearest'  # 滤波模式

    json_filename = f"{video_path}_with_speed.json"
    file_path = os.path.join(folder_path, json_filename)
    if not os.path.isfile(file_path):
        print(f"❗ 文件未找到: {file_path}")
        return []
    match = re.search(r'\d+', video_path)
    video_id = match.group(0) if match else video_path

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❗ JSON 读取失败: {file_path}, 错误: {e}")
        return []
    if not isinstance(data, list) or len(data) == 0:
        print(f"❗ 数据为空: {file_path}")
        return []

    filtered_data = [
        (frame, speed)
        for frame, speed in data
        if isinstance(speed, (int, float)) and not np.isnan(speed) and 0 < speed < thresh]

    if len(filtered_data) < 4:
        print(f"⚠️ 视频 {video_id} 数据点太少（{len(filtered_data)}），跳过！")
        return []

    frames, speeds = zip(*filtered_data)
    data_len = len(speeds)
    # =================== Savitzky-Golay 滤波 ===================
    window_length = min(7, data_len)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        window_length = 3
    if window_length >= data_len:
        window_length = data_len - 1 if data_len % 2 == 0 else data_len
        if window_length < 3:
            window_length = 3

    try:
        speeds_smooth_sg = savgol_filter(
            speeds,
            window_length=window_length,
            polyorder=min(savgol_polyorder, window_length - 1),
            mode=savgol_mode
        )
    except Exception as e:
        print(f"⚠️ Savitzky-Golay 失败: {file_path}, 错误: {e}")
        return []

    # =================== 样条插值平滑 ===================
    try:
        spl = UnivariateSpline(frames, speeds_smooth_sg, s=spline_s)
        frames_smooth = np.linspace(min(frames), max(frames), 300)
        speeds_smooth = spl(frames_smooth)
    except Exception as e:
        print(f"⚠️ 样条插值失败: {file_path}, 错误: {e}")
        return []

    # =================== 极小值检测 ===================
    try:
        minima_indices, _ = find_peaks(-speeds_smooth)
        extrema_frames = frames_smooth[minima_indices]
        extrema_speeds = speeds_smooth[minima_indices]
    except Exception as e:
        print(f"⚠️ 极值检测失败: {file_path}, 错误: {e}")
        return []

    if len(extrema_frames) == 0:
        print(f"⚠️ 视频 {video_id} 未找到极小值！")
        return []

    # 获取排序索引
    extrema_frames_int = np.rint(extrema_frames).astype(int)
    # =================== 从原始数据查找速度值 ===================
    extrema_frame_speed_pairs = []
    for ef in extrema_frames_int:
        close_indices = np.where(np.isclose(frames, ef))[0]
        if len(close_indices) > 0:
            idx = close_indices[0]
            matched_frame = frames[idx]
            speed = speeds[idx]
            extrema_frame_speed_pairs.append((matched_frame, speed))
        else:
            print(f"⚠️ 极小值帧 {ef} 不在原始帧列表中，尝试找最近点")

            # 兜底方案：找最接近的帧
            idx_nearest = np.argmin(np.abs(frames - ef))
            nearest_frame = frames[idx_nearest]
            speed = speeds[idx_nearest]
            extrema_frame_speed_pairs.append((nearest_frame, speed))

    if not extrema_frame_speed_pairs:
        print(f"⚠️ 视频 {video_id} 没有匹配到任何有效极小值帧！")
        return []
    # =================== 按速度排序 ===================
    sorted_pairs = sorted(extrema_frame_speed_pairs, key=lambda x: x[1])  # 按速度排序
    sorted_frames = [pair[0] for pair in sorted_pairs]

    print(f"视频 {video_id} 极小值帧序号（按原始速度排序）: {sorted_frames}")

    return sorted_frames


def adaptive_sample(minima_indices, mode='linear', exp_k=0.5):
    n = len(minima_indices)
    if n == 0:
        raise ValueError("minima_indices 列表为空，无法采样！")

    if mode == 'linear':
        # 线性递减权重，从 1 到 0.1（避免出现0）
        weights = np.linspace(1, 0.1, n)

    elif mode == 'exp':
        # 指数衰减：exp(-k * rank)
        ranks = np.arange(n)
        weights = np.exp(-exp_k * ranks)

    else:
        raise ValueError("mode 参数必须是 'linear' 或 'exp'")

    # 归一化为概率分布
    probabilities = weights / np.sum(weights)

    # 打印概率调试
    # print(f"➡️ 权重: {weights}")
    # print(f"➡️ 概率: {probabilities}")

    # 按概率采样
    selected_frame = np.random.choice(minima_indices, p=probabilities)
    return selected_frame


def extract_local_minima_frames_adaptive(
        video_path,
        folder_path="/home/EgoLoc/hand_data_drawer/3D_hand_speed_hamer",
        # folder_path="/home/bathroomCabinet/3D_hand_speed",
        output_folder="./vis_speed_curve"
):
    """
    自适应版：根据总帧数和速度自动设置四个关键参数并提取极小值帧。
    返回：极小值帧list、极小值帧对应速度list
    """
    json_file = os.path.join(folder_path, f"{video_path}_with_speed.json")
    if not os.path.isfile(json_file):
        print(f"❗ 文件未找到: {json_file}")
        return [], []
    data = json.load(open(json_file, 'r'))
    if not isinstance(data, list) or len(data) == 0:
        print(f"❗ 数据为空或格式错误")
        return [], []

    frames_all, speeds_all = zip(*data)
    frames_all = np.array(frames_all, float)
    speeds_all = np.array(speeds_all, float)

    mask = np.isfinite(speeds_all) & (speeds_all > 0)
    frames = frames_all[mask];
    speeds = speeds_all[mask]
    N = len(speeds)
    if N < 4:
        print(f"⚠️ 有效数据太少: {N}")
        return [], []

    polyorder = max(2, int(N / 13))
    window_length = max(8, int(N / 7))
    if window_length % 2 == 0: window_length += 1
    window_length = min(window_length, N - (1 if N % 2 == 0 else 0))
    mean_speed = np.mean(speeds)
    spline_s = mean_speed * 1e-3
    min_prominence = mean_speed * 0.3
    min_peak_distance = max(int(N * 0.03), 2)
    try:
        speeds_sg = savgol_filter(
            speeds, window_length=window_length,
            polyorder=min(polyorder, window_length - 1), mode='nearest'
        )
    except:
        speeds_sg = speeds
    try:
        spl = UnivariateSpline(frames, speeds_sg, s=spline_s)
        frames_smooth = np.linspace(frames.min(), frames.max(), max(300, N))
        speeds_smooth = spl(frames_smooth)
    except:
        frames_smooth, speeds_smooth = frames, speeds_sg
    peaks, props = find_peaks(
        -speeds_smooth,
        prominence=min_prominence,
        distance=min_peak_distance
    )
    cand_frames = np.unique(np.rint(frames_smooth[peaks]).astype(int))
    pairs = []
    for f in cand_frames:
        idx = np.argmin(np.abs(frames - f))
        pairs.append((int(frames[idx]), float(speeds[idx])))
    pairs.sort(key=lambda x: x[1])
    selected = []
    for f, s in pairs:
        if all(abs(f - sf) >= min_peak_distance for sf, _ in selected):
            selected.append((f, s))
    result = [f for f, s in selected]
    result_speeds = [s for f, s in selected]
    print(f"🖨️ {video_path}提取到极小值帧（{len(result)}个）： {result}")
    return result, result_speeds


# 速度越小概率越高的采样
def adaptive_sample_speed(minima_indices, minima_speeds):
    speeds = np.array(minima_speeds)
    eps = 1e-8
    inv_speeds = 1 / (speeds + eps)
    probabilities = inv_speeds / np.sum(inv_speeds)
    selected_frame = np.random.choice(minima_indices, p=probabilities)
    return selected_frame


# 以极小值点为中心，前后各两帧共5帧，按速度加权采样关键帧索引
def sample_keyframe_around_minima(selected_minima, frames, speeds, total_frames, window=2):
    idx = np.where(frames == selected_minima)[0]
    if len(idx) == 0:
        return selected_minima  # fallback
    idx = idx[0]
    candidate_indices = []
    candidate_speeds = []
    for offset in range(-window, window + 1):
        candidate_idx = idx + offset
        if 0 <= candidate_idx < len(frames):
            candidate_indices.append(frames[candidate_idx])
            candidate_speeds.append(speeds[candidate_idx])
    candidate_speeds = np.array(candidate_speeds)
    eps = 1e-8
    inv_speeds = 1 / (candidate_speeds + eps)
    probabilities = inv_speeds / np.sum(inv_speeds)
    selected_keyframe = np.random.choice(candidate_indices, p=probabilities)
    return selected_keyframe


def select_frames_near_average(filter_indices, grid_size, total_frames, invalid_list):
    avg_index = round(np.mean(filter_indices))
    # 确保采样范围不超出 [0, total_frames-1]
    start_index, end_index = avg_index, avg_index
    # 生成选取的帧索引列表
    used_frame_indices = []
    if avg_index not in invalid_list:
        used_frame_indices.append(avg_index)
    # 如果选取的帧数量小于 grid_size^2，补充一些帧以满足要求
    while len(used_frame_indices) < grid_size ** 2:
        if len(used_frame_indices) < grid_size ** 2:
            # 如果当前帧不够，可以从左边或右边继续补充
            if start_index >= 0:
                if start_index > 0:
                    start_index -= 1
                    if start_index not in invalid_list:
                        used_frame_indices.insert(0, start_index)
                if start_index == 0:
                    used_frame_indices.insert(0, start_index)
            if len(used_frame_indices) < grid_size ** 2 and end_index <= total_frames - 1:
                if end_index < total_frames - 1:
                    end_index += 1
                    if end_index not in invalid_list:
                        used_frame_indices.append(end_index)
                if end_index == total_frames - 1:
                    used_frame_indices.append(end_index)

    # 确保最终的索引列表数量为 grid_size^2
    used_frame_indices = used_frame_indices[:grid_size ** 2]
    index = used_frame_indices.index(filter_indices)
    return used_frame_indices, index


def select_frames_near_average1(filter_indices, grid_size, total_frames, invalid_list):
    avg_index = round(np.mean(filter_indices))
    # 确保采样范围不超出 [0, total_frames-1]
    start_index, end_index = avg_index, avg_index
    # 生成选取的帧索引列表
    used_frame_indices = []
    if avg_index not in invalid_list:
        used_frame_indices.append(avg_index)
    # 如果选取的帧数量小于 grid_size^2，补充一些帧以满足要求
    while len(used_frame_indices) < 20:
        if len(used_frame_indices) < 20:
            # 如果当前帧不够，可以从左边或右边继续补充
            if start_index >= 0:
                if start_index > 0:
                    start_index -= 1
                    if start_index not in invalid_list:
                        used_frame_indices.insert(0, start_index)
                if start_index == 0:
                    used_frame_indices.insert(0, start_index)
            if len(used_frame_indices) < 20 and end_index <= total_frames - 1:
                if end_index < total_frames - 1:
                    end_index += 1
                    if end_index not in invalid_list:
                        used_frame_indices.append(end_index)
                if end_index == total_frames - 1:
                    used_frame_indices.append(end_index)

    # 确保最终的索引列表数量为 grid_size^2
    used_frame_indices = used_frame_indices[:20]
    index = used_frame_indices.index(filter_indices)
    return used_frame_indices, index


def select_and_filter_keyframes_with_anchor(selected_indices, total_indices, grid_size, search_anchor, video_path):
    if not selected_indices:
        return []
    video = cv2.VideoCapture(video_path)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if search_anchor == 'start':
        # 保证所有关键帧都在视频的前半部分
        filtered_indices = [idx for idx in selected_indices if idx < total_frames // 2]
        # **如果筛选后的关键帧少于 grid_size，则补充前半部分的帧**
        if len(filtered_indices) < grid_size:
            num_needed = grid_size - len(filtered_indices)
            remaining_indices = [i for i in total_indices if i not in filtered_indices and i < total_frames // 2]
            # **补充速度次优的帧**
            for idx in remaining_indices:
                if len(filtered_indices) >= grid_size:
                    break  # 已经补充到足够数量
                filtered_indices.append(idx)

    elif search_anchor == 'end':
        filtered_indices = [idx for idx in selected_indices if idx >= total_frames // 2]

        if len(filtered_indices) < grid_size:
            num_needed = grid_size - len(filtered_indices)
            remaining_indices = [i for i in total_indices if i not in filtered_indices and i >= total_frames // 2]
            # **补充速度次优的帧**
            for idx in remaining_indices:
                if len(filtered_indices) >= grid_size:
                    break  # 已经补充到足够数量
                filtered_indices.append(idx)

    else:
        raise ValueError("search_anchor must be either 'start' or 'end'")
    filtered_indices_sorted = sorted(filtered_indices)
    return filtered_indices_sorted


# def get_json_path(video_name, base_dir="/home/hand/3D_hand_speed"):
def get_json_path(video_name, base_dir="/home/EgoLoc/hand_data_drawer/3D_hand_right"):
    # json_filename = f"{video_name}_with_speed.json"
    # json_path = os.path.join(base_dir, json_filename)
    # return json_path
    """
    e.g. video_name='video29' ->
    /home/EgoLoc/hand_data_drawer/3D_hand_speed_hamer/video29/video29_with_speed.json
    """
    json_dir = os.path.join(base_dir, video_name)
    os.makedirs(json_dir, exist_ok=True)  # 确保目录存在（读/写都安全）
    json_path = os.path.join(json_dir, f"{video_name}_with_speed.json")
    return json_path


def get_json_folder_path(video_name, base_dir="/home/EgoLoc/hand_data_drawer/3D_hand_right"):
    # json_filename = f"{video_name}_with_speed.json"
    # json_path = os.path.join(base_dir, json_filename)
    # return json_path
    """
    e.g. video_name='video29' ->
    /home/EgoLoc/hand_data_drawer/3D_hand_speed_hamer/video29/video29_with_speed.json
    """
    json_dir = os.path.join(base_dir, video_name)
    os.makedirs(json_dir, exist_ok=True)  # 确保目录存在（读/写都安全）

    return json_dir


def select_top_n_frames_from_json(json_path, n, frame_index=None, flag=None, receive_flag=None):
    with open(json_path, 'r') as file:
        data = json.load(file)
    if frame_index is None:
        valid_frames = [
            (index, speed) for index, speed in data if speed != 0.0 and not math.isnan(speed)
        ]

    else:
        if flag == "feedback":
            valid_frames = [
                (index, speed) for index, speed in data if
                speed != 0.0 and not math.isnan(speed) and index > frame_index
            ]
            invalid_list = [
                index for index, speed in data if speed != 0.0 and not math.isnan(speed) and index <= frame_index
            ]
        elif flag == "speed":
            valid_frames = [
                (index, speed) for index, speed in data if
                speed != 0.0 and not math.isnan(speed) and speed < frame_index
            ]
            invalid_list = [
                index for index, speed in data if speed != 0.0 and not math.isnan(speed) and speed >= frame_index
            ]
        else:
            valid_frames = [
                (index, speed) for index, speed in data if
                speed != 0.0 and not math.isnan(speed) and index != frame_index
            ]
            invalid_list = [
                index for index, speed in data if speed != 0.0 and not math.isnan(speed) and index == frame_index
            ]
    sorted_frames = sorted(valid_frames, key=lambda x: x[1])
    top_n_frames = [frame[0] for frame in sorted_frames[:n]]
    if receive_flag is None:
        return top_n_frames
    else:
        return invalid_list, top_n_frames


def image_resize_for_vlm(frame, inter=cv2.INTER_AREA):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    max_short_side = 768
    max_long_side = 2000
    if aspect_ratio > 1:
        new_width = min(width, max_long_side)
        new_height = int(new_width / aspect_ratio)
        if new_height > max_short_side:
            new_height = max_short_side
            new_width = int(new_height * aspect_ratio)
    else:
        new_height = min(height, max_long_side)
        new_width = int(new_height * aspect_ratio)
        if new_width > max_short_side:
            new_width = max_short_side
            new_height = int(new_width / aspect_ratio)
    resized_frame = cv2.resize(
        frame, (new_width, new_height), interpolation=inter)
    return resized_frame


# Extract JSON part from the response
def extract_json_part(text_output):
    text = text_output.strip().replace(" ", "").replace("\n", "")
    try:
        start = text.index('{"points":')
        text_json = text[start:].strip()
        end = text_json.index('}') + 1
        text_json = text_json[:end].strip()
        return text_json
    except ValueError:
        print("Text received:", text_output)
        return None


def get_contact_separation_pairs(results, speed_data):
    # 构建速度字典
    speed_dict = {frame: speed for frame, speed in speed_data}

    # 分离 Contact 和 Separation 事件，并提取帧和速度
    contacts = []
    separations = []
    for event in results:
        event_type, frame = event
        if frame not in speed_dict:
            continue
        if event_type == "Contact":
            contacts.append((frame, speed_dict[frame]))
        elif event_type == "Separation":
            separations.append((frame, speed_dict[frame]))

    # 按帧索引排序
    contacts.sort()
    separations.sort()

    pairs = []
    contact_candidates = []  # 修正点：存储格式改为 (frame, speed)

    # 混合排序所有事件（保持原有逻辑）
    all_events = sorted(
        [("C", frame, speed) for frame, speed in contacts] +
        [("S", frame, speed) for frame, speed in separations],
        key=lambda x: x[1]  # 按帧索引排序
    )

    for event in all_events:
        event_type, frame, speed = event

        if event_type == "C":
            # 修正点：使用 speed 比较（索引应为 1）
            if not contact_candidates or speed < contact_candidates[-1][1]:
                contact_candidates.append((frame, speed))

        elif event_type == "S":
            if contact_candidates:
                # 选择速度最小的 Contact（索引应为 1）
                best_contact = min(contact_candidates, key=lambda x: x[1])
                pairs.append((best_contact[0], frame))
                contact_candidates = []

    return pairs


# Perform scene understanding on the frame
def scene_understanding(credentials, frame, prompt_message, principle=None):
    frame = image_resize_for_vlm(frame)
    # frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frame = base64.b64encode(buffer).decode("utf-8")
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_message
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64Frame}",
                        "detail": "high"
                    },
                }
            ]
        },
    ]

    if len(credentials["AZURE_OPENAI_API_KEY"]) == 0:
        client_gpt4v = OpenAI(
            api_key=credentials["OPENAI_API_KEY"],
            base_url="https://api.chatanywhere.tech/v1"
        )
        params = {
            "model": "gpt-4o",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 200,
            "temperature": 0.1,
            "top_p": 0.5,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
    else:
        client_gpt4v = AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=credentials["AZURE_OPENAI_ENDPOINT"],
            api_key=credentials["AZURE_OPENAI_API_KEY"]
        )
        params = {
            "model": credentials["AZURE_OPENAI_DEPLOYMENT_NAME"],
            "messages": PROMPT_MESSAGES,
            "max_tokens": 200,
            "temperature": 0.1,
            "top_p": 0.5,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
    count = 0
    while True:
        if count > 5:
            raise Exception("Failed to get response from Azure OpenAI")
        try:
            result = client_gpt4v.chat.completions.create(**params)
            break
        except openai.BadRequestError as e:
            print(e)
            print('Bad Request error.')
            return None, None
        except openai.RateLimitError as e:
            print(e)
            print('Rate Limit. Waiting for 5 seconds...')
            time.sleep(5)
            count += 1
        except openai.APIStatusError as e:
            print(e)
            print('APIStatusError. Waiting for 1 second...')
            time.sleep(1)
            count += 1
    if principle == "state":
        # print("answer:",result.choices[0].message.content)
        state = extract_event_info(result.choices[0].message.content)
        return state
    elif principle == "feedback":
        # print("feedback:",result.choices[0].message.content)
        return result.choices[0].message.content
    else:
        # print("answer:",result.choices[0].message.content)
        frame = extract_frame_info(result.choices[0].message.content)
        return frame


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# Create a grid of frames
def create_frame_grid_with_keyframe(video_path, frame_indices, grid_size, minima_frame=None):
    spacer = 0
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for index in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = video.read()
        # image = Image.fromarray(frame)
        # image.save(f"grid_{index}.png")
        if success:
            frame = image_resize(frame, width=200)
            frames.append(frame)
        else:
            print(f"Warning: Frame {index} not found")
            print(f"Total frames: {total_frames}")
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video.read()
            frame = image_resize(frame, width=200)
            frame = frame * 0
            frames.append(frame)
    video.release()

    if len(frames) < grid_size ** 2:
        print(f"Not enough frames, need {grid_size ** 2} frames. Filling the remaining with black frames.")
        # Calculate how many more frames are needed
        missing_frames = grid_size ** 2 - len(frames)
        black_frame = np.zeros_like(frames[0])  # Generate a black frame
        frames.extend([black_frame] * missing_frames)  # Add black frames to fill the grid
        # raise ValueError("Not enough frames to create the grid.")

    frame_height, frame_width = frames[0].shape[:2]

    grid_height = grid_size * frame_height + (grid_size - 1) * spacer
    grid_width = grid_size * frame_width + (grid_size - 1) * spacer

    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            frame = frames[index]
            cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
            max_dim = int(min(frame.shape[:2]) * 0.5)
            overlay = frame.copy()
            # is_minima = (index == minima_frame)
            is_minima = 0
            circle_color = (255, 0, 0) if is_minima else (255, 255, 255)  # 红色 or 白色
            box_color = (255, 0, 0) if is_minima else None  # 红色方框 or 不画
            if render_pos == 'center':
                circle_center = (cX, cY)
            else:
                circle_center = (frame.shape[1] - max_dim // 2, max_dim // 2)
            cv2.circle(overlay, circle_center,
                       max_dim // 2, circle_color, -1)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.circle(frame, circle_center, max_dim // 2, circle_color, 2)
            font_scale = max_dim / 50
            text_size = cv2.getTextSize(
                str(index + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            if render_pos == 'center':
                text_x = cX - text_size[0] // 2
                text_y = cY + text_size[1] // 2
            else:
                text_x = frame.shape[1] - text_size[0] // 2 - max_dim // 2
                text_y = text_size[1] // 2 + max_dim // 2

            cv2.putText(frame, str(index + 1), (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            if is_minima:
                thickness = 4
                cv2.rectangle(frame, (0, 0), (frame_width - 1, frame_height - 1), box_color, thickness)
            y1 = i * (frame_height + spacer)
            y2 = y1 + frame_height
            x1 = j * (frame_width + spacer)
            x2 = x1 + frame_width
            grid_img[y1:y2, x1:x2] = frame

    return grid_img


def image_resize_state(image, width=None):
    height, w, _ = image.shape
    if width is None:
        return image
    ratio = width / float(w)
    dim = (width, int(height * ratio))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# def create_frame_grid_state(video_path, frame_indices):
#     assert len(frame_indices) == 2, "frame_indices 必须包含两个元素"

#     video = cv2.VideoCapture(video_path)
#     frames = []

#     for index in frame_indices:
#         video.set(cv2.CAP_PROP_POS_FRAMES, index)
#         success, frame = video.read()
#         if success:
#             frame = image_resize(frame, width=200)
#         else:
#             frame = np.zeros((112, 200, 3), dtype=np.uint8)  # 生成黑色填充帧
#         frames.append(frame)

#     video.release()
#     # 确保两帧可用
#     if len(frames) < 2:
#         missing_frames = 2 - len(frames)
#         black_frame = np.zeros_like(frames[0])
#         frames.extend([black_frame] * missing_frames)

#     frame_height, frame_width = frames[0].shape[:2]
#     grid_img = np.ones((frame_height, frame_width * 2, 3), dtype=np.uint8) * 255

#     # 左侧图像
#     grid_img[:, :frame_width] = frames[0]
#     # 右侧图像
#     grid_img[:, frame_width:] = frames[1]

#     return grid_img


def create_frame_grid_state(video_path, frame_indices, grid_size=(1, 2)):
    spacer = 0
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            # 调整尺寸（保持宽度 400）
            frame = image_resize(frame, width=400)
            # 手部检测并裁剪，再缩放回当前尺寸
            frame = run_groundingdino_and_crop(frame)
            frames.append(frame)
        else:
            print(f"Warning: Frame {idx} not found (total {total_frames}). Using black frame.")
            # 使用与前面帧相同尺寸的黑帧
            black = np.zeros_like(frames[0] if frames else np.zeros((400, 400, 3), dtype=np.uint8))
            frames.append(black)
    video.release()

    # 如果帧不足，填充黑帧
    total_needed = grid_size[0] * grid_size[1]
    if len(frames) < total_needed:
        missing = total_needed - len(frames)
        black = np.zeros_like(frames[0])
        frames.extend([black] * missing)

    fh, fw = frames[0].shape[:2]
    gh = grid_size[0] * fh + (grid_size[0] - 1) * spacer
    gw = grid_size[1] * fw + (grid_size[1] - 1) * spacer
    grid_img = np.ones((gh, gw, 3), dtype=np.uint8) * 255

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            idx = i * grid_size[1] + j
            frame = frames[idx]
            # 渲染圆与数字
            # cX, cY = fw // 2, fh // 2
            # max_dim = int(min(fh, fw) * 0.5)
            # overlay = frame.copy()
            # if render_pos == 'center':
            #     center = (cX, cY)
            # else:
            #     center = (fw - max_dim//2, max_dim//2)
            # cv2.circle(overlay, center, max_dim//2, (255,255,255), -1)
            # frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            # cv2.circle(frame, center, max_dim//2, (255,255,255), 2)
            # # 文本
            # text = str(idx+1)
            # ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, max_dim/50, 2)[0]
            # if render_pos == 'center':
            #     tx = cX - ts[0]//2
            #     ty = cY + ts[1]//2
            # else:
            #     tx = fw - ts[0]//2 - max_dim//2
            #     ty = ts[1]//2 + max_dim//2
            # cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, max_dim/50, (0,0,0), 2)

            y1 = i * (fh + spacer)
            y2 = y1 + fh
            x1 = j * (fw + spacer)
            x2 = x1 + fw
            grid_img[y1:y2, x1:x2] = frame

    return grid_img


def add_text_with_background(
        frame,
        text,
        position,
        font,
        font_scale,
        font_color,
        font_thickness,
        bg_color):
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x, text_y = position
    top_left = (text_x - 10, text_y - text_size[1] - 10)
    bottom_right = (text_x + text_size[0] + 10, text_y + 10)
    cv2.rectangle(frame, top_left, bottom_right, bg_color, -1)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                font_color, font_thickness, cv2.LINE_AA)


# Annotate the video with task times
def trim_video_with_annotations(
        video_path,
        start_time,
        end_time,
        text,
        output_path,
        buffer=0.5):
    """Trim and annotate video with specified start and end times and text."""
    if os.path.exists(output_path):
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - int(buffer * fps)))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cap.get(
                cv2.CAP_PROP_POS_FRAMES) > end_frame + int(buffer * fps):
            break
        if start_frame <= cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
            add_text_with_background(
                frame,
                text,
                (10,
                 height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,
                 0,
                 255),
                2,
                (255,
                 255,
                 255))
        out.write(frame)

    cap.release()
    out.release()

# ===== BEGIN PATCH: helpers =====
#11.4新增的保护机制，防止
from typing import Optional

def _nearest_valid_frame(i: int, all_frames_np: np.ndarray, total_frames: int) -> Optional[int]:
    """把任意 i 夹到 [0, total_frames-1]，若不在 all_frames 中，就近映射到最近存在的帧号。"""
    if total_frames <= 0:
        return None
    i = int(max(0, min(total_frames - 1, int(i))))
    if all_frames_np.size == 0:
        return None
    if np.any(all_frames_np == i):
        return i
    j = int(np.argmin(np.abs(all_frames_np - i)))
    return int(all_frames_np[j])

def _safe_choice(cands: List[int], probs: Optional[np.ndarray] = None) -> int:
    """对空/退化概率做兜底：若只有一个候选就返回它；若多于一个但概率非法则均匀选。"""
    if not cands:
        raise ValueError("_safe_choice received empty candidate list")
    if len(cands) == 1 or probs is None:
        return int(cands[0]) if len(cands) == 1 else int(np.random.choice(cands))
    probs = np.asarray(probs, dtype=float)
    if (probs.shape[0] != len(cands)) or (not np.all(np.isfinite(probs))) or (probs.sum() <= 0):
        return int(np.random.choice(cands))
    probs = probs / probs.sum()
    if not np.isfinite(probs).all():
        return int(np.random.choice(cands))
    return int(np.random.choice(cands, p=probs))

# def _build_mapped_window(center: int, window: int, all_frames_np: np.ndarray, total_frames: int) -> List[int]:
#     """围绕 center±window 生成候选，全部映射到 all_frames 的最近存在帧，并去重排序。"""
#     raw = list(range(center - window, center + window + 1))
#     raw = [r for r in raw if 0 <= r < total_frames]
#     mapped = []
#     for r in raw:
#         nf = _nearest_valid_frame(r, all_frames_np, total_frames)
#         if nf is not None:
#             mapped.append(nf)
#     # 去重并排序
#     return sorted(set(mapped))
def _build_mapped_window(center: int, window: int, all_frames_np: np.ndarray, total_frames: int) -> List[int]:
    raw = list(range(center - window, center + window + 1))
    raw = [r for r in raw if 0 <= r < total_frames]
    mapped = []
    for r in raw:
        nf = _nearest_valid_frame(r, all_frames_np, total_frames)
        if nf is not None:
            mapped.append(nf)
    # 顺序去重：保留靠近 center 的相对顺序
    uniq = list(dict.fromkeys(mapped))
    return uniq

# ===== END PATCH: helpers =====


def process_task(
        credentials,
        video_path,
        grid_size,
        total_frames,
        frame_index=None,
        flag=None,
        max_feedback=5,
        video_type="short",
        keyframe_sampling_mode="adaptive",
        use_feedback=False
):
    """Process a task to identify the start or end of an action in a video."""
    prompt_contact = (
        """
        Instruction:
        You will be given a time-ordered image grid containing a hand and a target object. Your task is to find the earliest contact moment, where the hand first contacts with the object, and return its index.

        Reasoning Steps:
        1. Analyze each frame to observe the relationship between the hand and the object.
        2. Identify the earliest transition from separation (hand not touching the object) → contact (hand touching the object).
        3. If no contact is detected within the grid, return -1.

        Strict Output Format:
        If a contact moment is detected:
        "Frame: X"  
        (Where X is the index of the earliest contact frame)
        If no contact is detected:
        "Frame: -1"
        """
    )

    prompt_separation = (
        """
        Instruction:
        You will be given a time-ordered image grid containing a hand and a target object. Your task is to find the earliest separation moment, where the hand first separates with the object, and return its index.

        Reasoning Steps:
        1. Analyze each frame to observe the relationship between the hand and the object.
        2. Identify the earliest transition from contact (hand touching the object) → separation (hand moving away from the object).
        3. If no separation is detected within the grid, return -1.

        Strict Output Format:
        If a separation moment is detected:
        "Frame: X"  
        (Where X is the index of the earliest separation frame)
        If no separation is detected:
        "Frame: -1"
        """
    )
    # prompt_state = (
    #     """
    #     Instruction:
    #     - You will be given two frames: a previous frame (Frame Left) and a next frame (Frame Right). Your goal is to detect whether a Contact Moment, a Separation Moment, or Neither occurs based on the change in hand-object interaction between the two frames.

    #     Definitions:
    #     - Contact: The hand is touching or making physical contact with the target object.
    #     - Separation: The hand is not touching the target object.

    #     Steps:
    #     - Confirm that you are looking at the same target object in both frames.
    #     - In Frame Left:
    #         - If the hand is touching the object, classify as Contact.
    #         - Otherwise, classify as Separation.
    #     - In Frame Right:
    #         - If the hand is touching the object, classify as Contact.
    #         - Otherwise, classify as Separation.
    #     - Determine the event based on the transition:
    #         - If Frame Left = Separation → Frame Right = Contact, output: Event: Contact
    #         - If Frame Left = Contact → Frame Right = Separation, output: Event: Separation
    #         - For all other combinations (Contact → Contact, Separation → Separation), output: Event: Neither

    #     Output Format (Strict):
    #     - After completing your reasoning, output exactly one of the following as the final line:
    #         - "Event: Contact"
    #         - "Event: Separation"
    #         - "Event: Neither"
    #     """
    # )

    prompt_state = (
        """
        Instruction:
        - You are given two image frames: a previous frame (Frame Left) and a future frame (Frame Right).
        - Your task is to detect a **Contact Moment** or **Separation Moment** by analyzing the change in interaction between the hand and a target object of the two frames.
        - Avoid outputting "Event: Neither" unless both frames clearly and confidently show the **same state with no visible change**.

        Definitions:
        - Contact: The hand is visibly touching or making physical contact with the object (e.g., fingers pressed against the surface, grasping).
        - Separation: The hand is clearly not in contact with the object (e.g., fingers hovering, obvious gap, no overlap).

        Steps:
        1. Ensure the same target object is being observed in both frames.
        2. For each frame:
        - Check whether the hand is **clearly touching** or **clearly not touching** the object.
        - If it’s ambiguous, lean toward detecting **Contact** if the hand is near or aligned for grasp; lean toward **Separation** if the hand is retreating or open.
        3. Decide the event based on the transition:
        - If Frame Left = Separation and Frame Right = Contact → output: **Event: Contact**
        - If Frame Left = Contact and Frame Right = Separation → output: **Event: Separation**
        - If both frames show **no significant change**, and you are confident the contact state stayed the same → output: **Event: Neither**
        - In cases of uncertainty or partial motion, prefer to output **Contact** or **Separation** over “Neither”.

        Output Format (Strict):
        Output exactly one of the following as the final line:
        - "Event: Contact"
        - "Event: Separation"
        - "Event: Neither"

        """
    )

    # prompt_state = (
    #     """
    #     Instruction:
    #     - You are given two image frames: a previous frame (Frame Left) and a next frame (Frame Right).
    #     - Your task is to determine whether a **Contact Moment** or a **Separation Moment** occurs, based on the change in interaction between the hand and a target object.

    #     Definitions:
    #     - Contact: The hand is visibly touching the object (e.g., fingers in contact, object gripped or pressed).
    #     - Separation: The hand is not in contact with the object (e.g., fingers open or pulled back, a visible gap between hand and object).

    #     Steps:
    #     1. Ensure the same object is present in both frames.
    #     2. For each frame, decide whether the hand is in a Contact or Separation state:
    #     - If unclear or ambiguous, **do not assume “no change”**.
    #     - Instead, infer the most likely state by considering hand posture, motion direction, and proximity to the object:
    #         - If the hand is moving toward or very close to the object with grasping posture → classify as Contact.
    #         - If the hand is withdrawing or open with distance from the object → classify as Separation.
    #     3. Use the following logic to determine the event:
    #     - If Frame Left = Separation and Frame Right = Contact → output: **Event: Contact**
    #     - If Frame Left = Contact and Frame Right = Separation → output: **Event: Separation**
    #     - If both frames appear to have the same state, but there is any uncertainty or motion → **choose the most likely transition** (Contact or Separation).

    #     Output Format (Strict):
    #     Output exactly one of the following as the final line:
    #     - "Event: Contact"
    #     - "Event: Separation"
    #     """
    # )

    # Iterate to narrow down the time
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # 建一个按视频名分组的调试图目录,方便保存图片
    debug_dir = Path("/home/EgoLoc/hand_data_drawer/debug_grids") / video_name
    debug_dir.mkdir(parents=True, exist_ok=True)

    json_path = get_json_path(video_name)
    json_folder_path = get_json_folder_path(video_name)
    # 读取全部帧和速度
    with open(json_path, 'r') as f:
        all_data = json.load(f)
    all_frames = np.array([x[0] for x in all_data])
    all_speeds = np.array([x[1] for x in all_data])
    # 根据video_type选择极小值点提取函数
    if video_type == "short":
        minima_indices = extract_local_minima_frames(video_name, json_folder_path)
        minima_speeds = [all_speeds[np.where(all_frames == idx)[0][0]] for idx in minima_indices]
    else:
        minima_indices, minima_speeds = extract_local_minima_frames_adaptive(video_name, json_folder_path)
    print(f"{video_name}获取的极小值列表为:{minima_indices}")
    selected_frame_index = []
    while minima_indices:
        # 速度越小概率越高采样极小值点
        selected_minima = adaptive_sample_speed(minima_indices, minima_speeds)
        idx = minima_indices.index(selected_minima)
        minima_indices.pop(idx)
        minima_speeds.pop(idx)
        # 采样关键帧索引
        window = 2
        minima_idx = selected_minima
        #candidate_indices = [i for i in range(minima_idx - window, minima_idx + window + 1) if 0 <= i < total_frames]
        # 候选集合（就近映射到实际存在的 all_frames）
        candidate_indices = _build_mapped_window(minima_idx, window, all_frames, total_frames)

        # 逐级回退：若为空，用 minima 自己的最近有效帧；再不行用全局中点；还不行就跳过这个极小值
        if not candidate_indices:
            nf = _nearest_valid_frame(minima_idx, all_frames, total_frames)
            if nf is not None:
                candidate_indices = [nf]
            else:
                mid = total_frames // 2
                nf = _nearest_valid_frame(mid, all_frames, total_frames)
                if nf is not None:
                    candidate_indices = [nf]
                else:
                    print(f"[process_task] empty candidate near minima {minima_idx}, skip this minima.")
                    continue

        print(candidate_indices)
        candidate_speeds = [all_speeds[np.where(all_frames == i)[0][0]] if np.any(all_frames == i) else 9999 for i in
                            candidate_indices]
        if keyframe_sampling_mode == "adaptive":
            inv_speeds = 1 / (np.array(candidate_speeds) + 1e-8)
            probabilities = inv_speeds / inv_speeds.sum()
            keyframe_index = np.random.choice(candidate_indices, p=probabilities)
        else:
            keyframe_index = np.random.choice(candidate_indices)
        state_frame_indices, minima_index = select_frames_near_average(keyframe_index, 3, total_frames, [])
        state_indices = [state_frame_indices[0], state_frame_indices[-1]]
        print(f"选取的判断帧为{state_frame_indices[0]}和{state_frame_indices[-1]}")
        image_state = create_frame_grid_state(
            video_path, state_indices)
        image_RGB = cv2.cvtColor(image_state, cv2.COLOR_BGR2RGB)
        grid_image = Image.fromarray(image_RGB)
        # grid_image.save(f"/home/grid_EgoLoc.png")
        grid_image.save(debug_dir / f"{video_name}_state.png")
        state = scene_understanding(
            credentials, image_state, prompt_state, principle="state")
        print("判断其状态为：", state)
        if state == "Contact" or state == "Separation":
            frame_indices, minima_index = select_frames_near_average(keyframe_index, grid_size, total_frames, [])
            print(f'{video_name}选择的关键帧为{frame_indices}')
            image = create_frame_grid_with_keyframe(
                video_path, frame_indices, grid_size, minima_index)
            image_RGB1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            grid_image1 = Image.fromarray(image_RGB1)
            # grid_image1.save(f"/home/grid_EgoLoc1.png")
            grid_image1.save(debug_dir / f"{video_name}_keyframes.png")
            prompt = prompt_contact if state == "Contact" else prompt_separation  # 统一为大写的contact
            description = scene_understanding(
                credentials, image, prompt)
            if description:
                if description != -1:
                    if int(description) - 1 > len(frame_indices) - 1:
                        print("Warning: Invalid frame index selected")
                        print(f"Selected frame index: {description}")
                    index_specified = max(
                        min(int(description) - 1, len(frame_indices) - 1), 0)
                    final_frame = frame_indices[index_specified]
                    # === 反馈机制：对final_frame再用VLM判断状态 ===
                    if use_feedback:
                        feedback_window = 3
                        tried_frames = set()
                        correct = False
                        feedback_count = 0
                        while feedback_count < max_feedback:
                            tried_frames.add(final_frame)
                            single_image = create_frame_grid_with_keyframe(video_path, [final_frame], 1)
                            if state == "Contact":
                                feedback_prompt = (
                                    "I will show an image of hand-object interaction. "
                                    "You need to help me determine whether the hand and the object in the current image are in contact rather than just appearing to be in contact. "
                                    "If yes, answer 1. If not, answer 0.")
                            else:
                                feedback_prompt = (
                                    "I will show an image of hand-object interaction. "
                                    "You need to help me determine whether the hand and the object in the current image are in seperate rather than just appearing to be in seperate. "
                                    "If yes, answer 1. If not, answer 0.")
                            feedback_result = scene_understanding(credentials, single_image, feedback_prompt, flag)
                            print(f"反馈VLM输出: {feedback_result}")
                            if is_positive_feedback(feedback_result):
                                correct = True
                                break
                            # 采样新帧
                            feedback_count += 1
                            feedback_candidates = [i for i in range(final_frame - feedback_window,
                                                                    final_frame + feedback_window + 1)
                                                   if 0 <= i < total_frames and i not in tried_frames]
                            if not feedback_candidates:
                                break
                            feedback_speeds = [
                                all_speeds[np.where(all_frames == i)[0][0]] if np.any(all_frames == i) else 9999 for i
                                in feedback_candidates]
                            if keyframe_sampling_mode == "adaptive":
                                inv_speeds = 1 / (np.array(feedback_speeds) + 1e-8)
                                probabilities = inv_speeds / inv_speeds.sum()
                                final_frame = np.random.choice(feedback_candidates, p=probabilities)
                            else:
                                final_frame = np.random.choice(feedback_candidates)
                        if not correct:
                            print("反馈后未找到合适帧或超出最大反馈次数，跳过该极小值点")
                            continue
                    # use_feedback为False时，直接采纳final_frame，无需反馈
                    selected_frame_index.append((state, final_frame))
    return selected_frame_index


def calculate_max_mode_average(list_of_pairs_lists):
    """
    处理多组 Contact-Separation 对数据：
    1. 找出帧对数量最多的模式（最大对数）
    2. 对具有该模式的所有实验结果按位置对齐计算平均值
    3. 返回四舍五入后的帧对列表
    """
    if not list_of_pairs_lists:
        return []

    # 统计每组的对数
    pair_counts = [len(pairs) for pairs in list_of_pairs_lists]
    if not pair_counts:
        return []

    max_count = max(pair_counts)
    if max_count == 0:
        return []

    # 筛选出具有最大对数的实验组
    max_mode_groups = [pairs for pairs in list_of_pairs_lists if len(pairs) == max_count]

    # 按位置对齐各组帧对并求平均
    averaged_pairs = []
    for grouped_pairs in zip(*max_mode_groups):
        contacts = [contact for contact, _ in grouped_pairs]
        separations = [separation for _, separation in grouped_pairs]

        avg_contact = round(sum(contacts) / len(contacts))
        avg_separation = round(sum(separations) / len(separations))

        averaged_pairs.append((avg_contact, avg_separation))

    return averaged_pairs


def convert_video(video_file_path: str, action: str, credentials, grid_size: int, video_type="short", max_feedback=1):
    video = cv2.VideoCapture(video_file_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    results = process_task(
        credentials,
        video_file_path,
        grid_size,
        total_frames,
        max_feedback=max_feedback,
        video_type=video_type)
    # print(results)
    video_name = os.path.splitext(os.path.basename(video_file_path))[0]
    json_path = get_json_path(video_name)
    with open(json_path, 'r') as f:
        speed_data = json.load(f)
    pair = get_contact_separation_pairs(results, speed_data)
    return pair


parser = argparse.ArgumentParser()
parser.add_argument("--video_type", default="short")
parser.add_argument("--credentials", help="credentials file")
parser.add_argument("--grid", help="grid size", default=3)
parser.add_argument(
    "--action",
    help="action label",
    default="grabbing towards the can")
parser.add_argument('--keyframe_sampling_mode', type=str, default='adaptive', choices=['adaptive', 'random'],
                    help='关键帧索引采样方式: adaptive(速度加权) or random(等概率)')
pargs, unknown = parser.parse_known_args()
credentials = dotenv.dotenv_values(pargs.credentials)
required_keys = ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
if not all(key in credentials for key in required_keys):
    raise ValueError("Required keys are missing in the credentials file")
render_pos = 'topright'  # center or topright
grid_size = int(pargs.grid)
video_folder = "/home/EgoLoc/hand_data_drawer/EgoPAT3D/EgoPAT3D"
action = pargs.action
video_type = pargs.video_type
folder_name = action.replace(" ", "_")
output_folder = f"results/{folder_name}"
# os.makedirs(output_folder, exist_ok=True)
if __name__ == "__main__":
    # 获取文件夹中的所有 MP4 文件并按顺序排序
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    save_path = "/home/EgoLoc/ManiTIL_prompt/right_grid4.json"
    speed_output_root = "/home/EgoLoc/results2"
    # save_path = "/home/VLM-Video-Action-Localization-main/VLM-Video-Action-Localization-main/result/greedyVLM_drawer_grid5.json"
    # 排序视频文件，基于文件名中 'c' 后的数字部分
    sorted_video_files = sorted(video_files, key=lambda x: int(x.split('.')[0][5:]))
    all_predictions = load_predictions(save_path)
    batch_process_videos(video_folder, speed_output_root, device="cuda", encoder="vits")  # 后续添加对已有文件的跳过
    # processed_video_files = {prediction[0] for prediction in all_predictions}
    #
    # # 按顺序遍历视频文件
    # for video_file in sorted_video_files:
    #     if video_file in processed_video_files:
    #         continue
    #     video_path = os.path.join(video_folder, video_file)
    #     # video_path = "/home/bathroomCabinet/video_cleaned/video32.mp4"
    #     if os.path.exists(video_path):
    #         list_of_pair = []
    #         for i in range(1):
    #             pair = convert_video(
    #                 video_path, action, credentials, grid_size, video_type=video_type, max_feedback=1)
    #             list_of_pair.append(pair)
    #
    #         averaged_pairs = calculate_max_mode_average(list_of_pair)
    #
    #         if len(averaged_pairs) == 0:
    #             print(f"{video_file} can't predict")
    #             all_predictions.append([video_file, [(0, 0)]])
    #         else:
    #             print(f"{video_file} pairs: {averaged_pairs}")
    #             all_predictions.append([video_file, averaged_pairs])
    #
    #         save_predictions(all_predictions, save_path)
    #
    # if video_type == "short":
    #     result = evaluate_predictions(
    #         json_path=save_path,
    #         gt_excel_path="/home/EgoLoc/ground_truth/KitchenCounter1.xlsx",
    #         sheet_name="Sheet9"  # 你也可以换其他sheet
    #     )
    #     print(result)
    #
    # elif video_type == "long":
    #     results = evaluate_all(
    #         pred_json=save_path,
    #         gt_xlsx="/home/EgoLoc/ground_truth/KitchenCounter1.xlsx",
    #         sheet_name="hand_data_cabinet",
    #         sr_tolerances=(1, 3, 5),
    #         psr_tolerance=10
    #     )
    #     print("Evaluation Results:")
    #     for k, v in results.items():
    #         print(f"{k}: {v:.4f}")




