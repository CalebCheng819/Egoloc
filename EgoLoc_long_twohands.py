# export HF_ENDPOINT=https://hf-mirror.com
# python ./EgoLoc_long_twohands.py --credentials auth.env  --grid_size 4  --video_type long

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
import subprocess
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import UnivariateSpline
import re
from pathlib import Path
from datetime import datetime

sys.path.append('/home/Egoloc/Egolocx')  # 将 /home 路径添加到模块搜索路径中
sys.path.append('/home/Egoloc')
sys.path.append('/home/EgoLoc/Grounded-Segment-Anything/GroundingDINO')  # 必需
from groundingdino.util.inference import load_model, load_image, predict
from EgoLocx.script.long_metric import (
    evaluate_all,
    evaluate_all_stages,
    evaluate_all_stages_pooled_twohands,
    load_ground_truth_json_twohands,
)
from EgoLocx.script.compute_metric import evaluate_predictions
import tempfile
from egoloc_speed_twohands import extract_3d_speed_and_visualize  # 新封装的生成速度文件的函数
from egoloc_speed_twohands import batch_process_videos  # 对文件夹内的所有视频执行extract_3d_speed_and_visualize
from typing import List, Optional   # ← 新增这一行

# 模型惰性加载，便于多 GPU 时子进程在各自 GPU 上加载
_model = None
# 主进程的 minima_cache 引用，供 process_task 在未传入 minima_cache 时使用
_MINIMA_CACHE_REF = None


def get_model():
    """首次调用时加载 GroundingDINO，之后返回已加载的模型（便于多 GPU 子进程各自加载）。"""
    global _model
    if _model is None:
        _model = load_model(
            "/home/EgoLoc/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "/home/EgoLoc/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"  # 直接放在weights目录外
        )
    return _model

# 速度 JSON 根目录；由 --speed_root 设置时覆盖 get_json_path / get_json_folder_path 及 extract_* 的 folder_path
_SPEED_BASE_DIR = None
# 输出根目录；由 --output_dir 指定或自动生成（日期+配置），所有预测/缓存/调试写入其下
_OUTPUT_ROOT = None


def _load_speed_scalar(json_path: str, hand: str = "right"):
    """
    把双手速度文件解析成 (frame, speed_scalar) 列表。
    支持两种格式：
      1) 旧单手: [frame, speed]
      2) 新双手: [frame, vL, vR]

    hand:
      - "left"  : 用 vL
      - "right" : 用 vR
      - "min"   : min(vL, vR)
      - "max"   : max(vL, vR)
      - "avg"   : 0.5*(vL+vR)
    """
    if not os.path.isfile(json_path):
        print(f"❗ 文件未找到: {json_path}")
        return []

    with open(json_path, "r") as f:
        raw = json.load(f)

    data = []
    for row in raw:
        if not isinstance(row, (list, tuple)):
            continue

        if len(row) == 2:
            # 旧格式: [frame, speed]
            frame, speed = row
        elif len(row) >= 3:
            # 新格式: [frame, vL, vR, ...]
            frame, vL, vR = row[:3]
            if hand == "left":
                speed = vL
            elif hand == "right":
                speed = vR
            elif hand == "min":
                speed = min(vL, vR)
            elif hand == "max":
                speed = max(vL, vR)
            elif hand == "avg":
                speed = 0.5 * (vL + vR)
            else:
                speed = vR  # 默认右手
        else:
            continue

        try:
            frame = int(frame)
            speed = float(speed)
        except Exception:
            continue

        data.append((frame, speed))

    return data
import numpy as np
import cv2
import tempfile

def _xyxy_from_cxcywh_norm(box, W, H):
    cx, cy, bw, bh = box
    x0 = int((cx - bw/2) * W)
    y0 = int((cy - bh/2) * H)
    x1 = int((cx + bw/2) * W)
    y1 = int((cy + bh/2) * H)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]

def _box_center_xy(box):
    x0,y0,x1,y1 = box
    return ((x0+x1)/2.0, (y0+y1)/2.0)

def _iou(a, b):
    ax0,ay0,ax1,ay1 = a
    bx0,by0,bx1,by1 = b
    ix0,iy0 = max(ax0,bx0), max(ay0,by0)
    ix1,iy1 = min(ax1,bx1), min(ay1,by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw*ih
    area_a = (ax1-ax0)*(ay1-ay0)
    area_b = (bx1-bx0)*(by1-by0)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def _expand_xyxy(box, W, H, expand_pixels=10, expand_ratio=None):
    x0,y0,x1,y1 = box
    if expand_ratio is not None:
        w = x1 - x0
        h = y1 - y0
        ex = int(w * expand_ratio)
        ey = int(h * expand_ratio)
    else:
        ex = ey = int(expand_pixels)
    x0 = max(0, x0 - ex); y0 = max(0, y0 - ey)
    x1 = min(W, x1 + ex); y1 = min(H, y1 + ey)
    return [x0,y0,x1,y1]

def run_groundingdino_and_crop_stable(
    image: np.ndarray,
    hand: str = "right",               # "left" / "right" / "both"
    box_thresh: float = 0.35,
    text_thresh: float = 0.25,
    expand_pixels: int = 10,
    expand_ratio: float = 0.2,         # 推荐用比例扩张更稳
    mirror: bool = False,              # 镜像视频把左右对调
    prev_box: Optional[List[int]] = None,  # ✅ 改这里     # 上一帧该手的 box，用于稳定
):
    H, W = image.shape[:2]
    if hand == "left":
        text_prompt="left hand"
    if hand == "right":
        text_prompt="right hand"
    # 1) 写临时文件给 load_image（沿用你现有方式）
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        pil_img, tensor_img = load_image(tmp.name)

    # 2) 一次性检测所有 hand（不要用 left/right prompt）
    boxes_norm, logits, phrases = predict(
        model=get_model(),
        image=tensor_img,
        caption=text_prompt,
        box_threshold=box_thresh,
        text_threshold=text_thresh
    )
    if len(boxes_norm) == 0:
        return image, None  # 返回原图 + None

    # 3) 转成像素框，并带上分数（logit）
    candidates = []
    for k, b in enumerate(boxes_norm):
        box = _xyxy_from_cxcywh_norm(b, W, H)
        if box is None:
            continue
        score = float(logits[k]) if logits is not None and len(logits) > k else 0.0
        candidates.append((box, score))

    if not candidates:
        return image, None

    # 4) 如果给了 prev_box：优先选 IoU 最大/距离最近的，避免左右互换
    if prev_box is not None and len(candidates) >= 1:
        candidates.sort(key=lambda bs: (_iou(bs[0], prev_box), bs[1]), reverse=True)
        chosen = candidates[0][0]
    else:
        # 5) 否则：按 x 中心排序分左右（画面坐标）
        candidates.sort(key=lambda bs: _box_center_xy(bs[0])[0])  # 从左到右
        left_box = candidates[0][0]
        right_box = candidates[-1][0]

        if mirror:
            # 镜像视频：左右对调
            left_box, right_box = right_box, left_box

        if hand == "left":
            chosen = left_box
        elif hand == "right":
            chosen = right_box
        elif hand == "both":
            # 两只手一起：取 union 框，把两只手都框住（给 VLM 更稳）
            x0 = min(left_box[0], right_box[0])
            y0 = min(left_box[1], right_box[1])
            x1 = max(left_box[2], right_box[2])
            y1 = max(left_box[3], right_box[3])
            chosen = [x0,y0,x1,y1]
        else:
            chosen = right_box

    # 6) 扩张框 + 裁剪 + resize 回原分辨率
    chosen = _expand_xyxy(chosen, W, H, expand_pixels=expand_pixels, expand_ratio=expand_ratio)
    x0,y0,x1,y1 = chosen
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return image, None
    resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
    return resized, chosen


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
    if _SPEED_BASE_DIR is not None:
        folder_path = os.path.join(_SPEED_BASE_DIR, video_path)
    thresh = 0.08  # 异常值阈值
    savgol_polyorder = 2  # 提高多项式阶数以更好拟合曲线
    spline_s = 1e-4  # 增大样条平滑因子抑制噪声
    min_peak_prominence = 0.2  # 极小值的最小突出度（需根据数据调整）
    min_peak_distance = 5  # 极小值间最小间隔（单位：帧）
    savgol_window_ratio = 0.2

    filename = f"{video_path}_with_speed_twohands.json"
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
    if _SPEED_BASE_DIR is not None:
        folder_path = os.path.join(_SPEED_BASE_DIR, video_path)
    # 返回速度从慢到快的极小值点列表
    thresh = 1  # 异常值阈值
    savgol_polyorder = 2  # Savitzky-Golay 多项式阶数
    spline_s = 1e-4  # 样条插值平滑因子
    savgol_mode = 'nearest'  # 滤波模式

    #json_filename = f"{video_path}_with_speed.json"
    json_filename = f"{video_path}_with_speed_twohands.json"
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



import os
import json
import numpy as np
import cv2
from typing import List, Tuple
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def extract_local_minima_frames_adaptive(
        video_path,
        folder_path="/home/EgoLoc/hand_data_drawer/ego4d_mp4_outputs",
        output_folder="./vis_speed_curve",
        hand="right",
) -> Tuple[List[int], List[float]]:
    """
    自适应版：使用
      1) 局部窗口极小值 + robust 阈值（中位数 + MAD）
      2) 先在强平滑曲线上找候选，再在原始曲线上局部精修
      3) 所有时间尺度参数由 fps 决定（而不是 N）

    参数：
        video_path: 可以是视频文件路径，或仅仅是 video_name（不含扩展名）。
                    - 用于推测 JSON 文件名时只取 basename 的 stem。
                    - 如果是实际视频文件路径，会尝试用 OpenCV 读取 fps。
        folder_path: 存放 `<video_stem>_with_speed_twohands.json` 的目录
        output_folder: 可视化输出目录（会保存速度曲线 + 极小值点）
        hand: "left" or "right" —— 传给 _load_speed_scalar

    返回：
        result: 极小值帧号列表（以你 JSON 里的 frame index 为准，一般是 0-based）
        result_speeds: 对应帧的速度
    """
    if _SPEED_BASE_DIR is not None:
        folder_path = os.path.join(_SPEED_BASE_DIR, os.path.splitext(os.path.basename(video_path))[0])

    # -------- 0. 解析 video_stem + JSON 路径 --------
    video_basename = os.path.basename(video_path)
    video_stem, _ext = os.path.splitext(video_basename)   # "xxx.mp4" -> "xxx"
    json_file = os.path.join(folder_path, f"{video_stem}_with_speed_twohands.json")

    if not os.path.isfile(json_file):
        print(f"❗ 文件未找到: {json_file}")
        return [], []

    # 这里沿用你之前的工具函数，hand="left" 或 "right" 决定取哪一列速度
    data = _load_speed_scalar(json_file, hand=hand)
    if not isinstance(data, list) or len(data) == 0:
        print(f"❗ 数据为空或格式错误")
        return [], []

    # data: [[frame, speed], ...]
    frames_all, speeds_all = zip(*data)
    frames_all = np.array(frames_all, dtype=float)
    speeds_all = np.array(speeds_all, dtype=float)


    # -------- 数据清洗：去掉无效值 + 去掉 speed>30 的异常点 --------
    # 你也可以把 30 抽成参数 max_speed=30.0
    max_speed = 30.0

    mask = np.isfinite(speeds_all) & (speeds_all > 0) & (speeds_all <= max_speed)
    frames = frames_all[mask].astype(int)
    speeds = speeds_all[mask].astype(float)

    # 保险：按 frame 排序（JSON 不一定严格有序）
    order = np.argsort(frames)
    frames = frames[order]
    speeds = speeds[order]

    # 保险：如果同一帧出现多条记录（少见，但可能），取最小速度或均值都行
    # 这里取最小速度，更偏向捕捉“停顿”
    uniq_frames = []
    uniq_speeds = []
    i = 0
    while i < len(frames):
        f = frames[i]
        j = i
        vals = []
        while j < len(frames) and frames[j] == f:
            vals.append(speeds[j])
            j += 1
        uniq_frames.append(f)
        uniq_speeds.append(np.min(vals))
        i = j
    frames = np.array(uniq_frames, dtype=int)
    speeds = np.array(uniq_speeds, dtype=float)

    N_raw = len(speeds)
    if N_raw < 4:
        print(f"⚠️ 有效数据太少（过滤speed>30后）: {N_raw}")
        return [], []

    # -------- 在真实帧轴上插值：补齐成每一帧一个速度 --------
    f_min, f_max = int(frames[0]), int(frames[-1])
    frames_full = np.arange(f_min, f_max + 1, dtype=int)

    # 线性插值（np.interp 会自动处理缺失帧；两端用端点值外推）
    speeds_full = np.interp(frames_full, frames, speeds)

    # 如果你不想两端“外推成常数”，可以选择只保留原始范围内（其实 frames_full 就是原始范围）
    # speeds_full = np.interp(frames_full, frames, speeds, left=np.nan, right=np.nan)  # 需要后续处理nan

    frames = frames_full.astype(float)  # 后续你的代码用 float 做 abs(frame-fi) 没问题
    speeds = speeds_full.astype(float)
    N = len(speeds)

    if N < 4:
        print(f"⚠️ 有效数据太少: {N}")
        return [], []

    # -------- 1. 估计 fps（用于把“秒”换成“帧数”） --------
    fps = 30.0  # 默认
    try:
        cap_fps = cv2.VideoCapture(video_path)
        if cap_fps.isOpened():
            fps_val = cap_fps.get(cv2.CAP_PROP_FPS)
            if fps_val and fps_val > 1e-3:
                fps = float(fps_val)
        cap_fps.release()
    except Exception as e:
        # 如果 video_path 不是实际视频文件，或者读取失败，就用默认 30fps
        pass

    # 你可以根据实际情况调这个
    # 典型设定：Savitzky-Golay 轻平滑窗口 ~0.3s，强平滑窗口 ~0.7s
    mild_win_time = 0.3   # 秒
    strong_win_time = 0.4 # 秒
    min_peak_distance_time = 0.12  # 两个极小值之间至少间隔 0.5 秒
    coarse_neighborhood_time = 0.4  # 判定“局部极小”时的邻域宽度
    refine_neighborhood_time = 0.07  # 在原始曲线上精修时的邻域宽度

    # 换算成帧数（使用“帧号差值”，不是有效样本 index）
    min_peak_distance_frames = max(1, int(min_peak_distance_time * fps))
    coarse_neighborhood_frames = max(1, int(coarse_neighborhood_time * fps))
    refine_neighborhood_frames = max(1, int(refine_neighborhood_time * fps))

    # -------- 2. Savitzky-Golay 轻平滑 + 强平滑 --------
    def _sg_smooth(x, win_time, polyorder=3):
        """按时间窗口长度（秒）设置 Savitzky-Golay 的窗口，并做好边界检查。"""
        if len(x) < 5:
            return x.copy()
        win = int(win_time * fps)
        if win < 3:
            win = 3
        if win % 2 == 0:
            win += 1
        # 最大不能超过 N（且必须是奇数）
        if win > len(x):
            win = len(x) if len(x) % 2 == 1 else len(x) - 1
        if win < 3:
            return x.copy()
        poly = min(polyorder, win - 1)
        try:
            return savgol_filter(x, window_length=win, polyorder=poly, mode='nearest')
        except Exception:
            return x.copy()

    speeds_mild = _sg_smooth(speeds, mild_win_time)
    speeds_strong = _sg_smooth(speeds_mild, strong_win_time)

    # -------- 3. robust 阈值（中位数 + MAD） --------
    med = np.median(speeds_strong)
    mad = np.median(np.abs(speeds_strong - med))
    if mad < 1e-6:
        # 如果 MAD 非常小，就退化用 std
        mad = np.std(speeds_strong) + 1e-6

    # “明显比整体慢”的阈值：越小越严格
    alpha = 0.8  # 你可以在 0.5~1.5 之间调
    slow_thr = med - alpha * mad
    slow_thr = max(0.0, slow_thr)  # 速度不能 < 0


    from scipy.signal import find_peaks

    # ---------- 4) 在强平滑曲线上找极小值候选：find_peaks(-x) ----------
    x = speeds_strong.copy()

    # 自适应 prominence：用 MAD 或 std 做尺度
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-6:
        mad = np.std(x) + 1e-6

    # prominence 取一个“不会太严”的值：你可以从 0.3~1.0 之间调
    prom_k = 0.4
    min_prom = prom_k * mad

    # distance：建议别用 0.12s 那么大，先小一点防漏检（比如 2~5 帧）
    min_peak_distance_frames = max(1, int(min_peak_distance_time * fps))
    min_peak_distance_frames = min(min_peak_distance_frames, 5)  # 防止过大漏掉密集谷

    # 直接在 -x 上找谷
    peaks, props = find_peaks(-x, prominence=min_prom, distance=min_peak_distance_frames)

    candidate_indices = peaks.tolist()

    # ---------- 4.1) 兜底：导数过零（补平台谷/浅谷） ----------
    dx = np.diff(x)
    sign = np.sign(dx)
    # 从负到正：谷（极小）
    zc = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0] + 1
    candidate_indices += zc.tolist()

    candidate_indices = sorted(set(int(i) for i in candidate_indices if 0 <= i < N))

    if not candidate_indices:
        print(f"⚠️ {video_stem} 未找到候选极小值（find_peaks + zero-cross）")
        return [], []

    # ---------- 5) refine：回到原始 speeds（或 mild）在局部找真正最小 ----------
    refine_neighborhood_frames = max(1, int(refine_neighborhood_time * fps))

    refined_pairs = []
    for idx_c in candidate_indices:
        fc = frames[idx_c]
        refine_mask = np.abs(frames - fc) <= refine_neighborhood_frames
        if not np.any(refine_mask):
            continue

        local_frames = frames[refine_mask]
        # 推荐用 speeds（原始插值后）或 speeds_mild（二者二选一）
        local_speeds = speeds[refine_mask]  # 或者 speeds_mild[refine_mask]

        j = int(np.argmin(local_speeds))
        f_ref = int(local_frames[j])
        s_ref = float(local_speeds[j])
        refined_pairs.append((f_ref, s_ref))

    if not refined_pairs:
        print(f"⚠️ {video_stem} 候选极小值 refine 后为空")
        return [], []

    # -------- 5. 在原始曲线上做局部精修（refine） --------
    refined_pairs = []  # (frame_refined, speed_refined)
    for idx_c in candidate_indices:
        fc = frames[idx_c]
        # 在“真实帧号”意义下 ±refine_neighborhood_frames 范围内找原始速度的最小值
        refine_mask = np.abs(frames - fc) <= refine_neighborhood_frames
        if not np.any(refine_mask):
            continue
        local_frames = frames[refine_mask]
        local_speeds = speeds[refine_mask]

        j_local_min = np.argmin(local_speeds)
        f_ref = int(local_frames[j_local_min])
        s_ref = float(local_speeds[j_local_min])
        refined_pairs.append((f_ref, s_ref))

    if not refined_pairs:
        print(f"⚠️ {video_stem} 候选极小值精修后为空")
        return [], []

    # 去重（同一帧可能被多次 refine 到）
    # 先按帧号聚合，再取该帧最小速度
    frame_to_speed = {}
    for f, s in refined_pairs:
        if f not in frame_to_speed or s < frame_to_speed[f]:
            frame_to_speed[f] = s
    pairs = sorted(frame_to_speed.items(), key=lambda x: x[1])  # 按速度从小到大排


    selected = pairs[:]  # 不做 NMS，全部保留

    result = [f for f, s in selected]
    result_speeds = [s for f, s in selected]
    print(f"🖨️ {video_stem}（{hand} hand）提取到极小值帧（{len(result)}个）： {result}")

    # -------- 7. 可视化速度曲线 + 极小值点（可选） --------
    try:
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            plt.figure(figsize=(12, 4))
            # 用 frames 做 x 轴，更符合真实时间
            plt.plot(frames, speeds, label="raw speed", alpha=0.4)
            plt.plot(frames, speeds_mild, label="mild smooth", linewidth=1.5)
            plt.plot(frames, speeds_strong, label="strong smooth", linewidth=1.5)

            # 画出选中的极小值点
            if len(result) > 0:
                sel_frames = np.array(result, dtype=float)
                # 在 mild 曲线上取对应速度，仅用于可视化
                vis_speeds = []
                for f in sel_frames:
                    # 找到离该帧最近的 index
                    idx_near = np.argmin(np.abs(frames - f))
                    vis_speeds.append(speeds_mild[idx_near])
                plt.scatter(sel_frames, vis_speeds, c="red", marker="o", label="selected minima")

            plt.xlabel("Frame index")
            plt.ylabel("Speed")
            plt.title(f"{video_stem} ({hand}) - local minima (fps≈{fps:.1f})")
            plt.legend()
            plt.tight_layout()

            out_png = os.path.join(output_folder, f"{video_stem}_{hand}_local_minima.png")
            plt.savefig(out_png)
            plt.close()
            print(f"🖼️ 速度曲线可视化已保存: {out_png}")
    except Exception as e:
        print(f"⚠️ 可视化保存失败: {e}")

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
def get_json_path(video_name, base_dir="/home/EgoLoc/hand_data_drawer/ego4d_mp4_outputs"):
    if _SPEED_BASE_DIR is not None:
        base_dir = _SPEED_BASE_DIR
    # json_filename = f"{video_name}_with_speed.json"
    # json_path = os.path.join(base_dir, json_filename)
    # return json_path
    """
    e.g. video_name='video29' ->
    /home/EgoLoc/hand_data_drawer/3D_hand_speed_hamer/video29/video29_with_speed.json
    """
    json_dir = os.path.join(base_dir, video_name)
    os.makedirs(json_dir, exist_ok=True)  # 确保目录存在（读/写都安全）
    #json_path = os.path.join(json_dir, f"{video_name}_with_speed.json")
    json_path = os.path.join(json_dir, f"{video_name}_with_speed_twohands.json")
    return json_path


def get_json_folder_path(video_name, base_dir="/home/EgoLoc/hand_data_drawer/mp4_outputs_sorted_test4"):
    if _SPEED_BASE_DIR is not None:
        base_dir = _SPEED_BASE_DIR
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


# def get_contact_separation_pairs(results, speed_data):
#     # 构建速度字典
#     speed_dict = {frame: speed for frame, speed in speed_data}
#
#     # 分离 Contact 和 Separation 事件，并提取帧和速度
#     contacts = []
#     separations = []
#     for event in results:
#         event_type, frame = event
#         if frame not in speed_dict:
#             continue
#         if event_type == "Contact":
#             contacts.append((frame, speed_dict[frame]))
#         elif event_type == "Separation":
#             separations.append((frame, speed_dict[frame]))
#
#     # 按帧索引排序
#     contacts.sort()
#     separations.sort()
#
#     pairs = []
#     contact_candidates = []  # 修正点：存储格式改为 (frame, speed)
#
#     # 混合排序所有事件（保持原有逻辑）
#     all_events = sorted(
#         [("C", frame, speed) for frame, speed in contacts] +
#         [("S", frame, speed) for frame, speed in separations],
#         key=lambda x: x[1]  # 按帧索引排序
#     )
#
#     for event in all_events:
#         event_type, frame, speed = event
#
#         if event_type == "C":
#             # 修正点：使用 speed 比较（索引应为 1）
#             if not contact_candidates or speed < contact_candidates[-1][1]:
#                 contact_candidates.append((frame, speed))
#
#         elif event_type == "S":
#             if contact_candidates:
#                 # 选择速度最小的 Contact（索引应为 1）
#                 best_contact = min(contact_candidates, key=lambda x: x[1])
#                 pairs.append((best_contact[0], frame))
#                 contact_candidates = []
#
#     return pairs
#加上了打印日志，同时增加权重问题
# def get_contact_separation_pairs(
#     results,
#     speed_data,
#     *,
#     min_gap=2,
#     max_gap=60,
#     w_time=0.6,
#     w_speed=0.4,
#     speed_norm=None,
#     verbose=True,
# ):
#     """
#     改进版 Contact-Separation 配对：
#     1) Contact 必须在 Separation 之前
#     2) 在时间窗内，选「离 Separation 最近 + 速度低」的 Contact
#     3) Contact 只用一次，避免误配
#     """
#
#     # -----------------------------
#     # 0. 准备
#     # -----------------------------
#     speed_dict = {f: s for f, s in speed_data}
#
#     contacts = []
#     separations = []
#
#     for etype, frame in results:
#         if frame not in speed_dict:
#             continue
#         if etype == "Contact":
#             contacts.append((frame, speed_dict[frame]))
#         elif etype == "Separation":
#             separations.append((frame, speed_dict[frame]))
#
#     contacts.sort()
#     separations.sort()
#
#     if not contacts or not separations:
#         if verbose:
#             print("[PAIR] No valid contacts or separations.")
#         return []
#
#     if speed_norm is None:
#         speed_norm = max(s for _, s in contacts) + 1e-6
#
#     if verbose:
#         print("\n[CONTACTS]")
#         for f, v in contacts:
#             print(f"  frame={f}, speed={v:.6f}")
#
#         print("\n[SEPARATIONS]")
#         for f, v in separations:
#             print(f"  frame={f}, speed={v:.6f}")
#
#     # -----------------------------
#     # 1. 主匹配循环
#     # -----------------------------
#     used_contacts = set()
#     pairs = []
#
#     for sep_frame, sep_speed in separations:
#
#         candidates = []
#
#         for c_frame, c_speed in contacts:
#             if c_frame >= sep_frame:
#                 continue
#             if c_frame in used_contacts:
#                 continue
#
#             dt = sep_frame - c_frame
#             if dt < min_gap or dt > max_gap:
#                 continue
#
#             # --------- 加权评分 ----------
#             score_time = dt / max_gap
#             score_speed = c_speed / speed_norm
#             score = w_time * score_time + w_speed * score_speed
#
#             candidates.append({
#                 "c_frame": c_frame,
#                 "c_speed": c_speed,
#                 "dt": dt,
#                 "score": score
#             })
#
#         if not candidates:
#             if verbose:
#                 print(f"\n[SKIP] Separation @{sep_frame}: no valid Contact in window")
#             continue
#
#         # 选 score 最小
#         best = min(candidates, key=lambda x: x["score"])
#
#         used_contacts.add(best["c_frame"])
#         pairs.append((best["c_frame"], sep_frame))
#
#         if verbose:
#             print(
#                 f"\n[PAIR] Separation @{sep_frame} matched with "
#                 f"Contact @{best['c_frame']} | "
#                 f"dt={best['dt']} "
#                 f"speed={best['c_speed']:.6f} "
#                 f"score={best['score']:.4f}"
#             )
#
#     if verbose:
#         print("\n[FINAL PAIRS]")
#         for c, s in pairs:
#             print(f"  ({c}, {s})")
#
#     return pairs
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from PIL import Image


# ----------------------------
# Small utils
# ----------------------------
def _layout_rc(n: int) -> Tuple[int, int]:
    """tight grid layout close to square"""
    if n <= 0:
        return 1, 1
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _parse_choice_number(text: str) -> Optional[int]:
    """Parse output like '3' or 'C3' or 'S2' -> returns integer index (1-based)."""
    if text is None:
        return None
    s = str(text).strip().upper()
    if s == "NONE":
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    return int(m.group(1))


def _read_frame_bgr(cap: cv2.VideoCapture, idx: int, fallback_shape=None) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame
    if fallback_shape is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    return np.zeros(fallback_shape, dtype=np.uint8)


def _resize_keep_w(img: np.ndarray, w: int = 260) -> np.ndarray:
    h0, w0 = img.shape[:2]
    if w0 <= 0:
        return img
    scale = w / float(w0)
    h = max(1, int(h0 * scale))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# def _make_referee_grid(
#     *,
#     video_path: str,
#     contact_frames: List[int],
#     separation_frames: List[int],
#     fixed_side: str,   # "S" or "C"
#     fixed_frame: int,
#     cell_w: int = 260,
#     debug_save_path: Optional[str] = None,
# ) -> np.ndarray:
#     """
#     Build a labeled grid for VLM:
#       - If fixed_side == "S": cells are C1..Ck and last cell is S (fixed_frame)
#       - If fixed_side == "C": cells are S1..Sk and last cell is C (fixed_frame)
#     """
#     assert fixed_side in ("S", "C")
#
#     if fixed_side == "S":
#         frames = list(contact_frames) + [fixed_frame]
#         labels = [f"C{i+1}" for i in range(len(contact_frames))] + ["S"]
#         highlight_last = True
#     else:
#         frames = list(separation_frames) + [fixed_frame]
#         labels = [f"S{i+1}" for i in range(len(separation_frames))] + ["C"]
#         highlight_last = True
#
#     n = len(frames)
#     rows, cols = _layout_rc(n)
#
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     # pre-read first valid for fallback shape
#     fallback = None
#     if total_frames > 0:
#         fallback = _resize_keep_w(_read_frame_bgr(cap, 0), cell_w).shape
#
#     tiles = []
#     for f in frames:
#         f = int(max(0, min(total_frames - 1, int(f)))) if total_frames > 0 else int(f)
#         img = _read_frame_bgr(cap, f, fallback_shape=fallback)
#         img = _resize_keep_w(img, cell_w)
#         tiles.append(img)
#     cap.release()
#
#     # pad to rows*cols
#     need = rows * cols
#     if len(tiles) < need:
#         black = np.zeros_like(tiles[0])
#         tiles.extend([black] * (need - len(tiles)))
#         labels = labels + [""] * (need - len(labels))
#
#     th, tw = tiles[0].shape[:2]
#     grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
#
#     for i in range(rows):
#         for j in range(cols):
#             k = i * cols + j
#             tile = tiles[k]
#             y1, y2 = i * th, (i + 1) * th
#             x1, x2 = j * tw, (j + 1) * tw
#             grid[y1:y2, x1:x2] = tile
#
#             lab = labels[k]
#             if lab:
#                 # label background
#                 cv2.rectangle(grid, (x1, y1), (x1 + 90, y1 + 35), (255, 255, 255), -1)
#                 cv2.putText(grid, lab, (x1 + 8, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
#
#             # highlight fixed frame (last real cell index n-1)
#             if highlight_last and (k == n - 1):
#                 cv2.rectangle(grid, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (0, 0, 255), 4)
#
#     if debug_save_path:
#         p = Path(debug_save_path)
#         p.parent.mkdir(parents=True, exist_ok=True)
#         rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
#         Image.fromarray(rgb).save(str(p))
#
#     return grid
def _make_referee_grid(
    *,
    video_path: str,
    contact_frames: List[int],
    separation_frames: List[int],
    fixed_side: str,   # "S" or "C"
    fixed_frame: int,
    cell_w: int = 260,
    debug_save_path: Optional[str] = None,
    sort_by_distance: bool = False,   # NEW: optional
) -> np.ndarray:
    """
    Build a labeled grid for VLM:
      - fixed_side == "S": cells are C1..Ck and last cell is S (fixed_frame)
      - fixed_side == "C": cells are S1..Sk and last cell is C (fixed_frame)

    Enhancements:
      - show frame index in each cell label (e.g., C3@001234)
      - mark fixed cell as FIXED explicitly
      - robust handling if video cannot be opened / no frames
      - dynamic label background width
    """
    assert fixed_side in ("S", "C")

    # ---- build frames + labels (without frame index yet) ----
    if fixed_side == "S":
        cand_frames = list(contact_frames)
        cand_labels = [f"C{i+1}" for i in range(len(cand_frames))]
        fixed_label = "S"
    else:
        cand_frames = list(separation_frames)
        cand_labels = [f"S{i+1}" for i in range(len(cand_frames))]
        fixed_label = "C"

    # optional: reorder candidates by distance to fixed_frame for VLM stability
    if sort_by_distance and len(cand_frames) > 1:
        order = sorted(range(len(cand_frames)), key=lambda i: abs(int(cand_frames[i]) - int(fixed_frame)))
        cand_frames = [cand_frames[i] for i in order]
        cand_labels = [cand_labels[i] for i in order]

    frames = cand_frames + [fixed_frame]
    labels = cand_labels + [fixed_label]
    n = len(frames)

    rows, cols = _layout_rc(n)

    # ---- open video ----
    cap = cv2.VideoCapture(video_path)
    opened = cap.isOpened()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0

    # fallback shape: try reading frame 0
    fallback_shape = None
    if opened and total_frames > 0:
        fr0 = _read_frame_bgr(cap, 0)
        if fr0 is not None:
            fr0r = _resize_keep_w(fr0, cell_w)
            fallback_shape = fr0r.shape

    # helper to create a black tile if needed
    def _black_tile():
        if fallback_shape is not None:
            return np.zeros(fallback_shape, dtype=np.uint8)
        # last resort
        return np.zeros((int(cell_w * 9 / 16), cell_w, 3), dtype=np.uint8)

    tiles = []
    norm_frames = []
    for f in frames:
        if opened and total_frames > 0:
            ff = int(max(0, min(total_frames - 1, int(f))))
        else:
            ff = int(f)
        norm_frames.append(ff)

        if not opened:
            img = None
        else:
            img = _read_frame_bgr(cap, ff, fallback_shape=fallback_shape)

        if img is None:
            tile = _black_tile()
        else:
            tile = _resize_keep_w(img, cell_w)

        tiles.append(tile)

    if opened:
        cap.release()

    # ---- pad to rows*cols ----
    need = rows * cols
    if len(tiles) < need:
        tiles.extend([_black_tile()] * (need - len(tiles)))
        labels = labels + [""] * (need - len(labels))
        norm_frames = norm_frames + [None] * (need - len(norm_frames))

    th, tw = tiles[0].shape[:2]
    grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)

    # ---- draw ----
    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            tile = tiles[k]
            y1, y2 = i * th, (i + 1) * th
            x1, x2 = j * tw, (j + 1) * tw
            grid[y1:y2, x1:x2] = tile

            lab = labels[k]
            ff = norm_frames[k]

            if lab:
                # compose label with frame index
                if ff is not None:
                    text = f"{lab}@{int(ff):06d}"
                else:
                    text = f"{lab}"

                # mark fixed cell explicitly
                if k == n - 1:
                    text = f"{text} FIXED"

                # dynamic background size
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.65
                thickness = 2
                (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
                pad_x, pad_y = 8, 6
                bg_w = min(tw - 4, w + 2 * pad_x)
                bg_h = h + 2 * pad_y

                cv2.rectangle(
                    grid,
                    (x1 + 2, y1 + 2),
                    (x1 + 2 + bg_w, y1 + 2 + bg_h),
                    (255, 255, 255),
                    -1,
                )
                cv2.putText(
                    grid,
                    text,
                    (x1 + 2 + pad_x, y1 + 2 + pad_y + h),
                    font,
                    scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )

            # highlight fixed frame (last real cell index n-1)
            if k == n - 1:
                cv2.rectangle(grid, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (0, 0, 255), 4)

    if debug_save_path:
        p = Path(debug_save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(str(p))

    return grid



def _vlm_choose_one(
    *,
    credentials,
    grid_bgr: np.ndarray,
    prompt: str,
) -> Optional[int]:
    """
    Returns:
      - chosen index (1-based) OR None (for NONE/invalid)
    NOTE: we call scene_understanding with principle="feedback" so it returns raw string.
    """
    ans_text = scene_understanding(credentials, grid_bgr, prompt, principle="feedback")
    if ans_text is None:
        return None
    return _parse_choice_number(ans_text)


def _fallback_choose_closest(
    *,
    fixed_side: str,  # "S" or "C"
    fixed_frame: int,
    candidates: List[int],
) -> Optional[int]:
    if not candidates:
        return None
    # choose minimal temporal distance to fixed_frame (best local heuristic)
    cand = min(candidates, key=lambda x: abs(int(x) - int(fixed_frame)))
    return int(cand)


# ----------------------------
# Referee prompts
# ----------------------------
PROMPT_S_FIXED = """
You will see a grid of frames.

- The FIXED frame is the SEPARATION frame labeled S (same episode target).
- The candidate frames are CONTACT frames labeled C1, C2, C3, ...

Task:
Choose the ONE contact candidate that belongs to the SAME interaction episode as S.
If none match, output NONE.

Rules:
1) Temporal: a valid Contact MUST occur BEFORE S in time.
2) Episode consistency: Match based on the same object, consistent hand-object relation, and the same ongoing action.
3) Distance consistency (important):
   - The matched Contact should typically show the hand touching or VERY CLOSE to the object,
     because S is a separation moment from that same interaction.
   - Do NOT choose a contact candidate where the hand is clearly far away from the object;
     that likely belongs to a different episode.
   - Exception: if motion clearly indicates a fast throw / rapid pull-away, larger distance changes may be acceptable.
4) Ignore tiny jitters, brief accidental touches, occlusion artifacts, or unrelated contacts.

Output format (STRICT):
- Output ONE token only: either a single integer (1,2,3,...) or NONE.
- Do NOT output any other words.
""".strip()

PROMPT_C_FIXED = """
You will see a grid of frames.

- The FIXED frame is the CONTACT frame labeled C (same episode target).
- The candidate frames are SEPARATION frames labeled S1, S2, S3, ...

Task:
Choose the ONE separation candidate that belongs to the SAME interaction episode as C.
If none match, output NONE.

Rules:
1) Temporal: a valid Separation MUST occur AFTER C in time.
2) Episode consistency: Match based on the same object, consistent hand-object relation, and the same ongoing action.
3) Distance consistency (important):
   - The matched Separation should look like a REAL release: hand and object go from touching/very close
     to a small visible gap (just separated), not instantly to very far.
   - Prefer candidates where the hand is still near the object right after release.
   - Exception: if motion clearly indicates a fast throw / rapid pull-away, a larger distance may be acceptable.
4) Ignore tiny releases that are not actual separation (e.g., brief slack, occlusion).

Output format (STRICT):
- Output ONE token only: either a single integer (1,2,3,...) or NONE.
- Do NOT output any other words.
""".strip()



# ----------------------------
# Main: level-3 pairing with iterative conflict resolution
# ----------------------------
def level3_pairing(
    *,
    results: List[Tuple[str, int]],
    video_path: str,
    credentials,
    speed_data: Optional[List[Tuple[int, float]]] = None,   # NEW
    matching_cfg: Optional[dict] = None,                    # NEW
    max_candidates_per_query: int = 12,
    max_rounds: int = 3,
    debug_dir: Optional[Path] = None,
) -> List[Tuple[int, int]]:

    """
    Level-3 pairing:
      1) initial: for each Separation, VLM chooses Contact (or NONE)
      2) iterative conflicts:
         - Contact matched multiple times -> for that Contact, VLM selects ONE Separation among conflicts
         - Separation matched multiple times -> for that Separation, VLM selects ONE Contact among conflicts
      3) iterate until stable or max_rounds
      4) final enforce one-to-one

    Returns:
      pairs: [(contact_frame, separation_frame), ...]
    """

    # ---- collect events ----
    contacts = sorted({int(f) for t, f in results if t == "Contact" and isinstance(f, int)})
    separations = sorted({int(f) for t, f in results if t == "Separation" and isinstance(f, int)})

    print("\n[Level-3] Raw Contacts:", contacts)
    print("[Level-3] Raw Separations:", separations)
    if not contacts or not separations:
        return []

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    # ---- helper: downsample candidates (keep closest + uniform) ----
    def _downsample(cands: List[int], center: int, limit: int) -> List[int]:
        cands = sorted(set(int(x) for x in cands))
        if len(cands) <= limit:
            return cands
        half = max(1, limit // 2)
        closest = sorted(cands, key=lambda x: abs(x - center))[:half]
        remain = [x for x in cands if x not in set(closest)]
        if not remain:
            return sorted(set(closest))[:limit]
        step = max(1, len(remain) // max(1, (limit - len(closest))))
        uniform = remain[::step][: (limit - len(closest))]
        return sorted(set(closest + uniform))[:limit]

    from collections import defaultdict

    pair_meta = defaultdict(lambda: {"picked": 0, "src": set()})
    # picked: 被 VLM 选中的次数（越多越可信）
    # src: 这条边来自哪个阶段：{"init", "confC", "confS"} 便于调试

    # ---- step 1: initial pairing (each S picks one C) ----
    pairs: List[Tuple[int, int]] = []
    for s in separations:
        cands = [c for c in contacts if c < s]
        if not cands:
            continue
        show_cs = _downsample(cands, center=s, limit=max_candidates_per_query)

        dbg = None
        if debug_dir is not None:
            dbg = str(debug_dir / f"init_S{s}.png")

        grid = _make_referee_grid(
            video_path=video_path,
            contact_frames=show_cs,
            separation_frames=[],
            fixed_side="S",
            fixed_frame=s,
            debug_save_path=dbg
        )
        k = _vlm_choose_one(credentials=credentials, grid_bgr=grid, prompt=PROMPT_S_FIXED)
        if k is None:
            continue

        # k refers to Ck (1-based)
        idx = k - 1
        if 0 <= idx < len(show_cs):
            c_chosen = int(show_cs[idx])
            s_int = int(s)
            pairs.append((c_chosen, s_int))
            pair_meta[(c_chosen, s_int)]["picked"] += 1
            pair_meta[(c_chosen, s_int)]["src"].add("init")

    if not pairs:
        print("[Level-3] No initial pairs.")
        return []

    # ---- conflict finder ----
    def _find_conflicts(pairs_: List[Tuple[int, int]]):
        c2s = defaultdict(list)
        s2c = defaultdict(list)
        for c, s in pairs_:
            c2s[int(c)].append(int(s))
            s2c[int(s)].append(int(c))
        contact_conf = {c: ss for c, ss in c2s.items() if len(ss) > 1}
        sep_conf = {s: cs for s, cs in s2c.items() if len(cs) > 1}
        print(f"[L3 CONFLICT] contact_conflicts={contact_conf}")
        print(f"[L3 CONFLICT] separation_conflicts={sep_conf}")

        return contact_conf, sep_conf

    # ---- resolve conflicts iteratively ----
    for rd in range(max_rounds):
        contact_conf, sep_conf = _find_conflicts(pairs)

        print(f"\n[Iter {rd}] pairs={len(pairs)} contact_conf={len(contact_conf)} sep_conf={len(sep_conf)}")

        if not contact_conf and not sep_conf:
            print("[Iter] stable, stop.")
            break

        changed = False

        # -------------------------
        # Phase A: resolve Contact conflicts (C matched to many S)
        # fixed C -> choose ONE S
        # -------------------------
        if contact_conf:
            new_pairs = []
            drop_keys = set(contact_conf.keys())

            # keep non-conflict pairs
            for c, s in pairs:
                if c not in drop_keys:
                    new_pairs.append((c, s))

            # solve each conflict group
            for c, ss in contact_conf.items():
                # only keep separations after contact
                ss = [s for s in ss if int(s) > int(c)]
                if not ss:
                    continue

                show_ss = _downsample(ss, center=c, limit=max_candidates_per_query)

                dbg = None
                if debug_dir is not None:
                    dbg = str(debug_dir / f"confC_C{c}.png")

                grid = _make_referee_grid(
                    video_path=video_path,
                    contact_frames=[],
                    separation_frames=show_ss,
                    fixed_side="C",
                    fixed_frame=c,
                    debug_save_path=dbg
                )
                k = _vlm_choose_one(credentials=credentials, grid_bgr=grid, prompt=PROMPT_C_FIXED)

                if k is None:
                    chosen = _fallback_choose_closest(fixed_side="C", fixed_frame=c, candidates=show_ss)
                else:
                    idx = k - 1
                    chosen = show_ss[idx] if 0 <= idx < len(show_ss) else None

                if chosen is not None:
                    c_int = int(c)
                    s_chosen = int(chosen)
                    new_pairs.append((c_int, s_chosen))
                    pair_meta[(c_int, s_chosen)]["picked"] += 1
                    pair_meta[(c_int, s_chosen)]["src"].add("confC")
                    changed = True

            pairs = new_pairs

        # recompute after phase A
        contact_conf, sep_conf = _find_conflicts(pairs)

        # -------------------------
        # Phase B: resolve Separation conflicts (S matched to many C)
        # fixed S -> choose ONE C
        # -------------------------
        if sep_conf:
            new_pairs = []
            drop_keys = set(sep_conf.keys())

            # keep non-conflict pairs
            for c, s in pairs:
                if s not in drop_keys:
                    new_pairs.append((c, s))

            for s, cs in sep_conf.items():
                # only keep contacts before separation
                cs = [c for c in cs if int(c) < int(s)]
                if not cs:
                    continue

                show_cs = _downsample(cs, center=s, limit=max_candidates_per_query)

                dbg = None
                if debug_dir is not None:
                    dbg = str(debug_dir / f"confS_S{s}.png")

                grid = _make_referee_grid(
                    video_path=video_path,
                    contact_frames=show_cs,
                    separation_frames=[],
                    fixed_side="S",
                    fixed_frame=s,
                    debug_save_path=dbg
                )
                k = _vlm_choose_one(credentials=credentials, grid_bgr=grid, prompt=PROMPT_S_FIXED)

                if k is None:
                    chosen = _fallback_choose_closest(fixed_side="S", fixed_frame=s, candidates=show_cs)
                else:
                    idx = k - 1
                    chosen = show_cs[idx] if 0 <= idx < len(show_cs) else None

                if chosen is not None:
                    c_chosen = int(chosen)
                    s_int = int(s)
                    new_pairs.append((c_chosen, s_int))
                    pair_meta[(c_chosen, s_int)]["picked"] += 1
                    pair_meta[(c_chosen, s_int)]["src"].add("confS")
                    changed = True

            pairs = new_pairs

        if not changed:
            print("[Iter] no change in this round, stop.")
            break
    speed_dict = None
    if speed_data:
        speed_dict = {int(f): float(v) for f, v in speed_data if np.isfinite(v)}

    def speed_score_fn(c, s):
        vc = speed_dict.get(c, 1e6)
        vs = speed_dict.get(s, 1e6)
        return 0.5 * vc + 0.5 * vs

    # ---- final enforce one-to-one (global optimal matching) ----
    # 可选：如果你有速度函数，就传进去；没有就 None
    matching_cfg = matching_cfg or {}
    final_pairs = _final_one_to_one_by_matching(
        pairs,
        pair_meta=pair_meta,
        W=matching_cfg.get("W", 300),
        penalty=matching_cfg.get("penalty", 1000.0),
        w_dt=matching_cfg.get("w_dt", 1.0),
        w_window=matching_cfg.get("w_window", 1.0),
        w_vlm=matching_cfg.get("w_vlm", -120.0),
        speed_score_fn=speed_score_fn,  # ✅ now enabled
        w_speed=matching_cfg.get("w_speed", 1.0),  # ✅ tune here
    )

    print("\n[Level-3] Final pairs:", final_pairs)
    return final_pairs


def speed_score_fn(c, s):
    # 举例：假设你有 dict: sep_to_min_frame[s] = frame_min
    m = sep_to_min_frame.get(s)
    if m is None:
        return 0.0
    return abs(c - m) * 0.1  # 权重0.1只是示例
def _hungarian_min_cost(cost):
    """
    Hungarian algorithm for rectangular matrices (min-cost assignment).
    cost: list[list[float]] shape (n, m)
    returns: assignment list a where a[i] = j matched column for row i, or -1
    """
    n = len(cost)
    m = len(cost[0]) if n else 0

    transposed = False
    if n > m:
        transposed = True
        cost = [list(row) for row in zip(*cost)]
        n, m = m, n

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1

    if transposed:
        # convert back: original rows assignment
        orig_n = m  # original rows
        inv = [-1] * orig_n
        for r, c in enumerate(assignment):
            if c != -1:
                inv[c] = r
        return inv

    return assignment
def _final_one_to_one_by_matching(
    pairs,
    pair_meta=None,
    *,
    W=300,                # 时间窗（帧数）：例如 300帧 ≈ 10s@30fps
    penalty=1000.0,       # 超窗惩罚：要大于典型 |s-c|
    w_dt=1.0,             # |s-c| 权重
    w_window=1.0,         # 超窗惩罚权重
    w_vlm=-120.0,         # VLM偏好权重（负号：picked越多 cost越低）
    speed_score_fn=None,  # 可选：额外成本函数 speed_score_fn(c,s)->float
    w_speed=1.0,          # speed项权重
    forbid_inf=1e12,      # 禁止边的超大成本
):
    """
    pairs: list[(c, s)] candidate edges
    pair_meta: dict[(c,s)] -> {"picked": int, "src": set()}
    returns: list[(c, s)] one-to-one set minimizing total cost.
    """
    pairs = sorted(set((int(c), int(s)) for c, s in pairs))
    if not pairs:
        return []

    pair_meta = pair_meta or {}

    Cs = sorted({c for c, _ in pairs})
    Ss = sorted({s for _, s in pairs})
    ci = {c: i for i, c in enumerate(Cs)}
    sj = {s: j for j, s in enumerate(Ss)}

    INF = float(forbid_inf)
    cost = [[INF] * len(Ss) for _ in range(len(Cs))]

    # Fill costs only for allowed edges present in pairs
    for c, s in pairs:
        dt = abs(s - c)
        window_cost = 0.0
        if dt > W:
            window_cost = penalty + (dt - W) * 0.1  # 超窗后随距离略增（可调）

        picked = 0
        meta = pair_meta.get((c, s))
        if meta is not None:
            picked = int(meta.get("picked", 0))

        vlm_bonus_cost = w_vlm * picked  # picked越大，cost越低（因为w_vlm为负）

        speed_cost = 0.0
        if speed_score_fn is not None:
            try:
                speed_cost = float(speed_score_fn(c, s))
            except Exception:
                speed_cost = 0.0

        total_cost = (
            w_dt * dt
            + w_window * window_cost
            + vlm_bonus_cost
            + w_speed * speed_cost
        )

        cost[ci[c]][sj[s]] = total_cost

    assign = _hungarian_min_cost(cost)

    final_pairs = []
    for i, j in enumerate(assign):
        if j == -1:
            continue
        if cost[i][j] >= INF / 2:
            continue
        final_pairs.append((Cs[i], Ss[j]))

    final_pairs.sort(key=lambda x: (x[1], x[0]))
    return final_pairs

def vlm_referee_all_contacts_for_separation(
    *,
    credentials,
    video_path: str,
    contact_frames: list,
    separation_frame: int,
    max_contacts: int = 12,
    principle: str = "referee_all",
    debug_save_path: str = None,
):
    """
    Given many Contact frames and one Separation frame,
    ask VLM to choose which Contact matches this Separation, or NONE.
    """

    # -------- sanity --------
    contacts_before = sorted(
        {c for c in contact_frames if isinstance(c, int) and c < separation_frame}
    )
    if not contacts_before:
        return None

    # -------- downsample if needed --------
    if len(contacts_before) > max_contacts:
        closest = sorted(
            contacts_before,
            key=lambda x: abs(separation_frame - x)
        )[: max_contacts // 2]

        remain = [c for c in contacts_before if c not in set(closest)]
        step = max(1, len(remain) // max(1, (max_contacts - len(closest))))
        uniform = remain[::step][: (max_contacts - len(closest))]

        contacts_show = sorted(set(closest + uniform))
    else:
        contacts_show = contacts_before

    # -------- build grid: C1..Ck + S --------
    frames = contacts_show + [separation_frame]
    minima_index = len(frames) - 1  # highlight S

    grid_bgr = create_frame_grid_with_keyframe(
        video_path,
        frames,
        grid_size=len(frames),

    )

    if debug_save_path:
        import cv2
        from PIL import Image
        rgb = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)
        from pathlib import Path
        p = Path(debug_save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(debug_save_path)

    # -------- prompt --------
    prompt = """
You will see a grid of frames.
Some frames are CONTACT candidates labeled C1, C2, C3, ...
The last frame is the SEPARATION frame labeled S.

Task:
Choose which CONTACT belongs to the SAME interaction episode as S.
If none match, output NONE.

Rules:
- Contact must occur before S
- Match based on same object and consistent interaction
- Ignore tiny hand jitters or unrelated touches

Output format (ONE token only):
- number (1,2,3,...) or NONE
"""

    # -------- VLM call --------
    answer = scene_understanding(
        credentials,
        grid_bgr,
        prompt,
        principle=principle
    )

    print(f"[VLM referee] S={separation_frame} → {answer}")

    if not answer:
        return None

    ans = str(answer).strip().upper()
    if ans == "NONE":
        return None

    import re
    m = re.search(r"(\d+)", ans)
    if not m:
        return None

    idx = int(m.group(1)) - 1
    if 0 <= idx < len(contacts_show):
        return contacts_show[idx]

    return None

def get_contact_separation_pairs(
    *,
    results,
    video_path,
    credentials,
    max_contacts_per_query=12,
    debug_dir=None,
):
    """
    Level-3: VLM-only pairing.
    For each Separation, ask VLM to choose the best Contact or NONE.

    Args:
        results: [(event_type, frame_idx), ...]
        video_path: path to video
        credentials: VLM credentials
        max_contacts_per_query: limit contacts shown to VLM
        debug_dir: optional Path to save referee grids

    Returns:
        pairs: [(contact_frame, separation_frame), ...]
    """

    # ----------------------------
    # 1) Collect raw events
    # ----------------------------
    contacts = sorted({f for t, f in results if t == "Contact"})
    separations = sorted({f for t, f in results if t == "Separation"})

    print("\n[Level-3] Raw Contacts:", contacts)
    print("[Level-3] Raw Separations:", separations)

    if not contacts or not separations:
        return []

    pairs = []

    # ----------------------------
    # 2) For each Separation → VLM chooses Contact
    # ----------------------------
    for s in separations:
        c = vlm_referee_all_contacts_for_separation(
            credentials=credentials,
            video_path=video_path,
            contact_frames=contacts,
            separation_frame=s,
            max_contacts=max_contacts_per_query,
            debug_save_path=(
                str(debug_dir / f"referee_S{s}.png")
                if debug_dir is not None else None
            ),
        )

        if c is not None:
            pairs.append((c, s))
            print(f"[PAIR] Contact {c}  →  Separation {s}")
        else:
            print(f"[PAIR] Separation {s}: NONE")

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
            "max_tokens": 400,
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
            "max_tokens": 500,
            "temperature": 0.0,
            "top_p": 1.0,
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
    if principle == "state_score":
        # 返回原始文本，交给 parse_state_scores
        return result.choices[0].message.content
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




def create_frame_grid_state(video_path, frame_indices, grid_size=None, hand="hand",
                            debug=False, show="cropped"):
    """
    show:
      - "cropped": grid 里放裁剪后的图（推荐给 VLM）
      - "vis":     grid 里放画了 prev_box 的原图（用于你肉眼检查裁剪是否选对手）
    """

    # ---------- 0) 自动 grid_size ----------
    n = len(frame_indices)
    if n == 0:
        return np.zeros((400, 400, 3), dtype=np.uint8)

    if grid_size is None:
        # 自动排版：尽量接近平方
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        grid_size = (rows, cols)
    else:
        rows, cols = grid_size
        if rows * cols < n:
            # 你传的格子不够就自动扩一下（避免丢帧）
            cols = max(cols, int(math.ceil(n / rows)))
            grid_size = (rows, cols)

    rows, cols = grid_size
    spacer = 0

    # ---------- 1) 读帧并裁剪 ----------
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    prev_box = None  # 同一个 grid 内保持稳定跟踪

    for idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame_bgr = video.read()

        if not success or frame_bgr is None:
            print(f"Warning: Frame {idx} not found (total {total_frames}). Using black frame.")
            black = np.zeros((400, 400, 3), dtype=np.uint8) if not frames else np.zeros_like(frames[0])
            frames.append(black)
            continue

        # 统一尺寸（保持 width=400）
        frame_bgr = image_resize(frame_bgr, width=400)

        # 保留裁剪前的原图（用于 debug 画框）
        orig = frame_bgr.copy()

        # 裁剪（返回 cropped_img + new_prev_box）
        cropped, prev_box = run_groundingdino_and_crop_stable(
            frame_bgr,
            hand=hand,
            mirror=False,
            prev_box=prev_box,
            expand_ratio=0.25,
            box_thresh=0.3
        )

        # 兜底：如果 cropped 失败就回退到原图
        if cropped is None or not isinstance(cropped, np.ndarray):
            cropped = orig

        # 给 cropped 打 frame 标记（你想看的“在哪里看出来”最直接就是这里）
        cv2.putText(
            cropped, f"{hand} frame={idx}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        # debug 可视化：在原图上画 prev_box
        if debug and prev_box is not None:
            vis = orig.copy()
            x0, y0, x1, y1 = map(int, prev_box)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                vis, f"{hand} box @ frame={idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            frames.append(vis if show == "vis" else cropped)
        else:
            frames.append(cropped)

    video.release()

    # ---------- 2) 填充黑帧到 rows*cols ----------
    total_needed = rows * cols
    if len(frames) < total_needed:
        black = np.zeros_like(frames[0])
        frames.extend([black] * (total_needed - len(frames)))

    # ---------- 3) 拼 grid ----------
    fh, fw = frames[0].shape[:2]
    gh = rows * fh + (rows - 1) * spacer
    gw = cols * fw + (cols - 1) * spacer
    grid_img = np.ones((gh, gw, 3), dtype=np.uint8) * 255

    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            frame = frames[k]
            if frame is None or not isinstance(frame, np.ndarray):
                frame = np.zeros((fh, fw, 3), dtype=np.uint8)
            y1 = i * (fh + spacer)
            y2 = y1 + fh
            x1 = j * (fw + spacer)
            x2 = x1 + fw
            grid_img[y1:y2, x1:x2] = frame

    return grid_img

import math
import numpy as np
import cv2




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

def build_context_frames(center_frame: int, all_frames_np, total_frames: int, ctx_win: int = 2):
    # ctx_win=2 => [t-2, t-1, t, t+1, t+2]（会映射到最近有效帧）
    return _build_mapped_window(center_frame, ctx_win, all_frames_np, total_frames)
#冲突检测
from collections import defaultdict



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
# Event Logic (Left -> Right):
# - If {contact_event} -> "Event: Contact"
# - If {sep_event} -> "Event: Separation"
# - Otherwise -> "Event: Neither"
# ===== END PATCH: helpers =====

# prompt_state = f"""
# Instruction:
#
# You are given a time-ordered image grid (earlier→later).
# Determine whether there is a POSSIBLY transition to Contact or to Separation within the grid, and output Event.
# Definitions (per frame):
# - {contact_def}
# - {sep_def}
#
# # Event Logic (Left -> Right):
# # - If {contact_event} -> "Event: Contact"
# # - If {sep_event} -> "Event: Separation"
# # - Otherwise -> "Event: Neither"
#
#
#
# Strict Output Format:
# Output exactly one line:
# "Event: Contact" OR "Event: Separation" OR "Event: Neither"
# """.strip()
import re

def parse_state_scores(vlm_output: str):
    """
    Parse:
    Contact: 0.73
    Separation: 0.21
    """
    if not isinstance(vlm_output, str):
        return None, None

    m1 = re.search(r"Contact\s*:\s*([0-9]*\.?[0-9]+)", vlm_output, re.I)
    m2 = re.search(r"Separation\s*:\s*([0-9]*\.?[0-9]+)", vlm_output, re.I)

    c = float(m1.group(1)) if m1 else None
    s = float(m2.group(1)) if m2 else None
    return c, s


def decide_state_from_scores(c, s, contact_th=0.35, separation_th=0.35, margin=0.10):
    """
    宽进严出：
    - 只有“明显更像 Contact/Separation”才下硬标签
    - 否则 Ambiguous，后续交给 feedback / pairing / referee
    """


    if c >= contact_th and c >= s + margin:
        return "Contact"
    if s >= separation_th and s >= c + margin:
        return "Separation"
    return "Ambiguous"
#2.6修改prompt，增加transition判断，原版如下
# prompt_state = f"""
#     Instruction:
#     You are given a time-ordered image grid (earlier → later) showing hands and an object.
#
#     Your task is NOT to make a hard decision.
#     Instead, estimate TWO independent confidence scores in [0.0, 1.0]:
#
#     Definitions (per frame):
#     - Contact: {contact_def}
#     - Separation: {sep_def}
#
#     Scoring rules:
#     - Scores are NOT mutually exclusive.
#     - It is allowed that both scores are high (ambiguous transition).
#     - It is allowed that both scores are low (no clear interaction).
#     - Use the entire grid as temporal context.
#
#     Strict Output Format (EXACTLY two lines):
#     Contact: <float between 0.0 and 1.0>
#     Separation: <float between 0.0 and 1.0>
#
#     Do NOT output any explanation.
#     """.strip()
def build_prompts(hand: str = "right"):
    # hand ∈ {"left","right","both","any","min","max","avg"} 你也可以只用前三个
    hand = hand.lower()

    if hand in ["left", "right"]:
        subj = f"the {hand} hand"
        # Contact/Separation 都只针对这一只手
        contact_def = f"Contact means {subj} is in physical contact with the object."
        sep_def = f"Separation means {subj} is NOT in physical contact with the object."
        contact_event = f'Left is Separation and Right is Contact w.r.t {subj}'
        sep_event = f'Left is Contact and Right is Separation w.r.t {subj}'
        who_contact = f"ONLY judge the interaction between {subj} and the object. Ignore the other hand even if it touches."
        sep_rule = f"If the {hand} hand is not touching, it is Separation even if the other hand is touching."



    prompt_contact = f"""
Instruction:
You will be given a time-ordered image grid containing hands and a target object.
Your task is to find the earliest contact moment, and return its grid index.

Focus:
- {who_contact}
- {contact_def}

Rules:
- If contact is already visible in all frames, return the earliest frame (usually the first).
- If no contact is detected within the grid, return -1.

Strict Output Format:
"Frame: X"  (X is the grid cell index, starting from 1)
or
"Frame: -1"
""".strip()

    prompt_separation = f"""
Instruction:
You will be given a time-ordered image grid containing hands and a target object.
Your task is to find the earliest separation moment, and return its grid index.

Focus:
- {who_contact}
- {sep_def}

Rules:
- {sep_rule}
- If no separation is detected within the grid, return -1.

Strict Output Format:
"Frame: X"  (X is the grid cell index, starting from 1)
or
"Frame: -1"
""".strip()

    prompt_state = f"""
    Instruction:
    You are given a time-ordered image grid (earlier → later) showing a hand and a target object.

    Your task is NOT to output a discrete event.
    Instead, estimate TWO independent confidence scores in [0.0, 1.0]:

    Definitions:
    - Contact: {contact_def}
    - Separation: {sep_def}

    Temporal Reasoning Steps:
    1. Ensure the same target object is observed across the grid.
    2. Compare earlier frames with later frames.
    3. Determine whether there is:
       - Increasing evidence of physical touch (toward Contact),
       - Increasing evidence of detachment (toward Separation),
       - Or no significant interaction change.

    Scoring Interpretation:
    - Contact score reflects how strongly the grid suggests a transition toward or presence of Contact.
    - Separation score reflects how strongly the grid suggests a transition toward or presence of Separation.
    - Scores are NOT mutually exclusive.
    - Both scores may be high during ambiguous transitions.
    - Both scores may be low if interaction is unclear or absent.

    Bias Rule (important):
    - If uncertain but the hand appears to approach or align with the object, lean toward higher Contact score.
    - If uncertain but the hand appears to move away or clearly open, lean toward higher Separation score.
    - Avoid assigning high scores to both unless there is clear transitional ambiguity.

    Strict Output Format (EXACTLY two lines):
    Contact: <float between 0.0 and 1.0>
    Separation: <float between 0.0 and 1.0>

    Do NOT output explanation.
    """.strip()

    feedback_prompt_contact = f"""
You will be given ONE image with hands and an object.
Decide whether it is CONTACT under this rule:
- {contact_def}
Answer 1 if yes, else 0. Output only 1 or 0.
""".strip()

    feedback_prompt_separation = f"""
You will be given ONE image with hands and an object.
Decide whether it is SEPARATION under this rule:
- {sep_def}
{sep_rule}
Answer 1 if yes, else 0. Output only 1 or 0.
""".strip()

    feedback_score_prompt_contact = f"""
You will be given ONE image OR a short time-ordered grid around a candidate frame.
Task: Judge whether the CENTER frame should be considered CONTACT under this rule:
- {contact_def}

Output MUST be a single-line JSON (no extra text):
{{"score": 0.0, "label": "contact" or "not_contact", "reason": "short reason"}}

Scoring rubric:
- 1.0 = definitely contact (clear physical touch)
- 0.7~0.9 = likely contact
- 0.4~0.6 = uncertain
- 0.0~0.3 = definitely not contact
""".strip()

    feedback_score_prompt_separation = f"""
You will be given ONE image OR a short time-ordered grid around a candidate frame.
Task: Judge whether the CENTER frame should be considered SEPARATION under this rule:
- {sep_def}
- {sep_rule}

Output MUST be a single-line JSON (no extra text):
{{"score": 0.0, "label": "separation" or "not_separation", "reason": "short reason"}}

Scoring rubric:
- 1.0 = definitely separation (clear gap, no touch)
- 0.7~0.9 = likely separation
- 0.4~0.6 = uncertain
- 0.0~0.3 = definitely not separation
""".strip()

    return (
        prompt_contact,
        prompt_separation,
        prompt_state,
        feedback_prompt_contact,
        feedback_prompt_separation,
        feedback_score_prompt_contact,
        feedback_score_prompt_separation,
    )
def extract_score_info(response: str):
    """
    解析 VLM 输出的 JSON:
    {"score": 0.83, "label": "...", "reason": "..."}
    允许模型偶尔多输出几句：会从文本中抓第一个 {...} 来尝试 json.loads
    返回: (score(float|None), label(str|None), reason(str|None))
    """
    if response is None:
        return None, None, None

    text = str(response).strip()

    # 1) 先尝试直接 loads
    try:
        obj = json.loads(text)
        score = float(obj.get("score", None))
        label = obj.get("label", None)
        reason = obj.get("reason", None)
        return score, label, reason
    except Exception:
        pass

    # 2) 再从文本里抓一个 JSON 子串（最常见：前后有多余解释）
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            score = float(obj.get("score", None))
            label = obj.get("label", None)
            reason = obj.get("reason", None)
            return score, label, reason
        except Exception:
            return None, None, None

    return None, None, None


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
        use_feedback=True,
        hand="right",      # ⭐ 新增
        anchors_cache=None,
        minima_cache=None,  # 多 GPU worker 时传入本地 dict，主进程用 _MINIMA_CACHE_REF
):
    """Process a task to identify the start or end of an action in a video."""
    global _MINIMA_CACHE_REF
    cache = minima_cache if minima_cache is not None else _MINIMA_CACHE_REF
    state_list = []  # 本视频本手每次极小值触发的 state

    # prompt_contact, prompt_separation, prompt_state, fb_contact, fb_separation = build_prompts(hand)
    prompt_contact, prompt_separation, prompt_state, fb_contact, fb_separation, fb_score_contact, fb_score_separation = build_prompts(
        hand)

    # Iterate to narrow down the time
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # 建一个按视频名分组的调试图目录,方便保存图片
    if _OUTPUT_ROOT:
        debug_dir = Path(_OUTPUT_ROOT) / "debug_grids" / video_name
    else:
        debug_dir = Path("/home/EgoLoc/hand_data_drawer/debug_grids") / video_name
    debug_dir.mkdir(parents=True, exist_ok=True)

    json_path = get_json_path(video_name)
    json_folder_path = get_json_folder_path(video_name)

    scalar_data = _load_speed_scalar(str(json_path), hand=hand)
    if not scalar_data:
        print(f"[process_task] {video_name}, hand={hand} 没有有效速度数据")
        return [], state_list

    all_frames = np.array([x[0] for x in scalar_data])
    all_speeds = np.array([x[1] for x in scalar_data])

    if video_type == "short":
        # minima_indices = extract_local_minima_frames(video_name, json_folder_path, hand=hand)
        minima_indices, _ = extract_local_minima_frames_adaptive(
            video_name,
            folder_path=json_folder_path,
            hand=hand,
        )        
        minima_speeds = [
            all_speeds[np.where(all_frames == idx)[0][0]]
            for idx in minima_indices if np.any(all_frames == idx)
        ]
    else:
        minima_indices, minima_speeds = extract_local_minima_frames_adaptive(
            video_name, json_folder_path, hand=hand
        )

    video_key = f"{video_name}.mp4"

    # 去重 + 排序，防止采样过程影响原始 minima
    clean_minima = sorted(set(int(x) for x in minima_indices))

    # 融合锚点：锚点不筛选，全部进入后续 VLM 判断
    if anchors_cache is None:
        anchors_cache = {}
    combined_frames = list(clean_minima)
    anchor_key = f"{video_name}_{hand}"
    anchors_raw = anchors_cache.get(anchor_key, {}).get("merged", [])
    if anchors_raw:
        anchors_in_range = [int(f) for f in anchors_raw if 0 <= int(f) < total_frames]
        combined_frames = sorted(set(anchors_in_range) | set(clean_minima))

    # 为 combined_frames 构造 minima_indices（可 pop）与 minima_speeds（与 while 循环兼容）
    minima_indices = list(combined_frames)
    fallback_speed = float(np.median(all_speeds)) if len(all_speeds) > 0 else 0.0
    minima_speeds = []
    for f in combined_frames:
        idx_in_all = np.where(all_frames == f)[0]
        if len(idx_in_all) > 0:
            minima_speeds.append(float(all_speeds[idx_in_all[0]]))
        else:
            nearest_idx = np.argmin(np.abs(all_frames - f))
            minima_speeds.append(float(all_speeds[nearest_idx]) if len(all_frames) > 0 else fallback_speed)

    # 写入全局缓存（融合后候选帧，含锚点 + 速度极小值）
    if cache is not None:
        cache.setdefault(hand, {})[video_key] = list(combined_frames)

    print(f"{video_name} 融合后候选帧（锚点+极小值）共 {len(combined_frames)} 个: {combined_frames}")
    selected_frame_index = []
    while minima_indices:
        # 速度越小概率越高采样极小值点
        #selected_minima = adaptive_sample_speed(minima_indices, minima_speeds)

        #idx = minima_indices.index(selected_minima)

        # minima_indices.pop(idx)
        # minima_speeds.pop(idx)
        # 直接顺序取第一个极小值
        selected_minima = minima_indices.pop(0)
        minima_speeds.pop(0)
        # 采样关键帧索引
        window =2
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
        #state_indices = list(state_frame_indices)  # 全部帧


        print(f"选取的判断帧为{state_frame_indices[0]}和{state_frame_indices[-1]}")
        print("state_indices:", state_indices)
        print("n_state_frames =", len(state_indices))

        image_state = create_frame_grid_state(
            video_path, state_indices,hand=hand)
        image_RGB = cv2.cvtColor(image_state, cv2.COLOR_BGR2RGB)
        grid_image = Image.fromarray(image_RGB)
        left_idx, right_idx = state_frame_indices[0],state_frame_indices[-1]
        if hand=="right":
            grid_image.save(f"/home/EgoLoc/grid/right2/{video_name}_L{left_idx}_R{right_idx}.png")
        if hand=="left":
            grid_image.save(f"/home/EgoLoc/grid/left2/{video_name}_L{left_idx}_R{right_idx}.png")
        grid_image.save(debug_dir / f"{video_name}_state.png")
        # state = scene_understanding(
        #     credentials, image_state, prompt_state, principle="state")
        state_raw = scene_understanding(
            credentials, image_state, prompt_state, principle="state_score"
        )
        print("[RAW STATE OUTPUT]")
        print(state_raw)
        c_score, s_score = parse_state_scores(state_raw)
        state = decide_state_from_scores(c_score, s_score)
        print(f"[STATE DECISION] c={c_score}, s={s_score} → {state}")
        # ✅ 记录下来（只接受三类）
        if state in ["Contact", "Separation", "Neither","Ambiguous"]:
            state_list.append(state)
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
            print(
                f"[L2.5 GRID LOC] state={state}, grid_frames={frame_indices}, "
                f"vlm_choice={description}"
            )

            if description:
                if description != -1:
                    if int(description) - 1 > len(frame_indices) - 1:
                        print("Warning: Invalid frame index selected")
                        print(f"Selected frame index: {description}")
                    index_specified = max(
                        min(int(description) - 1, len(frame_indices) - 1), 0)
                    final_frame = frame_indices[index_specified]
                    print(
                        f"description choose:{final_frame}"
                    )
                    #=== 反馈机制：对final_frame再用VLM判断状态 ===
                    if use_feedback:
                        # --- 可调参数（默认宽进严出） ---
                        score_thr = 0.65 if state == "Contact" else 0.70
                        feedback_window = 3  # 初始邻域
                        ctx_win = 2  # 上下文窗口：中心±2，共5帧（推荐）
                        tried_frames = set()
                        correct = False
                        feedback_count = 0

                        # 选用 score prompt
                        feedback_prompt = fb_score_contact if state == "Contact" else fb_score_separation

                        while feedback_count < max_feedback:
                            # 1) 构造候选集合（中心±feedback_window，排除试过的）
                            feedback_candidates = [
                                i for i in range(final_frame - feedback_window, final_frame + feedback_window + 1)
                                if 0 <= i < total_frames and i not in tried_frames
                            ]
                            if not feedback_candidates:
                                break

                            # 2) 对每个候选打分，选最高分
                            best = None  # (total_score, score_vlm, score_spd, frame, label, reason)
                            for cand in feedback_candidates:
                                tried_frames.add(cand)

                                # ---- 上下文 1×K grid：让 VLM 看前后帧（强烈推荐）----
                                ctx_frames = _build_mapped_window(cand, ctx_win, all_frames, total_frames)  # 你已有 helper
                                ctx_img = create_frame_grid_with_keyframe(video_path, ctx_frames,
                                                                          1)  # 1行grid（grid_size=1表示单格? 你这个函数是方阵！
                                # ⚠️ 如果 create_frame_grid_with_keyframe 只能做方阵：
                                # 你可以改用 create_frame_grid_state，它支持自动排版：
                                # ctx_img = create_frame_grid_state(video_path, ctx_frames, hand=hand)

                                feedback_result = scene_understanding(
                                    credentials, ctx_img, feedback_prompt, principle="feedback"
                                )
                                score_vlm, label, reason = extract_score_info(feedback_result)

                                if score_vlm is None or not np.isfinite(score_vlm):
                                    continue
                                score_vlm = float(np.clip(score_vlm, 0.0, 1.0))

                                # ---- 速度先验：速度越小越加分（可选但很有用）----
                                if np.any(all_frames == cand):
                                    spd = float(all_speeds[np.where(all_frames == cand)[0][0]])
                                else:
                                    spd = 9999.0
                                score_spd = 1.0 / (spd + 1e-6)  # 你也可以换 exp(-spd/tau)
                                # 归一化到 0~1（粗暴一点就够用）
                                score_spd = float(np.clip(score_spd / (score_spd + 1.0), 0.0, 1.0))

                                total_score = 0.85 * score_vlm + 0.15 * score_spd

                                if (best is None) or (total_score > best[0]):
                                    best = (total_score, score_vlm, score_spd, cand, label, reason)

                            if best is None:
                                feedback_count += 1
                                feedback_window += 2  # 没法评分就扩大范围
                                continue

                            # 3) 采用最高分帧作为新的 final_frame
                            total_score, score_vlm, score_spd, cand, label, reason = best
                            final_frame = cand

                            print(f"[FEEDBACK] state={state} best_frame={cand} "
                                  f"total={total_score:.3f} vlm={score_vlm:.3f} spd={score_spd:.3f} "
                                  f"label={label} reason={reason}")

                            # 4) 严出：过阈值才算 correct
                            if score_vlm >= score_thr:
                                correct = True
                                break

                            # 5) 仍未通过：扩大搜索范围，继续下一轮
                            feedback_count += 1
                            feedback_window += 2

                        if not correct:
                            print("反馈后未找到合适帧（评分未过阈值或超出最大反馈次数），跳过该极小值点")


                    # if use_feedback:
                    #     feedback_window = 3
                    #     tried_frames = set()
                    #     correct = False
                    #     feedback_count = 0
                    #     while feedback_count < max_feedback:
                    #         tried_frames.add(final_frame)
                    #         single_image = create_frame_grid_with_keyframe(video_path, [final_frame], 1)
                    #
                    #         feedback_prompt = fb_contact if state == "Contact" else fb_separation
                    #
                    #
                    #         feedback_result = scene_understanding(credentials, single_image, feedback_prompt, principle="feedback")
                    #         print(f"反馈VLM输出: {feedback_result}")
                    #         if is_positive_feedback(feedback_result):
                    #             correct = True
                    #             break
                    #         # 采样新帧
                    #         feedback_count += 1
                    #         feedback_candidates = [i for i in range(final_frame - feedback_window,
                    #                                                 final_frame + feedback_window + 1)
                    #                                if 0 <= i < total_frames and i not in tried_frames]
                    #         if not feedback_candidates:
                    #             break
                    #         feedback_speeds = [
                    #             all_speeds[np.where(all_frames == i)[0][0]] if np.any(all_frames == i) else 9999 for i
                    #             in feedback_candidates]
                    #         if keyframe_sampling_mode == "adaptive":
                    #             inv_speeds = 1 / (np.array(feedback_speeds) + 1e-8)
                    #             probabilities = inv_speeds / inv_speeds.sum()
                    #             final_frame = np.random.choice(feedback_candidates, p=probabilities)
                    #         else:
                    #             final_frame = np.random.choice(feedback_candidates)
                    #     if not correct:
                    #         print("反馈后未找到合适帧或超出最大反馈次数，但并不跳过该极小值点")

                    # use_feedback为False时，直接采纳final_frame，无需反馈
                    selected_frame_index.append((state, final_frame))
    return selected_frame_index, state_list

def load_json_safe(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def load_anchors_cache(anchors_path):
    """
    加载锚点缓存 JSON。key 为 "video_stem_left" / "video_stem_right"，
    value 为 {"merged": [...], "speed": [...], "visual": [...]}。
    若路径为 None 或文件不存在则返回 {}。
    """
    if anchors_path is None or not os.path.exists(anchors_path):
        return {}
    with open(anchors_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

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


def convert_video(video_file_path: str, action: str, credentials, grid_size: int, video_type="short", max_feedback=1, hand="right", anchors_cache=None, minima_cache=None):
    if anchors_cache is None:
        anchors_cache = {}
    video = cv2.VideoCapture(video_file_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    results, state_list = process_task(
        credentials,
        video_file_path,
        grid_size,
        total_frames,
        max_feedback=max_feedback,
        video_type=video_type,
        hand=hand,
        anchors_cache=anchors_cache,
        minima_cache=minima_cache,
    )
    # print(results)
    video_name = os.path.splitext(os.path.basename(video_file_path))[0]
    json_path = get_json_path(video_name)

    # 再次按 hand 取 (frame, speed)，给 get_contact_separation_pairs 用
    speed_data = _load_speed_scalar(str(json_path), hand=hand)
    # pair = get_contact_separation_pairs(results, speed_data)
    # pair = get_contact_separation_pairs(
    #     results=results,
    #     video_path=video_file_path,
    #     credentials=credentials,
    #     max_contacts_per_query=12,
    #     debug_dir=Path("/home/EgoLoc/debug_referee")  # 可选
    # )
    speed_data = _load_speed_scalar(str(json_path), hand=hand)

    _referee_debug = (Path(_OUTPUT_ROOT) / "debug_referee") if _OUTPUT_ROOT else Path("/home/EgoLoc/debug_referee")
    pair = level3_pairing(
        results=results,
        video_path=video_file_path,
        credentials=credentials,
        speed_data=speed_data,  # NEW
        matching_cfg={"w_speed": 2.0},  # NEW (示例权重)
        max_candidates_per_query=12,
        max_rounds=3,
        debug_dir=_referee_debug,
    )

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
parser.add_argument('--speed_root', type=str, default=None, help='速度 JSON 根目录，设置后覆盖 get_json_path 与 extract_* 的 folder_path')
parser.add_argument('--output_dir', type=str, default=None, help='本次运行输出根目录；未指定时自动生成 output/egoloc_<date>_<time>_<video_type>_grid<N>_<folder>')
parser.add_argument('--video_folder', type=str, default=None, help='视频所在目录（用于列表与输出目录命名）')
parser.add_argument('--anchors_path', type=str, default=None,
                    help='锚点缓存 JSON 路径，例如 .../short/merged_anchors_cache.json；若提供则与速度极小值合并作为候选帧，锚点不筛选全部进入 VLM 判断')
parser.add_argument('--gpus', type=str, default=None,
                    help='多 GPU 并行：逗号分隔的 GPU ID，如 "0,1,2,3"；不传或单卡时使用默认 GPU 0')
parser.add_argument('--worker_tasks', type=str, default=None,
                    help='[内部] Worker 模式：任务 JSON 路径，处理指定任务并写入 worker_output_dir')
parser.add_argument('--worker_output_dir', type=str, default=None,
                    help='[内部] Worker 模式：本进程输出目录')
pargs, unknown = parser.parse_known_args()
credentials = dotenv.dotenv_values(pargs.credentials)
required_keys = ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
if not all(key in credentials for key in required_keys):
    raise ValueError("Required keys are missing in the credentials file")
render_pos = 'topright'  # center or topright
grid_size = int(pargs.grid)
video_folder = pargs.video_folder or "/home/EgoLoc/hand_data_drawer/ego4dvideo"
action = pargs.action
video_type = pargs.video_type
folder_name = action.replace(" ", "_")
output_folder = f"results/{folder_name}"
# os.makedirs(output_folder, exist_ok=True)
if __name__ == "__main__":
    # ---------- Worker 模式：子进程处理指定任务，写入 worker_output_dir ----------
    if getattr(pargs, "worker_tasks", None) and getattr(pargs, "worker_output_dir", None):
        with open(pargs.worker_tasks, "r", encoding="utf-8") as f:
            worker_cfg = json.load(f)
        _SPEED_BASE_DIR = worker_cfg.get("speed_root")
        _OUTPUT_ROOT = pargs.worker_output_dir
        os.makedirs(_OUTPUT_ROOT, exist_ok=True)
        video_folder = worker_cfg["video_folder"]
        credentials = dotenv.dotenv_values(worker_cfg["credentials_path"])
        action = worker_cfg["action"]
        grid_size = int(worker_cfg["grid_size"])
        video_type = worker_cfg.get("video_type", "short")
        anchors_cache = load_anchors_cache(worker_cfg.get("anchors_path")) or {}
        minima_cache = {"left": {}, "right": {}}
        tasks = worker_cfg["tasks"]
        pred_left, pred_right = [], []
        for t in tasks:
            hand, video_file = t["hand"], t["video_file"]
            video_path = os.path.join(video_folder, video_file)
            if not os.path.exists(video_path):
                continue
            pair = convert_video(
                video_path, action, credentials, grid_size,
                video_type=video_type, max_feedback=3, hand=hand,
                anchors_cache=anchors_cache, minima_cache=minima_cache,
            )
            if hand == "left":
                pred_left.append([video_file, pair])
            else:
                pred_right.append([video_file, pair])
        save_predictions(pred_left, os.path.join(_OUTPUT_ROOT, "predictions_left.json"))
        save_predictions(pred_right, os.path.join(_OUTPUT_ROOT, "predictions_right.json"))
        with open(os.path.join(_OUTPUT_ROOT, "minima_left.json"), "w") as f:
            json.dump(minima_cache.get("left", {}), f, indent=2)
        with open(os.path.join(_OUTPUT_ROOT, "minima_right.json"), "w") as f:
            json.dump(minima_cache.get("right", {}), f, indent=2)
        sys.exit(0)

    # ---------- 主进程：设置 GPU 与输出目录 ----------
    gpus_str = getattr(pargs, "gpus", None)
    if gpus_str:
        gpu_list = [x.strip() for x in gpus_str.split(",") if x.strip()]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[0] if len(gpu_list) == 1 else gpus_str
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    if pargs.speed_root is not None:
        _SPEED_BASE_DIR = pargs.speed_root  # 覆盖模块级变量，供 get_json_* 与 extract_* 使用
    # 构建输出根目录：显式指定或按日期+配置自动生成
    if pargs.output_dir:
        _OUTPUT_ROOT = pargs.output_dir
    else:
        basename = os.path.basename(video_folder.rstrip(os.sep))
        _OUTPUT_ROOT = "output/egoloc_{}_{}_grid{}_{}".format(
            datetime.now().strftime("%Y-%m-%d_%H%M"),
            video_type,
            grid_size,
            basename,
        )
    os.makedirs(_OUTPUT_ROOT, exist_ok=True)
    meta = {
        "video_folder": video_folder,
        "speed_root": _SPEED_BASE_DIR,
        "video_type": video_type,
        "grid_size": grid_size,
        "action": action,
        "output_root": _OUTPUT_ROOT,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(_OUTPUT_ROOT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Output root:", _OUTPUT_ROOT)

    minima_cache = {
        "left": {},  # video.mp4 -> [frame, frame, ...]
        "right": {}
    }
    _MINIMA_CACHE_REF = minima_cache
    anchors_cache = load_anchors_cache(pargs.anchors_path) if getattr(pargs, "anchors_path", None) else {}
    if anchors_cache:
        print(f"已加载锚点缓存，共 {len(anchors_cache)} 条（video_stem_hand）")
    state_list = []  # ✅ 记录本视频本手，每次极小值触发的 state
    # 获取文件夹中的所有 MP4 文件并按顺序排序
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    # save_path = "/home/EgoLoc/ManiTIL_prompt/right_grid4.json"
    speed_output_root = "/home/EgoLoc/hand_data_drawer/ego4d_mp4_outputs"
    # save_path = "/home/VLM-Video-Action-Localization-main/VLM-Video-Action-Localization-main/result/greedyVLM_drawer_grid5.json"
    # 排序视频文件，基于文件名中 'c' 后的数字部分（videoN 格式）；其他格式按文件名
    def _video_sort_key(x):
        try:
            return int(x.split('.')[0][5:])
        except (ValueError, IndexError):
            return x
    sorted_video_files = sorted(video_files, key=_video_sort_key)

    # ---------- 多 GPU：分配任务并启动子进程，再合并结果 ----------
    gpu_list = [x.strip() for x in (gpus_str or "").split(",") if x.strip()] if gpus_str else []
    if len(gpu_list) > 1:
        # 构建 (hand, video_file) 任务列表（只含未处理的）
        tasks = []
        for hand in ["left", "right"]:
            save_path = os.path.join(_OUTPUT_ROOT, "predictions_right.json" if hand == "right" else "predictions_left.json")
            existing = load_predictions(save_path)
            processed = {p[0] for p in existing}
            for video_file in sorted_video_files:
                if video_file in processed:
                    continue
                tasks.append({"hand": hand, "video_file": video_file})
        if not tasks:
            print("没有待处理任务（可能已全部完成）。")
        else:
            n_workers = min(len(gpu_list), len(tasks))
            chunk_size = (len(tasks) + n_workers - 1) // n_workers
            chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
            worker_cfg_base = {
                "video_folder": video_folder,
                "speed_root": _SPEED_BASE_DIR,
                "output_root": _OUTPUT_ROOT,
                "credentials_path": pargs.credentials,
                "action": action,
                "grid_size": grid_size,
                "video_type": video_type,
                "anchors_path": pargs.anchors_path,
            }
            procs = []
            for i, chunk in enumerate(chunks[:n_workers]):
                gpu_id = gpu_list[i]
                worker_dir = os.path.join(_OUTPUT_ROOT, "worker_{}".format(i))
                os.makedirs(worker_dir, exist_ok=True)
                task_path = os.path.join(worker_dir, "tasks.json")
                with open(task_path, "w", encoding="utf-8") as f:
                    json.dump({"tasks": chunk, **worker_cfg_base}, f, indent=2)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                script_path = os.path.abspath(__file__)
                p = subprocess.Popen(
                    [sys.executable, script_path, "--worker_tasks", task_path, "--worker_output_dir", worker_dir,
                     "--credentials", pargs.credentials, "--grid", str(grid_size), "--video_type", video_type,
                     "--action", action],
                    env=env,
                    cwd=os.path.dirname(script_path) or ".",
                )
                procs.append((p, worker_dir, chunk))
            for p, wdir, chunk in procs:
                p.wait()
                if p.returncode != 0:
                    print("Worker {} 退出码 {}".format(wdir, p.returncode))
            # 合并 predictions 与 minima（以已有结果为基础，再合并各 worker 输出）
            pl_path = os.path.join(_OUTPUT_ROOT, "predictions_left.json")
            pr_path = os.path.join(_OUTPUT_ROOT, "predictions_right.json")
            merged_left = {p[0]: p[1] for p in (load_predictions(pl_path) if os.path.isfile(pl_path) else [])}
            merged_right = {p[0]: p[1] for p in (load_predictions(pr_path) if os.path.isfile(pr_path) else [])}
            for _, wdir, _ in procs:
                for hand, name in [("left", "predictions_left.json"), ("right", "predictions_right.json")]:
                    p = os.path.join(wdir, name)
                    if os.path.isfile(p):
                        for vid, pairs in load_predictions(p):
                            (merged_left if hand == "left" else merged_right)[vid] = pairs
            save_predictions([[k, v] for k, v in sorted(merged_left.items(), key=lambda x: x[0])], os.path.join(_OUTPUT_ROOT, "predictions_left.json"))
            save_predictions([[k, v] for k, v in sorted(merged_right.items(), key=lambda x: x[0])], os.path.join(_OUTPUT_ROOT, "predictions_right.json"))
            for _, wdir, _ in procs:
                for hand, fname in [("left", "minima_left.json"), ("right", "minima_right.json")]:
                    p = os.path.join(wdir, fname)
                    if os.path.isfile(p):
                        with open(p, "r") as f:
                            data = json.load(f)
                        minima_cache[hand].update(data)
            with open(os.path.join(_OUTPUT_ROOT, "minima_left.json"), "w") as f:
                json.dump(minima_cache["left"], f, indent=2)
            with open(os.path.join(_OUTPUT_ROOT, "minima_right.json"), "w") as f:
                json.dump(minima_cache["right"], f, indent=2)
            print("多 GPU 合并完成。")
        # 多 GPU 分支结束后仍执行评估（与单 GPU 一致）
    else:
        # ---------- 单 GPU：原有顺序循环 ----------
        for hand in ["left", "right"]:
            print(f"\n====== 处理 {hand} 手 ======\n")

            # 每只手一份结果文件（自适应输出时使用可读文件名）
            if _OUTPUT_ROOT:
                save_path = os.path.join(_OUTPUT_ROOT, "predictions_right.json" if hand == "right" else "predictions_left.json")
            else:
                save_path = "/home/EgoLoc/ManiTIL_prompt/r16.json" if hand == "right" else "/home/EgoLoc/ManiTIL_prompt/l16.json"

            all_predictions = load_predictions(save_path)
            processed_video_files = {prediction[0] for prediction in all_predictions}

            for video_file in sorted_video_files:
                if video_file in processed_video_files:
                    continue

                video_path = os.path.join(video_folder, video_file)
                if not os.path.exists(video_path):
                    continue

                list_of_pair = []
                for i in range(1):
                    pair = convert_video(
                        video_path,
                        action,
                        credentials,
                        grid_size,
                        video_type=video_type,
                        max_feedback=3,
                        hand=hand,  # ⭐ 关键：这一轮是 hand
                        anchors_cache=anchors_cache,
                    )
                    list_of_pair.append(pair)

                averaged_pairs = calculate_max_mode_average(list_of_pair)
                if _OUTPUT_ROOT:
                    minima_left_path = os.path.join(_OUTPUT_ROOT, "minima_left.json")
                    minima_right_path = os.path.join(_OUTPUT_ROOT, "minima_right.json")
                else:
                    minima_left_path = "/home/EgoLoc/ManiTIL_prompt/minima_left_16.json"
                    minima_right_path = "/home/EgoLoc/ManiTIL_prompt/minima_right_16.json"
                with open(minima_left_path, "w") as f:
                    json.dump(minima_cache["left"], f, indent=2)
                with open(minima_right_path, "w") as f:
                    json.dump(minima_cache["right"], f, indent=2)
                if len(averaged_pairs) == 0:
                    print(f"{video_file} ({hand}) can't predict")
                    all_predictions.append([video_file, []])
                else:
                    print(f"{video_file} ({hand}) pairs: {averaged_pairs}")
                    all_predictions.append([video_file, averaged_pairs])

                save_predictions(all_predictions, save_path)

    # 每只手各自评估一次（单 GPU 与多 GPU 合并后均执行）
    if video_type == "short":
        for hand in ["left", "right"]:
            save_path = os.path.join(_OUTPUT_ROOT, "predictions_right.json" if hand == "right" else "predictions_left.json") if _OUTPUT_ROOT else ("/home/EgoLoc/ManiTIL_prompt/r16.json" if hand == "right" else "/home/EgoLoc/ManiTIL_prompt/l16.json")
            result = evaluate_predictions(
                json_path=save_path,
                gt_excel_path="/home/EgoLoc/ground_truth/KitchenCounter1.xlsx",
                sheet_name="Sheet9"
            )
            print(f"\nEvaluation ({hand}):", result)

    elif video_type == "long":
        gt_json = "/data/EgoLoc/EgoDex/long/long.json"

        if _OUTPUT_ROOT:
            pred_left = os.path.join(_OUTPUT_ROOT, "predictions_left.json")
            pred_right = os.path.join(_OUTPUT_ROOT, "predictions_right.json")
            minima_left_path = os.path.join(_OUTPUT_ROOT, "minima_left.json")
            minima_right_path = os.path.join(_OUTPUT_ROOT, "minima_right.json")
        else:
            pred_left = "/home/EgoLoc/ManiTIL_prompt/l16.json"
            pred_right = "/home/EgoLoc/ManiTIL_prompt/r16.json"
            minima_left_path = "/home/EgoLoc/ManiTIL_prompt/minima_left_16.json"
            minima_right_path = "/home/EgoLoc/ManiTIL_prompt/minima_right_16.json"

        # GT 左右手都为空则该视频不纳入计算
        gts_left = load_ground_truth_json_twohands(gt_json, hand="left")
        gts_right = load_ground_truth_json_twohands(gt_json, hand="right")
        all_gt_videos = set(gts_left.keys()) | set(gts_right.keys())
        videos_to_exclude = {
            v for v in all_gt_videos
            if not gts_left.get(v, {}).get("pairs", []) and not gts_right.get(v, {}).get("pairs", [])
        }

        results_left = evaluate_all_stages(
            pred_json_path=pred_left,
            gt_json_path=gt_json,
            hand="left",
            minima_json_path=minima_left_path,
            videos_to_exclude=videos_to_exclude,
        )

        results_right = evaluate_all_stages(
            pred_json_path=pred_right,
            gt_json_path=gt_json,
            hand="right",
            minima_json_path=minima_right_path,
            videos_to_exclude=videos_to_exclude,
        )

        print("\nEvaluation (left):")
        for k, v in results_left.items():
            if isinstance(v, (int, float)) and v is not None:
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        print("\nEvaluation (right):")
        for k, v in results_right.items():
            if isinstance(v, (int, float)) and v is not None:
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        # ---- 左右手一起评估（按接触对合并，非简单 (L+R)/2）----
        results_both_pooled = evaluate_all_stages_pooled_twohands(
            pred_left_path=pred_left,
            pred_right_path=pred_right,
            gt_json_path=gt_json,
            sr_tolerances=(1, 3, 5),
            psr_tolerance=10,
        )
        keys_stage3 = ["SR@1", "SR@3", "SR@5", "PSR", "mae", "MoF", "IoU"]
        print("\nEvaluation (both-pooled):  # 左右手全部 (video,hand) 样本一起算 stage3 再平均，按接触对数自然加权")
        print("hand:", results_both_pooled["hand"])
        for k in keys_stage3:
            v = results_both_pooled.get("stage3", {}).get(k)
            print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: None")


