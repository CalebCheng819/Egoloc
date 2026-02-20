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
from egoloc_speed_twohands import extract_3d_speed_and_visualize  # 新封装的生成速度文件的函数
from egoloc_speed_twohands import batch_process_videos  # 对文件夹内的所有视频执行extract_3d_speed_and_visualize
from typing import List, Optional   # ← 新增这一行
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

_model = load_model(
    "/home/EgoLoc/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "/home/EgoLoc/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"  # 直接放在weights目录外
)
def load_predictions(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        with open(file_path, "w") as f:
            json.dump([], f)
        return []

if __name__ == "__main__":
    # 获取文件夹中的所有 MP4 文件并按顺序排序
    video_folder = "/home/EgoLoc/hand_data_drawer/ego4dvideo"
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    save_path = "/home/EgoLoc/ManiTIL_prompt/right_grid4.json"
    speed_output_root = "/home/EgoLoc/hand_data_drawer/EgoDex_10"
    #123分别为15，20，25
    #5为25，6为15，7也为15但加上了reg 质量信息，8为在7基础上加入了左右手修正，9修复了roi的问题，10修正会出现双手空缺的问题，11增加了对2d手腕的坐标，12修改roi为20
    #13增加对depth的调查,14增加对深度的网格处理最大限制0.2，15改成0.1,16修改成为三点速度,17使用roi深度
    # save_path = "/home/VLM-Video-Action-Localization-main/VLM-Video-Action-Localization-main/result/greedyVLM_drawer_grid5.json"
    # 排序视频文件，基于文件名中 'c' 后的数字部分
    #数据结果记录
    #mp4_outputs_sorted,
    #2为添加了hammer修改逻辑
    #最终结果全部放到3里面
    #4为将第一版结果处理之后得到的
    #重新生成了ego4d
    #用于生成速度EgoDex,short and long
    sorted_video_files = sorted(video_files, key=lambda x: int(x.split('.')[0][5:]))
    all_predictions = load_predictions(save_path)
    batch_process_videos(video_folder, speed_output_root, device="cuda", encoder="vits")  # 后续添加对已有文件的跳过