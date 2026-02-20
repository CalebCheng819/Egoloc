#!/usr/bin/env python3
"""
将 Moge2 输出的深度数据转换为 egoloc_speed.py 期望的格式

用法：
    python convert_moge2_depth.py \
        --input_dir output/depth_moge2 \
        --output_base_dir output_egoloc \
        --video_name video1

或者批量转换：
    python convert_moge2_depth.py \
        --input_dir output/depth_moge2 \
        --output_base_dir output_egoloc \
        --batch
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Optional
import cv2

# 尝试多种方式导入 EXR 读取库（按优先级排序）

# 方法1: pyexr（纯 Python，推荐）
HAS_PYEXR = False
try:
    import pyexr
    HAS_PYEXR = True
except ImportError:
    pass

# 方法2: OpenEXR（需要系统依赖）
HAS_EXR = False
OpenEXR = None
Imath = None
try:
    import OpenEXR
    import Imath
    HAS_EXR = True
except ImportError:
    try:
        import openexr as OpenEXR
        import Imath
        HAS_EXR = True
    except ImportError:
        pass

# 方法3: imageio（可能内部使用 OpenCV，不推荐）
HAS_IMAGEIO = False
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    pass

if not HAS_PYEXR and not HAS_EXR and not HAS_IMAGEIO:
    print("=" * 60)
    print("警告: 未找到可用的 EXR 读取库！")
    print("=" * 60)
    print("推荐安装 pyexr（纯 Python，最简单）：")
    print("  pip install pyexr")
    print("")
    print("或者安装 OpenEXR（需要系统依赖）：")
    print("  # Ubuntu/Debian:")
    print("  sudo apt-get install libopenexr-dev")
    print("  pip install OpenEXR")
    print("=" * 60)


def read_exr_depth(exr_path: Path) -> Optional[np.ndarray]:
    """
    读取 .exr 格式的深度图
    
    Returns:
        (H, W) float32 array，单位：米，或 None 如果读取失败
    """
    exr_path = Path(exr_path)
    if not exr_path.exists():
        return None
    
    # 方法1: 使用 pyexr（纯 Python，推荐）
    if HAS_PYEXR:
        try:
            # pyexr 可以直接读取 EXR 文件
            depth = pyexr.read(str(exr_path))
            # pyexr 可能返回多通道，取第一个通道
            if len(depth.shape) == 3:
                # 如果是 RGB，通常深度在 R 通道（索引 0）
                depth = depth[:, :, 0]
            elif len(depth.shape) == 2:
                # 已经是单通道
                pass
            else:
                print(f"  未知的深度图形状: {depth.shape}")
                return None
            return depth.astype(np.float32)
        except Exception as e:
            print(f"  pyexr 读取失败 {exr_path.name}: {e}")
            # 继续尝试其他方法
    
    # 方法2: 使用 OpenEXR 库（如果可用）
    if HAS_EXR:
        try:
            exr_file = OpenEXR.InputFile(str(exr_path))
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            # 读取 R 通道（深度图通常是单通道，存储在 R 通道）
            depth_str = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
            depth = np.frombuffer(depth_str, dtype=np.float32)
            depth = depth.reshape((height, width))
            return depth.astype(np.float32)
        except Exception as e:
            print(f"  OpenEXR 读取失败 {exr_path.name}: {e}")
            # 继续尝试其他方法
    
    # 方法3: 使用 imageio（可能内部使用 OpenCV，不推荐）
    if HAS_IMAGEIO:
        try:
            # imageio 可以直接读取 EXR 文件
            depth = imageio.imread(str(exr_path))
            # imageio 可能返回多通道，取第一个通道或转换为灰度
            if len(depth.shape) == 3:
                # 如果是 RGB，通常深度在 R 通道
                depth = depth[:, :, 0]
            elif len(depth.shape) == 2:
                # 已经是单通道
                pass
            else:
                print(f"  未知的深度图形状: {depth.shape}")
                return None
            return depth.astype(np.float32)
        except Exception as e:
            # imageio 可能内部使用 OpenCV，静默失败
            pass
    
    # 方法4: 使用 OpenCV（通常不支持 EXR，但尝试一下）
    try:
        depth = cv2.imread(str(exr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if depth is not None:
            # OpenCV 可能读取为多通道，取第一个通道
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
            return depth.astype(np.float32)
    except Exception as e:
        pass  # OpenCV 通常不支持 EXR，静默失败
    
    # 所有方法都失败了
    return None


def convert_video_depth(
    input_video_dir: Path,
    output_depth_dir: Path,
    video_name: Optional[str] = None
) -> int:
    """
    转换单个视频的深度数据
    
    Args:
        input_video_dir: Moge2 输出的视频目录（如 output/depth_moge2/video1/）
        output_depth_dir: 输出目录（如 output_egoloc/video1/depth/）
        video_name: 视频名称（用于日志）
    
    Returns:
        成功转换的帧数
    """
    input_video_dir = Path(input_video_dir)
    output_depth_dir = Path(output_depth_dir)
    output_depth_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 .exr 文件
    exr_files = sorted(input_video_dir.glob("frame_*_depth.exr"))
    
    if not exr_files:
        print(f"  未找到 .exr 文件在 {input_video_dir}")
        return 0
    
    converted = 0
    for exr_file in exr_files:
        # 从文件名提取帧索引：frame_000000_depth.exr -> 0
        try:
            frame_idx_str = exr_file.stem.replace("frame_", "").replace("_depth", "")
            frame_idx = int(frame_idx_str)
        except ValueError:
            print(f"  无法解析帧索引: {exr_file.name}")
            continue
        
        # 读取深度图
        depth = read_exr_depth(exr_file)
        if depth is None:
            print(f"  读取失败: {exr_file.name}")
            continue
        
        # 保存为 .npy 格式
        output_file = output_depth_dir / f"pred_depth_{frame_idx:06d}.npy"
        np.save(output_file, depth.astype(np.float32))
        converted += 1
    
    if video_name:
        print(f"  [{video_name}] 转换了 {converted}/{len(exr_files)} 帧")
    else:
        print(f"  转换了 {converted}/{len(exr_files)} 帧")
    
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="将 Moge2 深度数据转换为 egoloc_speed.py 格式"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Moge2 输出的根目录（如 output/depth_moge2）"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="输出根目录（每个视频会创建子目录，如 output_egoloc）"
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
        help="单个视频名称（如 video1），如果不指定则批量处理所有视频"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量处理所有视频（忽略 --video_name）"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_base_dir = Path(args.output_base_dir)
    
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    if args.batch or args.video_name is None:
        # 批量处理：遍历所有视频子目录
        video_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        if not video_dirs:
            print(f"未找到视频目录在 {input_dir}")
            return
        
        print(f"找到 {len(video_dirs)} 个视频，开始批量转换...\n")
        total_converted = 0
        
        for video_dir in sorted(video_dirs):
            video_name = video_dir.name
            output_depth_dir = output_base_dir / video_name / "depth"
            converted = convert_video_depth(video_dir, output_depth_dir, video_name)
            total_converted += converted
        
        print(f"\n总共转换了 {total_converted} 帧")
    else:
        # 单个视频处理
        video_dir = input_dir / args.video_name
        if not video_dir.exists():
            print(f"错误: 视频目录不存在: {video_dir}")
            return
        
        output_depth_dir = output_base_dir / args.video_name / "depth"
        convert_video_depth(video_dir, output_depth_dir, args.video_name)


if __name__ == "__main__":
    main()

