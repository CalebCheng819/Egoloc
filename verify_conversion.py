#!/usr/bin/env python3
"""
验证 MoGe2 深度转换结果是否正确

用法：
    python verify_conversion.py --output_base_dir output_egoloc
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def verify_video_depth(output_depth_dir: Path, video_name: str) -> Dict:
    """
    验证单个视频的深度转换结果
    
    Returns:
        包含验证结果的字典
    """
    result = {
        "video_name": video_name,
        "valid": True,
        "errors": [],
        "warnings": [],
        "file_count": 0,
        "expected_count": 0,
        "depth_stats": {}
    }
    
    output_depth_dir = Path(output_depth_dir)
    
    if not output_depth_dir.exists():
        result["valid"] = False
        result["errors"].append(f"目录不存在: {output_depth_dir}")
        return result
    
    # 查找所有深度文件
    depth_files = sorted(output_depth_dir.glob("pred_depth_*.npy"))
    result["file_count"] = len(depth_files)
    
    if result["file_count"] == 0:
        result["valid"] = False
        result["errors"].append("未找到任何深度文件")
        return result
    
    # 检查文件命名格式
    expected_indices = set()
    for depth_file in depth_files:
        try:
            # 从文件名提取索引：pred_depth_000000.npy -> 0
            idx_str = depth_file.stem.split("_")[-1]
            idx = int(idx_str)
            expected_indices.add(idx)
        except ValueError:
            result["errors"].append(f"文件名格式错误: {depth_file.name}")
            result["valid"] = False
    
    result["expected_count"] = len(expected_indices)
    
    # 检查是否有缺失的帧（检查前10个文件）
    if len(expected_indices) > 0:
        max_idx = max(expected_indices)
        missing = []
        for i in range(min(10, max_idx + 1)):
            if i not in expected_indices:
                missing.append(i)
        if missing:
            result["warnings"].append(f"前10帧中有缺失: {missing}")
    
    # 检查几个文件的数据格式
    sample_files = depth_files[:min(5, len(depth_files))]
    all_stats = []
    
    for depth_file in sample_files:
        try:
            depth = np.load(depth_file)
            
            # 检查数据类型
            if depth.dtype != np.float32:
                result["warnings"].append(
                    f"{depth_file.name}: dtype={depth.dtype}, 期望 float32"
                )
            
            # 检查形状
            if len(depth.shape) != 2:
                result["errors"].append(
                    f"{depth_file.name}: shape={depth.shape}, 期望 (H, W)"
                )
                result["valid"] = False
                continue
            
            # 检查深度值范围（metric depth 应该在合理范围内）
            valid_mask = np.isfinite(depth) & (depth > 0)
            if valid_mask.sum() == 0:
                result["errors"].append(
                    f"{depth_file.name}: 没有有效的深度值"
                )
                result["valid"] = False
                continue
            
            valid_depth = depth[valid_mask]
            stats = {
                "min": float(np.min(valid_depth)),
                "max": float(np.max(valid_depth)),
                "mean": float(np.mean(valid_depth)),
                "median": float(np.median(valid_depth)),
                "shape": depth.shape
            }
            all_stats.append(stats)
            
            # 检查深度值是否在合理范围（0.01m 到 10m）
            if stats["min"] < 0.01 or stats["max"] > 10.0:
                result["warnings"].append(
                    f"{depth_file.name}: 深度值范围异常 "
                    f"(min={stats['min']:.3f}m, max={stats['max']:.3f}m)"
                )
            
        except Exception as e:
            result["errors"].append(f"{depth_file.name}: 读取失败 - {e}")
            result["valid"] = False
    
    # 计算平均统计信息
    if all_stats:
        result["depth_stats"] = {
            "avg_min": np.mean([s["min"] for s in all_stats]),
            "avg_max": np.mean([s["max"] for s in all_stats]),
            "avg_mean": np.mean([s["mean"] for s in all_stats]),
            "avg_median": np.mean([s["median"] for s in all_stats]),
            "shape": all_stats[0]["shape"] if all_stats else None
        }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="验证 MoGe2 深度转换结果"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="输出根目录（如 output_egoloc）"
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
        help="单个视频名称（如 video1），如果不指定则检查所有视频"
    )
    
    args = parser.parse_args()
    
    output_base_dir = Path(args.output_base_dir)
    
    if not output_base_dir.exists():
        print(f"错误: 输出目录不存在: {output_base_dir}")
        return
    
    if args.video_name:
        # 检查单个视频
        video_dirs = [output_base_dir / args.video_name]
    else:
        # 检查所有视频
        video_dirs = [d for d in output_base_dir.iterdir() if d.is_dir()]
    
    if not video_dirs:
        print(f"未找到视频目录在 {output_base_dir}")
        return
    
    print(f"开始验证 {len(video_dirs)} 个视频的转换结果...\n")
    print("=" * 80)
    
    all_valid = True
    total_files = 0
    
    for video_dir in sorted(video_dirs):
        video_name = video_dir.name
        depth_dir = video_dir / "depth"
        
        result = verify_video_depth(depth_dir, video_name)
        
        # 打印结果
        status = "✓" if result["valid"] else "✗"
        print(f"{status} [{video_name}]")
        print(f"  文件数: {result['file_count']}")
        
        if result["depth_stats"]:
            stats = result["depth_stats"]
            print(f"  深度范围: {stats['avg_min']:.3f}m - {stats['avg_max']:.3f}m")
            print(f"  平均深度: {stats['avg_mean']:.3f}m")
            print(f"  图像尺寸: {stats['shape']}")
        
        if result["errors"]:
            print(f"  错误 ({len(result['errors'])}):")
            for err in result["errors"][:3]:  # 只显示前3个错误
                print(f"    - {err}")
            if len(result["errors"]) > 3:
                print(f"    ... 还有 {len(result['errors']) - 3} 个错误")
        
        if result["warnings"]:
            print(f"  警告 ({len(result['warnings'])}):")
            for warn in result["warnings"][:2]:  # 只显示前2个警告
                print(f"    - {warn}")
            if len(result["warnings"]) > 2:
                print(f"    ... 还有 {len(result['warnings']) - 2} 个警告")
        
        if not result["valid"]:
            all_valid = False
        
        total_files += result["file_count"]
        print()
    
    print("=" * 80)
    print(f"\n验证完成:")
    print(f"  总视频数: {len(video_dirs)}")
    print(f"  总文件数: {total_files}")
    print(f"  状态: {'全部通过 ✓' if all_valid else '有错误 ✗'}")


if __name__ == "__main__":
    main()

