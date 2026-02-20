from pathlib import Path
import shutil

ROOT = Path("/home/EgoLoc/results2")          # 根目录
OUT  = Path("speed_jsons")       # 输出目录
PATTERN = "*_with_speed.json"    # 速度 json 命名规则

OUT.mkdir(exist_ok=True)

count = 0

for video_dir in sorted(ROOT.glob("video*")):
    if not video_dir.is_dir():
        continue

    for json_file in video_dir.glob(PATTERN):
        dst = OUT / f"{video_dir.name}__{json_file.name}"

        # 防止极端情况下重名
        if dst.exists():
            i = 1
            while True:
                candidate = OUT / f"{video_dir.name}__{json_file.stem}_{i}{json_file.suffix}"
                if not candidate.exists():
                    dst = candidate
                    break
                i += 1

        shutil.copy2(json_file, dst)
        print(f"[OK] {json_file} -> {dst}")
        count += 1

print(f"\n[DONE] 共提取 {count} 个 speed json")
