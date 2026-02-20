#!/bin/bash
# 删除 EgoDex_short 目录下的所有点云文件

DIR="/home/chengjuntao/data0/EgoLoc/hand_data_drawer/EgoDex_short"

if [ ! -d "$DIR" ]; then
    echo "错误: 目录不存在: $DIR"
    exit 1
fi

# 查找所有点云文件
echo "正在扫描点云文件..."
FILES=$(find "$DIR" -type f \( -name "*.ply" -o -name "*.pcd" -o -name "*.xyz" -o -name "*.pts" -o -name "*.las" -o -name "*.laz" \))

COUNT=$(echo "$FILES" | grep -c .)
if [ "$COUNT" -eq 0 ]; then
    echo "未找到任何点云文件"
    exit 0
fi

echo "找到 $COUNT 个点云文件"

# 显示前10个文件示例
echo ""
echo "文件示例（前10个）:"
echo "$FILES" | head -10 | while read f; do
    echo "  $f"
done

# 确认删除
if [ "$1" != "--yes" ] && [ "$1" != "-y" ]; then
    echo ""
    read -p "确认删除这 $COUNT 个文件? (yes/no): " confirm
    if [ "$confirm" != "yes" ] && [ "$confirm" != "y" ]; then
        echo "已取消"
        exit 1
    fi
fi

# 执行删除
echo ""
echo "开始删除..."
DELETED=0
FAILED=0

while IFS= read -r f; do
    if [ -n "$f" ]; then
        if rm -f "$f" 2>/dev/null; then
            DELETED=$((DELETED + 1))
            if [ $((DELETED % 100)) -eq 0 ]; then
                echo "进度: 已删除 $DELETED 个文件..."
            fi
        else
            FAILED=$((FAILED + 1))
            echo "错误: 无法删除 $f" >&2
        fi
    fi
done <<< "$FILES"

echo ""
echo "完成!"
echo "成功删除: $DELETED 个文件"
if [ "$FAILED" -gt 0 ]; then
    echo "失败: $FAILED 个文件"
fi
