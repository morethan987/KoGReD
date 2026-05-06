#!/bin/bash
# 抑制效果可视化脚本
# 使用方法:
#   bash scripts/visualize_suppression.sh                                     # 默认: CoDEx-S 最新运行
#   bash scripts/visualize_suppression.sh CoDEx-S
#   bash scripts/visualize_suppression.sh CoDEx-S 20260503_155855            # 指定时间戳

DATASET="${1:-CoDEx-S}"

case "$DATASET" in
    CoDEx-S)  METRICS_BASE='loss_restraint_KGE_model/output/codex-s' ;;
    FB15k-237N) METRICS_BASE='loss_restraint_KGE_model/output/fb15k-237n' ;;
    *)
        echo "错误: 不支持的数据集 '$DATASET'"
        echo "支持: CoDEx-S, FB15k-237N"
        exit 1
        ;;
esac

if [ -n "$2" ]; then
    TIMESTAMP="$2"
else
    TIMESTAMP=$(ls -1t "$METRICS_BASE" 2>/dev/null | head -1)
    if [ -z "$TIMESTAMP" ]; then
        echo "错误: 未找到任何输出目录: $METRICS_BASE"
        exit 1
    fi
fi

INPUT="$METRICS_BASE/$TIMESTAMP/visualization_metrics/suppression_metrics.json"
OUTPUT="$METRICS_BASE/$TIMESTAMP/visualization_results"

if [ ! -f "$INPUT" ]; then
    echo "错误: 未找到指标文件: $INPUT"
    exit 1
fi

echo "=== 可视化抑制效果 ==="
echo "数据集:    $DATASET"
echo "时间戳:    $TIMESTAMP"
echo "指标文件:  $INPUT"
echo "输出目录:  $OUTPUT"
echo "======================"

mkdir -p "$OUTPUT"

python loss_restraint_KGE_model/visualize_suppression.py --input "$INPUT" --output "$OUTPUT"
