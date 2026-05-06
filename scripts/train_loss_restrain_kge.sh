#!/bin/bash
# Loss Restrain KGE 训练启动脚本
# 使用方法:
#   bash scripts/train_loss_restrain_kge.sh                                # 默认: CoDEx-S
#   bash scripts/train_loss_restrain_kge.sh CoDEx-S                        # 指定数据集
#   bash scripts/train_loss_restrain_kge.sh FB15k-237N
#

export CUDA_VISIBLE_DEVICES=1

# ==================== 参数解析 ====================
DATASET="${1:-CoDEx-S}"

case "$DATASET" in
    CoDEx-S) ;;
    FB15k-237N) ;;
    *)
        echo "错误: 不支持的数据集 '$DATASET'"
        echo "支持: CoDEx-S, FB15k-237N"
        exit 1
        ;;
esac

# ==================== 数据集相关路径 ====================
case "$DATASET" in
    CoDEx-S)
        NAME='codex_train'
        OUTPUT_BASE='loss_restraint_KGE_model/output/codex-s'
        AUX_TRIPLES="MCTS/output/codex-s-llm/20260503_114800/discovered_triplets.txt"
        AUX_CONFIDENCE="MCTS/output/codex-s-llm/20260503_114800/auxiliary_triples_confidence_kgbert.json"
        ;;
    FB15k-237N)
        NAME='fb15k_train'
        OUTPUT_BASE='loss_restraint_KGE_model/output/fb15k-237n'
        AUX_TRIPLES="MCTS/output/fb15k-237n-kge/20260425_215000/discovered_triplets.txt"
        AUX_CONFIDENCE="MCTS/output/fb15k-237n-kge/20260425_215000/auxiliary_triples_confidence_kgbert.json"
        ;;
esac

LOG_DIR='loss_restraint_KGE_model/logs'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIME_STAMP}"
LOG_FILE="$LOG_DIR/${DATASET}_${TIME_STAMP}.log"


# ==================== 创建目录 ====================
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ==================== 预检查 ====================
if [ ! -f "$AUX_TRIPLES" ]; then
    echo "错误: 未找到辅助三元组文件: $AUX_TRIPLES"
    echo "请先运行 MCTS 生成辅助三元组"
    exit 1
fi

if [ ! -f "$AUX_CONFIDENCE" ]; then
    echo "错误: 未找到置信度文件: $AUX_CONFIDENCE"
    echo "请先运行 MCTS 生成置信度文件"
    exit 1
fi

# ==================== 显示配置 ====================
echo "=== Loss Restrain KGE Configuration ==="
echo "数据集:          $DATASET"
echo "输出目录:        $OUTPUT_DIR"
echo "辅助三元组:      $AUX_TRIPLES"
echo "置信度文件:      $AUX_CONFIDENCE"
echo "========================================="
echo ""
echo "=== 输出文件位置 ==="
echo "日志文件:        $LOG_FILE"
echo "模型输出:        $OUTPUT_DIR"
echo "=========================="

if command -v npu-smi &> /dev/null; then
    npu-smi info
else
    nvidia-smi
fi
read -p "Continue? (Y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" && "$CONFIRM" != "" ]]; then
    echo "Canceled!"
    exit 0
fi

# ==================== 启动训练 ====================
nohup python loss_restraint_KGE_model/run.py \
    --name "$NAME" \
    --mode train \
    --keep_aux True \
    --time_string "$TIME_STAMP" \
    --data "$DATASET" \
    --save "$OUTPUT_DIR" \
    --score_func conve \
    --opn corr \
    --adapt_aggr 1 \
    --loss_delta 0.05 \
    --batch 256 \
    --lr 5e-4 \
    --epoch 10000 \
    --aux_triples "$AUX_TRIPLES" \
    --aux_confidence "$AUX_CONFIDENCE" \
    >> "$LOG_FILE" 2>&1 &

PID=$!
{
    echo "========================================================="
    echo "loss_restrain_kge训练进程已启动, PID: $PID"
    echo "数据集: $DATASET    日志文件: $LOG_FILE"
    echo "输出目录: $OUTPUT_DIR"
    echo "辅助三元组: $AUX_TRIPLES"
    echo "置信度文件: $AUX_CONFIDENCE"
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止进程: kill $PID"
    echo "备注：超参数实验：0.5 0.05"
    echo "========================================================="
} | tee -a "$LOG_FILE"
