#!/bin/bash
# KGE模型训练启动脚本
# 使用方法: cdko && acko && bash scripts/train_kge.sh

# NPU环境配置
export NPU_VISIBLE_DEVICES=0

# --- 路径配置 ---
DATA_PATH='data/FB15K-237N'
OUTPUT_DIR='LLM_Discriminator/train_kge_output'
LOG_DIR='LLM_Discriminator/logs'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_kge_RotatE_${TIME_STAMP}.log"

# --- 超参数配置 ---
# [来源: 您的论文图片]
MODEL_NAME="RotatE"
DIMENSION=512         # embedding dimension d_e = d_r = 512
NEGATIVE_SAMPLES=32   # sample K = 32 negative samples

# [来源: 您的论文图片, 从调优集合中选择一个典型值开始]
MARGIN=6.0            # margin γ is tuned among {0, 4, 6, 8, 12}. 6.0是一个很好的起始值.

# [来源: OpenKE官方示例, 作为合理的基线设置]
BATCH_SIZE=2048       # KGE训练通常使用较大的batch size以提升效率和稳定性
LEARNING_RATE=2e-5    # 一个在KGE任务中常见且有效的学习率
OPTIMIZER="Adam"      # Adam是KGE训练中最常用的优化器
EPOCHS=3000           # 训练轮数, 可根据收敛情况调整

# --- 脚本执行 ---

# 检查路径是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo "错误: 数据路径不存在: $DATA_PATH"
    exit 1
fi

# 创建日志和输出目录
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# 显示配置信息
echo -e "\n=== KGE模型训练配置信息 ==="
echo "模型: $MODEL_NAME"
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo "---------------------------------"
echo "维度 (Dimension): $DIMENSION"
echo "批次大小 (Batch Size): $BATCH_SIZE"
echo "学习率 (Learning Rate): $LEARNING_RATE"
echo "边际值 (Margin γ): $MARGIN"
echo "优化器 (Optimizer): $OPTIMIZER"
echo "负采样数 (Negative Samples K): $NEGATIVE_SAMPLES"
echo "训练轮数 (Epochs): $EPOCHS"
echo "==================================="

# 用户确认
read -p "是否继续训练? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "训练已取消。"
    exit 0
fi

# 启动训练脚本
nohup python LLM_Discriminator/train_kge.py \
    --model "$MODEL_NAME" \
    --in_path "$DATA_PATH" \
    --out_path "$OUTPUT_DIR" \
    --time_stamp "$TIME_STAMP" \
    --dimension "$DIMENSION" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --margin "$MARGIN" \
    --optimizer "$OPTIMIZER" \
    --neg_ent "$NEGATIVE_SAMPLES" \
    --epochs "$EPOCHS" \
    --test_triple \
    >> "$LOG_FILE" 2>&1 &

# 获取进程ID并提示用户
PID=$!
{
    echo "========================================================="
    echo "KGE训练进程已启动!"
    echo "进程ID (PID): $PID"
    echo "日志文件: $LOG_FILE"
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止进程: kill $PID"
    echo "========================================================="
} | tee -a "$LOG_FILE"
