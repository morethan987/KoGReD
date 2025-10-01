#!/bin/bash
# 微调启动脚本
# 使用方法: cdko && acko && bash scripts/run_embedding.sh

# 路径设置
MODEL_PATH='/home/ma-user/work/model/Qwen3-embedding-8B'
DATA_PATH='data/FB15K-237N'
OUTPUT_DIR='data/FB15K-237N'
KGE_MODEL='data/FB15K-237N-rotate.pth'
LOG_DIR='logs'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/get_embedding_${TIME_STAMP}.log"


# 创建目录及文件
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

# 显示NPU信息
echo "NPU信息:"
if command -v npu-smi &> /dev/null; then
    npu-smi info
else
    echo "npu-smi 命令不可用"
fi
# 让用户确认是否继续
read -p "Continue? (Y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" && "$CONFIRM" != "" ]]; then
    echo "Canceled!"
    exit 0
fi

nohup python data/run_embedding.py \
    --dataset $DATA_PATH \
    --embedding_dir $MODEL_PATH \
    --device npu \
    --batch_size 16 \
    --output_dir $OUTPUT_DIR \
    >> $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
{
    echo "========================================================="
    echo "LoRA微调进程已启动, PID: $PID    日志文件: $LOG_FILE"
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止进程: kill $PID"
    echo "========================================================="
} | tee -a "$LOG_FILE"
