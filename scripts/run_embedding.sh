#!/bin/bash
# 微调启动脚本
# 使用方法: cdko && acko && bash scripts/run_embedding.sh

export HF_HUB_OFFLINE=1

# 路径设置
MODEL_PATH='all-MiniLM-L6-v2'
DATA_PATH='data/CoDEx-S'
OUTPUT_DIR=${DATA_PATH}
LOG_DIR='data/logs'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/get_embedding_${TIME_STAMP}.log"


# 创建目录及文件
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

# 显示NPU信息
echo "Device status:"
if command -v npu-smi &> /dev/null; then
    npu-smi info
else
    nvidia-smi
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
    --device cuda:2 \
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
