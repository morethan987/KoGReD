#!/bin/bash
# 微调后的LLM的启动脚本
# 使用方法: cdko && ackopa && bash scripts/test_finetuned_llm.sh

# 路径设置
MODEL_PATH='wxjiao/alpaca-7b'
DATA_PATH='LLM_Discriminator/data/FB15K-237N-test.json'
LORA_PATH="LLM_Discriminator/output/alpaca7b_fb"
LOG_DIR='LLM_Discriminator/logs'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/test_finetuned_llm_${TIME_STAMP}.log"

# 设置 NPU 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2
export WORLD_SIZE=3
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503
NPROC=$(( $(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c) + 1 ))


# 创建目录及文件
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

# 显示NPU信息
echo "Show device info:"
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

nohup torchrun \
    --nnodes=1 \
    --nproc_per_node=$NPROC \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    LLM_Discriminator/test_finetuned_llm.py \
    --base_model $MODEL_PATH \
    --test_data $DATA_PATH \
    --lora_weights $LORA_PATH \
    >> $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
{
    echo "========================================================="
    echo "测试微调进程已启动, PID: $PID    日志文件: $LOG_FILE"
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止进程: kill $PID"
    echo "========================================================="
} | tee -a "$LOG_FILE"
