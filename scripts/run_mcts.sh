#!/bin/bash
# MCTS启动脚本
# 使用方法: cdko && ackopa && bash scripts/run_mcts.sh

# 路径设置
# DATA_PATH='data/FB15K-237N'
DATA_PATH='data/CoDEx-S'
MODEL_PATH='wxjiao/alpaca-7b'
# OUTPUT_DIR='MCTS/output/fb15k-237n'
OUTPUT_DIR='MCTS/output/codex-s'
PROCESSED_DATA="$OUTPUT_DIR/processed_data.pth"
# LORA_PATH="LLM_Discriminator/output/alpaca7b_fb"
LORA_PATH="LLM_Discriminator/output/alpaca7b_CoDeX-S"
EMBEDDING_PATH="$LORA_PATH/embeddings.pth"
ENTITY2EMBEDDING_PATH="$DATA_PATH/entity2embedding.pth"
# KGE_MODEL='LLM_Discriminator/data/FB15K-237N-rotate.pth'
KGE_MODEL='LLM_Discriminator/data/CoDeX-S-rotate.pth'
DISCRIMINATOR_FOLDER="$PWD/LLM_Discriminator"
LOG_DIR='MCTS/logs'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)
# LOG_FILE="$LOG_DIR/fb15k_${TIME_STAMP}.log"
LOG_FILE="$LOG_DIR/codex_${TIME_STAMP}.log"


# 设置 NPU 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2
export WORLD_SIZE=3
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503
NPROC=$(( $(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c) + 1 ))


# 创建目录及文件
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR
mkdir -p "$OUTPUT_DIR/checkpoints"

# 设置 tokenizers 并行化环境变量，避免警告
export TOKENIZERS_PARALLELISM=false

# 显示NPU信息
echo "NPU信息:"
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
    MCTS/run_mcts.py \
    --data_folder $DATA_PATH \
    --processed_data $PROCESSED_DATA \
    --output_folder $OUTPUT_DIR \
    --llm_path $MODEL_PATH \
    --lora_path $LORA_PATH \
    --embedding_path $EMBEDDING_PATH \
    --entity2embedding_path $ENTITY2EMBEDDING_PATH \
    --kge_path $KGE_MODEL \
    --discriminator_folder $DISCRIMINATOR_FOLDER \
    --root_dir $PWD \
    --dtype fp16 \
    --exploration_weight 1.0 \
    --leaf_threshold 32 \
    --mcts_iterations 10 \
    --budget_per_entity 200 \
    --checkpoint_interval 1 \
    >> $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
{
    echo "========================================================="
    echo "MCTS进程已启动, PID: $PID    日志文件: $LOG_FILE"
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止进程: kill $PID"
    echo "========================================================="
} | tee -a "$LOG_FILE"
