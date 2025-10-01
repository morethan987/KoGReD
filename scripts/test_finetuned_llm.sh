# cdkopa && acko && nohup python inference_kopa.py > test_fb.log 2>&1 &
#!/bin/bash
# 微调启动脚本
# 使用方法: cdko && acko && bash scripts/run_finetune_fb15k237n.sh

# 路径设置
MODEL_PATH='/home/ma-user/work/model/Alpaca-7B'
DATA_PATH='data/FB15K-237N-test.json'
OUTPUT_DIR='output/alpaka_7b_fb'
KGE_MODEL='data/FB15K-237N-rotate.pth'
LOG_DIR='logs'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)


# wandb 设置
export WANDB_DISABLED=true
wandb offline
wandb disabled

# 设置 NPU 环境变量
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503
NPROC=$(( $(echo "$NPU_VISIBLE_DEVICES" | tr -cd ',' | wc -c) + 1 ))


# 创建目录及文件
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR
LOG_FILE="$LOG_DIR/finetune_${TIME_STAMP}.log"

# 设置 tokenizers 并行化环境变量，避免警告
export TOKENIZERS_PARALLELISM=false

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

nohup torchrun \
    --nnodes=1 \
    --nproc_per_node=$NPROC \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    finetune_kopa.py \
    --base_model $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_epochs 3 \
    --lora_r 64 \
    --learning_rate 3e-4 \
    --batch_size 8 \
    --micro_batch_size 8 \
    --num_prefix 1 \
    --kge_model $KGE_MODEL \
    --lora_target_modules '["q_proj","k_proj","v_proj","o_proj"]' \
    --llm_dim 4096 \
    >> $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
{
    echo "========================================================="
    echo "LoRA微调进程已启动, PID: $PID    日志文件: $LOG_FILE"
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止进程: ps aux | grep finetune_kopa.py | grep -v grep | awk '{print \$2}' | xargs kill -9"
    echo "========================================================="
} | tee -a "$LOG_FILE"
