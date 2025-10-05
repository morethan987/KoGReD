#!/bin/bash
# loss_restrain_kge训练启动脚本
# 使用方法: cdko && ackopa && bash scripts/train_loss_restrain_kge.sh

# 路径设置
# DATA_SET='FB15K-237N'
DATA_SET='CoDEx-S'
# OUTPUT_DIR='loss_restraint_KGE_model/output/fb15k-237n'
OUTPUT_DIR='loss_restraint_KGE_model/output/codex-s'
LOG_DIR='loss_restraint_KGE_model/logs'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)
# LOG_FILE="$LOG_DIR/fb15k_${TIME_STAMP}.log"
LOG_FILE="$LOG_DIR/codex_${TIME_STAMP}.log"

# 创建目录及文件
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

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

nohup python loss_restraint_KGE_model/run.py \
    --name test_run \
    --time_string $TIME_STAMP \
    --data $DATA_SET \
    --save $OUTPUT_DIR \
    --score_func conve \
    --opn corr \
    --adapt_aggr 1 \
    --loss_delta 0.002 \
    --batch 256 \
    --lr 5e-4 \
    --epoch 500 \
    --gpu 2 \
    >> $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
{
    echo "========================================================="
    echo "loss_restrain_kge训练进程已启动, PID: $PID    日志文件: $LOG_FILE"
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止进程: kill $PID"
    echo "========================================================="
} | tee -a "$LOG_FILE"
