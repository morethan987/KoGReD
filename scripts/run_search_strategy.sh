#!/bin/bash
# 搜索策略消融实验启动脚本
# 使用方法:
#   bash scripts/run_search_strategy.sh                                # 默认: CoDEx-S + llm + mcts
#   bash scripts/run_search_strategy.sh CoDEx-S kgbert random          # 指定数据集 + 判别器 + 策略
#   bash scripts/run_search_strategy.sh CoDEx-S kgbert greedy
#   bash scripts/run_search_strategy.sh CoDEx-S kgbert beam
#   bash scripts/run_search_strategy.sh CoDEx-S kgbert mcts
#   bash scripts/run_search_strategy.sh CoDEx-S llm random
#
# 策略类型: random, greedy, beam, mcts
# 判别器类型: llm, kgbert, kge, random

# ==================== 参数解析 ====================
DATASET="${1:-CoDEx-S}"
DISCRIMINATOR="${2:-llm}"
STRATEGY="${3:-mcts}"

case "$DATASET" in
    CoDEx-S) ;;
    FB15k-237N) ;;
    *)
        echo "错误: 不支持的数据集 '$DATASET'"
        echo "支持: CoDEx-S, FB15k-237N"
        exit 1
        ;;
esac

case "$DISCRIMINATOR" in
    llm|kgbert|kge|random) ;;
    *)
        echo "错误: 不支持的判别器 '$DISCRIMINATOR'"
        echo "支持: llm, kgbert, kge, random"
        exit 1
        ;;
esac

case "$STRATEGY" in
    random|greedy|beam|mcts) ;;
    *)
        echo "错误: 不支持的策略 '$STRATEGY'"
        echo "支持: random, greedy, beam, mcts"
        exit 1
        ;;
esac

# ==================== 数据集相关路径 ====================
DATA_PATH="data/${DATASET}"
ENTITY2EMBEDDING_PATH="$DATA_PATH/entity2embedding.pth"

case "$DATASET" in
    CoDEx-S)
        THRESHOLD='5e-4'
        KGE_MODEL='LLM_Discriminator/data/CoDeX-S-rotate.pth'
        LLM_PATH='/data/yitingting/models/alpaca-7b'
        LORA_PATH='LLM_Discriminator/output/alpaka_7b_codex'
        KGBERT_MODEL_DIR='kg-bert/output/triple_classifier_CoDEx-S'
        KGBERT_DATA_DIR='kg-bert/data/CoDEx-S'
        KGE_DISCRIMINATOR_PATH='LLM_Discriminator/data/CoDeX-S-rotate.pth'
        TARGET_DEPTH=4
        ;;
    FB15k-237N)
        THRESHOLD='9e-5'
        KGE_MODEL='LLM_Discriminator/data/FB15k-237N-rotate.pth'
        LLM_PATH='/data/yitingting/models/alpaca-7b'
        LORA_PATH='LLM_Discriminator/output/alpaca7b_fb'
        KGBERT_MODEL_DIR='kg-bert/output/triple_classifier_FB15k-237N'
        KGBERT_DATA_DIR='kg-bert/data/FB15k-237N'
        KGE_DISCRIMINATOR_PATH='LLM_Discriminator/data/FB15k-237N-rotate.pth'
        TARGET_DEPTH=4
        ;;
esac

EMBEDDING_PATH="${LORA_PATH}/embeddings.pth"

# ==================== 通用配置 ====================
DISCRIMINATOR_FOLDER="$PWD/LLM_Discriminator"
TIME_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="MCTS/output/strategy_ablation/${DATASET,,}-${DISCRIMINATOR}-${STRATEGY}/${TIME_STAMP}"
PROCESSED_DATA="MCTS/output/processed_data_${DATASET,,}.pth"
LOG_DIR='MCTS/logs'
LOG_FILE="$LOG_DIR/strategy_${STRATEGY}_${DATASET,,}_${DISCRIMINATOR}_${TIME_STAMP}.log"

export CUDA_VISIBLE_DEVICES=0,1,2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503
export TOKENIZERS_PARALLELISM=false
NPROC=$(( $(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c) + 1 ))

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR/checkpoints"

# ==================== 数据预处理 ====================
PREPROCESS_OUTPUT_DIR="MCTS/output"
if [ ! -f "$PROCESSED_DATA" ]; then
    echo "未找到预处理数据: $PROCESSED_DATA"
    echo "正在运行数据预处理 (dataset=$DATASET, threshold=$THRESHOLD)..."
    mkdir -p "$PREPROCESS_OUTPUT_DIR"
    python MCTS/preprocess.py \
        --data_folder "$DATA_PATH" \
        --output_file "$PROCESSED_DATA" \
        --threshold "$THRESHOLD"
    if [ $? -ne 0 ]; then
        echo "错误: 数据预处理失败"
        exit 1
    fi
    echo "数据预处理完成"
fi

# ==================== 预检查 ====================
case "$DISCRIMINATOR" in
    llm)
        if [ ! -d "$LORA_PATH" ]; then
            echo "错误: 未找到 LoRA 权重: $LORA_PATH"
            exit 1
        fi
        ;;
    kgbert)
        if [ ! -f "${KGBERT_MODEL_DIR}/pytorch_model.bin" ]; then
            echo "错误: 未找到 KG-BERT 模型: ${KGBERT_MODEL_DIR}/pytorch_model.bin"
            echo "请先运行: bash scripts/train_kgbert_triple_classifier.sh $DATASET"
            exit 1
        fi
        ;;
    kge)
        if [ ! -f "$KGE_DISCRIMINATOR_PATH" ]; then
            echo "错误: 未找到 KGE 模型: $KGE_DISCRIMINATOR_PATH"
            echo "请先运行: bash scripts/train_kge.sh $DATASET"
            exit 1
        fi
        ;;
esac

# ==================== 显示配置 ====================
echo "=== Search Strategy Ablation ==="
echo "数据集:          $DATASET"
echo "搜索策略:        $STRATEGY"
echo "判别器类型:      $DISCRIMINATOR"
echo "输出目录:        $OUTPUT_DIR"
case "$DISCRIMINATOR" in
    llm)     echo "LLM:             $LLM_PATH" ;;
    kgbert)  echo "KG-BERT模型:     $KGBERT_MODEL_DIR" ;;
    kge)     echo "KGE模型:         $KGE_DISCRIMINATOR_PATH" ;;
esac
echo "=========================="
echo ""
echo "=== 输出文件位置 ==="
echo "discovered_triplets.txt: $OUTPUT_DIR/discovered_triplets.txt"
echo "日志文件:               $LOG_FILE"
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

# ==================== 构建判别器相关参数 ====================
DISC_ARGS="--discriminator_type $DISCRIMINATOR"
case "$DISCRIMINATOR" in
    llm)
        DISC_ARGS="$DISC_ARGS --llm_path $LLM_PATH --lora_path $LORA_PATH --embedding_path $EMBEDDING_PATH"
        ;;
    kgbert)
        DISC_ARGS="$DISC_ARGS --kgbert_model_dir $KGBERT_MODEL_DIR --kgbert_data_dir $KGBERT_DATA_DIR"
        ;;
    kge)
        DISC_ARGS="$DISC_ARGS --kge_discriminator_path $KGE_DISCRIMINATOR_PATH"
        ;;
esac

# ==================== 启动 ====================
nohup torchrun \
    --nnodes=1 \
    --nproc_per_node=$NPROC \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    MCTS/strategy_runner.py \
    --strategy $STRATEGY \
    --data_folder "$DATA_PATH" \
    --processed_data "$PROCESSED_DATA" \
    --output_folder "$OUTPUT_DIR" \
    --entity2embedding_path "$ENTITY2EMBEDDING_PATH" \
    --kge_path "$KGE_MODEL" \
    --discriminator_folder "$DISCRIMINATOR_FOLDER" \
    --root_dir "$PWD" \
    --dtype fp16 \
    --exploration_weight 1.0 \
    --leaf_threshold 32 \
    --mcts_iterations 10 \
    --budget_per_entity 200 \
    --beam_width 3 \
    --checkpoint_interval 1 \
    --target_depth $TARGET_DEPTH \
    $DISC_ARGS \
    >> "$LOG_FILE" 2>&1 &

PID=$!
{
    echo "========================================================="
    echo "Strategy=$STRATEGY ($DISCRIMINATOR) 进程已启动, PID: $PID"
    echo "数据集: $DATASET    日志文件: $LOG_FILE"
    echo "输出目录: $OUTPUT_DIR"
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止进程: kill $PID"
    echo "备注：llm判别器+随机搜索策略"
    echo "========================================================="
} | tee -a "$LOG_FILE"
