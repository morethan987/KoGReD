#!/bin/bash
# mcts预处理启动脚本
# 使用方法: cdko && ackopa && bash scripts/preprocess_mcts.sh
# for fb15k-237n, threshold=9e-5
# for codex-s, threshold=5e-4

python MCTS/preprocess.py \
    --data_folder data/CoDEx-S \
    --output_path MCTS/output/codex-s \
    --threshold 5e-4
