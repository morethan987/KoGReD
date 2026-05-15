#!/bin/bash
# 不同度数的实体的性能测试启动脚本
# 使用方法: bash scripts/test_entity_degree.sh

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python loss_restraint_KGE_model/run.py \
    --restore \
    --name fb15k_train_20260514_000228 \
    --mode test_entity_degree \
    --save loss_restraint_KGE_model/output/fb15k-237n/20260514_000228
