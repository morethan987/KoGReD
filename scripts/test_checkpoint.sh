#!/bin/bash
# 使用已保存的 checkpoint 进行测试
# 使用方法:
# bash scripts/test_checkpoint.sh
# 默认: test_entity_degree
# 修改 --mode 参数即可切换测试模式:
# --mode test_entity_degree    按实体度数区间评估
# --mode test_relation_type    按关系类型评估
# --mode overall               整体测试
# --mode case_study            案例分析
# --mode inference_time        推理时间基准测试

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python loss_restraint_KGE_model/run.py \
    --restore \
    --name codex_train_20260503_155855 \
    --mode inference_time \
    --save loss_restraint_KGE_model/output/codex-s/20260503_155855
