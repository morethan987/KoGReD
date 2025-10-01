export WANDB_DISABLED=true
wandb offline
CUDA_VISIBLE_DEVICES=0 nohup python finetune_kopa.py \
    --base_model 'wxjiao/alpaca-7b' \
    --data_path 'data/CoDeX-S-train.json' \
    --output_dir 'output/alpaca7b_CoDeX-S' \
    --num_epochs 3 \
    --lora_r 64 \
    --learning_rate 3e-4 \
    --batch_size 12 \
    --micro_batch_size 12 \
    --num_prefix 1 \
    --kge_model 'data/CoDeX-S-rotate.pth' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' > log_alpaca7b_CoDeX-S.txt &