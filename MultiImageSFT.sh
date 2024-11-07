#!/bin/bash
# CUDA_VISIBLE_DEVICES="4,5,6,7" bash MultiImageSFT.sh

export WANDB_API_KEY=e343c6f2df76e0cab534608ebffa94798062a270
export NCCL_ALGO=Tree

# pip install -U transformers accelerate
# pip install --upgrade Pillow
# pip install git+https://github.com/Dao-AILab/causal-conv1d

# need change 4 place
experiment_name=MultiImageSFT_241101
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

## For MultiNode
# ssh check
# apt-get install pdsh
# chown root:root /usr/lib/x86_64-linux-gnu/pdsh
# chown root:root /usr/lib
# chmod 755 /usr/lib/x86_64-linux-gnu/pdsh
# chmod 755 /usr/lib
# export NCCL_DEBUG=DEBUG
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

deepspeed --master_port 0 \
    --include localhost:6,7 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./ckpts/10SFT2dSenseLong176K \
    --version jamba \
    --data_path ./data/convert_mul_pic_241101.json \
    --vision_tower ./models/clip_vit_large_patch14_336 \
    --mm_projector_type mlp2x_gelu \
    --resamplePooling 2d \
    --group_by_modality_length True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./ckpts/${experiment_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 40960 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb > ${log_folder}/${log_name} 2>&1 &

