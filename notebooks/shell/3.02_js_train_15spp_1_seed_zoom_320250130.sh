#!/bin/bash

cd /project
python nachetmodel/HFTrainer_js.py \
    --train_dir data/processed/15spp_zoom_level_validation_models/1-seed/ \
    --output_dir models/15spp_zoom_level_validation_models/1_seed_model_320250130 \
    --do_train True \
    --ignore_mismatched_sizes True \
    --num_train_epochs 400.0 \
    --dataloader_num_workers 4 \
    --do_eval True \
    --do_predict True \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 32 \
    --model_name_or_path "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft" \
    --warmup_ratio 0.1 \
    --overwrite_output_dir True \
    # --learning_rate 0.00002 \
    --overwrite_output_dir True \
    --seed 7613 \
>> models/15spp_zoom_level_validation_models/1_seed_model_320250130/train_log.txt
