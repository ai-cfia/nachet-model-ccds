#!/bin/bash

cd /project
python nachetmodel/HFTrainer.py \
    --train_dir data/processed/15spp_zoom_level_validation_models/1-seed/ \
    --output_dir models/15spp_zoom_level_validation_models/1_seed_model_20250128 \
    --do_train True \
    --ignore_mismatched_sizes True \
    --num_train_epochs 400.0 \
    --dataloader_num_workers 8 \
    --do_eval True \
    --do_predict True \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 4 \
    --model_name_or_path "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft" \
    --warmup_ratio 0.1 \
    --overwrite_output_dir True \
    --learning_rate 0.00002 \
    --overwrite_output_dir True \
>> models/15spp_zoom_level_validation_models/1_seed_model_20250128/train_log.txt

# --model_name_or_path "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft" \

# start=500
# end=4000
# step=500

# Loop over checkpoints
# for ((i=start; i<=end; i+=step))
# do
#    echo "Running for checkpoint-$i <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
   
#     python HF_ValSet_evaluation.py \
#         --train_dir data/split_22Seeds/Training \
#         --output_dir  models/22Species-swinv2-large-patch4-192-6Seed-dataAug2 \
#         --do_train False \
#         --validation_dir  data/split_22Seeds/Testing \
#         --ignore_mismatched_sizes True \
#         --dataloader_num_workers 12 \
#         --per_device_train_batch_size 16 \
#         --do_eval True \
#         --do_predict True \
#         --model_name_or_path  models/22Species-swinv2-large-patch4-192-6Seed-dataAug2/checkpoint-$i 
# done 

# python HFTrainer.py
# --train_dir ./processed_images/6seed/trai
# --output_dir N_interactive_Run4_150E_SwinV2_large/ 
# --do_train True  
# --ignore_mismatched_sizes True
# --do_eval True
# --num_train_epochs 150.0
# --model_name_or_path microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft  
# --per_device_train_batch_size 32 
# --overwrite_output_dir True

