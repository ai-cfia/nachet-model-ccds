#!/bin/bash
TRAIN_DIR="data/processed/27spp_model/6-seed/"
OUTPUT_DIR="models/27spp_model/model_320250130/"
MODEL_PATH="microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"
ACCUM=2
BATCH_SIZE=16

cd /project
python3 nachetmodel/HFTrainer_js.py \
    --train_dir $TRAIN_DIR \
    --output_dir $OUTPUT_DIR \
    --do_train True \
    --ignore_mismatched_sizes True \
    --num_train_epochs 400.0 \
    --dataloader_num_workers 4 \
    --do_eval True \
    --do_predict True \
    --eval_strategy "steps" \
    --gradient_accumulation_steps $ACCUM \
    --per_device_train_batch_size $BATCH_SIZE \
    --model_name_or_path $MODEL_PATH \
    --warmup_ratio 0.1 \
    --overwrite_output_dir True \
    # --learning_rate 0.00002 \
    # --report_to "mlflow" \
    # --resume_from_checkpoint "models/27spp_model/model_120250130/checkpoint-1000" \
    --logging_first_step True \
    --save_strategy "steps" \
    --eval_on_start True \
    --seed 7613 \
    --data_seed 7613
