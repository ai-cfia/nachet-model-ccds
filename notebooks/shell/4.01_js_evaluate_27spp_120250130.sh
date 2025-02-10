#!/bin/bash

cd /project
python nachetmodel/ModelEvaluator.py \
    --model_path models/27spp_model/model_120250130 \
    --test_data_path data/processed/27spp_model/6-seed/test \
    --output_path models/27spp_model/model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --figsize 15 \
>> models/27spp_model/model_120250130/evaluate.log 2>&1
