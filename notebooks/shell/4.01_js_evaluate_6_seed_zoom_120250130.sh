#!/bin/bash

cd /project
python nachetmodel/ModelEvaluator.py \
    --model_path models/15spp_zoom_level_validation_models/6_seed_model_120250130 \
    --test_data_path data/processed/15spp_zoom_level_validation_models/6-seed/test \
    --output_path models/15spp_zoom_level_validation_models/6_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
>> models/15spp_zoom_level_validation_models/6_seed_model_120250130/evaluate.log 2>&1
