#!/bin/bash

cd /project
python nachetmodel/ModelEvaluator.py \
    --model_path models/15spp_zoom_level_validation_models/1_seed_model_20250127 \
    --test_data_path data/processed/15spp_zoom_level_validation_models/1-seed/test \
    --output_path models/15spp_zoom_level_validation_models/1_seed_model_20250127 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 20000 \
>> models/15spp_zoom_level_validation_models/1_seed_model_20250127/evaluate.log 2>&1