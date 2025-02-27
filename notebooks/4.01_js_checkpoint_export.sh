#!/bin/bash

python3 nachetmodel/CheckpointExporter.py \
  --checkpoint_path models/27spp_model/model_120250130/checkpoint-9000 \
  --model_name 27spp_model_1 \
  --version 1.0

python3 nachetmodel/CheckpointExporter.py \
  --checkpoint_path models/15spp_zoom_level_validation_models/6_seed_model_120250130/checkpoint-20500 \
  --model_name 15spp_model_1 \
  --version 1.0
