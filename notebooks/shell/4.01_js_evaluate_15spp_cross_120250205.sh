#!/bin/bash

BASE_DATA_DIR="data/processed/15spp_zoom_level_validation_models/"
BASE_MODEL_DIR="models/15spp_zoom_level_validation_models/"

cd /project
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}1-seed/test \
    --output_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --test_name "zoom_2_data_1"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}2-seed/test \
    --output_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --test_name "zoom_2_data_2"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}6-seed/test \
    --output_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --test_name "zoom_2_data_6"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}combine/test \
    --output_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --test_name "zoom_2_data_combine"

python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}1-seed/test \
    --output_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --test_name "zoom_1_data_1"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}2-seed/test \
    --output_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --test_name "zoom_1_data_2"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}6-seed/test \
    --output_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --test_name "zoom_1_data_6"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}combine/test \
    --output_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --test_name "zoom_1_data_combine"

python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}1-seed/test \
    --output_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --test_name "zoom_6_data_1"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}2-seed/test \
    --output_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --test_name "zoom_6_data_2"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}6-seed/test \
    --output_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --test_name "zoom_6_data_6"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}combine/test \
    --output_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 500 \
    --chkend 40000 \
    --test_name "zoom_6_data_combine"
    