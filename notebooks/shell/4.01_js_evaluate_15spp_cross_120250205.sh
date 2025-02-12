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
    --chkstart 5500 \
    --chkend 5500 \
    --test_name "_zoom_2_data_1"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}2-seed/test \
    --output_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 5500 \
    --chkend 5500 \
    --test_name "_zoom_2_data_2"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}6-seed/test \
    --output_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 5500 \
    --chkend 5500 \
    --test_name "_zoom_2_data_6"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}combine/test \
    --output_path {$BASE_MODEL_DIR}2_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 5500 \
    --chkend 5500 \
    --test_name "_zoom_2_data_combine"

python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}1-seed/test \
    --output_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 19500 \
    --chkend 19500 \
    --test_name "_zoom_1_data_1"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}2-seed/test \
    --output_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 19500 \
    --chkend 19500 \
    --test_name "_zoom_1_data_2"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}6-seed/test \
    --output_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 19500 \
    --chkend 19500 \
    --test_name "_zoom_1_data_6"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}combine/test \
    --output_path {$BASE_MODEL_DIR}1_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 19500 \
    --chkend 19500 \
    --test_name "_zoom_1_data_combine"

python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}1-seed/test \
    --output_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 20500 \
    --chkend 20500 \
    --test_name "_zoom_6_data_1"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}2-seed/test \
    --output_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 20500 \
    --chkend 20500 \
    --test_name "_zoom_6_data_2"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}6-seed/test \
    --output_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 20500 \
    --chkend 20500 \
    --test_name "_zoom_6_data_6"
python nachetmodel/ModelEvaluator.py \
    --model_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --test_data_path {$BASE_DATA_DIR}combine/test \
    --output_path {$BASE_MODEL_DIR}6_seed_model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 20500 \
    --chkend 20500 \
    --test_name "_zoom_6_data_combine"

python nachetmodel/ModelEvaluator.py \
    --model_path models/27spp_model/model_120250130 \
    --test_data_path {$BASE_DATA_DIR}1-seed/test \
    --output_path models/27spp_model/model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 10000 \
    --chkend 10000 \
    --test_name "_27spp_1_data_1"
python nachetmodel/ModelEvaluator.py \
    --model_path models/27spp_model/model_120250130 \
    --test_data_path {$BASE_DATA_DIR}2-seed/test \
    --output_path models/27spp_model/model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 10000 \
    --chkend 10000 \
    --test_name "_27spp_1_data_2"
python nachetmodel/ModelEvaluator.py \
    --model_path models/27spp_model/model_120250130 \
    --test_data_path {$BASE_DATA_DIR}6-seed/test \
    --output_path models/27spp_model/model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 10000 \
    --chkend 10000 \
    --test_name "_27spp_1_data_6"
python nachetmodel/ModelEvaluator.py \
    --model_path models/27spp_model/model_120250130 \
    --test_data_path {$BASE_DATA_DIR}combine/test \
    --output_path models/27spp_model/model_120250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 10000 \
    --chkend 10000 \
    --test_name "_27spp_1_data_combine"

python nachetmodel/ModelEvaluator.py \
    --model_path models/27spp_model/model_220250130 \
    --test_data_path {$BASE_DATA_DIR}1-seed/test \
    --output_path models/27spp_model/model_220250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 7500 \
    --chkend 7500 \
    --test_name "_27spp_2_data_1"
python nachetmodel/ModelEvaluator.py \
    --model_path models/27spp_model/model_220250130 \
    --test_data_path {$BASE_DATA_DIR}2-seed/test \
    --output_path models/27spp_model/model_220250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 7500 \
    --chkend 7500 \
    --test_name "_27spp_2_data_2"
python nachetmodel/ModelEvaluator.py \
    --model_path models/27spp_model/model_220250130 \
    --test_data_path {$BASE_DATA_DIR}6-seed/test \
    --output_path models/27spp_model/model_220250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 7500 \
    --chkend 7500 \
    --test_name "_27spp_2_data_6"
python nachetmodel/ModelEvaluator.py \
    --model_path models/27spp_model/model_220250130 \
    --test_data_path {$BASE_DATA_DIR}combine/test \
    --output_path models/27spp_model/model_220250130 \
    --batch_size 32 \
    --parent "true" \
    --chkstart 7500 \
    --chkend 7500 \
    --test_name "_27spp_2_data_combine"
    