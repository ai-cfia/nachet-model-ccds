#!/bin/bash

python3 nachetmodel/CheckpointExporter.py \
  --checkpoint_path models/27spp_model/model_120250130/checkpoint-9000 \
  --model_name 27spp_model_1 \
  --version 1.0

python3 nachetmodel/CheckpointExporter.py \
  --checkpoint_path models/15spp_zoom_level_validation_models/6_seed_model_120250130/checkpoint-20500 \
  --model_name 15spp_model_1 \
  --version 1.0

\cp -rf models/27spp_model/model_120250130/checkpoint-9000/27spp_model_1.mar environments/torchserve/gpu/artifacts/
\cp -rf models/15spp_zoom_level_validation_models/6_seed_model_120250130/checkpoint-20500/15spp_model_1.mar environments/torchserve/gpu/artifacts/

# python3 nachetmodel/CheckpointExporter.py \
#   --checkpoint_path models/27spp_model/model_120250130/checkpoint-9000-ensemble \
#   --model_name 27spp_model_e_1 \
#   --version 1.0

# python3 nachetmodel/CheckpointExporter.py \
#   --checkpoint_path models/15spp_zoom_level_validation_models/6_seed_model_120250130/checkpoint-20500-ensemble \
#   --model_name 15spp_model_e_1 \
#   --version 1.0

# python3 nachetmodel/CheckpointExporter.py \
#   --checkpoint_path models/27spp_model/model_120250130/checkpoint-9000-ensemble \
#   --model_name 27spp_model_e_1 \
#   --version 1.0

# torch-workflow-archiver -f \
#   --workflow-name ensemble_27spp \
#   --spec-file environments/torchserve/workflow/27spp_ensemble/workflow.yaml \
#   --handler environments/torchserve/workflow/27spp_ensemble/workflow_handler.py \
#   --export-path environments/torchserve/workflow/27spp_ensemble

# \cp -rf environments/torchserve/workflow/27spp_ensemble/ensemble_27spp.war environments/torchserve/gpu/artifacts/
# \cp -rf models/27spp_model/model_120250130/checkpoint-9000-ensemble/27spp_model_e_1.mar environments/torchserve/gpu/artifacts/
# \cp -rf models/15spp_zoom_level_validation_models/6_seed_model_120250130/checkpoint-20500-ensemble/15spp_model_e_1.mar environments/torchserve/gpu/artifacts/
