#!/bin/bash
TAG=$(date +%Y%m%d%H)
docker exec -it pytorch-gpu-24 /bin/bash -c "/project/notebooks/4.01_js_checkpoint_export.sh"
docker compose -f 27spp_model_1.gpu.compose.yaml build --progress=plain --no-cache
docker tag local/ai-cfia/nachet-model-ccds/gpu-classifier-27spp-model-1 local/ai-cfia/nachet-model-ccds/gpu-classifier-27spp-model-1:$TAG
docker tag local/ai-cfia/nachet-model-ccds/gpu-classifier-27spp-model-1:$TAG ghcr.io/ai-cfia/nachet-model-ccds/gpu-classifier-27spp-model-1:$TAG
docker push ghcr.io/ai-cfia/nachet-model-ccds/gpu-classifier-27spp-model-1:$TAG
