# Commands
```bash
cd environments/torchserve
git submodule add https://github.com/pytorch/serve.git
git submodule update --init --recursive
cd serve/docker
./build_image.sh -g -cv cu121 -py 3.11 -bt production -t ghcr.io/ai-cfia/nachet-model-ccds/serve-gpu-cu121-py311-prod:20250214
./build_image.sh -py 3.11 -bi ubuntu:24.04 -bt production -t ghcr.io/ai-cfia/nachet-model-ccds/serve-cpu-py311-prod:20250214

# retag built image
# docker tag pytorch-gpu-24:latest ghcr.io/ai-cfia/nachet-model-ccds/pytorch-gpu-24-devenv:20250120

export CR_PAT=sometoken
echo $CR_PAT | docker login ghcr.io -u ChromaticPanic --password-stdin
docker push ghcr.io/ai-cfia/nachet-model-ccds/serve-gpu-cu121-py311-prod:20250214
docker push ghcr.io/ai-cfia/nachet-model-ccds/serve-cpu-py311-prod:20250214

docker tag local/ai-cfia/nachet-model-ccds/gpu-classifier-27spp-model-1:20250303 ghcr.io/ai-cfia/nachet-model-ccds/gpu-classifier-27spp-model-1:20250303
docker push ghcr.io/ai-cfia/nachet-model-ccds/gpu-classifier-27spp-model-1:20250303
```

cpu is based on ubuntu 24.04
gpu is based on nvidia/cuda:12.1.0-base-ubuntu22.04