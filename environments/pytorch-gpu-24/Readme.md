# Commands
```bash
export CR_PAT=sometoken
echo $CR_PAT | docker login ghcr.io -u ChromaticPanic --password-stdin

cd nachet-model-ccds/environments/pytorch-gpu-24
docker build -t pytorch-gpu-24-devenv:2025031401 .
docker tag local/ai-cfia/nachet-model-ccds/pytorch-gpu-24-devenv:2025031401 ghcr.io/ai-cfia/nachet-model-ccds/pytorch-gpu-24-devenv:20250319
docker push ghcr.io/ai-cfia/nachet-model-ccds/pytorch-gpu-24-devenv:20250319

code tunnel --accept-terms-of-service
```