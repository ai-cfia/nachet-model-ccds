# Commands
```bash
export CR_PAT=sometoken
echo $CR_PAT | docker login ghcr.io -u ChromaticPanic --password-stdin
docker push ghcr.io/ai-cfia/nachet-model-ccds/pytorch-gpu-24-devenv:20250120
```