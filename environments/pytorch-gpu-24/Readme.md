# Commands

```bash
export $(xargs < .env)
export GHCR_PAT=sometoken
export GHCR_USER=someuser
export IMAGE_NAME=your-image-name
echo $GHCR_PAT | docker login ghcr.io -u $GHCR_USER --password-stdin

cd nachet-model-ccds/environments/pytorch-gpu-24
docker build --no-cache --secret id=user_password,env=SSH_PASSWORD -t $IMAGE_NAME .
# docker build -t pytorch-gpu-24-devenv:2025050801 .
# docker tag local/ai-cfia/nachet-model-ccds/$IMAGE_NAME ghcr.io/ai-cfia/nachet-model-ccds/$IMAGE_NAME
docker push $IMAGE_NAME

code tunnel --accept-terms-of-service
```

## Force Rebuilding with docker-compose

```bash
# Force rebuild with docker-compose
docker-compose build --no-cache

# Or with environment variable for the password
SSH_PASSWORD=your_secure_password docker-compose build --no-cache
```

## Running the Docker Container

### Using Docker Compose

```bash
# Start the container using docker-compose
docker-compose up -d

# Stop the container
docker-compose down
```

### Using Docker Run (directly)

```bash
# Run the image as a container (replace with your specific image name)
docker run -d \
  --name pytorch-gpu-24 \
  --gpus all \
  -p 2222:22 \
  -v "$(pwd)/../../:/project" \
  --shm-size=1gb \
  --ulimit stack=67108864 \
  --ulimit memlock=-1 \
  ghcr.io/ai-cfia/nachet-model-ccds/pytorch-gpu-24-devenv:2025050801

docker run -d --name pytorch-gpu-24 -p 2222:22 ghcr.io/ai-cfia/nachet-model-ccds/pytorch-gpu-24-devenv:2025050801

# Connect to the container via SSH
ssh -p 2222 nachetuser@localhost

# Stop and remove the container
docker stop pytorch-gpu-24
docker rm pytorch-gpu-24
```

### Additional Commands

```bash
# View running containers
docker ps

# View container logs
docker logs pytorch-gpu-24

# Execute commands in the running container
docker exec -it pytorch-gpu-24 bash
```
