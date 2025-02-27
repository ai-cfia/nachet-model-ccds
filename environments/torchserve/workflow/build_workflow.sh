#!/bin/bash
# Script to build and register the cascade classification workflow

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

# Create model store directory if it doesn't exist
MODEL_STORE="/home/ubuntu/projects/nachet-model-ccds/model-store"
mkdir -p $MODEL_STORE

echo "Building workflow archive..."
torch-workflow-archiver \
  --workflow-name cascade_classification \
  --spec-file cascade_classification.yaml \
  --handler workflow_handler.py \
  --export-path $MODEL_STORE

echo "Workflow archive created at $MODEL_STORE/cascade_classification.war"

# Check if TorchServe is running
if curl -s localhost:8081/models > /dev/null; then
    echo "TorchServe is running. Registering workflow..."
    curl -X POST "http://localhost:8081/workflows?url=cascade_classification.war"
else
    echo "TorchServe is not running. Start TorchServe and register the workflow manually."
    echo "Command to start TorchServe:"
    echo "torchserve --start --model-store $MODEL_STORE --workflow-store $MODEL_STORE --ncs"
    echo "Command to register workflow:"
    echo "curl -X POST \"http://localhost:8081/workflows?url=cascade_classification.war\""
fi

echo "Done"
