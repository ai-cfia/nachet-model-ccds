# Cascade Classification with TorchServe Workflows

This implementation uses TorchServe's workflow functionality to create a cascade of two models, where the second model is only used when specific classes are detected by the first model.

## Files Overview

- `cascade_classification.yaml`: Workflow specification defining models and data flow
- `workflow_handler.py`: Contains functions for preprocessing, routing logic, and result handling
- `model_handler.py`: Model handler for both primary and secondary classifiers (reused)

## How It Works

1. Input data flows to the preprocessing step
2. The primary model processes the input
3. The `check_trigger_class` function determines whether to route to the secondary model
4. If a trigger class is detected, the input is processed by the secondary model
5. Results from the appropriate model are returned

## Creating the Workflow Archive (WAR)

1. Create a workflow archive using the provided script:

```bash
cd /home/ubuntu/projects/nachet-model-ccds/environments/torchserve/workflow

# Create workflow archive
torch-workflow-archiver \
  --workflow-name cascade_classification \
  --spec-file cascade_classification.yaml \
  --handler workflow_handler.py \
  --export-path /home/ubuntu/projects/nachet-model-ccds/model-store
```

## Deploy Models and Workflow

1. First, ensure your models are archived as MAR files:

```bash
# Package primary model
torch-model-archiver \
  --model-name primary-classifier \
  --version 1.0 \
  --model-file path/to/model_definition.py \
  --serialized-file path/to/model_weights.pth \
  --handler model_handler.py \
  --export-path /home/ubuntu/projects/nachet-model-ccds/model-store

# Package secondary model
torch-model-archiver \
  --model-name secondary-classifier \
  --version 1.0 \
  --model-file path/to/model2_definition.py \
  --serialized-file path/to/model2_weights.pth \
  --handler model_handler.py \
  --export-path /home/ubuntu/projects/nachet-model-ccds/model-store
```

2. Start TorchServe if not already running:

```bash
torchserve --start \
  --model-store /home/ubuntu/projects/nachet-model-ccds/model-store \
  --workflow-store /home/ubuntu/projects/nachet-model-ccds/model-store \
  --ncs
```

3. Register the workflow:

```bash
# Register workflow
curl -X POST "http://localhost:8081/workflows?url=cascade_classification.war"
```

## Customization

Edit the `TRIGGER_CLASSES` list in `workflow_handler.py` to specify which classes from the primary model should trigger the secondary model:

```python
TRIGGER_CLASSES = ["your_class1", "your_class2", "your_class3"]
```

## Usage

Send inference requests to the workflow:

```bash
curl -X POST http://localhost:8080/predictions/cascade_classification \
  -T /path/to/your/image.jpg
```

## Monitoring Workflow

```bash
# List registered workflows
curl http://localhost:8081/workflows

# Get workflow description
curl http://localhost:8081/workflows/cascade_classification
```
