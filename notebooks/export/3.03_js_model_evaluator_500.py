#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import numpy as np
from transformers import Swinv2Model, Swinv2ForImageClassification, AutoImageProcessor
from torchvision.transforms import (
    Normalize,
    Lambda,
    Resize,
    CenterCrop,
    ToTensor,
    Compose,
)
from torchvision import datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm.notebook import tqdm

TEST_DIR = "../data/processed/15spp_zoom_level_validation_models/1-seed/test"  # adjust path as needed
CHECKPOINT_PATH = "../models/15spp_zoom_level_validation_models/1_seed_model_20250128/checkpoint-500"  # adjust path as needed
# RESIZE = 256
# IMAGE_SIZE = 192


# In[2]:


# # Load the checkpoint
# checkpoint = torch.load(CHECKPOINT_PATH)

# # Assuming the checkpoint contains model state dict
# model_state_dict = checkpoint['model_state_dict']

# # Create model instance (replace with your model architecture)
# model = Swinv2Model()  # Define your model class
# model.load_state_dict(model_state_dict)

model = Swinv2ForImageClassification.from_pretrained(CHECKPOINT_PATH)
model.eval()  # Set to evaluation mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# In[3]:


# Load the image processor
image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT_PATH)

# Define torchvision transforms to be applied to each image.
if "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
else:
    size = (image_processor.size["height"], image_processor.size["width"])

print("SIZE : ", size)
normalize = (
    Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
    else Lambda(lambda x: x)
)

# Define transformations
transform = Compose(
    [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ]
)



# In[4]:


# Load test dataset from local folder
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

# Get the class to index mapping
class_to_idx = test_dataset.class_to_idx

# Invert the dictionary to get index to class mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Print the mapping
print(idx_to_class)

BATCH_SIZE = 32
# Create data loader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# In[5]:


total_samples = len(test_dataset.imgs)
progress_bar = tqdm(total=total_samples, desc="Test set inference", unit="samples")

# Initialize lists to store predictions and ground truth
predictions = []
y_test = []

# Disable gradient calculation for inference
with torch.no_grad():
    for images, labels in test_loader:
        # Move images and labels to device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        # Get predictions
        _, preds = torch.max(outputs.logits, 1)

        # Append batch predictions and labels
        predictions.extend(preds.cpu().numpy())
        y_test.extend(labels.cpu().numpy())
        progress_bar.update(BATCH_SIZE)

# Convert lists to numpy arrays
predictions = np.array(predictions)
y_test = np.array(y_test)


# In[9]:


with torch.no_grad():
    torch.cuda.empty_cache()


# In[7]:


import matplotlib.pyplot as plt
# Compute confusion matrix
cm = confusion_matrix(y_test, predictions)

# Display non-normalized confusion matrix
# disp = ConfusionMatrixDisplay(
#     cm, display_labels=[idx_to_class[i] for i in range(len(idx_to_class))]
# )
# disp.plot()

# Compute normalized confusion matrix
cm_normalized = confusion_matrix(y_test, predictions, normalize="true")

# Display normalized confusion matrix
disp_normalized = ConfusionMatrixDisplay(
    cm_normalized, display_labels=[idx_to_class[i] for i in range(len(idx_to_class))]
)
# disp_normalized.plot()
# disp.plot()
# disp.ax_.set_title("Non-Normalized Confusion Matrix")
plt.xticks(rotation=80)
fig, ax = plt.subplots(figsize=(10, 10))
disp_normalized.plot(ax=ax)
disp_normalized.ax_.set_title("Normalized Confusion Matrix")
plt.xticks(rotation=80)
plt.show()


# In[8]:


# Generate classification report
report = classification_report(y_test, predictions, target_names=[idx_to_class[i] for i in range(len(idx_to_class))])

# Print the classification report
print(report)

