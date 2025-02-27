import torch
import torch.nn as nn
from transformers import Swinv2ForImageClassification, AutoImageProcessor

class Swinv2ImageClassifier(nn.Module):
    def __init__(self):
        """
        Args:
            model_dir (str): The directory containing the model checkpoint and configuration.
        """
        super(Swinv2ImageClassifier, self).__init__()
        model_dir = "/home/model-server/artifacts"
        self.model = Swinv2ForImageClassification.from_pretrained(model_dir)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor. When using TorchServe’s default image_classifier handler,
                              x is expected to be pre-processed (normalized, resized, etc.)
        Returns:
            torch.Tensor: The logits output from the model.
        """
        outputs = self.model(x)
        return outputs.logits
