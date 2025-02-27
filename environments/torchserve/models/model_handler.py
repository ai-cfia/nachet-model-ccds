import os
import torch
import json
import torch.nn.functional as F
from transformers import Swinv2ForImageClassification, AutoImageProcessor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import io

class ModelHandler(object):
    """
    A custom model handler implementation for Swinv2 image classification.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
        self.image_processor = None
        self.idx_to_class = None
        self.transform = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """
        self._context = context
        self.manifest = context.manifest
        
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        # model_dir = "/home/model-server/artifacts"
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        
        # Load the class mapping from index.json
        index_path = os.path.join(model_dir, "index_to_name.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self.idx_to_class = json.load(f)
        else:
            self.idx_to_class = None
            print("Warning: index_to_name.json not found. Class names will not be available.")
            
        # Option 1: Load the model using HuggingFace directly
        try:
            self.model = Swinv2ForImageClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            
            # Load image processor for preprocessing
            self.image_processor = AutoImageProcessor.from_pretrained(model_dir)
            
            # Create transforms based on image processor config
            if "shortest_edge" in self.image_processor.size:
                size = self.image_processor.size["shortest_edge"]
            else:
                size = (self.image_processor.size["height"], self.image_processor.size["width"])
                
            normalize = Normalize(
                mean=self.image_processor.image_mean, 
                std=self.image_processor.image_std
            ) if hasattr(self.image_processor, "image_mean") else None
            
            transform_list = [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
            ]
            if normalize:
                transform_list.append(normalize)
                
            self.transform = Compose(transform_list)
            
        except Exception as e:
            print(f"Error loading model with HuggingFace: {e}")
            # Option 2: Fall back to loading serialized model
            serialized_file = self.manifest['model']['serializedFile']
            model_pt_path = os.path.join(model_dir, serialized_file)
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")
            
            self.model = torch.jit.load(model_pt_path, map_location=self.device)

        self.initialized = True

    def preprocess(self, data):
        """
        Preprocess the input data
        """
        images = []
        for row in data:
            # Load image
            image = row.get("data") or row.get("body")
            if isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image)).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            images.append(image)
        
        # Stack all images into a batch
        image_tensor = torch.stack(images).to(self.device)
        return image_tensor

    def inference(self, image_tensor):
        """
        Run inference on the preprocessed data
        """
        with torch.no_grad():
            if hasattr(self.model, "forward"):
                outputs = self.model(image_tensor)
            else:
                # For torch.jit models
                outputs = self.model.forward(image_tensor)
                
            # Get logits - handle different model output formats
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
                
        return logits

    def postprocess(self, logits):
        """
        Post-process the model output to get top 5 predictions
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get top-5 predicted class indices and probabilities
        # Use min to handle cases with fewer than 5 classes
        k = min(5, probs.shape[1])
        top_probs, top_indices = torch.topk(probs, k, dim=1)
        
        result = []
        for i in range(len(top_indices)):
            top_predictions = []
            for j in range(k):
                idx = top_indices[i, j].item()
                prob = top_probs[i, j].item()
                
                # Map index to class name if available
                class_name = self.idx_to_class[str(idx)] if self.idx_to_class else str(idx)
                
                top_predictions.append({
                    'class': class_name,
                    'class_id': idx,
                    'confidence': prob
                })
            
            result.append(top_predictions)
            
        return result

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediction output
        """
        if not self.initialized:
            self.initialize(context)
            
        # Preprocess
        image_tensor = self.preprocess(data)
        
        # Inference
        logits = self.inference(image_tensor)
        
        # Postprocess
        result = self.postprocess(logits)
        
        return [result]
