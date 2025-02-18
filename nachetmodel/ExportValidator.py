import torch
import argparse
from transformers import Swinv2ForImageClassification, AutoImageProcessor
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda

def load_model_from_serialized(checkpoint_path, serialized_file):
    # Instantiate model architecture and load the saved state_dict
    model = Swinv2ForImageClassification.from_pretrained(checkpoint_path)
    state_dict = torch.load(serialized_file)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_dummy_input(checkpoint_path):
    # Derive dummy input tensor shape from the image processor config
    image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    elif "height" in image_processor.size and "width" in image_processor.size:
        s = image_processor.size
        size = (s["height"], s["width"])
    else:
        size = 224  # default
    dummy_input = torch.randn(1, 3, size, size) if isinstance(size, int) else torch.randn(1, 3, size[0], size[1])
    return dummy_input

def load_test_image(checkpoint_path, test_file):
    # Create a transform similar to ModelEvaluator and load a test image
    image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
        else Lambda(lambda x: x)
    )
    transform = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])
    image = Image.open(test_file).convert("RGB")
    return transform(image).unsqueeze(0)

def validate_serialized_file(checkpoint_path, serialized_file, test_file=None):
    model = load_model_from_serialized(checkpoint_path, serialized_file)
    if test_file:
        input_tensor = load_test_image(checkpoint_path, test_file)
    else:
        input_tensor = get_dummy_input(checkpoint_path)
    with torch.no_grad():
        outputs = model(input_tensor)
    print("Inference output:", outputs)

def main():
    parser = argparse.ArgumentParser(description="Validate exported serialized model file by running inference.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint or config directory.")
    parser.add_argument("--serialized_file", type=str, required=True, help="Path to the exported serialized model file.")
    parser.add_argument("--test_file", type=str, default=None, help="Path to a test image file to run inference.")
    args = parser.parse_args()

    validate_serialized_file(args.checkpoint_path, args.serialized_file, args.test_file)

if __name__ == "__main__":
    main()
