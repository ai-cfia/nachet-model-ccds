#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import json
import tarfile
import argparse
from transformers import Swinv2ForImageClassification, AutoImageProcessor
from PIL import Image
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    Lambda,
)


# In[ ]:


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
    dummy_input = (
        torch.randn(1, 3, size, size)
        if isinstance(size, int)
        else torch.randn(1, 3, size[0], size[1])
    )
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
        if hasattr(image_processor, "image_mean")
        and hasattr(image_processor, "image_std")
        else Lambda(lambda x: x)
    )
    transform = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )
    image = Image.open(test_file).convert("RGB")
    return transform(image).unsqueeze(0)


def validate_serialized_file(
    checkpoint_path, serialized_file, test_file=None, index_to_class=None
):
    model = load_model_from_serialized(checkpoint_path, serialized_file)
    if test_file:
        input_tensor = load_test_image(checkpoint_path, test_file)
    else:
        input_tensor = get_dummy_input(checkpoint_path)
    with torch.no_grad():
        outputs = model(input_tensor)
    print("Inference output:", outputs)
    # Compute predicted indices from logits.
    preds = torch.argmax(outputs.logits, dim=1)
    if index_to_class:
        import json

        with open(index_to_class, "r") as f:
            mapping = json.load(f)
        # Convert predicted index to class name using the mapping.
        pred_classes = [
            mapping.get(str(int(idx)), f"Class {int(idx)}") for idx in preds
        ]
        print("Predicted classes:", pred_classes)
    else:
        print("Predicted class indices:", preds)


# In[ ]:


def validate_mar_file(mar_file):
    try:
        with tarfile.open(mar_file, "r") as tar:
            members = tar.getnames()
            if "MAR-INF/manifest.json" not in members:
                print("ERROR: 'MAR-INF/manifest.json' not found in the MAR file.")
                return False
            manifest_member = tar.getmember("MAR-INF/manifest.json")
            with tar.extractfile(manifest_member) as f:
                manifest = json.load(f)
            print("Manifest loaded successfully:")
            print(json.dumps(manifest, indent=4))
            # Check for required keys in manifest. Adjust keys as needed.
            required_keys = ["model", "serializedFile", "handler"]
            missing_keys = [key for key in required_keys if key not in manifest]
            if missing_keys:
                print("Missing keys in manifest:", missing_keys)
                return False
            print("All required keys are present in the manifest.")
            return True
    except Exception as e:
        print("Failed to validate MAR file:", e)
        return False


# In[ ]:


def main():
    parser = argparse.ArgumentParser(
        description="Validate exported serialized model file by running inference."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint or config directory.",
    )
    parser.add_argument(
        "--serialized_file",
        type=str,
        required=True,
        help="Path to the exported serialized model file.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to a test image file to run inference.",
    )
    parser.add_argument(
        "--index_to_class",
        type=str,
        default=None,
        help="Path to the index_to_class.json file.",
    )
    parser.add_argument("--mar_file", type=str, help="Path to the MAR file.")

    print("Test args")
    argarr = [
        "--checkpoint_path ../environments/torchserve/gpu/artifacts",
        "--serialized_file ../environments/torchserve/gpu/artifacts/27spp_model_1_serialized.pt",
        "--test_file ../environments/torchserve/gpu/artifacts/solanum_nigrum.tiff",
        "--index_to_class ../environments/torchserve/gpu/artifacts/index_to_name.json",
        "--mar_file ../environments/torchserve/gpu/artifacts/27spp_model_1.mar"
    ]
    argstr = " ".join(argarr)

    print("Parsing args")
    args = parser.parse_args(argstr.split())
    # args = parser.parse_args()

    validate_serialized_file(
        args.checkpoint_path, args.serialized_file, args.test_file, args.index_to_class
    )

    if args.mar_file:
        validate_mar_file(args.mar_file)


# In[ ]:


# if __name__ == "__main__":
#     main()
main()

