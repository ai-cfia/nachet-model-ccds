#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import subprocess
import argparse
from transformers import (
    Swinv2ForImageClassification,
    AutoImageProcessor,
)


# In[ ]:


def load_model(checkpoint_path):
    # Load model similarly to ModelEvaluator; using from_pretrained
    model = Swinv2ForImageClassification.from_pretrained(checkpoint_path)
    model.eval()
    return model


def export_serialized_model(model, output_file):
    # Save the model's state dictionary to output_file
    torch.save(model.state_dict(), output_file)


def export_ensemble_models(model_states, output_file):
    # Save ensemble (dict of state_dicts) to output_file
    torch.save(model_states, output_file)


def export_to_onnx(model, output_file, checkpoint_path):
    # Derive dummy input tensor shape from the image processor
    image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    elif "height" in image_processor.size and "width" in image_processor.size:
        s = image_processor.size
        size = (s["height"], s["width"])
    else:
        size = 224  # fallback to default size
    if isinstance(size, int):
        dummy_input = torch.randn(1, 3, size, size)
    else:
        dummy_input = torch.randn(1, 3, size[0], size[1])
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )


def run_model_archiver(
    model_name,
    version,
    model_file,
    serialized_file,
    export_path,
    handler,
    requirements="",
    config="",
    extra_files="",
):
    # Build and run the torch-model-archiver command
    cmd = [
        "torch-model-archiver",
        "--model-name",
        model_name,
        "--version",
        version,
        "--model-file",
        model_file,
        "--serialized-file",
        serialized_file,
        "--handler",
        handler,
        "--export-path",
        export_path,
        "-f",
    ]
    if extra_files:
        cmd.extend(["--extra-files", extra_files])
    if requirements:
        cmd.extend(["--requirements-file", requirements])
    if config:
        cmd.extend(["--config", config])
    subprocess.run(cmd, check=True)


# In[ ]:


def get_parser():
    parser = argparse.ArgumentParser(description="Export and archive a PyTorch model.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint or directory",
    )
    parser.add_argument(
        "--serialized_output",
        type=str,
        default="model_serialized.pt",
        help="Output file for serialized model.",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name for the model archive."
    )
    parser.add_argument(
        "--version", type=str, default="1.0", help="Version for the model archive."
    )
    parser.add_argument(
        "--handler",
        type=str,
        default="image_classifier",
        help="Handler file required for model archiver.",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="model_store",
        help="Directory to store the model archive.",
    )
    parser.add_argument(
        "--extra_files",
        type=str,
        default="",
        help="Extra files to include (comma separated if multiple).",
    )
    parser.add_argument(
        "--requirements",
        type=str,
        default="",
        help="Requirements file for the model archive.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Config file for the model archive.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Enable ensemble mode to save multiple models in one serialized file.",
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        help="Export the model to ONNX format (disabled in ensemble mode).",
    )
    parser.add_argument(
        "--onnx_output",
        type=str,
        default="model.onnx",
        help="Output file for ONNX model export.",
    )
    return parser


# In[ ]:


def main():
    parser = get_parser()
    # args = parser.parse_args()

    argarr = [
        "--checkpoint_path models/27spp_model/model_120250130/checkpoint-10000",
        "--serialized_output models/27spp_model/model_120250130/checkpoint-10000/27spp_model_1_serialized.pt",
        "--model_name 27spp_model_1",
        "--version 1.0",
        "--handler image_classifier",
        "--export_path models/27spp_model/model_120250130/checkpoint-10000/",
    ]
    argstr = " ".join(argarr)

    args = parser.parse_args(argstr)

    if args.ensemble:
        ensemble_states = {}
        # Assume checkpoints are subdirectories in checkpoint_path
        for subdir in os.listdir(args.checkpoint_path):
            sub_path = os.path.join(args.checkpoint_path, subdir)
            if os.path.isdir(sub_path):
                print(f"Loading model from {sub_path}...")
                model = load_model(sub_path)
                ensemble_states[subdir] = model.state_dict()
        print("Serializing ensemble models...")
        export_ensemble_models(ensemble_states, args.serialized_output)
    else:
        print("Loading model...")
        model = load_model(args.checkpoint_path)
        print("Serializing model...")
        export_serialized_model(model, args.serialized_output)
        if args.export_onnx:
            print("Exporting model to ONNX format...")
            export_to_onnx(model, args.onnx_output, args.checkpoint_path)

    os.makedirs(args.export_path, exist_ok=True)

    print("Archiving model via torch-model-archiver...")
    run_model_archiver(
        args.model_name,
        args.version,
        args.serialized_output,
        args.export_path,
        args.handler,
        args.requirements,
        args.config,
        args.extra_files,
    )

    print("Model archive created successfully.")


# In[ ]:


main()

