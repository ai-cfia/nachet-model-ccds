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
    # torch.save(model.state_dict(), output_file)
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("model."):
            new_state_dict["model." + k] = v
        else:
            new_state_dict[k] = v
    torch.save(new_state_dict, output_file)


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
        "--model_name", type=str, required=True, help="Name for the model archive."
    )
    parser.add_argument(
        "--version", type=str, default="1.0", help="Version for the model archive."
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
        default="",
        help="Output file for ONNX model export.",
    )
    return parser


# In[ ]:


def main():
    print("Exporting model...")
    parser = get_parser()
    # args = parser.parse_args()

    print("Test args")
    base_model_path = "../environments/torchserve/gpu/artifacts"
    model_name = "27spp_model_1"

    argarr = [
        f"--checkpoint_path {base_model_path}",
        f"--model_name {model_name}",
        f"--version 1.0",
    ]
    argstr = " ".join(argarr)

    print("Parsing args")
    args = parser.parse_args(argstr.split())

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

    files = [
        f"{args.checkpoint_path}/config.properties",
        f"{args.checkpoint_path}/index_to_name.json",
        f"{args.checkpoint_path}/config.json",
        f"{args.checkpoint_path}/model.safetensors",
        f"{args.checkpoint_path}/preprocessor_config.json",
        # f"{args.checkpoint_path}/{args.model_name}_serialized.pt",
    ]
    extra_files = ",".join(files)
    if args.extra_files:
        extra_files += "," + args

    print("Archiving model via torch-model-archiver...")
    run_model_archiver(
        model_name=args.model_name,
        version=args.version,
        requirements=args.requirements,
        export_path=f"{args.checkpoint_path}/",
        handler=f"{args.checkpoint_path}/model_handler.py",
        config=f"{args.checkpoint_path}/config.properties",
        serialized_file=f"{args.checkpoint_path}/{args.model_name}_serialized.pt",
        extra_files=extra_files,
    )

    print("Model archive created successfully.")


# In[ ]:


main()

