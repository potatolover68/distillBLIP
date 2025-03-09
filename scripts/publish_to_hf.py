#!/usr/bin/env python
"""
Script to publish a distilled BLIP model to Hugging Face Hub.
"""

import os
import argparse
import json
import torch
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

from transformers import BlipProcessor
from huggingface_hub import HfApi, create_repo, upload_folder

from models.student_model import DistilledBLIPForConditionalGeneration, DistilledBLIPConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Publish distilled BLIP model to Hugging Face Hub")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained student model checkpoint"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        help="Name for the model on Hugging Face Hub (e.g., 'username/distilled-blip')"
    )
    parser.add_argument(
        "--teacher_model", 
        type=str, 
        default="Salesforce/blip-image-captioning-large",
        help="Teacher model name or path (for processor)"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Whether to make the model repository private"
    )
    parser.add_argument(
        "--description", 
        type=str,
        default="",
        help="Description for the model card"
    )
    
    return parser.parse_args()


def load_student_model(checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Load the trained student model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint.
        device: Device to load the model on.
        
    Returns:
        Dict containing the model and its configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if the checkpoint contains config information
    if "config" in checkpoint:
        config = checkpoint["config"]
        student_config = DistilledBLIPConfig(**config)
    else:
        # Use default configuration
        student_config = DistilledBLIPConfig()
    
    model = DistilledBLIPForConditionalGeneration(student_config)
    
    # Load state dict
    if "student_model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["student_model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return {
        "model": model,
        "config": student_config,
    }


def create_model_card(
    model_name: str,
    description: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    teacher_model: str = "Salesforce/blip-image-captioning-large",
) -> str:
    """
    Create a model card for the Hugging Face Hub.
    
    Args:
        model_name: Name of the model on Hugging Face Hub.
        description: Custom description for the model.
        metrics: Optional evaluation metrics to include.
        teacher_model: Name of the teacher model.
        
    Returns:
        Model card content as a string.
    """
    if not description:
        description = f"A distilled version of the {teacher_model} model for image captioning."
    
    model_card = f"""---
language: en
license: mit
tags:
- distillation
- image-captioning
- blip
- vision-language
datasets:
- coco
---

# {model_name.split('/')[-1]}

{description}

This model is a distilled version of [{teacher_model}](https://huggingface.co/{teacher_model}) created using knowledge distillation techniques. The goal is to provide a smaller, more efficient model while maintaining comparable performance to the original.

## Model Details

- **Model Type:** Distilled BLIP for Image Captioning
- **Original Model:** [{teacher_model}](https://huggingface.co/{teacher_model})
- **Task:** Image Captioning
- **Training Data:** COCO Captions
- **Framework:** PyTorch & Hugging Face Transformers

## Usage

```python
from transformers import BlipProcessor
from PIL import Image
import requests
import torch

# Load the model and processor
processor = BlipProcessor.from_pretrained("{model_name}")
model = DistilledBLIPForConditionalGeneration.from_pretrained("{model_name}")

# Prepare image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

# Generate caption
inputs = processor(image, return_tensors="pt")
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)
```
"""

    # Add metrics if available
    if metrics:
        model_card += "\n## Performance\n\n"
        model_card += "| Metric | Value |\n"
        model_card += "|--------|-------|\n"
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                model_card += f"| {metric_name} | {metric_value:.4f} |\n"
            else:
                model_card += f"| {metric_name} | {metric_value} |\n"
    
    # Add citation
    model_card += """
## Citation

If you use this model, please cite the original BLIP paper:

```bibtex
@inproceedings{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  booktitle={ICML},
  year={2022}
}
```
"""
    
    return model_card


def prepare_model_for_hf(
    model_dict: Dict[str, Any],
    processor,
    output_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
    model_name: str = "distilled-blip",
    description: str = "",
    teacher_model: str = "Salesforce/blip-image-captioning-large",
) -> None:
    """
    Prepare the model for upload to Hugging Face Hub.
    
    Args:
        model_dict: Dictionary containing the model and its configuration.
        processor: The processor to use with the model.
        output_dir: Directory to save the prepared model.
        metrics: Optional evaluation metrics to include in the model card.
        model_name: Name of the model on Hugging Face Hub.
        description: Custom description for the model.
        teacher_model: Name of the teacher model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model = model_dict["model"]
    config = model_dict["config"]
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save configuration
    config_dict = config.to_dict()
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Save processor
    processor.save_pretrained(output_dir)
    
    # Create model card
    model_card = create_model_card(
        model_name=model_name,
        description=description,
        metrics=metrics,
        teacher_model=teacher_model,
    )
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    # Create a simple example script
    example_script = """
from transformers import BlipProcessor
from PIL import Image
import requests
import torch
from models import DistilledBLIPForConditionalGeneration

# Load the model and processor
processor = BlipProcessor.from_pretrained("MODEL_NAME")
model = DistilledBLIPForConditionalGeneration.from_pretrained("MODEL_NAME")

# Prepare image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

# Generate caption
inputs = processor(image, return_tensors="pt")
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)
"""
    example_script = example_script.replace("MODEL_NAME", model_name)
    
    with open(os.path.join(output_dir, "example.py"), "w") as f:
        f.write(example_script)
    
    # Copy model files
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    
    # Get the path to the student_model.py file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    student_model_path = os.path.join(project_dir, "models", "student_model.py")
    
    # Copy the student model file
    shutil.copy(student_model_path, os.path.join(output_dir, "models", "student_model.py"))
    
    # Create an __init__.py file to expose the model class
    init_content = """
from .student_model import DistilledBLIPForConditionalGeneration, DistilledBLIPConfig

__all__ = ["DistilledBLIPForConditionalGeneration", "DistilledBLIPConfig"]
"""
    
    with open(os.path.join(output_dir, "models", "__init__.py"), "w") as f:
        f.write(init_content)


def upload_to_hf_hub(
    local_dir: str,
    repo_name: str,
    private: bool = False,
) -> str:
    """
    Upload the model to Hugging Face Hub.
    
    Args:
        local_dir: Local directory containing the model files.
        repo_name: Name of the repository on Hugging Face Hub.
        private: Whether to make the repository private.
        
    Returns:
        URL of the model on Hugging Face Hub.
    """
    # Create the repository
    api = HfApi()
    
    try:
        create_repo(repo_name, private=private, exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        print("You may need to run 'huggingface-cli login' first.")
        return None
    
    # Upload the model files
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_name,
            commit_message="Upload distilled BLIP model",
        )
        
        return f"https://huggingface.co/{repo_name}"
    except Exception as e:
        print(f"Error uploading model: {e}")
        return None


def load_metrics(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Try to load evaluation metrics if available.
    
    Args:
        model_path: Path to the model checkpoint.
        
    Returns:
        Dictionary of metrics if available, None otherwise.
    """
    # Check for evaluation results in the same directory
    model_dir = os.path.dirname(model_path)
    
    # Look for evaluation results
    eval_paths = [
        os.path.join(model_dir, "evaluation_results.json"),
        os.path.join(os.path.dirname(model_dir), "evaluation_results", "comparison_results.json"),
    ]
    
    for path in eval_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except:
                pass
    
    return None


def main():
    args = parse_args()
    
    print(f"Preparing to publish model from {args.model_path} to Hugging Face Hub as {args.model_name}")
    
    # Load the processor from the teacher model
    print(f"Loading processor from {args.teacher_model}")
    processor = BlipProcessor.from_pretrained(args.teacher_model)
    
    # Load the student model
    print(f"Loading student model from {args.model_path}")
    model_dict = load_student_model(args.model_path)
    
    # Try to load metrics
    metrics = load_metrics(args.model_path)
    
    # Create a temporary directory for preparing the model
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Preparing model for upload in {temp_dir}")
        
        # Prepare the model for upload
        prepare_model_for_hf(
            model_dict=model_dict,
            processor=processor,
            output_dir=temp_dir,
            metrics=metrics,
            model_name=args.model_name,
            description=args.description,
            teacher_model=args.teacher_model,
        )
        
        # Upload to Hugging Face Hub
        print(f"Uploading model to Hugging Face Hub as {args.model_name}")
        model_url = upload_to_hf_hub(
            local_dir=temp_dir,
            repo_name=args.model_name,
            private=args.private,
        )
        
        if model_url:
            print(f"Model successfully uploaded to {model_url}")
            print("\nYou can use the model with the following code:")
            print(f"""
from transformers import BlipProcessor
from PIL import Image
import requests
import torch
from models import DistilledBLIPForConditionalGeneration

# Load the model and processor
processor = BlipProcessor.from_pretrained("{args.model_name}")
model = DistilledBLIPForConditionalGeneration.from_pretrained("{args.model_name}")

# Prepare image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

# Generate caption
inputs = processor(image, return_tensors="pt")
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)
""")
        else:
            print("Failed to upload model to Hugging Face Hub.")
            print("Please check your credentials and try again.")


if __name__ == "__main__":
    main()
