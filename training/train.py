import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import warning suppression utility and apply it immediately
from utils.suppress_warnings import suppress_all_warnings
suppress_all_warnings()

from models.teacher_model import BLIPTeacherModel
from models.student_model import DistilledBLIPForConditionalGeneration, DistilledBLIPConfig
from models.distillation import CombinedDistillationLoss
from data.datasets import get_captioning_dataloader
from training.trainer import DistillationTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a distilled BLIP model")
    parser.add_argument("--config", type=str, default="configs/distill_config.yaml", help="Path to config file")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Set deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def optimize_gpu_memory():
    """Apply GPU memory optimizations."""
    # Empty cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Set memory fraction
    if torch.cuda.is_available():
        # Reserve 90% of GPU memory for this process
        for device in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(device)
            print(f"GPU {device}: {device_props.name} with {device_props.total_memory / 1e9:.2f} GB memory")


def main():
    """Main function for training the model."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    if args.profile:
        config["training"]["profile"] = True
    
    # Set device
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        config["training"]["device"] = device
    
    # Print GPU info if using CUDA
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} with {gpu_mem:.2f} GB memory")
    
    # Set random seeds for reproducibility
    if config["training"].get("deterministic", False):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Load teacher model
    print(f"Loading teacher model: {config['model']['teacher_model_name']}")
    teacher_model = BLIPTeacherModel(config["model"]["teacher_model_name"]).to(device)
    
    # Create student model
    print("Creating student model...")
    student_config = DistilledBLIPConfig(
        vision_model_name=config["model"]["vision_model_name"],
        text_model_name=config["model"]["text_model_name"],
        vision_hidden_size=config["model"]["vision_hidden_size"],
        text_hidden_size=config["model"]["text_hidden_size"],
        cross_attention_dim=config["model"]["cross_attention_dim"],
        num_visual_encoder_layers=config["model"]["num_visual_encoder_layers"],
        num_text_encoder_layers=config["model"]["num_text_encoder_layers"],
        num_text_decoder_layers=config["model"]["num_text_decoder_layers"],
        num_attention_heads=config["model"]["num_attention_heads"],
        intermediate_size=config["model"]["intermediate_size"],
        vocab_size=config["model"]["vocab_size"],
    )
    student_model = DistilledBLIPForConditionalGeneration(student_config).to(device)
    
    # Load data
    print("Loading data...")
    train_dataloader = get_captioning_dataloader(
        dataset_name=config["data"]["dataset"],
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        split="train",
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        subset_size=config["data"]["subset_size"],
        pin_memory=config["training"].get("dataloader_pin_memory", True),
        prefetch_factor=config["training"].get("dataloader_prefetch_factor", 2),
        persistent_workers=config["training"].get("dataloader_persistent_workers", False),
    )
    
    val_dataloader = get_captioning_dataloader(
        dataset_name=config["data"]["dataset"],
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        split="val",
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        subset_size=config["data"]["subset_size"],
        pin_memory=config["training"].get("dataloader_pin_memory", True),
        prefetch_factor=config["training"].get("dataloader_prefetch_factor", 2),
        persistent_workers=config["training"].get("dataloader_persistent_workers", False),
    )
    
    # Calculate total steps for the scheduler
    total_steps = len(train_dataloader) * config["training"]["num_epochs"] // config["training"]["gradient_accumulation_steps"]
    config["training"]["total_steps"] = total_steps
    
    # Add model dimensions to distillation config for proper feature projection
    if "distillation" not in config:
        config["distillation"] = {}
    
    config["distillation"]["student_dim"] = config["model"]["vision_hidden_size"]  # Student hidden size
    config["distillation"]["teacher_dim"] = config["model"]["teacher_hidden_size"]  # Teacher hidden size
    
    # Create trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        device=device,
        output_dir="checkpoints",
        log_dir="logs",
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train(
        num_epochs=config["training"]["num_epochs"],
        save_every=config["training"]["save_steps"],
        eval_every=config["training"]["eval_steps"],
    )
    
    print("Training complete!")


if __name__ == "__main__":
    main()
