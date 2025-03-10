import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.teacher_model import BLIPTeacherModel
from models.student_model import DistilledBLIPForConditionalGeneration, DistilledBLIPConfig
from models.distillation import CombinedDistillationLoss
from data.datasets import get_captioning_dataloader
from training.trainer import DistillationTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a distilled BLIP model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/distill_config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="logs",
        help="Directory to save training logs"
    )
    parser.add_argument(
        "--teacher_model", 
        type=str, 
        default="Salesforce/blip-image-captioning-large",
        help="Path or name of the teacher model"
    )
    parser.add_argument(
        "--resume_from", 
        type=str, 
        default=None,
        help="Path to a checkpoint to resume training from"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=10,
        help="Number of epochs to train for"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main function to train a distilled BLIP model."""
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load teacher model
    print(f"Loading teacher model: {args.teacher_model}")
    teacher_model = BLIPTeacherModel(model_path=args.teacher_model)
    
    # Create student model
    print("Creating student model...")
    student_config = config.get("model", {})
    student_model = DistilledBLIPForConditionalGeneration(
        DistilledBLIPConfig(**student_config)
    )
    
    # Load data
    print("Loading data...")
    data_config = config.get("data", {})
    
    train_dataloader = get_captioning_dataloader(
        dataset_name=data_config.get("dataset", "coco"),
        split="train",
        batch_size=data_config.get("batch_size", 32),
        image_size=data_config.get("image_size", 384),
        max_length=data_config.get("max_length", 30),
        subset_size=data_config.get("subset_size", None),
        num_workers=data_config.get("num_workers", 4),
    )
    
    val_dataloader = get_captioning_dataloader(
        dataset_name=data_config.get("dataset", "coco"),
        split="validation",
        batch_size=data_config.get("batch_size", 32),
        image_size=data_config.get("image_size", 384),
        max_length=data_config.get("max_length", 30),
        subset_size=data_config.get("subset_size", None),
        num_workers=data_config.get("num_workers", 4),
    )
    
    # Set up distillation loss
    print("Setting up distillation loss...")
    distill_config = config.get("distillation", {})
    loss_fn = CombinedDistillationLoss(
        temperature=distill_config.get("temperature", 2.0),
        alpha=distill_config.get("alpha", 0.5),
        lambda_logits=distill_config.get("lambda_logits", 1.0),
        lambda_feature=distill_config.get("lambda_feature", 0.5),
        lambda_attn=distill_config.get("lambda_attn", 0.5),
        use_feature_distillation=distill_config.get("use_feature_distillation", True),
        use_attn_distillation=distill_config.get("use_attn_distillation", True),
    )
    
    # Create optimizer
    print("Setting up optimizer...")
    training_config = config.get("training", {})
    
    # Ensure numeric values are properly converted to floats
    learning_rate = float(training_config.get("learning_rate", 5e-5))
    weight_decay = float(training_config.get("weight_decay", 0.01))
    beta1 = float(training_config.get("beta1", 0.9))
    beta2 = float(training_config.get("beta2", 0.999))
    
    # Get device from config, default to CUDA if available, otherwise CPU
    device = training_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move models to the appropriate device
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2) if "beta1" in training_config and "beta2" in training_config else (0.9, 0.999),
    )
    
    # Create learning rate scheduler
    total_steps = len(train_dataloader) * training_config.get("num_epochs", 10)
    warmup_steps = training_config.get("warmup_steps", 0)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        mixed_precision=training_config.get("mixed_precision", True),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
        # Extract epoch number from checkpoint name if possible
        try:
            checkpoint_name = os.path.basename(args.resume_from)
            if checkpoint_name.startswith("epoch_") and "_" in checkpoint_name:
                epoch_str = checkpoint_name.split("_")[1]
                if epoch_str.isdigit():
                    start_epoch = int(epoch_str) + 1
                    print(f"Resuming from epoch {start_epoch}")
        except:
            print("Could not determine starting epoch from checkpoint name. Starting from the beginning.")
    
    # Train the model
    print("Starting training...")
    try:
        trainer.train(
            num_epochs=training_config.get("num_epochs", 10),
            save_every=training_config.get("save_steps", 1),
            eval_every=training_config.get("eval_steps", 1),
        )
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        trainer._save_checkpoint("manual_interrupt.pth")
        print("Checkpoint saved. Exiting...")
    except Exception as e:
        print(f"Error during training: {e}")
        # Save checkpoint on error
        trainer._save_checkpoint("error_checkpoint.pth")
        raise


if __name__ == "__main__":
    main()
