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
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = load_config(args.config)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load teacher model
    print(f"Loading teacher model: {args.teacher_model}")
    teacher_model = BLIPTeacherModel(model_path=args.teacher_model)
    
    # Initialize student model
    student_config = DistilledBLIPConfig(
        vision_model_name=config.get("vision_model_name", "google/vit-base-patch16-224-in21k"),
        text_model_name=config.get("text_model_name", "bert-base-uncased"),
        vision_hidden_size=config.get("vision_hidden_size", 768),
        text_hidden_size=config.get("text_hidden_size", 768),
        cross_attention_dim=config.get("cross_attention_dim", 768),
        num_visual_encoder_layers=config.get("num_visual_encoder_layers", 6),
        num_text_encoder_layers=config.get("num_text_encoder_layers", 6),
        num_text_decoder_layers=config.get("num_text_decoder_layers", 6),
        num_attention_heads=config.get("num_attention_heads", 8),
        intermediate_size=config.get("intermediate_size", 2048),
    )
    
    print("Initializing student model...")
    student_model = DistilledBLIPForConditionalGeneration(student_config)
    
    # Create data loaders
    print("Creating data loaders...")
    data_config = config.get("data", {})
    
    train_dataloader = get_captioning_dataloader(
        image_dir=data_config.get("train_image_dir", "data/coco/train2017"),
        annotations_file=data_config.get("train_annotations", "data/coco/annotations/captions_train2017.json"),
        processor=teacher_model.processor,
        batch_size=data_config.get("batch_size", 32),
        max_length=data_config.get("max_length", 30),
        split="train",
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
    )
    
    if data_config.get("val_image_dir") and data_config.get("val_annotations"):
        val_dataloader = get_captioning_dataloader(
            image_dir=data_config.get("val_image_dir", "data/coco/val2017"),
            annotations_file=data_config.get("val_annotations", "data/coco/annotations/captions_val2017.json"),
            processor=teacher_model.processor,
            batch_size=data_config.get("batch_size", 32),
            max_length=data_config.get("max_length", 30),
            split="val",
            shuffle=False,
            num_workers=data_config.get("num_workers", 4),
        )
    else:
        val_dataloader = None
    
    # Create loss function
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
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=training_config.get("learning_rate", 5e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        betas=(training_config.get("beta1", 0.9), training_config.get("beta2", 0.999)),
    )
    
    # Create learning rate scheduler
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * training_config.get("warmup_ratio", 0.1))
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Create trainer
    print("Initializing trainer...")
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        distill_config=distill_config,
        device=device,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        mixed_precision=training_config.get("mixed_precision", True),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
    )
    
    # Resume from checkpoint if provided
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Train the model
    print(f"Starting training for {args.num_epochs} epochs...")
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=training_config.get("save_every", 1),
        eval_every=training_config.get("eval_every", 1),
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
