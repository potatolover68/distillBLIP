import os
import time
import logging
import warnings
import signal
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image

from models.teacher_model import BLIPTeacherModel
from models.student_model import DistilledBLIPForConditionalGeneration, DistilledBLIPConfig
from models.distillation import CombinedDistillationLoss

# Filter out FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*User provided device_type of 'cuda'.")

# Global flag for graceful exit
_GRACEFUL_EXIT = False

def graceful_exit_handler(signum, frame):
    """Signal handler for graceful exit."""
    global _GRACEFUL_EXIT
    print("\nReceived exit signal. Will save checkpoint and exit after current batch...")
    _GRACEFUL_EXIT = True

# Register signal handlers
signal.signal(signal.SIGINT, graceful_exit_handler)


class DistillationTrainer:
    """
    Trainer class for knowledge distillation of BLIP models.
    """
    
    def __init__(
        self,
        teacher_model: Optional[BLIPTeacherModel] = None,
        student_model: Optional[DistilledBLIPForConditionalGeneration] = None,
        teacher_model_name: str = "Salesforce/blip-image-captioning-large",
        student_config: Optional[DistilledBLIPConfig] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        distill_config: Dict[str, Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "checkpoints",
        log_dir: str = "logs",
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        fp16_opt_level: str = "O2",
        use_amp: bool = True,
        use_gradient_checkpointing: bool = False,
        use_compile: bool = False,
    ):
        """
        Initialize the distillation trainer.
        
        Args:
            teacher_model: The teacher model (original BLIP model).
            student_model: The student model (distilled BLIP model).
            teacher_model_name: Name or path of the teacher model to load.
            student_config: Configuration for the student model.
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            loss_fn: Loss function for distillation.
            optimizer: Optimizer for training the student model.
            lr_scheduler: Learning rate scheduler.
            distill_config: Configuration for distillation process.
            device: Device to run the training on ("cuda" or "cpu").
            output_dir: Directory to save model checkpoints.
            log_dir: Directory to save training logs.
            mixed_precision: Whether to use mixed precision training.
            gradient_accumulation_steps: Number of steps to accumulate gradients.
            max_grad_norm: Maximum gradient norm for gradient clipping.
            fp16_opt_level: Mixed precision optimization level.
            use_amp: Whether to use automatic mixed precision.
            use_gradient_checkpointing: Whether to use gradient checkpointing to save memory.
            use_compile: Whether to use torch.compile for faster training.
        """
        self.device = device
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.fp16_opt_level = fp16_opt_level
        self.use_amp = use_amp
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_compile = use_compile
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(log_dir, "training.log"))
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Setup teacher model
        if teacher_model is not None:
            self.teacher_model = teacher_model
        else:
            self.logger.info(f"Loading teacher model from {teacher_model_name}...")
            self.teacher_model = BLIPTeacherModel(model_path=teacher_model_name)
        
        self.teacher_model = self.teacher_model.to(device)
        self.teacher_model.eval()  # Teacher model is always in eval mode
        
        # Setup student model
        if student_model is not None:
            self.student_model = student_model
        else:
            student_config = student_config or DistilledBLIPConfig()
            self.logger.info("Initializing student model...")
            self.student_model = DistilledBLIPForConditionalGeneration(student_config)
        
        self.student_model = self.student_model.to(device)
        
        # Apply gradient checkpointing if enabled
        if self.use_gradient_checkpointing and hasattr(self.student_model, "gradient_checkpointing_enable"):
            self.logger.info("Enabling gradient checkpointing...")
            self.student_model.gradient_checkpointing_enable()
        
        # Apply torch.compile if enabled and available
        if self.use_compile and hasattr(torch, "compile") and torch.__version__ >= "2.0.0":
            self.logger.info("Compiling student model with torch.compile()...")
            self.student_model = torch.compile(self.student_model)
        
        # Setup distillation loss
        default_distill_config = {
            "temperature": 2.0,
            "alpha": 0.5,
            "lambda_logits": 1.0,
            "lambda_feature": 0.5,
            "lambda_attn": 0.5,
            "use_feature_distillation": True,
            "use_attn_distillation": True,
        }
        
        distill_config = {**default_distill_config, **(distill_config or {})}
        
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = CombinedDistillationLoss(**distill_config)
        
        # Setup optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.AdamW(
                self.student_model.parameters(),
                lr=5e-5,
                weight_decay=0.01,
                betas=(0.9, 0.999),
            )
        
        # Setup learning rate scheduler
        self.lr_scheduler = lr_scheduler
        
        # Setup gradient scaler for mixed precision training
        self.scaler = GradScaler() if mixed_precision else None
        
        # Setup dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        
        self.logger.info("Distillation trainer initialized successfully!")
        
        # Log GPU memory usage
        if torch.cuda.is_available():
            self.logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            self.logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    def train(self, num_epochs: int, save_every: int = 1, eval_every: int = 1):
        """
        Train the student model using knowledge distillation.
        
        Args:
            num_epochs (int): Number of epochs to train for.
            save_every (int): Save checkpoint every N epochs.
            eval_every (int): Evaluate on validation set every N epochs.
        """
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.mixed_precision}")
        self.logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        # Print keyboard interrupt instructions
        print("\n" + "="*80)
        print("Press Ctrl+C to gracefully exit training and save a checkpoint")
        print("="*80 + "\n")
        
        try:
            for epoch in range(self.epoch, num_epochs):
                self.epoch = epoch
                self.logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
                
                # Train for one epoch
                train_metrics = self._train_epoch()
                
                # Log training metrics
                self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Training metrics:")
                for k, v in train_metrics.items():
                    self.logger.info(f"  {k}: {v:.4f}")
                    self.writer.add_scalar(f"epoch/{k}", v, epoch)
                
                # Evaluate on validation set
                if self.val_dataloader is not None and (epoch + 1) % eval_every == 0:
                    self.logger.info(f"Evaluating on validation set...")
                    val_metrics = self._validate()
                    
                    # Log validation metrics
                    self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation metrics:")
                    for k, v in val_metrics.items():
                        self.logger.info(f"  {k}: {v:.4f}")
                        self.writer.add_scalar(f"val/{k}", v, epoch)
                    
                    # Save best model
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self._save_checkpoint("best.pth")
                        self.logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
                
                # Save checkpoint
                if (epoch + 1) % save_every == 0:
                    self._save_checkpoint(f"epoch_{epoch}.pth")
                    self.logger.info(f"Checkpoint saved at epoch {epoch+1}")
                
                # Check for graceful exit
                if _GRACEFUL_EXIT:
                    self.logger.info("Graceful exit requested. Saving final checkpoint and exiting...")
                    self._save_checkpoint(f"epoch_{epoch}_interrupted.pth")
                    break
                
            # Save final model
            self._save_checkpoint("final.pth")
            self.logger.info("Training completed!")
            
        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted by user. Saving checkpoint...")
            self._save_checkpoint(f"epoch_{self.epoch}_interrupted.pth")
            self.logger.info("Checkpoint saved. Exiting...")
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self._save_checkpoint("error_checkpoint.pth")
            self.logger.info("Error checkpoint saved.")
            raise
    
    def _train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            dict: Dictionary containing training metrics.
        """
        self.student_model.train()
        
        # Initialize metrics
        epoch_loss = 0
        epoch_metrics = {}
        
        # Create a progress bar with tqdm
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch+1}",
            leave=True,
            dynamic_ncols=True,
            unit="batch"
        )
        
        # Training loop
        for step, batch in enumerate(progress_bar):
            # Check for graceful exit
            if _GRACEFUL_EXIT:
                self.logger.info("Graceful exit requested. Saving checkpoint and exiting...")
                self._save_checkpoint(f"epoch_{self.epoch}_interrupted.pth")
                return {"total_loss": epoch_loss / (step + 1)}
                
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if "pixel_values" not in batch:
                # If batch contains raw images, process them
                images = batch.get("images", [])
                if images and isinstance(images[0], Image.Image):
                    batch = self.teacher_model.prepare_inputs(images, batch.get("captions", None))
                    batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with the teacher model
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                )
            
            # Forward pass with the student model
            if self.mixed_precision:
                with autocast():
                    student_outputs = self.student_model(
                        pixel_values=batch["pixel_values"],
                        input_ids=batch.get("input_ids"),
                        attention_mask=batch.get("attention_mask"),
                        labels=batch.get("labels"),
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                    
                    # Compute distillation losses
                    losses = self.loss_fn(
                        student_outputs=student_outputs,
                        teacher_outputs=teacher_outputs,
                        labels=batch.get("labels"),
                    )
                    
                    loss = losses["total_loss"]
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.max_grad_norm)
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
            else:
                student_outputs = self.student_model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                    output_attentions=True,
                    output_hidden_states=True,
                )
                
                # Compute distillation losses
                losses = self.loss_fn(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    labels=batch.get("labels"),
                )
                
                loss = losses["total_loss"]
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.max_grad_norm)
                    
                    # Update weights
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar with metrics
            metrics_to_display = {
                "loss": loss.item() * self.gradient_accumulation_steps,
            }
            
            # Add individual loss components to metrics
            for loss_name, loss_value in losses.items():
                if loss_name != "total_loss":
                    metrics_to_display[loss_name] = loss_value.item()
                    
            progress_bar.set_postfix(metrics_to_display)
            
            # Log to tensorboard
            self.writer.add_scalar("train/loss", loss.item() * self.gradient_accumulation_steps, self.global_step)
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            
            for loss_name, loss_value in losses.items():
                self.writer.add_scalar(f"train/{loss_name}", loss_value.item(), self.global_step)
            
            # Update epoch metrics
            for k, v in metrics_to_display.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = 0
                epoch_metrics[k] += v
            
            self.global_step += 1
            
            # Check for graceful exit
            if _GRACEFUL_EXIT:
                self.logger.info("Graceful exit requested. Saving checkpoint and exiting...")
                self._save_checkpoint(f"epoch_{self.epoch}_interrupted.pth")
                return {"total_loss": epoch_loss / (step + 1)}
        
        # Compute average metrics
        num_batches = len(self.train_dataloader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        return epoch_metrics
    
    def _validate(self):
        """
        Validate the model on the validation set.
        
        Returns:
            dict: Dictionary containing validation metrics.
        """
        self.student_model.eval()
        
        # Initialize metrics
        epoch_loss = 0
        epoch_metrics = {}
        
        # Create a progress bar with tqdm
        progress_bar = tqdm(
            self.val_dataloader,
            desc="Validation",
            leave=True,
            dynamic_ncols=True,
            unit="batch"
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                if "pixel_values" not in batch:
                    # If batch contains raw images, process them
                    images = batch.get("images", [])
                    if images and isinstance(images[0], Image.Image):
                        batch = self.teacher_model.prepare_inputs(images, batch.get("captions", None))
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with the teacher model
                teacher_outputs = self.teacher_model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                )
                
                # Forward pass with the student model
                student_outputs = self.student_model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                    output_attentions=True,
                    output_hidden_states=True,
                )
                
                # Compute distillation losses
                losses = self.loss_fn(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    labels=batch.get("labels"),
                )
                
                loss = losses["total_loss"]
                
                # Update metrics
                epoch_loss += loss.item()
                
                # Update progress bar with metrics
                metrics_to_display = {
                    "loss": loss.item(),
                }
                
                # Add individual loss components to metrics
                for loss_name, loss_value in losses.items():
                    if loss_name != "total_loss":
                        metrics_to_display[loss_name] = loss_value.item()
                        
                progress_bar.set_postfix(metrics_to_display)
                
                # Update epoch metrics
                for k, v in metrics_to_display.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = 0
                    epoch_metrics[k] += v
        
        # Compute average metrics
        num_batches = len(self.val_dataloader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        return epoch_metrics
    
    def _save_checkpoint(self, name: str):
        """
        Save a checkpoint of the model.
        
        Args:
            name (str): Name of the checkpoint file.
        """
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "student_model_state_dict": self.student_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        # Save the checkpoint
        checkpoint_path = os.path.join(self.output_dir, name)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
            load_optimizer (bool): Whether to load optimizer state.
        """
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.student_model.load_state_dict(checkpoint["student_model_state_dict"])
        
        # Load optimizer state if requested
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if available
        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        self.logger.info(f"Checkpoint loaded successfully from epoch {self.epoch}!")
