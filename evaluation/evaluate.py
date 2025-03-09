import os
import argparse
import json
import time
import torch
import yaml
from tqdm import tqdm
from typing import Dict, List, Optional, Any

from models.teacher_model import BLIPTeacherModel
from models.student_model import DistilledBLIPForConditionalGeneration, DistilledBLIPConfig
from data.datasets import get_captioning_dataloader
from evaluation.metrics import CaptioningMetrics, calculate_model_size, compute_latency, compare_models


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a distilled BLIP model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/eval_config.yaml",
        help="Path to the evaluation configuration file"
    )
    parser.add_argument(
        "--teacher_model", 
        type=str, 
        default="Salesforce/blip-image-captioning-large",
        help="Path or name of the teacher model"
    )
    parser.add_argument(
        "--student_model", 
        type=str, 
        required=True,
        help="Path to the trained student model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--compare_with_teacher", 
        action="store_true",
        help="Compare the student model with the teacher model"
    )
    parser.add_argument(
        "--gen_max_length", 
        type=int, 
        default=30,
        help="Maximum length for generated captions"
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_student_model(checkpoint_path, device):
    """
    Load the trained student model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint.
        device (str): Device to load the model on.
        
    Returns:
        model: The loaded student model.
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
    
    model = model.to(device)
    model.eval()
    
    return model


def generate_captions(model, dataloader, device, max_length=30, num_beams=4, processor=None):
    """
    Generate captions for the images in the dataloader.
    
    Args:
        model: The model to use for caption generation.
        dataloader: DataLoader containing the images.
        device: Device to run generation on.
        max_length: Maximum length for generated captions.
        num_beams: Number of beams for beam search.
        processor: Processor for tokenizing and detokenizing.
        
    Returns:
        dict: Dictionary containing reference and generated captions.
    """
    references = {}
    predictions = {}
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating captions")):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Get image IDs (use indices if not available)
        image_ids = batch.get("image_ids", [f"{batch_idx}_{i}" for i in range(len(batch["pixel_values"]))])
        
        # Get reference captions
        if "input_ids" in batch and processor:
            for i, ids in enumerate(batch["input_ids"]):
                image_id = image_ids[i]
                caption = processor.decode(ids, skip_special_tokens=True)
                if image_id not in references:
                    references[image_id] = []
                references[image_id].append(caption)
        
        # Generate captions
        with torch.no_grad():
            if hasattr(model, "generate_captions"):
                captions = model.generate_captions(
                    pixel_values=batch["pixel_values"],
                    max_length=max_length,
                    num_beams=num_beams
                )
            else:
                # For student model
                generated_ids = model.generate(
                    pixel_values=batch["pixel_values"],
                    max_length=max_length,
                    num_beams=num_beams
                )
                
                captions = processor.batch_decode(generated_ids, skip_special_tokens=True) if processor else generated_ids
            
            # Store predictions
            for i, caption in enumerate(captions):
                image_id = image_ids[i]
                if image_id not in predictions:
                    predictions[image_id] = []
                predictions[image_id].append(caption)
    
    return {"references": references, "predictions": predictions}


def evaluate_model(model, dataloader, device, processor=None, gen_kwargs=None):
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model: The model to evaluate.
        dataloader: DataLoader containing the evaluation data.
        device: Device to run evaluation on.
        processor: Processor for tokenizing and detokenizing.
        gen_kwargs: Keyword arguments for generation.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    gen_kwargs = gen_kwargs or {"max_length": 30, "num_beams": 4}
    
    # Generate captions
    generation_results = generate_captions(
        model=model,
        dataloader=dataloader,
        device=device,
        processor=processor,
        **gen_kwargs
    )
    
    # Calculate captioning metrics
    metrics = CaptioningMetrics(use_spice=False)
    captioning_scores = metrics.compute_metrics(
        generation_results["references"],
        generation_results["predictions"]
    )
    
    # Calculate model size
    model_size = calculate_model_size(model)
    
    # Compute latency (using a small batch for measurement)
    sample_inputs = next(iter(dataloader))
    sample_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_inputs.items()}
    
    latency_metrics = compute_latency(
        model=model,
        inputs=sample_inputs,
        device=device,
        warmup=5,
        num_runs=50
    )
    
    # Combine all metrics
    all_metrics = {
        **captioning_scores,
        **model_size,
        **latency_metrics,
    }
    
    return all_metrics, generation_results


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {}
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load teacher model if comparing
    teacher_model = None
    teacher_processor = None
    if args.compare_with_teacher:
        print(f"Loading teacher model: {args.teacher_model}")
        teacher_wrapper = BLIPTeacherModel(model_path=args.teacher_model)
        teacher_model = teacher_wrapper.model
        teacher_processor = teacher_wrapper.processor
        teacher_model = teacher_model.to(args.device)
        teacher_model.eval()
    
    # Load student model
    print(f"Loading student model from: {args.student_model}")
    student_model = load_student_model(args.student_model, args.device)
    
    # Create dataloader
    data_config = config.get("data", {})
    print("Creating evaluation dataloader...")
    dataloader = get_captioning_dataloader(
        image_dir=data_config.get("image_dir", "data/coco/val2017"),
        annotations_file=data_config.get("annotations", "data/coco/annotations/captions_val2017.json"),
        processor=teacher_processor,
        batch_size=args.batch_size,
        max_length=data_config.get("max_length", 30),
        split="val",
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
    )
    
    # Generation parameters
    gen_kwargs = {
        "max_length": args.gen_max_length,
        "num_beams": config.get("num_beams", 4),
    }
    
    # Evaluate student model
    print("Evaluating student model...")
    student_metrics, student_results = evaluate_model(
        model=student_model,
        dataloader=dataloader,
        device=args.device,
        processor=teacher_processor,
        gen_kwargs=gen_kwargs,
    )
    
    # Save student results
    student_output_path = os.path.join(args.output_dir, "student_results.json")
    with open(student_output_path, "w") as f:
        json.dump({
            "metrics": student_metrics,
            "generation_samples": {k: v for k, v in list(student_results["predictions"].items())[:10]},
        }, f, indent=2)
    
    print(f"Student evaluation results saved to: {student_output_path}")
    
    # Print key metrics
    print("\nStudent Model Metrics:")
    print(f"BLEU-4: {student_metrics.get('BLEU-4', 'N/A')}")
    print(f"CIDEr: {student_metrics.get('CIDEr', 'N/A')}")
    print(f"METEOR: {student_metrics.get('METEOR', 'N/A')}")
    print(f"Model Size (MB): {student_metrics.get('model_size_mb', 'N/A')}")
    print(f"Mean Latency (ms): {student_metrics.get('latency_mean_ms', 'N/A')}")
    
    # Evaluate teacher model and compare if requested
    if args.compare_with_teacher:
        print("\nEvaluating teacher model...")
        teacher_metrics, teacher_results = evaluate_model(
            model=teacher_model,
            dataloader=dataloader,
            device=args.device,
            processor=teacher_processor,
            gen_kwargs=gen_kwargs,
        )
        
        # Save teacher results
        teacher_output_path = os.path.join(args.output_dir, "teacher_results.json")
        with open(teacher_output_path, "w") as f:
            json.dump({
                "metrics": teacher_metrics,
                "generation_samples": {k: v for k, v in list(teacher_results["predictions"].items())[:10]},
            }, f, indent=2)
        
        print(f"Teacher evaluation results saved to: {teacher_output_path}")
        
        # Compare models
        print("\nComparing teacher and student models...")
        comparison_results = compare_models(
            original_model=teacher_model,
            distilled_model=student_model,
            original_metrics=teacher_metrics,
            distilled_metrics=student_metrics,
        )
        
        # Save comparison results
        comparison_output_path = os.path.join(args.output_dir, "comparison_results.json")
        with open(comparison_output_path, "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"Model comparison results saved to: {comparison_output_path}")
        
        # Print comparison summary
        print("\nComparison Summary:")
        print(f"Parameter Reduction: {comparison_results['size_reduction']['parameter_reduction'] * 100:.2f}%")
        print(f"Model Size Reduction: {comparison_results['size_reduction']['model_size_reduction'] * 100:.2f}%")
        print(f"Latency Speedup: {comparison_results['performance_metrics'].get('latency_mean_ms_speedup', 'N/A'):.2f}x")
        
        # Print performance metrics comparison
        print("\nPerformance Metrics Comparison:")
        print(f"BLEU-4 - Teacher: {teacher_metrics.get('BLEU-4', 'N/A')}, Student: {student_metrics.get('BLEU-4', 'N/A')}")
        print(f"CIDEr - Teacher: {teacher_metrics.get('CIDEr', 'N/A')}, Student: {student_metrics.get('CIDEr', 'N/A')}")
        print(f"METEOR - Teacher: {teacher_metrics.get('METEOR', 'N/A')}, Student: {student_metrics.get('METEOR', 'N/A')}")


if __name__ == "__main__":
    main()
