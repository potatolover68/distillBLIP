import torch
import numpy as np
from typing import List, Dict, Any, Union

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class CaptioningMetrics:
    """
    Class for computing captioning metrics (BLEU, METEOR, ROUGE, CIDEr, SPICE).
    """
    def __init__(self, use_spice=False):
        """
        Initialize the captioning metrics.
        
        Args:
            use_spice (bool): Whether to use SPICE metric (computationally expensive).
        """
        self.metrics = {
            'BLEU': Bleu(4),
            'METEOR': Meteor(),
            'ROUGE': Rouge(),
            'CIDEr': Cider(),
        }
        
        if use_spice:
            self.metrics['SPICE'] = Spice()
        
        self.use_spice = use_spice
    
    def compute_metrics(self, ground_truth: Dict[str, List[str]], predictions: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Compute captioning metrics.
        
        Args:
            ground_truth: Dictionary of image IDs to lists of reference captions.
            predictions: Dictionary of image IDs to lists of predicted captions.
            
        Returns:
            Dict: Dictionary of metric names to scores.
        """
        result = {}
        
        for metric_name, metric in self.metrics.items():
            score, scores = metric.compute_score(ground_truth, predictions)
            
            if metric_name == 'BLEU':
                # BLEU returns a list of scores for BLEU-1,2,3,4
                for i, bleu_score in enumerate(score):
                    result[f'BLEU-{i+1}'] = bleu_score
            else:
                result[metric_name] = score
        
        return result


def calculate_model_size(model) -> Dict[str, Union[int, float]]:
    """
    Calculate model size metrics.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Dict: Dictionary containing model size metrics.
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
    }


def compute_latency(model, inputs, device='cuda', warmup=10, num_runs=100) -> Dict[str, float]:
    """
    Compute inference latency.
    
    Args:
        model: PyTorch model.
        inputs: Model inputs.
        device: Device to run inference on.
        warmup: Number of warmup runs.
        num_runs: Number of timed runs.
        
    Returns:
        Dict: Dictionary containing latency metrics.
    """
    model.to(device)
    model.eval()
    
    # Prepare inputs
    if isinstance(inputs, dict):
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs) if isinstance(inputs, dict) else model(inputs)
    
    # Timed runs
    torch.cuda.synchronize() if device == 'cuda' else None
    start_event = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    end_event = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                start_event.record()
                _ = model(**inputs) if isinstance(inputs, dict) else model(inputs)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
            else:
                import time
                start_time = time.time()
                _ = model(**inputs) if isinstance(inputs, dict) else model(inputs)
                latencies.append((time.time() - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    latencies = np.array(latencies)
    metrics = {
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_median_ms': float(np.median(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'latency_min_ms': float(np.min(latencies)),
        'latency_max_ms': float(np.max(latencies)),
        'latency_p90_ms': float(np.percentile(latencies, 90)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
    }
    
    return metrics


def compare_models(original_model, distilled_model, original_metrics, distilled_metrics) -> Dict[str, Any]:
    """
    Compare original and distilled models.
    
    Args:
        original_model: Original model.
        distilled_model: Distilled model.
        original_metrics: Metrics for original model.
        distilled_metrics: Metrics for distilled model.
        
    Returns:
        Dict: Dictionary containing comparison results.
    """
    # Model size comparison
    original_size = calculate_model_size(original_model)
    distilled_size = calculate_model_size(distilled_model)
    
    # Calculate size reduction
    size_reduction = {
        'parameter_reduction': 1 - (distilled_size['total_parameters'] / original_size['total_parameters']),
        'model_size_reduction': 1 - (distilled_size['model_size_mb'] / original_size['model_size_mb']),
    }
    
    # Performance reduction
    perf_metrics = {}
    for metric in original_metrics:
        if metric in ['latency_mean_ms', 'latency_median_ms']:
            # For latency, lower is better
            perf_metrics[f'{metric}_speedup'] = original_metrics[metric] / distilled_metrics[metric]
        elif metric in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE', 'CIDEr', 'SPICE']:
            # For captioning metrics, higher is better
            if metric in original_metrics and metric in distilled_metrics:
                perf_metrics[f'{metric}_reduction'] = 1 - (distilled_metrics[metric] / original_metrics[metric])
    
    return {
        'original_size': original_size,
        'distilled_size': distilled_size,
        'size_reduction': size_reduction,
        'performance_metrics': perf_metrics,
    }
