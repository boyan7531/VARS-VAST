#!/usr/bin/env python3
"""
Temperature Scaling Calibration for MVFouls Model

This script performs temperature scaling on saved validation logits to improve
probability calibration before threshold optimization. Temperature scaling is
a simple post-processing technique that improves calibration without changing
the model's predictions.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from sklearn.metrics import log_loss, brier_score_loss
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils import get_task_metadata
except ImportError:
    get_task_metadata = None


def setup_logging(level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_validation_logits(logits_file: str) -> Dict:
    """Load validation logits from saved file."""
    logger = logging.getLogger(__name__)
    
    if not Path(logits_file).exists():
        raise FileNotFoundError(f"Logits file not found: {logits_file}")
    
    logger.info(f"📂 Loading validation logits from: {logits_file}")
    data = torch.load(logits_file, map_location='cpu')
    
    logger.info(f"   Epoch: {data.get('epoch', 'unknown')}")
    logger.info(f"   Tasks: {data.get('task_names', ['unknown'])}")
    logger.info(f"   Batches: {len(data.get('logits', []))}")
    
    return data


def aggregate_logits_and_targets(logits_list: List, targets_list: List, multi_task: bool = True) -> Tuple[Dict, Dict]:
    """
    Aggregate logits and targets from multiple batches.
    
    Args:
        logits_list: List of batch logits (dicts for multi-task, tensors for single-task)
        targets_list: List of batch targets (dicts for multi-task, tensors for single-task)
        multi_task: Whether this is multi-task data
        
    Returns:
        Tuple of (aggregated_logits_dict, aggregated_targets_dict)
    """
    logger = logging.getLogger(__name__)
    
    if multi_task:
        # Multi-task: aggregate by task
        task_logits = defaultdict(list)
        task_targets = defaultdict(list)
        
        for batch_logits, batch_targets in zip(logits_list, targets_list):
            if isinstance(batch_logits, dict) and isinstance(batch_targets, dict):
                for task_name in batch_logits.keys():
                    if task_name in batch_targets:
                        task_logits[task_name].append(batch_logits[task_name])
                        task_targets[task_name].append(batch_targets[task_name])
        
        # Concatenate all batches for each task
        aggregated_logits = {}
        aggregated_targets = {}
        
        for task_name in task_logits.keys():
            if task_logits[task_name]:
                aggregated_logits[task_name] = torch.cat(task_logits[task_name], dim=0)
                aggregated_targets[task_name] = torch.cat(task_targets[task_name], dim=0)
                logger.info(f"   Task '{task_name}': {aggregated_logits[task_name].shape[0]} samples, "
                           f"{aggregated_logits[task_name].shape[1]} classes")
    else:
        # Single-task: simple concatenation
        aggregated_logits = {'default': torch.cat(logits_list, dim=0)}
        aggregated_targets = {'default': torch.cat(targets_list, dim=0)}
        logger.info(f"   Single task: {aggregated_logits['default'].shape[0]} samples, "
                   f"{aggregated_logits['default'].shape[1]} classes")
    
    return aggregated_logits, aggregated_targets


class TemperatureScaling(nn.Module):
    """
    Temperature scaling module for calibrating neural network predictions.
    
    Temperature scaling is a post-processing technique that improves calibration
    by applying a learned temperature parameter to logits before softmax.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def calibrate(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        max_iter: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """
        Calibrate temperature using NLL loss on validation set.
        
        Args:
            logits: Raw logits tensor of shape (N, num_classes)
            targets: Target tensor of shape (N,) with class indices
            max_iter: Maximum iterations for optimization
            tolerance: Convergence tolerance
            
        Returns:
            Optimized temperature value
        """
        # Optimize temperature using LBFGS
        optimizer = LBFGS([self.temperature], max_iter=max_iter, tolerance_change=tolerance)
        
        def closure():
            optimizer.zero_grad()
            temperature_logits = self.forward(logits)
            loss = F.cross_entropy(temperature_logits, targets)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return float(self.temperature.item())


def compute_calibration_metrics(probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute calibration metrics for probability predictions.
    
    Args:
        probs: Predicted probabilities of shape (N, num_classes)
        targets: True class indices of shape (N,)
        
    Returns:
        Dict with calibration metrics
    """
    # Convert targets to one-hot for some metrics
    num_classes = probs.shape[1]
    targets_one_hot = np.eye(num_classes)[targets]
    
    # Negative log-likelihood (cross-entropy)
    nll = log_loss(targets, probs)
    
    # Brier score (lower is better)
    brier = brier_score_loss(targets_one_hot.ravel(), probs.ravel())
    
    # Expected Calibration Error (ECE)
    ece = compute_expected_calibration_error(probs, targets)
    
    # Maximum Calibration Error (MCE)
    mce = compute_maximum_calibration_error(probs, targets)
    
    return {
        'nll': float(nll),
        'brier_score': float(brier),
        'ece': float(ece),
        'mce': float(mce)
    }


def compute_expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the expected difference between predicted confidence and accuracy
    across different confidence levels.
    """
    predicted_probs = np.max(probs, axis=1)
    predicted_labels = np.argmax(probs, axis=1)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    total_samples = len(predicted_probs)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predicted_labels[in_bin] == targets[in_bin]).mean()
            avg_confidence_in_bin = predicted_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_maximum_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE is the maximum difference between predicted confidence and accuracy
    across all confidence bins.
    """
    predicted_probs = np.max(probs, axis=1)
    predicted_labels = np.argmax(probs, axis=1)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    max_calibration_error = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
        
        if in_bin.any():
            accuracy_in_bin = (predicted_labels[in_bin] == targets[in_bin]).mean()
            avg_confidence_in_bin = predicted_probs[in_bin].mean()
            calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_calibration_error = max(max_calibration_error, calibration_error)
    
    return max_calibration_error


def calibrate_task_temperature(
    task_logits: torch.Tensor, 
    task_targets: torch.Tensor,
    task_name: str
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Calibrate temperature for a single task.
    
    Args:
        task_logits: Logits tensor of shape (N, num_classes)
        task_targets: Target tensor of shape (N,) with class indices
        task_name: Name of the task
        
    Returns:
        Tuple of (optimal_temperature, before_metrics, after_metrics)
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔧 Calibrating temperature for task '{task_name}'")
    logger.info(f"   Samples: {task_logits.shape[0]}, Classes: {task_logits.shape[1]}")
    
    # Compute metrics before calibration
    probs_before = F.softmax(task_logits, dim=1).numpy()
    targets_np = task_targets.numpy()
    metrics_before = compute_calibration_metrics(probs_before, targets_np)
    
    logger.info(f"   Before calibration - NLL: {metrics_before['nll']:.4f}, "
               f"ECE: {metrics_before['ece']:.4f}, Brier: {metrics_before['brier_score']:.4f}")
    
    # Calibrate temperature
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.calibrate(task_logits, task_targets)
    
    # Compute metrics after calibration
    probs_after = F.softmax(task_logits / optimal_temp, dim=1).numpy()
    metrics_after = compute_calibration_metrics(probs_after, targets_np)
    
    logger.info(f"   Optimal temperature: {optimal_temp:.4f}")
    logger.info(f"   After calibration - NLL: {metrics_after['nll']:.4f}, "
               f"ECE: {metrics_after['ece']:.4f}, Brier: {metrics_after['brier_score']:.4f}")
    
    # Log improvement
    nll_improvement = metrics_before['nll'] - metrics_after['nll']
    ece_improvement = metrics_before['ece'] - metrics_after['ece']
    logger.info(f"   Improvement - NLL: {nll_improvement:+.4f}, ECE: {ece_improvement:+.4f}")
    
    return optimal_temp, metrics_before, metrics_after


def main():
    """Main temperature calibration function."""
    parser = argparse.ArgumentParser(description='Temperature Scaling Calibration for MVFouls')
    
    # Required arguments
    parser.add_argument('--logits-file', type=str, required=True,
                       help='Path to saved validation logits file (.pt)')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='temperature.json',
                       help='Output JSON file for temperature parameters (default: temperature.json)')
    parser.add_argument('--task', type=str, default='all',
                       help='Specific task to calibrate, or "all" for all tasks')
    parser.add_argument('--max-iter', type=int, default=100,
                       help='Maximum iterations for temperature optimization (default: 100)')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Convergence tolerance for optimization (default: 1e-6)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging(log_level)
    
    logger.info(f"🌡️ Starting temperature scaling calibration")
    logger.info(f"   Logits file: {args.logits_file}")
    logger.info(f"   Target task: {args.task}")
    
    try:
        # Load validation logits
        data = load_validation_logits(args.logits_file)
        
        # Determine if multi-task
        task_names = data.get('task_names', ['default'])
        multi_task = len(task_names) > 1 or task_names[0] != 'default'
        
        logger.info(f"🔍 Detected {'multi-task' if multi_task else 'single-task'} model")
        
        # Aggregate logits and targets
        logger.info("📊 Aggregating validation data...")
        logits_dict, targets_dict = aggregate_logits_and_targets(
            data['logits'], data['targets'], multi_task
        )
        
        # Filter tasks if specific task requested
        if args.task != 'all' and args.task in logits_dict:
            logits_dict = {args.task: logits_dict[args.task]}
            targets_dict = {args.task: targets_dict[args.task]}
            logger.info(f"🎯 Calibrating temperature only for task: {args.task}")
        elif args.task != 'all':
            available_tasks = list(logits_dict.keys())
            logger.error(f"❌ Task '{args.task}' not found. Available tasks: {available_tasks}")
            sys.exit(1)
        
        # Calibrate temperature for each task
        results = {
            'metadata': {
                'source_file': args.logits_file,
                'epoch': data.get('epoch', 'unknown'),
                'model_config': data.get('model_config', {}),
                'max_iter': args.max_iter,
                'tolerance': args.tolerance,
                'total_samples': sum(targets.shape[0] for targets in targets_dict.values())
            },
            'temperatures': {},
            'calibration_metrics': {}
        }
        
        for task_name in logits_dict.keys():
            logger.info(f"\n🔧 Processing task: {task_name}")
            
            optimal_temp, metrics_before, metrics_after = calibrate_task_temperature(
                logits_dict[task_name], 
                targets_dict[task_name],
                task_name
            )
            
            results['temperatures'][task_name] = optimal_temp
            results['calibration_metrics'][task_name] = {
                'before': metrics_before,
                'after': metrics_after,
                'improvement': {
                    'nll': metrics_before['nll'] - metrics_after['nll'],
                    'ece': metrics_before['ece'] - metrics_after['ece'],
                    'brier_score': metrics_before['brier_score'] - metrics_after['brier_score'],
                    'mce': metrics_before['mce'] - metrics_after['mce']
                }
            }
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results with proper serialization
        def json_serializer(obj):
            """Handle numpy types and other non-serializable objects."""
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return str(obj)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=json_serializer)
        
        logger.info(f"\n💾 Results saved to: {output_path}")
        logger.info("📋 Summary:")
        
        for task_name, temperature in results['temperatures'].items():
            metrics = results['calibration_metrics'][task_name]
            nll_improvement = metrics['improvement']['nll']
            ece_improvement = metrics['improvement']['ece']
            logger.info(f"   {task_name}: T={temperature:.4f}, "
                       f"NLL improvement: {nll_improvement:+.4f}, "
                       f"ECE improvement: {ece_improvement:+.4f}")
        
        logger.info("✅ Temperature calibration completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Temperature calibration failed: {e}")
        raise


if __name__ == '__main__':
    main() 