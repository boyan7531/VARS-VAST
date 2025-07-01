#!/usr/bin/env python3
"""
Threshold-based Prediction Utilities for MVFouls

This module provides utilities for making predictions using per-class decision thresholds
instead of standard argmax prediction. This can improve performance on imbalanced datasets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn.functional as F
import numpy as np


def load_thresholds(thresholds_file: str) -> Dict[str, Any]:
    """
    Load pre-computed thresholds from JSON file.
    
    Args:
        thresholds_file: Path to thresholds JSON file
        
    Returns:
        Dict containing threshold data
    """
    logger = logging.getLogger(__name__)
    
    if not Path(thresholds_file).exists():
        raise FileNotFoundError(f"Thresholds file not found: {thresholds_file}")
    
    with open(thresholds_file, 'r') as f:
        data = json.load(f)
    
    logger.info(f"📂 Loaded thresholds from: {thresholds_file}")
    if 'metadata' in data:
        metadata = data['metadata']
        logger.info(f"   Source epoch: {metadata.get('epoch', 'unknown')}")
        logger.info(f"   Metric used: {metadata.get('metric_used', 'unknown')}")
        logger.info(f"   Total samples: {metadata.get('total_samples', 'unknown')}")
    
    return data


def predict_with_thresholds_single_task(
    probs: torch.Tensor, 
    thresholds: Union[List[float], torch.Tensor],
    fallback_strategy: str = 'argmax'
) -> torch.Tensor:
    """
    Make predictions using per-class thresholds for a single task.
    
    Args:
        probs: Probability tensor of shape (batch_size, num_classes)
        thresholds: Threshold values for each class, shape (num_classes,)
        fallback_strategy: What to do when no class exceeds threshold
                          ('argmax', 'none', 'highest_prob')
        
    Returns:
        Predictions tensor of shape (batch_size,)
    """
    if isinstance(thresholds, list):
        thresholds = torch.tensor(thresholds, device=probs.device, dtype=probs.dtype)
    
    batch_size, num_classes = probs.shape
    
    # Check which classes exceed their thresholds
    above_threshold = probs >= thresholds.unsqueeze(0)  # (batch_size, num_classes)
    
    predictions = torch.zeros(batch_size, dtype=torch.long, device=probs.device)
    
    for i in range(batch_size):
        candidates = torch.where(above_threshold[i])[0]
        
        if len(candidates) > 0:
            # Among candidates exceeding threshold, choose the one with highest probability
            candidate_probs = probs[i, candidates]
            best_candidate_idx = torch.argmax(candidate_probs)
            predictions[i] = candidates[best_candidate_idx]
        else:
            # No class exceeds threshold, apply fallback strategy
            if fallback_strategy == 'argmax':
                predictions[i] = torch.argmax(probs[i])
            elif fallback_strategy == 'none':
                predictions[i] = -1  # Special value indicating no prediction
            elif fallback_strategy == 'highest_prob':
                predictions[i] = torch.argmax(probs[i])  # Same as argmax
            else:
                # Default to argmax
                predictions[i] = torch.argmax(probs[i])
    
    return predictions


def predict_with_thresholds_multi_task(
    probs_dict: Dict[str, torch.Tensor],
    thresholds_dict: Dict[str, List[float]],
    fallback_strategy: str = 'argmax'
) -> Dict[str, torch.Tensor]:
    """
    Make predictions using per-class thresholds for multiple tasks.
    
    Args:
        probs_dict: Dict mapping task names to probability tensors
        thresholds_dict: Dict mapping task names to threshold lists
        fallback_strategy: What to do when no class exceeds threshold
        
    Returns:
        Dict mapping task names to prediction tensors
    """
    predictions_dict = {}
    
    for task_name, probs in probs_dict.items():
        if task_name in thresholds_dict:
            thresholds = thresholds_dict[task_name]
            predictions_dict[task_name] = predict_with_thresholds_single_task(
                probs, thresholds, fallback_strategy
            )
        else:
            # No thresholds available for this task, use argmax
            predictions_dict[task_name] = torch.argmax(probs, dim=1)
    
    return predictions_dict


def predict_with_thresholds(
    probs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    thresholds: Union[List[float], torch.Tensor, Dict[str, List[float]]],
    fallback_strategy: str = 'argmax'
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Unified function for threshold-based prediction (single or multi-task).
    
    Args:
        probs: Probabilities (tensor for single-task, dict for multi-task)
        thresholds: Thresholds (list/tensor for single-task, dict for multi-task)
        fallback_strategy: What to do when no class exceeds threshold
        
    Returns:
        Predictions (tensor for single-task, dict for multi-task)
    """
    if isinstance(probs, dict) and isinstance(thresholds, dict):
        # Multi-task case
        return predict_with_thresholds_multi_task(probs, thresholds, fallback_strategy)
    elif isinstance(probs, torch.Tensor) and not isinstance(thresholds, dict):
        # Single-task case
        return predict_with_thresholds_single_task(probs, thresholds, fallback_strategy)
    else:
        raise ValueError("Inconsistent input types: probs and thresholds must both be "
                        "single-task (tensor/list) or multi-task (dict)")


def extract_task_thresholds(thresholds_data: Dict[str, Any], task_name: str) -> Optional[List[float]]:
    """
    Extract threshold list for a specific task from loaded threshold data.
    
    Args:
        thresholds_data: Loaded threshold data from load_thresholds()
        task_name: Name of the task to extract thresholds for
        
    Returns:
        List of thresholds for the task, or None if not found
    """
    if 'thresholds' not in thresholds_data:
        return None
    
    task_data = thresholds_data['thresholds'].get(task_name)
    if task_data is None:
        return None
    
    return task_data.get('thresholds')


def extract_all_task_thresholds(thresholds_data: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Extract thresholds for all tasks from loaded threshold data.
    
    Args:
        thresholds_data: Loaded threshold data from load_thresholds()
        
    Returns:
        Dict mapping task names to threshold lists
    """
    if 'thresholds' not in thresholds_data:
        return {}
    
    task_thresholds = {}
    for task_name, task_data in thresholds_data['thresholds'].items():
        if 'thresholds' in task_data:
            task_thresholds[task_name] = task_data['thresholds']
    
    return task_thresholds


def analyze_threshold_usage(
    probs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    thresholds: Union[List[float], torch.Tensor, Dict[str, List[float]]],
    return_details: bool = False
) -> Dict[str, Any]:
    """
    Analyze how often thresholds vs fallback predictions are used.
    
    Args:
        probs: Probabilities (tensor for single-task, dict for multi-task)
        thresholds: Thresholds (list/tensor for single-task, dict for multi-task)
        return_details: Whether to return detailed per-sample analysis
        
    Returns:
        Dict with usage statistics
    """
    stats = {}
    
    if isinstance(probs, dict) and isinstance(thresholds, dict):
        # Multi-task analysis
        for task_name, task_probs in probs.items():
            if task_name in thresholds:
                task_thresholds = torch.tensor(thresholds[task_name], 
                                             device=task_probs.device, dtype=task_probs.dtype)
                above_threshold = task_probs >= task_thresholds.unsqueeze(0)
                any_above = above_threshold.any(dim=1)
                
                stats[task_name] = {
                    'total_samples': len(task_probs),
                    'threshold_used': int(any_above.sum()),
                    'fallback_used': int((~any_above).sum()),
                    'threshold_rate': float(any_above.float().mean())
                }
    else:
        # Single-task analysis
        if isinstance(thresholds, list):
            thresholds = torch.tensor(thresholds, device=probs.device, dtype=probs.dtype)
        
        above_threshold = probs >= thresholds.unsqueeze(0)
        any_above = above_threshold.any(dim=1)
        
        stats['single_task'] = {
            'total_samples': len(probs),
            'threshold_used': int(any_above.sum()),
            'fallback_used': int((~any_above).sum()),
            'threshold_rate': float(any_above.float().mean())
        }
    
    return stats


def create_threshold_summary(thresholds_data: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of threshold data.
    
    Args:
        thresholds_data: Loaded threshold data from load_thresholds()
        
    Returns:
        Formatted summary string
    """
    lines = ["📊 THRESHOLD SUMMARY", "=" * 50]
    
    # Metadata
    if 'metadata' in thresholds_data:
        metadata = thresholds_data['metadata']
        lines.append(f"Source: {metadata.get('source_file', 'unknown')}")
        lines.append(f"Epoch: {metadata.get('epoch', 'unknown')}")
        lines.append(f"Metric: {metadata.get('metric_used', 'unknown')}")
        lines.append(f"Total samples: {metadata.get('total_samples', 'unknown')}")
        lines.append("")
    
    # Task-specific thresholds
    if 'thresholds' in thresholds_data:
        for task_name, task_data in thresholds_data['thresholds'].items():
            lines.append(f"🎯 Task: {task_name}")
            lines.append(f"   Classes: {task_data.get('num_classes', 'unknown')}")
            
            thresholds = task_data.get('thresholds', [])
            if thresholds:
                lines.append(f"   Thresholds: {len(thresholds)} values")
                lines.append(f"   Range: [{min(thresholds):.4f}, {max(thresholds):.4f}]")
                lines.append(f"   Mean: {np.mean(thresholds):.4f}")
            
            lines.append("")
    
    return "\n".join(lines)


# Calibration utilities
def apply_temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling to logits before converting to probabilities.
    
    Args:
        logits: Raw logits tensor
        temperature: Temperature parameter (T > 1 makes predictions softer)
        
    Returns:
        Temperature-scaled logits
    """
    return logits / temperature


def load_temperature_config(temperature_file: str) -> Dict[str, float]:
    """
    Load temperature scaling parameters from JSON file.
    
    Args:
        temperature_file: Path to temperature configuration file
        
    Returns:
        Dict mapping task names to temperature values
    """
    if not Path(temperature_file).exists():
        raise FileNotFoundError(f"Temperature file not found: {temperature_file}")
    
    with open(temperature_file, 'r') as f:
        data = json.load(f)
    
    return data.get('temperatures', {})


# Test utilities
def test_threshold_predictions():
    """Test threshold prediction functions with synthetic data."""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing threshold prediction functions...")
    
    # Test single-task
    probs = torch.tensor([[0.1, 0.3, 0.6], [0.8, 0.1, 0.1], [0.4, 0.4, 0.2]])
    thresholds = [0.5, 0.5, 0.5]
    
    preds = predict_with_thresholds_single_task(probs, thresholds)
    logger.info(f"Single-task test: {preds}")
    
    # Test multi-task
    probs_dict = {
        'task_a': torch.tensor([[0.1, 0.9], [0.6, 0.4]]),
        'task_b': torch.tensor([[0.3, 0.7], [0.8, 0.2]])
    }
    thresholds_dict = {
        'task_a': [0.5, 0.5],
        'task_b': [0.6, 0.6]
    }
    
    preds_dict = predict_with_thresholds_multi_task(probs_dict, thresholds_dict)
    logger.info(f"Multi-task test: {preds_dict}")
    
    # Test usage analysis
    stats = analyze_threshold_usage(probs_dict, thresholds_dict)
    logger.info(f"Usage stats: {stats}")
    
    logger.info("✅ All tests passed!")


if __name__ == '__main__':
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    test_threshold_predictions() 