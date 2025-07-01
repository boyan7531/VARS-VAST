#!/usr/bin/env python3
"""
Compute Optimal Decision Thresholds for MVFouls Model

This script takes saved validation logits and computes optimal decision thresholds
for each class using various metrics like Youden index, F1 score, or precision/recall trade-offs.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, f1_score
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


def optimize_threshold_youden(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, Dict]:
    """
    Optimize threshold using Youden's J statistic (sensitivity + specificity - 1).
    
    Args:
        y_true: Binary true labels for one class (0/1)
        y_probs: Predicted probabilities for that class
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # Youden's J statistic = sensitivity + specificity - 1 = tpr - fpr
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    optimal_threshold = thresholds[best_idx]
    
    metrics = {
        'threshold': float(optimal_threshold),
        'sensitivity': float(tpr[best_idx]),
        'specificity': float(1 - fpr[best_idx]),
        'youden_j': float(j_scores[best_idx]),
        'auc': float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    return optimal_threshold, metrics


def optimize_threshold_f1(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, Dict]:
    """
    Optimize threshold using F1 score.
    
    Args:
        y_true: Binary true labels for one class (0/1)
        y_probs: Predicted probabilities for that class
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Compute F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Handle edge case where thresholds might be shorter than precision/recall arrays
    if len(thresholds) < len(f1_scores):
        f1_scores = f1_scores[:-1]  # Remove last F1 score to match threshold length
    
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    
    metrics = {
        'threshold': float(optimal_threshold),
        'precision': float(precision[best_idx]),
        'recall': float(recall[best_idx]),
        'f1_score': float(f1_scores[best_idx]),
        'auc': float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    return optimal_threshold, metrics


def optimize_threshold_recall_at_precision(y_true: np.ndarray, y_probs: np.ndarray, 
                                         target_precision: float = 0.9) -> Tuple[float, Dict]:
    """
    Optimize threshold to achieve target precision while maximizing recall.
    
    Args:
        y_true: Binary true labels for one class (0/1)
        y_probs: Predicted probabilities for that class
        target_precision: Target precision value (default: 0.9)
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Find thresholds that meet or exceed target precision
    valid_indices = np.where(precision >= target_precision)[0]
    
    if len(valid_indices) == 0:
        # If no threshold achieves target precision, use the one with highest precision
        best_idx = np.argmax(precision[:-1])  # Exclude last element (precision=1, recall=0)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        achieved_precision = precision[best_idx]
        achieved_recall = recall[best_idx]
    else:
        # Among valid thresholds, choose the one with highest recall
        best_valid_idx = valid_indices[np.argmax(recall[valid_indices])]
        optimal_threshold = thresholds[best_valid_idx] if best_valid_idx < len(thresholds) else 0.5
        achieved_precision = precision[best_valid_idx]
        achieved_recall = recall[best_valid_idx]
    
    metrics = {
        'threshold': float(optimal_threshold),
        'precision': float(achieved_precision),
        'recall': float(achieved_recall),
        'target_precision': float(target_precision),
        'f1_score': float(2 * achieved_precision * achieved_recall / (achieved_precision + achieved_recall + 1e-8)),
        'auc': float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    return optimal_threshold, metrics


def compute_task_thresholds(task_logits: torch.Tensor, task_targets: torch.Tensor, 
                          task_name: str, metric: str = 'youden',
                          target_precision: float = 0.9) -> Dict:
    """
    Compute optimal thresholds for all classes in a task using one-vs-rest approach.
    
    Args:
        task_logits: Logits tensor of shape (N, num_classes)
        task_targets: Target tensor of shape (N,) with class indices
        task_name: Name of the task
        metric: Metric to optimize ('youden', 'f1', 'recall@precision')
        target_precision: Target precision for 'recall@precision' metric
        
    Returns:
        Dict with threshold information for this task
    """
    logger = logging.getLogger(__name__)
    
    # Convert logits to probabilities
    probs = F.softmax(task_logits, dim=1).numpy()
    targets = task_targets.numpy()
    
    num_classes = probs.shape[1]
    logger.info(f"🎯 Computing thresholds for task '{task_name}' ({num_classes} classes)")
    
    thresholds = []
    class_metrics = {}
    
    for class_idx in range(num_classes):
        # Create binary labels: 1 if this class, 0 otherwise
        binary_targets = (targets == class_idx).astype(int)
        class_probs = probs[:, class_idx]
        
        # Skip if no positive samples for this class
        if np.sum(binary_targets) == 0:
            logger.warning(f"   Class {class_idx}: No positive samples, using default threshold 0.5")
            thresholds.append(0.5)
            class_metrics[f'class_{class_idx}'] = {
                'threshold': 0.5,
                'num_positive': 0,
                'num_negative': len(binary_targets),
                'metric_used': metric
            }
            continue
        
        # Optimize threshold based on chosen metric
        try:
            if metric == 'youden':
                optimal_threshold, metrics = optimize_threshold_youden(binary_targets, class_probs)
            elif metric == 'f1':
                optimal_threshold, metrics = optimize_threshold_f1(binary_targets, class_probs)
            elif metric.startswith('recall@precision'):
                # Parse target precision from metric string if provided
                if '=' in metric:
                    target_precision = float(metric.split('=')[1])
                optimal_threshold, metrics = optimize_threshold_recall_at_precision(
                    binary_targets, class_probs, target_precision
                )
            else:
                logger.warning(f"Unknown metric '{metric}', falling back to Youden")
                optimal_threshold, metrics = optimize_threshold_youden(binary_targets, class_probs)
            
            thresholds.append(optimal_threshold)
            
            # Add metadata
            metrics['num_positive'] = int(np.sum(binary_targets))
            metrics['num_negative'] = int(len(binary_targets) - np.sum(binary_targets))
            metrics['metric_used'] = metric
            
            class_metrics[f'class_{class_idx}'] = metrics
            
            logger.info(f"   Class {class_idx}: threshold={optimal_threshold:.4f}, "
                       f"pos={metrics['num_positive']}, neg={metrics['num_negative']}")
            
        except Exception as e:
            logger.error(f"   Class {class_idx}: Failed to optimize threshold: {e}")
            thresholds.append(0.5)  # Fallback
            class_metrics[f'class_{class_idx}'] = {
                'threshold': 0.5,
                'num_positive': int(np.sum(binary_targets)),
                'num_negative': int(len(binary_targets) - np.sum(binary_targets)),
                'metric_used': metric,
                'error': str(e)
            }
    
    return {
        'task_name': task_name,
        'thresholds': thresholds,
        'num_classes': num_classes,
        'metric_used': metric,
        'class_metrics': class_metrics,
        'total_samples': len(targets)
    }


def main():
    """Main threshold computation function."""
    parser = argparse.ArgumentParser(description='Compute Optimal Decision Thresholds')
    
    # Required arguments
    parser.add_argument('--logits-file', type=str, required=True,
                       help='Path to saved validation logits file (.pt)')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='thresholds.json',
                       help='Output JSON file for thresholds (default: thresholds.json)')
    parser.add_argument('--metric', type=str, default='youden',
                       choices=['youden', 'f1', 'recall@precision=0.9', 'recall@precision=0.8'],
                       help='Metric to optimize (default: youden)')
    parser.add_argument('--task', type=str, default='all',
                       help='Specific task to compute thresholds for, or "all" for all tasks')
    parser.add_argument('--target-precision', type=float, default=0.9,
                       help='Target precision for recall@precision metric (default: 0.9)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging(log_level)
    
    logger.info(f"🎯 Computing optimal decision thresholds")
    logger.info(f"   Logits file: {args.logits_file}")
    logger.info(f"   Metric: {args.metric}")
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
        
        # Compute thresholds for requested tasks
        results = {
            'metadata': {
                'source_file': args.logits_file,
                'epoch': data.get('epoch', 'unknown'),
                'metric_used': args.metric,
                'target_precision': args.target_precision if 'precision' in args.metric else None,
                'model_config': data.get('model_config', {}),
                'total_samples': sum(targets.shape[0] for targets in targets_dict.values())
            },
            'thresholds': {}
        }
        
        # Filter tasks if specific task requested
        if args.task != 'all' and args.task in logits_dict:
            logits_dict = {args.task: logits_dict[args.task]}
            targets_dict = {args.task: targets_dict[args.task]}
            logger.info(f"🎯 Computing thresholds only for task: {args.task}")
        elif args.task != 'all':
            available_tasks = list(logits_dict.keys())
            logger.error(f"❌ Task '{args.task}' not found. Available tasks: {available_tasks}")
            sys.exit(1)
        
        # Compute thresholds for each task
        for task_name in logits_dict.keys():
            logger.info(f"\n🔧 Processing task: {task_name}")
            
            task_result = compute_task_thresholds(
                logits_dict[task_name], 
                targets_dict[task_name],
                task_name,
                args.metric,
                args.target_precision
            )
            
            results['thresholds'][task_name] = task_result
        
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
        
        for task_name, task_result in results['thresholds'].items():
            thresholds = task_result['thresholds']
            logger.info(f"   {task_name}: {len(thresholds)} thresholds")
            logger.info(f"      Range: [{min(thresholds):.4f}, {max(thresholds):.4f}]")
        
        logger.info("✅ Threshold computation completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Threshold computation failed: {e}")
        raise


if __name__ == '__main__':
    main() 