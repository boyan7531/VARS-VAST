#!/usr/bin/env python3
"""Calculate comprehensive accuracy metrics for MVFouls model.

This script loads a trained model checkpoint, runs inference on a test dataset,
and computes detailed accuracy metrics including:
- Normal accuracy (top-1)
- Macro recall (average recall across all classes)
- Macro precision
- Macro F1-score
- Per-class precision, recall, F1
- Confusion matrices
- Class-wise accuracy breakdown

Supports both single-clip and bag-of-clips evaluation modes.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from dataset import MVFoulsDataset, bag_of_clips_collate_fn
from transforms import get_val_transforms
from utils import get_task_metadata, compute_task_metrics, format_metrics_table
from model.mvfouls_model import MVFoulsModel, build_multi_task_model


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def load_model(ckpt_path: Path, device: torch.device) -> MVFoulsModel:
    """Load trained model from checkpoint."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model from {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg = ckpt.get("config", {}).get("model_config", {})
    
    # Extract key parameters and avoid conflicts
    backbone_arch = cfg.get("backbone_arch", "mvitv2_s")
    
    # Create clean config without conflicting keys
    # These parameters are explicitly set by build_multi_task_model
    conflicting_keys = [
        'backbone_arch', 'backbone_pretrained', 'backbone_freeze_mode', 'backbone_checkpointing',
        'multi_task',  # This is always True for build_multi_task_model
    ]
    clean_cfg = {k: v for k, v in cfg.items() if k not in conflicting_keys}
    
    # Build model with same architecture as training
    model = build_multi_task_model(
        backbone_arch=backbone_arch,
        backbone_pretrained=False,  # Don't load pretrained weights
        backbone_freeze_mode="none",  # No freezing for inference
        backbone_checkpointing=False,  # Disable for inference
        **clean_cfg
    )
    
    # Load trained weights
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    
    logger.info(f"✅ Model loaded successfully!")
    logger.info(f"   Training epoch: {ckpt.get('epoch', 'unknown')}")
    logger.info(f"   Best metric: {ckpt.get('best_metric', 'unknown')}")
    
    return model


def collect_predictions_and_targets(
    model: MVFoulsModel,
    loader: DataLoader,
    device: torch.device,
    task_names: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Collect all predictions and ground truth targets."""
    logger = logging.getLogger(__name__)
    
    all_predictions = {task: [] for task in task_names}
    all_targets = {task: [] for task in task_names}
    
    logger.info(f"Collecting predictions for {len(loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Inference")):
            # Handle different batch formats
            if len(batch) == 2:
                # Standard format: (videos, targets)
                videos, targets = batch
            elif len(batch) == 4:
                # Bag-of-clips format: (videos, targets, clip_masks, num_clips)
                videos, targets, clip_masks, num_clips = batch
            else:
                logger.error(f"Unexpected batch format with {len(batch)} elements")
                continue
            
            # Move to device
            videos = videos.to(device)
            
            # Get model predictions
            logits_dict, _ = model(videos, return_dict=True)
            
            # Process each task
            for task_name in task_names:
                if task_name not in logits_dict:
                    logger.warning(f"Task {task_name} not found in model output")
                    continue
                
                # Get predictions (class indices)
                task_logits = logits_dict[task_name]
                task_preds = torch.argmax(task_logits, dim=1).cpu().numpy()
                
                # Get ground truth targets
                if isinstance(targets, dict):
                    task_targets = targets[task_name].cpu().numpy()
                else:
                    # Assume targets is a tensor with shape [batch_size, num_tasks]
                    task_idx = task_names.index(task_name)
                    task_targets = targets[:, task_idx].cpu().numpy()
                
                # Store predictions and targets
                all_predictions[task_name].extend(task_preds)
                all_targets[task_name].extend(task_targets)
    
    # Convert to numpy arrays
    for task_name in task_names:
        all_predictions[task_name] = np.array(all_predictions[task_name])
        all_targets[task_name] = np.array(all_targets[task_name])
        
        logger.info(f"Task {task_name}: {len(all_predictions[task_name])} samples collected")
    
    return all_predictions, all_targets


def compute_detailed_metrics(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    class_names: Dict[str, List[str]]
) -> Dict[str, Dict]:
    """Compute detailed accuracy metrics for each task."""
    logger = logging.getLogger(__name__)
    
    detailed_metrics = {}
    
    for task_name in predictions.keys():
        logger.info(f"\n📊 Computing metrics for task: {task_name}")
        
        task_preds = predictions[task_name]
        task_targets = targets[task_name]
        task_classes = class_names[task_name]
        
        # Define expected classes to handle missing classes in predictions/targets
        num_expected_classes = len(task_classes)
        labels = list(range(num_expected_classes))
        
        # Basic accuracy
        accuracy = accuracy_score(task_targets, task_preds)
        
        # Precision, Recall, F1 (macro and weighted)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            task_targets, task_preds, average='macro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            task_targets, task_preds, average='weighted', zero_division=0
        )
        
        # Per-class metrics - specify labels to handle missing classes
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            task_targets, task_preds, labels=labels, average=None, zero_division=0
        )
        
        # Confusion matrix - specify labels to handle missing classes
        conf_matrix = confusion_matrix(task_targets, task_preds, labels=labels)
        
        # Classification report - explicitly specify labels to handle missing classes
        
        class_report = classification_report(
            task_targets, task_preds, 
            labels=labels,
            target_names=task_classes,
            output_dict=True,
            zero_division=0
        )
        
        # Store all metrics
        detailed_metrics[task_name] = {
            'accuracy': accuracy,
            'macro_precision': precision_macro,
            'macro_recall': recall_macro,
            'macro_f1': f1_macro,
            'weighted_precision': precision_weighted,
            'weighted_recall': recall_weighted,
            'weighted_f1': f1_weighted,
            'per_class_precision': precision_per_class,
            'per_class_recall': recall_per_class,
            'per_class_f1': f1_per_class,
            'per_class_support': support_per_class,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'num_samples': len(task_targets),
            'num_classes': len(task_classes)
        }
        
        logger.info(f"   ✅ Accuracy: {accuracy:.4f}")
        logger.info(f"   ✅ Macro Recall: {recall_macro:.4f}")
        logger.info(f"   ✅ Macro Precision: {precision_macro:.4f}")
        logger.info(f"   ✅ Macro F1: {f1_macro:.4f}")
    
    return detailed_metrics


def print_comprehensive_results(
    detailed_metrics: Dict[str, Dict],
    class_names: Dict[str, List[str]]
):
    """Print comprehensive results in a formatted way."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE ACCURACY RESULTS")
    print("="*80)
    
    # Overall summary table
    print("\n📋 OVERALL SUMMARY")
    print("-" * 60)
    print(f"{'Task':<15} {'Accuracy':<10} {'Macro Recall':<12} {'Macro Prec':<12} {'Macro F1':<10}")
    print("-" * 60)
    
    overall_metrics = {}
    for task_name, metrics in detailed_metrics.items():
        print(f"{task_name:<15} {metrics['accuracy']:<10.4f} {metrics['macro_recall']:<12.4f} "
              f"{metrics['macro_precision']:<12.4f} {metrics['macro_f1']:<10.4f}")
        
        # Store for overall calculation
        overall_metrics[task_name] = {
            'accuracy': metrics['accuracy'],
            'macro_recall': metrics['macro_recall'],
            'macro_precision': metrics['macro_precision'],
            'macro_f1': metrics['macro_f1']
        }
    
    # Calculate cross-task averages
    avg_accuracy = np.mean([m['accuracy'] for m in overall_metrics.values()])
    avg_macro_recall = np.mean([m['macro_recall'] for m in overall_metrics.values()])
    avg_macro_precision = np.mean([m['macro_precision'] for m in overall_metrics.values()])
    avg_macro_f1 = np.mean([m['macro_f1'] for m in overall_metrics.values()])
    
    print("-" * 60)
    print(f"{'AVERAGE':<15} {avg_accuracy:<10.4f} {avg_macro_recall:<12.4f} "
          f"{avg_macro_precision:<12.4f} {avg_macro_f1:<10.4f}")
    print("-" * 60)
    
    # Detailed per-task results
    for task_name, metrics in detailed_metrics.items():
        print(f"\n📊 DETAILED RESULTS FOR {task_name.upper()}")
        print("-" * 50)
        
        task_classes = class_names[task_name]
        
        # Per-class breakdown
        print(f"\n🔍 Per-Class Breakdown:")
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        print("-" * 60)
        
        for i, class_name in enumerate(task_classes):
            if i < len(metrics['per_class_precision']):
                precision = metrics['per_class_precision'][i]
                recall = metrics['per_class_recall'][i]
                f1 = metrics['per_class_f1'][i]
                support = metrics['per_class_support'][i]
                
                print(f"{class_name[:19]:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
        
        # Confusion matrix
        print(f"\n🔢 Confusion Matrix:")
        conf_matrix = metrics['confusion_matrix']
        
        # Print header
        print("      ", end="")
        for i, class_name in enumerate(task_classes):
            print(f"{class_name[:8]:>8}", end="")
        print()
        
        # Print matrix rows
        for i, class_name in enumerate(task_classes):
            print(f"{class_name[:5]:>5} ", end="")
            for j in range(len(task_classes)):
                if i < conf_matrix.shape[0] and j < conf_matrix.shape[1]:
                    print(f"{conf_matrix[i, j]:>8}", end="")
                else:
                    print(f"{'0':>8}", end="")
            print()
        
        # Additional statistics
        print(f"\n📈 Additional Statistics:")
        print(f"   Total samples: {metrics['num_samples']}")
        print(f"   Number of classes: {metrics['num_classes']}")
        print(f"   Weighted Precision: {metrics['weighted_precision']:.4f}")
        print(f"   Weighted Recall: {metrics['weighted_recall']:.4f}")
        print(f"   Weighted F1: {metrics['weighted_f1']:.4f}")


def save_results_to_json(
    detailed_metrics: Dict[str, Dict],
    output_path: Path
):
    """Save results to JSON file."""
    logger = logging.getLogger(__name__)
    
    # Convert numpy arrays to lists for JSON serialization
    json_metrics = {}
    for task_name, metrics in detailed_metrics.items():
        json_metrics[task_name] = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                json_metrics[task_name][key] = value.tolist()
            elif isinstance(value, np.integer):
                json_metrics[task_name][key] = int(value)
            elif isinstance(value, np.floating):
                json_metrics[task_name][key] = float(value)
            else:
                json_metrics[task_name][key] = value
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    logger.info(f"📁 Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calculate comprehensive accuracy metrics for MVFouls model")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to model checkpoint (.pth)")
    parser.add_argument("--test-dir", required=True, type=str, help="Directory with test split videos")
    parser.add_argument("--test-annotations", required=True, type=str, help="Path to test annotations.json")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--bag-of-clips", action="store_true", help="Enable bag-of-clips mode")
    parser.add_argument("--max-clips-per-action", type=int, default=8, help="Max clips per action")
    parser.add_argument("--min-clips-per-action", type=int, default=1, help="Min clips per action")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"], help="Dataset split to evaluate")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(Path(args.checkpoint), device)
    
    # Get task metadata
    metadata = get_task_metadata()
    task_names = metadata["task_names"]
    class_names = metadata["class_names"]
    
    logger.info(f"📋 Evaluating tasks: {task_names}")
    
    # Build dataset
    transforms = get_val_transforms()
    
    # Determine root directory from test_dir
    test_dir_path = Path(args.test_dir)
    if test_dir_path.name.endswith('_720p'):
        root_dir = test_dir_path.parent
    else:
        root_dir = test_dir_path
    
    dataset = MVFoulsDataset(
        root_dir=str(root_dir),
        split=args.split,
        transform=transforms,
        bag_of_clips=args.bag_of_clips,
        max_clips_per_action=args.max_clips_per_action,
        min_clips_per_action=args.min_clips_per_action,
        clip_sampling_strategy="uniform",
        load_annotations=True,
        cache_mode="none",
    )
    
    # Build dataloader
    collate_fn = bag_of_clips_collate_fn if args.bag_of_clips else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    logger.info(f"📊 Dataset: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Collect predictions and targets
    predictions, targets = collect_predictions_and_targets(
        model, dataloader, device, task_names
    )
    
    # Compute detailed metrics
    detailed_metrics = compute_detailed_metrics(
        predictions, targets, class_names
    )
    
    # Print results
    print_comprehensive_results(detailed_metrics, class_names)
    
    # Save to JSON if requested
    if args.output_json:
        save_results_to_json(detailed_metrics, Path(args.output_json))
    
    logger.info("🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
