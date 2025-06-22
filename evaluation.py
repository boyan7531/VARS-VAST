#!/usr/bin/env python3
"""
MVFouls Model Evaluation Script

This script loads a trained MVFouls model from checkpoint and evaluates it on the test dataset.
Supports both single-task and multi-task models with comprehensive metrics reporting.

Usage:
    python evaluation.py --checkpoint path/to/model.pth --root-dir mvfouls --split test
    python evaluation.py --checkpoint outputs/best_model.pth --root-dir mvfouls --split test --multi-task
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model.mvfouls_model import MVFoulsModel, build_multi_task_model, build_single_task_model
from dataset import MVFoulsDataset
from transforms import get_video_transforms
from utils import get_task_metadata, compute_task_metrics, format_metrics_table


def setup_logging(level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> MVFoulsModel:
    """Load model from checkpoint file."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    try:
        # Try with weights_only=True first (PyTorch 2.6+ default)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        logger.warning(f"Failed to load with weights_only=True: {e}")
        logger.info("Retrying with weights_only=False (trusted source)")
        # Fall back to weights_only=False for compatibility
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model configuration from checkpoint metadata
    if 'metadata' in checkpoint and 'args' in checkpoint['metadata']:
        args = checkpoint['metadata']['args']
        multi_task = args.get('multi_task', False)
        num_classes = args.get('num_classes', 2)
    else:
        # Try to infer from model state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Check if multi-task based on head structure
        multi_task = any('task_heads' in key for key in state_dict.keys())
        num_classes = 2  # Default
        
        logger.warning("No metadata found in checkpoint. Inferring model configuration...")
        logger.info(f"Inferred multi_task: {multi_task}")
    
    # Create model
    if multi_task:
        model = build_multi_task_model(
            backbone_pretrained=False,  # We'll load weights from checkpoint
            backbone_freeze_mode='none'
        )
    else:
        model = build_single_task_model(
            num_classes=num_classes,
            backbone_pretrained=False,
            backbone_freeze_mode='none'
        )
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully!")
    logger.info(f"Multi-task: {model.multi_task}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def create_test_dataloader(root_dir: str, split: str, batch_size: int, num_workers: int, 
                          max_frames: int = 32, image_size: int = 224) -> DataLoader:
    """Create test dataloader."""
    logger = logging.getLogger(__name__)
    
    # Create transforms (no augmentation for evaluation)
    transforms = get_video_transforms(image_size=image_size, augment_train=False)
    
    # Create dataset
    logger.info(f"Creating {split} dataset...")
    dataset = MVFoulsDataset(
        root_dir=root_dir,
        split=split,
        transform=transforms['val'],  # Use validation transforms (no augmentation)
        load_annotations=True,
        num_frames=max_frames
    )
    
    logger.info(f"{split.capitalize()} dataset created: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader, dataset


def evaluate_single_task(model: MVFoulsModel, dataloader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """Evaluate single-task model."""
    logger = logging.getLogger(__name__)
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    total_loss = 0.0
    num_batches = 0
    
    logger.info("Running single-task evaluation...")
    
    with torch.no_grad():
        for batch_idx, (videos, targets) in enumerate(dataloader):
            videos = videos.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits, extras = model(videos, return_dict=False)
            
            # Compute loss
            loss = model.compute_loss(logits, targets)
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions and probabilities
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Compute metrics
    accuracy = (all_predictions == all_targets).mean()
    avg_loss = total_loss / num_batches
    
    # Classification report
    class_names = [f"Class_{i}" for i in range(all_probabilities.shape[1])]
    clf_report = classification_report(
        all_targets, all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'classification_report': clf_report,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'num_samples': len(all_predictions)
    }
    
    logger.info(f"Single-task evaluation completed!")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Loss: {avg_loss:.4f}")
    
    return results


def evaluate_multi_task(model: MVFoulsModel, dataloader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """Evaluate multi-task model."""
    logger = logging.getLogger(__name__)
    
    model.eval()
    all_predictions = {}
    all_targets = {}
    all_probabilities = {}
    total_loss = 0.0
    num_batches = 0
    
    # Get task names
    if hasattr(model.head, 'task_names'):
        task_names = model.head.task_names
    else:
        metadata = get_task_metadata()
        task_names = metadata['task_names']
    
    # Initialize storage
    for task_name in task_names:
        all_predictions[task_name] = []
        all_targets[task_name] = []
        all_probabilities[task_name] = []
    
    logger.info("Running multi-task evaluation...")
    logger.info(f"Tasks: {task_names}")
    
    with torch.no_grad():
        for batch_idx, (videos, targets) in enumerate(dataloader):
            videos = videos.to(device)
            
            # Handle target format
            if isinstance(targets, dict):
                targets = {k: v.to(device) for k, v in targets.items()}
            else:
                # Convert tensor to dict format
                targets = targets.to(device)
                targets_dict = {}
                for i, task_name in enumerate(task_names):
                    if i < targets.shape[1]:
                        targets_dict[task_name] = targets[:, i]
                    else:
                        targets_dict[task_name] = torch.zeros(targets.shape[0], dtype=torch.long, device=device)
                targets = targets_dict
            
            # Forward pass
            logits_dict, extras = model(videos, return_dict=True)
            
            # Compute loss
            loss_dict = model.compute_loss(logits_dict, targets, return_dict=True)
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
            
            # Process each task
            for task_name in task_names:
                if task_name in logits_dict and task_name in targets:
                    task_logits = logits_dict[task_name]
                    task_targets = targets[task_name]
                    
                    # Get predictions and probabilities
                    task_probabilities = F.softmax(task_logits, dim=1)
                    task_predictions = torch.argmax(task_logits, dim=1)
                    
                    # Store results
                    all_predictions[task_name].extend(task_predictions.cpu().numpy())
                    all_targets[task_name].extend(task_targets.cpu().numpy())
                    all_probabilities[task_name].extend(task_probabilities.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Convert to numpy arrays and compute metrics
    task_results = {}
    overall_accuracy = []
    
    for task_name in task_names:
        if all_predictions[task_name]:  # Check if task has data
            predictions = np.array(all_predictions[task_name])
            targets = np.array(all_targets[task_name])
            probabilities = np.array(all_probabilities[task_name])
            
            # Compute metrics
            accuracy = (predictions == targets).mean()
            overall_accuracy.append(accuracy)
            
            # Classification report
            num_classes = probabilities.shape[1]
            class_names = [f"{task_name}_class_{i}" for i in range(num_classes)]
            
            try:
                clf_report = classification_report(
                    targets, predictions,
                    target_names=class_names,
                    output_dict=True,
                    zero_division=0
                )
            except:
                clf_report = {}
            
            # Confusion matrix
            try:
                conf_matrix = confusion_matrix(targets, predictions)
            except:
                conf_matrix = np.array([[0]])
            
            task_results[task_name] = {
                'accuracy': accuracy,
                'classification_report': clf_report,
                'confusion_matrix': conf_matrix,
                'predictions': predictions,
                'targets': targets,
                'probabilities': probabilities,
                'num_samples': len(predictions)
            }
    
    # Overall metrics
    avg_loss = total_loss / num_batches
    overall_acc = np.mean(overall_accuracy) if overall_accuracy else 0.0
    
    # Use utils function if available
    try:
        if compute_task_metrics is not None:
            # Convert to tensor format for utils function
            logits_tensor = {}
            targets_tensor = {}
            for task_name in task_names:
                if task_name in all_predictions and all_predictions[task_name]:
                    logits_tensor[task_name] = torch.from_numpy(all_probabilities[task_name])
                    targets_tensor[task_name] = torch.from_numpy(all_targets[task_name])
            
            if logits_tensor:
                detailed_metrics = compute_task_metrics(logits_tensor, targets_tensor, list(logits_tensor.keys()))
                
                # Format metrics table
                if format_metrics_table is not None:
                    metrics_table = format_metrics_table(detailed_metrics)
                else:
                    metrics_table = None
            else:
                detailed_metrics = {}
                metrics_table = None
        else:
            detailed_metrics = {}
            metrics_table = None
    except Exception as e:
        logger.warning(f"Could not compute detailed metrics: {e}")
        detailed_metrics = {}
        metrics_table = None
    
    results = {
        'overall_accuracy': overall_acc,
        'loss': avg_loss,
        'task_results': task_results,
        'detailed_metrics': detailed_metrics,
        'metrics_table': metrics_table,
        'task_names': task_names,
        'num_samples': len(all_predictions[task_names[0]]) if task_names and all_predictions[task_names[0]] else 0
    }
    
    logger.info(f"Multi-task evaluation completed!")
    logger.info(f"Overall accuracy: {overall_acc:.4f}")
    logger.info(f"Loss: {avg_loss:.4f}")
    
    # Print per-task accuracies
    for task_name, task_result in task_results.items():
        logger.info(f"  {task_name}: {task_result['accuracy']:.4f}")
    
    return results


def save_results(results: Dict[str, Any], output_dir: Path, model_multi_task: bool):
    """Save evaluation results to files."""
    logger = logging.getLogger(__name__)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if model_multi_task:
        # Save multi-task results
        summary = {
            'overall_accuracy': results['overall_accuracy'],
            'loss': results['loss'],
            'num_samples': results['num_samples'],
            'task_accuracies': {
                task_name: task_result['accuracy'] 
                for task_name, task_result in results['task_results'].items()
            }
        }
        
        # Save summary
        with open(output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_results = {}
        for task_name, task_result in results['task_results'].items():
            detailed_results[task_name] = {
                'accuracy': task_result['accuracy'],
                'num_samples': task_result['num_samples'],
                'classification_report': task_result['classification_report']
            }
        
        with open(output_dir / 'detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save metrics table if available
        if results.get('metrics_table'):
            with open(output_dir / 'metrics_table.txt', 'w') as f:
                f.write(results['metrics_table'])
        
        # Save confusion matrices
        for task_name, task_result in results['task_results'].items():
            conf_matrix = task_result['confusion_matrix']
            if conf_matrix.size > 1:
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {task_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(output_dir / f'confusion_matrix_{task_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    else:
        # Save single-task results
        summary = {
            'accuracy': results['accuracy'],
            'loss': results['loss'],
            'num_samples': results['num_samples']
        }
        
        # Save summary
        with open(output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save classification report
        with open(output_dir / 'classification_report.json', 'w') as f:
            json.dump(results['classification_report'], f, indent=2)
        
        # Save confusion matrix plot
        conf_matrix = results['confusion_matrix']
        if conf_matrix.size > 1:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    logger.info(f"Results saved to: {output_dir}")


def print_results(results: Dict[str, Any], model_multi_task: bool):
    """Print evaluation results to console."""
    print("\n" + "="*80)
    print("🎯 EVALUATION RESULTS")
    print("="*80)
    
    if model_multi_task:
        print(f"📊 Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"📉 Loss: {results['loss']:.4f}")
        print(f"📝 Total Samples: {results['num_samples']}")
        print(f"🎯 Tasks: {len(results['task_results'])}")
        
        print("\n📋 Per-Task Results:")
        print("-" * 60)
        for task_name, task_result in results['task_results'].items():
            print(f"  {task_name:20s}: {task_result['accuracy']:.4f} ({task_result['num_samples']} samples)")
        
        # Print metrics table if available
        if results.get('metrics_table'):
            print("\n📊 Detailed Metrics:")
            print("-" * 60)
            print(results['metrics_table'])
    
    else:
        print(f"📊 Accuracy: {results['accuracy']:.4f}")
        print(f"📉 Loss: {results['loss']:.4f}")
        print(f"📝 Total Samples: {results['num_samples']}")
        
        # Print classification report summary
        if 'classification_report' in results:
            clf_report = results['classification_report']
            if 'macro avg' in clf_report:
                macro_avg = clf_report['macro avg']
                print(f"\n📊 Macro Average:")
                print(f"  Precision: {macro_avg['precision']:.4f}")
                print(f"  Recall: {macro_avg['recall']:.4f}")
                print(f"  F1-Score: {macro_avg['f1-score']:.4f}")
    
    print("="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate MVFouls Model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint file')
    parser.add_argument('--root-dir', type=str, required=True,
                        help='Path to MVFouls root directory')
    
    # Data arguments
    parser.add_argument('--split', type=str, default='test', choices=['test', 'valid', 'train'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max-frames', type=int, default=32,
                        help='Maximum number of frames per video')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save individual predictions to file')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging(log_level)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Evaluating on {args.split} split")
    
    try:
        # Load model
        model = load_model_from_checkpoint(args.checkpoint, device)
        
        # Create dataloader
        dataloader, dataset = create_test_dataloader(
            root_dir=args.root_dir,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_frames=args.max_frames,
            image_size=args.image_size
        )
        
        # Run evaluation
        if model.multi_task:
            results = evaluate_multi_task(model, dataloader, device)
        else:
            results = evaluate_single_task(model, dataloader, device)
        
        # Print results
        print_results(results, model.multi_task)
        
        # Save results
        output_dir = Path(args.output_dir)
        save_results(results, output_dir, model.multi_task)
        
        # Save predictions if requested
        if args.save_predictions:
            if model.multi_task:
                # Save predictions for each task
                for task_name, task_result in results['task_results'].items():
                    pred_df = pd.DataFrame({
                        'predictions': task_result['predictions'],
                        'targets': task_result['targets']
                    })
                    pred_df.to_csv(output_dir / f'predictions_{task_name}.csv', index=False)
            else:
                pred_df = pd.DataFrame({
                    'predictions': results['predictions'],
                    'targets': results['targets']
                })
                pred_df.to_csv(output_dir / 'predictions.csv', index=False)
            
            logger.info("Predictions saved to CSV files")
        
        logger.info("Evaluation completed successfully! 🎉")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()
