#!/usr/bin/env python3
"""
MVFouls Training Script with Proper Class Imbalance Handling

This script addresses the severe class imbalance in MVFouls dataset by:
1. Computing proper class weights for each task
2. Using focal loss with effective number weighting
3. Implementing task-specific loss strategies
4. Monitoring per-class metrics during training
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model.mvfouls_model import MVFoulsModel, build_multi_task_model, build_single_task_model
from dataset import MVFoulsDataset
from transforms import get_video_transforms
from training_utils import MultiTaskTrainer
from utils import (
    get_task_metadata, 
    compute_class_weights, 
    get_recommended_loss_config,
    get_task_class_weights,
    analyze_class_distribution
)


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


def analyze_dataset_imbalance(dataset: MVFoulsDataset) -> Dict[str, Dict]:
    """Analyze class imbalance in the dataset and recommend solutions."""
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Analyzing dataset class imbalance...")
    
    # Get dataset statistics
    stats = dataset.get_task_statistics()
    
    # Get task metadata for expected number of classes
    metadata = get_task_metadata()
    
    recommendations = {}
    
    for task_name, task_stats in stats.items():
        class_counts = task_stats['class_counts']
        expected_num_classes = len(metadata['class_names'][task_name])
        
        # Ensure class_counts has the right length (pad with zeros if needed)
        if len(class_counts) < expected_num_classes:
            class_counts = class_counts + [0] * (expected_num_classes - len(class_counts))
        elif len(class_counts) > expected_num_classes:
            # This shouldn't happen, but just in case
            class_counts = class_counts[:expected_num_classes]
            
        total_samples = sum(class_counts)
        
        if total_samples == 0:
            continue
            
        # Calculate imbalance ratio
        max_count = max(class_counts)
        min_count = min([c for c in class_counts if c > 0]) if any(c > 0 for c in class_counts) else 1
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Create dummy labels for weight calculation, ensuring all classes are represented
        dummy_labels = []
        for class_idx, count in enumerate(class_counts):
            if count > 0:
                dummy_labels.extend([class_idx] * count)
            else:
                # Add at least one dummy sample for missing classes to ensure proper weight calculation
                dummy_labels.append(class_idx)
        
        config = get_recommended_loss_config(dummy_labels, num_classes=expected_num_classes, severity_threshold=20.0)
        
        # Ensure class weights have the correct length
        if config['class_weights'] is not None:
            weights = config['class_weights']
            if len(weights) != expected_num_classes:
                # Fix weight tensor to have correct number of classes
                if len(weights) < expected_num_classes:
                    # Pad with mean weight for missing classes
                    mean_weight = float(torch.mean(weights))
                    padding = [mean_weight] * (expected_num_classes - len(weights))
                    weights = torch.cat([weights, torch.tensor(padding, dtype=torch.float32)])
                else:
                    # Truncate if too long (shouldn't happen)
                    weights = weights[:expected_num_classes]
            
            # Scale down weights to be less aggressive (prevent overcompensation)
            # Use square root to reduce the strength while maintaining relative differences
            weights = torch.sqrt(weights)
            # Normalize to have mean of 1.0
            weights = weights / weights.mean()
            config['class_weights'] = weights
        
        recommendations[task_name] = {
            'imbalance_ratio': imbalance_ratio,
            'total_samples': total_samples,
            'class_counts': class_counts,
            'class_names': task_stats['class_names'],
            'recommended_config': config,
            'severity': 'SEVERE' if imbalance_ratio > 10 else 'MODERATE' if imbalance_ratio > 3 else 'MILD'
        }
        
        logger.info(f"📊 {task_name}:")
        logger.info(f"   Imbalance ratio: {imbalance_ratio:.1f}:1 ({recommendations[task_name]['severity']})")
        logger.info(f"   Recommendation: {config['recommendation']}")
        
        # Show class distribution
        for class_idx, (class_name, count) in enumerate(zip(task_stats['class_names'], class_counts)):
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            logger.info(f"     {class_name}: {count} ({percentage:.1f}%)")
    
    return recommendations


def create_balanced_model(
    multi_task: bool = True,
    imbalance_analysis: Optional[Dict] = None,
    **model_kwargs
) -> MVFoulsModel:
    """Create a model with proper class imbalance handling."""
    logger = logging.getLogger(__name__)
    
    if multi_task:
        # Get task metadata
        metadata = get_task_metadata()
        task_names = metadata['task_names']
        
        # Prepare task-specific configurations
        task_weights = {}
        loss_types_per_task = []
        
        for task_name in task_names:
            if imbalance_analysis and task_name in imbalance_analysis:
                analysis = imbalance_analysis[task_name]
                config = analysis['recommended_config']
                
                # Use recommended class weights
                if config['class_weights'] is not None:
                    # Ensure weights are tensors and will be moved to device later
                    weights = config['class_weights']
                    if not isinstance(weights, torch.Tensor):
                        weights = torch.tensor(weights, dtype=torch.float32)
                    task_weights[task_name] = weights
                
                # Use recommended loss type
                loss_types_per_task.append(config['loss_type'])
                
                logger.info(f"🎯 {task_name}: {config['loss_type']} loss, "
                          f"weights: {config['class_weights'] is not None}")
            else:
                # Default to focal loss for unknown tasks
                loss_types_per_task.append('focal')
        
        # Create multi-task model with balanced configuration
        model = build_multi_task_model(
            backbone_pretrained=True,
            backbone_freeze_mode='gradual',
            loss_types_per_task=loss_types_per_task,
            class_weights=task_weights,
            **model_kwargs
        )
        
    else:
        # Single-task model with class weights
        class_weights = None
        if imbalance_analysis and 'offence' in imbalance_analysis:
            config = imbalance_analysis['offence']['recommended_config']
            if config['class_weights'] is not None:
                weights = config['class_weights']
                if not isinstance(weights, torch.Tensor):
                    weights = torch.tensor(weights, dtype=torch.float32)
                class_weights = weights
            loss_type = config['loss_type']
        else:
            loss_type = 'focal'
        
        # Filter out conflicting kwargs
        filtered_kwargs = {k: v for k, v in model_kwargs.items() 
                          if k not in ['head_loss_type', 'head_label_smoothing']}
        
        model = build_single_task_model(
            num_classes=2,
            backbone_pretrained=True,
            backbone_freeze_mode='gradual',
            head_loss_type=loss_type,
            class_weights=class_weights,
            **filtered_kwargs
        )
    
    return model


def main():
    """Main training function with class imbalance handling."""
    parser = argparse.ArgumentParser(description='Train MVFouls Model with Class Balance')
    
    # Data arguments
    parser.add_argument('--train-dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--val-dir', type=str, required=True, help='Validation data directory')
    parser.add_argument('--train-annotations', type=str, required=True, help='Training annotations file')
    parser.add_argument('--val-annotations', type=str, required=True, help='Validation annotations file')
    
    # Model arguments
    parser.add_argument('--multi-task', action='store_true', help='Use multi-task learning')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes (single-task only)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--freeze-mode', type=str, default='gradual', help='Backbone freeze mode')
    
    # Balance-specific arguments
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze imbalance, dont train')
    parser.add_argument('--force-weights', type=str, choices=['balanced', 'effective', 'focal'], 
                       help='Force specific weighting method')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma parameter')
    
    # Other arguments
    parser.add_argument('--output-dir', type=str, default='./outputs_balanced', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging(log_level)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"🚀 Starting balanced MVFouls training")
    logger.info(f"Device: {device}")
    logger.info(f"Multi-task: {args.multi_task}")
    
    try:
        # Create transforms
        transforms = get_video_transforms(image_size=224, augment_train=True)
        
        # Create datasets
        logger.info("📂 Creating datasets...")
        
        # Extract root directory
        root_dir = str(Path(args.train_dir).parent)
        train_split = Path(args.train_dir).name.replace('_720p', '')
        val_split = Path(args.val_dir).name.replace('_720p', '')
        
        train_dataset = MVFoulsDataset(
            root_dir=root_dir,
            split=train_split,
            transform=transforms['train'],
            load_annotations=True,
            num_frames=32
        )
        
        val_dataset = MVFoulsDataset(
            root_dir=root_dir,
            split=val_split,
            transform=transforms['val'],
            load_annotations=True,
            num_frames=32
        )
        
        logger.info(f"📊 Dataset sizes: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Analyze class imbalance
        logger.info("🔍 Analyzing class imbalance...")
        imbalance_analysis = analyze_dataset_imbalance(train_dataset)
        
        if args.analyze_only:
            logger.info("✅ Analysis complete. Exiting (--analyze-only specified).")
            return
        
        # Create balanced model
        logger.info("🏗️ Creating balanced model...")
        model = create_balanced_model(
            multi_task=args.multi_task,
            imbalance_analysis=imbalance_analysis
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Create scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
        
        # Move model to device and ensure class weights are on correct device
        model.to(device)
        
        # Move class weights to device if they exist
        if hasattr(model.head, 'task_weights') and model.head.task_weights:
            for task_name, weights in model.head.task_weights.items():
                if isinstance(weights, torch.Tensor):
                    model.head.task_weights[task_name] = weights.to(device)
        
        # Create trainer
        trainer = MultiTaskTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            log_interval=50,
            eval_interval=1,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0
        )
        
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            'args': vars(args),
            'imbalance_analysis': imbalance_analysis,
            'model_config': model.config
        }
        
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Setup tensorboard
        writer = SummaryWriter(output_dir / 'tensorboard')
        
        # Training loop
        logger.info("🎯 Starting training with balanced losses...")
        
        best_metric = 0.0
        
        for epoch in range(args.epochs):
            logger.info(f"\n📅 Epoch {epoch + 1}/{args.epochs}")
            
            # Training
            model.train()
            train_metrics = []
            
            for batch_idx, (videos, targets) in enumerate(train_loader):
                metrics = trainer.train_step(videos, targets)
                train_metrics.append(metrics)
                
                if (batch_idx + 1) % 50 == 0:
                    avg_loss = sum(m['total_loss'] for m in train_metrics[-50:]) / min(50, len(train_metrics))
                    logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {avg_loss:.4f}")
            
            # Validation
            logger.info("🔬 Running validation...")
            val_results = trainer.evaluate(val_loader, compute_detailed_metrics=True)
            
            # Log metrics
            avg_train_loss = sum(m['total_loss'] for m in train_metrics) / len(train_metrics)
            val_loss = val_results['avg_loss']
            
            logger.info(f"📊 Epoch {epoch + 1} Results:")
            logger.info(f"   Train Loss: {avg_train_loss:.4f}")
            logger.info(f"   Val Loss: {val_loss:.4f}")
            
            # Print detailed metrics if available
            if 'metrics_table' in val_results:
                logger.info("📋 Validation Metrics:")
                print(val_results['metrics_table'])
            
            # Save best model
            current_metric = val_results.get('overall_accuracy', 1.0 - val_loss)
            if current_metric > best_metric:
                best_metric = current_metric
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_metric': best_metric,
                    'config': config
                }
                
                torch.save(checkpoint, output_dir / 'best_model.pth')
                logger.info(f"💾 New best model saved! Metric: {best_metric:.4f}")
            
            # Step scheduler
            if scheduler:
                scheduler.step()
            
            # Tensorboard logging
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Metric/Best', best_metric, epoch)
        
        writer.close()
        logger.info("🎉 Training completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        raise


if __name__ == '__main__':
    main() 