#!/usr/bin/env python3
"""
MVFouls Model Training Script
============================

Complete training script for the MVFouls video analysis model.
Supports both single-task and multi-task training with comprehensive logging,
checkpointing, and evaluation.

Usage:
    python train.py --config config.yaml
    python train.py --train-dir /path/to/train --val-dir /path/to/val \
                    --train-annotations train.csv --val-annotations val.csv \
                    --multi-task --epochs 100 --batch-size 16
"""

import argparse
import os
import sys
import json
import yaml
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Import our components
try:
    from model.mvfouls_model import MVFoulsModel, build_multi_task_model, build_single_task_model
    from training_utils import MultiTaskTrainer, create_mvfouls_trainer, create_task_schedulers
    from dataset import MVFoulsDataset, create_train_val_split
    from transforms import get_video_transforms
    from utils import get_task_metadata, format_metrics_table
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required files are in the correct locations.")
    sys.exit(1)


def setup_logging(log_dir: Path, level: str = 'INFO'):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_dir / 'training.log')
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=[file_handler, console_handler]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def create_dataloaders(args, transforms):
    """Create training and validation dataloaders from separate directories."""
    logger = logging.getLogger(__name__)
    
    # Create training dataset
    logger.info("Creating training dataset...")
    train_dataset = MVFoulsDataset(
        video_dir=args.train_dir,
        annotations_file=args.train_annotations,
        transforms=transforms['train'],
        multi_task=args.multi_task,
        max_frames=args.max_frames,
        fps=args.fps
    )
    
    # Create validation dataset
    logger.info("Creating validation dataset...")
    val_dataset = MVFoulsDataset(
        video_dir=args.val_dir,
        annotations_file=args.val_annotations,
        transforms=transforms['val'],
        multi_task=args.multi_task,
        max_frames=args.max_frames,
        fps=args.fps
    )
    
    logger.info(f"Datasets created: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_dataloader(args, transforms, split='test'):
    """Create test or challenge dataloader."""
    logger = logging.getLogger(__name__)
    
    if split == 'test':
        video_dir = args.test_dir
        annotations_file = args.test_annotations
    elif split == 'challenge':
        video_dir = args.challenge_dir
        annotations_file = args.challenge_annotations
    else:
        raise ValueError(f"Invalid split: {split}. Use 'test' or 'challenge'.")
    
    if video_dir is None or annotations_file is None:
        logger.warning(f"{split.capitalize()} directory or annotations not provided. Skipping {split} dataloader.")
        return None
    
    logger.info(f"Creating {split} dataset...")
    test_dataset = MVFoulsDataset(
        video_dir=video_dir,
        annotations_file=annotations_file,
        transforms=transforms['val'],  # Use validation transforms (no augmentation)
        multi_task=args.multi_task,
        max_frames=args.max_frames,
        fps=args.fps
    )
    
    logger.info(f"{split.capitalize()} dataset created: {len(test_dataset)} samples")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return test_loader


def train_epoch(trainer, train_loader, scheduler, epoch, writer, args):
    """Train for one epoch."""
    logger = logging.getLogger(__name__)
    
    epoch_results = trainer.train_epoch(train_loader, scheduler)
    
    # Log metrics
    writer.add_scalar('Train/Loss', epoch_results['avg_loss'], epoch)
    writer.add_scalar('Train/LearningRate', trainer.optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('Train/EpochTime', epoch_results['epoch_time'], epoch)
    
    logger.info(f"Epoch {epoch}/{args.epochs} - "
                f"Loss: {epoch_results['avg_loss']:.4f}, "
                f"Time: {epoch_results['epoch_time']:.1f}s, "
                f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")
    
    return epoch_results


def validate_epoch(trainer, val_loader, epoch, writer, args):
    """Validate for one epoch."""
    logger = logging.getLogger(__name__)
    
    logger.info("Running validation...")
    eval_results = trainer.evaluate(val_loader, compute_detailed_metrics=True)
    
    # Log basic metrics
    writer.add_scalar('Val/Loss', eval_results['avg_loss'], epoch)
    
    # Log detailed metrics if available
    if 'overall_metrics' in eval_results:
        overall = eval_results['overall_metrics']
        for metric_name, value in overall.items():
            writer.add_scalar(f'Val/{metric_name}', value, epoch)
    
    # Log task-specific metrics
    if 'task_metrics' in eval_results:
        for task_name, metrics in eval_results['task_metrics'].items():
            for metric_name, value in metrics.items():
                writer.add_scalar(f'Val/{task_name}/{metric_name}', value, epoch)
    
    # Print metrics table
    if 'metrics_table' in eval_results:
        logger.info("Validation Results:")
        logger.info("\n" + eval_results['metrics_table'])
    
    return eval_results


def save_best_model(trainer, eval_results, best_metric, epoch, save_dir, args):
    """Save model if it's the best so far."""
    logger = logging.getLogger(__name__)
    
    # Determine current metric
    if 'overall_metrics' in eval_results:
        current_metric = eval_results['overall_metrics'].get('accuracy', 0.0)
    else:
        current_metric = -eval_results['avg_loss']  # Use negative loss if no accuracy
    
    # Check if this is the best model
    is_best = current_metric > best_metric
    
    if is_best:
        best_metric = current_metric
        
        # Save best model
        best_path = save_dir / 'best_model.pth'
        trainer.save_checkpoint(
            str(best_path),
            best_metric=best_metric,
            metadata={
                'epoch': epoch,
                'eval_results': eval_results,
                'args': vars(args)
            }
        )
        logger.info(f"New best model saved! Metric: {best_metric:.4f}")
    
    # Save regular checkpoint
    if epoch % args.save_every == 0:
        checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
        trainer.save_checkpoint(
            str(checkpoint_path),
            best_metric=best_metric,
            metadata={'epoch': epoch, 'args': vars(args)}
        )
    
    return best_metric, is_best


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MVFouls Model')
    
    # Data arguments
    parser.add_argument('--train-dir', type=str, required=True,
                        help='Path to training video directory')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='Path to validation video directory')
    parser.add_argument('--test-dir', type=str, default=None,
                        help='Path to test video directory (optional)')
    parser.add_argument('--challenge-dir', type=str, default=None,
                        help='Path to challenge video directory (optional)')
    parser.add_argument('--train-annotations', type=str, required=True,
                        help='Path to training annotations file')
    parser.add_argument('--val-annotations', type=str, required=True,
                        help='Path to validation annotations file')
    parser.add_argument('--test-annotations', type=str, default=None,
                        help='Path to test annotations file (optional)')
    parser.add_argument('--challenge-annotations', type=str, default=None,
                        help='Path to challenge annotations file (optional)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    
    # Model arguments
    parser.add_argument('--multi-task', action='store_true',
                        help='Use multi-task learning')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes (single-task only)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone during training')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'warmup_cosine'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Warmup steps for warmup_cosine scheduler')
    
    # Data processing arguments
    parser.add_argument('--max-frames', type=int, default=32,
                        help='Maximum number of frames per video')
    parser.add_argument('--fps', type=int, default=8,
                        help='Target FPS for video processing')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')

    
    # Training configuration
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Steps between logging')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='Epochs between evaluation')
    parser.add_argument('--save-every', type=int, default=2,
                        help='Epochs between saving checkpoints')
    
    # Other arguments
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"mvfouls_{'multi' if args.multi_task else 'single'}_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, 'INFO')
    logger.info(f"Starting training experiment: {exp_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup tensorboard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    try:
        # Create transforms
        logger.info("Setting up data transforms...")
        transforms = get_video_transforms(
            image_size=args.image_size,
            augment_train=True
        )
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(args, transforms)
        
        # Create model and trainer
        logger.info("Creating model and trainer...")
        model_config = {
            'multi_task': args.multi_task,
            'num_classes': args.num_classes,
            'backbone_pretrained': args.pretrained,
            'backbone_freeze_mode': 'freeze_all' if args.freeze_backbone else 'none',
            'head_dropout': 0.5,
            'head_pooling': 'avg',
            'head_loss_type': 'focal'
        }
        
        optimizer_config = {
            'type': args.optimizer,
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }
        
        trainer_config = {
            'log_interval': args.log_interval,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'max_grad_norm': args.max_grad_norm
        }
        
        model, trainer = create_mvfouls_trainer(
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            device=device
        )
        
        # Create scheduler
        total_steps = len(train_loader) * args.epochs
        scheduler_kwargs = {
            'T_max': args.epochs,
            'warmup_steps': args.warmup_steps,
            'total_steps': total_steps
        }
        scheduler = create_task_schedulers(
            trainer.optimizer, 
            args.scheduler, 
            **scheduler_kwargs
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        best_metric = float('-inf')
        
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = trainer.load_checkpoint(args.resume)
            start_epoch = checkpoint.get('epoch', 0)
            best_metric = checkpoint.get('best_metric', float('-inf'))
        
        # Training loop
        logger.info("Starting training...")
        logger.info(f"Training for {args.epochs} epochs, starting from epoch {start_epoch}")
        
        for epoch in range(start_epoch, args.epochs):
            # Training
            train_results = train_epoch(trainer, train_loader, scheduler, epoch, writer, args)
            
            # Validation
            if epoch % args.eval_interval == 0:
                eval_results = validate_epoch(trainer, val_loader, epoch, writer, args)
                
                # Update curriculum if applicable
                trainer.update_curriculum(eval_results)
                
                # Save best model
                best_metric, is_best = save_best_model(
                    trainer, eval_results, best_metric, epoch, output_dir, args
                )
                
                if is_best:
                    logger.info("🎉 New best model!")
        
        # Final evaluation
        logger.info("Training completed! Running final evaluation...")
        final_results = validate_epoch(trainer, val_loader, args.epochs, writer, args)
        
        # Save final model
        final_path = output_dir / 'final_model.pth'
        trainer.save_checkpoint(
            str(final_path),
            best_metric=best_metric,
            metadata={
                'final_results': final_results,
                'args': vars(args)
            }
        )
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Best metric: {best_metric:.4f}")
        logger.info(f"Models saved in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        writer.close()


if __name__ == '__main__':
    main() 