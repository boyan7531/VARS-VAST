#!/usr/bin/env python3
"""
MVFouls Training Script with Proper Class Imbalance Handling

This script addresses the severe class imbalance in MVFouls dataset by:
1. Computing proper class weights for each task
2. Using focal loss with effective number weighting
3. Implementing task-specific loss strategies
4. Monitoring per-class metrics during training
5. Automatic gradual backbone unfreezing
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import nullcontext
from collections import Counter

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

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
    analyze_class_distribution,
    compute_task_metrics,
    format_metrics_table
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


def get_unfreeze_schedule(total_epochs: int, freeze_mode: str) -> Dict[int, str]:
    """
    Create a schedule for gradual backbone unfreezing.
    
    Args:
        total_epochs: Total number of training epochs
        freeze_mode: Backbone freeze mode
        
    Returns:
        Dictionary mapping epoch numbers to unfreeze actions
    """
    schedule = {}
    
    if freeze_mode == 'gradual':
        # Gradual unfreezing schedule based on total epochs
        if total_epochs >= 20:
            # For 20+ epochs: unfreeze every 3-4 epochs
            schedule[3] = "patch_embed"      # Epoch 4: unfreeze patch embedding
            schedule[6] = "stage_0"          # Epoch 7: unfreeze first stage
            schedule[9] = "stage_1"          # Epoch 10: unfreeze second stage
            schedule[12] = "stage_2"         # Epoch 13: unfreeze third stage
            schedule[15] = "stage_3"         # Epoch 16: unfreeze fourth stage
        elif total_epochs >= 15:
            # For 15+ epochs: unfreeze every 3 epochs
            schedule[2] = "patch_embed"      # Epoch 3: unfreeze patch embedding
            schedule[5] = "stage_0"          # Epoch 6: unfreeze first stage
            schedule[8] = "stage_1"          # Epoch 9: unfreeze second stage
            schedule[11] = "stage_2"         # Epoch 12: unfreeze third stage
            schedule[14] = "stage_3"         # Epoch 15: unfreeze fourth stage
        elif total_epochs >= 10:
            # For 10+ epochs: unfreeze every 2 epochs
            schedule[1] = "patch_embed"      # Epoch 2: unfreeze patch embedding
            schedule[3] = "stage_0"          # Epoch 4: unfreeze first stage
            schedule[5] = "stage_1"          # Epoch 6: unfreeze second stage
            schedule[7] = "stage_2"          # Epoch 8: unfreeze third stage
            schedule[9] = "stage_3"          # Epoch 10: unfreeze fourth stage
        else:
            # For short training: unfreeze every epoch after first
            for i in range(min(5, total_epochs - 1)):
                schedule[i + 1] = ["patch_embed", "stage_0", "stage_1", "stage_2", "stage_3"][i]
    
    return schedule


def apply_gradual_unfreezing(model: MVFoulsModel, epoch: int, unfreeze_schedule: Dict[int, str], logger, trainer=None, args=None, optimizer=None):
    """
    Apply gradual unfreezing based on the schedule with optional adaptive learning rate scaling.
    
    Args:
        model: The MVFouls model
        epoch: Current epoch (0-indexed)
        unfreeze_schedule: Schedule mapping epochs to unfreeze actions
        logger: Logger instance
        trainer: Optional trainer instance for batch size reduction
        args: Optional args for batch size and learning rate configuration
        optimizer: Optional optimizer for learning rate scaling
    """
    if epoch in unfreeze_schedule and model.backbone.freeze_mode == 'gradual':
        group_name = unfreeze_schedule[epoch]
        
        # Get current stage before unfreezing
        current_stage = model.backbone.get_current_unfreeze_stage()
        
        # Check if we need to reduce batch size on first unfreeze
        if (args and trainer and args.reduce_batch_on_unfreeze and 
            current_stage == -1 and hasattr(trainer, 'train_loader')):
            
            logger.info(f"🔄 Reducing batch size from {args.batch_size} to {args.unfreeze_batch_size} for backbone unfreezing")
            
            # Get original datasets from the trainer's data loaders
            train_dataset = trainer.train_loader.dataset
            val_dataset = trainer.val_loader.dataset
            
            # Create new data loaders with reduced batch size
            from torch.utils.data import DataLoader
            
            trainer.train_loader = DataLoader(
                train_dataset, 
                batch_size=args.unfreeze_batch_size, 
                shuffle=True, 
                num_workers=4, 
                pin_memory=True,
                drop_last=True
            )
            trainer.val_loader = DataLoader(
                val_dataset, 
                batch_size=args.unfreeze_batch_size, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True
            )
            
            logger.info(f"   ✅ Updated data loaders with batch size {args.unfreeze_batch_size}")
            logger.info(f"   📊 New train batches: {len(trainer.train_loader)}, val batches: {len(trainer.val_loader)}")
        
        # Unfreeze next group
        model.unfreeze_backbone_gradually()
        
        # Get new stage after unfreezing
        new_stage = model.backbone.get_current_unfreeze_stage()
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = (trainable_params / total_params) * 100
        
        logger.info(f"🔓 Epoch {epoch + 1}: Unfroze backbone group '{group_name}' "
                   f"(stage {current_stage} → {new_stage})")
        logger.info(f"   Trainable parameters: {trainable_params:,} / {total_params:,} "
                   f"({trainable_pct:.1f}%)")
        
        # Apply adaptive learning rate scaling if enabled
        if args and args.adaptive_lr and optimizer is not None:
            apply_adaptive_lr_scaling(
                optimizer=optimizer,
                group_name=group_name,
                trainable_params=trainable_params,
                current_stage=current_stage,
                new_stage=new_stage,
                args=args,
                logger=logger
            )


def apply_adaptive_lr_scaling(optimizer, group_name: str, trainable_params: int, current_stage: int, new_stage: int, args, logger):
    """
    Apply adaptive learning rate scaling based on the unfreezing stage and parameter count.
    
    Args:
        optimizer: The optimizer to scale learning rate for
        group_name: Name of the unfrozen group (e.g., 'patch_embed', 'stage_0', etc.)
        trainable_params: Number of trainable parameters after unfreezing
        current_stage: Stage before unfreezing
        new_stage: Stage after unfreezing
        args: Arguments containing scaling factors
        logger: Logger instance
    """
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    original_lr = current_lr
    
    # Determine scaling factor based on unfreezing type
    scale_factor = 1.0
    
    # Define unfreezing categories
    if group_name in ['patch_embed']:
        # Minor unfreezing: small parameter increase
        scale_factor = args.lr_scale_minor
        category = "MINOR"
    elif group_name in ['stage_0']:
        # Minor to moderate unfreezing
        scale_factor = args.lr_scale_minor
        category = "MINOR"
    elif group_name in ['stage_1']:
        # Major unfreezing: significant parameter increase
        scale_factor = args.lr_scale_major
        category = "MAJOR"
    elif group_name in ['stage_2', 'stage_3']:
        # Massive unfreezing: huge parameter increase (10M+ parameters)
        if trainable_params > 10_000_000:  # More than 10M parameters
            scale_factor = args.lr_scale_massive
            category = "MASSIVE"
        else:
            scale_factor = args.lr_scale_major
            category = "MAJOR"
    else:
        # Unknown group, use moderate scaling
        scale_factor = args.lr_scale_major
        category = "UNKNOWN"
    
    # Apply scaling
    new_lr = current_lr * scale_factor
    
    # Update all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    logger.info(f"🔧 Adaptive LR Scaling ({category}):")
    logger.info(f"   Group: {group_name}")
    logger.info(f"   Trainable params: {trainable_params:,}")
    logger.info(f"   LR: {original_lr:.2e} → {new_lr:.2e} (×{scale_factor:.1f})")
    logger.info(f"   Reason: {category} unfreezing detected")


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
    primary_task_weights: Optional[Dict[str, float]] = None,
    backbone_checkpointing: bool = True,
    task_focal_gamma_map: Optional[Dict[str, float]] = None,
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
            backbone_checkpointing=backbone_checkpointing,
            task_focal_gamma_map=task_focal_gamma_map
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
            backbone_checkpointing=backbone_checkpointing,
            **filtered_kwargs
        )
    
    return model


def create_balanced_sampler(dataset: MVFoulsDataset, task_name: str = 'action_class') -> WeightedRandomSampler:
    """Create a weighted sampler that balances classes for a specific task."""
    # Get labels directly from dataset annotations without loading videos
    labels = []
    
    # Access annotations directly - this is much faster than dataset[i]
    for annotation in dataset.annotations:
        if task_name in annotation:
            labels.append(annotation[task_name])
        else:
            labels.append(0)  # Default class
    
    # Count class frequencies
    class_counts = Counter(labels)
    num_samples = len(labels)
    
    # Compute weights: inverse frequency
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    
    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )


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
    parser.add_argument('--primary-focal-gamma', type=float, default=4.0, 
                       help='More aggressive focal loss gamma for primary tasks (default: 4.0)')
    parser.add_argument('--primary-focal-alpha', type=float, default=0.75,
                       help='Focal loss alpha parameter for primary tasks (default: 0.75)')
    
    # Unfreezing arguments
    parser.add_argument('--disable-gradual-unfreezing', action='store_true', 
                       help='Disable automatic gradual unfreezing')
    parser.add_argument('--unfreeze-schedule', type=str, 
                       help='Custom unfreeze schedule as JSON (e.g., {"3": "patch_embed", "6": "stage_0"})')
    parser.add_argument('--reduce-batch-on-unfreeze', action='store_true',
                       help='Automatically reduce batch size when unfreezing backbone')
    parser.add_argument('--unfreeze-batch-size', type=int, default=4,
                       help='Batch size to use after unfreezing (default: 4)')
    
    # Other arguments
    parser.add_argument('--output-dir', type=str, default='./outputs_balanced', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    # Primary task weighting arguments
    parser.add_argument('--primary-task-weight', type=float, default=3.0,
                        help='Weight multiplier for primary tasks (action_class, severity) (default: 3.0)')
    parser.add_argument('--auxiliary-task-weight', type=float, default=1.0,
                        help='Weight multiplier for auxiliary tasks (default: 1.0)')
    parser.add_argument('--primary-tasks', nargs='+', default=['action_class', 'severity'],
                        help='List of primary task names (default: action_class severity)')
    
    # Adaptive learning rate arguments
    parser.add_argument('--adaptive-lr', action='store_true',
                        help='Enable adaptive learning rate scaling during unfreezing')
    parser.add_argument('--lr-scale-minor', type=float, default=1.2,
                        help='LR multiplier for minor unfreezing (patch_embed, stage_0) (default: 1.2)')
    parser.add_argument('--lr-scale-major', type=float, default=1.5,
                        help='LR multiplier for major unfreezing (stage_1, stage_2, stage_3) (default: 1.5)')
    parser.add_argument('--lr-scale-massive', type=float, default=2.0,
                        help='LR multiplier for massive unfreezing (stage_2, stage_3 with >10M params) (default: 2.0)')
    
    # Add balanced sampling arguments
    parser.add_argument('--balanced-sampling', action='store_true',
                       help='Use balanced sampling to ensure equal class representation')
    parser.add_argument('--balance-task', type=str, default='action_class',
                       help='Task to balance sampling for (default: action_class)')
    
    # Add new sophisticated weighting strategy
    parser.add_argument('--use-smart-weighting', action='store_true',
                        help='Use sophisticated task weighting based on semantic relevance')
    parser.add_argument('--core-task-weight', type=float, default=20.0,
                        help='Weight for core tasks (action_class, severity) (default: 20.0)')
    parser.add_argument('--support-task-weight', type=float, default=2.0,
                        help='Weight for supporting tasks (contact, bodypart, offence) (default: 2.0)')
    parser.add_argument('--context-task-weight', type=float, default=0.5,
                        help='Weight for contextual tasks (remaining tasks) (default: 0.5)')
    
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
        
        # Create task weighting strategy
        primary_task_weights = {}
        # Build per-task focal gamma map (default 2.0, override for primary tasks)
        gamma_map = {}
        if args.multi_task:
            # Get all task names from metadata
            from utils import get_task_metadata
            metadata = get_task_metadata()
            all_tasks = metadata['task_names']
            
            if args.use_smart_weighting:
                # Smart weighting based on semantic relevance
                # Core tasks (main objectives)
                core_tasks = ['action_class', 'severity']
                
                # Support tasks (directly relevant to core tasks)
                support_tasks = ['contact', 'bodypart', 'offence', 'upper_body_part']
                
                # Context tasks (provide additional context)
                context_tasks = ['multiple_fouls', 'try_to_play', 'touch_ball', 'handball', 'handball_offence']
                
                for task_name in all_tasks:
                    if task_name in core_tasks:
                        primary_task_weights[task_name] = args.core_task_weight
                    elif task_name in support_tasks:
                        primary_task_weights[task_name] = args.support_task_weight
                    elif task_name in context_tasks:
                        primary_task_weights[task_name] = args.context_task_weight
                    else:
                        primary_task_weights[task_name] = args.context_task_weight
                
                logger.info("🎯 Smart task weighting enabled:")
                logger.info(f"   Core tasks ({args.core_task_weight}x): {core_tasks}")
                logger.info(f"   Support tasks ({args.support_task_weight}x): {support_tasks}")
                logger.info(f"   Context tasks ({args.context_task_weight}x): {context_tasks}")
                
                # Set focal gamma per task: use primary_focal_gamma for core tasks, else default 2.0
                for task_name in all_tasks:
                    if task_name in ['action_class', 'severity']:
                        gamma_map[task_name] = args.primary_focal_gamma
                    else:
                        gamma_map[task_name] = 2.0
            
            else:
                # Simple primary/auxiliary weighting
                logger.info(f"🎯 Setting up primary task weighting:")
                logger.info(f"   Primary tasks: {args.primary_tasks} (weight: {args.primary_task_weight}x)")
                logger.info(f"   Auxiliary tasks: all others (weight: {args.auxiliary_task_weight}x)")
                
                for task_name in all_tasks:
                    if task_name in args.primary_tasks:
                        primary_task_weights[task_name] = args.primary_task_weight
                    else:
                        primary_task_weights[task_name] = args.auxiliary_task_weight
            
            logger.info(f"   Task weights: {primary_task_weights}")
        
        model = create_balanced_model(
            multi_task=args.multi_task,
            imbalance_analysis=imbalance_analysis,
            primary_task_weights=primary_task_weights,
            backbone_checkpointing=True,  # Enable gradient checkpointing
            task_focal_gamma_map=gamma_map
        )
        
        # Create unfreezing schedule
        if args.freeze_mode == 'gradual' and not args.disable_gradual_unfreezing:
            if args.unfreeze_schedule:
                # Parse custom schedule
                unfreeze_schedule = {int(k): v for k, v in json.loads(args.unfreeze_schedule).items()}
                logger.info(f"📅 Using custom unfreeze schedule: {unfreeze_schedule}")
            else:
                # Generate automatic schedule
                unfreeze_schedule = get_unfreeze_schedule(args.epochs, args.freeze_mode)
                logger.info(f"📅 Generated unfreeze schedule for {args.epochs} epochs:")
                for epoch, group in unfreeze_schedule.items():
                    logger.info(f"   Epoch {epoch + 1}: unfreeze {group}")
        else:
            unfreeze_schedule = {}
            if args.freeze_mode == 'gradual':
                logger.info("⚠️ Gradual unfreezing disabled by --disable-gradual-unfreezing")
        
        # Create dataloaders with optional balanced sampling
        train_sampler = None
        shuffle = True
        
        if args.balanced_sampling:
            logger.info(f"🎯 Creating balanced sampler for task: {args.balance_task}")
            train_sampler = create_balanced_sampler(train_dataset, args.balance_task)
            shuffle = False  # Cannot shuffle when using custom sampler
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=shuffle,
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
            max_grad_norm=1.0,
            primary_task_weights=primary_task_weights
        )
        
        # Store data loaders in trainer so they can be updated during gradual unfreezing
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
        
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            'args': vars(args),
            'imbalance_analysis': imbalance_analysis,
            'model_config': model.config,
            'unfreeze_schedule': unfreeze_schedule
        }
        
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Setup tensorboard
        writer = SummaryWriter(output_dir / 'tensorboard')
        
        # Training loop
        logger.info("🎯 Starting training with balanced losses...")
        
        # Log primary task weighting if enabled
        if args.multi_task and primary_task_weights:
            logger.info("🎯 Primary task weighting enabled:")
            for task_name, weight in primary_task_weights.items():
                logger.info(f"   {task_name}: {weight}x weight")
        
        # Log adaptive learning rate configuration
        if args.adaptive_lr:
            logger.info("🔧 Adaptive learning rate scaling enabled:")
            logger.info(f"   Base LR: {args.lr:.2e}")
            logger.info(f"   Minor unfreezing scale: {args.lr_scale_minor}x")
            logger.info(f"   Major unfreezing scale: {args.lr_scale_major}x")
            logger.info(f"   Massive unfreezing scale: {args.lr_scale_massive}x")
        else:
            logger.info("⚠️  Adaptive learning rate scaling disabled (use --adaptive-lr to enable)")
        
        # Setup mixed precision training
        use_amp = device.type == 'cuda'
        scaler = GradScaler() if use_amp else None
        if use_amp:
            logger.info("🚀 Automatic Mixed Precision (AMP) enabled")
        
        best_metric = 0.0
        
        for epoch in range(args.epochs):
            print(f"\n{'='*80}")
            logger.info(f"📅 Epoch {epoch + 1}/{args.epochs}")
            print(f"{'='*80}")
            
            # Apply gradual unfreezing if scheduled
            apply_gradual_unfreezing(model, epoch, unfreeze_schedule, logger, trainer, args, optimizer)
            
            # Training
            model.train()
            train_metrics = []
            
            # Create progress bar for training
            train_pbar = tqdm(
                trainer.train_loader, 
                desc=f"Epoch {epoch + 1}/{args.epochs} [Train]",
                leave=False,
                ncols=100,
                unit="batch"
            )
            
            for batch_idx, (videos, targets) in enumerate(train_pbar):
                # Use mixed precision if available
                with autocast() if use_amp else nullcontext():
                    metrics = trainer.train_step(videos, targets, scaler=scaler if use_amp else None)
                train_metrics.append(metrics)
                
                # Update progress bar with current loss
                if len(train_metrics) >= 10:  # Update every 10 batches for smoother display
                    recent_loss = sum(m['total_loss'] for m in train_metrics[-10:]) / 10
                    train_pbar.set_postfix({
                        'Loss': f'{recent_loss:.4f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
            
            # Validation
            print()  # Add space before validation
            val_results = trainer.evaluate(
                trainer.val_loader, 
                compute_detailed_metrics=True,
                compute_task_metrics=compute_task_metrics,
                format_metrics_table=format_metrics_table
            )
            
            # Log metrics
            avg_train_loss = sum(m['total_loss'] for m in train_metrics) / len(train_metrics)
            val_loss = val_results['avg_loss']
            
            print(f"\n📊 Epoch {epoch + 1} Results:")
            print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Print detailed metrics if available
            if 'metrics_table' in val_results:
                print("\n📋 Validation Metrics:")
                print(val_results['metrics_table'])
            
            # Log current backbone status
            if args.freeze_mode == 'gradual':
                current_stage = model.backbone.get_current_unfreeze_stage()
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                trainable_pct = (trainable_params / total_params) * 100
                logger.info(f"🔧 Backbone status: stage {current_stage}, "
                           f"{trainable_params:,} trainable ({trainable_pct:.1f}%)")
            
            # Save best model
            current_metric = val_results.get('overall_metrics', {}).get('accuracy', 1.0 - val_loss)
            if current_metric > best_metric:
                best_metric = current_metric
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_metric': best_metric,
                    'config': config,
                    'backbone_stage': model.backbone.get_current_unfreeze_stage()
                }
                
                # Save with unique filename including epoch and metric
                model_filename = f'best_model_epoch_{epoch+1:02d}_metric_{best_metric:.4f}.pth'
                model_path = output_dir / model_filename
                torch.save(checkpoint, model_path)
                
                # Also save as latest best model (for easy loading)
                latest_path = output_dir / 'best_model_latest.pth'
                torch.save(checkpoint, latest_path)
                
                logger.info(f"💾 New best model saved! Metric: {best_metric:.4f}")
                logger.info(f"   Saved as: {model_filename}")
                logger.info(f"   Also saved as: best_model_latest.pth")
            
            # Step scheduler
            if scheduler:
                scheduler.step()
            
            # Tensorboard logging
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Metric/Best', best_metric, epoch)
            writer.add_scalar('LearningRate/Current', optimizer.param_groups[0]['lr'], epoch)
            
            # Log backbone unfreezing progress
            if args.freeze_mode == 'gradual':
                current_stage = model.backbone.get_current_unfreeze_stage()
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                writer.add_scalar('Backbone/Stage', current_stage, epoch)
                writer.add_scalar('Backbone/TrainableParams', trainable_params, epoch)
        
        writer.close()
        logger.info("🎉 Training completed successfully!")
        
        # Final backbone status
        if args.freeze_mode == 'gradual':
            final_stage = model.backbone.get_current_unfreeze_stage()
            final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            final_total = sum(p.numel() for p in model.parameters())
            logger.info(f"🏁 Final backbone status: stage {final_stage}, "
                       f"{final_trainable:,}/{final_total:,} trainable "
                       f"({final_trainable/final_total*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        raise


if __name__ == '__main__':
    main() 