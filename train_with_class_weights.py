#!/usr/bin/env python3
"""
MVFouls Training Script with Proper Class Imbalance Handling

This script addresses the severe class imbalance in MVFouls dataset by:
1. Computing proper class weights for each task
2. Using focal loss with effective number weighting
3. Implementing task-specific loss strategies
4. Monitoring per-class metrics during training
5. Automatic gradual backbone unfreezing

Stage 1 Fixes Applied (Learning Rate Sanity):
- FIXED: Conservative LR scaling prevents aggressive LR increases that cause training collapse
- FIXED: Separate parameter groups for backbone vs head with independent LR tracking
- FIXED: Only backbone LR is reduced when unfreezing (head LR remains stable)
- NEW: Detailed LR logging per epoch shows backbone/head learning rates
- NEW: Scale factors: patch_embed/stage_0 → ×1.0, stage_1 → ×0.5, stage_2/3 → ×0.25

Stage 2 Fixes Applied (Option A: Simplified Loss Strategy):
- IMPLEMENTED: WeightedRandomSampler + CrossEntropy approach for stability
- REMOVED: Focal loss, effective number weights, complex loss configurations
- SIMPLIFIED: Basic inverse frequency class weights (capped at 10x max)
- ELIMINATED: Triple over-compensation that caused training instability
- STREAMLINED: Single loss type (CrossEntropy) across all tasks for consistency
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
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
            
            # Get original datasets and sampler from the trainer's data loaders
            train_dataset = trainer.train_loader.dataset
            val_dataset = trainer.val_loader.dataset
            
            # CRITICAL FIX: Preserve the original sampler to maintain balanced sampling
            original_sampler = trainer.train_loader.sampler
            
            # Create new data loaders with reduced batch size
            from torch.utils.data import DataLoader
            
            # FIXED: Preserve sampler and set shuffle=False when using custom sampler
            if original_sampler is not None and hasattr(original_sampler, 'weights'):
                # We have a WeightedRandomSampler - preserve it
                trainer.train_loader = DataLoader(
                    train_dataset, 
                    batch_size=args.unfreeze_batch_size, 
                    sampler=original_sampler,  # PRESERVE balanced sampling
                    shuffle=False,  # Must be False when using custom sampler
                    num_workers=4, 
                    pin_memory=True,
                    drop_last=True
                )
                logger.info(f"   ✅ Preserved WeightedRandomSampler for balanced training")
            else:
                # No custom sampler, use shuffle
                trainer.train_loader = DataLoader(
                    train_dataset, 
                    batch_size=args.unfreeze_batch_size, 
                    shuffle=True, 
                    num_workers=4, 
                    pin_memory=True,
                    drop_last=True
                )
                logger.info(f"   ⚠️  No custom sampler detected, using shuffle=True")
            
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
    FIXED: Use conservative LR scaling to prevent training instability.
    
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
    
    # NOTE: The CLI arguments --lr-scale-minor, --lr-scale-major, --lr-scale-massive
    # are defined but currently not used. The current implementation uses FIXED
    # conservative scaling values to prevent training instability.
    # TODO: If you want to use the CLI args, uncomment the code below and comment the fixed values.
    
    # Determine scaling factor based on unfreezing type
    # FIXED: Use conservative scaling to prevent LR explosions
    scale_factor = 1.0
    
    # Define unfreezing categories with CONSERVATIVE scaling
    if group_name in ['patch_embed']:
        # Keep same LR for initial unfreezing
        scale_factor = 1.0
        category = "MINOR"
        # Optional: Use CLI arg instead: scale_factor = getattr(args, 'lr_scale_minor', 1.0)
    elif group_name in ['stage_0']:
        # Keep same LR for first stage
        scale_factor = 1.0
        category = "MINOR"
        # Optional: Use CLI arg instead: scale_factor = getattr(args, 'lr_scale_minor', 1.0)
    elif group_name in ['stage_1']:
        # Reduce LR for medium stages (prevents overfitting newly unfrozen layers)
        scale_factor = 0.5
        category = "MAJOR"
        # Optional: Use CLI arg instead: scale_factor = getattr(args, 'lr_scale_major', 1.0) * 0.5
    elif group_name in ['stage_2', 'stage_3']:
        # Significantly reduce LR for massive unfreezing (prevents instability)
        scale_factor = 0.25
        category = "MASSIVE"
        # Optional: Use CLI arg instead: scale_factor = getattr(args, 'lr_scale_massive', 1.0) * 0.25
    else:
        # Unknown group, use conservative scaling
        scale_factor = 0.5
        category = "UNKNOWN"
    
    # Apply scaling ONLY if scale_factor < 1.0 (conservative approach)
    if scale_factor < 1.0:
        new_lr = current_lr * scale_factor
        
        # Update ONLY backbone parameter groups (head LR stays unchanged)
        updated_groups = []
        for param_group in optimizer.param_groups:
            group_name_in_optimizer = param_group.get('name', 'unknown')
            if 'backbone' in group_name_in_optimizer.lower():
                param_group['lr'] = new_lr
                updated_groups.append(group_name_in_optimizer)
        
        logger.info(f"🔧 Adaptive LR Scaling ({category}) - CONSERVATIVE:")
        logger.info(f"   Unfrozen group: {group_name}")
        logger.info(f"   Trainable params: {trainable_params:,}")
        logger.info(f"   Backbone LR: {original_lr:.2e} → {new_lr:.2e} (×{scale_factor:.1f})")
        logger.info(f"   Updated groups: {updated_groups}")
        logger.info(f"   Reason: {category} unfreezing - reducing backbone LR for stability")
        if hasattr(args, 'lr_scale_minor') and (args.lr_scale_minor != 1.2 or args.lr_scale_major != 1.5 or args.lr_scale_massive != 2.0):
            logger.warning(f"   ⚠️  CLI LR scaling args are set but not used (using fixed conservative values)")
    else:
        logger.info(f"🔧 Adaptive LR Scaling ({category}) - NO CHANGE:")
        logger.info(f"   Unfrozen group: {group_name}")
        logger.info(f"   Trainable params: {trainable_params:,}")
        logger.info(f"   LR: {original_lr:.2e} (unchanged)")
        logger.info(f"   Reason: Conservative scaling - no LR increase")


def analyze_dataset_imbalance(dataset) -> Dict[str, Dict]:
    """
    Analyze class imbalance in the dataset for Option A approach.
    SIMPLIFIED: Just provides class counts and imbalance ratios.
    """
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Analyzing dataset class imbalance (Option A: Simple Analysis)...")
    
    # Handle Subset objects (when using train_fraction < 1.0)
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        # This is a Subset object, get the underlying dataset
        base_dataset = dataset.dataset
        subset_indices = dataset.indices
        logger.info(f"🔍 Working with subset of {len(subset_indices)} samples from {len(base_dataset)} total")
        logger.info(f"🔍 Base dataset has {len(base_dataset.annotations)} annotations")
        logger.info(f"🔍 Subset indices range: {min(subset_indices)} to {max(subset_indices)}")
    else:
        base_dataset = dataset
        subset_indices = None
    
    # Get dataset statistics from the base dataset
    stats = base_dataset.get_task_statistics()
    
    # Get task metadata for expected number of classes
    metadata = get_task_metadata()
    
    analysis_results = {}
    
    for task_name, task_stats in stats.items():
        if subset_indices is not None:
            # Recalculate class counts for the subset
            class_counts_dict = {}
            valid_indices = 0
            
            for idx in subset_indices:
                # idx is a clip index into the dataset_index
                if idx < len(base_dataset.dataset_index):
                    clip_info = base_dataset.dataset_index[idx]
                    if clip_info.numeric_labels is not None:
                        valid_indices += 1
                        # Get the task index to extract the label for this task
                        task_names_list = list(stats.keys())
                        if task_name in task_names_list:
                            task_idx = task_names_list.index(task_name)
                            class_idx = clip_info.numeric_labels[task_idx].item()
                            class_counts_dict[class_idx] = class_counts_dict.get(class_idx, 0) + 1
            
            if task_name == list(stats.keys())[0]:  # Only log once for the first task
                logger.info(f"🔍 Found valid annotations for {valid_indices}/{len(subset_indices)} subset clips")
            
            # Convert to list format expected by the rest of the function
            expected_num_classes = len(metadata['class_names'][task_name])
            class_counts = [0] * expected_num_classes
            for class_idx, count in class_counts_dict.items():
                if 0 <= class_idx < expected_num_classes:
                    class_counts[class_idx] = count
        else:
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
        
        # Simple severity classification
        severity = 'SEVERE' if imbalance_ratio > 20 else 'MODERATE' if imbalance_ratio > 5 else 'MILD'
        
        analysis_results[task_name] = {
            'imbalance_ratio': imbalance_ratio,
            'total_samples': total_samples,
            'class_counts': class_counts,
            'class_names': task_stats['class_names'],
            'severity': severity
        }
        
        logger.info(f"📊 {task_name}:")
        logger.info(f"   Imbalance ratio: {imbalance_ratio:.1f}:1 ({severity})")
        logger.info(f"   Strategy: WeightedRandomSampler + CrossEntropy + Class Weights")
        
        # Show class distribution
        for class_idx, (class_name, count) in enumerate(zip(task_stats['class_names'], class_counts)):
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            logger.info(f"     {class_name}: {count} ({percentage:.1f}%)")
    
    return analysis_results


def create_balanced_model(
    multi_task: bool = True,
    imbalance_analysis: Optional[Dict] = None,
    primary_task_weights: Optional[Dict[str, float]] = None,
    backbone_checkpointing: bool = True,
    use_class_weights: bool = True,
    class_weight_cap: float = 10.0,
    **model_kwargs
) -> MVFoulsModel:
    """
    Create a model with Option A: WeightedRandomSampler + Cross-Entropy.
    SIMPLIFIED: Uses only CrossEntropy loss with optional simple class weights.
    """
    logger = logging.getLogger(__name__)
    
    if multi_task:
        # Get task metadata
        metadata = get_task_metadata()
        task_names = metadata['task_names']
        
        # Prepare simplified task-specific configurations
        task_weights = {}
        loss_types_per_task = []
        
        for task_name in task_names:
            # OPTION A: Always use Cross-Entropy loss
            loss_types_per_task.append('ce')
            
            # Optional: Add simple inverse frequency class weights
            if use_class_weights and imbalance_analysis and task_name in imbalance_analysis:
                analysis = imbalance_analysis[task_name]
                class_counts = analysis['class_counts']
                
                # Compute simple inverse frequency weights
                total_samples = sum(class_counts)
                if total_samples > 0:
                    weights = []
                    for count in class_counts:
                        if count > 0:
                            # Inverse frequency, but capped to prevent extreme weights
                            weight = min(total_samples / count, class_weight_cap)
                        else:
                            weight = 1.0  # Default weight for missing classes
                        weights.append(weight)
                    
                    # Normalize weights to have mean of 1.0
                    weights = torch.tensor(weights, dtype=torch.float32)
                    weights = weights / weights.mean()
                    task_weights[task_name] = weights
                    
                    logger.info(f"🎯 {task_name}: CrossEntropy loss, "
                              f"class weights: {weights.tolist()}")
                else:
                    logger.info(f"🎯 {task_name}: CrossEntropy loss, no weights")
            else:
                logger.info(f"🎯 {task_name}: CrossEntropy loss, no weights")
        
        # Create multi-task model with simplified configuration
        model = build_multi_task_model(
            backbone_pretrained=True,
            backbone_freeze_mode='gradual',
            loss_types_per_task=loss_types_per_task,
            class_weights=task_weights,
            task_loss_weights=primary_task_weights,
            backbone_checkpointing=backbone_checkpointing
        )
        
    else:
        # Single-task model with simple class weights
        class_weights = None
        if use_class_weights and imbalance_analysis and 'offence' in imbalance_analysis:
            analysis = imbalance_analysis['offence']
            class_counts = analysis['class_counts']
            total_samples = sum(class_counts)
            
            if total_samples > 0:
                weights = []
                for count in class_counts:
                    if count > 0:
                        weight = min(total_samples / count, class_weight_cap)
                    else:
                        weight = 1.0
                    weights.append(weight)
                
                weights = torch.tensor(weights, dtype=torch.float32)
                weights = weights / weights.mean()
                class_weights = weights
        
        # Filter out conflicting kwargs
        filtered_kwargs = {k: v for k, v in model_kwargs.items() 
                          if k not in ['head_loss_type', 'head_label_smoothing']}
        
        model = build_single_task_model(
            num_classes=2,
            backbone_pretrained=True,
            backbone_freeze_mode='gradual',
            head_loss_type='ce',  # Always use CrossEntropy
            class_weights=class_weights,
            backbone_checkpointing=backbone_checkpointing,
            **filtered_kwargs
        )
    
    return model


def create_balanced_sampler(dataset: MVFoulsDataset, task_names: Union[str, List[str]] = 'action_class') -> WeightedRandomSampler:
    """Create a weighted sampler that balances classes for one or multiple tasks.
    
    Args:
        dataset: The MVFoulsDataset to sample from
        task_names: Either a single task name (str) or list of task names to balance jointly
        
    Returns:
        WeightedRandomSampler that balances the specified task(s)
    """
    # Handle Subset datasets (e.g., for smoke tests)
    subset_indices = None
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        # It's a torch.utils.data.Subset
        subset_indices = list(dataset.indices)
        base_dataset = dataset.dataset  # underlying MVFoulsDataset
        print(f"🔍 Detected Subset dataset: {len(subset_indices)} samples from {len(base_dataset)} total")
    else:
        base_dataset = dataset
    
    # Normalize task_names to list
    if isinstance(task_names, str):
        task_names = [task_names]
    
    # Build joint_labels list
    joint_labels = []
    
    # FIXED: Handle subset indices correctly
    if subset_indices is not None:
        # For subsets, only process the subset indices
        for subset_idx, original_idx in enumerate(subset_indices):
            annotation = base_dataset.annotations[original_idx]
            joint_key = []
            for task_name in task_names:
                joint_key.append(annotation.get(task_name, 0))
            joint_labels.append(tuple(joint_key))
    else:
        # For full datasets, process all annotations
        for idx in range(len(base_dataset.annotations)):
            annotation = base_dataset.annotations[idx]
            joint_key = []
            for task_name in task_names:
                joint_key.append(annotation.get(task_name, 0))
            joint_labels.append(tuple(joint_key))
    
    # Count frequencies etc.
    class_counts = Counter(joint_labels)
    num_samples = len(joint_labels)
    
    # VALIDATION: Ensure sample count matches dataset size
    expected_size = len(subset_indices) if subset_indices is not None else len(base_dataset)
    if num_samples != expected_size:
        raise ValueError(f"Sample count mismatch: got {num_samples}, expected {expected_size}")
    
    # Log joint class distribution for multi-task sampling
    if len(task_names) > 1:
        print(f"\n📊 Joint class distribution for tasks {task_names}:")
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for joint_class, count in sorted_classes[:10]:  # Show top 10
            percentage = (count / num_samples) * 100
            print(f"   {joint_class}: {count} samples ({percentage:.1f}%)")
        if len(sorted_classes) > 10:
            print(f"   ... and {len(sorted_classes) - 10} more joint classes")
        
        # Calculate imbalance ratio
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        print(f"   📈 Joint class imbalance ratio: {imbalance_ratio:.1f}:1")
    
    # Compute weights: inverse frequency for joint classes
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    
    # Create sample weights - FIXED: length must match dataset size exactly
    sample_weights = [class_weights[label] for label in joint_labels]
    
    # VALIDATION: Ensure weights list matches dataset size
    if len(sample_weights) != expected_size:
        raise ValueError(f"Sample weights length mismatch: got {len(sample_weights)}, expected {expected_size}")
    
    print(f"✅ Created balanced sampler: {num_samples} samples, {len(class_counts)} unique classes")
    
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
    
    # Balance-specific arguments (Option A: WeightedRandomSampler + CrossEntropy)
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze imbalance, dont train')
    parser.add_argument('--disable-class-weights', action='store_true', 
                       help='Disable simple inverse frequency class weights')
    parser.add_argument('--class-weight-cap', type=float, default=10.0,
                       help='Maximum class weight multiplier to prevent extreme weights (default: 10.0)')
    
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
    parser.add_argument('--balance-tasks', nargs='+', default=['action_class'],
                       help='Task(s) to balance sampling for. For joint balancing, specify multiple tasks (default: action_class)')
    parser.add_argument('--balance-task', type=str, default=None,
                       help='Single task to balance sampling for (deprecated, use --balance-tasks)')
    parser.add_argument('--joint-severity-sampling', action='store_true',
                       help='Convenience flag to enable joint sampling for action_class and severity (equivalent to --balance-tasks action_class severity)')
    
    # Add new sophisticated weighting strategy
    parser.add_argument('--use-smart-weighting', action='store_true',
                        help='Use sophisticated task weighting based on semantic relevance')
    parser.add_argument('--core-task-weight', type=float, default=20.0,
                        help='Weight for core tasks (action_class, severity) (default: 20.0)')
    parser.add_argument('--support-task-weight', type=float, default=2.0,
                        help='Weight for supporting tasks (contact, bodypart, offence) (default: 2.0)')
    parser.add_argument('--context-task-weight', type=float, default=0.5,
                        help='Weight for contextual tasks (remaining tasks) (default: 0.5)')
    
    # Add train-fraction argument
    parser.add_argument('--train-fraction', type=float, default=1.0,
                        help='Fraction of training data to use (e.g., 0.2 for 20% smoke test)')
    
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
        
        # Apply fractional subset for smoke tests
        if 0 < args.train_fraction < 1.0:
            import random
            subset_size = int(len(train_dataset) * args.train_fraction)
            indices = list(range(len(train_dataset)))
            random.shuffle(indices)
            subset_indices = indices[:subset_size]
            from torch.utils.data import Subset
            train_dataset = Subset(train_dataset, subset_indices)
            logger.info(f"🔍 Using subset of training data: {subset_size}/{len(indices)} samples ({args.train_fraction*100:.1f}%) for smoke test")
        
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
        
        # Create task weighting strategy (simplified for Option A)
        primary_task_weights = {}
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
            logger.info("🎯 Using Option A: WeightedRandomSampler + CrossEntropy")
        
        model = create_balanced_model(
            multi_task=args.multi_task,
            imbalance_analysis=imbalance_analysis,
            primary_task_weights=primary_task_weights,
            backbone_checkpointing=True,  # Enable gradient checkpointing
            use_class_weights=not args.disable_class_weights,  # Use simple inverse frequency class weights
            class_weight_cap=args.class_weight_cap
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
        
        # Create dataloaders with balanced sampling (Option A requires this)
        train_sampler = None
        shuffle = True
        
        if args.balanced_sampling:
            # Handle convenience flag for joint severity sampling
            if args.joint_severity_sampling:
                balance_tasks = ['action_class', 'severity']
                logger.info("🎯 Using convenience flag --joint-severity-sampling")
            else:
                # Handle backward compatibility: use --balance-task if specified, otherwise use --balance-tasks
                balance_tasks = args.balance_tasks
                if args.balance_task is not None:
                    balance_tasks = [args.balance_task]
                    logger.info("⚠️  Using deprecated --balance-task, consider switching to --balance-tasks")
            
            if len(balance_tasks) == 1:
                logger.info(f"🎯 Creating balanced sampler for single task: {balance_tasks[0]}")
            else:
                logger.info(f"🎯 Creating joint balanced sampler for tasks: {balance_tasks}")
                logger.info("   📊 Joint balancing ensures both tasks are balanced together")
            
            train_sampler = create_balanced_sampler(train_dataset, balance_tasks)
            shuffle = False  # Cannot shuffle when using custom sampler
            logger.info("✅ Option A: WeightedRandomSampler enabled")
        else:
            logger.info("⚠️  Option A: WeightedRandomSampler disabled (use --balanced-sampling to enable)")
        
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
        
        # Create optimizer with separate parameter groups for backbone and head
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.head.parameters())
        
        param_groups = [
            {
                'params': head_params,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'name': 'head'
            },
            {
                'params': backbone_params,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'name': 'backbone'
            }
        ]
        
        optimizer = optim.AdamW(param_groups)
        
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
        logger.info("🎯 Starting training with Option A: WeightedRandomSampler + CrossEntropy...")
        
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
            
            # Log current learning rates (head/backbone)
            logger.info(f"📈 Learning Rates:")
            for i, group in enumerate(optimizer.param_groups):
                group_name = group.get('name', f'group_{i}')
                logger.info(f"   {group_name}: {group['lr']:.2e}")
            
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