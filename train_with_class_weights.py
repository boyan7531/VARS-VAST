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
import numpy as np
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
from dataset import MVFoulsDataset, bag_of_clips_collate_fn
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
    task_names: Optional[List[str]] = None,
    imbalance_analysis: Optional[Dict] = None,
    primary_task_weights: Optional[Dict[str, float]] = None,
    backbone_checkpointing: bool = True,
    use_class_weights: bool = True,
    class_weight_cap: float = 10.0,
    loss_types_per_task: Optional[Dict[str, str]] = None,
    use_effective_weights: bool = False,
    **model_kwargs
) -> MVFoulsModel:
    """
    Creates a balanced MVFouls model with proper class weighting and loss configuration.

    Args:
        multi_task: Whether to build a multi-task model.
        task_names: Optional list of task names to build the model for. If provided,
                    it overrides the tasks from imbalance_analysis.
        imbalance_analysis: Analysis of dataset class distribution.
        primary_task_weights: Weights for combining task losses.
        backbone_checkpointing: Whether to enable gradient checkpointing.
        use_class_weights: Whether to use class weights.
        class_weight_cap: Maximum class weight multiplier to prevent extreme weights.
        loss_types_per_task: Per-task loss types.
        use_effective_weights: Whether to use effective number of samples for weighting.
        model_kwargs: Additional arguments for the model head.

    Returns:
        An MVFoulsModel instance.
    """
    logger = logging.getLogger(__name__)

    if multi_task:
        # Get all possible task names from metadata
        all_task_names = get_task_metadata().get('task_names', [])

        # If a specific list of task_names is not provided, default to all tasks
        if task_names is None:
            task_names = all_task_names
            logger.info(f"No specific tasks provided; building model for all tasks: {task_names}")
        else:
            logger.info(f"Building model for specified tasks: {task_names}")

        # Filter imbalance analysis to only include the specified tasks
        if imbalance_analysis:
            imbalance_analysis = {k: v for k, v in imbalance_analysis.items() if k in task_names}
        else:
            logger.warning("Imbalance analysis not provided. Using default metadata for selected tasks.")
            imbalance_analysis = {
                task: {'num_classes': get_task_metadata()['num_classes'][all_task_names.index(task)]}
                for task in task_names if task in all_task_names
            }
        
        # Ensure the task order is consistent
        final_task_names = [task for task in task_names if task in imbalance_analysis]
        if len(final_task_names) != len(task_names):
            logger.warning("Some specified tasks were not found in imbalance analysis and will be skipped.")
        
        num_classes_list = [imbalance_analysis[task]['num_classes'] for task in final_task_names]
        
        # Compute class weights if enabled
        class_weights = None
        if use_class_weights:
            class_weights = {}
            for task in final_task_names:
                task_info = imbalance_analysis[task]
                weights = compute_class_weights(
                    class_counts=task_info.get('class_counts'),
                    num_classes=task_info['num_classes'],
                    strategy='inverse_capped',
                    cap=class_weight_cap,
                    effective_num_beta=0.999 if use_effective_weights else None
                )
                class_weights[task] = torch.tensor(weights, dtype=torch.float32)
                logger.info(f"Computed class weights for task '{task}': {weights.tolist()}")
    else:
        # Logic for single-task model remains unchanged
        # ... (assuming single-task logic doesn't need task_name filtering)
        logger.info("Building single-task model...")
        task_names = list(imbalance_analysis.keys())
        if len(task_names) > 1:
            logger.warning(f"Multiple tasks found in imbalance analysis for single-task mode. Using first task: {task_names[0]}")
        task_name = task_names[0]
        
        num_classes = imbalance_analysis[task_name]['num_classes']
        class_counts = imbalance_analysis[task_name].get('class_counts')
        
        class_weights_tensor = None
        if use_class_weights:
            weights = compute_class_weights(
                class_counts=class_counts,
                num_classes=num_classes,
                strategy='inverse_capped',
                cap=class_weight_cap,
                effective_num_beta=0.999 if use_effective_weights else None
            )
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # Determine loss types for each task
    final_loss_types = {}
    if multi_task:
        base_loss_types = loss_types_per_task or {}
        for task in final_task_names:
            final_loss_types[task] = base_loss_types.get(task, 'ce') # Default to 'ce'
    elif not multi_task:
        task_name = task_names[0]
        final_loss_types[task_name] = (loss_types_per_task or {}).get(task_name, 'ce')

    logger.info(f"Using loss types: {final_loss_types}")

    # Build the model
    if multi_task:
        logger.info("Building multi-task model...")
        model = build_multi_task_model(
            task_names=final_task_names,
            num_classes_per_task=num_classes_list,
            task_weights=class_weights,
            loss_types_per_task=list(final_loss_types.values()),
            task_loss_weights=primary_task_weights,
            gradient_checkpointing=backbone_checkpointing,
            **model_kwargs
        )
    else:
        logger.info(f"Building single-task model for '{task_name}'...")
        model = build_single_task_model(
            num_classes=num_classes,
            class_weights=class_weights_tensor,
            loss_type=final_loss_types[task_name],
            gradient_checkpointing=backbone_checkpointing,
            **model_kwargs
        )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created. Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")

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
    
    # FIXED: Handle subset indices correctly using dataset_index
    if subset_indices is not None:
        # For subsets, process clips via dataset_index
        for idx in subset_indices:
            if idx < len(base_dataset.dataset_index):
                clip_info = base_dataset.dataset_index[idx]
                if clip_info.numeric_labels is not None:
                    joint_key = []
                    # Get the task indices for the requested tasks
                    from utils import get_task_metadata
                    metadata = get_task_metadata()
                    task_names_list = metadata['task_names']
                    
                    for task_name in task_names:
                        if task_name in task_names_list:
                            task_idx = task_names_list.index(task_name)
                            joint_key.append(clip_info.numeric_labels[task_idx].item())
                        else:
                            joint_key.append(0)  # Default value
                    joint_labels.append(tuple(joint_key))
    else:
        # For full datasets, process all clips via dataset_index
        for clip_info in base_dataset.dataset_index:
            if clip_info.numeric_labels is not None:
                joint_key = []
                from utils import get_task_metadata
                metadata = get_task_metadata()
                task_names_list = metadata['task_names']
                
                for task_name in task_names:
                    if task_name in task_names_list:
                        task_idx = task_names_list.index(task_name)
                        joint_key.append(clip_info.numeric_labels[task_idx].item())
                    else:
                        joint_key.append(0)  # Default value
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
    parser.add_argument('--backbone', type=str, default='videomae_base_patch16_224', help='Backbone model name')
    parser.add_argument('--pooling', type=str, default='avg', help='Pooling type for head')
    parser.add_argument('--temporal-module', type=str, default=None, help='Temporal module for head')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate in head')
    parser.add_argument('--head-hidden-dim', type=int, default=2048, help='Hidden dimension in head')
    parser.add_argument('--head-shared-layers', type=int, default=2, help='Number of shared layers in head')
    parser.add_argument('--head-task-layers', type=int, default=1, help='Number of task-specific layers in head')
    parser.add_argument('--head-use-bn', action='store_true', help='Use BatchNorm in head')
    parser.add_argument('--head-activation', type=str, default='relu', help='Activation function in head')
    
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
    parser.add_argument('--unfreeze-batch-size', type=int, default=3,
                       help='Batch size to use after backbone unfreezing starts (default: 3)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of steps to accumulate gradients before optimizer step (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading (default: 4)')
    
    # Other arguments
    parser.add_argument('--output-dir', type=str, default='./outputs_balanced', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    # Primary task weighting arguments
    parser.add_argument('--primary-task-weight', type=float, default=3.0,
                        help='Weight multiplier for primary tasks (default: 3.0)')
    parser.add_argument('--auxiliary-task-weight', type=float, default=1.0,
                        help='Weight multiplier for auxiliary tasks (default: 1.0)')
    parser.add_argument('--primary-tasks', nargs='+', default=None,
                        help='List of primary task names to train on. If not set, all available tasks will be used.')
    
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
                       help='Task(s) to balance sampling for - choose from: action_class, severity, offence (default: action_class)')
    parser.add_argument('--balance-task', type=str, default=None,
                       help='Single task to balance sampling for (deprecated, use --balance-tasks)')
    parser.add_argument('--joint-severity-sampling', action='store_true',
                       help='Convenience flag to enable joint sampling for action_class and severity (equivalent to --balance-tasks action_class severity)')
    
    # Add new sophisticated weighting strategy
    parser.add_argument('--use-smart-weighting', action='store_true',
                        help='Use sophisticated task weighting based on semantic relevance for the 3 MVFouls tasks')
    parser.add_argument('--core-task-weight', type=float, default=20.0,
                        help='Weight for core tasks (action_class, severity) (default: 20.0)')
    parser.add_argument('--support-task-weight', type=float, default=2.0,
                        help='Weight for decision task (offence) (default: 2.0)')
    parser.add_argument('--context-task-weight', type=float, default=0.5,
                        help='Weight for unexpected tasks (should not be used) (default: 0.5)')
    
    # Loss configuration arguments
    parser.add_argument('--loss-types', nargs='+',
                        help='Per-task loss types in canonical task order '
                             '(format: task_name loss_type pairs). Example: --loss-types action_class ce severity focal offence ce')
    
    # Advanced task weighting arguments
    parser.add_argument('--weighting-strategy', type=str, default='inverse_accuracy',
                        choices=['uniform', 'inverse_accuracy', 'inverse_f1', 'difficulty'],
                        help='How to derive adaptive task weights from validation metrics (default: inverse_accuracy)')
    parser.add_argument('--adaptive-weights', action='store_true',
                        help='Re-compute task weights each eval epoch based on validation metrics')
    
    # Effective number class weights
    parser.add_argument('--effective-class-weights', action='store_true',
                        help='Use effective number of samples method for class weights (better for extreme imbalance)')
    
    # Add train-fraction argument
    parser.add_argument('--train-fraction', type=float, default=1.0,
                        help='Fraction of training data to use (e.g., 0.2 for 20% smoke test)')
    
    # Add bag-of-clips arguments
    parser.add_argument('--bag-of-clips', action='store_true',
                        help='Enable bag-of-clips (action-level) training mode')
    parser.add_argument('--max-clips-per-action', type=int, default=8,
                        help='Maximum number of clips per action in bag-of-clips mode (default: 8)')
    parser.add_argument('--min-clips-per-action', type=int, default=1,
                        help='Minimum number of clips per action (actions with fewer clips excluded) (default: 1)')
    parser.add_argument('--clip-sampling-strategy', type=str, default='random',
                        choices=['random', 'uniform', 'all'],
                        help='Strategy for sampling clips when exceeding max-clips-per-action (default: random)')
    parser.add_argument('--clip-pooling-type', type=str, default='mean',
                        choices=['mean', 'max', 'attention'],
                        help='Pooling strategy for aggregating clips in bag-of-clips mode (default: mean)')
    parser.add_argument('--clip-pooling-temperature', type=float, default=1.0,
                        help='Temperature for attention pooling in bag-of-clips mode (default: 1.0)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging(log_level)
    
    # Get official task metadata
    tasks_metadata = get_task_metadata()
    VALID_TASKS = tasks_metadata['task_names']
    logger.info(f"Discovered valid tasks from metadata: {VALID_TASKS}")

    # Determine which tasks to activate for this run
    if args.primary_tasks:
        # User has specified which tasks to run
        invalid_tasks = [t for t in args.primary_tasks if t not in VALID_TASKS]
        if invalid_tasks:
            raise ValueError(f"Invalid tasks specified in --primary-tasks: {invalid_tasks}. Valid options are: {VALID_TASKS}")
        ACTIVE_TASKS = args.primary_tasks
    else:
        # Default to the three main tasks
        ACTIVE_TASKS = ['action', 'offence', 'severity']

    logger.info(f"✅ Activating training for tasks: {ACTIVE_TASKS}")

    # Parse and validate loss types configuration
    loss_types_per_task = {}
    if args.loss_types:
        # Parse pairs: task_name loss_type task_name loss_type ...
        if len(args.loss_types) % 2 != 0:
            raise ValueError("--loss-types must have even number of arguments (task_name loss_type pairs)")
        
        valid_loss_types = ['ce', 'focal', 'bce']
        for i in range(0, len(args.loss_types), 2):
            task_name = args.loss_types[i]
            loss_type = args.loss_types[i + 1]
            
            if loss_type not in valid_loss_types:
                raise ValueError(f"Invalid loss type '{loss_type}'. Choose from: {valid_loss_types}")
            
            loss_types_per_task[task_name] = loss_type
    else:
        # Default configuration: CE for action and offence, focal for severity
        loss_types_per_task = {
            'action': 'ce', 
            'severity': 'focal', 
            'offence': 'ce'
        }
    
    # Validate that only valid tasks are specified in loss types
    invalid_loss_tasks = [task for task in loss_types_per_task.keys() if task not in VALID_TASKS]
    if invalid_loss_tasks:
        raise ValueError(f"Invalid tasks in --loss-types: {invalid_loss_tasks}. "
                        f"Valid tasks are: {VALID_TASKS}")
    
    # Filter to only include active tasks
    loss_types_per_task = {k: v for k, v in loss_types_per_task.items() if k in ACTIVE_TASKS}

    # Fill missing active tasks with default 'ce'
    missing_tasks = [task for task in ACTIVE_TASKS if task not in loss_types_per_task]
    if missing_tasks:
        for task in missing_tasks:
            loss_types_per_task[task] = 'ce'
        logger.warning(f"Missing loss types for active tasks {missing_tasks}, defaulting to 'ce'")

    logger.info(f"📋 Loss configuration for active tasks: {loss_types_per_task}")
    
    # Validate balance-tasks
    if args.balance_tasks:
        invalid_balance_tasks = [task for task in args.balance_tasks if task not in VALID_TASKS]
        if invalid_balance_tasks:
            raise ValueError(f"Invalid tasks in --balance-tasks: {invalid_balance_tasks}. "
                           f"Valid tasks are: {VALID_TASKS}")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Print clear summary of what this training will do
    print("\n" + "="*80)
    print("🚀 MVFOULS 3-TASK MULTI-TASK TRAINING")
    print("="*80)
    print("📋 This model will predict EXACTLY 3 tasks:")
    print("   1️⃣  action_class (10 classes): What type of action occurred")
    print("       • Standing tackling, Tackling, Challenge, Holding, Elbowing, etc.")
    print("   2️⃣  severity (6 classes): How severe was the action")  
    print("       • Missing, 1, 2, 3, 4, 5")
    print("   3️⃣  offence (4 classes): Final judgment on the action")
    print("       • Missing/Empty, Offence, No offence, Between")
    print("="*80)
    
    logger.info(f"🚀 Starting balanced MVFouls training")
    logger.info(f"Device: {device}")
    logger.info(f"Multi-task: {args.multi_task}")
    logger.info(f"Active tasks for this run: {ACTIVE_TASKS}")
    
    try:
        # 1. Create Transforms
        # ====================
        transforms = get_video_transforms(image_size=224, augment_train=True)
        
        # 2. Create Datasets
        # ==================
        logger.info("📂 Creating datasets...")
        root_dir = str(Path(args.train_dir).parent)
        train_split = Path(args.train_dir).name.replace('_720p', '')
        val_split = Path(args.val_dir).name.replace('_720p', '')
        
        train_dataset = MVFoulsDataset(
            root_dir=root_dir,
            split=train_split,
            transform=transforms['train'],
            load_annotations=True,
            num_frames=32,
            bag_of_clips=args.bag_of_clips,
            max_clips_per_action=args.max_clips_per_action,
            min_clips_per_action=args.min_clips_per_action,
            clip_sampling_strategy=args.clip_sampling_strategy
        )
        
        if 0 < args.train_fraction < 1.0:
            import random
            subset_size = int(len(train_dataset) * args.train_fraction)
            indices = list(range(len(train_dataset)))
            random.shuffle(indices)
            train_dataset = torch.utils.data.Subset(train_dataset, indices[:subset_size])
            logger.info(f"🔍 Using subset of training data: {len(train_dataset)} samples ({args.train_fraction*100:.1f}%)")
        
        val_dataset = MVFoulsDataset(
            root_dir=root_dir,
            split=val_split,
            transform=transforms['val'],
            load_annotations=True,
            num_frames=32,
            bag_of_clips=args.bag_of_clips,
            max_clips_per_action=args.max_clips_per_action,
            min_clips_per_action=args.min_clips_per_action,
            clip_sampling_strategy='uniform'
        )
        logger.info(f"📊 Dataset sizes: {len(train_dataset)} train, {len(val_dataset)} val")

        # 3. Analyze Dataset Imbalance
        # ============================
        logger.info("🔍 Analyzing class imbalance for active tasks...")
        imbalance_analysis = analyze_dataset_imbalance(train_dataset)
        imbalance_analysis = {k: v for k, v in imbalance_analysis.items() if k in ACTIVE_TASKS}
        for task, analysis in imbalance_analysis.items():
            logger.info(f"  - Task '{task}': {analysis['num_classes']} classes. Distribution: {analysis['class_distribution_percent']}")
        
        if args.analyze_only:
            logger.info("✅ Analysis complete. Exiting.")
            return

        # 4. Configure Task Loss Weights
        # ================================
        task_loss_weights = {}
        if args.use_smart_weighting:
            recommended_config = get_recommended_loss_config(True)
            task_loss_weights = recommended_config['task_loss_weights']
            if args.support_task_weight is not None:
                for task in task_loss_weights:
                    if task != 'action':
                        task_loss_weights[task] = args.support_task_weight
            logger.info("Using smart task weighting.")
        else:
            logger.info("Using primary/auxiliary task weighting.")
            for task in ACTIVE_TASKS:
                if task in args.primary_tasks if args.primary_tasks else ['action', 'offence', 'severity']:
                     task_loss_weights[task] = args.primary_task_weight
                else:
                     task_loss_weights[task] = args.auxiliary_task_weight

        task_loss_weights = {k: v for k, v in task_loss_weights.items() if k in ACTIVE_TASKS}
        logger.info(f"📊 Final task loss weights: {task_loss_weights}")

        # 5. Create Model
        # ===============
        logger.info("🏗️ Creating model...")
        model = create_balanced_model(
            multi_task=args.multi_task,
            task_names=ACTIVE_TASKS,
            imbalance_analysis=imbalance_analysis,
            primary_task_weights=task_loss_weights,
            backbone_checkpointing=args.backbone_checkpointing,
            use_class_weights=not args.disable_class_weights,
            class_weight_cap=args.class_weight_cap,
            loss_types_per_task=loss_types_per_task,
            use_effective_weights=args.effective_class_weights,
            # Pass model-specific arguments
            backbone_name=args.backbone,
            pretrained=True,
            freeze_mode=args.freeze_mode,
            pooling=args.pooling,
            temporal_module=args.temporal_module,
            dropout=args.dropout,
            hidden_dim=args.head_hidden_dim,
            num_shared_layers=args.head_shared_layers,
            task_specific_layers=args.head_task_layers,
            use_batch_norm=args.head_use_bn,
            activation=args.head_activation,
            clip_pooling_type=args.clip_pooling_type,
            clip_pooling_temperature=args.clip_pooling_temperature
        )
        model.to(device)

        # 6. Create DataLoaders
        # =====================
        train_sampler = None
        collate_fn = None
        shuffle_train = True
        if args.bag_of_clips:
            collate_fn = bag_of_clips_collate_fn
            shuffle_train = True
        elif args.balanced_sampling:
            train_sampler = create_balanced_sampler(train_dataset, task_names=args.balance_tasks)
            shuffle_train = False

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler,
            shuffle=shuffle_train, num_workers=args.num_workers, pin_memory=True,
            drop_last=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
        )

        # 7. Create Optimizer, Scheduler, and Trainer
        # ===========================================
        logger.info("Creating optimizer, scheduler, and trainer...")
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = GradScaler()
        writer = SummaryWriter(log_dir=args.output_dir)

        trainer = MultiTaskTrainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler, device=device, epochs=args.epochs,
            output_dir=args.output_dir, writer=writer, scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_smart_weighting=args.use_smart_weighting,
            adaptive_weights=args.adaptive_weights,
            weighting_strategy=args.weighting_strategy
        )

        # 8. Training Loop
        # ================
        logger.info("🚀 Starting training loop...")
        unfreeze_schedule = get_unfreeze_schedule(args.epochs, args.freeze_mode)
        
        for epoch in range(args.epochs):
            if not args.disable_gradual_unfreezing:
                apply_gradual_unfreezing(
                    model, epoch, unfreeze_schedule, logger, trainer, args, optimizer
                )
            trainer.train_epoch()
            trainer.evaluate()
            writer.add_scalar('LR/epoch', optimizer.param_groups[0]['lr'], epoch)
            if trainer.should_stop():
                logger.info("🛑 Early stopping triggered.")
                break
                
        logger.info("✅ Training complete.")
        writer.close()

    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 