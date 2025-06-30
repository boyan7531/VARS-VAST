"""
Advanced Data Loading Utilities for MVFouls Dataset

This module implements advanced data augmentation techniques that require
batch-level operations, specifically:
- Video MixUp / TimeMix: Mix clips with same action_class labels
- ClipCutMix: Spatial CutMix between minority class clips
- Enhanced minority class sampling strategies

These techniques are applied at the batch level through custom collate functions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict

# Import dataset utilities
try:
    from dataset import TASKS_INFO, LABEL2IDX, IDX2LABEL
    from utils import get_task_metadata
except ImportError:
    print("Warning: Could not import dataset utilities. Some functions may not work.")
    TASKS_INFO = None
    LABEL2IDX = None
    IDX2LABEL = None


def beta_distribution_sample(alpha: float = 0.3, beta: float = 0.3) -> float:
    """
    Sample from Beta distribution for MixUp lambda parameter.
    
    Args:
        alpha: Beta distribution alpha parameter
        beta: Beta distribution beta parameter
        
    Returns:
        float: Sampled lambda value
    """
    return np.random.beta(alpha, beta)


def mixup_data(video1: torch.Tensor, video2: torch.Tensor, 
               targets1: Dict[str, torch.Tensor], targets2: Dict[str, torch.Tensor],
               lam: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Apply MixUp to video data and targets.
    
    Args:
        video1: First video tensor (C, T, H, W)
        video2: Second video tensor (C, T, H, W)
        targets1: First target dict {task_name: target_tensor}
        targets2: Second target dict {task_name: target_tensor}
        lam: MixUp lambda parameter
        
    Returns:
        Tuple of mixed video and mixed targets
    """
    # Mix videos
    mixed_video = lam * video1 + (1 - lam) * video2
    
    # Mix targets (create soft labels)
    mixed_targets = {}
    for task_name in targets1.keys():
        if task_name in targets2:
            # Convert targets to one-hot if they're not already
            target1 = targets1[task_name]
            target2 = targets2[task_name]
            
            if len(target1.shape) == 0:  # Scalar target
                target1 = target1.unsqueeze(0)
            if len(target2.shape) == 0:  # Scalar target
                target2 = target2.unsqueeze(0)
            
            # For multi-task learning, we keep hard labels for the primary task
            # and create soft labels for auxiliary tasks
            if task_name == 'action_class':
                # Keep hard labels for action_class since we only mix same-class clips
                mixed_targets[task_name] = target1  # They should be the same
            else:
                # Create soft labels for auxiliary tasks
                num_classes = len(TASKS_INFO[task_name]) if TASKS_INFO else 2
                
                # Convert to one-hot
                target1_onehot = F.one_hot(target1.long(), num_classes).float()
                target2_onehot = F.one_hot(target2.long(), num_classes).float()
                
                # Mix and convert back to class indices for loss computation
                mixed_onehot = lam * target1_onehot + (1 - lam) * target2_onehot
                mixed_targets[task_name] = mixed_onehot
        else:
            mixed_targets[task_name] = targets1[task_name]
    
    return mixed_video, mixed_targets


def cutmix_data(video1: torch.Tensor, video2: torch.Tensor,
                targets1: Dict[str, torch.Tensor], targets2: Dict[str, torch.Tensor],
                lam: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Apply CutMix to video data and targets.
    
    Args:
        video1: First video tensor (C, T, H, W)
        video2: Second video tensor (C, T, H, W)
        targets1: First target dict {task_name: target_tensor}
        targets2: Second target dict {task_name: target_tensor}
        lam: CutMix lambda parameter
        
    Returns:
        Tuple of mixed video and mixed targets
    """
    C, T, H, W = video1.shape
    
    # Calculate cut area
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    
    # Random cut position
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cut to all frames
    mixed_video = video1.clone()
    mixed_video[:, :, bby1:bby2, bbx1:bbx2] = video2[:, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual cut area
    actual_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    # Mix targets for all tasks
    mixed_targets = {}
    for task_name in targets1.keys():
        if task_name in targets2:
            target1 = targets1[task_name]
            target2 = targets2[task_name]
            
            if len(target1.shape) == 0:  # Scalar target
                target1 = target1.unsqueeze(0)
            if len(target2.shape) == 0:  # Scalar target
                target2 = target2.unsqueeze(0)
            
            # Create soft labels for all tasks in CutMix
            num_classes = len(TASKS_INFO[task_name]) if TASKS_INFO else 2
            
            # Convert to one-hot
            target1_onehot = F.one_hot(target1.long(), num_classes).float()
            target2_onehot = F.one_hot(target2.long(), num_classes).float()
            
            # Mix with actual lambda
            mixed_onehot = actual_lam * target1_onehot + (1 - actual_lam) * target2_onehot
            mixed_targets[task_name] = mixed_onehot
        else:
            mixed_targets[task_name] = targets1[task_name]
    
    return mixed_video, mixed_targets


def is_minority_class(targets: Dict[str, torch.Tensor], 
                     minority_threshold: float = 0.1) -> bool:
    """
    Check if a sample belongs to minority class for action_class task.
    
    Args:
        targets: Target dict {task_name: target_tensor}
        minority_threshold: Threshold for considering a class as minority
        
    Returns:
        bool: True if sample belongs to minority class
    """
    if 'action_class' not in targets:
        return False
    
    action_class = targets['action_class'].item()
    
    # Define minority classes (these are typically rare in MVFouls)
    # Based on the TASKS_INFO: ["Missing/Empty", "Standing tackling", "Tackling", 
    # "Challenge", "Holding", "Elbowing", "High leg", "Pushing", "Dont know", "Dive"]
    minority_classes = {5, 6, 9}  # "Elbowing", "High leg", "Dive"
    
    return action_class in minority_classes


def create_mixup_cutmix_collate_fn(
    mixup_prob: float = 0.5,
    cutmix_prob: float = 0.3,
    mixup_alpha: float = 0.3,
    cutmix_alpha: float = 1.0,
    minority_only_cutmix: bool = True,
    same_class_mixup: bool = True
):
    """
    Create a collate function that applies MixUp and CutMix augmentations.
    
    Args:
        mixup_prob: Probability of applying MixUp
        cutmix_prob: Probability of applying CutMix
        mixup_alpha: Alpha parameter for MixUp beta distribution
        cutmix_alpha: Alpha parameter for CutMix beta distribution
        minority_only_cutmix: Only apply CutMix to minority class pairs
        same_class_mixup: Only mix clips with same action_class label
        
    Returns:
        Collate function for DataLoader
    """
    
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Custom collate function with MixUp/CutMix augmentation.
        
        Args:
            batch: List of (video, targets) tuples
            
        Returns:
            Batched and potentially mixed data
        """
        if len(batch) < 2:
            # Not enough samples for mixing, use default collate
            videos = torch.stack([item[0] for item in batch])
            targets = torch.stack([item[1] for item in batch])
            
            # Convert targets to dict format
            if TASKS_INFO:
                task_names = list(TASKS_INFO.keys())
                targets_dict = {}
                for i, task_name in enumerate(task_names):
                    targets_dict[task_name] = targets[:, i]
            else:
                targets_dict = {'default': targets}
            
            return videos, targets_dict
        
        # Separate videos and targets
        videos = [item[0] for item in batch]
        targets_raw = [item[1] for item in batch]
        
        # Convert targets to dict format
        targets_list = []
        if TASKS_INFO:
            task_names = list(TASKS_INFO.keys())
            for target in targets_raw:
                target_dict = {}
                for i, task_name in enumerate(task_names):
                    target_dict[task_name] = target[i] if len(target.shape) > 0 else target
                targets_list.append(target_dict)
        else:
            targets_list = [{'default': target} for target in targets_raw]
        
        batch_size = len(videos)
        mixed_videos = []
        mixed_targets_list = []
        
        # Track which samples have been processed
        processed = set()
        
        for i in range(batch_size):
            if i in processed:
                continue
                
            video_i = videos[i]
            targets_i = targets_list[i]
            
            # Decide whether to apply augmentation
            apply_mixup = random.random() < mixup_prob
            apply_cutmix = random.random() < cutmix_prob
            
            if (apply_mixup or apply_cutmix) and len(processed) < batch_size - 1:
                # Find a suitable partner
                partner_found = False
                
                for j in range(i + 1, batch_size):
                    if j in processed:
                        continue
                    
                    video_j = videos[j]
                    targets_j = targets_list[j]
                    
                    # Check constraints
                    can_mix = True
                    
                    if same_class_mixup and 'action_class' in targets_i and 'action_class' in targets_j:
                        if targets_i['action_class'] != targets_j['action_class']:
                            can_mix = False
                    
                    if apply_cutmix and minority_only_cutmix:
                        if not (is_minority_class(targets_i) and is_minority_class(targets_j)):
                            apply_cutmix = False
                            if not apply_mixup:
                                can_mix = False
                    
                    if can_mix:
                        # Apply augmentation
                        if apply_cutmix:
                            lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                            mixed_video, mixed_targets = cutmix_data(
                                video_i, video_j, targets_i, targets_j, lam
                            )
                        elif apply_mixup:
                            lam = np.random.beta(mixup_alpha, mixup_alpha)
                            mixed_video, mixed_targets = mixup_data(
                                video_i, video_j, targets_i, targets_j, lam
                            )
                        else:
                            mixed_video, mixed_targets = video_i, targets_i
                        
                        mixed_videos.append(mixed_video)
                        mixed_targets_list.append(mixed_targets)
                        
                        processed.add(i)
                        processed.add(j)
                        partner_found = True
                        break
                
                if not partner_found:
                    # No suitable partner found, use original
                    mixed_videos.append(video_i)
                    mixed_targets_list.append(targets_i)
                    processed.add(i)
            else:
                # No augmentation
                mixed_videos.append(video_i)
                mixed_targets_list.append(targets_i)
                processed.add(i)
        
        # Stack videos
        batch_videos = torch.stack(mixed_videos)
        
        # Combine targets
        batch_targets = {}
        for task_name in mixed_targets_list[0].keys():
            task_targets = []
            for target_dict in mixed_targets_list:
                task_targets.append(target_dict[task_name])
            batch_targets[task_name] = torch.stack(task_targets)
        
        return batch_videos, batch_targets
    
    return collate_fn


def create_minority_boost_sampler(dataset, boost_factor: float = 2.0, 
                                 task_name: str = 'action_class') -> torch.utils.data.WeightedRandomSampler:
    """
    Create a weighted sampler that boosts minority classes.
    
    Args:
        dataset: MVFouls dataset instance
        boost_factor: Multiplication factor for minority class weights
        task_name: Task to balance (default: 'action_class')
        
    Returns:
        WeightedRandomSampler instance
    """
    # Get labels for the specified task
    labels = []
    for i in range(len(dataset)):
        _, targets = dataset[i]
        if isinstance(targets, dict):
            label = targets[task_name].item()
        else:
            # Assume targets is tensor and extract task index
            task_idx = list(TASKS_INFO.keys()).index(task_name)
            label = targets[task_idx].item()
        labels.append(label)
    
    # Compute class weights
    class_counts = torch.bincount(torch.tensor(labels))
    total_samples = len(labels)
    
    # Standard inverse frequency weights
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    
    # Boost minority classes
    minority_classes = {5, 6, 9}  # "Elbowing", "High leg", "Dive"
    for cls in minority_classes:
        if cls < len(class_weights):
            class_weights[cls] *= boost_factor
    
    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]
    
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


if __name__ == "__main__":
    # Test the collate function
    print("🧪 Testing Advanced Data Loading Utilities")
    print("=" * 50)
    
    # Create dummy batch data
    batch_size = 4
    dummy_videos = [torch.randn(3, 32, 224, 224) for _ in range(batch_size)]
    dummy_targets = [torch.tensor([1, 2, 1]) for _ in range(batch_size)]  # Same action_class
    dummy_batch = list(zip(dummy_videos, dummy_targets))
    
    # Test collate function
    collate_fn = create_mixup_cutmix_collate_fn(
        mixup_prob=1.0,  # Force apply for testing
        cutmix_prob=0.0,
        same_class_mixup=True
    )
    
    try:
        batched_videos, batched_targets = collate_fn(dummy_batch)
        print(f"✅ MixUp collate function works!")
        print(f"   Video batch shape: {batched_videos.shape}")
        print(f"   Target keys: {list(batched_targets.keys())}")
        
        for task, targets in batched_targets.items():
            print(f"   {task} shape: {targets.shape}")
    except Exception as e:
        print(f"❌ MixUp collate function failed: {e}")
    
    # Test CutMix
    collate_fn_cutmix = create_mixup_cutmix_collate_fn(
        mixup_prob=0.0,
        cutmix_prob=1.0,  # Force apply for testing
        minority_only_cutmix=False  # Disable for testing
    )
    
    try:
        batched_videos, batched_targets = collate_fn_cutmix(dummy_batch)
        print(f"✅ CutMix collate function works!")
        print(f"   Video batch shape: {batched_videos.shape}")
    except Exception as e:
        print(f"❌ CutMix collate function failed: {e}")
    
    print("\n✨ Advanced data loading utilities ready!") 