import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Union
import math


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss
    
    Paper: Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
    https://arxiv.org/abs/1906.07413
    
    The key idea is to apply class-dependent margins to the logits before computing
    cross-entropy loss. Minority classes get larger margins, which pushes their
    decision boundaries further out and improves recall.
    
    Args:
        cls_num_list: List of number of samples per class
        max_m: Maximum margin value (default: 0.5)
        weight: Optional class weights tensor 
        s: Scale parameter (default: 30.0)
        reduction: Loss reduction ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        cls_num_list: List[int],
        max_m: float = 0.5,
        weight: Optional[torch.Tensor] = None,
        s: float = 30.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.cls_num_list = cls_num_list
        self.max_m = max_m
        self.s = s
        self.reduction = reduction
        
        # Compute margins based on class frequencies
        # m_c = C / n_c^(1/4) where C is chosen so max margin = max_m
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        
        # Avoid division by zero
        cls_num_list = torch.clamp(cls_num_list, min=1.0)
        
        # Compute raw margins (inversely proportional to class frequency)
        m_list = 1.0 / (cls_num_list ** 0.25)
        
        # Scale so maximum margin equals max_m
        m_list = m_list * (max_m / m_list.max())
        
        # Register as buffer so it moves with the model
        self.register_buffer('m_list', m_list)
        
        # Register class weights
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Apply LDAM loss.
        
        Args:
            logits: Raw logits tensor of shape (batch_size, num_classes)
            targets: Target class indices of shape (batch_size,)
            
        Returns:
            LDAM loss value
        """
        batch_size, num_classes = logits.shape
        
        # Create margin tensor for this batch
        batch_m = self.m_list[targets]  # Shape: (batch_size,)
        
        # Apply margins to the true class logits
        # For each sample, subtract the margin from the logit of the true class
        logits_m = logits.clone()
        logits_m[range(batch_size), targets] -= batch_m
        
        # Scale logits
        logits_m = logits_m * self.s
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits_m, targets, weight=self.weight, reduction=self.reduction)
        
        return loss
    
    def update_class_counts(self, cls_num_list: List[int]):
        """Update class counts and recompute margins."""
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        cls_num_list = torch.clamp(cls_num_list, min=1.0)
        
        # Recompute margins
        m_list = 1.0 / (cls_num_list ** 0.25)
        m_list = m_list * (self.max_m / m_list.max())
        
        # Update buffer
        self.m_list.data = m_list.to(self.m_list.device)


class MultiTaskLDAMLoss(nn.Module):
    """
    Multi-task LDAM loss for handling multiple classification tasks.
    Each task can have its own class distribution and margins.
    """
    
    def __init__(
        self,
        task_cls_num_lists: Dict[str, List[int]],
        max_m: float = 0.5,
        task_weights: Optional[Dict[str, torch.Tensor]] = None,
        s: float = 30.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.task_names = list(task_cls_num_lists.keys())
        self.max_m = max_m
        self.s = s
        self.reduction = reduction
        
        # Create LDAM loss for each task
        self.task_losses = nn.ModuleDict()
        for task_name, cls_num_list in task_cls_num_lists.items():
            task_weight = task_weights.get(task_name, None) if task_weights else None
            self.task_losses[task_name] = LDAMLoss(
                cls_num_list=cls_num_list,
                max_m=max_m,
                weight=task_weight,
                s=s,
                reduction=reduction
            )
    
    def forward(
        self,
        logits_dict: Dict[str, torch.Tensor],
        targets_dict: Dict[str, torch.Tensor],
        task_loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task LDAM loss.
        
        Args:
            logits_dict: Dict mapping task names to logits tensors
            targets_dict: Dict mapping task names to target tensors
            task_loss_weights: Optional weights for combining task losses
            
        Returns:
            Dict with individual task losses and total loss
        """
        loss_dict = {}
        total_loss = 0.0
        
        task_weights = task_loss_weights or {}
        
        for task_name in self.task_names:
            if task_name not in logits_dict or task_name not in targets_dict:
                continue
                
            logits = logits_dict[task_name]
            targets = targets_dict[task_name]
            
            # Compute LDAM loss for this task
            task_loss = self.task_losses[task_name](logits, targets)
            
            # Apply task weight
            task_weight = task_weights.get(task_name, 1.0)
            weighted_task_loss = task_loss * task_weight
            
            loss_dict[f'{task_name}_loss'] = task_loss
            loss_dict[f'{task_name}_weighted_loss'] = weighted_task_loss
            total_loss += weighted_task_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def update_class_counts(self, task_cls_num_lists: Dict[str, List[int]]):
        """Update class counts for all tasks."""
        for task_name, cls_num_list in task_cls_num_lists.items():
            if task_name in self.task_losses:
                self.task_losses[task_name].update_class_counts(cls_num_list)


def compute_class_counts_from_targets(targets: torch.Tensor, num_classes: int) -> List[int]:
    """
    Compute class counts from target tensor.
    
    Args:
        targets: Target tensor of shape (num_samples,)
        num_classes: Total number of classes
        
    Returns:
        List of counts per class
    """
    class_counts = []
    for c in range(num_classes):
        count = (targets == c).sum().item()
        class_counts.append(max(count, 1))  # Avoid zero counts
    return class_counts


def compute_multi_task_class_counts(
    targets_dict: Dict[str, torch.Tensor],
    num_classes_per_task: Dict[str, int]
) -> Dict[str, List[int]]:
    """
    Compute class counts for multiple tasks.
    
    Args:
        targets_dict: Dict mapping task names to target tensors
        num_classes_per_task: Dict mapping task names to number of classes
        
    Returns:
        Dict mapping task names to class count lists
    """
    task_cls_counts = {}
    
    for task_name, targets in targets_dict.items():
        if task_name in num_classes_per_task:
            num_classes = num_classes_per_task[task_name]
            class_counts = compute_class_counts_from_targets(targets, num_classes)
            task_cls_counts[task_name] = class_counts
    
    return task_cls_counts 