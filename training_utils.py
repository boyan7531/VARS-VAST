"""
Training utilities for multi-task MVFouls learning.

This module provides utilities for:
- Multi-task training loops
- Learning rate scheduling per task
- Curriculum learning strategies
- Task balancing and weighting
- Training monitoring and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import numpy as np
import time
import logging
from collections import defaultdict
from tqdm import tqdm

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from utils import get_task_metadata, compute_task_metrics, format_metrics_table
    from model.mvfouls_model import MVFoulsModel
except ImportError:
    print("Warning: Some imports failed. Make sure utils.py and model/mvfouls_model.py are available.")


class MultiTaskTrainer:
    """
    Trainer for multi-task MVFouls learning using the complete MVFoulsModel.
    """
    
    def __init__(
        self,
        model: MVFoulsModel,
        optimizer: optim.Optimizer,
        device: torch.device = torch.device('cpu'),
        task_schedulers: Optional[Dict[str, Any]] = None,
        weighting_strategy: str = 'uniform',
        curriculum_strategy: Optional[str] = None,
        log_interval: int = 100,
        eval_interval: int = 1000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        primary_task_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize multi-task trainer.
        
        Args:
            model: Complete MVFoulsModel instance
            optimizer: Optimizer for training
            device: Training device
            task_schedulers: Optional per-task learning rate schedulers
            weighting_strategy: Strategy for task loss weighting
            curriculum_strategy: Strategy for curriculum learning
            log_interval: Steps between logging
            eval_interval: Steps between evaluation
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            primary_task_weights: Optional weights for primary vs auxiliary tasks
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.task_schedulers = task_schedulers or {}
        self.weighting_strategy = weighting_strategy
        self.curriculum_strategy = curriculum_strategy
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.primary_task_weights = primary_task_weights or {}
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_metrics = {}
        self.training_history = defaultdict(list)
        
        # Task curriculum state
        self.task_difficulties = {}
        if hasattr(model, 'head') and hasattr(model.head, 'task_names'):
            self.active_tasks = set(model.head.task_names)
        elif model.multi_task and get_task_metadata is not None:
            metadata = get_task_metadata()
            self.active_tasks = set(metadata['task_names'])
        else:
            self.active_tasks = {'default'}
        
        # Move model to device
        self.model.to(device)
    
    def train_step(
        self,
        batch_data: Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        scaler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Perform a single training step.
        
        Args:
            batch_data: Either (videos, targets) for standard mode or 
                       (videos, targets, clip_masks, num_clips) for bag-of-clips mode
            scaler: Optional GradScaler for mixed precision training
            
        Returns:
            Dict with loss and metrics information
        """
        self.model.train()
        
        # Unpack batch data
        if len(batch_data) == 4:
            # Bag-of-clips mode: (videos, targets, clip_masks, num_clips)
            videos, targets, clip_masks, num_clips = batch_data
            clip_masks = clip_masks.to(self.device)
        else:
            # Standard mode: (videos, targets)
            videos, targets = batch_data
            clip_masks = None
        
        # Move inputs to device
        videos = videos.to(self.device)
        
        # Handle target format
        if self.model.multi_task:
            if not isinstance(targets, dict):
                # Convert tensor to dict format
                targets = targets.to(self.device)
                if get_task_metadata is not None:
                    metadata = get_task_metadata()
                    targets_dict = {}
                    for i, task_name in enumerate(metadata['task_names']):
                        targets_dict[task_name] = targets[:, i]
                else:
                    targets_dict = {'default': targets[:, 0]}
            else:
                targets_dict = {k: v.to(self.device) for k, v in targets.items()}
        else:
            if isinstance(targets, dict):
                targets = torch.cat(list(targets.values()), dim=1)
            targets = targets.to(self.device)
            
            # For single-task training, extract the main task (offence detection)
            # Get offence task index from metadata
            if targets.dim() > 1 and get_task_metadata is not None:
                metadata = get_task_metadata()
                task_names = metadata['task_names']
                if 'offence' in task_names:
                    offence_idx = task_names.index('offence')
                    if targets.size(1) > offence_idx:
                        offence_targets = targets[:, offence_idx]  # Shape: [batch_size]
                        # Convert offence labels: 0: Missing/Empty -> 0, 1: Offence -> 1, 2: No offence -> 0
                        targets = (offence_targets == 1).long()  # Convert to binary
            
            targets_dict = targets
        
        # Filter active tasks for curriculum learning
        if self.curriculum_strategy and isinstance(targets_dict, dict):
            filtered_targets = {k: v for k, v in targets_dict.items() if k in self.active_tasks}
        else:
            filtered_targets = targets_dict
        
        # Forward pass
        if self.model.multi_task:
            logits_dict, extras = self.model(videos, clip_mask=clip_masks, return_dict=True)
            
            # Filter logits for curriculum learning
            if self.curriculum_strategy:
                filtered_logits = {k: v for k, v in logits_dict.items() if k in self.active_tasks}
            else:
                filtered_logits = logits_dict
            
            # Check if we should use unified adaptive loss
            if (hasattr(self, 'adaptive_weights') and self.adaptive_weights and 
                hasattr(self.model.head, 'compute_unified_loss')):
                # Use unified adaptive loss computation
                focal_gamma = getattr(self.model, 'task_focal_gamma_map', {}).get('default', 2.0)
                
                loss_dict = self.model.head.compute_unified_loss(
                    filtered_logits,
                    filtered_targets,
                    weighting_strategy=self.weighting_strategy,
                    focal_gamma=focal_gamma,
                    adaptive_weights=True
                )
                total_loss = loss_dict.get('total_loss', loss_dict.get('total_loss_adaptive', 0.0))
            else:
                # Use standard loss computation
                loss_dict = self.model.compute_loss(filtered_logits, filtered_targets, return_dict=True)
                total_loss = loss_dict['total_loss']
        else:
            logits, extras = self.model(videos, clip_mask=clip_masks, return_dict=False)
            targets = targets.to(self.device)
            
            # For single-task training, extract the main task (offence detection)
            if targets.dim() > 1 and get_task_metadata is not None:
                metadata = get_task_metadata()
                task_names = metadata['task_names']
                if 'offence' in task_names:
                    offence_idx = task_names.index('offence')
                    if targets.size(1) > offence_idx:
                        offence_targets = targets[:, offence_idx]  # Shape: [batch_size]
                        # Convert offence labels: 0: Missing/Empty -> 0, 1: Offence -> 1, 2: No offence -> 0
                        targets = (offence_targets == 1).long()  # Convert to binary
            
            # Compute loss
            loss_dict = self.model.compute_loss(logits, targets, return_dict=True)
            total_loss = loss_dict['total_loss']
        
        # Scale loss for gradient accumulation
        scaled_loss = total_loss / self.gradient_accumulation_steps
        
        # Backward pass with optional mixed precision
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Gradient accumulation
        if (self.step + 1) % self.gradient_accumulation_steps == 0:
            if scaler is not None:
                # Mixed precision gradient clipping and optimizer step
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # Standard gradient clipping and optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Update per-task schedulers
            for task_name, scheduler in self.task_schedulers.items():
                if hasattr(scheduler, 'step'):
                    scheduler.step()
        
        # Update training state
        self.step += 1
        
        return {
            'loss_dict': loss_dict,
            'total_loss': total_loss.item(),
            'extras': extras,
            'step': self.step
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Train for one complete epoch.
        
        Args:
            dataloader: Training dataloader
            scheduler: Optional main learning rate scheduler
            
        Returns:
            Dict with epoch training metrics
        """
        self.model.train()
        epoch_losses = []
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # Training step - pass the entire batch to train_step
            step_results = self.train_step(batch)
            epoch_losses.append(step_results['total_loss'])
            
            # Logging (disabled in favor of custom progress bar)
            # if (batch_idx + 1) % self.log_interval == 0:
            #     avg_loss = np.mean(epoch_losses[-self.log_interval:])
            #     print(f"  Batch {batch_idx + 1}/{len(dataloader)}: Loss = {avg_loss:.4f}")
            
            # Update main scheduler
            if scheduler is not None and hasattr(scheduler, 'step'):
                if hasattr(scheduler, 'step_update'):
                    # For schedulers that update per step
                    scheduler.step_update(self.step)
                else:
                    # For schedulers that update per epoch (will be called once per epoch)
                    pass
        
        # Update epoch-based scheduler
        if scheduler is not None and hasattr(scheduler, 'step'):
            if not hasattr(scheduler, 'step_update'):
                scheduler.step()
        
        self.epoch += 1
        epoch_time = time.time() - epoch_start_time
        
        return {
            'avg_loss': np.mean(epoch_losses),
            'epoch_time': epoch_time,
            'total_batches': len(dataloader)
        }
    
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        compute_detailed_metrics: bool = True,
        compute_task_metrics: Optional[callable] = None,
        format_metrics_table: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Evaluation dataloader
            max_batches: Maximum number of batches to evaluate (None = all)
            compute_detailed_metrics: Whether to compute detailed metrics
            
        Returns:
            Dict with evaluation results
        """
        self.model.eval()
        eval_losses = []
        all_logits = []
        all_targets = []
        
        # Create progress bar for evaluation
        eval_pbar = tqdm(
            dataloader,
            desc="Validation",
            leave=False,
            ncols=100,
            unit="batch"
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_pbar):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Unpack batch data
                if len(batch) == 4:
                    # Bag-of-clips mode: (videos, targets, clip_masks, num_clips)
                    videos, targets, clip_masks, num_clips = batch
                    clip_masks = clip_masks.to(self.device)
                else:
                    # Standard mode: (videos, targets)
                    videos, targets = batch
                    clip_masks = None
                
                videos = videos.to(self.device)
                
                # Forward pass
                if self.model.multi_task:
                    logits_dict, extras = self.model(videos, clip_mask=clip_masks, return_dict=True)
                    
                    # Handle target format for multi-task
                    if isinstance(targets, dict):
                        targets_dict = {k: v.to(self.device) for k, v in targets.items()}
                    else:
                        # Convert tensor to dict format for multi-task
                        targets = targets.to(self.device)
                        if get_task_metadata is not None:
                            metadata = get_task_metadata()
                            targets_dict = {}
                            for i, task_name in enumerate(metadata['task_names']):
                                targets_dict[task_name] = targets[:, i]
                        else:
                            targets_dict = {'default': targets[:, 0]}
                    
                    # Compute loss
                    loss_dict = self.model.compute_loss(logits_dict, targets_dict, return_dict=True)
                    eval_losses.append(loss_dict['total_loss'].item())
                    
                    all_logits.append(logits_dict)
                    all_targets.append(targets_dict)
                else:
                    print(f"DEBUG: Using single-task evaluation path")  # Debug line
                    logits, extras = self.model(videos, clip_mask=clip_masks, return_dict=False)
                    targets = targets.to(self.device)
                    
                    # For single-task training, extract the main task (offence detection)
                    if targets.dim() > 1 and get_task_metadata is not None:
                        metadata = get_task_metadata()
                        task_names = metadata['task_names']
                        if 'offence' in task_names:
                            offence_idx = task_names.index('offence')
                            if targets.size(1) > offence_idx:
                                offence_targets = targets[:, offence_idx]  # Shape: [batch_size]
                                # Convert offence labels: 0: Missing/Empty -> 0, 1: Offence -> 1, 2: No offence -> 0
                                targets = (offence_targets == 1).long()  # Convert to binary
                    
                    # Compute loss
                    loss_dict = self.model.compute_loss(logits, targets, return_dict=True)
                    eval_losses.append(loss_dict['total_loss'].item())
                    
                    all_logits.append(logits)
                    all_targets.append(targets)
        
        results = {
            'avg_loss': np.mean(eval_losses),
            'total_batches': len(all_logits)
        }
        
        # Compute detailed metrics if requested
        if compute_detailed_metrics and self.model.multi_task:
            # Combine all logits and targets
            combined_logits = {}
            combined_targets = {}
            
            for batch_logits, batch_targets in zip(all_logits, all_targets):
                for task_name in batch_logits.keys():
                    if task_name not in combined_logits:
                        combined_logits[task_name] = []
                        combined_targets[task_name] = []
                    
                    combined_logits[task_name].append(batch_logits[task_name])
                    if isinstance(batch_targets, dict):
                        combined_targets[task_name].append(batch_targets[task_name])
                    else:
                        # Handle tensor targets (convert to dict format)
                        if batch_targets.dim() > 1 and get_task_metadata is not None:
                            metadata = get_task_metadata()
                            task_names = metadata['task_names']
                            if task_name in task_names:
                                task_idx = task_names.index(task_name)
                                combined_targets[task_name].append(batch_targets[:, task_idx])
                            else:
                                combined_targets[task_name].append(batch_targets[:, 0])
                        else:
                            combined_targets[task_name].append(batch_targets)
            
            # Convert to tensors
            for task_name in combined_logits.keys():
                combined_logits[task_name] = torch.cat(combined_logits[task_name], dim=0)
                combined_targets[task_name] = torch.cat(combined_targets[task_name], dim=0)
            
            # Compute metrics
            if compute_task_metrics is not None:
                task_metrics = compute_task_metrics(
                    combined_logits, combined_targets, list(combined_logits.keys())
                )
                results['task_metrics'] = task_metrics
                
                # Compute overall metrics
                overall_accuracy = np.mean([metrics.get('accuracy', 0) for metrics in task_metrics.values()])
                results['overall_metrics'] = {'accuracy': overall_accuracy}
                
                # Format metrics table
                if format_metrics_table is not None:
                    results['metrics_table'] = format_metrics_table(task_metrics)
        
        return results
    
    def save_checkpoint(
        self, 
        path: str, 
        include_optimizer: bool = True,
        best_metric: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': self.epoch,
            'step': self.step,
            'best_metrics': self.best_metrics,
            'model_config': self.model.config if hasattr(self.model, 'config') else {},
            'training_history': dict(self.training_history)
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if best_metric is not None:
            checkpoint['best_metric'] = best_metric
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
        strict: bool = True
    ) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load training state
        self.epoch = checkpoint.get('epoch', 0)
        self.step = checkpoint.get('step', 0)
        self.best_metrics = checkpoint.get('best_metrics', {})
        
        if 'training_history' in checkpoint:
            self.training_history = defaultdict(list, checkpoint['training_history'])
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def update_curriculum(self, eval_metrics: Dict[str, Any]):
        """Update curriculum learning based on evaluation metrics."""
        if self.curriculum_strategy is None:
            return
        
        if 'task_metrics' in eval_metrics:
            # Update task difficulties based on performance
            for task_name, metrics in eval_metrics['task_metrics'].items():
                accuracy = metrics.get('accuracy', 0.0)
                self.task_difficulties[task_name] = 1.0 - accuracy  # Higher difficulty = lower accuracy
            
            # Simple curriculum: activate tasks with difficulty below threshold
            difficulty_threshold = 0.7  # Activate tasks with >30% accuracy
            self.active_tasks = {
                task for task, difficulty in self.task_difficulties.items()
                if difficulty < difficulty_threshold
            }
            
            # Ensure at least one task is active
            if not self.active_tasks:
                easiest_task = min(self.task_difficulties.keys(), 
                                 key=lambda t: self.task_difficulties[t])
                self.active_tasks = {easiest_task}
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            'training_state': {
                'epoch': self.epoch,
                'step': self.step,
                'active_tasks': list(self.active_tasks),
                'task_difficulties': self.task_difficulties
            },
            'model_info': {
                'multi_task': self.model.multi_task,
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'training_config': {
                'weighting_strategy': self.weighting_strategy,
                'curriculum_strategy': self.curriculum_strategy,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'max_grad_norm': self.max_grad_norm
            },
            'best_metrics': self.best_metrics
        }

    def adjust_lr_on_unfreeze(self, scale_factor: float = 1.5):
        """Adjust learning rate when unfreezing backbone layers."""
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = old_lr * scale_factor
                logger.info(f"📈 Adaptive LR Scaling: {old_lr:.1e} → {param_group['lr']:.1e} (×{scale_factor})")
        else:
            logger.warning("No optimizer found for LR adjustment")


def create_mvfouls_trainer(
    model_config: Optional[Dict] = None,
    optimizer_config: Optional[Dict] = None,
    trainer_config: Optional[Dict] = None,
    device: Optional[torch.device] = None
) -> Tuple[MVFoulsModel, MultiTaskTrainer]:
    """
    Factory function to create MVFouls model and trainer.
    
    Args:
        model_config: Configuration for MVFoulsModel
        optimizer_config: Configuration for optimizer
        trainer_config: Configuration for trainer
        device: Training device
        
    Returns:
        Tuple of (model, trainer)
    """
    # Default configurations
    model_config = model_config or {}
    optimizer_config = optimizer_config or {}
    trainer_config = trainer_config or {}
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = MVFoulsModel(**model_config)
    
    # Create optimizer
    optimizer_type = optimizer_config.pop('type', 'adamw')
    lr = optimizer_config.pop('lr', 1e-4)
    weight_decay = optimizer_config.pop('weight_decay', 1e-4)
    
    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_config)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_config)
    elif optimizer_type.lower() == 'sgd':
        momentum = optimizer_config.pop('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, 
                             momentum=momentum, **optimizer_config)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        **trainer_config
    )
    
    return model, trainer


def create_task_schedulers(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'cosine',
    **scheduler_kwargs
) -> Any:
    """
    Create learning rate scheduler for the main optimizer.
    
    Args:
        optimizer: Main optimizer
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau')
        **scheduler_kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=scheduler_kwargs.get('T_max', 1000),
            eta_min=scheduler_kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_kwargs.get('step_size', 100),
            gamma=scheduler_kwargs.get('gamma', 0.9)
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=scheduler_kwargs.get('factor', 0.5),
            patience=scheduler_kwargs.get('patience', 10)
        )
    elif scheduler_type == 'warmup_cosine':
        # Custom warmup + cosine scheduler
        from torch.optim.lr_scheduler import LambdaLR
        
        warmup_steps = scheduler_kwargs.get('warmup_steps', 1000)
        total_steps = scheduler_kwargs.get('total_steps', 10000)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


if __name__ == '__main__':
    # Example usage
    print("Multi-task training utilities for MVFoulsModel loaded successfully!")
    print("Available components:")
    print("- MultiTaskTrainer: Main training loop for complete MVFoulsModel")
    print("- create_mvfouls_trainer: Factory function for model + trainer")
    print("- create_task_schedulers: Learning rate schedulers")
    
    # Test basic functionality
    try:
        model, trainer = create_mvfouls_trainer(
            model_config={'multi_task': True, 'backbone_pretrained': False},
            trainer_config={'log_interval': 50}
        )
        print(f"✓ Successfully created model and trainer")
        print(f"  Model: Multi-task: {model.multi_task}")
        print(f"  Trainer: {trainer.__class__.__name__}")
        
    except Exception as e:
        print(f"✗ Error creating model/trainer: {e}") 