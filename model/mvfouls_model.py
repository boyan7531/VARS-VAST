"""
Complete MVFouls Model
=====================

A comprehensive end-to-end model for MVFouls video analysis that combines:
- Video Swin B backbone for feature extraction
- Custom MVFouls head for classification
- Support for both single-task and multi-task learning
- Training, evaluation, and inference capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import json
from collections import OrderedDict
import warnings

# Import components
from .backbone import VideoSwinBackbone, build_backbone as build_swin_backbone
from .factory_backbone import build_backbone
from .head import MVFoulsHead, build_head, build_multi_task_head

# Import utilities
try:
    from utils import (
        get_task_metadata, get_task_class_weights, concat_task_logits, split_concat_logits,
        compute_task_metrics, compute_confusion_matrices, compute_task_weights_from_metrics,
        format_metrics_table, compute_overall_metrics, get_mvfouls_class_weights
    )
except ImportError:
    # Fallback if utils not available
    get_task_metadata = None
    get_task_class_weights = None
    concat_task_logits = None
    split_concat_logits = None
    compute_task_metrics = None
    compute_confusion_matrices = None
    compute_task_weights_from_metrics = None
    format_metrics_table = None
    compute_overall_metrics = None
    get_mvfouls_class_weights = None


class MVFoulsModel(nn.Module):
    """
    Complete MVFouls model combining Video Swin B backbone with custom classification head.
    
    Features:
    - End-to-end video analysis from raw video to predictions
    - Support for both single-task and multi-task learning
    - Configurable backbone freezing strategies
    - Advanced loss computation with class weighting
    - Comprehensive evaluation metrics
    - Model checkpointing and resuming
    - ONNX export capability
    """
    
    def __init__(
        self,
        # Backbone configuration
        backbone_arch: str = 'swin',
        backbone_pretrained: bool = True,
        backbone_freeze_mode: str = 'none',
        backbone_checkpointing: bool = False,
        
        # Head configuration
        num_classes: int = 2,
        head_dropout: Optional[float] = 0.5,
        head_pooling: Optional[str] = 'avg',
        head_temporal_module: Optional[str] = None,
        head_loss_type: str = 'focal',
        head_label_smoothing: float = 0.0,
        head_enable_localizer: bool = False,
        head_gradient_checkpointing: bool = False,
        
        # Multi-task configuration
        multi_task: bool = False,
        task_names: Optional[List[str]] = None,
        num_classes_per_task: Optional[List[int]] = None,
        loss_types_per_task: Optional[List[str]] = None,
        
        # Training configuration
        class_weights: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        task_loss_weights: Optional[Dict[str, float]] = None,
        task_focal_gamma_map: Optional[Dict[str, float]] = None,
        
        # Model configuration
        model_name: str = "MVFoulsModel",
        model_version: str = "1.0",
        **kwargs
    ):
        """
        Initialize the complete MVFouls model.
        
        Args:
            backbone_arch: Backbone architecture ('swin', 'mvit')
            backbone_pretrained: Whether to use pretrained backbone weights
            backbone_freeze_mode: Backbone freeze strategy ('none', 'freeze_all', 'freeze_stages{k}', 'gradual')
            backbone_checkpointing: Enable gradient checkpointing in backbone
            
            num_classes: Number of classes for single-task mode
            head_dropout: Dropout probability in head
            head_pooling: Pooling strategy ('avg', 'max', 'attention', None)
            head_temporal_module: Temporal processing ('tconv', 'lstm', None)  
            head_loss_type: Loss type ('ce', 'focal', 'bce')
            head_label_smoothing: Label smoothing factor
            head_enable_localizer: Enable frame-level predictions
            head_gradient_checkpointing: Enable gradient checkpointing in head
            
            multi_task: Enable multi-task learning mode
            task_names: List of task names (auto-detected if None)
            num_classes_per_task: Number of classes per task (auto-detected if None)
            loss_types_per_task: Loss type per task (defaults to head_loss_type)
            
            class_weights: Class weights for loss computation
            task_loss_weights: Weights for combining task losses
            task_focal_gamma_map: Focal gamma map for multi-task loss computation
            
            model_name: Name identifier for the model
            model_version: Version identifier for the model
        """
        super().__init__()
        
        # Store configuration
        self.config = {
            'backbone_arch': backbone_arch,
            'backbone_pretrained': backbone_pretrained,
            'backbone_freeze_mode': backbone_freeze_mode,
            'backbone_checkpointing': backbone_checkpointing,
            'num_classes': num_classes,
            'head_dropout': head_dropout,
            'head_pooling': head_pooling,
            'head_temporal_module': head_temporal_module,
            'head_loss_type': head_loss_type,
            'head_label_smoothing': head_label_smoothing,
            'head_enable_localizer': head_enable_localizer,
            'head_gradient_checkpointing': head_gradient_checkpointing,
            'multi_task': multi_task,
            'task_names': task_names,
            'num_classes_per_task': num_classes_per_task,
            'loss_types_per_task': loss_types_per_task,
            'class_weights': class_weights,
            'task_loss_weights': task_loss_weights,
            'task_focal_gamma_map': task_focal_gamma_map,
            'model_name': model_name,
            'model_version': model_version,
        }
        
        self.multi_task = multi_task
        self.task_loss_weights = task_loss_weights or {}
        self.task_focal_gamma_map = task_focal_gamma_map
        
        # Build backbone
        print(f"Building {backbone_arch.upper()} backbone...")
        self.backbone = build_backbone(
            arch=backbone_arch,
            pretrained=backbone_pretrained,
            return_pooled=head_pooling is None,  # If head has no pooling, backbone should return pooled features
            freeze_mode=backbone_freeze_mode,
            checkpointing=backbone_checkpointing
        )
        
        # Get backbone output dimension
        backbone_dim = self.backbone.out_dim
        
        # Build head based on mode
        print("Building head...")
        if multi_task and get_task_metadata is not None:
            # Multi-task mode with automatic task detection
            self.head = build_multi_task_head(
                in_dim=backbone_dim,
                dropout=head_dropout,
                pooling=head_pooling,
                temporal_module=head_temporal_module,
                loss_type=head_loss_type,
                label_smoothing=head_label_smoothing,
                enable_localizer=head_enable_localizer,
                gradient_checkpointing=head_gradient_checkpointing,
                task_names=task_names,
                num_classes_per_task=num_classes_per_task,
                loss_types_per_task=loss_types_per_task,
                task_weights=class_weights if isinstance(class_weights, dict) else None,
                task_loss_weights=task_loss_weights,
                clip_pooling_type=kwargs.get('clip_pooling_type', 'mean'),
                clip_pooling_temperature=kwargs.get('clip_pooling_temperature', 1.0)
            )
        else:
            # Single-task mode
            self.head = build_head(
                in_dim=backbone_dim,
                num_classes=num_classes,
                dropout=head_dropout,
                pooling=head_pooling,
                temporal_module=head_temporal_module,
                loss_type=head_loss_type,
                label_smoothing=head_label_smoothing,
                enable_localizer=head_enable_localizer,
                gradient_checkpointing=head_gradient_checkpointing,
                class_weights=class_weights if isinstance(class_weights, torch.Tensor) else None,
                task_loss_weights=task_loss_weights,
                clip_pooling_type=kwargs.get('clip_pooling_type', 'mean'),
                clip_pooling_temperature=kwargs.get('clip_pooling_temperature', 1.0)
            )
        
        # Initialize training state
        self.training_step = 0
        self.validation_step = 0
        
        # Print model summary
        self.print_model_summary()
        
        # Multi-task configuration
        if multi_task and get_task_metadata is not None:
            # Use task metadata from utils
            if task_names is None or num_classes_per_task is None:
                metadata = get_task_metadata()
                self.task_names = metadata['task_names']
                self.num_classes_per_task = metadata['num_classes']
                
                # Validation: Ensure exactly 3 tasks
                expected_tasks = ['action_class', 'severity', 'offence']
                if set(self.task_names) != set(expected_tasks):
                    raise ValueError(f"Multi-task model expects exactly these 3 tasks: {expected_tasks}, "
                                   f"but found: {self.task_names}")
                
                print(f"✅ Multi-task model configured for exactly 3 MVFouls tasks:")
                for task_name, num_cls in zip(self.task_names, self.num_classes_per_task):
                    print(f"   📋 {task_name}: {num_cls} classes")
            else:
                self.task_names = task_names
                self.num_classes_per_task = num_classes_per_task
    
    def forward(
        self, 
        x: torch.Tensor, 
        clip_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_dict: Optional[bool] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], 
               Tuple[Dict[str, torch.Tensor], torch.Tensor], 
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the complete model.
        
        Args:
            x: Input video tensor, supports multiple formats:
               - (B, 3, T, H, W): Standard single-clip format
               - (B, T, H, W, C): Standard single-clip format (alternative)
               - (B, N_clips, 3, T, H, W): Bag-of-clips format
               - (B, N_clips, T, H, W, C): Bag-of-clips format (alternative)
            clip_mask: Optional mask for bag-of-clips (B, N_clips) indicating valid clips
            return_features: Whether to return backbone features
            return_dict: Whether to return dict of logits (None = auto-detect from multi_task)
            
        Returns:
            Based on configuration:
            - Single-task: logits (B, num_classes) or (logits, features) if return_features=True
            - Multi-task: logits_dict {task: (B, classes)} or (logits_dict, features) if return_features=True
            - With extras: (logits/logits_dict, features, extras_dict)
        """
        # Detect bag-of-clips format
        if x.dim() == 6:  # (B, N_clips, 3, T, H, W) or (B, N_clips, T, H, W, C)
            batch_size, n_clips = x.shape[:2]
            
            if x.shape[2] == 3:
                # Format: (B, N_clips, 3, T, H, W) - standard bag-of-clips
                pass
            elif x.shape[-1] == 3:
                # Format: (B, N_clips, T, H, W, C) - convert to (B, N_clips, 3, T, H, W)
                x = x.permute(0, 1, 5, 2, 3, 4)
            else:
                raise ValueError(f"Unsupported bag-of-clips format: {x.shape}")
            
            # Process each clip through backbone
            x = x.view(-1, *x.shape[2:])  # (B*N_clips, 3, T, H, W)
            features = self.backbone(x)  # (B*N_clips, feature_dim) or (B*N_clips, T', H', W', feature_dim)
            
            # Reshape back to bag format and pass to head with clip pooling
            if features.dim() == 2:  # (B*N_clips, feature_dim)
                features = features.view(batch_size, n_clips, -1)  # (B, N_clips, feature_dim)
            else:  # (B*N_clips, T', H', W', feature_dim)
                features = features.view(batch_size, n_clips, *features.shape[1:])  # (B, N_clips, T', H', W', feature_dim)
            
        else:
            # Handle standard single-clip input format conversion if needed
            if x.dim() == 5 and x.shape[1] == 3:
                # Input is (B, C, T, H, W) - standard format
                pass
            elif x.dim() == 5 and x.shape[-1] == 3:
                # Input is (B, T, H, W, C) - convert to (B, C, T, H, W)
                x = x.permute(0, 4, 1, 2, 3)
            else:
                raise ValueError(f"Unsupported input shape: {x.shape}. Expected (B,C,T,H,W), (B,T,H,W,C), or bag-of-clips format")
            
            # Extract features through backbone
            features = self.backbone(x)  # (B, feature_dim) or (B, T', H', W', feature_dim)
        
        # Forward through head
        if return_dict is None:
            return_dict = self.multi_task
            
        if return_dict:
            # Multi-task mode
            logits_dict, extras = self.head.forward(features, clip_mask=clip_mask, return_dict=True)
            
            if return_features:
                return logits_dict, features, extras
            else:
                return logits_dict, extras
        else:
            # Single-task mode
            logits, extras = self.head.forward(features, clip_mask=clip_mask, return_dict=False)
            
            if return_features:
                return logits, features, extras
            else:
                return logits, extras
    
    def compute_loss(
        self,
        logits: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
        reduction: str = 'mean',
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss based on model configuration.
        
        Args:
            logits: Model predictions (tensor or dict)
            targets: Ground truth labels (tensor or dict)
            reduction: Loss reduction ('mean', 'sum', 'none')
            return_dict: Whether to return detailed loss breakdown
            
        Returns:
            Loss tensor or dict with loss breakdown
        """
        if self.multi_task and isinstance(logits, dict) and isinstance(targets, dict):
            # Multi-task loss computation
            loss_dict = self.head.compute_multi_task_loss(
                logits, targets, self.task_loss_weights,
                focal_gamma=self.task_focal_gamma_map if self.task_focal_gamma_map is not None else 2.0
            )
            
            if return_dict:
                return loss_dict
            else:
                return loss_dict['total_loss']
        else:
            # Single-task loss computation
            if isinstance(logits, dict):
                # If dict provided but single-task mode, concatenate
                if concat_task_logits is not None:
                    logits = concat_task_logits(logits)
                else:
                    # Fallback concatenation
                    logits = torch.cat(list(logits.values()), dim=1)
            
            if isinstance(targets, dict):
                # Same for targets
                targets = torch.cat(list(targets.values()), dim=1)
            
            loss = self.head.compute_loss(logits, targets)
            
            if return_dict:
                return {'total_loss': loss, 'main_loss': loss}
            else:
                return loss
    
    def predict(
        self, 
        x: torch.Tensor, 
        return_probs: bool = False,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Make predictions on input video.
        
        Args:
            x: Input video tensor
            return_probs: Whether to return probabilities
            temperature: Temperature for probability calibration
            
        Returns:
            Dict containing predictions, probabilities, and metadata
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            if self.multi_task:
                logits_dict, extras = self.forward(x, return_dict=True)
                
                # Convert to predictions
                predictions = {}
                probabilities = {}
                
                for task_name, task_logits in logits_dict.items():
                    # Apply temperature scaling
                    if temperature != 1.0:
                        task_logits = task_logits / temperature
                    
                    # Get predictions
                    task_probs = F.softmax(task_logits, dim=1)
                    task_preds = torch.argmax(task_logits, dim=1)
                    
                    predictions[task_name] = task_preds
                    if return_probs:
                        probabilities[task_name] = task_probs
                
                result = {
                    'predictions': predictions,
                    'logits': logits_dict,
                    'extras': extras
                }
                
                if return_probs:
                    result['probabilities'] = probabilities
                    
            else:
                logits, extras = self.forward(x, return_dict=False)
                
                # Apply temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Get predictions
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                result = {
                    'predictions': preds,
                    'logits': logits,
                    'extras': extras
                }
                
                if return_probs:
                    result['probabilities'] = probs
        
        return result
    
    def evaluate(
        self,
        dataloader,
        device: Optional[torch.device] = None,
        compute_confusion_matrix: bool = True,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: PyTorch DataLoader
            device: Device to run evaluation on
            compute_confusion_matrix: Whether to compute confusion matrices
            return_predictions: Whether to return all predictions
            
        Returns:
            Dict containing evaluation results
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_logits = []
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, (videos, targets) in enumerate(dataloader):
                videos = videos.to(device)
                
                # Handle target format
                if self.multi_task:
                    if not isinstance(targets, dict):
                        # Convert tensor to dict format
                        targets = targets.to(device)
                        if get_task_metadata is not None:
                            metadata = get_task_metadata()
                            targets_dict = {}
                            for i, task_name in enumerate(metadata['task_names']):
                                if i < targets.size(1):
                                    targets_dict[task_name] = targets[:, i]
                                else:
                                    # Fallback for missing task
                                    targets_dict[task_name] = torch.zeros_like(targets[:, 0])
                        else:
                            # Fallback for single task
                            targets_dict = {'default': targets.flatten() if targets.dim() > 1 else targets}
                    else:
                        targets_dict = {k: v.to(device) for k, v in targets.items()}
                else:
                    if isinstance(targets, dict):
                        # Convert dict to single tensor for single-task mode
                        targets = torch.cat(list(targets.values()), dim=1)
                    targets = targets.to(device)
                    targets_dict = targets
                
                # Forward pass
                if self.multi_task:
                    logits_dict, extras = self.forward(videos, return_dict=True)
                    loss = self.compute_loss(logits_dict, targets_dict)
                    
                    all_logits.append(logits_dict)
                    all_targets.append(targets_dict)
                    
                    if return_predictions:
                        preds_dict = {}
                        for task_name, task_logits in logits_dict.items():
                            preds_dict[task_name] = torch.argmax(task_logits, dim=1)
                        all_predictions.append(preds_dict)
                        
                else:
                    logits, extras = self.forward(videos, return_dict=False)
                    loss = self.compute_loss(logits, targets_dict)
                    
                    all_logits.append(logits)
                    all_targets.append(targets_dict)
                    
                    if return_predictions:
                        preds = torch.argmax(logits, dim=1)
                        all_predictions.append(preds)
                
                total_loss += loss.item()
                num_batches += 1
        
        # Aggregate results
        avg_loss = total_loss / num_batches
        
        results = {
            'avg_loss': avg_loss,
            'num_batches': num_batches,
            'num_samples': len(dataloader.dataset)
        }
        
        # Compute detailed metrics if utilities are available
        if self.multi_task and compute_task_metrics is not None:
            # Combine all logits and targets
            combined_logits = {}
            combined_targets = {}
            
            if get_task_metadata is not None:
                metadata = get_task_metadata()
                task_names = metadata['task_names']
                
                for task_name in task_names:
                    task_logits_list = []
                    task_targets_list = []
                    
                    for batch_logits, batch_targets in zip(all_logits, all_targets):
                        if task_name in batch_logits and task_name in batch_targets:
                            task_logits_list.append(batch_logits[task_name])
                            task_targets_list.append(batch_targets[task_name])
                    
                    if task_logits_list and task_targets_list:
                        combined_logits[task_name] = torch.cat(task_logits_list, dim=0)
                        combined_targets[task_name] = torch.cat(task_targets_list, dim=0)
                
                # Compute metrics
                metrics = compute_task_metrics(combined_logits, combined_targets, task_names)
                results['task_metrics'] = metrics
                
                # Compute overall metrics
                if compute_overall_metrics is not None:
                    overall_metrics = compute_overall_metrics(metrics)
                    results['overall_metrics'] = overall_metrics
                
                # Format metrics table
                if format_metrics_table is not None:
                    metrics_table = format_metrics_table(metrics)
                    results['metrics_table'] = metrics_table
                
                # Compute confusion matrices
                if compute_confusion_matrix and compute_confusion_matrices is not None:
                    confusion_matrices = compute_confusion_matrices(combined_logits, combined_targets, task_names)
                    results['confusion_matrices'] = confusion_matrices
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['logits'] = all_logits
            results['targets'] = all_targets
        
        return results
    
    def fit_epoch(
        self,
        train_dataloader,
        optimizer,
        device: Optional[torch.device] = None,
        scheduler=None,
        accumulation_steps: int = 1,
        clip_grad_norm: Optional[float] = None,
        log_interval: int = 100
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_dataloader: Training data loader
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler
            accumulation_steps: Gradient accumulation steps
            clip_grad_norm: Gradient clipping value
            log_interval: Logging interval
            
        Returns:
            Dict with training metrics
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, (videos, targets) in enumerate(train_dataloader):
            videos = videos.to(device)
            
            # Handle target format (same as evaluate)
            if self.multi_task:
                if not isinstance(targets, dict):
                    # Convert tensor to dict format
                    targets = targets.to(device)
                    if get_task_metadata is not None:
                        metadata = get_task_metadata()
                        targets_dict = {}
                        for i, task_name in enumerate(metadata['task_names']):
                            if i < targets.size(1):
                                targets_dict[task_name] = targets[:, i]
                            else:
                                # Fallback for missing task
                                targets_dict[task_name] = torch.zeros_like(targets[:, 0])
                    else:
                        # Fallback for single task
                        targets_dict = {'default': targets.flatten() if targets.dim() > 1 else targets}
                else:
                    targets_dict = {k: v.to(device) for k, v in targets.items()}
            else:
                targets_dict = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            if self.multi_task:
                logits_dict, extras = self.forward(videos, return_dict=True)
                loss = self.compute_loss(logits_dict, targets_dict)
            else:
                logits, extras = self.forward(videos, return_dict=False)
                loss = self.compute_loss(logits, targets_dict)
            
            # Scale loss for accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Scheduler step
                if scheduler is not None:
                    scheduler.step()
            
            total_loss += loss.item() * accumulation_steps
            num_batches += 1
            self.training_step += 1
            
            # Logging
            if batch_idx % log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = optimizer.param_groups[0]['lr']
                logging.info(f"Batch {batch_idx}/{len(train_dataloader)}, "
                          f"Loss: {avg_loss:.4f}, LR: {lr:.2e}")
        
        # Handle remaining gradients
        if num_batches % accumulation_steps != 0:
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches
        return {'avg_loss': avg_loss, 'num_batches': num_batches}
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer=None,
        scheduler=None,
        epoch: int = 0,
        best_metric: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch number
            best_metric: Best validation metric achieved
            metadata: Additional metadata to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'training_step': self.training_step,
            'validation_step': self.validation_step,
            'best_metric': best_metric,
            'model_name': self.config['model_name'],
            'model_version': self.config['model_version'],
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(
        self,
        path: Union[str, Path],
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            device: Device to load checkpoint on
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Dict containing checkpoint metadata
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Load model state
        missing_keys, unexpected_keys = self.load_state_dict(
            checkpoint['model_state_dict'], strict=strict
        )
        
        if missing_keys:
            logging.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        
        # Restore training state
        self.training_step = checkpoint.get('training_step', 0)
        self.validation_step = checkpoint.get('validation_step', 0)
        
        logging.info(f"Checkpoint loaded from {path}")
        logging.info(f"Model: {checkpoint.get('model_name', 'Unknown')} "
                    f"v{checkpoint.get('model_version', 'Unknown')}")
        logging.info(f"Epoch: {checkpoint.get('epoch', 0)}")
        
        return checkpoint
    
    def export_onnx(
        self,
        path: Union[str, Path],
        input_shape: Tuple[int, ...] = (1, 3, 32, 224, 224),
        export_mode: str = 'concat'
    ):
        """
        Export model to ONNX format.
        
        Args:
            path: Output ONNX file path
            input_shape: Input tensor shape (B, C, T, H, W)
            export_mode: 'concat' for single output, 'separate' for multi-task outputs
        """
        self.eval()
        
        # Create wrapper for ONNX export
        if self.multi_task and export_mode == 'separate':
            class ONNXMultiTaskWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    logits_dict, _ = self.model(x, return_dict=True)
                    # Return as tuple for ONNX
                    if get_task_metadata is not None:
                        metadata = get_task_metadata()
                        return tuple(logits_dict[task] for task in metadata['task_names'])
                    else:
                        return tuple(logits_dict.values())
            
            onnx_model = ONNXMultiTaskWrapper(self)
            
            if get_task_metadata is not None:
                metadata = get_task_metadata()
                output_names = [f'logits_{task}' for task in metadata['task_names']]
            else:
                output_names = [f'logits_{i}' for i in range(len(self.head.task_names))]
                
        else:
            class ONNXWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    if self.model.multi_task:
                        logits_dict, _ = self.model(x, return_dict=True)
                        if concat_task_logits is not None:
                            return concat_task_logits(logits_dict)
                        else:
                            return torch.cat(list(logits_dict.values()), dim=1)
                    else:
                        logits, _ = self.model(x, return_dict=False)
                        return logits
            
            onnx_model = ONNXWrapper(self)
            output_names = ['logits']
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Dynamic axes for batch size
        dynamic_axes = {'input': {0: 'batch_size'}}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch_size'}
        
        # Export to ONNX
        torch.onnx.export(
            onnx_model,
            dummy_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        
        logging.info(f"Model exported to ONNX: {path}")
    
    def set_backbone_freeze_mode(self, mode: str):
        """Dynamically change backbone freeze mode."""
        self.backbone.set_freeze(mode)
        self.config['backbone_freeze_mode'] = mode
    
    def unfreeze_backbone_gradually(self):
        """Unfreeze next group in gradual mode."""
        if self.backbone.freeze_mode == 'gradual':
            self.backbone.next_unfreeze()
    
    def print_model_summary(self):
        """Print comprehensive model summary."""
        print("\n" + "="*70)
        print(f"🚀 {self.config['model_name']} v{self.config['model_version']}")
        print("="*70)
        
        # Model configuration
        print("📋 MODEL CONFIGURATION:")
        print(f"  Mode: {'Multi-task' if self.multi_task else 'Single-task'}")
        
        if self.multi_task:
            if hasattr(self.head, 'task_names'):
                print(f"  Tasks: {len(self.head.task_names)} ({', '.join(self.head.task_names[:3])}...)")
                print(f"  Classes per task: {self.head.num_classes_per_task[:3]}...")
            else:
                print(f"  Tasks: Multi-task mode enabled")
        else:
            print(f"  Classes: {self.config['num_classes']}")
        
        arch_name = self.config.get('backbone_arch', 'swin').upper()
        print(f"  Backbone: {arch_name} ({'pretrained' if self.config['backbone_pretrained'] else 'random init'})")
        print(f"  Freeze mode: {self.config['backbone_freeze_mode']}")
        print(f"  Head pooling: {self.config['head_pooling']}")
        print(f"  Temporal module: {self.config['head_temporal_module']}")
        print(f"  Loss type: {self.config['head_loss_type']}")
        
        # Parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        
        print(f"\n📊 PARAMETER SUMMARY:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"  Backbone: {backbone_params:,}")
        print(f"  Head: {head_params:,}")
        
        # Memory estimation (rough)
        model_size_mb = total_params * 4 / (1024**2)  # Assume float32
        print(f"  Estimated size: {model_size_mb:.1f} MB")
        
        print("="*70 + "\n")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': self.config['model_name'],
            'model_version': self.config['model_version'],
            'multi_task': self.multi_task,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'config': self.config.copy(),
            'training_step': self.training_step,
            'validation_step': self.validation_step,
        }
        
        if self.multi_task and hasattr(self.head, 'task_names'):
            info['task_names'] = self.head.task_names
            info['num_classes_per_task'] = self.head.num_classes_per_task
        
        return info


# Factory functions for easy model creation

def build_mvfouls_model(**kwargs) -> MVFoulsModel:
    """
    Factory function to build MVFouls model with default settings.
    
    Args:
        **kwargs: Arguments passed to MVFoulsModel constructor
        
    Returns:
        Configured MVFoulsModel instance
    """
    return MVFoulsModel(**kwargs)


def build_single_task_model(
    num_classes: int = 2,
    backbone_arch: str = 'swin',
    backbone_pretrained: bool = True,
    backbone_freeze_mode: str = 'none',
    **kwargs
) -> MVFoulsModel:
    """
    Build single-task MVFouls model with common settings.
    
    Args:
        num_classes: Number of classes
        backbone_arch: Backbone architecture ('swin', 'mvit')
        backbone_pretrained: Use pretrained backbone
        backbone_freeze_mode: Backbone freeze mode ('none', 'freeze_all', etc.)
        **kwargs: Additional arguments
        
    Returns:
        Single-task MVFoulsModel
    """
    return MVFoulsModel(
        backbone_arch=backbone_arch,
        backbone_pretrained=backbone_pretrained,
        backbone_freeze_mode=backbone_freeze_mode,
        num_classes=num_classes,
        multi_task=False,
        **kwargs
    )


def build_multi_task_model(
    backbone_arch: str = 'swin',
    backbone_pretrained: bool = True,
    backbone_freeze_mode: str = 'none',
    **kwargs
) -> MVFoulsModel:
    """
    Build multi-task MVFouls model with automatic task detection.
    
    Args:
        backbone_arch: Backbone architecture ('swin', 'mvit')
        backbone_pretrained: Use pretrained backbone
        backbone_freeze_mode: Backbone freeze mode ('none', 'freeze_all', etc.)
        **kwargs: Additional arguments
        
    Returns:
        Multi-task MVFoulsModel
    """
    if get_task_metadata is None:
        raise ImportError("Multi-task model requires task metadata from utils.py")
    
    return MVFoulsModel(
        backbone_arch=backbone_arch,
        backbone_pretrained=backbone_pretrained,
        backbone_freeze_mode=backbone_freeze_mode,
        multi_task=True,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing MVFouls Complete Model")
    print("=" * 50)
    
    # Test single-task model
    print("\n1. Single-task model:")
    try:
        model = build_single_task_model(num_classes=2, pretrained=False)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 32, 224, 224)
        logits, extras = model(dummy_input)
        
        print(f"   ✓ Input shape: {dummy_input.shape}")
        print(f"   ✓ Output shape: {logits.shape}")
        print(f"   ✓ Extras keys: {list(extras.keys())}")
        
        # Test loss computation
        dummy_targets = torch.randint(0, 2, (2,))
        loss = model.compute_loss(logits, dummy_targets)
        print(f"   ✓ Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"   ✗ Single-task model failed: {e}")
    
    # Test multi-task model if possible
    print("\n2. Multi-task model:")
    try:
        if get_task_metadata is not None:
            model = build_multi_task_model(pretrained=False)
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 32, 224, 224)
            logits_dict, extras = model(dummy_input)
            
            print(f"   ✓ Input shape: {dummy_input.shape}")
            print(f"   ✓ Tasks: {list(logits_dict.keys())[:3]}...")
            print(f"   ✓ Sample task output: {list(logits_dict.values())[0].shape}")
            
            # Test prediction
            pred_result = model.predict(dummy_input)
            print(f"   ✓ Prediction keys: {list(pred_result.keys())}")
            
        else:
            print("   ⚠ Multi-task model requires task metadata (utils.py)")
            
    except Exception as e:
        print(f"   ✗ Multi-task model failed: {e}")
    
    print("\n✅ Model testing complete!")
