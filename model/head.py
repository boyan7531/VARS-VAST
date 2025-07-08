import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List, Union
import pandas as pd

# Import task metadata utilities
try:
    from utils import (
        get_task_metadata, get_task_class_weights, concat_task_logits, split_concat_logits,
        compute_task_metrics, compute_confusion_matrices, compute_task_weights_from_metrics,
        format_metrics_table, compute_overall_metrics
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


class ClipPooling(nn.Module):
    """Pooling module for aggregating multiple clips into a single representation."""
    
    def __init__(self, pooling_type: str = 'mean', in_dim: int = 1024, temperature: float = 1.0):
        """
        Args:
            pooling_type: Type of pooling ('mean', 'max', 'attention')
            in_dim: Input feature dimension
            temperature: Temperature for attention softmax
        """
        super().__init__()
        self.pooling_type = pooling_type
        self.temperature = temperature
        
        if pooling_type == 'attention':
            # Learnable attention pooling
            self.attention = nn.Sequential(
                nn.Linear(in_dim, in_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim // 4, 1)
            )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N_clips, D)
            mask: Optional mask of shape (B, N_clips) indicating valid clips
            
        Returns:
            Pooled tensor of shape (B, D)
        """
        if self.pooling_type == 'mean':
            if mask is not None:
                # Masked mean pooling
                mask_expanded = mask.unsqueeze(-1).float()  # (B, N_clips, 1)
                masked_x = x * mask_expanded
                return masked_x.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
            else:
                return x.mean(dim=1)
        
        elif self.pooling_type == 'max':
            if mask is not None:
                # Masked max pooling
                mask_expanded = mask.unsqueeze(-1)  # (B, N_clips, 1)
                masked_x = x.masked_fill(~mask_expanded, float('-inf'))
                return masked_x.max(dim=1)[0]
            else:
                return x.max(dim=1)[0]
        
        elif self.pooling_type == 'attention':
            # Learnable attention pooling
            attn_weights = self.attention(x)  # (B, N_clips, 1)
            
            if mask is not None:
                # Apply mask to attention weights
                mask_expanded = mask.unsqueeze(-1)  # (B, N_clips, 1)
                attn_weights = attn_weights.masked_fill(~mask_expanded, float('-inf'))
            
            attn_weights = F.softmax(attn_weights / self.temperature, dim=1)  # (B, N_clips, 1)
            return (x * attn_weights).sum(dim=1)  # (B, D)
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class AttentionPool(nn.Module):
    """Attention-based pooling over spatial or temporal dimensions."""
    
    def __init__(self, in_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 4, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass handling different input dimensions."""
        if x.dim() == 5:  # (B, T, H, W, C) - global spatio-temporal attention
            B, T, H, W, C = x.shape
            x_flat = x.view(B, T * H * W, C)  # (B, T*H*W, C)
            attn_weights = self.attention(x_flat)  # (B, T*H*W, 1)
            attn_weights = F.softmax(attn_weights, dim=1)  # softmax over T*H*W
            return (x_flat * attn_weights).sum(dim=1)  # (B, C)
        elif x.dim() == 3:  # (B, T, C) - temporal attention
            attn_weights = self.attention(x)  # (B, T, 1)
            attn_weights = F.softmax(attn_weights, dim=1)  # softmax over T
            return (x * attn_weights).sum(dim=1)  # (B, C)
        else:
            raise ValueError(f"AttentionPool received unexpected input dimensions: {x.shape}. Expected 3D (B,T,C) or 5D (B,T,H,W,C).")


class TemporalConv(nn.Module):
    """Depthwise 1D temporal convolution stack."""
    
    def __init__(self, in_dim: int, dilations: List[int] = [1, 2, 3], kernel_size: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        for dilation in dilations:
            self.convs.append(nn.Conv1d(
                in_dim, in_dim, kernel_size=kernel_size,
                padding=dilation * (kernel_size - 1) // 2,
                dilation=dilation, groups=in_dim
            ))
        self.norm = nn.LayerNorm(in_dim)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        
        for conv in self.convs:
            residual = x
            x = conv(x)
            x = x + residual  # residual connection
            x = self.activation(x)
        
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        return x.mean(dim=1)  # (B, C)


class TemporalLSTM(nn.Module):
    """Bidirectional LSTM with attention pooling."""
    
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim // 2
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.attention_pool = AttentionPool(hidden_dim * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim*2)
        return self.attention_pool(lstm_out)  # (B, hidden_dim*2)


class MVFoulsHead(nn.Module):
    """
    Comprehensive head for MVFouls predictions.
    Converts backbone features into predictions with configurable pooling, temporal processing, and losses.
    Supports both single-clip and bag-of-clips (action-level) training.
    """
    
    def __init__(
        self,
        in_dim: int = 1024,
        num_classes: int = 2,
        dropout: Optional[float] = 0.5,
        pooling: Optional[str] = 'avg',  # 'avg', 'max', 'attention', None
        temporal_module: Optional[str] = None,  # None, 'tconv', 'lstm'
        loss_type: str = 'focal',  # 'ce', 'focal', 'bce'
        label_smoothing: float = 0.0,
        freeze_mode: str = 'trainable',  # 'trainable', 'freeze'
        init_std: float = 0.02,
        enable_localizer: bool = False,
        gradient_checkpointing: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        aux_heads: Optional[List[nn.Module]] = None,
        # Multi-task parameters
        multi_task: bool = False,
        task_names: Optional[List[str]] = None,
        num_classes_per_task: Optional[List[int]] = None,
        loss_types_per_task: Optional[List[str]] = None,
        task_weights: Optional[Dict[str, torch.Tensor]] = None,
        task_loss_weights: Optional[Dict[str, float]] = None,
        # Enhanced head parameters
        hidden_dim: int = 2048,  # Intermediate layer size
        num_shared_layers: int = 2,  # Number of shared layers before task-specific heads
        task_specific_layers: int = 1,  # Number of task-specific layers
        use_batch_norm: bool = True,  # Use batch normalization
        activation: str = 'relu',  # Activation function
        # Bag-of-clips parameters
        clip_pooling_type: str = 'mean',  # 'mean', 'max', 'attention'
        clip_pooling_temperature: float = 1.0,  # Temperature for attention pooling
        # Logit-adjustment (Balanced Softmax) parameters
        use_logit_adjustment: bool = False,
        logit_adjustment_temperature: float = 1.0,
        class_prior_logits: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None  # Pre-computed prior shifts
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_shared_layers = num_shared_layers
        self.task_specific_layers = task_specific_layers
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.dropout_p = dropout
        self.pooling_type = pooling
        self.temporal_module_type = temporal_module
        self.label_smoothing = label_smoothing
        self.freeze_mode = freeze_mode
        self.gradient_checkpointing = gradient_checkpointing
        self.enable_localizer = enable_localizer
        
        # Bag-of-clips parameters
        self.clip_pooling_type = clip_pooling_type
        self.clip_pooling_temperature = clip_pooling_temperature
        
        # ------------------------------------------------------------------
        # Balanced Softmax / Logit-Adjustment configuration
        # ------------------------------------------------------------------
        self.use_logit_adjustment = use_logit_adjustment
        self.logit_adjustment_temperature = logit_adjustment_temperature
        if self.use_logit_adjustment:
            if self.multi_task:
                # Expect a dict mapping task_name -> tensor(C,)
                if not isinstance(class_prior_logits, dict):
                    raise ValueError("For multi-task heads, class_prior_logits must be a dict {task: tensor} when use_logit_adjustment=True")
                self.class_prior_logits = {}
                for task_name, prior in class_prior_logits.items():
                    prior = prior.float()
                    # Register each as separate buffer to keep device synchronization & state dict support
                    buf_name = f"class_prior_logits_{task_name}"
                    self.register_buffer(buf_name, prior, persistent=True)
                    self.class_prior_logits[task_name] = getattr(self, buf_name)
            else:
                # Single-task – expect Tensor
                if not isinstance(class_prior_logits, torch.Tensor):
                    raise ValueError("class_prior_logits tensor must be provided for single-task when use_logit_adjustment=True")
                self.register_buffer('class_prior_logits', class_prior_logits.float(), persistent=True)
        
        # Multi-task configuration
        self.multi_task = multi_task
        if multi_task and get_task_metadata is not None:
            # Use task metadata from utils
            if task_names is None or num_classes_per_task is None:
                metadata = get_task_metadata()
                self.task_names = metadata['task_names']
                self.num_classes_per_task = metadata['num_classes']
            else:
                self.task_names = task_names
                self.num_classes_per_task = num_classes_per_task
            
            # Set up per-task loss types
            if loss_types_per_task is None:
                self.loss_types_per_task = [loss_type] * len(self.task_names)
            else:
                self.loss_types_per_task = loss_types_per_task
                
            # Set up per-task weights (class weights)
            self.task_weights = task_weights or {}
            
            # Store task loss weights (scalar weights for loss combination)
            self.task_weights_scalar = task_loss_weights or {}
            
            # For backward compatibility
            self.num_classes = sum(self.num_classes_per_task)
            self.loss_type = loss_type  # Default loss type
        else:
            # Single-task mode (backward compatibility)
            self.task_names = ['default']
            self.num_classes_per_task = [num_classes]
            self.loss_types_per_task = [loss_type]
            self.task_weights = {}
            self.task_weights_scalar = task_loss_weights or {}
            self.num_classes = num_classes
            self.loss_type = loss_type
        
        # Build clip pooling module
        self.clip_pooling = ClipPooling(
            pooling_type=clip_pooling_type,
            in_dim=in_dim,
            temperature=clip_pooling_temperature
        )
        
        # Build pooling module
        self._pool = self._build_pooling_module()
        
        # Build temporal module
        self._temporal = self._build_temporal_module()
        
        # Determine classifier input dimension
        classifier_dim = self._get_classifier_dim()
        
        # Build enhanced architecture
        if self.multi_task and len(self.task_names) > 1:
            self._build_enhanced_multi_task_head(classifier_dim)
        else:
            self._build_simple_single_task_head(classifier_dim)
        
        # Build optional localizer
        self.localizer = None
        if enable_localizer:
            if self.multi_task:
                # For multi-task, use total classes for localizer
                self.localizer = nn.Conv3d(in_dim, self.num_classes, kernel_size=1)
            else:
                self.localizer = nn.Conv3d(in_dim, self.num_classes, kernel_size=1)
        
        # Learnable logit scale
        self.logit_scale = nn.Parameter(torch.ones(1))
        
        # Loss weights
        self.register_buffer('class_weights', class_weights)
        
        # Auxiliary heads
        self.aux_heads = aux_heads or []
        
        # Metrics buffers
        if self.multi_task:
            # Per-task metrics
            self.running_acc = nn.ModuleDict()
            self.confusion_matrices = nn.ModuleDict()
            for task_name, num_cls in zip(self.task_names, self.num_classes_per_task):
                self.register_buffer(f'running_acc_{task_name}', torch.zeros(1))
                self.register_buffer(f'confusion_matrix_{task_name}', torch.zeros(num_cls, num_cls))
        else:
            # Single task metrics (backward compatibility)
            self.register_buffer('running_acc', torch.zeros(1))
            self.register_buffer('confusion_matrix', torch.zeros(self.num_classes, self.num_classes))
        
        # Initialize weights
        self._init_weights(init_std)
        
        # Apply freeze mode
        self._apply_freeze_mode()
    
    def _build_enhanced_multi_task_head(self, classifier_dim: int):
        """Build enhanced multi-task head with shared and task-specific layers."""
        
        # Activation function
        if self.activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif self.activation == 'gelu':
            act_fn = nn.GELU()
        elif self.activation == 'swish':
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU(inplace=True)
        
        # Shared feature extraction layers
        shared_layers = []
        current_dim = classifier_dim
        
        for i in range(self.num_shared_layers):
            # Determine output dimension for this layer
            if i == self.num_shared_layers - 1:
                # Last shared layer goes to hidden_dim
                out_dim = self.hidden_dim
            else:
                # Intermediate layers gradually increase dimension
                out_dim = classifier_dim + (self.hidden_dim - classifier_dim) * (i + 1) // self.num_shared_layers
            
            # Linear layer
            shared_layers.append(nn.Linear(current_dim, out_dim))
            
            # Batch normalization - use GroupNorm for batch_size=1 compatibility
            if self.use_batch_norm:
                # Calculate appropriate number of groups for GroupNorm
                # Use 32 groups if possible, otherwise use factors of out_dim
                num_groups = 32
                if out_dim % num_groups != 0:
                    # Find largest factor of out_dim that's <= 32
                    for ng in [16, 8, 4, 2, 1]:
                        if out_dim % ng == 0:
                            num_groups = ng
                            break
                
                shared_layers.append(nn.GroupNorm(num_groups, out_dim))
                print(f"✅ Added GroupNorm: {num_groups} groups, {out_dim} channels")
            
            # Activation
            shared_layers.append(act_fn)
            
            # Dropout
            if self.dropout_p is not None and self.dropout_p > 0:
                shared_layers.append(nn.Dropout(self.dropout_p))
            
            current_dim = out_dim
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, num_cls in zip(self.task_names, self.num_classes_per_task):
            task_layers = []
            task_current_dim = self.hidden_dim
            
            # Task-specific intermediate layers
            for i in range(self.task_specific_layers):
                if i == self.task_specific_layers - 1:
                    # Last layer goes to task's number of classes
                    out_dim = num_cls
                else:
                    # Intermediate task-specific layers
                    out_dim = self.hidden_dim // 2
                
                task_layers.append(nn.Linear(task_current_dim, out_dim))
                
                # Only add activation/dropout for intermediate layers, not final layer
                if i < self.task_specific_layers - 1:
                    if self.use_batch_norm:
                        # Calculate appropriate number of groups for GroupNorm
                        # Use 32 groups if possible, otherwise use factors of out_dim
                        num_groups = 32
                        if out_dim % num_groups != 0:
                            # Find largest factor of out_dim that's <= 32
                            for ng in [16, 8, 4, 2, 1]:
                                if out_dim % ng == 0:
                                    num_groups = ng
                                    break
                        
                        task_layers.append(nn.GroupNorm(num_groups, out_dim))
                        print(f"✅ Added GroupNorm: {num_groups} groups, {out_dim} channels")
                    task_layers.append(act_fn)
                    if self.dropout_p is not None and self.dropout_p > 0:
                        task_layers.append(nn.Dropout(self.dropout_p))
                
                task_current_dim = out_dim
            
            self.task_heads[task_name] = nn.Sequential(*task_layers)
        
        # Keep single classifier for backward compatibility (will be deprecated)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
    
    def _build_simple_single_task_head(self, classifier_dim: int):
        """Build simple single-task head (backward compatibility)."""
        # Build single classifier
        if self.hidden_dim > classifier_dim and self.num_shared_layers > 0:
            # Use enhanced architecture even for single task
            shared_layers = []
            shared_layers.append(nn.Linear(classifier_dim, self.hidden_dim))
            if self.use_batch_norm:
                shared_layers.append(nn.BatchNorm1d(self.hidden_dim))
            shared_layers.append(nn.ReLU(inplace=True))
            if self.dropout_p is not None and self.dropout_p > 0:
                shared_layers.append(nn.Dropout(self.dropout_p))
            
            self.shared_layers = nn.Sequential(*shared_layers)
            self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
        else:
            # Simple single layer
            self.shared_layers = nn.Identity()
            self.classifier = nn.Linear(classifier_dim, self.num_classes)
        
        # For consistency, also create task_heads for single task
        self.task_heads = nn.ModuleDict({'default': self.classifier})
    
    def _build_pooling_module(self) -> Optional[nn.Module]:
        """Build the pooling module based on pooling_type."""
        if self.pooling_type == 'attention':
            return AttentionPool(self.in_dim)
        # For 'avg', 'max', or None, we handle in forward pass
        return None
    
    def _build_temporal_module(self) -> Optional[nn.Module]:
        """Build the temporal processing module."""
        if self.temporal_module_type == 'tconv':
            return TemporalConv(self.in_dim)
        elif self.temporal_module_type == 'lstm':
            return TemporalLSTM(self.in_dim)
        return None
    
    def _get_classifier_dim(self) -> int:
        """Get the input dimension for the classifier by testing temporal module output."""
        if self._temporal is None:
            return self.in_dim
        
        # Test with dummy input to get actual output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 8, self.in_dim)  # (B=1, T=8, C=in_dim)
            if hasattr(self._temporal, 'forward'):
                try:
                    dummy_output = self._temporal(dummy_input)
                    return dummy_output.shape[-1]
                except Exception:
                    # Fallback to default if testing fails
                    pass
        
        # Fallback logic for known temporal modules
        if self.temporal_module_type == 'lstm':
            # TemporalLSTM: bidirectional LSTM with hidden_dim = in_dim//2 -> output = in_dim
            return self.in_dim
        elif self.temporal_module_type == 'tconv':
            # TemporalConv: maintains input dimension
            return self.in_dim
        
        return self.in_dim
    
    def _init_weights(self, std: float):
        """Initialize weights with given standard deviation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _apply_freeze_mode(self):
        """Apply freeze mode to all parameters."""
        if self.freeze_mode == 'freeze':
            for param in self.parameters():
                param.requires_grad = False
    
    def _reduce_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce spatial dimensions (H, W) from (B,T,H,W,C) to (B,T,C)."""
        if self.pooling_type == 'avg':
            return x.mean(dim=(2, 3))  # mean over H, W
        elif self.pooling_type == 'max':
            return x.amax(dim=(2, 3))  # max over H, W
        else:
            raise ValueError(f"Unsupported pooling type for spatial reduction: {self.pooling_type}")
    
    def forward(self, x: torch.Tensor, clip_mask: Optional[torch.Tensor] = None, return_dict: bool = None) -> Union[Tuple[torch.Tensor, Dict[str, Any]], Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]:
        """
        Forward pass through the head.
        
        Args:
            x: Input tensor, supports multiple formats:
               - (B, C): Single feature vector per sample
               - (B, T, H, W, C): Video with temporal and spatial dimensions
               - (B, N_clips, C): Bag-of-clips with pre-extracted features
               - (B, N_clips, T, H, W, C): Bag-of-clips with full video dimensions
            clip_mask: Optional mask for bag-of-clips (B, N_clips) indicating valid clips
            return_dict: If True, return dict of logits. If False, return single tensor.
                        If None, use self.multi_task to decide.
            
        Returns:
            If return_dict=True (multi-task mode):
                logits_dict: Dict[str, torch.Tensor] mapping task names to logits
                extras: Dict with additional outputs like features, frame_logits
            If return_dict=False (single-task mode):
                logits: (B, num_classes) 
                extras: Dict with additional outputs like features, frame_logits
        """
        if return_dict is None:
            return_dict = self.multi_task
            
        extras = {}
        
        # Handle bag-of-clips input (B, N_clips, ...) -> flatten to (B*N_clips, ...)
        original_batch_size = x.shape[0]
        if x.dim() == 6:  # (B, N_clips, T, H, W, C)
            batch_size, n_clips = x.shape[:2]
            x = x.view(-1, *x.shape[2:])  # (B*N_clips, T, H, W, C)
            bag_of_clips = True
        elif x.dim() == 3 and clip_mask is not None:  # (B, N_clips, C) with mask
            batch_size, n_clips = x.shape[:2]
            x = x.view(-1, x.shape[-1])  # (B*N_clips, C)
            bag_of_clips = True
        else:
            bag_of_clips = False
        
        # Handle 5D input (B,T,H,W,C) or (B*N_clips,T,H,W,C)
        if x.dim() == 5:
            if self.pooling_type is None:
                raise ValueError('pooling=None but 5-D input provided')
            
            if self.pooling_type == 'attention':
                x = self._pool(x)  # (B, C) or (B*N_clips, C) - attention pools over all spatial-temporal dims
            else:
                x = self._reduce_spatial(x)  # (B, T, C) or (B*N_clips, T, C)
        
        # ------------------------------------------------------------------
        # NEW: Handle 4-D feature maps coming from some backbones
        # ------------------------------------------------------------------
        if x.dim() == 4:  # Possible shapes: (B, C, H, W) *or* (B, T, N, C)
            # We interpret the last dimension as the embedding (feature) dimension and
            # collapse all preceding dims (except batch) into a single token/time axis.
            # This turns the tensor into the standard (B, T, C) format that the rest
            # of the head is designed to handle.
            x = x.flatten(1, -2)  # (B, T, C)
            # Dynamically align the feature dimension if it differs from the configured in_dim
            if x.shape[-1] != self.in_dim:
                if not hasattr(self, '_input_proj') or self._input_proj.in_features != x.shape[-1]:
                    # Lazily create the projection layer on first mismatch
                    self._input_proj = nn.Linear(x.shape[-1], self.in_dim).to(x.device)
                x = self._input_proj(x)
        
        # Store frame logits before temporal pooling if localizer is enabled
        if self.localizer is not None and x.dim() == 3:
            # x is (B, T, C), need to add spatial dims for conv3d
            x_for_localizer = x.unsqueeze(2).unsqueeze(3)  # (B, T, 1, 1, C)
            x_for_localizer = x_for_localizer.permute(0, 4, 1, 2, 3)  # (B, C, T, 1, 1)
            frame_logits = self.localizer(x_for_localizer)  # (B, total_classes, T, 1, 1)
            frame_logits = frame_logits.squeeze(-1).squeeze(-1).transpose(1, 2)  # (B, T, total_classes)
            
            if return_dict and self.multi_task and split_concat_logits is not None:
                # Split frame logits by task
                frame_logits_dict = {}
                frame_logits_split = split_concat_logits(frame_logits)
                for task_name in self.task_names:
                    frame_logits_dict[f'frame_logits_{task_name}'] = frame_logits_split[task_name]
                extras.update(frame_logits_dict)
            else:
                extras['frame_logits'] = frame_logits
        
        # Handle temporal dimension
        if x.dim() == 3:  # (B, T, C) or (B*N_clips, T, C)
            if self._temporal is not None:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(self._temporal, x)
                else:
                    x = self._temporal(x)  # (B, C) or (B*N_clips, C)
            else:
                x = x.mean(dim=1)  # Average over temporal dimension
        
        # Handle bag-of-clips pooling
        if bag_of_clips:
            # Reshape back to (B, N_clips, C) for clip pooling
            x = x.view(batch_size, n_clips, -1)
            # Apply clip pooling to aggregate clips into action-level features
            x = self.clip_pooling(x, clip_mask)  # (B, C)
        
        # Store features before shared layers
        extras['feat'] = x
        
        # Pass through shared layers (if any)
        if hasattr(self, 'shared_layers'):
            x = self.shared_layers(x)
        
        # Classification
        if return_dict:
            # Multi-task mode: return dict of logits
            logits_dict = {}
            for task_name in self.task_names:
                task_logits = self.task_heads[task_name](x) * self.logit_scale
                logits_dict[task_name] = task_logits
            return logits_dict, extras
        else:
            # Single-task mode: return single tensor (backward compatibility)
            if self.multi_task:
                # If in multi-task mode but single tensor requested, concatenate all logits
                logits_dict = {}
                for task_name in self.task_names:
                    task_logits = self.task_heads[task_name](x) * self.logit_scale
                    logits_dict[task_name] = task_logits
                
                if concat_task_logits is not None:
                    logits = concat_task_logits(logits_dict)
                else:
                    # Fallback: manual concatenation
                    logits = torch.cat([logits_dict[task] for task in self.task_names], dim=1)
            else:
                # True single-task mode
                logits = self.classifier(x) * self.logit_scale
            
            return logits, extras
    
    def forward_single(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Backward compatibility method: always returns single tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            logits: (B, total_classes) - concatenated logits from all tasks
            extras: Dict with additional outputs
        """
        return self.forward(x, return_dict=False)
    
    def forward_multi(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Multi-task method: always returns dict of logits.
        
        Args:
            x: Input tensor
            
        Returns:
            logits_dict: Dict[task_name, logits] for each task
            extras: Dict with additional outputs
        """
        return self.forward(x, return_dict=True)
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss based on loss_type.
        
        Args:
            logits: (B, num_classes)
            targets: (B,) for CE/focal or (B, num_classes) for BCE
            focal_gamma: Gamma parameter for focal loss
            class_weights: Optional class weights
            
        Returns:
            loss: Scalar loss tensor
        """
        class_weights = class_weights or self.class_weights
        
        # Apply logit adjustment (Balanced Softmax) ONLY when using CE loss.
        # Applying logit adjustment with focal/BCE is not theoretically sound and
        # can result in degenerate solutions (e.g., predicting a single class).
        if self.use_logit_adjustment and self.loss_type == 'ce':
            logits = logits + self.class_prior_logits.to(logits.device)
        
        if self.loss_type == 'ce':
            loss_fn = nn.CrossEntropyLoss(
                weight=class_weights, 
                label_smoothing=self.label_smoothing
            )
            return loss_fn(logits, targets)
        
        elif self.loss_type == 'focal':
            return self._focal_loss(logits, targets, focal_gamma, class_weights)
        
        elif self.loss_type == 'bce':
            loss_fn = nn.BCEWithLogitsLoss(weight=class_weights)
            return loss_fn(logits, targets)
        
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def compute_multi_task_loss(
        self,
        logits_dict: Dict[str, torch.Tensor],
        targets_dict: Dict[str, torch.Tensor],
        task_loss_weights: Optional[Dict[str, float]] = None,
        focal_gamma: Union[float, Dict[str, float]] = 2.0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with task weights flowing from CLI.
        
        Args:
            logits_dict: Dict mapping task names to logits tensors
            targets_dict: Dict mapping task names to target tensors  
            task_loss_weights: Optional weights for each task loss (overrides stored weights)
            focal_gamma: Gamma parameter for focal loss
            
        Returns:
            loss_dict: Dict with 'total_loss' and per-task losses
        """
        loss_dict = {}
        total_loss = 0.0
        # Use provided weights, fallback to stored weights, then empty dict
        task_weights = task_loss_weights or self.task_weights_scalar or {}
        
        for task_name in self.task_names:
            if task_name not in logits_dict or task_name not in targets_dict:
                continue
                
            logits = logits_dict[task_name]
            targets = targets_dict[task_name]
            
            # ------------------------------------------------------------------
            # Apply logit adjustment (Balanced Softmax) ONLY when the task uses
            # Cross-Entropy loss. Applying the class-prior shift when using focal
            # loss (or other loss types) can negate the intended re-weighting and
            # lead to collapsed predictions (e.g. predicting just the majority
            # class). Restricting it to CE avoids this conflict while preserving
            # the benefits of Balanced Softmax for CE tasks.
            # ------------------------------------------------------------------
            if (
                self.use_logit_adjustment
                and task_name in self.class_prior_logits
                and self.loss_types_per_task[self.task_names.index(task_name)] == 'ce'
            ):
                logits = logits + self.class_prior_logits[task_name].to(logits.device)
            
            # Get task-specific loss type and class weights
            task_idx = self.task_names.index(task_name)
            loss_type = self.loss_types_per_task[task_idx]
            class_weights = self.task_weights.get(task_name, None)
            
            # Compute task loss
            if loss_type == 'focal':
                # Support per-task focal gamma: if focal_gamma is a dict, fall back to default 2.0 if task not specified
                if isinstance(focal_gamma, dict):
                    gamma_val = focal_gamma.get(task_name, 2.0)
                else:
                    gamma_val = focal_gamma
                task_loss = self._focal_loss(logits, targets, gamma_val, class_weights)
            elif loss_type == 'ce':
                if class_weights is not None:
                    task_loss = F.cross_entropy(logits, targets, weight=class_weights, 
                                              label_smoothing=self.label_smoothing)
                else:
                    task_loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
            elif loss_type == 'bce':
                if targets.dim() == 1:
                    # Convert to one-hot
                    targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
                else:
                    targets_one_hot = targets.float()
                
                if class_weights is not None:
                    # Apply class weights to BCE
                    pos_weight = class_weights[1] / class_weights[0] if class_weights.numel() >= 2 else 1.0
                    task_loss = F.binary_cross_entropy_with_logits(logits, targets_one_hot, pos_weight=pos_weight)
                else:
                    task_loss = F.binary_cross_entropy_with_logits(logits, targets_one_hot)
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
            
            # Apply task weight
            task_weight = task_weights.get(task_name, 1.0)
            weighted_task_loss = task_loss * task_weight
            
            loss_dict[f'{task_name}_loss'] = task_loss
            loss_dict[f'{task_name}_weighted_loss'] = weighted_task_loss
            total_loss += weighted_task_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def compute_unified_loss(
        self,
        logits_dict: Dict[str, torch.Tensor],
        targets_dict: Dict[str, torch.Tensor],
        weighting_strategy: str = 'uniform',
        focal_gamma: float = 2.0,
        adaptive_weights: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Unified loss computation with advanced weighting strategies.
        
        Args:
            logits_dict: Dict mapping task names to logits tensors
            targets_dict: Dict mapping task names to target tensors
            weighting_strategy: Strategy for task weighting
            focal_gamma: Gamma parameter for focal loss
            adaptive_weights: If True, compute weights based on current performance
            
        Returns:
            loss_dict: Dict with comprehensive loss information
        """
        # First compute basic multi-task loss
        loss_dict = self.compute_multi_task_loss(
            logits_dict, targets_dict, None, focal_gamma, **kwargs
        )
        
        # Compute advanced metrics if available
        if compute_task_metrics is not None:
            with torch.no_grad():
                metrics = compute_task_metrics(logits_dict, targets_dict, self.task_names)
                loss_dict['metrics'] = metrics
                
                # Compute overall metrics
                if compute_overall_metrics is not None:
                    overall_metrics = compute_overall_metrics(metrics)
                    loss_dict['overall_metrics'] = overall_metrics
        
        # Adaptive weighting based on performance
        if adaptive_weights and 'metrics' in loss_dict:
            if compute_task_weights_from_metrics is not None:
                adaptive_task_weights = compute_task_weights_from_metrics(
                    loss_dict['metrics'], weighting_strategy
                )
                
                # Recompute weighted losses with adaptive weights
                total_loss_adaptive = 0.0
                for task_name in self.task_names:
                    if f'{task_name}_loss' in loss_dict:
                        task_loss = loss_dict[f'{task_name}_loss']
                        adaptive_weight = adaptive_task_weights.get(task_name, 1.0)
                        weighted_loss = task_loss * adaptive_weight
                        loss_dict[f'{task_name}_adaptive_loss'] = weighted_loss
                        total_loss_adaptive += weighted_loss
                
                loss_dict['total_loss_adaptive'] = total_loss_adaptive
                loss_dict['adaptive_weights'] = adaptive_task_weights
        
        return loss_dict
    
    def compute_comprehensive_metrics(
        self,
        logits_dict: Dict[str, torch.Tensor],
        targets_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Compute comprehensive metrics including confusion matrices.
        
        Args:
            logits_dict: Dict mapping task names to logits tensors
            targets_dict: Dict mapping task names to target tensors
            
        Returns:
            Dict with metrics, confusion matrices, and formatted tables
        """
        results = {}
        
        if compute_task_metrics is not None:
            # Compute detailed metrics
            metrics = compute_task_metrics(logits_dict, targets_dict, self.task_names)
            results['metrics'] = metrics
            
            # Compute overall metrics
            if compute_overall_metrics is not None:
                overall_metrics = compute_overall_metrics(metrics)
                results['overall_metrics'] = overall_metrics
            
            # Format metrics table
            if format_metrics_table is not None:
                metrics_table = format_metrics_table(metrics)
                results['metrics_table'] = metrics_table
        
        if compute_confusion_matrices is not None:
            # Compute confusion matrices
            confusion_matrices = compute_confusion_matrices(logits_dict, targets_dict, self.task_names)
            results['confusion_matrices'] = confusion_matrices
        
        return results
    
    def _focal_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def update_metrics(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update running metrics (single-task mode)."""
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == targets).float().mean()
            self.running_acc = 0.9 * self.running_acc + 0.1 * acc
            
            # Update confusion matrix
            for t, p in zip(targets.view(-1), preds.view(-1)):
                self.confusion_matrix[t.long(), p.long()] += 1
    
    def update_multi_task_metrics(self, logits_dict: Dict[str, torch.Tensor], targets_dict: Dict[str, torch.Tensor]):
        """Update running metrics for multi-task mode."""
        with torch.no_grad():
            for task_name in self.task_names:
                if task_name not in logits_dict or task_name not in targets_dict:
                    continue
                    
                logits = logits_dict[task_name]
                targets = targets_dict[task_name]
                
                preds = logits.argmax(dim=1)
                acc = (preds == targets).float().mean()
                
                # Update per-task running accuracy
                running_acc_attr = f'running_acc_{task_name}'
                if hasattr(self, running_acc_attr):
                    current_acc = getattr(self, running_acc_attr)
                    setattr(self, running_acc_attr, 0.9 * current_acc + 0.1 * acc)
                
                # Update per-task confusion matrix
                confusion_attr = f'confusion_matrix_{task_name}'
                if hasattr(self, confusion_attr):
                    confusion_matrix = getattr(self, confusion_attr)
                    for t, p in zip(targets.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
    
    def get_task_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current task performance.
        
        Returns:
            Dict with performance summary for each task
        """
        summary = {}
        
        if self.multi_task:
            for task_name in self.task_names:
                task_summary = {}
                
                # Running accuracy
                acc_attr = f'running_acc_{task_name}'
                if hasattr(self, acc_attr):
                    task_summary['running_accuracy'] = getattr(self, acc_attr).item()
                
                # Confusion matrix stats
                cm_attr = f'confusion_matrix_{task_name}'
                if hasattr(self, cm_attr):
                    cm = getattr(self, cm_attr)
                    total_samples = cm.sum().item()
                    if total_samples > 0:
                        diagonal_sum = cm.diag().sum().item()
                        task_summary['confusion_matrix_accuracy'] = diagonal_sum / total_samples
                        task_summary['total_samples'] = total_samples
                
                summary[task_name] = task_summary
        else:
            # Single task summary
            summary['default'] = {
                'running_accuracy': self.running_acc.item(),
                'confusion_matrix_accuracy': self.confusion_matrix.diag().sum().item() / max(self.confusion_matrix.sum().item(), 1),
                'total_samples': self.confusion_matrix.sum().item()
            }
        
        return summary
    
    def reset_metrics(self):
        """Reset all metrics to zero."""
        if self.multi_task:
            for task_name in self.task_names:
                acc_attr = f'running_acc_{task_name}'
                cm_attr = f'confusion_matrix_{task_name}'
                
                if hasattr(self, acc_attr):
                    getattr(self, acc_attr).zero_()
                if hasattr(self, cm_attr):
                    getattr(self, cm_attr).zero_()
        else:
            self.running_acc.zero_()
            self.confusion_matrix.zero_()
    
    def print_performance_summary(self):
        """Print a formatted performance summary."""
        summary = self.get_task_performance_summary()
        
        print("\n" + "="*60)
        print("📊 TASK PERFORMANCE SUMMARY")
        print("="*60)
        
        if self.multi_task:
            for task_name, task_summary in summary.items():
                print(f"\n🎯 {task_name.upper()}")
                print("-" * 30)
                
                if 'running_accuracy' in task_summary:
                    print(f"  Running Accuracy: {task_summary['running_accuracy']:.4f}")
                
                if 'confusion_matrix_accuracy' in task_summary:
                    print(f"  CM Accuracy:      {task_summary['confusion_matrix_accuracy']:.4f}")
                
                if 'total_samples' in task_summary:
                    print(f"  Total Samples:    {task_summary['total_samples']}")
        else:
            task_summary = summary['default']
            print(f"Running Accuracy:     {task_summary['running_accuracy']:.4f}")
            print(f"CM Accuracy:          {task_summary['confusion_matrix_accuracy']:.4f}")
            print(f"Total Samples:        {task_summary['total_samples']}")
        
        print("="*60)
    
    def print_structure(self):
        """Print model structure with parameter counts."""
        print("MVFoulsHead Structure:")
        print("=" * 50)
        
        total_params = 0
        trainable_params = 0
        
        for name, module in self.named_children():
            if hasattr(module, 'parameters'):
                module_params = sum(p.numel() for p in module.parameters())
                module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            else:
                module_params = 0
                module_trainable = 0
            
            total_params += module_params
            trainable_params += module_trainable
            
            print(f"{name:20} | {module_params:>10,} params | {module_trainable:>10,} trainable")
        
        print("=" * 50)
        print(f"{'Total':20} | {total_params:>10,} params | {trainable_params:>10,} trainable")
        print(f"Freeze mode: {self.freeze_mode}")
        print(f"Pooling: {self.pooling_type}")
        print(f"Temporal: {self.temporal_module_type}")
        print(f"Loss type: {self.loss_type}")
    
    def summary_table(self) -> pd.DataFrame:
        """Return a DataFrame summary of parameters."""
        data = []
        
        for name, param in self.named_parameters():
            data.append({
                'name': name,
                'shape': str(tuple(param.shape)),
                'numel': param.numel(),
                'trainable': param.requires_grad,
                'dtype': str(param.dtype)
            })
        
        return pd.DataFrame(data)
    
    def export_onnx(self, path: str, input_shape: Tuple[int, ...] = (1, 1024), export_mode: str = 'concat'):
        """
        Export to ONNX format.
        
        Args:
            path: Output ONNX file path
            input_shape: Input tensor shape
            export_mode: 'concat' for single concatenated output, 'separate' for multiple outputs
        """
        if self.pooling_type == 'attention':
            raise ValueError("ONNX export not supported for attention pooling")
        
        self.eval()
        dummy_input = torch.randn(input_shape)
        
        if self.multi_task and export_mode == 'separate':
            # Create wrapper that returns separate outputs for each task
            class ONNXMultiTaskWrapper(nn.Module):
                def __init__(self, head):
                    super().__init__()
                    self.head = head
                
                def forward(self, x):
                    logits_dict, _ = self.head.forward_multi(x)
                    # Return as tuple for ONNX
                    return tuple(logits_dict[task] for task in self.head.task_names)
            
            onnx_model = ONNXMultiTaskWrapper(self)
            output_names = [f'logits_{task}' for task in self.task_names]
            dynamic_axes = {'input': {0: 'batch_size'}}
            for name in output_names:
                dynamic_axes[name] = {0: 'batch_size'}
                
        else:
            # Create wrapper that returns concatenated logits for ONNX compatibility
            class ONNXWrapper(nn.Module):
                def __init__(self, head):
                    super().__init__()
                    self.head = head
                
                def forward(self, x):
                    logits, _ = self.head.forward_single(x)
                    return logits
            
            onnx_model = ONNXWrapper(self)
            output_names = ['logits']
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        
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
    
    def set_freeze_mode(self, mode: str):
        """Dynamically change freeze mode."""
        self.freeze_mode = mode
        if mode == 'freeze':
            for param in self.parameters():
                param.requires_grad = False
        elif mode == 'trainable':
            for param in self.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Invalid freeze mode: {mode}. Use 'freeze' or 'trainable'.")


def build_head(**kwargs) -> MVFoulsHead:
    """
    Factory function to build MVFoulsHead with default parameters.
    
    Args:
        **kwargs: Arguments to pass to MVFoulsHead constructor
        
    Returns:
        MVFoulsHead instance
    """
    return MVFoulsHead(**kwargs)


def build_multi_task_head(**kwargs) -> MVFoulsHead:
    """
    Factory function to build multi-task MVFoulsHead with MVFouls task metadata.
    
    Args:
        **kwargs: Additional arguments to pass to MVFoulsHead constructor
        
    Returns:
        MVFoulsHead instance configured for multi-task MVFouls classification
    """
    if get_task_metadata is None:
        raise ImportError("Task metadata utilities not available. Make sure utils.py is accessible.")
    
    # Get task metadata
    metadata = get_task_metadata()
    
    # Set multi-task defaults with enhanced architecture
    defaults = {
        'multi_task': True,
        'task_names': metadata['task_names'],
        'num_classes_per_task': metadata['num_classes'],
        'loss_type': 'focal',  # Default loss type
        # Enhanced head architecture for better multi-task learning
        'hidden_dim': 2048,  # Large intermediate layer
        'num_shared_layers': 2,  # Shared feature extraction
        'task_specific_layers': 1,  # Task-specific processing
        'use_batch_norm': True,  # Normalization for stability
        'activation': 'relu',  # Activation function
        'dropout': 0.3,  # Regularization
    }
    
    # Override with user-provided kwargs
    defaults.update(kwargs)
    
    return MVFoulsHead(**defaults)


# Example usage and testing helpers
def test_head_shapes():
    """Test different input shapes and configurations with robust assertions."""
    print("Testing MVFoulsHead shapes...")
    
    # Test configurations
    configs = [
        {'pooling': 'avg', 'temporal_module': None, 'num_classes': 2},
        {'pooling': 'max', 'temporal_module': 'tconv', 'num_classes': 3},
        {'pooling': 'attention', 'temporal_module': 'lstm', 'num_classes': 2},
        {'pooling': None, 'temporal_module': None, 'num_classes': 2},  # For pre-pooled input
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}: {config}")
        
        head = build_head(**config)
        num_classes = config['num_classes']
        
        # Test (B, C) input
        batch_size = 4
        x1 = torch.randn(batch_size, 1024)
        logits1, extras1 = head(x1)
        
        # Assert correct shapes
        assert logits1.shape == (batch_size, num_classes), f"Expected logits shape ({batch_size}, {num_classes}), got {logits1.shape}"
        assert extras1['feat'].shape == (batch_size, 1024), f"Expected feat shape ({batch_size}, 1024), got {extras1['feat'].shape}"
        print(f"  ✓ (B,C) -> logits: {logits1.shape}, feat: {extras1['feat'].shape}")
        
        # Test (B,T,H,W,C) input if pooling is not None
        if config['pooling'] is not None:
            batch_size_2 = 2
            x2 = torch.randn(batch_size_2, 4, 7, 7, 1024)
            logits2, extras2 = head(x2)
            
            # For attention pooling, output shape is different
            if config['pooling'] == 'attention':
                expected_logits_shape = (batch_size_2, num_classes)
                expected_feat_shape = (batch_size_2, 1024)
            else:
                expected_logits_shape = (batch_size_2, num_classes)
                expected_feat_shape = (batch_size_2, 1024)
            
            assert logits2.shape == expected_logits_shape, f"Expected logits shape {expected_logits_shape}, got {logits2.shape}"
            assert 'feat' in extras2, "extras should contain 'feat' key"
            print(f"  ✓ (B,T,H,W,C) -> logits: {logits2.shape}, feat: {extras2['feat'].shape}")
        
        # Test loss computation
        targets = torch.randint(0, num_classes, (batch_size,))
        loss = head.compute_loss(logits1, targets)
        
        # Assert loss is finite and non-NaN
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
        assert not torch.isnan(loss), f"Loss should not be NaN, got {loss}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
        print(f"  ✓ Loss: {loss.item():.4f}")
    
    print("\n" + "="*50)
    print("Testing freeze mode functionality...")
    
    # Test freeze mode
    head_trainable = build_head(freeze_mode='trainable')
    head_frozen = build_head(freeze_mode='freeze')
    
    trainable_params = sum(p.numel() for p in head_trainable.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in head_frozen.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in head_trainable.parameters())
    
    assert trainable_params == total_params, f"All params should be trainable in 'trainable' mode, got {trainable_params}/{total_params}"
    assert frozen_params == 0, f"No params should be trainable in 'freeze' mode, got {frozen_params}"
    print(f"  ✓ Trainable mode: {trainable_params}/{total_params} trainable")
    print(f"  ✓ Frozen mode: {frozen_params}/{total_params} trainable")
    
    # Test dynamic freeze mode change
    head_dynamic = build_head(freeze_mode='trainable')
    head_dynamic.set_freeze_mode('freeze')
    frozen_after_change = sum(p.numel() for p in head_dynamic.parameters() if p.requires_grad)
    assert frozen_after_change == 0, f"Should be frozen after set_freeze_mode('freeze'), got {frozen_after_change} trainable"
    
    head_dynamic.set_freeze_mode('trainable')
    trainable_after_change = sum(p.numel() for p in head_dynamic.parameters() if p.requires_grad)
    assert trainable_after_change > 0, f"Should have trainable params after set_freeze_mode('trainable'), got {trainable_after_change}"
    print(f"  ✓ Dynamic freeze mode changes work correctly")
    
    print("\n" + "="*50)
    print("Testing localizer functionality...")
    
    # Test localizer
    head_with_localizer = build_head(enable_localizer=True, pooling='avg')
    x_temporal = torch.randn(2, 3, 7, 7, 1024)
    logits, extras = head_with_localizer(x_temporal)
    
    assert 'frame_logits' in extras, "extras should contain 'frame_logits' when localizer is enabled"
    expected_frame_shape = (2, 3, 2)  # (B, T, num_classes)
    assert extras['frame_logits'].shape == expected_frame_shape, f"Expected frame_logits shape {expected_frame_shape}, got {extras['frame_logits'].shape}"
    print(f"  ✓ Localizer produces frame_logits: {extras['frame_logits'].shape}")
    
    print("\n" + "="*50)
    print("Testing different loss types...")
    
    # Test different loss types
    test_logits = torch.randn(4, 2)
    test_targets = torch.randint(0, 2, (4,))
    
    for loss_type in ['ce', 'focal', 'bce']:
        head_loss = build_head(loss_type=loss_type, num_classes=2)
        
        if loss_type == 'bce':
            # BCE needs different target format
            test_targets_bce = torch.randint(0, 2, (4, 2)).float()
            loss = head_loss.compute_loss(test_logits, test_targets_bce)
        else:
            loss = head_loss.compute_loss(test_logits, test_targets)
        
        assert torch.isfinite(loss), f"{loss_type} loss should be finite"
        assert not torch.isnan(loss), f"{loss_type} loss should not be NaN"
        print(f"  ✓ {loss_type.upper()} loss: {loss.item():.4f}")
    
    print("\n" + "="*50)
    print("✅ All tests passed successfully!")


if __name__ == "__main__":
    test_head_shapes()
