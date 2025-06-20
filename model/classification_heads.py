"""
Classification heads for Video Swin Transformer.

This module provides various classification heads that can be attached to the
Video Swin Transformer backbone for different tasks in the MVFouls dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import math


class SimpleClassificationHead(nn.Module):
    """Simple classification head with dropout and linear layer."""
    
    def __init__(self, 
                 in_features: int, 
                 num_classes: int,
                 dropout: float = 0.5,
                 hidden_dim: Optional[int] = None):
        """
        Args:
            in_features: Number of input features from backbone
            num_classes: Number of output classes
            dropout: Dropout probability
            hidden_dim: Optional hidden dimension for additional layer
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features from backbone (B, in_features)
        
        Returns:
            Classification logits (B, num_classes)
        """
        x = self.dropout(x)
        return self.classifier(x)


class MultiTaskClassificationHead(nn.Module):
    """
    Multi-task classification head for MVFouls dataset.
    
    Handles multiple classification tasks simultaneously:
    - Action class (main task)
    - Offence type
    - Contact type
    - Body part
    - Upper body part
    - Try to play (binary)
    - Touch ball (binary)
    - Handball (binary)
    """
    
    def __init__(self, 
                 in_features: int,
                 task_configs: Dict[str, int],
                 dropout: float = 0.5,
                 shared_hidden_dim: Optional[int] = None):
        """
        Args:
            in_features: Number of input features from backbone
            task_configs: Dict mapping task name to number of classes
                Example: {
                    'action_class': 10,
                    'offence': 5,
                    'contact': 4,
                    'bodypart': 8,
                    'upper_body_part': 4,
                    'try_to_play': 2,
                    'touch_ball': 2,
                    'handball': 2
                }
            dropout: Dropout probability
            shared_hidden_dim: Optional shared hidden layer dimension
        """
        super().__init__()
        self.task_configs = task_configs
        self.dropout = nn.Dropout(dropout)
        
        # Shared feature processing
        if shared_hidden_dim is not None:
            self.shared_features = nn.Sequential(
                nn.Linear(in_features, shared_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            classifier_input_dim = shared_hidden_dim
        else:
            self.shared_features = nn.Identity()
            classifier_input_dim = in_features
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_configs.items():
            self.task_heads[task_name] = nn.Linear(classifier_input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input features from backbone (B, in_features)
        
        Returns:
            Dictionary mapping task names to logits
        """
        x = self.dropout(x)
        x = self.shared_features(x)
        
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(x)
        
        return outputs


class SeverityRegressionHead(nn.Module):
    """Regression head for severity prediction (continuous value)."""
    
    def __init__(self, 
                 in_features: int,
                 dropout: float = 0.5,
                 hidden_dim: Optional[int] = None,
                 output_activation: str = 'sigmoid'):  # 'sigmoid', 'tanh', 'none'
        """
        Args:
            in_features: Number of input features from backbone
            dropout: Dropout probability
            hidden_dim: Optional hidden dimension
            output_activation: Output activation function
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        if hidden_dim is not None:
            self.regressor = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.regressor = nn.Linear(in_features, 1)
        
        # Output activation
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features from backbone (B, in_features)
        
        Returns:
            Severity predictions (B, 1)
        """
        x = self.dropout(x)
        x = self.regressor(x)
        return self.output_activation(x)


class HierarchicalClassificationHead(nn.Module):
    """
    Hierarchical classification head that models dependencies between tasks.
    
    Uses action class predictions to condition other predictions.
    """
    
    def __init__(self, 
                 in_features: int,
                 action_classes: int,
                 secondary_tasks: Dict[str, int],
                 dropout: float = 0.5,
                 hidden_dim: int = 512):
        """
        Args:
            in_features: Number of input features from backbone
            action_classes: Number of action classes (main task)
            secondary_tasks: Dict of secondary tasks and their class counts
            dropout: Dropout probability
            hidden_dim: Hidden dimension for feature processing
        """
        super().__init__()
        self.action_classes = action_classes
        self.secondary_tasks = secondary_tasks
        
        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Action classification head (primary)
        self.action_head = nn.Linear(hidden_dim, action_classes)
        
        # Secondary heads that use both features and action predictions
        self.secondary_heads = nn.ModuleDict()
        for task_name, num_classes in secondary_tasks.items():
            self.secondary_heads[task_name] = nn.Sequential(
                nn.Linear(hidden_dim + action_classes, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input features from backbone (B, in_features)
        
        Returns:
            Dictionary mapping task names to logits
        """
        # Process features
        features = self.feature_processor(x)
        
        # Action classification (primary task)
        action_logits = self.action_head(features)
        action_probs = F.softmax(action_logits, dim=1)
        
        # Secondary tasks conditioned on action predictions
        outputs = {'action_class': action_logits}
        
        # Concatenate features with action probabilities for secondary tasks
        conditioned_features = torch.cat([features, action_probs], dim=1)
        
        for task_name, head in self.secondary_heads.items():
            outputs[task_name] = head(conditioned_features)
        
        return outputs


class AttentionPoolingHead(nn.Module):
    """
    Classification head with attention-based pooling.
    
    Applies attention mechanism to select important parts of the feature vector
    before classification.
    """
    
    def __init__(self, 
                 in_features: int,
                 num_classes: int,
                 attention_dim: int = 256,
                 dropout: float = 0.5):
        """
        Args:
            in_features: Number of input features
            num_classes: Number of output classes
            attention_dim: Dimension for attention mechanism
            dropout: Dropout probability
        """
        super().__init__()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_features, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, in_features) or (B, N, in_features) for sequence
        
        Returns:
            Classification logits (B, num_classes)
        """
        if x.dim() == 3:
            # Sequence input: apply attention pooling
            attention_weights = self.attention(x)  # (B, N, 1)
            x = torch.sum(x * attention_weights, dim=1)  # (B, in_features)
        
        return self.classifier(x)


class EnsembleHead(nn.Module):
    """
    Ensemble of multiple classification heads for improved performance.
    """
    
    def __init__(self, 
                 in_features: int,
                 num_classes: int,
                 num_heads: int = 3,
                 dropout: float = 0.5,
                 hidden_dim: int = 512):
        """
        Args:
            in_features: Number of input features
            num_classes: Number of output classes
            num_heads: Number of ensemble heads
            dropout: Dropout probability
            hidden_dim: Hidden dimension for each head
        """
        super().__init__()
        self.num_heads = num_heads
        
        # Multiple classification heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
            for _ in range(num_heads)
        ])
        
        # Optional learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
    
    def forward(self, x: torch.Tensor, return_individual: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            x: Input features (B, in_features)
            return_individual: Whether to return individual head outputs
        
        Returns:
            Ensemble prediction or tuple of (ensemble, individual_predictions)
        """
        individual_outputs = [head(x) for head in self.heads]
        
        # Weighted ensemble
        ensemble_weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_output = sum(w * output for w, output in zip(ensemble_weights, individual_outputs))
        
        if return_individual:
            return ensemble_output, individual_outputs
        else:
            return ensemble_output


# Utility function to create heads based on dataset analysis
def create_mvfouls_heads(backbone_features: int, dataset_info: Dict) -> Dict[str, nn.Module]:
    """
    Create appropriate classification heads for MVFouls dataset.
    
    Args:
        backbone_features: Number of features from the backbone
        dataset_info: Information about the dataset (classes, tasks, etc.)
    
    Returns:
        Dictionary of task names to classification heads
    """
    heads = {}
    
    # Example configuration based on MVFouls dataset
    if 'action_classes' in dataset_info:
        # Simple action classification
        heads['action_classifier'] = SimpleClassificationHead(
            in_features=backbone_features,
            num_classes=len(dataset_info['action_classes']),
            dropout=0.5,
            hidden_dim=512
        )
    
    # Multi-task head for all classification tasks
    if 'task_configs' in dataset_info:
        heads['multi_task'] = MultiTaskClassificationHead(
            in_features=backbone_features,
            task_configs=dataset_info['task_configs'],
            dropout=0.5,
            shared_hidden_dim=512
        )
    
    # Severity regression
    heads['severity_regressor'] = SeverityRegressionHead(
        in_features=backbone_features,
        dropout=0.5,
        hidden_dim=256,
        output_activation='sigmoid'  # Assuming severity is in [0, 1]
    )
    
    return heads


if __name__ == "__main__":
    # Test the classification heads
    print("Testing classification heads...")
    
    batch_size = 4
    in_features = 1024  # Example backbone feature dimension
    
    # Test simple head
    simple_head = SimpleClassificationHead(in_features, num_classes=10)
    test_input = torch.randn(batch_size, in_features)
    
    output = simple_head(test_input)
    print(f"Simple head output shape: {output.shape}")
    
    # Test multi-task head
    task_configs = {
        'action_class': 8,
        'offence': 4,
        'contact': 3,
        'bodypart': 6
    }
    
    multi_head = MultiTaskClassificationHead(in_features, task_configs)
    outputs = multi_head(test_input)
    
    print("Multi-task head outputs:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")
    
    # Test severity regression
    severity_head = SeverityRegressionHead(in_features)
    severity_output = severity_head(test_input)
    print(f"Severity head output shape: {severity_output.shape}")
    
    print("✓ All classification heads working correctly!") 