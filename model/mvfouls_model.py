"""
Complete Video Swin Transformer model for MVFouls dataset.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass

from .video_swin_transformer import VideoSwinTransformer, VideoSwinConfig, create_video_swin_base
from .classification_heads import SimpleClassificationHead, MultiTaskClassificationHead, SeverityRegressionHead


@dataclass
class MVFoulsModelConfig:
    """Configuration for MVFouls model."""
    backbone_config: VideoSwinConfig
    use_multi_task: bool = True
    task_configs: Optional[Dict[str, int]] = None
    head_dropout: float = 0.5
    head_hidden_dim: Optional[int] = 512
    freeze_backbone: bool = False
    freeze_backbone_layers: Optional[List[str]] = None


class MVFoulsVideoModel(nn.Module):
    """Complete Video Swin Transformer model for MVFouls dataset."""
    
    def __init__(self, config: MVFoulsModelConfig):
        super().__init__()
        self.config = config
        
        # Create backbone
        self.backbone = VideoSwinTransformer(config.backbone_config)
        self.backbone_features = self.backbone.get_feature_dim()
        
        # Create classification heads
        self._create_heads()
        
        # Freeze backbone if specified
        if config.freeze_backbone:
            self._freeze_backbone()
        elif config.freeze_backbone_layers:
            self._freeze_backbone_layers(config.freeze_backbone_layers)
    
    def _create_heads(self):
        """Create appropriate classification heads based on configuration."""
        self.heads = nn.ModuleDict()
        
        if self.config.use_multi_task and self.config.task_configs:
            # Multi-task head
            self.heads['multi_task'] = MultiTaskClassificationHead(
                in_features=self.backbone_features,
                task_configs=self.config.task_configs,
                dropout=self.config.head_dropout,
                shared_hidden_dim=self.config.head_hidden_dim
            )
        else:
            # Individual heads for each task
            if self.config.task_configs:
                for task_name, num_classes in self.config.task_configs.items():
                    if task_name == 'severity':
                        # Regression head for severity
                        self.heads[task_name] = SeverityRegressionHead(
                            in_features=self.backbone_features,
                            dropout=self.config.head_dropout,
                            hidden_dim=self.config.head_hidden_dim // 2 if self.config.head_hidden_dim else None
                        )
                    else:
                        # Classification head
                        self.heads[task_name] = SimpleClassificationHead(
                            in_features=self.backbone_features,
                            num_classes=num_classes,
                            dropout=self.config.head_dropout,
                            hidden_dim=self.config.head_hidden_dim
                        )
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def _freeze_backbone_layers(self, layer_names: List[str]):
        """Freeze specific backbone layers."""
        for name, param in self.backbone.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
        print(f"Backbone layers frozen: {layer_names}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the complete model."""
        # Extract features using backbone
        features = self.backbone(x)  # (B, backbone_features)
        
        # Apply classification heads
        outputs = {}
        
        if 'multi_task' in self.heads:
            # Multi-task head returns dictionary
            multi_outputs = self.heads['multi_task'](features)
            outputs.update(multi_outputs)
        else:
            # Individual heads
            for task_name, head in self.heads.items():
                outputs[task_name] = head(features)
        
        return outputs
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone only."""
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        self._freeze_backbone()
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get count of trainable parameters by component."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.heads.parameters() if p.requires_grad)
        
        return {
            'backbone': backbone_params,
            'heads': head_params,
            'total': backbone_params + head_params
        }


def create_mvfouls_model(
    num_classes: Optional[int] = None,
    task_configs: Optional[Dict[str, int]] = None,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **kwargs
) -> MVFoulsVideoModel:
    """Factory function to create MVFouls model with reasonable defaults."""
    
    # Create backbone configuration
    backbone_config = VideoSwinConfig(
        embed_dim=96,
        depths=(2, 2, 6, 2),  # Reduced for memory efficiency
        num_heads=(3, 6, 12, 24),
        pretrained_2d=pretrained,
        **kwargs
    )
    
    # Handle task configuration
    if task_configs is None and num_classes is not None:
        task_configs = {'action_class': num_classes}
    elif task_configs is None:
        # Default MVFouls tasks
        task_configs = {
            'action_class': 8,
            'offence': 4,
            'contact': 3,
            'try_to_play': 2,
            'touch_ball': 2,
            'handball': 2
        }
    
    # Create model configuration
    model_config = MVFoulsModelConfig(
        backbone_config=backbone_config,
        use_multi_task=len(task_configs) > 1,
        task_configs=task_configs,
        freeze_backbone=freeze_backbone
    )
    
    return MVFoulsVideoModel(model_config)


def create_mvfouls_model_from_dataset(dataset, pretrained: bool = True, **kwargs) -> MVFoulsVideoModel:
    """Create MVFouls model based on dataset information."""
    
    # Get dataset information
    dataset_info = dataset.get_split_info()
    
    # Create task configurations based on dataset
    task_configs = {}
    
    if dataset_info.get('has_annotations', False):
        # Get unique action classes
        action_classes = dataset.get_action_classes()
        if action_classes:
            task_configs['action_class'] = len(action_classes)
        
        # Add other common MVFouls tasks
        task_configs.update({
            'offence': 4,
            'contact': 3,
            'try_to_play': 2,
            'touch_ball': 2,
            'handball': 2,
            'severity': 1  # Regression task
        })
    else:
        # Default classification task
        task_configs = {'action_class': 10}
    
    print(f"Creating model with tasks: {list(task_configs.keys())}")
    
    return create_mvfouls_model(
        task_configs=task_configs,
        pretrained=pretrained,
        **kwargs
    )


if __name__ == "__main__":
    # Test the complete model
    print("Testing MVFouls Video Model...")
    
    # Create model with default configuration
    model = create_mvfouls_model(pretrained=False)
    
    # Test input
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 32, 224, 224)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(test_input)
    
    print("Model outputs:")
    for task_name, output in outputs.items():
        print(f"  {task_name}: {output.shape}")
    
    # Check parameter counts
    param_counts = model.get_trainable_parameters()
    print(f"Trainable parameters:")
    for component, count in param_counts.items():
        print(f"  {component}: {count:,}")
    
    print("MVFouls Video Model test completed successfully!") 