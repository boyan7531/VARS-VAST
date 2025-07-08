"""
Video Swin B Backbone for MVFouls
================================

A self-contained Video Swin B feature extractor tailored for MVFouls detection.
Provides flexible freeze policies including gradual unfreezing and optional 
gradient checkpointing.
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models
from typing import Optional, List, Union
import warnings
import os
import urllib.request
from collections import OrderedDict


class VideoSwinBackbone(nn.Module):
    """
    Video Swin B backbone for MVFouls detection.
    
    Features:
    - Pretrained on KINETICS-600
    - Multiple freeze policies including gradual unfreezing
    - Optional gradient checkpointing
    - Configurable output (pooled vs raw feature maps)
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        return_pooled: bool = True,
        freeze_mode: str = 'none',
        checkpointing: bool = False
    ):
        super().__init__()
        
        self.return_pooled = return_pooled
        self.freeze_mode = freeze_mode
        self._current = -1  # For gradual mode
        
        # Load Video Swin B model
        if pretrained:
            # Load model without pretrained weights first
            self.model = video_models.swin3d_b(weights=None)
            # Load Kinetics-600 weights from official repository
            self._load_kinetics600_weights()
        else:
            self.model = video_models.swin3d_b(weights=None)
        
        # Remove classifier head and final norm
        self._remove_classifier_head()
        
        # Build groups for freeze control
        self._build_groups()
        
        # Apply freeze policy
        self.set_freeze(freeze_mode)
        
        # Enable gradient checkpointing if requested
        if checkpointing:
            self._enable_checkpointing()
        
        # Determine output dimensions dynamically
        self._determine_output_dimensions()
        
        # Print parameter summary
        self._print_parameter_summary()
    
    def _load_kinetics600_weights(self):
        """Load Kinetics-600 pretrained weights from the official repository."""
        # URL for the official Kinetics-600 Video Swin B weights
        weights_url = "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth"
        weights_path = "swin_base_kinetics600.pth"
        
        # Download weights if not already cached
        if not os.path.exists(weights_path):
            print(f"Downloading Kinetics-600 weights from {weights_url}")
            try:
                urllib.request.urlretrieve(weights_url, weights_path)
                print("Successfully downloaded Kinetics-600 weights")
            except Exception as e:
                print(f"Failed to download Kinetics-600 weights: {e}")
                print("Falling back to torchvision Kinetics-400 weights...")
                try:
                    # Use best available torchvision weights as fallback
                    self.model = video_models.swin3d_b(weights=video_models.Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
                    print("Loaded KINETICS400_IMAGENET22K_V1 weights as fallback")
                    return
                except:
                    self.model = video_models.swin3d_b(weights=video_models.Swin3D_B_Weights.KINETICS400_V1)
                    print("Loaded KINETICS400_V1 weights as fallback")
                    return
        
        # Load the downloaded weights
        try:
            print(f"Loading Kinetics-600 weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # Extract state dict (official weights might be wrapped)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Clean up state dict keys and map from official format to torchvision format
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # Remove common prefixes
                name = k
                if name.startswith('backbone.'):
                    name = name[9:]  # Remove 'backbone.'
                if name.startswith('module.'):
                    name = name[7:]   # Remove 'module.'
                
                # Map from official Video Swin format to torchvision format
                # Official uses 'layers.X.blocks.Y' while torchvision uses 'features.X.Y'
                if name.startswith('layers.'):
                    # Convert 'layers.0.blocks.0.' to 'features.0.0.'
                    parts = name.split('.')
                    if len(parts) >= 4 and parts[2] == 'blocks':
                        layer_idx = parts[1]
                        block_idx = parts[3]
                        rest = '.'.join(parts[4:])
                        # Map to torchvision structure
                        if layer_idx == '0':  # Stage 0
                            name = f'features.0.{block_idx}.{rest}'
                        elif layer_idx == '1':  # Stage 1 (after first patch merging)
                            name = f'features.2.{block_idx}.{rest}'
                        elif layer_idx == '2':  # Stage 2 (after second patch merging)  
                            name = f'features.4.{block_idx}.{rest}'
                        elif layer_idx == '3':  # Stage 3 (after third patch merging)
                            name = f'features.6.{block_idx}.{rest}'
                    elif len(parts) >= 3 and parts[2] == 'downsample':
                        # Handle downsample/patch merging layers
                        layer_idx = parts[1]
                        rest = '.'.join(parts[3:])
                        if layer_idx == '0':
                            name = f'features.1.{rest}'
                        elif layer_idx == '1':
                            name = f'features.3.{rest}'
                        elif layer_idx == '2':
                            name = f'features.5.{rest}'
                
                # Handle MLP layer naming differences (fc1/fc2 vs 0/3)
                if '.mlp.fc1.' in name:
                    name = name.replace('.mlp.fc1.', '.mlp.0.')
                elif '.mlp.fc2.' in name:
                    name = name.replace('.mlp.fc2.', '.mlp.3.')
                
                # Skip classifier head weights as we remove them anyway
                if name.startswith('cls_head.') or name.startswith('head.'):
                    continue
                
                new_state_dict[name] = v
            
            # Load state dict into model
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5 for brevity
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5 for brevity
            
            print("Successfully loaded Kinetics-600 pretrained weights!")
            
        except Exception as e:
            print(f"Failed to load Kinetics-600 weights: {e}")
            print("Falling back to torchvision weights...")
            try:
                self.model = video_models.swin3d_b(weights=video_models.Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
                print("Loaded KINETICS400_IMAGENET22K_V1 weights as fallback")
            except:
                self.model = video_models.swin3d_b(weights=video_models.Swin3D_B_Weights.KINETICS400_V1)
                print("Loaded KINETICS400_V1 weights as fallback")
    
    def _remove_classifier_head(self):
        """Remove the classifier head and final norm layer."""
        # Video Swin has a classifier head that we need to remove
        if hasattr(self.model, 'head'):
            delattr(self.model, 'head')
        if hasattr(self.model, 'norm'):
            delattr(self.model, 'norm')
    
    def _build_groups(self):
        """Build ordered list of module references for freeze control."""
        self._groups = []
        
        # Group 1: Patch embedding modules
        patch_embed_modules = []
        if hasattr(self.model, 'patch_embed'):
            patch_embed_modules.append(self.model.patch_embed)
        if hasattr(self.model, 'pos_drop'):
            patch_embed_modules.append(self.model.pos_drop)
        self._groups.append(patch_embed_modules)
        
        # Groups 2-5: Stages 0-3 from features
        # Identify stage modules more robustly
        features = self.model.features
        stage_modules = []
        
        for name, module in features.named_children():
            # Look for Sequential modules that contain transformer blocks
            if isinstance(module, nn.Sequential) and len(module) > 0:
                # Check if this contains transformer-like blocks
                first_child = next(iter(module.children()))
                if hasattr(first_child, 'attn') or 'transformer' in str(type(first_child)).lower():
                    stage_modules.append(module)
                elif hasattr(first_child, 'norm1'):  # SwinTransformerBlock has norm1
                    stage_modules.append(module)
        
        # If we couldn't identify stages properly, fall back to simple division
        if len(stage_modules) == 0:
            print("Warning: Could not identify stage modules, using fallback division")
            # Group features modules by index ranges
            n_modules = len(features)
            if n_modules >= 4:
                # Try to identify potential stages by grouping every other module
                for i in range(0, min(8, n_modules), 2):  # Take modules at even indices
                    if i < len(features):
                        stage_modules.append(features[i])
            else:
                # Very simple fallback - just take first few modules
                for i in range(min(4, n_modules)):
                    stage_modules.append(features[i])
        
        # Add stage modules to groups (up to 4 stages)
        for stage in stage_modules[:4]:
            self._groups.append([stage])
        
        # Ensure we have exactly 5 groups (patch_embed + 4 stages)
        while len(self._groups) < 5:
            self._groups.append([])  # Empty group if needed
    
    def set_freeze(self, mode: str):
        """
        Apply freeze policy to the backbone.
        
        Args:
            mode: One of 'none', 'freeze_all', 'freeze_stages{k}', 'gradual'
        """
        self.freeze_mode = mode
        
        if mode == 'none':
            # Leave all parameters trainable
            for param in self.model.parameters():
                param.requires_grad = True
                
        elif mode == 'freeze_all':
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
                
        elif mode.startswith('freeze_stages'):
            # Extract number of stages to freeze
            try:
                k = int(mode.replace('freeze_stages', ''))
                if k < 0 or k > 3:
                    raise ValueError(f"Invalid freeze_stages value: {k}. Must be 0-3.")
            except ValueError:
                raise ValueError(f"Invalid freeze_mode: {mode}. Expected 'freeze_stages{{k}}' where k is 0-3.")
            
            # First unfreeze all
            for param in self.model.parameters():
                param.requires_grad = True
            
            # Then freeze patch embed + first k stages
            # freeze_stages0: freeze patch_embed only (group 0)
            # freeze_stages1: freeze patch_embed + stage0 (groups 0,1)  
            # freeze_stages2: freeze patch_embed + stage0 + stage1 (groups 0,1,2)
            # freeze_stages3: freeze patch_embed + stage0 + stage1 + stage2 (groups 0,1,2,3)
            for i in range(k + 1):  # +1 to include patch embed
                if i < len(self._groups):
                    self._freeze_group(i)
                        
        elif mode == 'gradual':
            # Freeze everything initially
            for param in self.model.parameters():
                param.requires_grad = False
            self._current = -1
            
        else:
            raise ValueError(f"Unknown freeze_mode: {mode}")
        
        self._print_parameter_summary()
    
    def _freeze_group(self, group_idx: int):
        """Freeze all parameters in a specific group."""
        for module in self._groups[group_idx]:
            for param in module.parameters():
                param.requires_grad = False
    
    def _unfreeze_group(self, group_idx: int):
        """Unfreeze all parameters in a specific group."""
        for module in self._groups[group_idx]:
            for param in module.parameters():
                param.requires_grad = True
    
    def next_unfreeze(self):
        """
        Unfreeze the next group of parameters in gradual mode.
        """
        if self.freeze_mode != 'gradual':
            print("next_unfreeze() only works in gradual mode")
            return
        
        if self._current >= len(self._groups) - 1:
            print("All groups already unfrozen")
            return
        
        self._current += 1
        group_name = ['patch_embed', 'stage_0', 'stage_1', 'stage_2', 'stage_3'][self._current]
        
        # Unfreeze current group using helper method
        self._unfreeze_group(self._current)
        
        print(f"Unfroze group {self._current}: {group_name}")
        self._print_parameter_summary()
    
    def _enable_checkpointing(self):
        """Enable gradient checkpointing if supported."""
        checkpointing_enabled = False
        
        try:
            # Method 1: Try timm-style model-level checkpointing
            if hasattr(self.model, 'set_grad_checkpointing'):
                self.model.set_grad_checkpointing(True)
                print("✅ Gradient checkpointing enabled (timm-style)")
                checkpointing_enabled = True
            
            # Method 2: Check if gradient checkpointing is available
            elif hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
                print("✅ Gradient checkpointing enabled (model-level)")
                checkpointing_enabled = True
            
            # Method 3: Try alternative methods on modules
            else:
                for module in self.model.modules():
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = True
                        checkpointing_enabled = True
                if checkpointing_enabled:
                    print("✅ Gradient checkpointing enabled (module-level)")
            
            # Method 4: Manual checkpointing using torch.utils.checkpoint for Video Swin
            if not checkpointing_enabled and hasattr(self.model, 'features'):
                # Store original features and wrap them with checkpoint
                import torch.utils.checkpoint as checkpoint
                original_features = self.model.features
                
                # Create a wrapper that uses checkpoint
                class CheckpointedSequential(nn.Module):
                    def __init__(self, features):
                        super().__init__()
                        self.features = features
                    
                    def forward(self, x):
                        if self.training:
                            return checkpoint.checkpoint(self.features, x)
                        else:
                            return self.features(x)
                
                self.model.features = CheckpointedSequential(original_features)
                print("✅ Gradient checkpointing enabled (manual torch.utils.checkpoint)")
                checkpointing_enabled = True
                
        except Exception as e:
            print(f"⚠️  Could not enable gradient checkpointing: {e}")
        
        if not checkpointing_enabled:
            print("⚠️  Gradient checkpointing not available for this Video Swin implementation")
    
    def _determine_output_dimensions(self):
        """Determine output dimensions dynamically by running a test forward pass."""
        self.model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 16, 224, 224)
            
            # Get features through the pipeline (same as forward method)
            features = self.model.patch_embed(test_input)
            features = self.model.pos_drop(features)
            features = self.model.features(features)
            # Note: norm layer was removed in _remove_classifier_head
            
            # Set output dimensions
            if features.dim() == 5:  # (B, T, H, W, C)
                self.out_dim = features.shape[-1]  # C dimension
                self.spatial_dims = features.shape[2:4]  # H, W
                self.temporal_dims = features.shape[1]  # T
            else:
                self.out_dim = features.shape[-1]
                self.spatial_dims = None
                self.temporal_dims = None
    
    def _print_parameter_summary(self):
        """Print summary of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        total = trainable + frozen
        
        print(f"Parameter summary:")
        print(f"  Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
        print(f"  Frozen: {frozen:,} ({frozen/total*100:.1f}%)")
        print(f"  Total: {total:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input video tensor of shape (batch, 3, 16, 224, 224)
            
        Returns:
            Feature tensor of shape (batch, out_dim) if return_pooled=True,
            otherwise (batch, out_dim, T', H', W') feature maps
        """
        # Extract features manually following the Video Swin architecture
        features = x
        
        # Pass through patch embedding
        features = self.model.patch_embed(features)
        features = self.model.pos_drop(features)
        
        # Pass through all feature stages
        features = self.model.features(features)
        
        # Note: norm layer was removed in _remove_classifier_head
        
        # Handle pooling
        if self.return_pooled:
            # Manual pooling over spatial and temporal dimensions
            # features shape: (B, T, H, W, C) = (B, 16, 7, 7, 1024)
            if features.dim() == 5:  # (B, T, H, W, C)
                features = features.mean(dim=[1, 2, 3])  # Pool over T, H, W → (B, C)
            elif features.dim() == 4:  # (B, C, H, W)
                features = features.mean(dim=[2, 3])  # (B, C)
            elif features.dim() == 3:  # (B, T, C) - some models return this
                features = features.mean(dim=1)  # (B, C)
            
            # Ensure we have the correct shape
            if features.dim() > 2:
                features = features.view(features.size(0), -1)  # (B, 1024)
        
        return features
    
    def get_current_unfreeze_stage(self) -> int:
        """Get current unfreeze stage for gradual mode."""
        return self._current
    
    def get_freeze_mode(self) -> str:
        """Get current freeze mode."""
        return self.freeze_mode
    
    def get_output_dimensions(self) -> dict:
        """Get output dimensions information."""
        return {
            'feature_dim': self.out_dim,
            'spatial_dims': self.spatial_dims,
            'temporal_dims': self.temporal_dims
        }


def build_swin_backbone(
    pretrained: bool = True,
    return_pooled: bool = True,
    freeze_mode: str = 'none',
    checkpointing: bool = False
) -> VideoSwinBackbone:
    """
    Factory function to build a Video Swin B backbone.
    
    Args:
        pretrained: Whether to load KINETICS-600 pretrained weights
        return_pooled: Whether to return pooled features (True) or raw feature maps (False)
        freeze_mode: Freeze policy - 'none', 'freeze_all', 'freeze_stages{k}', or 'gradual'
        checkpointing: Whether to enable gradient checkpointing for memory efficiency
    
    Returns:
        Configured VideoSwinBackbone instance
    """
    return VideoSwinBackbone(
        pretrained=pretrained,
        return_pooled=return_pooled,
        freeze_mode=freeze_mode,
        checkpointing=checkpointing
    )


# Backward compatibility alias
def build_backbone(
    pretrained: bool = True,
    return_pooled: bool = True,
    freeze_mode: str = 'none',
    checkpointing: bool = False
) -> VideoSwinBackbone:
    """
    DEPRECATED: Use build_swin_backbone() or the generalized factory_backbone.build_backbone().
    Kept for backward compatibility.
    """
    import warnings
    warnings.warn(
        "build_backbone() from backbone.py is deprecated. "
        "Use build_swin_backbone() or factory_backbone.build_backbone(arch='swin').",
        DeprecationWarning,
        stacklevel=2
    )
    return build_swin_backbone(
        pretrained=pretrained,
        return_pooled=return_pooled,
        freeze_mode=freeze_mode,
        checkpointing=checkpointing
    )


# Example usage and testing
if __name__ == "__main__":
    # Test different configurations
    print("=== Testing Video Swin Backbone ===")
    
    # Test basic functionality
    backbone = build_backbone(pretrained=True, freeze_mode='none')
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 16, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    # Test pooled output
    backbone.return_pooled = True
    pooled_out = backbone(x)
    print(f"Pooled output shape: {pooled_out.shape}")
    
    # Test unpooled output
    backbone.return_pooled = False
    unpooled_out = backbone(x)
    print(f"Unpooled output shape: {unpooled_out.shape}")
    
    # Test gradual unfreezing
    print("\n=== Testing Gradual Unfreezing ===")
    backbone = build_backbone(pretrained=True, freeze_mode='gradual')
    
    for i in range(6):  # Try to unfreeze more than available
        print(f"Unfreeze step {i}:")
        backbone.next_unfreeze()
        print()
    
    # Test freeze stages
    print("=== Testing Freeze Stages ===")
    for k in range(4):
        print(f"\nTesting freeze_stages{k}:")
        backbone = build_backbone(pretrained=True, freeze_mode=f'freeze_stages{k}')
        print()
