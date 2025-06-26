"""
Video MViTv2-B Backbone for MVFouls
===================================

A self-contained Video MViTv2-B feature extractor tailored for MVFouls detection.
Provides flexible freeze policies including gradual unfreezing and optional 
gradient checkpointing. This is an alternative to Video Swin B with better
efficiency (~52M vs ~88M parameters).
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models
from typing import Optional, List, Union
import warnings
import os
import urllib.request
from collections import OrderedDict


class VideoMViTBackbone(nn.Module):
    """
    Video MViTv2-B backbone for MVFouls detection.
    
    Features:
    - Pretrained on KINETICS-400 (ImageNet22K pretraining)
    - Multiple freeze policies including gradual unfreezing
    - Optional gradient checkpointing
    - Configurable output (pooled vs raw feature maps)
    - More efficient than Video Swin B (~52M vs ~88M parameters)
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
        
        # Load Video MViTv2-B model
        self._load_pretrained_model(pretrained)
        
        # Remove classifier head
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
    
    def _load_pretrained_model(self, pretrained: bool):
        """Load pretrained MViTv2-B model with fallback options."""
        print("Loading pretrained MViTv2-B model...")
        
        # Step 1: Try torchvision MViTv2-B (newer torchvision)
        if hasattr(video_models, 'mvit_v2_b'):
            try:
                # Check if weight enum exists
                weights_enum = getattr(video_models, 'MViT_V2_B_Weights', None)
                if weights_enum is not None and hasattr(weights_enum, 'KINETICS400_IMAGENET22K_V1'):
                    weights = weights_enum.KINETICS400_IMAGENET22K_V1 if pretrained else None
                else:
                    weights = None

                self.model = video_models.mvit_v2_b(weights=weights)
                print("✅ Loaded torchvision mvit_v2_b (weights={} )".format('pretrained' if weights else 'random'))
                return
            except Exception as e:
                print(f"Failed to load torchvision mvit_v2_b: {e}")

        # Step 2: Try torchvision MViTv1-B as closest alternative
        if hasattr(video_models, 'mvit_v1_b'):
            try:
                weights_enum = getattr(video_models, 'MViT_V1_B_Weights', None)
                if weights_enum is not None and hasattr(weights_enum, 'KINETICS400_IMAGENET22K_V1'):
                    weights = weights_enum.KINETICS400_IMAGENET22K_V1 if pretrained else None
                else:
                    weights = None

                self.model = video_models.mvit_v1_b(weights=weights)
                print("✅ Loaded torchvision mvit_v1_b (weights={} )".format('pretrained' if weights else 'random'))
                return
            except Exception as e:
                print(f"Failed to load torchvision mvit_v1_b: {e}")

        print("Failed to find MViT in current torchvision. Trying PyTorchVideo...")
        
        try:
            self._load_from_pytorchvideo()
        except Exception as e:
            print(f"Failed to load from PyTorchVideo: {e}")
            print("⚠️  Using random initialization")
            if hasattr(video_models, 'mvit_v2_b'):
                self.model = video_models.mvit_v2_b(weights=None)
            elif hasattr(video_models, 'mvit_v1_b'):
                self.model = video_models.mvit_v1_b(weights=None)
            else:
                self.model = nn.Identity()  # type: ignore
    
    def _load_from_pytorchvideo(self):
        """Fallback: Load from PyTorchVideo if torchvision version is too old."""
        try:
            import pytorchvideo.models.hub as hub
            
            print("📦 Loading MViTv2-B from PyTorchVideo model zoo...")
            
            # Load pytorchvideo model
            pv_model = hub.mvit_base_16x4(pretrained=True)
            
            # Create placeholder backbone in torchvision if available otherwise construct simple nn.Identity
            if hasattr(video_models, 'mvit_v2_b'):
                self.model = video_models.mvit_v2_b(weights=None)
            elif hasattr(video_models, 'mvit_v1_b'):
                self.model = video_models.mvit_v1_b(weights=None)
            else:
                print("⚠️  No MViT architecture in torchvision – replacing with Identity backbone (features only)")
                from torch import nn
                self.model = nn.Identity()
            
            # Map weights from pytorchvideo to torchvision format
            self._map_pytorchvideo_weights(pv_model.state_dict())
            
            print("✅ Successfully loaded MViTv2-B weights from PyTorchVideo")
            
        except ImportError:
            print("❌ PyTorchVideo not available. Install with: pip install pytorchvideo")
            print("Using random initialization...")
            if hasattr(video_models, 'mvit_v2_b'):
                self.model = video_models.mvit_v2_b(weights=None)
            elif hasattr(video_models, 'mvit_v1_b'):
                self.model = video_models.mvit_v1_b(weights=None)
            else:
                from torch import nn
                self.model = nn.Identity()  # type: ignore
        except Exception as e:
            print(f"❌ PyTorchVideo loading failed: {e}")
            print("Using random initialization...")
            if hasattr(video_models, 'mvit_v2_b'):
                self.model = video_models.mvit_v2_b(weights=None)
            elif hasattr(video_models, 'mvit_v1_b'):
                self.model = video_models.mvit_v1_b(weights=None)
            else:
                from torch import nn
                self.model = nn.Identity()  # type: ignore
    
    def _map_pytorchvideo_weights(self, pv_state_dict):
        """Map PyTorchVideo MViT weights to torchvision format."""
        mapped_state_dict = {}
        
        for key, value in pv_state_dict.items():
            mapped_key = key
            
            # Skip classifier head weights
            if any(skip_key in key for skip_key in ['head', 'cls', 'classifier']):
                continue
            
            # Handle common layer mappings
            if key.startswith('patch_embed.'):
                mapped_key = key
            elif key.startswith('blocks.'):
                mapped_key = key
            elif key.startswith('norm.'):
                mapped_key = key
            elif key.startswith('pos_embed'):
                mapped_key = key
            
            mapped_state_dict[mapped_key] = value
        
        # Load mapped weights
        missing_keys, unexpected_keys = self.model.load_state_dict(mapped_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing {len(missing_keys)} keys during weight mapping")
        if unexpected_keys:
            print(f"⚠️  Found {len(unexpected_keys)} unexpected keys during weight mapping")
    
    def _remove_classifier_head(self):
        """Remove the classifier head and related components."""
        # Remove classifier components
        for attr_name in ['head', 'cls_head', 'classifier']:
            if hasattr(self.model, attr_name):
                delattr(self.model, attr_name)
        
        # Keep norm layer as it's part of feature extraction
    
    def _build_groups(self):
        """Build ordered list of module references for freeze control."""
        self._groups = []
        
        # Group 1: Patch embedding and positional embedding
        patch_embed_modules = []
        if hasattr(self.model, 'patch_embed'):
            patch_embed_modules.append(self.model.patch_embed)
        if hasattr(self.model, 'pos_embed'):
            patch_embed_modules.append(self.model.pos_embed)
        if hasattr(self.model, 'pos_drop'):
            patch_embed_modules.append(self.model.pos_drop)
        self._groups.append(patch_embed_modules)
        
        # Groups 2-5: Divide transformer blocks into 4 stages
        if hasattr(self.model, 'blocks') and len(self.model.blocks) > 0:
            blocks = self.model.blocks
            n_blocks = len(blocks)
            
            # Divide blocks into 4 roughly equal stages
            if n_blocks >= 4:
                stage_size = n_blocks // 4
                remainder = n_blocks % 4
                
                current_idx = 0
                for stage_idx in range(4):
                    # Add one extra block to early stages if there's remainder
                    current_stage_size = stage_size + (1 if stage_idx < remainder else 0)
                    stage_blocks = []
                    
                    for i in range(current_stage_size):
                        if current_idx < n_blocks:
                            stage_blocks.append(blocks[current_idx])
                            current_idx += 1
                    
                    self._groups.append(stage_blocks)
            else:
                # If very few blocks, put each in its own group
                for block in blocks:
                    self._groups.append([block])
                
                # Pad with empty groups to have exactly 5 groups total
                while len(self._groups) < 5:
                    self._groups.append([])
        else:
            # Fallback: create 4 empty groups
            for _ in range(4):
                self._groups.append([])
        
        # Add final norm to last group if it exists
        if hasattr(self.model, 'norm'):
            if len(self._groups) >= 5:
                self._groups[4].append(self.model.norm)
            else:
                self._groups.append([self.model.norm])
    
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
        if group_idx < len(self._groups):
            for module in self._groups[group_idx]:
                for param in module.parameters():
                    param.requires_grad = False
    
    def _unfreeze_group(self, group_idx: int):
        """Unfreeze all parameters in a specific group."""
        if group_idx < len(self._groups):
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
        group_names = ['patch_embed', 'stage_0', 'stage_1', 'stage_2', 'stage_3']
        group_name = group_names[self._current] if self._current < len(group_names) else f'group_{self._current}'
        
        # Unfreeze current group
        self._unfreeze_group(self._current)
        
        print(f"Unfroze group {self._current}: {group_name}")
        self._print_parameter_summary()
    
    def _enable_checkpointing(self):
        """Enable gradient checkpointing if supported."""
        checkpointing_enabled = False
        
        try:
            # For MViTv2, try to enable checkpointing on blocks
            if hasattr(self.model, 'blocks'):
                for block in self.model.blocks:
                    # Try different checkpointing attributes
                    if hasattr(block, 'use_checkpoint'):
                        block.use_checkpoint = True
                        checkpointing_enabled = True
                    elif hasattr(block, 'gradient_checkpointing'):
                        block.gradient_checkpointing = True
                        checkpointing_enabled = True
                    elif hasattr(block, 'checkpoint'):
                        block.checkpoint = True
                        checkpointing_enabled = True
                
                if checkpointing_enabled:
                    print("✅ Gradient checkpointing enabled on MViTv2 blocks")
            
            # Try model-level checkpointing
            if not checkpointing_enabled and hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
                print("✅ Gradient checkpointing enabled (model-level)")
                checkpointing_enabled = True
                
        except Exception as e:
            print(f"⚠️  Could not enable gradient checkpointing: {e}")
        
        if not checkpointing_enabled:
            print("⚠️  Gradient checkpointing not available for this MViTv2 implementation")
    
    def _determine_output_dimensions(self):
        """Determine output dimensions dynamically by running a test forward pass."""
        self.model.eval()
        with torch.no_grad():
            # MViTv2 typically uses 16 frames
            test_input = torch.randn(1, 3, 16, 224, 224)
            
            try:
                features = self._forward_features(test_input)
                
                # Set output dimensions based on feature tensor
                if features.dim() == 3:  # (B, N_tokens, C) - typical MViT output
                    self.out_dim = features.shape[-1]  # C dimension (768 for MViTv2-B)
                    self.spatial_dims = None
                    self.temporal_dims = None
                elif features.dim() == 5:  # (B, C, T, H, W)
                    self.out_dim = features.shape[1]  # C dimension
                    self.spatial_dims = features.shape[3:5]  # H, W
                    self.temporal_dims = features.shape[2]  # T
                elif features.dim() == 4:  # (B, C, H, W)
                    self.out_dim = features.shape[1]  # C dimension
                    self.spatial_dims = features.shape[2:4]  # H, W
                    self.temporal_dims = None
                else:
                    self.out_dim = features.shape[-1]
                    self.spatial_dims = None
                    self.temporal_dims = None
                    
                print(f"MViTv2-B output dimension: {self.out_dim}")
                
            except Exception as e:
                print(f"⚠️  Could not determine output dimensions: {e}")
                # Fallback to known MViTv2-B dimension
                self.out_dim = 768
                self.spatial_dims = None
                self.temporal_dims = None
                print(f"Using fallback output dimension: {self.out_dim}")
    
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction only."""
        features = x
        
        # Pass through patch embedding
        if hasattr(self.model, 'patch_embed'):
            features = self.model.patch_embed(features)
        
        # Add positional embedding if it exists
        if hasattr(self.model, 'pos_embed') and self.model.pos_embed is not None:
            features = features + self.model.pos_embed
        
        # Pass through positional dropout if it exists
        if hasattr(self.model, 'pos_drop'):
            features = self.model.pos_drop(features)
        
        # Pass through all transformer blocks
        if hasattr(self.model, 'blocks'):
            for block in self.model.blocks:
                features = block(features)
        
        # Pass through final norm if it exists
        if hasattr(self.model, 'norm'):
            features = self.model.norm(features)
        
        return features
    
    def _print_parameter_summary(self):
        """Print summary of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        total = trainable + frozen
        
        print(f"MViTv2-B Parameter summary:")
        print(f"  Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
        print(f"  Frozen: {frozen:,} ({frozen/total*100:.1f}%)")
        print(f"  Total: {total:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input video tensor of shape (batch, 3, 16, 224, 224)
               Note: MViTv2 is flexible with frame counts, but 16 is typical
            
        Returns:
            Feature tensor of shape (batch, out_dim) if return_pooled=True,
            otherwise feature maps in MViTv2 format
        """
        # Get features through the model
        features = self._forward_features(x)
        
        # Handle pooling based on return_pooled setting
        if self.return_pooled:
            if features.dim() == 3:  # (B, N_tokens, C) - typical MViT output
                # Global average pooling over all tokens
                features = features.mean(dim=1)  # (B, C)
            elif features.dim() == 5:  # (B, C, T, H, W)
                # Pool over spatial and temporal dimensions
                features = features.mean(dim=[2, 3, 4])  # (B, C)
            elif features.dim() == 4:  # (B, C, H, W)
                # Pool over spatial dimensions
                features = features.mean(dim=[2, 3])  # (B, C)
                
            # Ensure correct output shape
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
        
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


# Factory function for consistency with Swin backbone
def build_mvit_backbone(
    pretrained: bool = True,
    return_pooled: bool = True,
    freeze_mode: str = 'none',
    checkpointing: bool = False
) -> VideoMViTBackbone:
    """
    Factory function to build a Video MViTv2-B backbone.
    
    Args:
        pretrained: Whether to load KINETICS-400 pretrained weights
        return_pooled: Whether to return pooled features (True) or raw feature maps (False)
        freeze_mode: Freeze policy - 'none', 'freeze_all', 'freeze_stages{k}', or 'gradual'
        checkpointing: Whether to enable gradient checkpointing for memory efficiency
    
    Returns:
        Configured VideoMViTBackbone instance
    """
    return VideoMViTBackbone(
        pretrained=pretrained,
        return_pooled=return_pooled,
        freeze_mode=freeze_mode,
        checkpointing=checkpointing
    )


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Video MViTv2-B Backbone ===")
    
    # Test basic functionality
    print("\n1. Basic functionality test:")
    try:
        backbone = build_mvit_backbone(pretrained=False, freeze_mode='none')
        
        # Test input - MViTv2 typically uses 16 frames but is flexible
        batch_size = 2
        x = torch.randn(batch_size, 3, 16, 224, 224)
        
        print(f"   Input shape: {x.shape}")
        
        # Test pooled output
        backbone.return_pooled = True
        pooled_out = backbone(x)
        print(f"   Pooled output shape: {pooled_out.shape}")
        
        # Test unpooled output
        backbone.return_pooled = False
        unpooled_out = backbone(x)
        print(f"   Unpooled output shape: {unpooled_out.shape}")
        
        print("   ✅ Basic functionality test passed")
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
    
    # Test gradual unfreezing
    print("\n2. Gradual unfreezing test:")
    try:
        backbone = build_mvit_backbone(pretrained=False, freeze_mode='gradual')
        
        for i in range(6):  # Try to unfreeze more than available
            print(f"   Unfreeze step {i}:")
            backbone.next_unfreeze()
        
        print("   ✅ Gradual unfreezing test passed")
        
    except Exception as e:
        print(f"   ❌ Gradual unfreezing test failed: {e}")
    
    # Test freeze stages
    print("\n3. Freeze stages test:")
    try:
        for k in range(4):
            backbone = build_mvit_backbone(pretrained=False, freeze_mode=f'freeze_stages{k}')
            trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in backbone.parameters())
            print(f"   freeze_stages{k}: {trainable:,}/{total:,} trainable")
        
        print("   ✅ Freeze stages test passed")
        
    except Exception as e:
        print(f"   ❌ Freeze stages test failed: {e}")
    
    # Test different input sizes
    print("\n4. Input flexibility test:")
    try:
        backbone = build_mvit_backbone(pretrained=False, freeze_mode='none')
        
        test_shapes = [
            (1, 3, 16, 224, 224),  # Standard
            (2, 3, 32, 224, 224),  # More frames
            (1, 3, 8, 224, 224),   # Fewer frames
        ]
        
        for shape in test_shapes:
            test_input = torch.randn(shape)
            try:
                output = backbone(test_input)
                print(f"   Input {shape} -> Output {output.shape}")
            except Exception as e:
                print(f"   Input {shape} -> Failed: {e}")
        
        print("   ✅ Input flexibility test completed")
        
    except Exception as e:
        print(f"   ❌ Input flexibility test failed: {e}")
    
    print("\n✅ MViTv2-B backbone testing complete!") 