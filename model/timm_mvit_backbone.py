"""
Timm MViTv2-B Backbone Wrapper
==============================

Provides a Video MViTv2-B backbone using the timm model-zoo. Designed to plug
into MVFouls just like the Swin / torchvision backbones.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import importlib, sys, types

try:
    import timm
except ImportError as e:  # pragma: no cover
    raise ImportError("timm is required for the timm MViTv2-B backbone. Install via `pip install timm`. ") from e


class VideoTimmMViTBackbone(nn.Module):
    """Wraps timm's mvitv2_base model so it matches our backbone API."""

    def __init__(
        self,
        pretrained: bool = True,
        return_pooled: bool = True,
        freeze_mode: str = "none",
        checkpointing: bool = False,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------------------
        # Build timm model (no classifier / no pooling)
        # ---------------------------------------------------------------------
        self._fix_checkpoint_import()
        
        self.model = timm.create_model(
            "mvitv2_base", 
            pretrained=pretrained, 
            num_classes=0, 
            global_pool=""
        )

        # Force pooled output to be compatible with existing head architecture
        # The head expects a single feature vector per frame, not unpooled tokens
        self.return_pooled = True
        self.freeze_mode = freeze_mode
        self._current = -1  # for gradual mode

        # Split blocks into 4 roughly equal stages for freeze control
        self._build_groups()
        self.set_freeze(freeze_mode)

        # Configure timm's internal gradient checkpointing based on flag
        # Disable by default due to TIMM 1.0.16 compatibility issues
        if checkpointing and hasattr(self.model, "set_grad_checkpointing"):
            try:
                self.model.set_grad_checkpointing(True)
                print("Gradient checkpointing enabled successfully")
            except Exception as e:
                print(f"Warning: Could not enable gradient checkpointing: {e}")
                print("Continuing without gradient checkpointing...")
        else:
            if hasattr(self.model, "set_grad_checkpointing"):
                try:
                    self.model.set_grad_checkpointing(False)
                except Exception:
                    pass  # Ignore errors when disabling

        self.out_dim = self.model.num_features
        
        # Test forward pass to determine actual output shape and ensure correct out_dim
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_output = self.model.forward_features(test_input)
            if isinstance(test_output, (list, tuple)):
                test_output = test_output[-1]
            
            if test_output.dim() == 3:  # (B, tokens, C) - need to pool
                self.out_dim = test_output.shape[-1]  # Use actual feature dimension from tokens
            elif test_output.dim() == 2:  # (B, C) - already pooled
                self.out_dim = test_output.shape[-1]

    def _fix_checkpoint_import(self):
        """
        Robust fix for timm's checkpoint import issue.
        
        TIMM 1.0.16 has compatibility issues with gradient checkpointing.
        This method attempts to fix the import issues.
        """
        import torch.utils as torch_utils
        
        # Ensure torch.utils.checkpoint is imported as a module
        try:
            checkpoint_module = importlib.import_module("torch.utils.checkpoint")
            # Force torch.utils.checkpoint to be the module, not the function
            torch_utils.checkpoint = checkpoint_module
        except Exception:
            pass
            
        # Additional monkey-patch for TIMM 1.0.16 compatibility
        try:
            import timm.models.mvitv2 as mvitv2_module
            
            # Check if checkpoint is imported incorrectly
            if hasattr(mvitv2_module, 'checkpoint'):
                checkpoint_attr = mvitv2_module.checkpoint
                
                # If checkpoint is a function but doesn't have a checkpoint attribute, fix it
                if callable(checkpoint_attr) and not hasattr(checkpoint_attr, 'checkpoint'):
                    # Create a proper module-like wrapper
                    class CheckpointModule:
                        def __init__(self, checkpoint_fn):
                            self.checkpoint = checkpoint_fn
                        
                        def __call__(self, *args, **kwargs):
                            return self.checkpoint(*args, **kwargs)
                    
                    mvitv2_module.checkpoint = CheckpointModule(checkpoint_attr)
                
                # Alternative: if it's a CheckpointWrapper, make it callable
                elif hasattr(checkpoint_attr, '__class__') and 'CheckpointWrapper' in str(checkpoint_attr.__class__):
                    if not callable(checkpoint_attr):
                        # Make the wrapper callable by adding __call__ method
                        def make_callable(wrapper):
                            def __call__(self, *args, **kwargs):
                                return self.checkpoint(*args, **kwargs)
                            wrapper.__call__ = __call__.__get__(wrapper, wrapper.__class__)
                            return wrapper
                        mvitv2_module.checkpoint = make_callable(checkpoint_attr)
                        
        except Exception as e:
            # If all else fails, disable gradient checkpointing in the model
            print(f"Warning: Could not fix checkpoint import, will disable gradient checkpointing: {e}")
            pass

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers (similar interface to other backbones)
    # ------------------------------------------------------------------
    def _build_groups(self):
        """Patch embed + 4 stages split from self.model.blocks"""
        self._groups: List[List[nn.Module]] = []

        # Group 0: patch embed & pos drop
        first_group = []
        if hasattr(self.model, "patch_embed"):
            first_group.append(self.model.patch_embed)
        if hasattr(self.model, "pos_drop"):
            first_group.append(self.model.pos_drop)
        self._groups.append(first_group)

        # Remaining groups: divide blocks equally into 4 stages
        if hasattr(self.model, "blocks"):
            blocks = list(self.model.blocks)
            n = len(blocks) // 4 or 1
            for i in range(4):
                start, end = i * n, (i + 1) * n if i < 3 else len(blocks)
                self._groups.append([nn.Sequential(*blocks[start:end])])
        else:
            # Fallback: treat whole model as single stage
            self._groups.append([self.model])

    def set_freeze(self, mode: str):
        self.freeze_mode = mode
        if mode == "none":
            for p in self.parameters():
                p.requires_grad = True
        elif mode == "freeze_all":
            for p in self.parameters():
                p.requires_grad = False
        elif mode.startswith("freeze_stages"):
            for p in self.parameters():
                p.requires_grad = True
            k = int(mode.replace("freeze_stages", ""))
            for i in range(k + 1):
                for m in self._groups[i]:
                    for p in m.parameters():
                        p.requires_grad = False
        elif mode == "gradual":
            for p in self.parameters():
                p.requires_grad = False
            self._current = -1
        else:
            raise ValueError(f"Unknown freeze_mode: {mode}")

    def next_unfreeze(self):
        if self.freeze_mode != "gradual":
            return
        if self._current >= len(self._groups) - 1:
            return
        self._current += 1
        for m in self._groups[self._current]:
            for p in m.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Support video tensor (B, C, T, H, W) by per-frame embedding then temporal pool."""
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)
            features = self.model.forward_features(x)  # (B*T, tokens, C) or (B*T, C)

            # Select last stage if list returned
            if isinstance(features, (list, tuple)):
                features = features[-1]

            # Handle different output formats
            if features.dim() == 3:  # (B*T, tokens, C) - unpooled tokens
                if self.return_pooled:
                    # Global average pooling over spatial tokens
                    features = features.mean(dim=1)  # (B*T, C)
                else:
                    # Flatten spatial tokens
                    features = features.view(B * T, -1)  # (B*T, tokens*C)
            elif features.dim() == 2:  # (B*T, C) - already pooled
                pass  # Use as-is
            else:
                # Flatten any unexpected dimensions
                features = features.view(B * T, -1)
                
            # Get the actual feature dimension
            feat_dim = features.shape[-1]
            
            # Reshape back to (B, T, feat_dim)
            features = features.view(B, T, feat_dim)

            if self.return_pooled:
                # Temporal pooling
                features = features.mean(dim=1)  # (B, feat_dim)
            return features

        # Fallback for 4D input
        features = self.model.forward_features(x)  # shape (B, tokens, C) or (B, C)
        if isinstance(features, (list, tuple)):
            features = features[-1]
            
        # Handle different output formats for 4D input
        if features.dim() == 3:  # (B, tokens, C)
            if self.return_pooled:
                features = features.mean(dim=1)  # (B, C)
            else:
                features = features.view(features.shape[0], -1)  # (B, tokens*C)
        elif features.dim() == 2:  # (B, C)
            pass  # Already pooled
        else:
            # Flatten unexpected dimensions
            features = features.view(features.shape[0], -1)
            
        return features

    # Convenience getters
    def get_current_unfreeze_stage(self) -> int:
        return self._current

    def get_freeze_mode(self) -> str:
        return self.freeze_mode

    def get_output_dimensions(self):
        return {
            "feature_dim": self.out_dim,
            "spatial_dims": None,
            "temporal_dims": None,
        }


def build_timm_mvit_backbone(
    pretrained: bool = True,
    return_pooled: bool = True,
    freeze_mode: str = "none",
    checkpointing: bool = False,
):
    """Factory for the timm MViTv2-B backbone."""
    return VideoTimmMViTBackbone(
        pretrained=pretrained,
        return_pooled=return_pooled,
        freeze_mode=freeze_mode,
        checkpointing=checkpointing,
    ) 