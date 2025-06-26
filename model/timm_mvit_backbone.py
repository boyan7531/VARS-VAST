"""
Timm MViTv2-B Backbone Wrapper
==============================

Provides a Video MViTv2-B backbone using the timm model-zoo. Designed to plug
into MVFouls just like the Swin / torchvision backbones.
"""

from typing import List, Optional

import torch
import torch.nn as nn

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
        self.model = timm.create_model(
            "mvitv2_base", pretrained=pretrained, num_classes=0, global_pool=""
        )

        self.return_pooled = return_pooled
        self.freeze_mode = freeze_mode
        self._current = -1  # for gradual mode

        # Split blocks into 4 roughly equal stages for freeze control
        self._build_groups()
        self.set_freeze(freeze_mode)

        # Allow optional checkpointing (timm natively supports it via set_grad_checkpointing)
        if checkpointing and hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing()

        self.out_dim = self.model.num_features

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

            # Pool tokens if needed
            if features.dim() == 3:  # (B*T, tokens, C)
                if self.return_pooled:
                    features = features.mean(dim=1)  # (B*T, C)
            # Reshape back to (B, T, ...)
            features = features.view(B, T, -1)  # (B, T, C)

            if self.return_pooled:
                features = features.mean(dim=1)  # (B, C)
            return features

        # Fallback for 4D input
        features = self.model.forward_features(x)  # shape (B, tokens, C) or (B, C)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.dim() == 3 and self.return_pooled:
            features = features.mean(dim=1)
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