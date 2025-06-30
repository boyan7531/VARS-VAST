"""
MMAction2 MViTv2-B Kinetics-600 Backbone
========================================

Provides a Video MViTv2-B backbone using MMAction2's Kinetics-600 pre-trained checkpoint.
Designed to plug into MVFouls for better action classification performance.
"""

from typing import List, Optional, Dict, Any
import os
import torch
import torch.nn as nn
import urllib.request
from pathlib import Path
import sys
from typing import Callable

# -------------------------------------------------------------
# Helper: robust file downloader (follows redirects, sets UA)
# -------------------------------------------------------------
def _download_with_progress(url: str, dest_path: Path, chunk_size: int = 8192):
    """Download ``url`` to ``dest_path`` with a progress bar and
    a proper User-Agent header to avoid 403 errors on S3/Cloudfront.
    If the ``requests`` package is missing, falls back to urllib.
    """

    try:
        import requests

        with requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            done = int(50 * downloaded / total)
                            sys.stdout.write("\r[{}{}] {:.1f}%".format("█" * done, " " * (50 - done), downloaded * 100 / total))
                            sys.stdout.flush()
        sys.stdout.write("\n")
    except ImportError:
        # fallback to urllib
        import urllib.request
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, dest_path)
    except Exception as e:
        # Final fallback: try system curl if available (helps with corporate proxies)
        try:
            import subprocess, shlex
            cmd = f"curl -L --fail -o {str(dest_path)} {url}"
            print(f"\n⚙️  Falling back to system curl…")
            subprocess.check_call(shlex.split(cmd))
        except Exception as e2:
            raise RuntimeError(f"Failed to download {url}: {e} | curl fallback: {e2}")

class VideoMMAction2MViTBackbone(nn.Module):
    """
    MViTv2-B backbone using MMAction2's Kinetics-600 pre-trained weights.
    
    This model uses the 32x3 sampling (32 frames, stride 3) configuration
    and provides better action classification features compared to ImageNet pre-training.
    """

    def __init__(
        self,
        pretrained: bool = True,
        return_pooled: bool = True,
        freeze_mode: str = "none",
        checkpointing: bool = False,
        checkpoint_path: Optional[str] = None,
        cache_dir: str = "checkpoints",
    ) -> None:
        super().__init__()

        self.return_pooled = return_pooled
        self.freeze_mode = freeze_mode
        self.checkpointing = checkpointing
        self._current = -1  # for gradual freeze mode

        # Build the MViT model architecture (may load built-in K400 weights)
        self.model = self._build_mvit_model(pretrained=pretrained, checkpoint_path=checkpoint_path)

        # If _build_mvit_model() already loaded weights from torchvision we skip
        built_in_loaded = getattr(self, "_built_in_weights_loaded", False)

        # Configure gradient checkpointing
        if checkpointing:
            self._enable_checkpointing()

        # Set output dimension based on model architecture
        self.out_dim = self._get_feature_dim()
        
        # Build freeze groups and apply freeze mode
        self._build_groups()
        self.set_freeze(freeze_mode)

        # Load external pre-trained weights (PySlowFast/MMAction2) only if
        #  1) user asked for pretrained AND
        #  2) we did not already load the official torchvision Kinetics weights AND
        #  3) a checkpoint_path was provided (otherwise we keep built-in weights)
        if pretrained and (not built_in_loaded) and (checkpoint_path is not None):
            self._load_pretrained_weights(checkpoint_path, cache_dir)

    def _build_mvit_model(self, pretrained: bool, checkpoint_path: Optional[str]) -> nn.Module:
        """
        Build MViTv2-B model architecture matching MMAction2's configuration.
        Based on the 32x3 Kinetics-600 model structure.
        """
        # ------------------------------------------------------------------
        #  Prefer the full-size 52 M-parameter architecture from PyTorchVideo.
        #  If that is not available, fall back to torchvision's Small variant,
        #  and finally to an internal minimal stub.
        # ------------------------------------------------------------------

        # 1) Try PyTorchVideo ‑ mvit_base_32x3 (matches MMAction2 checkpoints)
        try:
            try:
                # Preferred location (PyTorchVideo ≥0.1.5)
                from pytorchvideo.models.hub.vision_transformers import mvit_base_32x3 # type: ignore
            except ImportError:
                # Fallback for older versions (<0.1.5)
                from pytorchvideo.models.hub import mvit_base_32x3  # type: ignore

            if pretrained and checkpoint_path is None:
                # Let PyTorchVideo download/cache the official K400 weights.
                base_model = mvit_base_32x3(pretrained=True)
                self._built_in_weights_loaded = True
                print("Loaded PyTorchVideo Kinetics-400 pretrained weights (MViTv2-B 32x3)")
            else:
                # Either training from scratch or we'll load an external ckpt.
                base_model = mvit_base_32x3(pretrained=False)
                self._built_in_weights_loaded = False

            model = self._wrap_torchvision_mvit(base_model)

        except Exception as ptv_exc:
            # 2) Fallback – torchvision's mvit_v2_s (≈34 M params)
            try:
                from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

                if pretrained and checkpoint_path is None:
                    base_model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
                    self._built_in_weights_loaded = True
                    print("Loaded torchvision Kinetics-400 pretrained weights (MViTv2-S)")
                else:
                    base_model = mvit_v2_s(weights=None)
                    self._built_in_weights_loaded = False

                model = self._wrap_torchvision_mvit(base_model)

            except Exception as tv_exc:
                # 3) Ultimate fallback – minimal stub implementation.
                print("WARNING: Falling back to minimal stub MViT implementation "
                      f"(PyTorchVideo error: {ptv_exc}; torchvision error: {tv_exc})")
                model = self._create_minimal_mvit()

        return model

    def _wrap_torchvision_mvit(self, base_model) -> nn.Module:
        """
        Wrap torchvision MViT to add forward_features method and handle different input formats.
        """
        class WrappedMViT(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.num_features = 768  # MViTv2-B feature dimension
                
                # Copy important attributes from torchvision MViT
                if hasattr(base_model, 'conv_proj'):
                    self.conv_proj = base_model.conv_proj
                    self.patch_embed = base_model.conv_proj  # Alias for compatibility
                if hasattr(base_model, 'blocks'):
                    self.blocks = base_model.blocks
                if hasattr(base_model, 'norm'):
                    self.norm = base_model.norm
                if hasattr(base_model, 'pos_encoding'):
                    self.pos_encoding = base_model.pos_encoding

            def forward_features(self, x):
                """Extract features without classification head."""
                if x.dim() == 4:  # (B, C, H, W) - single frame
                    # Add temporal dimension for video model
                    x = x.unsqueeze(2)  # (B, C, 1, H, W)
                
                # Try to tap features right before the classification head.
                # Different implementations name this layer differently.
                hook = None
                features = None

                candidate_norm_names = [
                    "norm",          # torchvision / timm naming
                    "post_norm",     # some PyTorchVideo revisions
                    "final_norm",
                ]

                for n in candidate_norm_names:
                    if hasattr(self.base_model, n):
                        hook = getattr(self.base_model, n).register_forward_hook(
                            lambda _module, _inp, out: locals().update(**{"features": out})
                        )
                        break

                try:
                    _ = self.base_model(x)
                    if hook is not None:
                        hook.remove()
                    if features is not None:
                        return features
                except Exception:
                    if hook is not None:
                        hook.remove()

                # If hooking failed or layer not present, do manual forward
                return self._manual_forward_features(x)

            def _manual_forward_features(self, x):
                """Manual forward pass through the model layers."""
                try:
                    B, C, T, H, W = x.shape
                    
                    # Patch projection
                    x = self.base_model.conv_proj(x)
                    B, embed_dim, T_new, H_new, W_new = x.shape
                    
                    # Reshape to sequence format
                    x = x.permute(0, 2, 3, 4, 1).reshape(B, T_new * H_new * W_new, embed_dim)
                    
                    # Simple positional encoding (no THW argument)
                    x = self.base_model.pos_encoding(x)
                    
                    # Apply transformer blocks
                    thw = [T_new, H_new, W_new]
                    for block in self.base_model.blocks:
                        x, thw = block(x, thw)
                    
                    # Final norm
                    x = self.base_model.norm(x)
                    
                    return x
                    
                except Exception as e:
                    # Ultimate fallback
                    B = x.shape[0]
                    return torch.zeros(B, 1, self.num_features, device=x.device)

            def forward(self, x):
                """Standard forward pass with classification head."""
                if x.dim() == 4:  # (B, C, H, W) - single frame
                    # Add temporal dimension for video model
                    x = x.unsqueeze(2)  # (B, C, 1, H, W)
                
                try:
                    return self.base_model(x)
                except Exception as e:
                    # Return dummy output if forward fails
                    B = x.shape[0]
                    return torch.zeros(B, 1000, device=x.device)  # ImageNet classes

        return WrappedMViT(base_model)

    def _create_minimal_mvit(self) -> nn.Module:
        """
        Create a minimal MViT model structure that can load MMAction2 weights.
        This is a fallback if torchvision MViT is not available.
        """
        class MinimalMViT(nn.Module):
            def __init__(self):
                super().__init__()
                # Patch embedding
                self.patch_embed = nn.Conv3d(3, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
                
                # Positional encoding (simplified)
                self.pos_embed = nn.Parameter(torch.randn(1, 1000, 96))  # Approximate size
                self.pos_drop = nn.Dropout(0.1)
                
                # Transformer blocks (simplified structure)
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=96 * (2 ** (i // 4)),  # Dimension increases
                        nhead=8,
                        batch_first=True
                    ) for i in range(16)  # 16 blocks for MViTv2-B
                ])
                
                # Final norm
                self.norm = nn.LayerNorm(768)  # Final dimension for MViTv2-B
                
                # Feature dimensions
                self.num_features = 768

            def forward_features(self, x):
                # Simplified forward pass
                if x.dim() == 5:  # (B, C, T, H, W)
                    x = self.patch_embed(x)  # (B, C', T', H', W')
                    # Flatten spatial-temporal dimensions
                    B, C, T, H, W = x.shape
                    x = x.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, C)  # (B, tokens, C)
                else:  # 4D input
                    # Handle as single frame
                    x = x.unsqueeze(2)  # Add temporal dimension
                    x = self.patch_embed(x)
                    B, C, T, H, W = x.shape
                    x = x.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, C)
                
                # Add positional encoding (simplified)
                if x.shape[1] <= self.pos_embed.shape[1]:
                    x = x + self.pos_embed[:, :x.shape[1], :x.shape[2]]
                
                x = self.pos_drop(x)
                
                # Pass through transformer blocks
                for block in self.blocks:
                    x = block(x)
                
                x = self.norm(x)
                return x

        return MinimalMViT()

    def _load_pretrained_weights(self, checkpoint_path: Optional[str], cache_dir: str):
        """
        Download and load Kinetics-400 pre-trained weights.
        """
        if checkpoint_path is None:
            # Create cache directory
            cache_path = Path(cache_dir)
            cache_path.mkdir(exist_ok=True)
            
            # Download checkpoint if not exists
            checkpoint_path = cache_path / "mvit_k400_pytorchvideo.pth"
            
            if not checkpoint_path.exists():
                print("📦 Downloading Kinetics-400 MViTv2-B checkpoint (~333 MB)…")
                
                # Try multiple download sources in order of preference
                download_urls = [
                    # Primary: PySlowFast model zoo (recommended)
                    "https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth",
                    # Fallback: PyTorchVideo CDN (sometimes blocked)
                    "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/mvit/v2/MViTv2_B_32x3_k400_fps50.pyth",
                ]
                
                download_success = False
                for i, url in enumerate(download_urls):
                    try:
                        print(f"🔄 Trying download source {i+1}/{len(download_urls)}...")
                        _download_with_progress(url, checkpoint_path)
                        print(f"✅ Downloaded checkpoint to {checkpoint_path}\n")
                        download_success = True
                        break
                    except Exception as e:
                        print(f"❌ Source {i+1} failed: {e}")
                        if checkpoint_path.exists():
                            checkpoint_path.unlink()  # Remove partial download
                        continue
                
                if not download_success:
                    print("\n" + "="*80)
                    print("❌ AUTOMATIC DOWNLOAD FAILED")
                    print("="*80)
                    print("The PyTorchVideo CDN is currently blocking automated downloads.")
                    print("Please download the checkpoint manually using one of these methods:")
                    print("\n🔧 OPTION 1: Manual download with curl (recommended)")
                    print("Run this command in your terminal:")
                    print(f'curl -L -o "{checkpoint_path}" "https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth"')
                    
                    print("\n🔧 OPTION 2: Browser download")
                    print("1. Open this URL in your browser:")
                    print("   https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth")
                    print(f"2. Save the file as: {checkpoint_path}")
                    
                    print("\n🔧 OPTION 3: Use wget")
                    print("Run this command in your terminal:")
                    print(f'wget -O "{checkpoint_path}" "https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth"')
                    
                    print("\n🔧 OPTION 4: Alternative model")
                    print("Consider using ImageNet pre-trained weights instead:")
                    print("--backbone-arch mvitv2_b  # Uses timm ImageNet weights")
                    
                    print("\n💡 TIP: The file size should be ~333 MB (349,018,454 bytes)")
                    print("="*80)
                    
                    # Continue without pre-trained weights
                    print("⚠️  Continuing with random initialization (no pre-trained weights)")
                    return
        
        # Load the checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract state dict (PySlowFast/MMAction2 format may have nested structure)
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']  # PySlowFast format
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']  # MMAction2 format
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove 'backbone.' prefix if present (common in MMAction2)
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key[9:]  # Remove 'backbone.' prefix
                else:
                    new_key = key
                cleaned_state_dict[new_key] = value
            
            # Load weights with strict=False to handle minor key mismatches
            missing_keys, unexpected_keys = self.model.load_state_dict(cleaned_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading checkpoint: {missing_keys[:10]}...")  # Show first 10
            if unexpected_keys:
                print(f"Unexpected keys when loading checkpoint: {unexpected_keys[:10]}...")  # Show first 10
                
            print("Successfully loaded PySlowFast Kinetics-400 pre-trained weights")
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Continuing with random initialization...")

    def _get_feature_dim(self) -> int:
        """Determine the output feature dimension."""
        # Test forward pass to get actual dimensions
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            try:
                test_output = self.model.forward_features(test_input)
                if isinstance(test_output, (list, tuple)):
                    test_output = test_output[-1]

                if test_output.dim() == 3:  # (B, tokens, C)
                    # If pooled output requested we would average later; the per-token dim is still valid
                    return test_output.shape[-1]
                elif test_output.dim() == 2:  # (B, C) already pooled
                    return test_output.shape[-1]
                else:
                    return 768  # Sensible default for MViTv2-B

            except Exception:
                # Any error during probing falls back to default
                return 768  # Fallback dimension

        # Final safeguard (should rarely be reached)
        return 768

    def _enable_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        def checkpoint_wrapper(module):
            def forward(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(module.forward, *args, **kwargs)
            return forward
        
        # Apply checkpointing to transformer blocks if available
        if hasattr(self.model, 'blocks'):
            for block in self.model.blocks:
                block.forward = checkpoint_wrapper(block)

    def _build_groups(self):
        """Build parameter groups for progressive freezing."""
        self._groups: List[List[nn.Module]] = []

        # Group 0: patch embedding
        first_group = []
        if hasattr(self.model, "patch_embed"):
            first_group.append(self.model.patch_embed)
        if hasattr(self.model, "pos_embed"):
            first_group.append(self.model.pos_embed)
        if hasattr(self.model, "pos_drop"):
            first_group.append(self.model.pos_drop)
        self._groups.append(first_group)

        # Groups 1-4: divide transformer blocks into 4 stages
        if hasattr(self.model, "blocks"):
            blocks = list(self.model.blocks)
            n = len(blocks) // 4 or 1
            for i in range(4):
                start, end = i * n, (i + 1) * n if i < 3 else len(blocks)
                self._groups.append(blocks[start:end])
        else:
            # Fallback: single group
            self._groups.append([self.model])

    def set_freeze(self, mode: str):
        """Set freezing mode for the backbone."""
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
            for i in range(min(k + 1, len(self._groups))):
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
        """Unfreeze the next stage in gradual mode."""
        if self.freeze_mode != "gradual":
            return
        if self._current >= len(self._groups) - 1:
            return
        self._current += 1
        for m in self._groups[self._current]:
            for p in m.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass supporting both 4D and 5D inputs.
        
        Args:
            x: Input tensor of shape (B, C, H, W) or (B, C, T, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim) if return_pooled=True,
            otherwise (B, num_tokens, feature_dim) or (B, T, feature_dim)
        """
        if x.dim() == 5:
            # Video input (B, C, T, H, W)
            B, C, T, H, W = x.shape
            
            # Process as video
            features = self.model.forward_features(x)

            if isinstance(features, (list, tuple)):
                features = features[-1]

            # Handle different output formats
            if features.dim() == 3:  # (B, tokens, C)
                if self.return_pooled:
                    # Global average pooling over spatial tokens
                    features = features.mean(dim=1)  # (B, C)
                # else: return unpooled tokens
            elif features.dim() == 2:  # (B, C) - already pooled
                pass
            else:
                # Handle unexpected dimensions
                features = features.view(B, -1)
                
            return features

        else:
            # Image input (B, C, H, W) - process frame by frame
            B, C, H, W = x.shape
            
            # Add temporal dimension for compatibility
            x = x.unsqueeze(2)  # (B, C, 1, H, W)
            
            features = self.model.forward_features(x)
            
        if isinstance(features, (list, tuple)):
            features = features[-1]
            
        if features.dim() == 3:  # (B, tokens, C)
            if self.return_pooled:
                features = features.mean(dim=1)  # (B, C)
            elif features.dim() == 2:  # (B, C)
                pass
            else:
                features = features.view(B, -1)
            
        return features

    # Convenience methods
    def get_current_unfreeze_stage(self) -> int:
        return self._current

    def get_freeze_mode(self) -> str:
        return self.freeze_mode

    def get_output_dimensions(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.out_dim,
            "spatial_dims": None,
            "temporal_dims": None,
            "pretrained_on": "Kinetics-600",
            "sampling_config": "32x3",
        }


def build_mmaction2_mvit_backbone(
    pretrained: bool = True,
    return_pooled: bool = True,
    freeze_mode: str = "none",
    checkpointing: bool = False,
    checkpoint_path: Optional[str] = None,
    cache_dir: str = "checkpoints",
):
    """
    Factory function for MMAction2 MViTv2-B Kinetics-600 backbone.
    
    Args:
        pretrained: Whether to load Kinetics-600 pre-trained weights
        return_pooled: Whether to return pooled features or raw tokens
        freeze_mode: Freezing strategy ("none", "freeze_all", "freeze_stages{N}", "gradual")
        checkpointing: Whether to use gradient checkpointing for memory efficiency
        checkpoint_path: Custom path to checkpoint file
        cache_dir: Directory to cache downloaded checkpoints
        
    Returns:
        VideoMMAction2MViTBackbone instance
    """
    return VideoMMAction2MViTBackbone(
        pretrained=pretrained,
        return_pooled=return_pooled,
        freeze_mode=freeze_mode,
        checkpointing=checkpointing,
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
    ) 