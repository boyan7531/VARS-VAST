"""
Backbone Factory for MVFouls
============================

Generalized factory for building different video backbone architectures.
Currently supports:
- Video Swin B (88M params, high accuracy)
- Video MViTv2-B (52M params, efficient)
"""

from typing import Union
import typing

# Ensure type names are available at runtime for get_type_hints or similar reflection
try:
    from .backbone import VideoSwinBackbone  # noqa: F401
except Exception:
    VideoSwinBackbone = None  # type: ignore

try:
    from .mvit_backbone import VideoMViTBackbone  # noqa: F401
except Exception:
    VideoMViTBackbone = None  # type: ignore


def build_backbone(
    arch: str = 'swin',
    pretrained: bool = True,
    return_pooled: bool = True,
    freeze_mode: str = 'none',
    checkpointing: bool = False,
    **kwargs
) -> 'typing.Any':
    """
    Factory function to build video backbone architectures.
    
    Args:
        arch: Architecture name ('swin', 'mvit')
        pretrained: Whether to load pretrained weights
        return_pooled: Whether to return pooled features
        freeze_mode: Freeze policy ('none', 'freeze_all', 'freeze_stages{k}', 'gradual')
        checkpointing: Whether to enable gradient checkpointing
        **kwargs: Additional architecture-specific arguments
        
    Returns:
        Configured backbone instance
        
    Raises:
        ValueError: If architecture is not supported
    """
    print(f"Building {arch.upper()} backbone...")
    
    if arch.lower() == 'swin':
        if VideoSwinBackbone is None:
            raise ValueError("VideoSwinBackbone is not available")
        
        print("  Architecture: Video Swin B")
        print("  Parameters: ~88M")
        print("  Memory: ~14GB VRAM @ BS=1")
        print("  Strengths: High accuracy, proven performance")
        
        return VideoSwinBackbone(
            pretrained=pretrained,
            return_pooled=return_pooled,
            freeze_mode=freeze_mode,
            checkpointing=checkpointing,
            **kwargs
        )
        
    elif arch.lower() == 'mvit':
        if VideoMViTBackbone is None:
            raise ValueError("VideoMViTBackbone is not available")
        
        print("  Architecture: Video MViTv2-B")
        print("  Parameters: ~52M")
        print("  Memory: ~8GB VRAM @ BS=1")
        print("  Strengths: Efficiency, lower memory usage")
        
        return VideoMViTBackbone(
            pretrained=pretrained,
            return_pooled=return_pooled,
            freeze_mode=freeze_mode,
            checkpointing=checkpointing,
            **kwargs
        )
        
    else:
        available_archs = ['swin', 'mvit']
        raise ValueError(
            f"Unknown backbone architecture: '{arch}'. "
            f"Available architectures: {available_archs}"
        )


def get_backbone_info(arch: str) -> dict:
    """
    Get information about a specific backbone architecture.
    
    Args:
        arch: Architecture name
        
    Returns:
        Dict with architecture information
    """
    info = {
        'swin': {
            'name': 'Video Swin B',
            'params': '~88M',
            'memory_bs1': '~14GB VRAM',
            'memory_bs4': '~32GB VRAM',
            'strengths': ['High accuracy', 'Proven performance', 'Strong pretrained weights'],
            'weaknesses': ['High memory usage', 'Slower inference'],
            'recommended_for': ['High-accuracy requirements', 'Research', 'When compute is not limited']
        },
        'mvit': {
            'name': 'Video MViTv2-B',
            'params': '~52M',
            'memory_bs1': '~8GB VRAM',
            'memory_bs4': '~18GB VRAM',
            'strengths': ['Memory efficient', 'Faster inference', 'Good accuracy'],
            'weaknesses': ['Slightly lower accuracy than Swin', 'Newer architecture'],
            'recommended_for': ['Limited compute', 'Production deployment', 'Efficiency requirements']
        }
    }
    
    return info.get(arch.lower(), {})


def list_available_backbones() -> list:
    """Get list of available backbone architectures."""
    return ['swin', 'mvit']


def print_backbone_comparison():
    """Print a comparison table of available backbones."""
    print("\n" + "="*80)
    print("🏗️  AVAILABLE BACKBONE ARCHITECTURES")
    print("="*80)
    
    for arch in list_available_backbones():
        info = get_backbone_info(arch)
        if info:
            print(f"\n📋 {info['name'].upper()} (--backbone-arch {arch})")
            print(f"   Parameters: {info['params']}")
            print(f"   Memory (BS=1): {info['memory_bs1']}")
            print(f"   Memory (BS=4): {info['memory_bs4']}")
            print(f"   Strengths: {', '.join(info['strengths'])}")
            print(f"   Best for: {', '.join(info['recommended_for'])}")
    
    print("\n💡 RECOMMENDATIONS:")
    print("   • Use 'swin' for maximum accuracy (if you have ≥16GB VRAM)")
    print("   • Use 'mvit' for efficiency and deployment (works with ≥8GB VRAM)")
    print("   • Both support identical freeze modes and training features")
    print("="*80 + "\n")


# Example usage
if __name__ == "__main__":
    print_backbone_comparison()
    
    # Test both architectures
    for arch in ['swin', 'mvit']:
        try:
            print(f"\nTesting {arch} backbone...")
            backbone = build_backbone(arch=arch, pretrained=False, freeze_mode='none')
            print(f"✅ {arch} backbone created successfully")
            print(f"   Output dim: {backbone.out_dim}")
            print(f"   Total params: {sum(p.numel() for p in backbone.parameters()):,}")
        except Exception as e:
            print(f"❌ {arch} backbone failed: {e}") 