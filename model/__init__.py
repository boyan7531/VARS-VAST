"""
Model package for MVFouls video classification.

This package contains:
- Video Swin Transformer backbone implementations
- Custom classification heads
- Model utilities and configurations
"""

from .video_swin_transformer import VideoSwinTransformer, VideoSwinConfig
from .classification_heads import (
    SimpleClassificationHead,
    MultiTaskClassificationHead,
    SeverityRegressionHead
)
from .mvfouls_model import (
    MVFoulsVideoModel,
    MVFoulsModelConfig,
    create_mvfouls_model,
    create_mvfouls_model_from_dataset
)

__all__ = [
    'VideoSwinTransformer',
    'VideoSwinConfig', 
    'SimpleClassificationHead',
    'MultiTaskClassificationHead',
    'SeverityRegressionHead',
    'MVFoulsVideoModel',
    'MVFoulsModelConfig',
    'create_mvfouls_model',
    'create_mvfouls_model_from_dataset'
] 