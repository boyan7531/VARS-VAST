"""
Video Transforms for Deep Learning Models (e.g., Video Swin Transformer)

This module provides comprehensive video transformation pipelines optimized for 
training deep learning models on video data. The transforms handle temporal 
consistency and proper dimension ordering for video models.

Key Design Principles:
1. Resize-then-Crop Strategy: For proper data augmentation, we resize to a larger 
   size first, then crop to the target size. This provides spatial variety in training.
2. Temporal Consistency: All spatial transforms (crop, flip) are applied consistently 
   across all frames in a video clip.
3. Proper Dimension Ordering: Converts from (T,H,W,C) numpy format to (C,T,H,W) 
   PyTorch tensor format expected by video models.

Usage Patterns:
- Training: Use get_train_transforms() for data augmentation
- Validation/Testing: Use get_val_transforms() for consistent evaluation
- Simple preprocessing: Use get_minimal_transforms() for basic conversion

Important Notes:
- When using transforms with VideoResize, set target_size=None in dataset to avoid double resizing
- For target_size=224, training transforms resize to ~256 then crop to 224
- All transforms expect input videos as (T, H, W, C) numpy arrays with uint8 values [0-255]
- Output tensors are (C, T, H, W) with float32 values normalized appropriately
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random
from typing import Dict, Tuple, List, Union, Optional
from torchvision import transforms as T


class VideoResize:
    """Resize all frames in a video to a specified (height, width)."""
    
    def __init__(self, size: Union[int, Tuple[int, int]], interpolation=cv2.INTER_LINEAR):
        """
        Args:
            size: Target size as (height, width) or single int for square
            interpolation: OpenCV interpolation method
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size  # (height, width)
        self.interpolation = interpolation
    
    def __call__(self, sample: Dict) -> Dict:
        video = sample['video']  # Shape: (T, H, W, C)
        h_target, w_target = self.size
        
        # Ensure video is numpy array for cv2.resize
        if isinstance(video, torch.Tensor):
            video = video.numpy()
        
        resized_frames = []
        for frame in video:
            # Ensure frame is contiguous numpy array for cv2.resize
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            
            # cv2.resize expects (width, height) but we store (height, width)
            resized_frame = cv2.resize(frame, (w_target, h_target), 
                                     interpolation=self.interpolation)
            resized_frames.append(resized_frame)
        
        sample['video'] = np.stack(resized_frames, axis=0)
        return sample


class VideoRandomCrop:
    """Extract a random crop from all frames in a video."""
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        """
        Args:
            size: Crop size as (height, width) or single int for square
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, sample: Dict) -> Dict:
        video = sample['video']  # Shape: (T, H, W, C)
        
        # Ensure video is numpy array
        if isinstance(video, torch.Tensor):
            video = video.numpy()
        
        t, h, w, c = video.shape
        h_crop, w_crop = self.size
        
        if h < h_crop or w < w_crop:
            raise ValueError(f"Video size ({h}, {w}) smaller than crop size {self.size}")
        
        # Random crop position
        top = random.randint(0, h - h_crop)
        left = random.randint(0, w - w_crop)
        
        # Apply same crop to all frames
        cropped_video = video[:, top:top+h_crop, left:left+w_crop, :]
        sample['video'] = cropped_video
        return sample


class VideoCenterCrop:
    """Extract the center crop from all frames in a video."""
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        """
        Args:
            size: Crop size as (height, width) or single int for square
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, sample: Dict) -> Dict:
        video = sample['video']  # Shape: (T, H, W, C)
        
        # Ensure video is numpy array
        if isinstance(video, torch.Tensor):
            video = video.numpy()
        
        t, h, w, c = video.shape
        h_crop, w_crop = self.size
        
        if h < h_crop or w < w_crop:
            raise ValueError(f"Video size ({h}, {w}) smaller than crop size {self.size}")
        
        # Center crop position
        top = (h - h_crop) // 2
        left = (w - w_crop) // 2
        
        # Apply same crop to all frames
        cropped_video = video[:, top:top+h_crop, left:left+w_crop, :]
        sample['video'] = cropped_video
        return sample


class VideoRandomHorizontalFlip:
    """Randomly flip all frames in a video horizontally."""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying the flip
        """
        self.p = p
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            video = sample['video']  # Shape: (T, H, W, C)
            
            # Ensure video is numpy array
            if isinstance(video, torch.Tensor):
                video = video.numpy()
            
            # Flip all frames horizontally (flip along width axis)
            flipped_video = np.flip(video, axis=2).copy()
            sample['video'] = flipped_video
        return sample


class VideoToTensor:
    """Convert video from NumPy array (T, H, W, C) with values 0-255 to PyTorch tensor (C, T, H, W) with values 0.0-1.0."""
    
    def __call__(self, sample: Dict) -> Dict:
        video = sample['video']  # Shape: (T, H, W, C), dtype=uint8, range=0-255
        
        # Handle both numpy arrays and tensors
        if isinstance(video, torch.Tensor):
            # Convert tensor to numpy for consistent processing
            video = video.numpy()
        
        # Convert to float and normalize to [0, 1]
        video = video.astype(np.float32) / 255.0
        
        # Convert to tensor and transpose from (T, H, W, C) to (C, T, H, W)
        video_tensor = torch.from_numpy(video).permute(3, 0, 1, 2)
        
        sample['video'] = video_tensor
        return sample


class VideoNormalize:
    """Normalize video tensor with given mean and standard deviation."""
    
    def __init__(self, mean: List[float], std: List[float]):
        """
        Args:
            mean: Mean values for each channel (RGB)
            std: Standard deviation values for each channel (RGB)
        """
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)  # Shape: (C, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)
    
    def __call__(self, sample: Dict) -> Dict:
        video = sample['video']  # Should be tensor with shape (C, T, H, W)
        
        if not isinstance(video, torch.Tensor):
            raise TypeError("VideoNormalize expects a torch.Tensor")
        
        # Normalize each channel
        normalized_video = (video - self.mean) / self.std
        sample['video'] = normalized_video
        return sample


class VideoCompose:
    """Compose multiple video transforms together."""
    
    def __init__(self, transforms: List):
        """
        Args:
            transforms: List of transform objects
        """
        self.transforms = transforms
    
    def __call__(self, sample: Dict) -> Dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


# Predefined transform compositions for common use cases
def get_train_transforms(size: int = 224) -> VideoCompose:
    """
    Get standard training transforms for video data.
    
    Args:
        size: Target size for frames (square)
    
    Returns:
        Composed transforms for training
    """
    # Resize to larger size first, then crop to target size for augmentation
    resize_size = int(size * 1.143)  # ~256 for size=224, gives room for cropping
    
    return VideoCompose([
        VideoResize((resize_size, resize_size)),  # Resize to larger size first
        VideoRandomCrop((size, size)),           # Then crop to target size
        VideoRandomHorizontalFlip(p=0.5),
        VideoToTensor(),
        VideoNormalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])


def get_val_transforms(size: int = 224) -> VideoCompose:
    """
    Get standard validation/test transforms for video data.
    
    Args:
        size: Target size for frames (square)
    
    Returns:
        Composed transforms for validation/testing
    """
    # For validation, resize to slightly larger then center crop for consistency
    resize_size = int(size * 1.143)  # ~256 for size=224
    
    return VideoCompose([
        VideoResize((resize_size, resize_size)),  # Resize to larger size first
        VideoCenterCrop((size, size)),           # Then center crop to target size
        VideoToTensor(),
        VideoNormalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])


def get_minimal_transforms(size: int = 224) -> VideoCompose:
    """
    Get minimal transforms (just resize and tensor conversion).
    
    Args:
        size: Target size for frames (square)
    
    Returns:
        Minimal transforms
    """
    return VideoCompose([
        VideoResize((size, size)),  # Direct resize for minimal transforms
        VideoToTensor()
    ])


# Additional function for different augmentation strategies
def get_strong_train_transforms(size: int = 224) -> VideoCompose:
    """
    Get stronger training transforms with more augmentation.
    
    Args:
        size: Target size for frames (square)
    
    Returns:
        Composed transforms for training with stronger augmentation
    """
    # Even larger resize for more cropping variety
    resize_size = int(size * 1.25)  # ~280 for size=224
    
    return VideoCompose([
        VideoResize((resize_size, resize_size)),
        VideoRandomCrop((size, size)),
        VideoRandomHorizontalFlip(p=0.5),
        VideoToTensor(),
        VideoNormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# Custom transform for handling different input sizes
class VideoResizeKeepAspect:
    """Resize video while keeping aspect ratio, then pad or crop to target size."""
    
    def __init__(self, size: Union[int, Tuple[int, int]], 
                 mode: str = 'pad', fill_value: int = 0):
        """
        Args:
            size: Target size as (height, width) or single int for square
            mode: 'pad' to pad with fill_value, 'crop' to center crop
            fill_value: Fill value for padding
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.mode = mode
        self.fill_value = fill_value
    
    def __call__(self, sample: Dict) -> Dict:
        video = sample['video']  # Shape: (T, H, W, C)
        t, h, w, c = video.shape
        h_target, w_target = self.size
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(h_target / h, w_target / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize to maintain aspect ratio
        resized_frames = []
        for frame in video:
            resized_frame = cv2.resize(frame, (new_w, new_h), 
                                     interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized_frame)
        
        resized_video = np.stack(resized_frames, axis=0)
        
        if self.mode == 'pad':
            # Pad to target size
            pad_h = h_target - new_h
            pad_w = w_target - new_w
            
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            padded_video = np.pad(
                resized_video,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=self.fill_value
            )
            sample['video'] = padded_video
            
        elif self.mode == 'crop':
            # Center crop to target size
            if new_h >= h_target and new_w >= w_target:
                top = (new_h - h_target) // 2
                left = (new_w - w_target) // 2
                cropped_video = resized_video[:, top:top+h_target, left:left+w_target, :]
                sample['video'] = cropped_video
            else:
                # If still too small after scaling, pad
                return VideoResizeKeepAspect(self.size, mode='pad', 
                                           fill_value=self.fill_value)(sample)
        
        return sample


# Example usage function
def example_usage():
    """Example of how to use the transforms with the MVFouls dataset."""
    
    # Training transforms with data augmentation
    train_transform = get_train_transforms(size=224)
    
    # Validation transforms without augmentation
    val_transform = get_val_transforms(size=224)
    
    # Custom transform composition
    custom_transform = VideoCompose([
        VideoResizeKeepAspect(224, mode='pad'),
        VideoRandomHorizontalFlip(p=0.3),
        VideoToTensor(),
        VideoNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] range
    ])
    
    return train_transform, val_transform, custom_transform


if __name__ == "__main__":
    # Test the transforms
    print("Video transforms implemented successfully!")
    
    # Create dummy video data for testing
    dummy_video = np.random.randint(0, 256, (32, 256, 256, 3), dtype=np.uint8)
    dummy_sample = {'video': dummy_video, 'action_id': 1}
    
    # Test each transform
    transforms_to_test = [
        ("Resize", VideoResize(224)),
        ("RandomCrop", VideoRandomCrop(224)),
        ("CenterCrop", VideoCenterCrop(224)),
        ("RandomFlip", VideoRandomHorizontalFlip()),
        ("ToTensor", VideoToTensor()),
    ]
    
    for name, transform in transforms_to_test:
        try:
            result = transform(dummy_sample.copy())
            print(f"✓ {name}: Output shape = {result['video'].shape}")
        except Exception as e:
            print(f"✗ {name}: Error = {e}")
    
    # Test full pipeline
    try:
        train_transform = get_train_transforms(224)
        result = train_transform(dummy_sample.copy())
        print(f"✓ Full training pipeline: Output shape = {result['video'].shape}")
        print(f"  Output dtype: {result['video'].dtype}")
        print(f"  Output range: [{result['video'].min():.3f}, {result['video'].max():.3f}]")
    except Exception as e:
        print(f"✗ Full training pipeline: Error = {e}")
    
    # Test the resize/crop pipeline step by step
    print(f"\n{'='*50}")
    print("TESTING RESIZE/CROP PIPELINE")
    print(f"{'='*50}")
    
    # Test training transforms step by step
    print("\nTraining transforms pipeline:")
    sample = dummy_sample.copy()
    print(f"1. Input video shape: {sample['video'].shape}")
    
    # Step 1: Resize to larger size
    resize_transform = VideoResize(int(224 * 1.143))  # ~256
    sample = resize_transform(sample)
    print(f"2. After resize to ~256: {sample['video'].shape}")
    
    # Step 2: Random crop to target size
    crop_transform = VideoRandomCrop(224)
    sample = crop_transform(sample)
    print(f"3. After random crop to 224: {sample['video'].shape}")
    
    # Test validation transforms step by step
    print("\nValidation transforms pipeline:")
    sample = dummy_sample.copy()
    print(f"1. Input video shape: {sample['video'].shape}")
    
    # Step 1: Resize to larger size
    sample = resize_transform(sample)
    print(f"2. After resize to ~256: {sample['video'].shape}")
    
    # Step 2: Center crop to target size
    center_crop_transform = VideoCenterCrop(224)
    sample = center_crop_transform(sample)
    print(f"3. After center crop to 224: {sample['video'].shape}")
    
    print(f"\n✓ Resize/crop pipeline working correctly!")
    print(f"  - Training: resize to {int(224 * 1.143)} → random crop to 224")
    print(f"  - Validation: resize to {int(224 * 1.143)} → center crop to 224")
    print("  - This provides proper augmentation space for training!")
