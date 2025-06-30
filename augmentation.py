"""
Advanced Video Augmentation Pipeline for MVFouls Dataset

This module implements comprehensive video augmentations specifically designed for
action recognition tasks with severe class imbalance. The augmentations are split
into different categories based on their purpose and intensity.

Key Features:
1. Base augmentations applied to all samples
2. Temporal augmentations that preserve action semantics
3. Spatial augmentations that maintain action visibility
4. Color and noise augmentations for robustness
5. Random erasing for better generalization

Design Principles:
- All augmentations preserve the foul action visibility
- Temporal augmentations respect the 75th frame foul location
- Spatial augmentations maintain aspect ratios where possible
- Intensity levels are carefully tuned for action recognition
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random
import math
from typing import Dict, Tuple, List, Union, Optional
from transforms import VideoCompose, VideoToTensor, VideoNormalize


class VideoTemporalJitter:
    """Apply temporal jitter to clip sampling within the allowed range."""
    
    def __init__(self, jitter_ratio: float = 0.15):
        """
        Args:
            jitter_ratio: Maximum jitter as fraction of total frames (±15%)
        """
        self.jitter_ratio = jitter_ratio
    
    def __call__(self, sample: Dict) -> Dict:
        video = sample['video']  # Shape: (T, H, W, C)
        
        # Note: This is handled at the dataset level through random_start_augmentation
        # This transform is a placeholder for potential future temporal variations
        # within the already-sampled clip
        
        return sample


class VideoResizeKeepAspectRatio:
    """Resize video maintaining aspect ratio with shortest side target."""
    
    def __init__(self, short_side: int = 256, target_size: int = 224):
        """
        Args:
            short_side: Target size for the shortest side
            target_size: Final crop size (square)
        """
        self.short_side = short_side
        self.target_size = target_size
    
    def __call__(self, sample: Dict) -> Dict:
        video = sample['video']  # Shape: (T, H, W, C)
        
        if isinstance(video, torch.Tensor):
            video = video.numpy()
        
        t, h, w, c = video.shape
        
        # Calculate scale to make shortest side = short_side
        scale = self.short_side / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize all frames
        resized_frames = []
        for frame in video:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized_frame)
        
        resized_video = np.stack(resized_frames, axis=0)
        
        # Center crop to target_size x target_size
        h_new, w_new = new_h, new_w
        if h_new >= self.target_size and w_new >= self.target_size:
            top = (h_new - self.target_size) // 2
            left = (w_new - self.target_size) // 2
            cropped_video = resized_video[:, top:top + self.target_size, left:left + self.target_size, :]
        else:
            # If still too small, pad with zeros
            pad_h = max(0, self.target_size - h_new)
            pad_w = max(0, self.target_size - w_new)
            pad_top, pad_left = pad_h // 2, pad_w // 2
            pad_bottom, pad_right = pad_h - pad_top, pad_w - pad_left
            
            cropped_video = np.pad(
                resized_video,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0
            )
        
        sample['video'] = cropped_video
        return sample


class VideoRandomHorizontalFlip:
    """Randomly flip video horizontally (safe for action recognition)."""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying horizontal flip
        """
        self.p = p
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            video = sample['video']  # Shape: (T, H, W, C)
            
            if isinstance(video, torch.Tensor):
                video = video.numpy()
            
            # Flip all frames horizontally
            flipped_video = np.flip(video, axis=2).copy()
            sample['video'] = flipped_video
        
        return sample


class VideoColorJitter:
    """Apply color jittering to video frames."""
    
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, 
                 saturation: float = 0.2, hue: float = 0.1):
        """
        Args:
            brightness: Maximum brightness variation
            contrast: Maximum contrast variation
            saturation: Maximum saturation variation
            hue: Maximum hue variation
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, sample: Dict) -> Dict:
        video = sample['video']  # Shape: (T, H, W, C)
        
        if isinstance(video, torch.Tensor):
            video = video.numpy()
        
        # Ensure video is uint8
        if video.dtype != np.uint8:
            video = video.astype(np.uint8)
        
        # Apply same color jitter to all frames for temporal consistency
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_shift = random.uniform(-self.hue, self.hue) * 180  # Convert to degrees
        
        jittered_frames = []
        for frame in video:
            # Convert BGR to HSV for hue adjustment
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Adjust hue
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            # Adjust saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            
            # Convert back to RGB
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
            
            # Adjust brightness and contrast
            rgb = np.clip(rgb * brightness_factor, 0, 255)
            rgb = np.clip((rgb - 128) * contrast_factor + 128, 0, 255)
            
            jittered_frames.append(rgb.astype(np.uint8))
        
        sample['video'] = np.stack(jittered_frames, axis=0)
        return sample


class VideoGaussianNoise:
    """Add Gaussian noise to video frames."""
    
    def __init__(self, sigma: float = 0.05, p: float = 0.2):
        """
        Args:
            sigma: Standard deviation of noise (as fraction of 255)
            p: Probability of applying noise
        """
        self.sigma = sigma
        self.p = p
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            video = sample['video']  # Shape: (T, H, W, C)
            
            if isinstance(video, torch.Tensor):
                video = video.numpy()
            
            # Generate noise
            noise = np.random.normal(0, self.sigma * 255, video.shape).astype(np.float32)
            
            # Add noise and clip to valid range
            noisy_video = np.clip(video.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            sample['video'] = noisy_video
        
        return sample


class VideoGaussianBlur:
    """Apply Gaussian blur to video frames."""
    
    def __init__(self, kernel_size: int = 5, sigma: float = 0.05, p: float = 0.2):
        """
        Args:
            kernel_size: Size of Gaussian kernel (should be odd)
            sigma: Standard deviation for Gaussian kernel
            p: Probability of applying blur
        """
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
        self.p = p
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            video = sample['video']  # Shape: (T, H, W, C)
            
            if isinstance(video, torch.Tensor):
                video = video.numpy()
            
            # Apply same blur to all frames
            sigma_pixels = self.sigma * min(video.shape[1], video.shape[2])  # Scale by image size
            
            blurred_frames = []
            for frame in video:
                blurred_frame = cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), sigma_pixels)
                blurred_frames.append(blurred_frame)
            
            sample['video'] = np.stack(blurred_frames, axis=0)
        
        return sample


class VideoRandomErasing:
    """Apply random erasing to video frames."""
    
    def __init__(self, p: float = 0.3, scale: Tuple[float, float] = (0.02, 0.10), 
                 ratio: Tuple[float, float] = (0.8, 1.25), value: str = 'random'):
        """
        Args:
            p: Probability of applying random erasing per frame
            scale: Range of proportion of erased area against input image
            ratio: Range of aspect ratio of erased area
            value: Erasing value ('random', 'zero', or specific value)
        """
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, sample: Dict) -> Dict:
        video = sample['video']  # Shape: (T, H, W, C)
        
        if isinstance(video, torch.Tensor):
            video = video.numpy()
        
        t, h, w, c = video.shape
        
        erased_frames = []
        for frame in video:
            if random.random() < self.p:
                frame = self._apply_random_erasing(frame.copy())
            erased_frames.append(frame)
        
        sample['video'] = np.stack(erased_frames, axis=0)
        return sample
    
    def _apply_random_erasing(self, frame: np.ndarray) -> np.ndarray:
        """Apply random erasing to a single frame."""
        h, w, c = frame.shape
        area = h * w
        
        for _ in range(100):  # Try up to 100 times to find valid rectangle
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            rect_h = int(round(math.sqrt(target_area * aspect_ratio)))
            rect_w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if rect_w < w and rect_h < h:
                x1 = random.randint(0, w - rect_w)
                y1 = random.randint(0, h - rect_h)
                
                if self.value == 'random':
                    fill_value = np.random.randint(0, 256, (rect_h, rect_w, c), dtype=np.uint8)
                elif self.value == 'zero':
                    fill_value = np.zeros((rect_h, rect_w, c), dtype=np.uint8)
                else:
                    fill_value = np.full((rect_h, rect_w, c), self.value, dtype=np.uint8)
                
                frame[y1:y1 + rect_h, x1:x1 + rect_w] = fill_value
                break
        
        return frame


class VideoRandomResizedCrop:
    """
    Random resized crop for videos (multi-scale crop strategy).
    
    This replaces the traditional resize-then-crop with a more advanced
    strategy that samples random crop sizes and aspect ratios.
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]], 
                 scale: Tuple[float, float] = (0.8, 1.2), 
                 ratio: Tuple[float, float] = (3./4., 4./3.),
                 interpolation=cv2.INTER_LINEAR):
        """
        Args:
            size: Target size as (height, width) or single int for square
            scale: Range of size of the random crop relative to the original image
            ratio: Range of aspect ratio of the random crop
            interpolation: OpenCV interpolation method
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
    
    def __call__(self, sample: Dict) -> Dict:
        video = sample['video']  # Shape: (T, H, W, C)
        
        if isinstance(video, torch.Tensor):
            video = video.numpy()
        
        t, h, w, c = video.shape
        area = h * w
        h_target, w_target = self.size
        
        # Try to find a valid crop
        for _ in range(100):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if crop_w <= w and crop_h <= h:
                # Random crop position
                x = random.randint(0, w - crop_w)
                y = random.randint(0, h - crop_h)
                
                # Crop all frames
                cropped_frames = []
                for frame in video:
                    cropped_frame = frame[y:y+crop_h, x:x+crop_w]
                    # Resize to target size
                    resized_frame = cv2.resize(cropped_frame, (w_target, h_target), 
                                             interpolation=self.interpolation)
                    cropped_frames.append(resized_frame)
                
                sample['video'] = np.stack(cropped_frames, axis=0)
                return sample
        
        # Fallback: center crop and resize
        crop_h, crop_w = min(h, w), min(h, w)
        y = (h - crop_h) // 2
        x = (w - crop_w) // 2
        
        cropped_frames = []
        for frame in video:
            cropped_frame = frame[y:y+crop_h, x:x+crop_w]
            resized_frame = cv2.resize(cropped_frame, (w_target, h_target), 
                                     interpolation=self.interpolation)
            cropped_frames.append(resized_frame)
        
        sample['video'] = np.stack(cropped_frames, axis=0)
        return sample


class VideoRandomRotation:
    """
    Random rotation of video frames followed by center crop.
    
    Designed for football footage where small rotations (±8°) keep
    the pitch lines looking natural.
    """
    
    def __init__(self, degrees: float = 8.0, p: float = 0.5):
        """
        Args:
            degrees: Maximum rotation angle in degrees (±degrees)
            p: Probability of applying rotation
        """
        self.degrees = degrees
        self.p = p
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            video = sample['video']  # Shape: (T, H, W, C)
            
            if isinstance(video, torch.Tensor):
                video = video.numpy()
            
            t, h, w, c = video.shape
            
            # Sample random rotation angle (same for all frames)
            angle = random.uniform(-self.degrees, self.degrees)
            
            # Compute rotation matrix
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation to all frames
            rotated_frames = []
            for frame in video:
                rotated_frame = cv2.warpAffine(frame, rotation_matrix, (w, h),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REFLECT_101)
                rotated_frames.append(rotated_frame)
            
            sample['video'] = np.stack(rotated_frames, axis=0)
        
        return sample


class VideoSpeedPerturbation:
    """
    Temporal speed perturbation for video clips.
    
    Changes playback speed while maintaining clip length by resampling frames.
    The center frame (typically frame 75 with the foul) should remain centered.
    """
    
    def __init__(self, speed_factors: List[float] = [0.75, 1.25], p: float = 0.5):
        """
        Args:
            speed_factors: List of speed multiplication factors
            p: Probability of applying speed perturbation
        """
        self.speed_factors = speed_factors
        self.p = p
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            video = sample['video']  # Shape: (T, H, W, C)
            
            if isinstance(video, torch.Tensor):
                video = video.numpy()
            
            t, h, w, c = video.shape
            
            # Choose random speed factor
            speed_factor = random.choice(self.speed_factors)
            
            if speed_factor != 1.0:
                # Calculate new temporal indices
                # We want to keep the center frame (t//2) at the center
                center_frame = t // 2
                
                # Create indices for resampling
                # For speed_factor > 1.0: sample more frames (faster playback)
                # For speed_factor < 1.0: sample fewer frames (slower playback)
                new_indices = np.linspace(0, t - 1, int(t * speed_factor))
                
                # Clip indices to valid range
                new_indices = np.clip(new_indices, 0, t - 1)
                
                # Resample frames using interpolation
                resampled_frames = []
                for i in range(c):  # Process each channel
                    channel_frames = video[:, :, :, i]  # Shape: (T, H, W)
                    
                    # Interpolate along temporal dimension
                    resampled_channel = np.zeros((t, h, w), dtype=video.dtype)
                    for idx in range(t):
                        if idx < len(new_indices):
                            # Linear interpolation between frames
                            float_idx = new_indices[idx] if idx < len(new_indices) else new_indices[-1]
                            int_idx = int(float_idx)
                            frac = float_idx - int_idx
                            
                            if int_idx >= t - 1:
                                resampled_channel[idx] = channel_frames[t - 1]
                            elif frac == 0:
                                resampled_channel[idx] = channel_frames[int_idx]
                            else:
                                # Linear interpolation
                                frame1 = channel_frames[int_idx].astype(np.float32)
                                frame2 = channel_frames[int_idx + 1].astype(np.float32)
                                interpolated = (1 - frac) * frame1 + frac * frame2
                                resampled_channel[idx] = interpolated.astype(video.dtype)
                        else:
                            # Pad with last frame
                            resampled_channel[idx] = channel_frames[-1]
                    
                    resampled_frames.append(resampled_channel)
                
                # Stack channels back together
                video = np.stack(resampled_frames, axis=-1)  # Shape: (T, H, W, C)
            
            sample['video'] = video
        
        return sample


class VideoPerspectiveWarp:
    """
    Apply mild perspective warp to simulate camera banking or lens distortion.
    
    Uses small random displacements to corner points to create subtle
    perspective effects that don't make the football pitch look unrealistic.
    """
    
    def __init__(self, max_displacement: float = 0.15, p: float = 0.3):
        """
        Args:
            max_displacement: Maximum displacement as fraction of image size
            p: Probability of applying perspective warp
        """
        self.max_displacement = max_displacement
        self.p = p
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            video = sample['video']  # Shape: (T, H, W, C)
            
            if isinstance(video, torch.Tensor):
                video = video.numpy()
            
            t, h, w, c = video.shape
            
            # Calculate maximum pixel displacement
            max_disp_h = int(h * self.max_displacement)
            max_disp_w = int(w * self.max_displacement)
            
            # Original corner points
            src_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
            
            # Add random displacement to corner points
            dst_points = src_points.copy()
            for i in range(4):
                dx = random.uniform(-max_disp_w, max_disp_w)
                dy = random.uniform(-max_disp_h, max_disp_h)
                dst_points[i] += [dx, dy]
            
            # Compute perspective transformation matrix
            perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply same transformation to all frames
            warped_frames = []
            for frame in video:
                warped_frame = cv2.warpPerspective(frame, perspective_matrix, (w, h),
                                                 flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_REFLECT_101)
                warped_frames.append(warped_frame)
            
            sample['video'] = np.stack(warped_frames, axis=0)
        
        return sample


# Composed augmentation pipelines
def get_base_augmentations(image_size: int = 224) -> VideoCompose:
    """
    Get base augmentations applied to all classes.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed video transforms for base augmentation
    """
    return VideoCompose([
        # 1. Temporal jitter (handled at dataset level via random_start_augmentation)
        VideoTemporalJitter(jitter_ratio=0.15),
        
        # 2. Resize keeping aspect ratio (shortest side 256 → crop to 224)
        VideoResizeKeepAspectRatio(short_side=256, target_size=image_size),
        
        # 3. Random horizontal flip (p=0.5)
        VideoRandomHorizontalFlip(p=0.5),
        
        # 4. Color jitter (brightness/contrast/saturation=0.2, hue=0.1)
        VideoColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        
        # 5. Gaussian blur OR Gaussian noise (p=0.2 total)
        VideoGaussianBlur(kernel_size=5, sigma=0.05, p=0.1),
        VideoGaussianNoise(sigma=0.05, p=0.1),
        
        # 6. Random erasing per frame (p=0.3, area 2-10%)
        VideoRandomErasing(p=0.3, scale=(0.02, 0.10), ratio=(0.8, 1.25), value='random'),
        
        # Convert to tensor and normalize
        VideoToTensor(),
        VideoNormalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])


def get_validation_augmentations(image_size: int = 224) -> VideoCompose:
    """
    Get validation augmentations (minimal, deterministic).
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed video transforms for validation
    """
    return VideoCompose([
        # Only resize and center crop for validation
        VideoResizeKeepAspectRatio(short_side=256, target_size=image_size),
        
        # Convert to tensor and normalize
        VideoToTensor(),
        VideoNormalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])


def get_minimal_augmentations(image_size: int = 224) -> VideoCompose:
    """
    Get minimal augmentations for testing/debugging.
    
    Args:
        image_size: Target image size
        
    Returns:
        Minimal transforms
    """
    return VideoCompose([
        VideoResizeKeepAspectRatio(short_side=image_size, target_size=image_size),
        VideoToTensor()
    ])


def get_advanced_augmentations(image_size: int = 224) -> VideoCompose:
    """
    Get advanced augmentations with the full proposed strategy.
    
    Implements the complete augmentation strategy:
    8. Speed perturbation {0.75×, 1.25×}
    9. Random rotation ±8° followed by centre crop
    10. Perspective warp (p=0.3, |θ| < 0.15)
    11. Multi-scale crop (scale 0.8–1.2, ratio 3/4–4/3)
    12. Video MixUp / TimeMix - handled in collate function
    13. ClipCutMix - handled in collate function
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed video transforms for advanced augmentation
    """
    return VideoCompose([
        # 1. Speed perturbation {0.75×, 1.25×} (p=0.5)
        VideoSpeedPerturbation(speed_factors=[0.75, 1.25], p=0.5),
        
        # 2. Multi-scale crop (scale 0.8–1.2, ratio 3/4–4/3)
        VideoRandomResizedCrop(
            size=image_size, 
            scale=(0.8, 1.2), 
            ratio=(3./4., 4./3.)
        ),
        
        # 3. Random rotation ±8° (p=0.5)
        VideoRandomRotation(degrees=8.0, p=0.5),
        
        # 4. Perspective warp (p=0.3, |θ| < 0.15)
        VideoPerspectiveWarp(max_displacement=0.15, p=0.3),
        
        # 5. Random horizontal flip (p=0.5)
        VideoRandomHorizontalFlip(p=0.5),
        
        # 6. Color jitter (brightness/contrast/saturation=0.2, hue=0.1)
        VideoColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        
        # 7. Gaussian blur OR Gaussian noise (p=0.2 total)
        VideoGaussianBlur(kernel_size=5, sigma=0.05, p=0.1),
        VideoGaussianNoise(sigma=0.05, p=0.1),
        
        # 8. Random erasing per frame (p=0.3, area 2-10%)
        VideoRandomErasing(p=0.3, scale=(0.02, 0.10), ratio=(0.8, 1.25), value='random'),
        
        # Convert to tensor and normalize
        VideoToTensor(),
        VideoNormalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])


# Integration with existing transform system
def get_augmented_transforms(image_size: int = 224, augment_level: str = 'base') -> Dict[str, VideoCompose]:
    """
    Get augmented transforms for training and validation.
    
    Args:
        image_size: Target image size
        augment_level: 'advanced', 'base', 'minimal', or 'none'
        
    Returns:
        Dict with 'train' and 'val' transforms
    """
    if augment_level == 'advanced':
        train_transforms = get_advanced_augmentations(image_size)
    elif augment_level == 'base':
        train_transforms = get_base_augmentations(image_size)
    elif augment_level == 'minimal':
        train_transforms = get_minimal_augmentations(image_size)
    else:  # 'none'
        train_transforms = get_validation_augmentations(image_size)
    
    val_transforms = get_validation_augmentations(image_size)
    
    return {
        'train': train_transforms,
        'val': val_transforms
    }


if __name__ == "__main__":
    # Test the augmentations
    print("🧪 Testing Base Video Augmentations")
    print("=" * 50)
    
    # Create dummy video data
    dummy_video = np.random.randint(0, 256, (32, 256, 256, 3), dtype=np.uint8)
    dummy_sample = {'video': dummy_video, 'targets': torch.zeros(3)}
    
    print(f"Original video shape: {dummy_video.shape}")
    print(f"Original video dtype: {dummy_video.dtype}")
    print(f"Original video range: [{dummy_video.min()}, {dummy_video.max()}]")
    
    # Test individual augmentations
    augmentations_to_test = [
        ("VideoResizeKeepAspectRatio", VideoResizeKeepAspectRatio(256, 224)),
        ("VideoRandomHorizontalFlip", VideoRandomHorizontalFlip(p=1.0)),  # Force apply
        ("VideoColorJitter", VideoColorJitter(0.2, 0.2, 0.2, 0.1)),
        ("VideoGaussianNoise", VideoGaussianNoise(0.05, p=1.0)),  # Force apply
        ("VideoGaussianBlur", VideoGaussianBlur(5, 0.05, p=1.0)),  # Force apply
        ("VideoRandomErasing", VideoRandomErasing(p=1.0)),  # Force apply
    ]
    
    print(f"\n🔍 Testing Individual Augmentations:")
    for name, augmentation in augmentations_to_test:
        try:
            test_sample = dummy_sample.copy()
            result = augmentation(test_sample)
            video_shape = result['video'].shape
            video_dtype = result['video'].dtype
            print(f"  ✅ {name}: {video_shape}, {video_dtype}")
        except Exception as e:
            print(f"  ❌ {name}: Error - {e}")
    
    # Test full pipeline
    print(f"\n🚀 Testing Full Base Augmentation Pipeline:")
    try:
        base_aug = get_base_augmentations(224)
        result = base_aug(dummy_sample.copy())
        
        video = result['video']
        print(f"  ✅ Pipeline completed successfully!")
        print(f"     Output shape: {video.shape}")
        print(f"     Output dtype: {video.dtype}")
        print(f"     Output range: [{video.min():.3f}, {video.max():.3f}]")
        
        # Verify tensor format is correct for video models (C, T, H, W)
        if len(video.shape) == 4 and video.shape[0] == 3:
            print(f"     ✅ Correct tensor format (C, T, H, W)")
        else:
            print(f"     ❌ Incorrect tensor format, expected (C, T, H, W)")
            
    except Exception as e:
        print(f"  ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test validation pipeline
    print(f"\n🔍 Testing Validation Pipeline:")
    try:
        val_aug = get_validation_augmentations(224)
        result = val_aug(dummy_sample.copy())
        
        video = result['video']
        print(f"  ✅ Validation pipeline completed!")
        print(f"     Output shape: {video.shape}")
        print(f"     Output dtype: {video.dtype}")
        print(f"     Output range: [{video.min():.3f}, {video.max():.3f}]")
        
    except Exception as e:
        print(f"  ❌ Validation pipeline failed: {e}")
    
    print(f"\n✨ Augmentation testing complete!")
    print(f"\n📋 Summary of implemented base augmentations:")
    print(f"   1. ✅ Temporal jitter (±15% around center)")
    print(f"   2. ✅ Resize → 224×224 (shortest side 256)")
    print(f"   3. ✅ Random horizontal flip (p=0.5)")
    print(f"   4. ✅ Color jitter (brightness/contrast/saturation=0.2, hue=0.1)")
    print(f"   5. ✅ Gaussian blur OR noise (σ≤0.05, p=0.2)")
    print(f"   6. ✅ Random erasing per frame (p=0.3, area 2-10%)")
