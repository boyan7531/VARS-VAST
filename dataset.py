import torch
from torch.utils.data import Dataset
import json
import os
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import glob

# Import transforms (make sure this module is in the same directory)
try:
    from transforms import get_train_transforms, get_val_transforms, get_minimal_transforms
except ImportError:
    print("Warning: Could not import transforms. Make sure transforms.py is in the same directory.")
    get_train_transforms = get_val_transforms = get_minimal_transforms = None


class MVFoulsDataset(Dataset):
    """
    PyTorch Dataset for MVFouls video dataset.
    
    Dataset structure:
    - mvfouls/
      - train_720p/
        - annotations.json
        - action_0/ (clip_0.mp4, clip_1.mp4, ...)
        - action_1/ (clip_0.mp4, clip_1.mp4, ...)
        - ...
      - test_720p/
        - annotations.json
        - action_0/ (clip_0.mp4, clip_1.mp4, ...)
        - ...
      - valid_720p/
        - annotations.json
        - action_0/ (clip_0.mp4, clip_1.mp4, ...)
        - ...
      - challenge_720p/
        - action_0/ (clip_0.mp4, clip_1.mp4, ...)
        - ... (no annotations for challenge)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        load_annotations: bool = True,
        max_frames: Optional[int] = None,
        frame_rate: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None  # (height, width)
    ):
        """
        Args:
            root_dir (str): Path to mvfouls directory
            split (str): One of 'train', 'test', 'valid', 'challenge'
            transform: Optional transform to be applied on video frames
            load_annotations (bool): Whether to load annotations (False for challenge)
            max_frames (int): Maximum number of frames to load per video
            frame_rate (int): Target frame rate for video loading
            target_size (Tuple[int, int]): Optional target size for resizing frames
                NOTE: If using transforms that include VideoResize, set this to None
                to avoid double resizing. Use target_size for simple resizing without
                transforms, or when transforms don't include VideoResize.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.load_annotations = load_annotations and split != 'challenge'
        self.max_frames = max_frames
        self.frame_rate = frame_rate
        self.target_size = target_size
        
        # Set split directory
        self.split_dir = self.root_dir / f"{split}_720p"
        
        if not self.split_dir.exists():
            raise ValueError(f"Split directory {self.split_dir} does not exist")
        
        # Load annotations if available
        self.annotations = {}
        if self.load_annotations:
            self._load_annotations()
        
        # Get all action directories
        self.action_dirs = self._get_action_directories()
        
        # Build dataset index
        self.dataset_index = self._build_dataset_index()
        
    def _load_annotations(self):
        """Load annotations from JSON file."""
        annotations_path = self.split_dir / "annotations.json"
        if not annotations_path.exists():
            print(f"Warning: No annotations found at {annotations_path}")
            return
            
        try:
            with open(annotations_path, 'r') as f:
                data = json.load(f)
            self.annotations = data.get('Actions', {})
            print(f"Loaded {len(self.annotations)} annotations for {self.split} split")
        except Exception as e:
            print(f"Error loading annotations: {e}")
            self.annotations = {}
    
    def _get_action_directories(self) -> List[Path]:
        """Get all action directories in the split."""
        action_dirs = []
        for action_dir in self.split_dir.iterdir():
            if action_dir.is_dir() and action_dir.name.startswith('action_'):
                action_dirs.append(action_dir)
        
        # Sort by action number
        action_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
        return action_dirs
    
    def _build_dataset_index(self) -> List[Dict]:
        """Build index of all individual video clips in the dataset."""
        dataset_index = []
        
        for action_dir in self.action_dirs:
            action_id = action_dir.name.split('_')[1]
            
            # Get all video clips in this action directory
            video_files = list(action_dir.glob("*.mp4"))
            video_files.sort()  # Sort to ensure consistent ordering
            
            if not video_files:
                continue
                
            # Create a separate entry for each individual clip (always use all clips)
            for clip_path in video_files:
                clip_info = {
                    'action_id': action_id,
                    'action_dir': action_dir,
                    'clip_path': clip_path,
                    'clip_name': clip_path.name,
                    'annotations': self.annotations.get(action_id, {}) if self.load_annotations else {}
                }
                dataset_index.append(clip_info)
        
        return dataset_index
    
    def _safe_float_conversion(self, value) -> float:
        """Safely convert a value to float, handling empty strings and invalid values."""
        if value is None or value == '':
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert '{value}' to float, using 0.0")
            return 0.0
    
    def _load_video(self, video_path: Path) -> np.ndarray:
        """Load video from file and return as numpy array (frames 59-90 inclusive)."""
        cap = cv2.VideoCapture(str(video_path))
        
        # --- Get clip length first so we can adjust the frame window if needed ---
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Desired slice parameters
        desired_frames = 32
        default_start_frame = 59
        start_frame = default_start_frame
        end_frame = start_frame + desired_frames - 1

        # If the video is too short to reach our default [59-90] window, shift the
        # window backwards so that we still try to grab 32 frames ending at the
        # last available frame.
        if total_frames > 0 and end_frame >= total_frames:
            # New window end is the last frame in the clip
            end_frame = total_frames - 1
            # New window start makes a 32-frame segment (but not < 0)
            start_frame = max(0, end_frame - desired_frames + 1)
            print(f"Warning: Video {video_path} is too short, shifting window to {start_frame}-{end_frame}")

        # After shifting, we may still have less than 32 frames (very short clip).
        # In that case we will collect whatever frames exist and later pad by
        # repeating the last available frame so that the returned tensor always
        # has `desired_frames` frames.
        num_frames_to_read = end_frame - start_frame + 1

        # Seek to the adjusted start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []

        try:
            for i in range(num_frames_to_read):
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Could not read frame {start_frame + i} from {video_path}")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize to target_size if requested
                if self.target_size is not None:
                    # target_size given as (H, W) but cv2 uses (W, H)
                    h, w = self.target_size
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                frames.append(frame)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
        finally:
            cap.release()

        # If no frames could be read, return black frames with proper dimensions
        if not frames:
            print(f"Warning: No frames loaded from {video_path} (frames {start_frame}-{end_frame})")
            
            # Determine frame size for black frames
            if self.target_size is not None:
                h, w = self.target_size
            else:
                # Try to get frame size from video properties
                cap_temp = cv2.VideoCapture(str(video_path))
                frame_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH) or 224)
                frame_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT) or 224)
                cap_temp.release()
                h, w = frame_height, frame_width
                
                # If video properties are invalid, use default
                if h <= 0 or w <= 0:
                    h, w = 224, 224
                    print(f"Warning: Could not determine video dimensions, using default {h}x{w}")
            
            # Create black frame with determined dimensions
            black_frame = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Create 32 black frames
            frames = [black_frame.copy() for _ in range(desired_frames)]
            print(f"Warning: Using {desired_frames} black frames of size {h}x{w} for {video_path}")
            return np.stack(frames, axis=0)

        # Pad very short clips by repeating the last frame so that length == desired_frames
        while len(frames) < desired_frames:
            frames.append(frames[-1].copy())

        if len(frames) != desired_frames:
            print(f"Warning: Expected {desired_frames} frames but got {len(frames)} from {video_path}")

        return np.stack(frames, axis=0)  # Shape: (32, H, W, C)
    
    def __len__(self) -> int:
        """Return the total number of video clips in the dataset."""
        return len(self.dataset_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single video clip from the dataset."""
        if idx >= len(self.dataset_index):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.dataset_index)}")
        
        clip_info = self.dataset_index[idx]
        
        # Load the single video clip
        video = self._load_video(clip_info['clip_path'])
        
        # Prepare the sample
        sample = {
            'video': video,  # Single video array (32, H, W, C)
            'action_id': int(clip_info['action_id']),
            'clip_name': clip_info['clip_name'],
            'clip_path': str(clip_info['clip_path'])
        }
        
        # Add annotations if available
        if self.load_annotations and clip_info['annotations']:
            annotations = clip_info['annotations']
            sample.update({
                'offence': annotations.get('Offence', ''),
                'contact': annotations.get('Contact', ''),
                'bodypart': annotations.get('Bodypart', ''),
                'upper_body_part': annotations.get('Upper body part', ''),
                'action_class': annotations.get('Action class', ''),
                'severity': self._safe_float_conversion(annotations.get('Severity', 0.0)),
                'try_to_play': annotations.get('Try to play', ''),
                'touch_ball': annotations.get('Touch ball', ''),
                'handball': annotations.get('Handball', ''),
                'url_local': annotations.get('UrlLocal', ''),
                'clips_info': annotations.get('Clips', [])
            })
        
        # Apply transforms if specified
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_action_ids(self) -> List[int]:
        """Get list of all unique action IDs in the dataset."""
        return sorted(list(set(int(item['action_id']) for item in self.dataset_index)))
    
    def get_action_classes(self) -> List[str]:
        """Get list of all unique action classes in the dataset."""
        if not self.load_annotations:
            return []
        
        classes = set()
        for item in self.dataset_index:
            if item['annotations']:
                action_class = item['annotations'].get('Action class', '')
                if action_class:
                    classes.add(action_class)
        return sorted(list(classes))
    
    def get_split_info(self) -> Dict:
        """Get information about the dataset split."""
        # Count unique actions
        unique_actions = len(set(int(item['action_id']) for item in self.dataset_index))
        
        info = {
            'split': self.split,
            'total_clips': len(self.dataset_index),
            'total_actions': unique_actions,
            'has_annotations': self.load_annotations
        }
        
        if self.load_annotations:
            info['action_classes'] = self.get_action_classes()
            info['num_classes'] = len(info['action_classes'])
        
        return info


def create_mvfouls_datasets(
    root_dir: str,
    splits: List[str] = ['train', 'test', 'valid'],
    **kwargs
) -> Dict[str, MVFoulsDataset]:
    """
    Create MVFouls datasets for multiple splits.
    
    Args:
        root_dir (str): Path to mvfouls directory
        splits (List[str]): List of splits to create datasets for
        **kwargs: Additional arguments passed to MVFoulsDataset
    
    Returns:
        Dict[str, MVFoulsDataset]: Dictionary mapping split names to datasets
    """
    datasets = {}
    
    for split in splits:
        try:
            dataset = MVFoulsDataset(root_dir, split=split, **kwargs)
            datasets[split] = dataset
            info = dataset.get_split_info()
            print(f"Created {split} dataset with {len(dataset)} clips ({info['total_actions']} actions)")
        except Exception as e:
            print(f"Error creating {split} dataset: {e}")
    
    return datasets


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    root_dir = "mvfouls"
    
    # Create datasets WITHOUT transforms first to show raw data
    datasets = create_mvfouls_datasets(root_dir, splits=['train', 'test', 'valid'])
    
    # Print dataset information
    for split, dataset in datasets.items():
        info = dataset.get_split_info()
        print(f"\n{split.upper()} Dataset:")
        print(f"  Total clips: {info['total_clips']}")
        print(f"  Total actions: {info['total_actions']}")
        print(f"  Has annotations: {info['has_annotations']}")
        if info['has_annotations']:
            print(f"  Number of action classes: {info['num_classes']}")
            print(f"  Action classes: {info['action_classes'][:5]}...")  # Show first 5
    
    # Example: Load a single sample from train dataset (without transforms)
    if 'train' in datasets:
        train_dataset = datasets['train']
        if len(train_dataset) > 0:
            sample = train_dataset[2]
            print(f"\nSample from train dataset (raw data):")
            print(f"  Action ID: {sample['action_id']}")
            print(f"  Video shape: {sample['video'].shape}")
            print(f"  Video dtype: {sample['video'].dtype}")
            print(f"  Video range: [{sample['video'].min()}, {sample['video'].max()}]")
            print(f"  Clip name: {sample['clip_name']}")
            if 'action_class' in sample:
                print(f"  Action class: {sample['action_class']}")
                print(f"  Severity: {sample['severity']}")
    
    # Example: Using datasets WITH transforms
    if get_train_transforms and get_val_transforms:
        print(f"\n{'='*50}")
        print("CREATING DATASETS WITH TRANSFORMS")
        print(f"{'='*50}")
        
        # Create train dataset with training transforms (augmentation)
        train_transform = get_train_transforms(size=224)
        train_dataset_transformed = MVFoulsDataset(
            root_dir=root_dir,
            split='train',
            transform=train_transform,
            target_size=None  # Let transforms handle resizing to avoid double resizing
        )
        
        # Create validation dataset with validation transforms (no augmentation)
        val_transform = get_val_transforms(size=224)
        val_dataset_transformed = MVFoulsDataset(
            root_dir=root_dir,
            split='valid',
            transform=val_transform,
            target_size=None  # Let transforms handle resizing to avoid double resizing
        )
        
        print(f"\nTrain dataset with transforms: {len(train_dataset_transformed)} clips")
        print(f"Validation dataset with transforms: {len(val_dataset_transformed)} clips")
        
        # Test the transforms
        if len(train_dataset_transformed) > 0:
            sample_transformed = train_dataset_transformed[0]
            print(f"\nTransformed sample (training):")
            print(f"  Video shape: {sample_transformed['video'].shape}")
            print(f"  Video dtype: {sample_transformed['video'].dtype}")
            print(f"  Video range: [{sample_transformed['video'].min():.3f}, {sample_transformed['video'].max():.3f}]")
            print(f"  Expected shape for models: (C=3, T=32, H=224, W=224)")
            print(f"  Ready for Video Swin Transformer: {'✓' if sample_transformed['video'].shape == torch.Size([3, 32, 224, 224]) else '✗'}")
        
        if len(val_dataset_transformed) > 0:
            sample_val = val_dataset_transformed[0]
            print(f"\nTransformed sample (validation):")
            print(f"  Video shape: {sample_val['video'].shape}")
            print(f"  Video dtype: {sample_val['video'].dtype}")
            print(f"  Video range: [{sample_val['video'].min():.3f}, {sample_val['video'].max():.3f}]")
    
    else:
        print(f"\nTransforms not available. Make sure transforms.py is in the same directory.") 