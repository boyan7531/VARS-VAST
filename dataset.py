import torch
from torch.utils.data import Dataset
import json
import os
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import glob


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
        clip_selection: str = 'all',  # 'all', 'first', 'last', or int index
        transform=None,
        load_annotations: bool = True,
        max_frames: Optional[int] = None,
        frame_rate: Optional[int] = None
    ):
        """
        Args:
            root_dir (str): Path to mvfouls directory
            split (str): One of 'train', 'test', 'valid', 'challenge'
            clip_selection (str or int): How to select clips from each action
                - 'all': return all clips for each action
                - 'first': return only the first clip
                - 'last': return only the last clip
                - int: return clip at specific index
            transform: Optional transform to be applied on video frames
            load_annotations (bool): Whether to load annotations (False for challenge)
            max_frames (int): Maximum number of frames to load per video
            frame_rate (int): Target frame rate for video loading
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.clip_selection = clip_selection
        self.transform = transform
        self.load_annotations = load_annotations and split != 'challenge'
        self.max_frames = max_frames
        self.frame_rate = frame_rate
        
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
        """Build index of all actions in the dataset (grouped by action)."""
        dataset_index = []
        
        for action_dir in self.action_dirs:
            action_id = action_dir.name.split('_')[1]
            
            # Get all video clips in this action directory
            video_files = list(action_dir.glob("*.mp4"))
            video_files.sort()  # Sort to ensure consistent ordering
            
            if not video_files:
                continue
                
            # Apply clip selection to determine which clips to include
            selected_clips = self._select_clips(video_files)
            
            if selected_clips:  # Only add if we have clips after selection
                action_info = {
                    'action_id': action_id,
                    'action_dir': action_dir,
                    'clip_paths': selected_clips,  # All selected clips for this action
                    'clip_names': [clip.name for clip in selected_clips],
                    'annotations': self.annotations.get(action_id, {}) if self.load_annotations else {}
                }
                dataset_index.append(action_info)
        
        return dataset_index
    
    def _select_clips(self, video_files: List[Path]) -> List[Path]:
        """Select clips based on clip_selection parameter."""
        if not video_files:
            return []
            
        if self.clip_selection == 'all':
            return video_files
        elif self.clip_selection == 'first':
            return [video_files[0]]
        elif self.clip_selection == 'last':
            return [video_files[-1]]
        elif isinstance(self.clip_selection, int):
            idx = self.clip_selection
            if 0 <= idx < len(video_files):
                return [video_files[idx]]
            else:
                return []
        else:
            raise ValueError(f"Invalid clip_selection: {self.clip_selection}")
    
    def _load_video(self, video_path: Path) -> np.ndarray:
        """Load video from file and return as numpy array."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
                # Limit number of frames if specified
                if self.max_frames and len(frames) >= self.max_frames:
                    break
                    
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            
        finally:
            cap.release()
        
        if not frames:
            print(f"Warning: No frames loaded from {video_path}")
            return np.array([])
            
        return np.array(frames)  # Shape: (T, H, W, C)
    
    def __len__(self) -> int:
        """Return the total number of actions in the dataset."""
        return len(self.dataset_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single action from the dataset (with all its clips)."""
        if idx >= len(self.dataset_index):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.dataset_index)}")
        
        action_info = self.dataset_index[idx]
        
        # Load all videos for this action
        videos = []
        for clip_path in action_info['clip_paths']:
            video = self._load_video(clip_path)
            videos.append(video)
        
        # Prepare the sample
        sample = {
            'videos': videos,  # List of video arrays
            'action_id': int(action_info['action_id']),
            'clip_names': action_info['clip_names'],  # List of clip names
            'clip_paths': [str(path) for path in action_info['clip_paths']],  # List of clip paths
            'num_clips': len(videos)
        }
        
        # Add annotations if available
        if self.load_annotations and action_info['annotations']:
            annotations = action_info['annotations']
            sample.update({
                'offence': annotations.get('Offence', ''),
                'contact': annotations.get('Contact', ''),
                'bodypart': annotations.get('Bodypart', ''),
                'upper_body_part': annotations.get('Upper body part', ''),
                'action_class': annotations.get('Action class', ''),
                'severity': float(annotations.get('Severity', 0.0)),
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
        total_clips = sum(len(item['clip_paths']) for item in self.dataset_index)
        
        info = {
            'split': self.split,
            'total_actions': len(self.dataset_index),
            'total_clips': total_clips,
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
            print(f"Created {split} dataset with {len(dataset)} actions ({info['total_clips']} clips)")
        except Exception as e:
            print(f"Error creating {split} dataset: {e}")
    
    return datasets


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    root_dir = "mvfouls"
    
    # Create datasets for all splits
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
    
    # Example: Load a single sample from train dataset
    if 'train' in datasets:
        train_dataset = datasets['train']
        if len(train_dataset) > 0:
            sample = train_dataset[2]
            print(f"\nSample from train dataset:")
            print(f"  Action ID: {sample['action_id']}")
            print(f"  Number of clips: {sample['num_clips']}")
            print(f"  Video shapes: {[video.shape for video in sample['videos']]}")
            print(f"  Clip names: {sample['clip_names']}")
            if 'action_class' in sample:
                print(f"  Action class: {sample['action_class']}")
                print(f"  Severity: {sample['severity']}") 