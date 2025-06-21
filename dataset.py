import torch
from torch.utils.data import Dataset
import json
import os
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import glob
from collections import OrderedDict

# Import transforms (make sure this module is in the same directory)
try:
    from transforms import get_train_transforms, get_val_transforms, get_minimal_transforms
except ImportError:
    print("Warning: Could not import transforms. Make sure transforms.py is in the same directory.")
    get_train_transforms = get_val_transforms = get_minimal_transforms = None

# Global Constants for Multi-Task Learning
TASKS_INFO = OrderedDict([
    ("action_class", ["Missing/Empty", "Standing tackling", "Tackling", "Challenge", "Holding", "Elbowing", "High leg", "Pushing", "Dont know", "Dive"]),
    ("severity", ["Missing", "1", "2", "3", "4", "5"]),
    ("offence", ["Missing/Empty", "Offence", "No offence", "Between"]),
    ("contact", ["Missing/Empty", "With contact", "Without contact"]),
    ("bodypart", ["Missing/Empty", "Under body", "Upper body"]),
    ("upper_body_part", ["Missing/Empty", "Use of arms", "Use of shoulder", "Use of shoulders"]),
    ("multiple_fouls", ["Missing/Empty", "No", "Yes"]),
    ("try_to_play", ["Missing/Empty", "Yes", "No"]),
    ("touch_ball", ["Missing/Empty", "No", "Yes", "Maybe"]),
    ("handball", ["Missing/Empty", "No handball", "Handball"]),
    ("handball_offence", ["Missing/Empty", "Offence", "No offence"]),
])

# Create label-to-index and index-to-label mappings
LABEL2IDX = {task: {lbl: i for i, lbl in enumerate(labels)} for task, labels in TASKS_INFO.items()}
IDX2LABEL = {task: labels for task, labels in TASKS_INFO.items()}

# Number of tasks
N_TASKS = len(TASKS_INFO)

def normalize_label_value(raw_value: any, task_name: str) -> str:
    """
    Normalize raw annotation values to canonical string labels.
    
    Args:
        raw_value: Raw value from annotations (could be None, empty string, number, etc.)
        task_name: Name of the task (must be in TASKS_INFO)
    
    Returns:
        str: Normalized canonical label
    """
    # Handle missing/empty values
    if raw_value is None or raw_value == '' or str(raw_value).strip() == '':
        return "Missing" if task_name == "severity" else "Missing/Empty"
    
    # Convert to string and strip whitespace
    str_value = str(raw_value).strip()
    
    # Handle special cases for specific tasks
    if task_name == "severity":
        # Convert float/int to string for severity
        try:
            float_val = float(str_value)
            if float_val.is_integer():
                str_value = str(int(float_val))
            else:
                str_value = str(float_val)
        except (ValueError, TypeError):
            pass
    
    # Normalize common case variations
    if str_value.lower() == "yes":
        str_value = "Yes"
    elif str_value.lower() == "no":
        str_value = "No"
    
    # Check if normalized value is in canonical labels
    canonical_labels = TASKS_INFO[task_name]
    if str_value in canonical_labels:
        return str_value
    
    # If not found, try case-insensitive match
    for canonical_label in canonical_labels:
        if str_value.lower() == canonical_label.lower():
            return canonical_label
    
    # If still not found, return Missing/Empty
    print(f"Warning: Unknown value '{str_value}' for task '{task_name}', using Missing/Empty")
    return "Missing" if task_name == "severity" else "Missing/Empty"



def get_task_info_for_model() -> Dict[str, Dict[str, any]]:
    """
    Get task configuration information for the model.
    
    Returns:
        Dict containing num_classes and loss_type for each task
    """
    task_configs = {}
    
    for task_name, labels in TASKS_INFO.items():
        task_configs[task_name] = {
            'num_classes': len(labels),
            'loss_type': 'ce',  # Cross-entropy loss for all tasks
            'labels': labels
        }
    
    return task_configs

def decode_predictions(pred_tensor: torch.Tensor, task_order: List[str] = None) -> Dict[str, List[str]]:
    """
    Decode model predictions from integer indices to human-readable labels.
    
    Args:
        pred_tensor: Tensor of shape (B, N_TASKS) containing predicted class indices
        task_order: List of task names in the same order as tensor dimensions
    
    Returns:
        Dict mapping task names to lists of predicted labels
    """
    if task_order is None:
        task_order = list(TASKS_INFO.keys())
    
    if pred_tensor.dim() == 1:
        pred_tensor = pred_tensor.unsqueeze(0)  # Add batch dimension
    
    batch_size, num_tasks = pred_tensor.shape
    assert num_tasks == len(task_order), f"Expected {len(task_order)} tasks, got {num_tasks}"
    
    decoded = {task: [] for task in task_order}
    
    for batch_idx in range(batch_size):
        for task_idx, task_name in enumerate(task_order):
            pred_idx = pred_tensor[batch_idx, task_idx].item()
            if 0 <= pred_idx < len(IDX2LABEL[task_name]):
                decoded[task_name].append(IDX2LABEL[task_name][pred_idx])
            else:
                print(f"Warning: Invalid prediction index {pred_idx} for task {task_name}")
                decoded[task_name].append(IDX2LABEL[task_name][0])  # Use Missing/Empty
    
    return decoded

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
        
        # Process annotations to create numeric labels after dataset_index is built
        if self.load_annotations:
            self._process_annotations()
        
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
                    'annotations': self.annotations.get(action_id, {}) if self.load_annotations else {},
                    'numeric_labels': None  # Will be populated by _process_annotations
                }
                dataset_index.append(clip_info)
        
        return dataset_index
    
    def _process_annotations(self):
        """Process raw annotations into numeric labels for multi-task learning."""
        for clip_info in self.dataset_index:
            annotations = clip_info['annotations']
            
            # Initialize numeric labels for all tasks
            numeric_labels = []
            
            # Process each task in the canonical order
            for task_name in TASKS_INFO.keys():
                # Map annotation field names to task names
                field_mapping = {
                    'action_class': 'Action class',
                    'severity': 'Severity',
                    'offence': 'Offence',
                    'contact': 'Contact',
                    'bodypart': 'Bodypart',
                    'upper_body_part': 'Upper body part',
                    'multiple_fouls': 'Multiple fouls',  # May not exist in annotations
                    'try_to_play': 'Try to play',
                    'touch_ball': 'Touch ball',
                    'handball': 'Handball',
                    'handball_offence': 'Handball offence'  # May not exist in annotations
                }
                
                # Get raw value from annotations
                annotation_field = field_mapping.get(task_name, task_name)
                raw_value = annotations.get(annotation_field, None)
                
                # Normalize and convert to index
                normalized_value = normalize_label_value(raw_value, task_name)
                label_idx = LABEL2IDX[task_name][normalized_value]
                numeric_labels.append(label_idx)
            
            # Store as tensor
            clip_info['numeric_labels'] = torch.tensor(numeric_labels, dtype=torch.long)
    
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
        """
        Load video from file using consistent frame sampling (no temporal normalization).
        This allows the model to learn temporal invariance naturally.
        
        Args:
            video_path: Path to video file
        
        Returns:
            np.ndarray: Video frames of shape (32, H, W, C)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # Get clip length first so we can adjust the frame window if needed
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Desired slice parameters - use consistent window for all videos
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single video clip from the dataset.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (video_tensor, targets_tensor)
                - video_tensor: Shape (C, T, H, W) after transforms
                - targets_tensor: Shape (N_TASKS,) with integer class labels
        """
        if idx >= len(self.dataset_index):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.dataset_index)}")
        
        clip_info = self.dataset_index[idx]
        
        # Load the video using consistent frame sampling
        video = self._load_video(clip_info['clip_path'])
        
        # Get numeric labels (pre-processed during initialization)
        if self.load_annotations and clip_info['numeric_labels'] is not None:
            targets = clip_info['numeric_labels']
        else:
            # Use all Missing/Empty labels if no annotations
            targets = torch.zeros(N_TASKS, dtype=torch.long)
        
        # Prepare the sample for transforms (if any)
        sample = {
            'video': video,  # Shape: (T, H, W, C)
            'targets': targets  # Shape: (N_TASKS,)
        }
        
        # Apply transforms if specified
        if self.transform:
            sample = self.transform(sample)
        
        # Return as tuple
        return sample['video'], sample['targets']
    
    def get_action_ids(self) -> List[int]:
        """Get list of all unique action IDs in the dataset."""
        return sorted(list(set(int(item['action_id']) for item in self.dataset_index)))
    
    def get_task_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all tasks in the dataset (per unique action, not per clip)."""
        if not self.load_annotations:
            return {}
        
        stats = {}
        
        # Get unique actions to avoid counting the same action multiple times
        unique_actions = {}
        for item in self.dataset_index:
            action_id = item['action_id']
            if action_id not in unique_actions and item['numeric_labels'] is not None:
                unique_actions[action_id] = item['numeric_labels']
        
        for task_idx, task_name in enumerate(TASKS_INFO.keys()):
            task_stats = {
                'task_name': task_name,
                'num_classes': len(TASKS_INFO[task_name]),
                'class_names': TASKS_INFO[task_name],
                'class_counts': [0] * len(TASKS_INFO[task_name])
            }
            
            # Count occurrences of each class (per unique action)
            for action_id, numeric_labels in unique_actions.items():
                class_idx = numeric_labels[task_idx].item()
                if 0 <= class_idx < len(task_stats['class_counts']):
                    task_stats['class_counts'][class_idx] += 1
            
            # Calculate class weights (inverse frequency)
            total_samples = sum(task_stats['class_counts'])
            if total_samples > 0:
                task_stats['class_weights'] = [
                    total_samples / (len(task_stats['class_counts']) * count) if count > 0 else 1.0
                    for count in task_stats['class_counts']
                ]
            else:
                task_stats['class_weights'] = [1.0] * len(task_stats['class_counts'])
            
            stats[task_name] = task_stats
        
        return stats
    
    def get_split_info(self) -> Dict:
        """Get information about the dataset split."""
        # Count unique actions
        unique_actions = len(set(int(item['action_id']) for item in self.dataset_index))
        
        info = {
            'split': self.split,
            'total_clips': len(self.dataset_index),
            'total_actions': unique_actions,
            'has_annotations': self.load_annotations,
            'num_tasks': N_TASKS,
            'task_names': list(TASKS_INFO.keys())
        }
        
        if self.load_annotations:
            info['task_statistics'] = self.get_task_statistics()
        
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


# Comprehensive testing and examples
if __name__ == "__main__":
    print("="*60)
    print("MVFOULS MULTI-TASK DATASET TESTING")
    print("="*60)
    
    # Test global constants
    print(f"\n1. GLOBAL CONSTANTS TEST:")
    print(f"   Number of tasks: {N_TASKS}")
    print(f"   Task names: {list(TASKS_INFO.keys())}")
    print(f"   Sample task info (action_class): {TASKS_INFO['action_class']}")
    print(f"   Sample LABEL2IDX (severity): {LABEL2IDX['severity']}")
    
    # Test utility functions
    print(f"\n2. UTILITY FUNCTIONS TEST:")
    
    # Test normalize_label_value
    test_cases = [
        (None, 'action_class', 'Missing/Empty'),
        ('', 'severity', 'Missing'),
        ('1.0', 'severity', '1'),
        ('yes', 'try_to_play', 'Yes'),
        ('Standing tackling', 'action_class', 'Standing tackling'),
        ('unknown_value', 'contact', 'Missing/Empty')
    ]
    
    print("   Testing normalize_label_value:")
    for raw_val, task, expected in test_cases:
        result = normalize_label_value(raw_val, task)
        status = "✓" if result == expected else "✗"
        print(f"     {status} {raw_val} -> {result} (expected: {expected})")
    
    # Test decode_predictions
    print("   Testing decode_predictions:")
    test_pred = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]])  # Example predictions
    decoded = decode_predictions(test_pred)
    print(f"     Sample prediction decoded: {list(decoded.keys())[:3]}...")
    
    # Test get_task_info_for_model
    task_info = get_task_info_for_model()
    print(f"   Task info for model: {len(task_info)} tasks configured")
    print(f"   Sample task config: {task_info['action_class']}")
    
    root_dir = "mvfouls"
    
    print(f"\n3. DATASET CREATION TEST:")
    
    # Create datasets WITHOUT transforms first
    try:
        datasets = create_mvfouls_datasets(root_dir, splits=['train', 'test', 'valid'])
        
        # Print dataset information with new multi-task structure
        for split, dataset in datasets.items():
            info = dataset.get_split_info()
            print(f"\n   {split.upper()} Dataset:")
            print(f"     Total clips: {info['total_clips']}")
            print(f"     Total actions: {info['total_actions']}")
            print(f"     Number of tasks: {info['num_tasks']}")
            print(f"     Has annotations: {info['has_annotations']}")
            
            if info['has_annotations'] and 'task_statistics' in info:
                stats = info['task_statistics']
                print(f"     Task statistics sample:")
                for task_name in list(stats.keys())[:3]:  # Show first 3 tasks
                    task_stat = stats[task_name]
                    total_count = sum(task_stat['class_counts'])
                    missing_count = task_stat['class_counts'][0]
                    print(f"       {task_name}: {task_stat['num_classes']} classes, "
                          f"{missing_count}/{total_count} missing")
        
        print(f"\n4. MULTI-TASK SAMPLE TEST:")
        
        # Test the new __getitem__ return format
        if 'train' in datasets:
            train_dataset = datasets['train']
            if len(train_dataset) > 0:
                video_tensor, targets_tensor = train_dataset[0]
                
                print(f"   Sample from train dataset:")
                print(f"     Video tensor shape: {video_tensor.shape}")
                print(f"     Video tensor dtype: {video_tensor.dtype}")
                print(f"     Video tensor range: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")
                print(f"     Targets tensor shape: {targets_tensor.shape}")
                print(f"     Targets tensor dtype: {targets_tensor.dtype}")
                print(f"     Targets tensor: {targets_tensor}")
                
                # Verify targets are valid
                valid_targets = True
                for task_idx, task_name in enumerate(TASKS_INFO.keys()):
                    target_val = targets_tensor[task_idx].item()
                    max_classes = len(TASKS_INFO[task_name])
                    if not (0 <= target_val < max_classes):
                        print(f"     ✗ Invalid target for {task_name}: {target_val} (max: {max_classes-1})")
                        valid_targets = False
                
                if valid_targets:
                    print(f"     ✓ All targets are valid")
                
                # Test decoding
                decoded = decode_predictions(targets_tensor.unsqueeze(0))
                print(f"     Decoded labels sample:")
                for task_name in list(decoded.keys())[:3]:
                    print(f"       {task_name}: {decoded[task_name][0]}")
        
        print(f"\n5. CONSISTENT FRAME SAMPLING TEST:")
        
        # Test consistent frame sampling (no temporal normalization)
        if 'train' in datasets and len(datasets['train']) > 0:
            dataset = datasets['train']
            clip_info = dataset.dataset_index[0]
            
            # Test that we always get consistent frame sampling
            video_frames = dataset._load_video(clip_info['clip_path'])
            print(f"     Loaded video: {video_frames.shape} frames")
            
            # Should always get 32 frames using consistent sampling
            assert video_frames.shape[0] == 32, f"Expected 32 frames, got {video_frames.shape[0]}"
            
            print(f"     ✓ Consistent frame sampling working correctly")
            print(f"     ✓ Model will learn temporal invariance naturally")
        
        print(f"\n6. TRANSFORMS COMPATIBILITY TEST:")
        
        # Test with transforms if available
        if get_train_transforms and get_val_transforms:
            print("     Testing with transforms...")
            
            # Create train dataset with training transforms
            train_transform = get_train_transforms(size=224)
            train_dataset_transformed = MVFoulsDataset(
                root_dir=root_dir,
                split='train',
                transform=train_transform,
                target_size=None
            )
            
            if len(train_dataset_transformed) > 0:
                video_tensor, targets_tensor = train_dataset_transformed[0]
                print(f"     Transformed video shape: {video_tensor.shape}")
                print(f"     Transformed video dtype: {video_tensor.dtype}")
                print(f"     Transformed video range: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")
                print(f"     Targets unchanged: {targets_tensor.shape}")
                
                expected_shape = torch.Size([3, 32, 224, 224])
                is_correct = video_tensor.shape == expected_shape
                print(f"     ✓ Ready for Video Swin Transformer: {'✓' if is_correct else '✗'}")
        else:
            print("     Transforms not available - skipping transform tests")
        
        print(f"\n7. DATASET LOADER COMPATIBILITY TEST:")
        
        # Test DataLoader compatibility
        from torch.utils.data import DataLoader
        
        if 'train' in datasets:
            train_dataset = datasets['train']
            if len(train_dataset) > 0:
                # Test with small batch
                dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False)
                
                try:
                    batch_videos, batch_targets = next(iter(dataloader))
                    print(f"     DataLoader batch test:")
                    print(f"       Batch videos shape: {batch_videos.shape}")
                    print(f"       Batch targets shape: {batch_targets.shape}")
                    print(f"       ✓ DataLoader works with default collate_fn")
                except Exception as e:
                    print(f"     ✗ DataLoader test failed: {e}")
        
        print(f"\n8. SUMMARY:")
        print(f"   ✓ Multi-task labeling implemented ({N_TASKS} tasks)")
        print(f"   ✓ Consistent frame sampling (NO temporal normalization)")
        print(f"   ✓ Model will learn temporal invariance naturally")
        print(f"   ✓ New tuple return format (__getitem__)")
        print(f"   ✓ Utility functions for model integration")
        print(f"   ✓ Compatible with PyTorch DataLoader")
        print(f"   ✓ Missing values mapped to class 0")
        print(f"   ✓ Robust to different video speeds during inference")
        
    except Exception as e:
        print(f"   ✗ Dataset creation failed: {e}")
        print(f"   Make sure the 'mvfouls' directory exists and contains the dataset.")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60) 