#!/usr/bin/env python3
"""
Test script to demonstrate how data batches look when feeding to the model.
Shows both raw dataset format and transformed format ready for training.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import MVFoulsDataset, decode_predictions, TASKS_INFO

# Try to import transforms
try:
    from transforms import get_train_transforms, get_val_transforms
    TRANSFORMS_AVAILABLE = True
except ImportError:
    print("⚠️  Transforms not available - showing raw format only")
    TRANSFORMS_AVAILABLE = False

def print_separator(title):
    """Print a nice separator with title."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_batch_info(batch_videos, batch_targets, title="BATCH INFO"):
    """Print detailed information about a batch."""
    print(f"\n📊 {title}")
    print(f"   Video batch shape: {batch_videos.shape}")
    print(f"   Video batch dtype: {batch_videos.dtype}")
    print(f"   Video memory usage: {batch_videos.element_size() * batch_videos.numel() / (1024**2):.1f} MB")
    
    if batch_videos.dtype == torch.uint8:
        print(f"   Video value range: [{batch_videos.min()}, {batch_videos.max()}]")
    else:
        print(f"   Video value range: [{batch_videos.min():.3f}, {batch_videos.max():.3f}]")
    
    print(f"   Targets batch shape: {batch_targets.shape}")
    print(f"   Targets dtype: {batch_targets.dtype}")
    
    # Show sample targets for first item in batch
    sample_targets = batch_targets[0]
    decoded = decode_predictions(sample_targets.unsqueeze(0))
    print(f"\n   📋 SAMPLE LABELS (first item in batch):")
    print(f"   Raw targets: {sample_targets.tolist()}")
    for i, (task_name, labels) in enumerate(decoded.items()):
        if i < 5:  # Show first 5 tasks
            print(f"   {task_name:15} → {labels[0]}")
        elif i == 5:
            print(f"   {'...':15}   (showing first 5 of {len(decoded)} tasks)")
            break

def main():
    print_separator("🎬 MVFOULS DATASET BATCH TESTING")
    
    # Dataset configuration
    batch_size = 4
    num_workers = 0  # Use 0 for Windows compatibility
    
    print(f"\n🔧 CONFIGURATION:")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of workers: {num_workers}")
    print(f"   Tasks: {len(TASKS_INFO)} multi-task labels")
    
    # Create datasets
    print(f"\n📁 Creating datasets...")
    try:
        train_dataset = MVFoulsDataset(
            root_dir='mvfouls',
            split='train',
            num_frames=32,
            center_frame=75,
            return_uint8=True,
            cache_mode="none"
        )
        print(f"   ✓ Train dataset: {len(train_dataset)} clips")
    except Exception as e:
        print(f"   ✗ Error creating dataset: {e}")
        return
    
    print_separator("RAW DATASET FORMAT (no transforms)")
    
    # Test raw format (no transforms)
    raw_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Get one batch
    try:
        raw_batch_videos, raw_batch_targets = next(iter(raw_dataloader))
        print_batch_info(raw_batch_videos, raw_batch_targets, "RAW FORMAT")
        
        print(f"\n📝 RAW FORMAT NOTES:")
        print(f"   • Video format: (Batch, Time, Height, Width, Channels)")
        print(f"   • Dtype: uint8 for memory efficiency (0-255 range)")
        print(f"   • Ready for transforms that expect (T,H,W,C) format")
        print(f"   • Memory efficient: 4x less than float32")
        
    except Exception as e:
        print(f"   ✗ Error loading raw batch: {e}")
        return
    
    # Test with transforms if available
    if TRANSFORMS_AVAILABLE:
        print_separator("TRANSFORMED FORMAT (ready for model)")
        
        try:
            # Create dataset with transforms
            train_transform = get_train_transforms(size=224)
            transformed_dataset = MVFoulsDataset(
                root_dir='mvfouls',
                split='train',
                transform=train_transform,
                num_frames=32,
                center_frame=75,
                return_uint8=True,  # Still return uint8, let transforms handle conversion
                cache_mode="none"
            )
            
            transformed_dataloader = DataLoader(
                transformed_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False
            )
            
            # Get transformed batch
            transformed_videos, transformed_targets = next(iter(transformed_dataloader))
            print_batch_info(transformed_videos, transformed_targets, "TRANSFORMED FORMAT")
            
            print(f"\n📝 TRANSFORMED FORMAT NOTES:")
            print(f"   • Video format: (Batch, Channels, Time, Height, Width)")
            print(f"   • Dtype: float32 normalized to [0.0, 1.0] or [-1.0, 1.0]")
            print(f"   • Ready for Video Swin Transformer or similar models")
            print(f"   • Includes data augmentation (if enabled in transforms)")
            
        except Exception as e:
            print(f"   ✗ Error with transforms: {e}")
            print(f"   💡 Make sure transforms.py exists and get_train_transforms() works")
    
    print_separator("DATALOADER ITERATION EXAMPLE")
    
    # Show how to iterate through batches
    print(f"\n🔄 Example training loop iteration:")
    print(f"""
    for batch_idx, (videos, targets) in enumerate(dataloader):
        # videos.shape = {raw_batch_videos.shape}
        # targets.shape = {raw_batch_targets.shape}
        
        # Move to device
        videos = videos.to(device)    # GPU/CPU
        targets = targets.to(device)  # GPU/CPU
        
        # Forward pass
        outputs = model(videos)       # Shape: (batch_size, num_tasks, num_classes)
        
        # Multi-task loss computation
        loss = compute_multi_task_loss(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
    """)
    
    print_separator("MEMORY & PERFORMANCE ANALYSIS")
    
    # Memory analysis
    video_memory_mb = raw_batch_videos.element_size() * raw_batch_videos.numel() / (1024**2)
    targets_memory_mb = raw_batch_targets.element_size() * raw_batch_targets.numel() / (1024**2)
    total_memory_mb = video_memory_mb + targets_memory_mb
    
    print(f"\n💾 MEMORY USAGE per batch:")
    print(f"   Videos:  {video_memory_mb:6.1f} MB")
    print(f"   Targets: {targets_memory_mb:6.1f} MB")
    print(f"   Total:   {total_memory_mb:6.1f} MB")
    
    # Estimate GPU memory for different batch sizes
    print(f"\n🎯 GPU MEMORY ESTIMATES (approximate):")
    for bs in [1, 2, 4, 8, 16]:
        estimated_mb = total_memory_mb * bs / batch_size
        if TRANSFORMS_AVAILABLE:
            estimated_mb *= 4  # float32 vs uint8
        print(f"   Batch size {bs:2d}: ~{estimated_mb:5.0f} MB")
    
    print_separator("DATASET STATISTICS")
    
    # Dataset info
    info = train_dataset.get_split_info()
    print(f"\n📈 DATASET INFO:")
    print(f"   Total clips: {info['total_clips']:,}")
    print(f"   Total actions: {info['total_actions']:,}")
    print(f"   Tasks: {info['num_tasks']}")
    print(f"   Frames per clip: {info['num_frames']}")
    print(f"   Center frame: {info['center_frame']}")
    print(f"   Cache mode: {info['cache_mode']}")
    print(f"   Return format: {'uint8' if info['return_uint8'] else 'float32'}")
    
    # Batches per epoch
    batches_per_epoch = len(train_dataset) // batch_size
    print(f"\n⚡ TRAINING ESTIMATES:")
    print(f"   Batches per epoch: {batches_per_epoch:,}")
    print(f"   Samples per epoch: {len(train_dataset):,}")
    print(f"   Data loading time per batch: ~0.1-1.0s (depends on decoder)")
    
    print_separator("READY FOR TRAINING! 🚀")
    
    print(f"""
✅ The dataset is ready for training with:
   • Multi-task learning ({len(TASKS_INFO)} tasks)
   • Efficient video loading (decord/PyAV/OpenCV)
   • Memory-optimized uint8 tensors
   • Configurable temporal windows
   • Robust data pipeline

🔥 Next steps:
   1. Create your model (Video Swin Transformer, etc.)
   2. Define multi-task loss function
   3. Set up training loop with this DataLoader
   4. Monitor per-task metrics during training
    """)

if __name__ == "__main__":
    main()
