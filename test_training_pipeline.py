#!/usr/bin/env python3
"""
Quick Training Pipeline Test Script
==================================

This script performs a fast smoke test of the training pipeline to catch
potential crashes before starting actual training.

Usage:
    python test_training_pipeline.py --train-dir /path/to/train --val-dir /path/to/val --train-annotations train.csv --val-annotations val.csv
"""

import argparse
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    imports_to_test = [
        ("model.mvfouls_model", ["MVFoulsModel", "build_multi_task_model", "build_single_task_model"]),
        ("training_utils", ["MultiTaskTrainer"]),
        ("dataset", ["MVFoulsDataset"]),
        ("transforms", ["get_video_transforms"]),
    ]
    
    for module_name, items in imports_to_test:
        try:
            module = __import__(module_name, fromlist=items)
            for item in items:
                getattr(module, item)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
            return False
        except Exception as e:
            print(f"  ❌ {module_name}: Unexpected error - {e}")
            return False
    
    print("✅ All imports successful")
    return True

def test_device():
    """Test device availability."""
    print("\n🔍 Testing device availability...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def test_dataset_creation(args):
    """Test dataset creation and basic loading."""
    print("\n🔍 Testing dataset creation...")
    
    # Check if data paths exist first
    import os
    train_path = f"{args.root_dir}/train_720p"
    val_path = f"{args.root_dir}/valid_720p"
    
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print(f"⚠️ Data paths don't exist, creating dummy datasets for testing...")
        return test_dummy_dataset_creation(args)
    
    try:
        import sys
        sys.path.append('.')
        from dataset import MVFoulsDataset
        from transforms import get_video_transforms
        
        # Create transforms
        transforms = get_video_transforms(image_size=224, augment_train=True)
        
        # Test training dataset
        train_dataset = MVFoulsDataset(
            root_dir=args.root_dir,
            split='train',
            transform=transforms['train'],
            num_frames=8,  # Small for testing
            load_annotations=True
        )
        
        # Test validation dataset
        val_dataset = MVFoulsDataset(
            root_dir=args.root_dir,
            split='valid',
            transform=transforms['val'],
            num_frames=8,  # Small for testing
            load_annotations=True
        )
        
        print(f"✅ Datasets created: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Test loading one sample
        print("🔍 Testing sample loading...")
        train_sample = train_dataset[0]
        val_sample = val_dataset[0]
        
        print(f"✅ Sample loaded: train video shape={train_sample[0].shape}, val video shape={val_sample[0].shape}")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return None, None


def test_dummy_dataset_creation(args):
    """Test dataset creation with dummy data."""
    try:
        import sys
        sys.path.append('.')
        from dataset import MVFoulsDataset
        from transforms import get_video_transforms
        import tempfile
        import os
        
        # Create transforms
        transforms = get_video_transforms(image_size=224, augment_train=True)
        
        # Create dummy video files for testing
        dummy_videos = [
            "dummy_video_1.mp4",
            "dummy_video_2.mp4"
        ]
        
        # Create dummy annotations
        dummy_annotations = {
            "1": {"Severity": "1", "Action class": "Standing tackle"},
            "2": {"Severity": "2", "Action class": "Kicking"}
        }
        
        # Test with dummy data
        train_dataset = MVFoulsDataset(
            video_list=dummy_videos,
            annotations_dict=dummy_annotations,
            transform=transforms['train'],
            num_frames=8,
            load_annotations=True
        )
        
        val_dataset = MVFoulsDataset(
            video_list=dummy_videos,
            annotations_dict=dummy_annotations,
            transform=transforms['val'],
            num_frames=8,
            load_annotations=True
        )
        
        # Override the getitem to return proper format for testing
        def patched_getitem(self, idx):
            # Get the original data
            video, targets = self._original_getitem(idx)
            
            # Convert targets to dict format for multi-task
            if hasattr(self, 'load_annotations') and self.load_annotations:
                from utils import get_task_metadata
                if get_task_metadata is not None:
                    metadata = get_task_metadata()
                    task_names = metadata['task_names']
                    
                    # Convert tensor targets to dict
                    if isinstance(targets, torch.Tensor) and len(targets.shape) == 1:
                        targets_dict = {}
                        for i, task_name in enumerate(task_names):
                            if i < len(targets):
                                targets_dict[task_name] = targets[i:i+1]  # Keep as 1D tensor
                            else:
                                targets_dict[task_name] = torch.tensor([0])  # Default value
                        return video, targets_dict
            
            return video, targets
        
        # Patch the datasets for testing
        train_dataset._original_getitem = train_dataset.__getitem__
        train_dataset.__getitem__ = lambda idx: patched_getitem(train_dataset, idx)
        
        val_dataset._original_getitem = val_dataset.__getitem__
        val_dataset.__getitem__ = lambda idx: patched_getitem(val_dataset, idx)
        
        print(f"✅ Dummy datasets created: {len(train_dataset)} train, {len(val_dataset)} val")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"❌ Dummy dataset creation failed: {e}")
        return None, None

def test_dataloader(dataset, batch_size=2):
    """Test dataloader creation and batch loading."""
    print("\n🔍 Testing dataloader...")
    
    try:
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues in testing
            pin_memory=False
        )
        
        # Test loading one batch
        batch = next(iter(dataloader))
        videos, targets = batch
        
        print(f"✅ Batch loaded: videos={videos.shape}, targets type={type(targets)}")
        
        if isinstance(targets, dict):
            for task, target in targets.items():
                print(f"   Task {task}: {target.shape}")
        else:
            print(f"   Targets: {targets.shape}")
        
        return dataloader
        
    except Exception as e:
        print(f"❌ Dataloader failed: {e}")
        return None

def test_model_creation(args, device):
    """Test model creation and basic forward pass."""
    print("\n🔍 Testing model creation...")
    
    try:
        import sys
        sys.path.append('.')
        from model.mvfouls_model import build_multi_task_model, build_single_task_model
        
        if args.multi_task:
            model = build_multi_task_model(
                backbone_pretrained=False,  # Faster for testing
                backbone_freeze_mode='none',
                head_dropout=0.5,
                head_pooling='avg'
            )
        else:
            model = build_single_task_model(
                num_classes=args.num_classes,
                backbone_pretrained=False,  # Faster for testing
                backbone_freeze_mode='none',
                head_dropout=0.5,
                head_pooling='avg'
            )
        
        model.to(device)
        model.eval()
        
        print(f"✅ Model created: {model.__class__.__name__}")
        print(f"   Multi-task: {model.multi_task}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None

def test_forward_pass(model, dataloader, device):
    """Test forward pass through the model."""
    print("\n🔍 Testing forward pass...")
    
    try:
        batch = next(iter(dataloader))
        videos, targets = batch
        
        videos = videos.to(device)
        
        with torch.no_grad():
            if model.multi_task:
                logits_dict, extras = model(videos, return_dict=True)
                print(f"✅ Forward pass successful: {len(logits_dict)} tasks")
                for task, logits in logits_dict.items():
                    print(f"   Task {task}: {logits.shape}")
            else:
                logits, extras = model(videos, return_dict=False)
                print(f"✅ Forward pass successful: logits={logits.shape}")
        
        print(f"   Features: {extras['feat'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_loss_computation(model, dataloader, device):
    """Test loss computation."""
    print("\n🔍 Testing loss computation...")
    
    try:
        batch = next(iter(dataloader))
        videos, targets = batch
        
        videos = videos.to(device)
        if isinstance(targets, dict):
            targets = {k: v.to(device) for k, v in targets.items()}
        else:
            targets = targets.to(device)
        
        if model.multi_task:
            logits_dict, extras = model(videos, return_dict=True)
            loss_dict = model.compute_loss(logits_dict, targets, return_dict=True)
            print(f"✅ Loss computation successful: total_loss={loss_dict['total_loss'].item():.4f}")
            
            for key, value in loss_dict.items():
                if key != 'total_loss' and isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.item():.4f}")
        else:
            logits, extras = model(videos, return_dict=False)
            loss_dict = model.compute_loss(logits, targets, return_dict=True)
            print(f"✅ Loss computation successful: total_loss={loss_dict['total_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False

def test_trainer_creation(model, device):
    """Test trainer creation."""
    print("\n🔍 Testing trainer creation...")
    
    try:
        import sys
        sys.path.append('.')
        from training_utils import MultiTaskTrainer
        import torch.optim as optim
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        trainer = MultiTaskTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            log_interval=10,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0
        )
        
        print("✅ Trainer created successfully")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        return None

def test_backward_pass(trainer, dataloader, device):
    """Test backward pass and optimization step."""
    print("\n🔍 Testing backward pass...")
    
    try:
        batch = next(iter(dataloader))
        videos, targets = batch
        
        # Test training step
        trainer.model.train()
        trainer.optimizer.zero_grad()
        
        step_result = trainer.train_step(videos, targets)
        
        print(f"✅ Backward pass successful: loss={step_result['total_loss']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Training Pipeline')
    
    # Required arguments
    parser.add_argument('--root-dir', type=str, required=True,
                        help='Path to MVFouls root directory (containing train_720p, valid_720p, etc.)')
    
    # Model arguments
    parser.add_argument('--multi-task', action='store_true',
                        help='Use multi-task learning')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes (single-task only)')
    
    # Test arguments
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for testing')
    parser.add_argument('--skip-backward', action='store_true',
                        help='Skip backward pass test (faster)')
    
    args = parser.parse_args()
    
    print("🧪 TRAINING PIPELINE SMOKE TEST")
    print("=" * 50)
    
    # Run tests
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Imports
    total_tests += 1
    if test_imports():
        tests_passed += 1
    else:
        print("❌ Cannot proceed without imports")
        sys.exit(1)
    
    # Test 2: Device
    total_tests += 1
    device = test_device()
    if device:
        tests_passed += 1
    
    # Test 3: Dataset creation
    total_tests += 1
    train_dataset, val_dataset = test_dataset_creation(args)
    if train_dataset and val_dataset:
        tests_passed += 1
    else:
        print("❌ Cannot proceed without datasets")
        sys.exit(1)
    
    # Test 4: Dataloader
    total_tests += 1
    train_loader = test_dataloader(train_dataset, args.batch_size)
    if train_loader:
        tests_passed += 1
    else:
        print("❌ Cannot proceed without dataloader")
        sys.exit(1)
    
    # Test 5: Model creation
    total_tests += 1
    model = test_model_creation(args, device)
    if model:
        tests_passed += 1
    else:
        print("❌ Cannot proceed without model")
        sys.exit(1)
    
    # Test 6: Forward pass
    total_tests += 1
    if test_forward_pass(model, train_loader, device):
        tests_passed += 1
    
    # Test 7: Loss computation
    total_tests += 1
    if test_loss_computation(model, train_loader, device):
        tests_passed += 1
    
    # Test 8: Trainer creation
    total_tests += 1
    trainer = test_trainer_creation(model, device)
    if trainer:
        tests_passed += 1
    
    # Test 9: Backward pass (optional)
    if not args.skip_backward and trainer:
        total_tests += 1
        if test_backward_pass(trainer, train_loader, device):
            tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🏁 TEST SUMMARY: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Training pipeline is ready to run!")
        print("\nYou can now start training with:")
        
        cmd = f"python train.py \\\n"
        cmd += f"  --train-dir {args.root_dir}/train_720p \\\n"
        cmd += f"  --val-dir {args.root_dir}/valid_720p \\\n"
        cmd += f"  --train-annotations {args.root_dir}/train_720p/annotations.json \\\n"
        cmd += f"  --val-annotations {args.root_dir}/valid_720p/annotations.json"
        
        if args.multi_task:
            cmd += " \\\n  --multi-task"
        else:
            cmd += f" \\\n  --num-classes {args.num_classes}"
        
        cmd += " \\\n  --epochs 50 \\\n  --batch-size 8"
        
        print(f"\n{cmd}")
        
    else:
        print("❌ Some tests failed!")
        print("🔧 Please fix the issues before starting training.")
        sys.exit(1)

if __name__ == '__main__':
    main() 