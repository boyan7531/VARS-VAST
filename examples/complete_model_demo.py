"""
Complete MVFouls Model Demo
==========================

This script demonstrates how to use the complete MVFouls model end-to-end:
1. Load and preprocess data using MVFoulsDataset and transforms
2. Create single-task and multi-task models
3. Train and evaluate the models
4. Save and load checkpoints
5. Make predictions on new data

Usage:
    python examples/complete_model_demo.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model components
from model.mvfouls_model import (
    MVFoulsModel, build_single_task_model, build_multi_task_model
)

# Import dataset and transforms
from dataset import MVFoulsDataset, create_mvfouls_datasets
from transforms import get_train_transforms, get_val_transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_single_task_training():
    """Demo single-task model training pipeline."""
    print("\n" + "="*60)
    print("🎯 SINGLE-TASK MODEL DEMO")
    print("="*60)
    
    # Configuration
    config = {
        'num_classes': 2,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 2,  # Short demo
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Using device: {config['device']}")
    
    # Create model
    logger.info("Creating single-task model...")
    model = build_single_task_model(
        num_classes=config['num_classes'],
        pretrained=False,  # Use False for faster demo
        freeze_backbone=False,
        head_dropout=0.3,
        head_loss_type='focal',
        model_name="Demo_SingleTask",
        model_version="1.0"
    )
    
    model = model.to(config['device'])
    
    # Create dummy dataset (replace with real dataset)
    logger.info("Creating dummy dataset...")
    train_transform = get_train_transforms(size=224)
    val_transform = get_val_transforms(size=224)
    
    # For demo purposes, create synthetic data
    # In practice, use: datasets = create_mvfouls_datasets("path/to/mvfouls", transforms=...)
    class DummyDataset:
        def __init__(self, size=100, transform=None):
            self.size = size
            self.transform = transform
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Generate random video (T, H, W, C) format
            video = np.random.randint(0, 256, (32, 256, 256, 3), dtype=np.uint8)
            target = np.random.randint(0, 2)  # Binary classification
            
            sample = {'video': video, 'targets': torch.tensor(target, dtype=torch.long)}
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample['video'], sample['targets']
    
    train_dataset = DummyDataset(size=50, transform=train_transform)
    val_dataset = DummyDataset(size=20, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    logger.info("Starting training...")
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        train_metrics = model.fit_epoch(
            train_loader, 
            optimizer, 
            device=config['device'],
            scheduler=scheduler,
            log_interval=10
        )
        
        logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
        
        # Validation
        val_results = model.evaluate(val_loader, device=config['device'])
        val_loss = val_results['avg_loss']
        
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                'best_single_task_model.pth',
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_val_loss
            )
            logger.info("✓ Saved best model checkpoint")
    
    # Test prediction
    logger.info("Testing prediction...")
    model.eval()
    
    with torch.no_grad():
        # Get a sample
        sample_video, sample_target = next(iter(val_loader))
        sample_video = sample_video[:1].to(config['device'])  # Take first sample
        
        # Make prediction
        pred_result = model.predict(sample_video, return_probs=True)
        
        logger.info(f"Prediction: {pred_result['predictions'].item()}")
        logger.info(f"Probabilities: {pred_result['probabilities'][0].cpu().numpy()}")
        logger.info(f"True label: {sample_target[0].item()}")
    
    # Test model export
    logger.info("Testing ONNX export...")
    try:
        model.export_onnx('single_task_model.onnx', input_shape=(1, 3, 32, 224, 224))
        logger.info("✓ ONNX export successful")
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
    
    logger.info("Single-task demo completed! ✅")
    return model


def demo_multi_task_training():
    """Demo multi-task model training pipeline."""
    print("\n" + "="*60)
    print("🎯 MULTI-TASK MODEL DEMO")
    print("="*60)
    
    try:
        from utils import get_task_metadata
        
        # Get task information
        metadata = get_task_metadata()
        logger.info(f"Detected {metadata['total_tasks']} tasks: {metadata['task_names'][:3]}...")
        
    except ImportError:
        logger.warning("Multi-task demo requires utils.py with task metadata")
        return None
    
    # Configuration
    config = {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 2,  # Short demo
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create model
    logger.info("Creating multi-task model...")
    model = build_multi_task_model(
        pretrained=False,  # Use False for faster demo
        freeze_backbone=False,
        head_dropout=0.3,
        head_loss_type='focal',
        model_name="Demo_MultiTask",
        model_version="1.0"
    )
    
    model = model.to(config['device'])
    
    # Create dummy multi-task dataset
    logger.info("Creating dummy multi-task dataset...")
    train_transform = get_train_transforms(size=224)
    val_transform = get_val_transforms(size=224)
    
    class DummyMultiTaskDataset:
        def __init__(self, size=100, transform=None):
            self.size = size
            self.transform = transform
            self.num_tasks = metadata['total_tasks']
            self.num_classes_per_task = metadata['num_classes']
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Generate random video
            video = np.random.randint(0, 256, (32, 256, 256, 3), dtype=np.uint8)
            
            # Generate random targets for all tasks
            targets = []
            for num_classes in self.num_classes_per_task:
                targets.append(np.random.randint(0, num_classes))
            
            targets = torch.tensor(targets, dtype=torch.long)
            
            sample = {'video': video, 'targets': targets}
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample['video'], sample['targets']
    
    train_dataset = DummyMultiTaskDataset(size=50, transform=train_transform)
    val_dataset = DummyMultiTaskDataset(size=20, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    logger.info("Starting multi-task training...")
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        train_metrics = model.fit_epoch(
            train_loader, 
            optimizer, 
            device=config['device'],
            scheduler=scheduler,
            log_interval=10
        )
        
        logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
        
        # Validation
        val_results = model.evaluate(val_loader, device=config['device'])
        val_loss = val_results['avg_loss']
        
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        # Print detailed metrics if available
        if 'task_metrics' in val_results:
            logger.info("Task-specific metrics:")
            for task_name, metrics in val_results['task_metrics'].items():
                logger.info(f"  {task_name}: Acc={metrics['accuracy']:.3f}, F1={metrics['macro_f1']:.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                'best_multi_task_model.pth',
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_val_loss
            )
            logger.info("✓ Saved best model checkpoint")
    
    # Test prediction
    logger.info("Testing multi-task prediction...")
    model.eval()
    
    with torch.no_grad():
        # Get a sample
        sample_video, sample_targets = next(iter(val_loader))
        sample_video = sample_video[:1].to(config['device'])  # Take first sample
        
        # Make prediction
        pred_result = model.predict(sample_video, return_probs=True)
        
        logger.info("Multi-task predictions:")
        for task_name in metadata['task_names'][:3]:  # Show first 3 tasks
            if task_name in pred_result['predictions']:
                pred = pred_result['predictions'][task_name].item()
                if task_name in pred_result.get('probabilities', {}):
                    probs = pred_result['probabilities'][task_name][0].cpu().numpy()
                    logger.info(f"  {task_name}: pred={pred}, probs={probs}")
                else:
                    logger.info(f"  {task_name}: pred={pred}")
    
    # Test model export
    logger.info("Testing multi-task ONNX export...")
    try:
        model.export_onnx('multi_task_model.onnx', input_shape=(1, 3, 32, 224, 224), export_mode='concat')
        logger.info("✓ Multi-task ONNX export successful")
    except Exception as e:
        logger.warning(f"Multi-task ONNX export failed: {e}")
    
    logger.info("Multi-task demo completed! ✅")
    return model


def demo_model_features():
    """Demo advanced model features."""
    print("\n" + "="*60)
    print("🔧 ADVANCED FEATURES DEMO")
    print("="*60)
    
    # Create a model for feature testing
    logger.info("Creating model for feature testing...")
    model = build_single_task_model(
        num_classes=3,
        pretrained=False,
        freeze_backbone=False,
        head_temporal_module='tconv',  # Test temporal convolution
        head_pooling='attention',      # Test attention pooling
        model_name="Demo_Features",
        model_version="1.0"
    )
    
    # Test dynamic backbone freezing
    logger.info("Testing dynamic backbone freezing...")
    logger.info(f"Initial freeze mode: {model.backbone.freeze_mode}")
    
    # Freeze backbone
    model.set_backbone_freeze_mode('freeze_all')
    logger.info(f"After freezing: {model.backbone.freeze_mode}")
    
    # Test gradual unfreezing
    model.set_backbone_freeze_mode('gradual')
    logger.info("Testing gradual unfreezing...")
    for i in range(3):
        model.unfreeze_backbone_gradually()
        current_stage = model.backbone.get_current_unfreeze_stage()
        logger.info(f"  Unfreeze step {i+1}: stage {current_stage}")
    
    # Test model info
    logger.info("Getting model information...")
    model_info = model.get_model_info()
    logger.info(f"Model has {model_info['total_params']:,} total parameters")
    logger.info(f"Trainable: {model_info['trainable_params']:,} ({model_info['trainable_params']/model_info['total_params']*100:.1f}%)")
    
    # Test different input formats
    logger.info("Testing different input formats...")
    
    # Format 1: (B, C, T, H, W) - standard
    input1 = torch.randn(2, 3, 32, 224, 224)
    try:
        output1, extras1 = model(input1)
        logger.info(f"✓ Standard format (B,C,T,H,W): {input1.shape} -> {output1.shape}")
    except Exception as e:
        logger.error(f"✗ Standard format failed: {e}")
    
    # Format 2: (B, T, H, W, C) - dataset format
    input2 = torch.randn(2, 32, 224, 224, 3)
    try:
        output2, extras2 = model(input2)
        logger.info(f"✓ Dataset format (B,T,H,W,C): {input2.shape} -> {output2.shape}")
    except Exception as e:
        logger.error(f"✗ Dataset format failed: {e}")
    
    # Test prediction with temperature scaling
    logger.info("Testing temperature scaling...")
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 32, 224, 224)
        
        # Normal prediction
        pred_normal = model.predict(test_input, return_probs=True, temperature=1.0)
        
        # High temperature (more confident)
        pred_confident = model.predict(test_input, return_probs=True, temperature=0.5)
        
        # Low temperature (less confident)
        pred_uncertain = model.predict(test_input, return_probs=True, temperature=2.0)
        
        logger.info("Temperature scaling effects:")
        logger.info(f"  Normal (T=1.0): {pred_normal['probabilities'][0].numpy()}")
        logger.info(f"  Confident (T=0.5): {pred_confident['probabilities'][0].numpy()}")
        logger.info(f"  Uncertain (T=2.0): {pred_uncertain['probabilities'][0].numpy()}")
    
    # Test checkpoint loading
    logger.info("Testing checkpoint save/load...")
    try:
        # Save checkpoint
        checkpoint_path = 'test_checkpoint.pth'
        model.save_checkpoint(checkpoint_path, epoch=5, best_metric=0.85)
        
        # Create new model and load checkpoint
        new_model = build_single_task_model(num_classes=3, pretrained=False)
        checkpoint_data = new_model.load_checkpoint(checkpoint_path)
        
        logger.info(f"✓ Checkpoint loaded: epoch {checkpoint_data['epoch']}, metric {checkpoint_data['best_metric']}")
        
        # Clean up
        os.remove(checkpoint_path)
        
    except Exception as e:
        logger.error(f"✗ Checkpoint test failed: {e}")
    
    logger.info("Advanced features demo completed! ✅")


def demo_real_dataset_integration():
    """Demo integration with real MVFouls dataset (if available)."""
    print("\n" + "="*60)
    print("📁 REAL DATASET INTEGRATION DEMO")
    print("="*60)
    
    # Check if MVFouls dataset is available
    mvfouls_path = "mvfouls"
    if not Path(mvfouls_path).exists():
        logger.warning(f"MVFouls dataset not found at {mvfouls_path}")
        logger.info("To test with real data:")
        logger.info("1. Download the MVFouls dataset")
        logger.info("2. Extract to 'mvfouls/' directory")
        logger.info("3. Run this demo again")
        return
    
    logger.info(f"Found MVFouls dataset at {mvfouls_path}")
    
    try:
        # Create datasets with transforms
        logger.info("Loading datasets...")
        train_transform = get_train_transforms(size=224)
        val_transform = get_val_transforms(size=224)
        
        # Create datasets - set target_size=None since transforms handle resizing
        datasets = create_mvfouls_datasets(
            mvfouls_path,
            splits=['train', 'valid'],
            target_size=None,  # Let transforms handle resizing
            num_frames=32,
            center_frame=75,
            cache_mode='none'
        )
        
        # Apply transforms
        datasets['train'].transform = train_transform
        datasets['valid'].transform = val_transform
        
        # Print dataset info
        for split, dataset in datasets.items():
            info = dataset.get_split_info()
            logger.info(f"{split}: {info['total_clips']} clips, {info['total_actions']} actions")
        
        # Create data loaders
        train_loader = DataLoader(datasets['train'], batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(datasets['valid'], batch_size=2, shuffle=False, num_workers=0)
        
        # Test with single-task model
        logger.info("Testing with single-task model...")
        model = build_single_task_model(
            num_classes=2,  # Binary: foul vs no foul
            pretrained=False,
            freeze_backbone=True,  # Freeze for faster demo
            model_name="Real_Data_Demo"
        )
        
        # Test one batch
        model.eval()
        with torch.no_grad():
            videos, targets = next(iter(train_loader))
            logger.info(f"Batch shape: videos={videos.shape}, targets={targets.shape}")
            
            # Forward pass
            logits, extras = model(videos)
            logger.info(f"Model output: {logits.shape}")
            
            # Compute loss
            if targets.dim() > 1:
                # Multi-task targets - use first task for single-task demo
                targets_single = targets[:, 0]
            else:
                targets_single = targets
                
            loss = model.compute_loss(logits, targets_single)
            logger.info(f"Loss: {loss.item():.4f}")
        
        # Test with multi-task model if possible
        try:
            logger.info("Testing with multi-task model...")
            multi_model = build_multi_task_model(
                pretrained=False,
                freeze_backbone=True,
                model_name="Real_MultiTask_Demo"
            )
            
            multi_model.eval()
            with torch.no_grad():
                logits_dict, extras = multi_model(videos)
                logger.info(f"Multi-task outputs: {len(logits_dict)} tasks")
                
                # Convert targets to dict format
                if targets.dim() > 1:
                    from utils import get_task_metadata
                    metadata = get_task_metadata()
                    targets_dict = {}
                    for i, task_name in enumerate(metadata['task_names']):
                        targets_dict[task_name] = targets[:, i]
                    
                    loss_dict = multi_model.compute_loss(logits_dict, targets_dict, return_dict=True)
                    logger.info(f"Multi-task loss: {loss_dict['total_loss'].item():.4f}")
                
        except Exception as e:
            logger.warning(f"Multi-task test failed: {e}")
        
        logger.info("Real dataset integration successful! ✅")
        
    except Exception as e:
        logger.error(f"Real dataset demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demo function."""
    print("🎬 MVFouls Complete Model Demo")
    print("=" * 80)
    
    logger.info("Starting comprehensive model demo...")
    
    # Run demos
    try:
        # Demo 1: Single-task training
        single_model = demo_single_task_training()
        
        # Demo 2: Multi-task training (if available)
        multi_model = demo_multi_task_training()
        
        # Demo 3: Advanced features
        demo_model_features()
        
        # Demo 4: Real dataset integration (if available)
        demo_real_dataset_integration()
        
        print("\n" + "="*80)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Summary
        logger.info("Demo Summary:")
        logger.info("✅ Single-task model: Training, evaluation, prediction")
        logger.info("✅ Multi-task model: Advanced metrics, task-specific losses")
        logger.info("✅ Advanced features: Dynamic freezing, temperature scaling")
        logger.info("✅ Real dataset: Integration with MVFouls data (if available)")
        logger.info("✅ Model export: ONNX format for deployment")
        logger.info("✅ Checkpointing: Save/load model state")
        
        print("\nNext steps:")
        print("1. 📊 Use your own dataset by modifying the dataset loading")
        print("2. 🎯 Tune hyperparameters for your specific task")
        print("3. 🚀 Deploy using the ONNX export functionality")
        print("4. 📈 Monitor training with the built-in metrics")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 