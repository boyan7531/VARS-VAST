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
    python complete_model_demo.py
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


def demo_complete_model():
    """Demo the complete model functionality."""
    print("\n" + "="*60)
    print("🎯 COMPLETE MVFOULS MODEL DEMO")
    print("="*60)
    
    # Configuration
    config = {
        'num_classes': 2,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 3,  # Short demo
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Using device: {config['device']}")
    
    # 1. Create model
    logger.info("Creating complete MVFouls model...")
    model = build_single_task_model(
        num_classes=config['num_classes'],
        pretrained=False,  # Use False for faster demo
        freeze_backbone=False,
        head_dropout=0.3,
        head_loss_type='focal',
        head_pooling='avg',
        head_temporal_module='tconv',
        model_name="Complete_MVFouls_Demo",
        model_version="1.0"
    )
    
    model = model.to(config['device'])
    
    # 2. Create dummy dataset with transforms
    logger.info("Creating dataset with transforms...")
    train_transform = get_train_transforms(size=224)
    val_transform = get_val_transforms(size=224)
    
    class DemoDataset:
        def __init__(self, size=100, transform=None):
            self.size = size
            self.transform = transform
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Generate random video in (T, H, W, C) format as dataset would provide
            video = np.random.randint(0, 256, (32, 256, 256, 3), dtype=np.uint8)
            target = np.random.randint(0, 2)  # Binary classification
            
            sample = {'video': video, 'targets': torch.tensor(target, dtype=torch.long)}
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample['video'], sample['targets']
    
    train_dataset = DemoDataset(size=50, transform=train_transform)
    val_dataset = DemoDataset(size=20, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 3. Test data flow
    logger.info("Testing data flow...")
    sample_video, sample_target = next(iter(train_loader))
    logger.info(f"Sample batch: video={sample_video.shape}, target={sample_target.shape}")
    logger.info(f"Video dtype: {sample_video.dtype}, range: [{sample_video.min():.3f}, {sample_video.max():.3f}]")
    
    # 4. Test model forward pass
    logger.info("Testing model forward pass...")
    sample_video = sample_video.to(config['device'])
    with torch.no_grad():
        model.eval()
        logits, extras = model(sample_video)
        logger.info(f"Model output: logits={logits.shape}, extras={list(extras.keys())}")
        
        # Test prediction function
        pred_result = model.predict(sample_video[:1], return_probs=True)
        logger.info(f"Prediction: {pred_result['predictions'].item()}")
        logger.info(f"Probabilities: {pred_result['probabilities'][0].cpu().numpy()}")
    
    # 5. Setup training
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    logger.info("Starting training...")
    
    best_val_loss = float('inf')
    
    # 6. Training loop
    for epoch in range(config['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        train_metrics = model.fit_epoch(
            train_loader, 
            optimizer, 
            device=config['device'],
            scheduler=scheduler,
            log_interval=5  # More frequent logging for demo
        )
        
        logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
        
        # Validation
        val_results = model.evaluate(val_loader, device=config['device'])
        val_loss = val_results['avg_loss']
        
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Samples: {val_results['num_samples']}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                'best_complete_model.pth',
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_val_loss
            )
            logger.info("✓ Saved best model checkpoint")
    
    # 7. Test model features
    logger.info("Testing advanced features...")
    
    # Dynamic backbone freezing
    logger.info("Testing dynamic backbone freezing...")
    original_mode = model.backbone.freeze_mode
    model.set_backbone_freeze_mode('freeze_stages2')
    logger.info(f"Changed freeze mode to: {model.backbone.freeze_mode}")
    model.set_backbone_freeze_mode(original_mode)
    logger.info(f"Restored freeze mode to: {model.backbone.freeze_mode}")
    
    # Temperature scaling
    logger.info("Testing temperature scaling...")
    test_video = sample_video[:1]
    with torch.no_grad():
        pred_normal = model.predict(test_video, temperature=1.0, return_probs=True)
        pred_confident = model.predict(test_video, temperature=0.5, return_probs=True)
        
        logger.info(f"Normal (T=1.0): {pred_normal['probabilities'][0].numpy()}")
        logger.info(f"Confident (T=0.5): {pred_confident['probabilities'][0].numpy()}")
    
    # 8. Test model export
    logger.info("Testing ONNX export...")
    try:
        model.export_onnx('complete_model.onnx', input_shape=(1, 3, 32, 224, 224))
        logger.info("✓ ONNX export successful")
        
        # Clean up
        if os.path.exists('complete_model.onnx'):
            os.remove('complete_model.onnx')
            
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
    
    # 9. Test checkpoint loading
    logger.info("Testing checkpoint loading...")
    try:
        # Create new model and load checkpoint
        new_model = build_single_task_model(
            num_classes=config['num_classes'],
            pretrained=False
        )
        checkpoint_data = new_model.load_checkpoint('best_complete_model.pth')
        logger.info(f"✓ Loaded checkpoint from epoch {checkpoint_data['epoch']}")
        
        # Test that loaded model works
        new_model = new_model.to(config['device'])
        with torch.no_grad():
            new_logits, _ = new_model(test_video)
            logger.info(f"✓ Loaded model produces output: {new_logits.shape}")
        
        # Clean up
        if os.path.exists('best_complete_model.pth'):
            os.remove('best_complete_model.pth')
            
    except Exception as e:
        logger.error(f"Checkpoint loading failed: {e}")
    
    # 10. Model information
    logger.info("Getting model information...")
    model_info = model.get_model_info()
    logger.info(f"Model: {model_info['model_name']} v{model_info['model_version']}")
    logger.info(f"Parameters: {model_info['total_params']:,} total, {model_info['trainable_params']:,} trainable")
    logger.info(f"Training steps: {model_info['training_step']}")
    
    logger.info("Complete model demo finished! ✅")
    return model


def demo_multi_task_model():
    """Demo multi-task model if available."""
    print("\n" + "="*60)
    print("🎯 MULTI-TASK MODEL DEMO")
    print("="*60)
    
    try:
        from utils import get_task_metadata
        
        # Get task information
        metadata = get_task_metadata()
        logger.info(f"Found {metadata['total_tasks']} tasks for multi-task learning")
        logger.info(f"Task names: {metadata['task_names'][:3]}...")
        logger.info(f"Classes per task: {metadata['num_classes'][:3]}...")
        
        # Create multi-task model
        logger.info("Creating multi-task model...")
        model = build_multi_task_model(
            pretrained=False,
            freeze_backbone=False,
            head_dropout=0.3,
            model_name="Demo_MultiTask"
        )
        
        # Test forward pass
        logger.info("Testing multi-task forward pass...")
        dummy_input = torch.randn(2, 3, 32, 224, 224)
        
        model.eval()
        with torch.no_grad():
            logits_dict, extras = model(dummy_input, return_dict=True)
            
            logger.info(f"Multi-task outputs:")
            for task_name, task_logits in logits_dict.items():
                logger.info(f"  {task_name}: {task_logits.shape}")
            
            # Test prediction
            pred_result = model.predict(dummy_input[:1], return_probs=True)
            logger.info(f"Multi-task prediction keys: {list(pred_result.keys())}")
            
            # Show sample predictions
            logger.info("Sample predictions:")
            for task_name in list(pred_result['predictions'].keys())[:3]:
                pred = pred_result['predictions'][task_name].item()
                logger.info(f"  {task_name}: {pred}")
        
        logger.info("Multi-task demo completed! ✅")
        return model
        
    except ImportError:
        logger.warning("Multi-task model requires utils.py with task metadata")
        logger.info("Skipping multi-task demo...")
        return None
    except Exception as e:
        logger.error(f"Multi-task demo failed: {e}")
        return None


def demo_with_real_dataset():
    """Demo with real MVFouls dataset if available."""
    print("\n" + "="*60)
    print("📁 REAL DATASET DEMO")
    print("="*60)
    
    mvfouls_path = "mvfouls"
    if not Path(mvfouls_path).exists():
        logger.info(f"MVFouls dataset not found at {mvfouls_path}")
        logger.info("To test with real data:")
        logger.info("1. Download the MVFouls dataset")
        logger.info("2. Extract to 'mvfouls/' directory")
        logger.info("3. Run this demo again")
        return
    
    logger.info(f"Found MVFouls dataset at {mvfouls_path}")
    
    try:
        # Create transforms
        train_transform = get_train_transforms(size=224)
        val_transform = get_val_transforms(size=224)
        
        # Load datasets
        logger.info("Loading MVFouls datasets...")
        datasets = create_mvfouls_datasets(
            mvfouls_path,
            splits=['train', 'valid'],
            target_size=None,  # Let transforms handle resizing
            num_frames=32,
            center_frame=75
        )
        
        # Apply transforms
        datasets['train'].transform = train_transform
        datasets['valid'].transform = val_transform
        
        # Print dataset info
        for split, dataset in datasets.items():
            info = dataset.get_split_info()
            logger.info(f"{split}: {info['total_clips']} clips, {info['total_actions']} actions")
            
            if 'task_statistics' in info:
                logger.info(f"  Task statistics available for {len(info['task_statistics'])} tasks")
        
        # Create model
        logger.info("Creating model for real dataset...")
        model = build_single_task_model(
            num_classes=2,
            pretrained=False,
            freeze_backbone=True,  # Freeze for demo
            model_name="Real_Data_Demo"
        )
        
        # Test with real data
        train_loader = DataLoader(datasets['train'], batch_size=2, shuffle=True, num_workers=0)
        
        logger.info("Testing with real data...")
        model.eval()
        with torch.no_grad():
            videos, targets = next(iter(train_loader))
            logger.info(f"Real data batch: videos={videos.shape}, targets={targets.shape}")
            
            # Forward pass
            logits, extras = model(videos)
            logger.info(f"Model output on real data: {logits.shape}")
            
            # Handle multi-task targets for single-task model
            if targets.dim() > 1:
                targets_single = targets[:, 0]  # Use first task
            else:
                targets_single = targets
            
            loss = model.compute_loss(logits, targets_single)
            logger.info(f"Loss on real data: {loss.item():.4f}")
        
        logger.info("Real dataset demo completed! ✅")
        
    except Exception as e:
        logger.error(f"Real dataset demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demo function."""
    print("🎬 Complete MVFouls Model Demo")
    print("=" * 80)
    
    logger.info("Starting comprehensive model demo...")
    
    try:
        # Demo 1: Complete model functionality
        complete_model = demo_complete_model()
        
        # Demo 2: Multi-task model (if available)
        multi_model = demo_multi_task_model()
        
        # Demo 3: Real dataset integration (if available)
        demo_with_real_dataset()
        
        print("\n" + "="*80)
        print("🎉 ALL DEMOS COMPLETED!")
        print("="*80)
        
        # Final summary
        logger.info("Demo Summary:")
        logger.info("✅ Complete model: backbone + head integration")
        logger.info("✅ End-to-end pipeline: data -> transforms -> model -> predictions")
        logger.info("✅ Training and evaluation loops")
        logger.info("✅ Advanced features: freezing, temperature scaling, export")
        logger.info("✅ Checkpointing and model persistence")
        
        if multi_model is not None:
            logger.info("✅ Multi-task learning capabilities")
        else:
            logger.info("⚠️  Multi-task demo skipped (requires utils.py)")
        
        print("\n🚀 Your complete MVFouls model is ready to use!")
        print("\nNext steps:")
        print("1. 📊 Replace dummy data with your actual MVFouls dataset")
        print("2. 🎯 Tune hyperparameters for optimal performance")  
        print("3. 🔄 Train for more epochs with proper validation")
        print("4. 📈 Monitor metrics and adjust model architecture")
        print("5. 🚀 Deploy using ONNX export for production use")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 