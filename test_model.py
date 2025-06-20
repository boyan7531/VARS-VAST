#!/usr/bin/env python3
"""
Simple test script for Video Swin Transformer implementation.
"""

import sys
import torch
import numpy as np

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from transforms import get_train_transforms, get_val_transforms
        print("✅ Transforms imported successfully")
    except Exception as e:
        print(f"❌ Error importing transforms: {e}")
        return False
    
    try:
        from model.video_swin_transformer import VideoSwinTransformer, VideoSwinConfig
        print("✅ Video Swin Transformer imported successfully")
    except Exception as e:
        print(f"❌ Error importing Video Swin Transformer: {e}")
        return False
    
    try:
        from model.classification_heads import SimpleClassificationHead
        print("✅ Classification heads imported successfully")
    except Exception as e:
        print(f"❌ Error importing classification heads: {e}")
        return False
    
    try:
        from model.mvfouls_model import create_mvfouls_model
        print("✅ MVFouls model imported successfully")
    except Exception as e:
        print(f"❌ Error importing MVFouls model: {e}")
        return False
    
    return True


def test_transforms():
    """Test the transform pipeline."""
    print("\n🔄 Testing transforms...")
    
    try:
        from transforms import get_train_transforms, get_val_transforms
        
        # Create dummy video data
        dummy_video = np.random.randint(0, 256, (32, 224, 224, 3), dtype=np.uint8)
        dummy_sample = {'video': dummy_video, 'action_id': 0}
        
        # Test training transforms
        train_transform = get_train_transforms(size=224)
        train_result = train_transform(dummy_sample.copy())
        
        expected_shape = torch.Size([3, 32, 224, 224])
        if train_result['video'].shape == expected_shape:
            print(f"✅ Training transforms: {train_result['video'].shape}")
        else:
            print(f"❌ Training transforms shape mismatch: got {train_result['video'].shape}, expected {expected_shape}")
            return False
        
        # Test validation transforms
        val_transform = get_val_transforms(size=224)
        val_result = val_transform(dummy_sample.copy())
        
        if val_result['video'].shape == expected_shape:
            print(f"✅ Validation transforms: {val_result['video'].shape}")
        else:
            print(f"❌ Validation transforms shape mismatch: got {val_result['video'].shape}, expected {expected_shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing transforms: {e}")
        return False


def test_model():
    """Test the Video Swin Transformer model."""
    print("\n🧠 Testing Video Swin Transformer model...")
    
    try:
        from model import create_mvfouls_model
        
        # Create model configuration
        task_configs = {
            'action_class': 8,
            'offence': 4,
            'severity': 1
        }
        
        # Create model
        model = create_mvfouls_model(
            task_configs=task_configs,
            pretrained=False,  # Set to False for testing
            freeze_backbone=False
        )
        
        print(f"✅ Model created successfully")
        
        # Test parameter counting
        param_counts = model.get_trainable_parameters()
        print(f"✅ Parameter counts: {param_counts}")
        
        # Test forward pass
        batch_size = 1  # Reduced batch size
        test_input = torch.randn(batch_size, 3, 32, 224, 224)
        
        with torch.no_grad():
            outputs = model(test_input)
        
        print(f"✅ Forward pass successful")
        print("Output shapes:")
        for task_name, output in outputs.items():
            print(f"  {task_name}: {output.shape}")
        
        # Test backbone freezing
        model.freeze_backbone()
        frozen_param = next(model.backbone.parameters())
        if not frozen_param.requires_grad:
            print("✅ Backbone freezing works")
        else:
            print("❌ Backbone freezing failed")
            return False
        
        model.unfreeze_backbone()
        unfrozen_param = next(model.backbone.parameters())
        if unfrozen_param.requires_grad:
            print("✅ Backbone unfreezing works")
        else:
            print("❌ Backbone unfreezing failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Video Swin Transformer Test Suite")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your dependencies.")
        sys.exit(1)
    
    # Test 2: Transforms
    if not test_transforms():
        print("\n❌ Transform tests failed.")
        sys.exit(1)
    
    # Test 3: Model
    if not test_model():
        print("\n❌ Model tests failed.")
        sys.exit(1)
    
    print("\n🎉 All tests passed!")
    print("\n✨ Your Video Swin Transformer implementation is ready!")
    print("\nNext steps:")
    print("1. Download the MVFouls dataset using download_dataset.py")
    print("2. Run example_usage.py to train your model")
    print("3. Experiment with different configurations and parameters")


if __name__ == "__main__":
    main() 