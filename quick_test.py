#!/usr/bin/env python3
"""
Quick Test Script for MVFouls Training Pipeline
==============================================

Simple test to check if basic components work.
"""

import os
import sys
import torch

def test_basic_imports():
    """Test basic PyTorch and system imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import numpy as np
        import pandas as pd
        print("  ✓ numpy, pandas")
        
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  ✓ GPU: {torch.cuda.get_device_name()}")
        
        return True
    except Exception as e:
        print(f"  ❌ Basic imports failed: {e}")
        return False

def test_file_structure():
    """Test if required files exist."""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        'model/mvfouls_model.py',
        'model/backbone.py',
        'model/head.py',
        'dataset.py',
        'transforms.py',
        'training_utils.py',
        'utils.py',
        'train.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    return True

def test_data_paths(train_dir, val_dir, train_annotations, val_annotations):
    """Test if data paths exist."""
    print("\n🔍 Testing data paths...")
    
    paths_to_check = [
        ("Train directory", train_dir),
        ("Val directory", val_dir),
        ("Train annotations", train_annotations),
        ("Val annotations", val_annotations)
    ]
    
    missing_paths = []
    for name, path in paths_to_check:
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
            
            # Check if directory has files
            if os.path.isdir(path):
                files = os.listdir(path)
                print(f"    Contains {len(files)} items")
        else:
            print(f"  ❌ {name}: {path}")
            missing_paths.append(path)
    
    if missing_paths:
        print(f"\n❌ Missing paths: {missing_paths}")
        return False
    
    return True

def test_simple_model():
    """Test creating a simple model."""
    print("\n🔍 Testing simple model creation...")
    
    try:
        # Try to import and create a basic model
        sys.path.append('.')
        
        from model.backbone import VideoSwinBackbone
        print("  ✓ Backbone import successful")
        
        from model.head import MVFoulsHead
        print("  ✓ Head import successful")
        
        # Create simple components
        backbone = VideoSwinBackbone(pretrained=False, freeze_mode='none')
        head = MVFoulsHead(in_dim=1024, num_classes=2)
        
        print("  ✓ Components created successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 8, 224, 224)  # (B, C, T, H, W)
        
        with torch.no_grad():
            features = backbone(dummy_input)
            logits, extras = head(features)
        
        print(f"  ✓ Forward pass: input {dummy_input.shape} -> features {features.shape} -> logits {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick MVFouls Test')
    parser.add_argument('--root-dir', type=str, help='MVFouls root directory')
    
    args = parser.parse_args()
    
    print("🧪 QUICK MVFOULS TEST")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Basic imports
    total_tests += 1
    if test_basic_imports():
        tests_passed += 1
    
    # Test 2: File structure
    total_tests += 1
    if test_file_structure():
        tests_passed += 1
    
    # Test 3: Data paths (if provided)
    if args.root_dir:
        total_tests += 1
        train_dir = f"{args.root_dir}/train_720p"
        val_dir = f"{args.root_dir}/valid_720p"
        train_annotations = f"{args.root_dir}/train_720p/annotations.json"
        val_annotations = f"{args.root_dir}/valid_720p/annotations.json"
        if test_data_paths(train_dir, val_dir, train_annotations, val_annotations):
            tests_passed += 1
    
    # Test 4: Simple model
    total_tests += 1
    if test_simple_model():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 40)
    print(f"🏁 SUMMARY: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 Basic tests passed!")
        print("✅ Try running the full test script next:")
        print("   python test_training_pipeline.py --train-dir ... --val-dir ...")
    else:
        print("❌ Some basic tests failed!")
        print("🔧 Fix these issues before proceeding.")

if __name__ == '__main__':
    main() 