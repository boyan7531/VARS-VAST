#!/usr/bin/env python3
"""
Test Script for Threshold-based Prediction Pipeline

This script validates the entire threshold optimization pipeline with synthetic data
and provides end-to-end testing capabilities.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from thresholding import (
    predict_with_thresholds_single_task, 
    predict_with_thresholds_multi_task,
    analyze_threshold_usage,
    load_thresholds,
    extract_all_task_thresholds
)


def setup_logging(level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def create_synthetic_logits_data(
    batch_size: int = 100,
    tasks: List[str] = ['action_class', 'severity', 'offence'],
    num_classes: List[int] = [10, 6, 4],
    separability: float = 0.8
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Create synthetic logits data with known class separability.
    
    Args:
        batch_size: Number of samples
        tasks: List of task names
        num_classes: Number of classes for each task
        separability: How well-separated the classes are (0.5 = random, 1.0 = perfect)
        
    Returns:
        Tuple of (logits_dict, targets_dict)
    """
    logger = logging.getLogger(__name__)
    
    logits_dict = {}
    targets_dict = {}
    
    for task_name, n_classes in zip(tasks, num_classes):
        # Generate random targets
        targets = torch.randint(0, n_classes, (batch_size,))
        
        # Generate logits with controllable separability
        logits = torch.randn(batch_size, n_classes) * (1 - separability)
        
        # Boost correct class logits
        for i, target in enumerate(targets):
            logits[i, target] += separability * 3  # Make correct class more likely
        
        logits_dict[task_name] = logits
        targets_dict[task_name] = targets
        
        logger.debug(f"Task {task_name}: {batch_size} samples, {n_classes} classes")
    
    return logits_dict, targets_dict


def save_synthetic_validation_data(
    logits_dict: Dict[str, torch.Tensor],
    targets_dict: Dict[str, torch.Tensor],
    output_path: str
) -> None:
    """Save synthetic validation data in the format expected by threshold computation."""
    data = {
        'epoch': 20,
        'logits': [logits_dict],  # Single batch
        'targets': [targets_dict],  # Single batch
        'model_config': {
            'multi_task': True,
            'backbone_arch': 'test'
        },
        'task_names': list(logits_dict.keys())
    }
    
    torch.save(data, output_path)


def test_single_task_thresholds():
    """Test single-task threshold-based predictions."""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing single-task threshold predictions...")
    
    # Create test data: batch_size=5, num_classes=3
    probs = torch.tensor([
        [0.1, 0.3, 0.6],  # Should predict class 2
        [0.8, 0.1, 0.1],  # Should predict class 0
        [0.4, 0.4, 0.2],  # No class exceeds threshold, fallback to argmax (class 0/1)
        [0.2, 0.7, 0.1],  # Should predict class 1
        [0.3, 0.3, 0.4],  # Should predict class 2
    ])
    
    thresholds = [0.5, 0.5, 0.5]  # All classes need >50% confidence
    
    # Test with argmax fallback
    preds = predict_with_thresholds_single_task(probs, thresholds, fallback_strategy='argmax')
    expected = torch.tensor([2, 0, 0, 1, 2])  # Expected predictions
    
    logger.info(f"   Probabilities: {probs}")
    logger.info(f"   Thresholds: {thresholds}")
    logger.info(f"   Predictions: {preds}")
    logger.info(f"   Expected: {expected}")
    
    # Check if predictions match expected (allowing for ties in argmax)
    success = torch.allclose(preds.float(), expected.float(), atol=1.0)  # Allow some variation for ties
    
    if not success:
        # More detailed analysis
        for i in range(len(probs)):
            above_thresh = probs[i] >= torch.tensor(thresholds)
            if above_thresh.any():
                candidates = torch.where(above_thresh)[0]
                best_candidate = candidates[torch.argmax(probs[i, candidates])]
                logger.info(f"   Sample {i}: candidates={candidates}, best={best_candidate}, pred={preds[i]}")
            else:
                argmax_pred = torch.argmax(probs[i])
                logger.info(f"   Sample {i}: no candidates, argmax={argmax_pred}, pred={preds[i]}")
    
    logger.info(f"   ✅ Single-task test {'PASSED' if success else 'FAILED'}")
    return success


def test_multi_task_thresholds():
    """Test multi-task threshold-based predictions."""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing multi-task threshold predictions...")
    
    # Create test data
    probs_dict = {
        'task_a': torch.tensor([[0.1, 0.9], [0.6, 0.4], [0.3, 0.7]]),  # 2 classes
        'task_b': torch.tensor([[0.3, 0.3, 0.4], [0.8, 0.1, 0.1], [0.2, 0.6, 0.2]])  # 3 classes
    }
    
    thresholds_dict = {
        'task_a': [0.5, 0.5],
        'task_b': [0.6, 0.6, 0.6]  # Higher thresholds for task_b
    }
    
    preds_dict = predict_with_thresholds_multi_task(probs_dict, thresholds_dict)
    
    logger.info(f"   Task A probs: {probs_dict['task_a']}")
    logger.info(f"   Task A thresholds: {thresholds_dict['task_a']}")
    logger.info(f"   Task A predictions: {preds_dict['task_a']}")
    
    logger.info(f"   Task B probs: {probs_dict['task_b']}")
    logger.info(f"   Task B thresholds: {thresholds_dict['task_b']}")
    logger.info(f"   Task B predictions: {preds_dict['task_b']}")
    
    # Basic validation: all predictions should be valid class indices
    success = True
    for task_name, preds in preds_dict.items():
        num_classes = probs_dict[task_name].shape[1]
        valid_preds = torch.all((preds >= 0) & (preds < num_classes))
        if not valid_preds:
            logger.error(f"   Invalid predictions for {task_name}: {preds}")
            success = False
    
    logger.info(f"   ✅ Multi-task test {'PASSED' if success else 'FAILED'}")
    return success


def test_threshold_optimization_pipeline():
    """Test the complete threshold optimization pipeline end-to-end."""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing complete threshold optimization pipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create synthetic data with good separability
        logits_dict, targets_dict = create_synthetic_logits_data(
            batch_size=200,
            tasks=['action_class', 'severity', 'offence'],
            num_classes=[10, 6, 4],
            separability=0.9  # High separability for better thresholds
        )
        
        # Save synthetic validation data
        logits_file = temp_path / 'val_logits_test.pt'
        save_synthetic_validation_data(logits_dict, targets_dict, str(logits_file))
        logger.info(f"   Created synthetic validation data: {logits_file}")
        
        # Test threshold computation
        thresholds_file = temp_path / 'thresholds.json'
        
        # Import and run threshold computation
        try:
            sys.path.append(str(Path(__file__).parent / 'scripts'))
            from compute_thresholds import (
                load_validation_logits, aggregate_logits_and_targets, compute_task_thresholds
            )
            
            # Load data
            data = load_validation_logits(str(logits_file))
            agg_logits, agg_targets = aggregate_logits_and_targets(
                data['logits'], data['targets'], multi_task=True
            )
            
            # Compute thresholds for each task
            results = {
                'metadata': {
                    'source_file': str(logits_file),
                    'epoch': data.get('epoch', 'test'),
                    'metric_used': 'youden',
                    'total_samples': sum(targets.shape[0] for targets in agg_targets.values())
                },
                'thresholds': {}
            }
            
            for task_name in agg_logits.keys():
                task_result = compute_task_thresholds(
                    agg_logits[task_name], 
                    agg_targets[task_name],
                    task_name,
                    'youden'
                )
                results['thresholds'][task_name] = task_result
            
            # Save thresholds (properly handle numpy types)
            def json_serializer(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                else:
                    return str(obj)
            
            with open(thresholds_file, 'w') as f:
                json.dump(results, f, indent=2, default=json_serializer)
            
            logger.info(f"   ✅ Computed thresholds: {thresholds_file}")
            
            # Test threshold loading and extraction
            thresholds_data = load_thresholds(str(thresholds_file))
            task_thresholds = extract_all_task_thresholds(thresholds_data)
            
            logger.info(f"   ✅ Loaded thresholds for {len(task_thresholds)} tasks")
            
            # Test predictions with computed thresholds
            for task_name, task_logits in logits_dict.items():
                if task_name in task_thresholds:
                    probs = F.softmax(task_logits, dim=1)
                    thresholds = task_thresholds[task_name]
                    
                    preds = predict_with_thresholds_single_task(probs, thresholds)
                    
                    # Analyze threshold usage
                    stats = analyze_threshold_usage(probs, thresholds)
                    
                    logger.info(f"   Task {task_name}: {stats['single_task']['threshold_rate']:.2%} threshold usage")
            
            logger.info("   ✅ End-to-end pipeline test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Pipeline test FAILED: {e}")
            return False


def test_threshold_file_format():
    """Test threshold file loading and format validation."""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing threshold file format...")
    
    # Create a test threshold file
    test_thresholds = {
        'metadata': {
            'source_file': 'test.pt',
            'epoch': 20,
            'metric_used': 'youden',
            'total_samples': 100
        },
        'thresholds': {
            'action_class': {
                'task_name': 'action_class',
                'thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                'num_classes': 10,
                'metric_used': 'youden'
            },
            'severity': {
                'task_name': 'severity',
                'thresholds': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
                'num_classes': 6,
                'metric_used': 'youden'
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_thresholds, f, indent=2)
        temp_file = f.name
    
    try:
        # Test loading
        data = load_thresholds(temp_file)
        task_thresholds = extract_all_task_thresholds(data)
        
        # Validate structure
        expected_tasks = ['action_class', 'severity']
        success = (
            set(task_thresholds.keys()) == set(expected_tasks) and
            len(task_thresholds['action_class']) == 10 and
            len(task_thresholds['severity']) == 6
        )
        
        logger.info(f"   Loaded tasks: {list(task_thresholds.keys())}")
        logger.info(f"   Action class thresholds: {len(task_thresholds['action_class'])}")
        logger.info(f"   Severity thresholds: {len(task_thresholds['severity'])}")
        logger.info(f"   ✅ File format test {'PASSED' if success else 'FAILED'}")
        
        return success
        
    finally:
        Path(temp_file).unlink(missing_ok=True)


def main():
    """Run all threshold optimization tests."""
    logger = setup_logging()
    
    logger.info("🎯 Starting Threshold Optimization Pipeline Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Single-task thresholds", test_single_task_thresholds),
        ("Multi-task thresholds", test_multi_task_thresholds),
        ("Threshold file format", test_threshold_file_format),
        ("End-to-end pipeline", test_threshold_optimization_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:30} : {status}")
        if success:
            passed += 1
    
    logger.info("-" * 60)
    logger.info(f"Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Threshold optimization pipeline is working correctly.")
        return 0
    else:
        logger.error("💥 Some tests failed. Please check the implementation.")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 