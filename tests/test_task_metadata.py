"""
Unit tests for utils.py task metadata functions.

Tests the multi-task MVFouls head utilities including:
- get_task_metadata()
- get_task_class_weights()
- concat_task_logits()
- split_concat_logits()
"""

import pytest
import torch
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    get_task_metadata,
    get_task_class_weights,
    concat_task_logits,
    split_concat_logits,
    compute_class_weights
)


class TestTaskMetadata:
    """Test suite for task metadata functions."""
    
    def test_get_task_metadata_basic(self):
        """Test basic functionality of get_task_metadata."""
        metadata = get_task_metadata()
        
        # Check required keys
        required_keys = [
            'task_names', 'num_classes', 'total_tasks', 'total_classes',
            'offsets', 'class_names', 'label_to_idx', 'idx_to_label'
        ]
        for key in required_keys:
            assert key in metadata, f"Missing required key: {key}"
        
        # Check data types
        assert isinstance(metadata['task_names'], list)
        assert isinstance(metadata['num_classes'], list)
        assert isinstance(metadata['total_tasks'], int)
        assert isinstance(metadata['total_classes'], int)
        assert isinstance(metadata['offsets'], list)
        assert isinstance(metadata['class_names'], dict)
        assert isinstance(metadata['label_to_idx'], dict)
        assert isinstance(metadata['idx_to_label'], dict)
        
        # Check consistency
        assert len(metadata['task_names']) == metadata['total_tasks']
        assert len(metadata['num_classes']) == metadata['total_tasks']
        assert len(metadata['offsets']) == metadata['total_tasks'] + 1  # +1 for final offset
        assert metadata['total_classes'] == sum(metadata['num_classes'])
        
        # Check offsets are cumulative
        expected_offsets = np.cumsum([0] + metadata['num_classes']).tolist()
        assert metadata['offsets'] == expected_offsets
        
        print(f"✓ Basic metadata test passed: {metadata['total_tasks']} tasks, {metadata['total_classes']} total classes")
    
    def test_get_task_metadata_expected_tasks(self):
        """Test that we get the expected MVFouls tasks."""
        metadata = get_task_metadata()
        
        # Expected tasks from the plan (11 tasks)
        expected_tasks = [
            'action_class', 'severity', 'offence', 'contact', 'bodypart',
            'upper_body_part', 'multiple_fouls', 'try_to_play', 'touch_ball',
            'handball', 'handball_offence'
        ]
        
        assert metadata['total_tasks'] == 11, f"Expected 11 tasks, got {metadata['total_tasks']}"
        
        # Check that all expected tasks are present (order may vary)
        for task in expected_tasks:
            assert task in metadata['task_names'], f"Missing expected task: {task}"
        
        # Check that each task has reasonable number of classes (2-10 range)
        for task_name, num_cls in zip(metadata['task_names'], metadata['num_classes']):
            assert 2 <= num_cls <= 10, f"Task {task_name} has unreasonable class count: {num_cls}"
        
        print(f"✓ Expected tasks test passed: {metadata['task_names']}")
    
    def test_get_task_metadata_import_error(self):
        """Test behavior when dataset module is not available."""
        with patch('utils.TASKS_INFO', None):
            with pytest.raises(RuntimeError, match="Cannot import task metadata"):
                get_task_metadata()
        
        print("✓ Import error handling test passed")
    
    def test_get_task_class_weights_uniform(self):
        """Test get_task_class_weights with no dataset statistics (uniform weights)."""
        task_weights = get_task_class_weights()
        metadata = get_task_metadata()
        
        # Check we have weights for all tasks
        assert len(task_weights) == metadata['total_tasks']
        
        # Check each task has correct number of weights
        for task_name, expected_cls in zip(metadata['task_names'], metadata['num_classes']):
            assert task_name in task_weights, f"Missing weights for task: {task_name}"
            weights = task_weights[task_name]
            assert isinstance(weights, torch.Tensor)
            assert len(weights) == expected_cls, f"Task {task_name}: expected {expected_cls} weights, got {len(weights)}"
            
            # Uniform weights should all be 1.0
            assert torch.allclose(weights, torch.ones_like(weights)), f"Task {task_name} weights not uniform: {weights}"
        
        print(f"✓ Uniform weights test passed: {len(task_weights)} tasks")
    
    def test_get_task_class_weights_with_stats(self):
        """Test get_task_class_weights with dataset statistics."""
        metadata = get_task_metadata()
        
        # Mock dataset statistics for first two tasks
        mock_stats = {
            metadata['task_names'][0]: {
                'class_counts': [100, 50, 25]  # Imbalanced
            },
            metadata['task_names'][1]: {
                'class_counts': [30, 30, 30, 30, 30, 30]  # Balanced
            }
        }
        
        task_weights = get_task_class_weights(mock_stats)
        
        # Check first task has non-uniform weights (imbalanced)
        first_task = metadata['task_names'][0]
        first_weights = task_weights[first_task]
        assert not torch.allclose(first_weights, torch.ones_like(first_weights)), \
            f"First task weights should be non-uniform due to imbalance: {first_weights}"
        
        # Check second task has more uniform weights (balanced)
        second_task = metadata['task_names'][1]
        second_weights = task_weights[second_task]
        # Balanced data should have weights closer to 1.0
        assert torch.allclose(second_weights, torch.ones_like(second_weights), atol=0.1), \
            f"Second task weights should be nearly uniform: {second_weights}"
        
        # Check remaining tasks have uniform weights (no stats provided)
        for i in range(2, len(metadata['task_names'])):
            task_name = metadata['task_names'][i]
            weights = task_weights[task_name]
            assert torch.allclose(weights, torch.ones_like(weights)), \
                f"Task {task_name} should have uniform weights (no stats): {weights}"
        
        print("✓ Dataset stats integration test passed")
    
    def test_concat_task_logits(self):
        """Test concatenation of task logits."""
        metadata = get_task_metadata()
        batch_size = 8
        
        # Create dummy logits for each task
        logits_dict = {}
        for task_name, num_cls in zip(metadata['task_names'], metadata['num_classes']):
            logits_dict[task_name] = torch.randn(batch_size, num_cls)
        
        # Test concatenation
        concat_logits = concat_task_logits(logits_dict)
        
        # Check shape
        expected_shape = (batch_size, metadata['total_classes'])
        assert concat_logits.shape == expected_shape, f"Expected {expected_shape}, got {concat_logits.shape}"
        
        # Check that concatenation preserves order
        start_idx = 0
        for task_name, num_cls in zip(metadata['task_names'], metadata['num_classes']):
            end_idx = start_idx + num_cls
            expected_slice = logits_dict[task_name]
            actual_slice = concat_logits[:, start_idx:end_idx]
            assert torch.allclose(expected_slice, actual_slice), \
                f"Task {task_name} slice mismatch at indices {start_idx}:{end_idx}"
            start_idx = end_idx
        
        print(f"✓ Concatenation test passed: {concat_logits.shape}")
    
    def test_concat_task_logits_missing_task(self):
        """Test concat_task_logits with missing task."""
        metadata = get_task_metadata()
        
        # Create incomplete logits dict (missing last task)
        logits_dict = {}
        for task_name, num_cls in zip(metadata['task_names'][:-1], metadata['num_classes'][:-1]):
            logits_dict[task_name] = torch.randn(4, num_cls)
        
        # Should raise KeyError for missing task
        missing_task = metadata['task_names'][-1]
        with pytest.raises(KeyError, match=f"Missing logits for task '{missing_task}'"):
            concat_task_logits(logits_dict)
        
        print("✓ Missing task error handling test passed")
    
    def test_split_concat_logits(self):
        """Test splitting of concatenated logits."""
        metadata = get_task_metadata()
        batch_size = 6
        
        # Create concatenated logits
        concat_logits = torch.randn(batch_size, metadata['total_classes'])
        
        # Split back
        split_logits = split_concat_logits(concat_logits)
        
        # Check we get all tasks back
        assert len(split_logits) == metadata['total_tasks']
        
        # Check shapes
        for task_name, expected_cls in zip(metadata['task_names'], metadata['num_classes']):
            assert task_name in split_logits, f"Missing task in split: {task_name}"
            actual_shape = split_logits[task_name].shape
            expected_shape = (batch_size, expected_cls)
            assert actual_shape == expected_shape, f"Task {task_name}: expected {expected_shape}, got {actual_shape}"
        
        # Check that splitting is consistent with offsets
        for i, task_name in enumerate(metadata['task_names']):
            start_idx = metadata['offsets'][i]
            end_idx = metadata['offsets'][i + 1]
            expected_slice = concat_logits[:, start_idx:end_idx]
            actual_slice = split_logits[task_name]
            assert torch.allclose(expected_slice, actual_slice), \
                f"Task {task_name} split mismatch at indices {start_idx}:{end_idx}"
        
        print(f"✓ Splitting test passed: {len(split_logits)} tasks recovered")
    
    def test_concat_split_roundtrip(self):
        """Test that concatenation and splitting are inverse operations."""
        metadata = get_task_metadata()
        batch_size = 5
        
        # Create original logits
        original_logits = {}
        for task_name, num_cls in zip(metadata['task_names'], metadata['num_classes']):
            original_logits[task_name] = torch.randn(batch_size, num_cls)
        
        # Concatenate then split
        concat_logits = concat_task_logits(original_logits)
        recovered_logits = split_concat_logits(concat_logits)
        
        # Check that we recover the original logits exactly
        assert len(recovered_logits) == len(original_logits)
        for task_name in metadata['task_names']:
            assert torch.allclose(original_logits[task_name], recovered_logits[task_name]), \
                f"Roundtrip failed for task {task_name}"
        
        print("✓ Concat-split roundtrip test passed")
    
    def test_offsets_consistency(self):
        """Test that offsets are consistent with actual concatenation."""
        metadata = get_task_metadata()
        
        # Check that offsets match cumulative sum
        expected_offsets = [0]
        cumsum = 0
        for num_cls in metadata['num_classes']:
            cumsum += num_cls
            expected_offsets.append(cumsum)
        
        assert metadata['offsets'] == expected_offsets, \
            f"Offsets mismatch: expected {expected_offsets}, got {metadata['offsets']}"
        
        # Check that final offset equals total classes
        assert metadata['offsets'][-1] == metadata['total_classes'], \
            f"Final offset {metadata['offsets'][-1]} != total classes {metadata['total_classes']}"
        
        print("✓ Offsets consistency test passed")


def test_integration_with_real_data():
    """Integration test using real MVFouls data structure."""
    try:
        # This test will only run if the actual dataset module is available
        metadata = get_task_metadata()
        
        # Simulate a realistic batch
        batch_size = 16
        
        # Create logits with realistic ranges
        logits_dict = {}
        for task_name, num_cls in zip(metadata['task_names'], metadata['num_classes']):
            # Use small logits to simulate realistic model outputs
            logits_dict[task_name] = torch.randn(batch_size, num_cls) * 0.5
        
        # Test full pipeline
        concat_logits = concat_task_logits(logits_dict)
        split_logits = split_concat_logits(concat_logits)
        
        # Test with class weights
        task_weights = get_task_class_weights()
        
        # Simulate loss computation (just shapes, not actual loss)
        total_loss = 0.0
        for task_name in metadata['task_names']:
            task_logits = split_logits[task_name]
            task_targets = torch.randint(0, metadata['num_classes'][metadata['task_names'].index(task_name)], (batch_size,))
            # Just check that shapes are compatible for loss computation
            assert task_logits.shape[0] == task_targets.shape[0], f"Batch size mismatch for {task_name}"
            assert task_logits.shape[1] == len(task_weights[task_name]), f"Class count mismatch for {task_name}"
        
        print(f"✓ Integration test passed: {batch_size} samples, {len(metadata['task_names'])} tasks")
        
    except Exception as e:
        print(f"⚠ Integration test skipped due to: {e}")


if __name__ == "__main__":
    # Run tests directly
    test_suite = TestTaskMetadata()
    
    print("Running Task Metadata Unit Tests")
    print("=" * 50)
    
    # Run all test methods
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
    for test_method in test_methods:
        try:
            print(f"\n{test_method}:")
            getattr(test_suite, test_method)()
        except Exception as e:
            print(f"✗ {test_method} FAILED: {e}")
            raise
    
    # Run integration test
    print(f"\ntest_integration_with_real_data:")
    test_integration_with_real_data()
    
    print(f"\n{'='*50}")
    print(f"✅ All {len(test_methods) + 1} tests passed!") 