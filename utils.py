"""
Utility functions for MVFouls model training and evaluation.
"""

import torch
import numpy as np
from typing import Union, Dict, List, Optional, Tuple
from collections import Counter, OrderedDict
import warnings

# Import task metadata from dataset
try:
    from dataset import TASKS_INFO, LABEL2IDX, IDX2LABEL, N_TASKS
except ImportError:
    # Fallback if dataset module is not available
    TASKS_INFO = None
    LABEL2IDX = None
    IDX2LABEL = None
    N_TASKS = None


def compute_class_weights(
    labels: Union[torch.Tensor, np.ndarray, List],
    method: str = 'balanced',
    beta: float = 0.9999,
    smooth: float = 1e-7,
    return_dict: bool = False,
    num_classes: Optional[int] = None
) -> Union[torch.Tensor, Dict[int, float]]:
    """
    Compute class weights for imbalanced datasets using various strategies.
    
    Args:
        labels: Ground truth labels (1D array/tensor/list)
        method: Weight computation method:
            - 'balanced': sklearn-style balanced weights (default)
            - 'inverse': Simple inverse frequency weights
            - 'sqrt_inverse': Square root of inverse frequency
            - 'effective': Effective number of samples (for extreme imbalance)
            - 'focal': Weights designed for focal loss
        beta: Beta parameter for effective number method (default: 0.9999)
        smooth: Smoothing factor to avoid division by zero (default: 1e-7)
        return_dict: If True, return dict mapping class_id -> weight
        num_classes: Expected number of classes (if None, inferred from labels)
        
    Returns:
        torch.Tensor or Dict: Class weights in same order as unique classes
        
    Examples:
        >>> labels = [0, 0, 0, 1, 1, 2]  # Imbalanced: 3 class-0, 2 class-1, 1 class-2
        >>> weights = compute_class_weights(labels, method='balanced')
        >>> print(weights)  # tensor([0.67, 1.00, 2.00])
        
        >>> weights_dict = compute_class_weights(labels, method='balanced', return_dict=True)
        >>> print(weights_dict)  # {0: 0.67, 1: 1.0, 2: 2.0}
    """
    # Convert to numpy for easier handling
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels_np = np.array(labels)
    else:
        labels_np = labels
    
    # Flatten if needed
    labels_np = labels_np.flatten()
    
    # Get class counts
    unique_classes, counts = np.unique(labels_np, return_counts=True)
    n_samples = len(labels_np)
    
    # Handle expected number of classes
    if num_classes is not None:
        # Ensure we have counts for all expected classes
        full_counts = np.zeros(num_classes, dtype=int)
        for cls, count in zip(unique_classes, counts):
            if cls < num_classes:
                full_counts[cls] = count
        
        # Update variables to use full class range
        unique_classes = np.arange(num_classes)
        counts = full_counts
        n_classes = num_classes
    else:
        n_classes = len(unique_classes)
    
    # Compute weights based on method
    if method == 'balanced':
        # sklearn-style balanced weights: n_samples / (n_classes * class_count)
        # For zero counts, use a small value to avoid division by zero
        safe_counts = np.where(counts == 0, smooth, counts)
        weights = n_samples / (n_classes * safe_counts)
        
    elif method == 'inverse':
        # Simple inverse frequency
        weights = 1.0 / (counts + smooth)
        
    elif method == 'sqrt_inverse':
        # Square root of inverse frequency (less aggressive)
        weights = 1.0 / np.sqrt(counts + smooth)
        
    elif method == 'effective':
        # Effective number of samples for extreme imbalance
        # Paper: "Class-Balanced Loss Based on Effective Number of Samples"
        # For zero counts, use minimal effective number
        safe_counts = np.where(counts == 0, 1, counts)
        effective_num = 1.0 - np.power(beta, safe_counts)
        weights = (1.0 - beta) / (effective_num + smooth)
        
    elif method == 'focal':
        # Weights designed to work well with focal loss
        # Less aggressive than inverse, more than sqrt_inverse
        safe_counts = np.where(counts == 0, smooth * n_samples, counts)
        freq = safe_counts / n_samples
        weights = 1.0 / np.power(freq + smooth, 0.25)
        
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'balanced', 'inverse', 'sqrt_inverse', 'effective', 'focal'")
    
    # Normalize weights so they sum to n_classes (maintains relative importance)
    weights = weights / weights.mean()
    
    if return_dict:
        return {int(cls): float(weight) for cls, weight in zip(unique_classes, weights)}
    else:
        # Return as tensor in order of unique classes
        return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights_from_dataset(
    dataset,
    label_key: str = 'label',
    method: str = 'balanced',
    **kwargs
) -> torch.Tensor:
    """
    Compute class weights directly from a dataset object.
    
    Args:
        dataset: Dataset object with indexable items
        label_key: Key to extract labels from dataset items (for dict-like items)
        method: Weight computation method (see compute_class_weights)
        **kwargs: Additional arguments passed to compute_class_weights
        
    Returns:
        torch.Tensor: Class weights
    """
    labels = []
    
    for item in dataset:
        if isinstance(item, (tuple, list)):
            # Assume (data, label) format
            label = item[1]
        elif isinstance(item, dict):
            # Dict format
            label = item[label_key]
        else:
            # Single label
            label = item
            
        if isinstance(label, torch.Tensor):
            label = label.item()
            
        labels.append(label)
    
    return compute_class_weights(labels, method=method, **kwargs)


def analyze_class_distribution(
    labels: Union[torch.Tensor, np.ndarray, List],
    class_names: Optional[List[str]] = None,
    print_stats: bool = True
) -> Dict:
    """
    Analyze class distribution and return statistics.
    
    Args:
        labels: Ground truth labels
        class_names: Optional list of class names for pretty printing
        print_stats: Whether to print statistics
        
    Returns:
        Dict: Statistics including counts, frequencies, imbalance ratios
    """
    # Convert to numpy
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels_np = np.array(labels)
    else:
        labels_np = labels
    
    labels_np = labels_np.flatten()
    
    # Get counts
    unique_classes, counts = np.unique(labels_np, return_counts=True)
    n_samples = len(labels_np)
    
    # Compute statistics
    frequencies = counts / n_samples
    max_count = counts.max()
    imbalance_ratios = max_count / counts
    
    stats = {
        'n_samples': n_samples,
        'n_classes': len(unique_classes),
        'classes': unique_classes.tolist(),
        'counts': counts.tolist(),
        'frequencies': frequencies.tolist(),
        'imbalance_ratios': imbalance_ratios.tolist(),
        'max_imbalance': imbalance_ratios.max(),
        'min_frequency': frequencies.min(),
        'max_frequency': frequencies.max()
    }
    
    if print_stats:
        print("Class Distribution Analysis:")
        print("=" * 50)
        print(f"Total samples: {n_samples}")
        print(f"Number of classes: {len(unique_classes)}")
        print(f"Max imbalance ratio: {imbalance_ratios.max():.2f}:1")
        print()
        
        for i, (cls, count, freq, ratio) in enumerate(zip(unique_classes, counts, frequencies, imbalance_ratios)):
            class_name = class_names[i] if class_names else f"Class {cls}"
            print(f"{class_name:12} | Count: {count:6} | Freq: {freq:6.3f} | Ratio: {ratio:6.2f}:1")
    
    return stats


def get_recommended_loss_config(
    labels: Union[torch.Tensor, np.ndarray, List],
    severity_threshold: float = 5.0,
    num_classes: Optional[int] = None
) -> Dict:
    """
    Get recommended loss configuration based on class imbalance severity.
    
    Args:
        labels: Ground truth labels
        severity_threshold: Imbalance ratio threshold for recommendations
        num_classes: Expected number of classes (if None, inferred from labels)
        
    Returns:
        Dict: Recommended configuration with loss_type, class_weights, etc.
    """
    stats = analyze_class_distribution(labels, print_stats=False)
    max_imbalance = stats['max_imbalance']
    
    if max_imbalance < 2.0:
        # Balanced dataset
        config = {
            'loss_type': 'ce',
            'class_weights': None,
            'label_smoothing': 0.0,
            'focal_gamma': 0.0,
            'recommendation': 'Dataset is well balanced. Standard cross-entropy should work well.'
        }
    elif max_imbalance < severity_threshold:
        # Moderate imbalance
        weights = compute_class_weights(labels, method='balanced', num_classes=num_classes)
        config = {
            'loss_type': 'ce',
            'class_weights': weights,
            'label_smoothing': 0.1,
            'focal_gamma': 0.0,
            'recommendation': f'Moderate imbalance ({max_imbalance:.1f}:1). Use balanced class weights with CE loss.'
        }
    else:
        # Severe imbalance
        weights = compute_class_weights(labels, method='effective', num_classes=num_classes)
        config = {
            'loss_type': 'focal',
            'class_weights': weights,
            'label_smoothing': 0.0,
            'focal_gamma': 2.0,
            'recommendation': f'Severe imbalance ({max_imbalance:.1f}:1). Use focal loss with effective number weights.'
        }
    
    return config


def compare_weighting_methods(
    labels: Union[torch.Tensor, np.ndarray, List],
    methods: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Compare different class weighting methods side by side.
    
    Args:
        labels: Ground truth labels
        methods: List of methods to compare (default: all methods)
        
    Returns:
        Dict: Method name -> class weights tensor
    """
    if methods is None:
        methods = ['balanced', 'inverse', 'sqrt_inverse', 'effective', 'focal']
    
    results = {}
    
    print("Class Weighting Methods Comparison:")
    print("=" * 60)
    
    # Get class distribution first
    stats = analyze_class_distribution(labels, print_stats=False)
    unique_classes = stats['classes']
    counts = stats['counts']
    
    print(f"Class distribution: {dict(zip(unique_classes, counts))}")
    print()
    
    for method in methods:
        try:
            weights = compute_class_weights(labels, method=method)
            results[method] = weights
            
            # Pretty print
            weight_str = ", ".join([f"{w:.3f}" for w in weights])
            print(f"{method:12} | Weights: [{weight_str}]")
            
        except Exception as e:
            print(f"{method:12} | Error: {e}")
            warnings.warn(f"Failed to compute weights with method '{method}': {e}")
    
    return results


# Convenience function for MVFouls head integration
def get_mvfouls_class_weights(
    labels: Union[torch.Tensor, np.ndarray, List],
    num_classes: int = 2,
    method: str = 'auto'
) -> torch.Tensor:
    """
    Get class weights specifically optimized for MVFouls binary/multi-class classification.
    
    Args:
        labels: Ground truth labels (0: no foul, 1: foul, 2: severe foul, etc.)
        num_classes: Number of classes (2 for binary, 3+ for multi-class)
        method: 'auto' for automatic selection, or specific method name
        
    Returns:
        torch.Tensor: Class weights ready for MVFoulsHead
    """
    if method == 'auto':
        # Auto-select method based on imbalance severity
        config = get_recommended_loss_config(labels)
        if config['class_weights'] is not None:
            return config['class_weights']
        else:
            # Return uniform weights for balanced case
            return torch.ones(num_classes, dtype=torch.float32)
    else:
        return compute_class_weights(labels, method=method)


# Multi-task MVFouls head utilities
def get_task_metadata() -> Dict:
    """
    Get comprehensive metadata for all MVFouls tasks.
    
    This is the single source of truth for task configuration in the multi-task head.
    Pulls information from the canonical TASKS_INFO defined in dataset.py.
    
    Returns:
        Dict containing:
            - task_names: List of task names in canonical order
            - num_classes: List of number of classes per task
            - total_tasks: Total number of tasks
            - offsets: Cumulative offsets for concatenated logits (useful for ONNX export)
            - class_names: Dict mapping task names to their class label lists
            - label_to_idx: Dict mapping task names to label->index mappings
            - idx_to_label: Dict mapping task names to index->label mappings
            
    Raises:
        RuntimeError: If dataset module cannot be imported or TASKS_INFO is not available
        
    Example:
        >>> metadata = get_task_metadata()
        >>> print(metadata['task_names'])
        ['action_class', 'severity', 'offence', ...]
        >>> print(metadata['num_classes'])
        [10, 6, 4, ...]
        >>> print(metadata['offsets'])
        [0, 10, 16, 20, ...]  # Cumulative sum for concatenation
    """
    if TASKS_INFO is None:
        raise RuntimeError(
            "Cannot import task metadata from dataset module. "
            "Make sure dataset.py is available and contains TASKS_INFO."
        )
    
    # Extract basic info
    task_names = list(TASKS_INFO.keys())
    num_classes = [len(TASKS_INFO[task]) for task in task_names]
    total_tasks = len(task_names)
    
    # Compute offsets for concatenated logits (useful for ONNX export and decoding)
    offsets = np.cumsum([0] + num_classes).tolist()  # [0, n1, n1+n2, n1+n2+n3, ...]
    
    # Comprehensive metadata
    metadata = {
        'task_names': task_names,
        'num_classes': num_classes,
        'total_tasks': total_tasks,
        'total_classes': sum(num_classes),  # Total classes across all tasks
        'offsets': offsets,
        'class_names': dict(TASKS_INFO),  # Copy to avoid mutation
        'label_to_idx': dict(LABEL2IDX) if LABEL2IDX is not None else {},
        'idx_to_label': dict(IDX2LABEL) if IDX2LABEL is not None else {},
    }
    
    return metadata


def get_task_class_weights(dataset_stats: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    """
    Get class weights for all tasks based on dataset statistics.
    
    Args:
        dataset_stats: Optional dataset statistics from dataset.get_task_statistics()
                      If None, returns uniform weights for all tasks
                      
    Returns:
        Dict mapping task names to class weight tensors
        
    Example:
        >>> # From dataset
        >>> dataset = MVFoulsDataset(...)
        >>> stats = dataset.get_task_statistics()
        >>> weights = get_task_class_weights(stats)
        >>> print(weights['action_class'])  # tensor([1.2, 0.8, 2.1, ...])
    """
    metadata = get_task_metadata()
    task_weights = {}
    
    for task_name, num_cls in zip(metadata['task_names'], metadata['num_classes']):
        if dataset_stats and task_name in dataset_stats:
            # Use computed class weights from dataset statistics
            class_counts = dataset_stats[task_name]['class_counts']
            weights = compute_class_weights(
                # Create dummy labels based on counts for weight computation
                labels=np.repeat(np.arange(len(class_counts)), class_counts),
                method='balanced'
            )
        else:
            # Uniform weights if no statistics available
            weights = torch.ones(num_cls, dtype=torch.float32)
        
        task_weights[task_name] = weights
    
    return task_weights


def concat_task_logits(logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Concatenate per-task logits into a single tensor for ONNX export or unified processing.
    
    Args:
        logits_dict: Dict mapping task names to logit tensors (B, num_classes_i)
        
    Returns:
        torch.Tensor: Concatenated logits of shape (B, total_classes)
        
    Example:
        >>> logits = {'action_class': torch.randn(4, 10), 'severity': torch.randn(4, 6)}
        >>> concat_logits = concat_task_logits(logits)
        >>> print(concat_logits.shape)  # torch.Size([4, 16])
    """
    metadata = get_task_metadata()
    
    # Ensure logits are in canonical task order
    ordered_logits = []
    for task_name in metadata['task_names']:
        if task_name not in logits_dict:
            raise KeyError(f"Missing logits for task '{task_name}'")
        ordered_logits.append(logits_dict[task_name])
    
    return torch.cat(ordered_logits, dim=1)


def split_concat_logits(concat_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Split concatenated logits back into per-task logits.
    
    Args:
        concat_logits: Concatenated logits tensor of shape (B, total_classes)
        
    Returns:
        Dict mapping task names to individual logit tensors
        
    Example:
        >>> concat_logits = torch.randn(4, 16)  # 10 + 6 classes
        >>> logits_dict = split_concat_logits(concat_logits)
        >>> print(logits_dict['action_class'].shape)  # torch.Size([4, 10])
    """
    metadata = get_task_metadata()
    logits_dict = {}
    
    for i, task_name in enumerate(metadata['task_names']):
        start_idx = metadata['offsets'][i]
        end_idx = metadata['offsets'][i + 1]
        logits_dict[task_name] = concat_logits[:, start_idx:end_idx]
    
    return logits_dict


# Testing and example functions
def test_class_weights():
    """Test the class weights computation with various scenarios."""
    print("Testing Class Weights Computation")
    print("=" * 50)
    
    # Test case 1: Moderate imbalance
    print("\n1. Moderate Imbalance (Binary):")
    labels1 = [0] * 70 + [1] * 30  # 70% class 0, 30% class 1
    analyze_class_distribution(labels1, class_names=['No Foul', 'Foul'])
    weights1 = compute_class_weights(labels1, method='balanced')
    print(f"Balanced weights: {weights1}")
    
    # Test case 2: Severe imbalance
    print("\n2. Severe Imbalance (Multi-class):")
    labels2 = [0] * 85 + [1] * 12 + [2] * 3  # 85%, 12%, 3%
    analyze_class_distribution(labels2, class_names=['No Foul', 'Minor Foul', 'Major Foul'])
    compare_weighting_methods(labels2, methods=['balanced', 'effective', 'focal'])
    
    # Test case 3: Recommendation system
    print("\n3. Automatic Recommendations:")
    config = get_recommended_loss_config(labels2)
    print(f"Recommended config: {config['recommendation']}")
    print(f"Loss type: {config['loss_type']}")
    if config['class_weights'] is not None:
        print(f"Class weights: {config['class_weights']}")


def test_task_metadata():
    """Test task metadata functions."""
    print("Testing Task Metadata Functions")
    print("=" * 50)
    
    try:
        # Test basic metadata extraction
        metadata = get_task_metadata()
        
        print(f"✓ Found {metadata['total_tasks']} tasks")
        print(f"✓ Task names: {metadata['task_names'][:3]}...")  # Show first 3
        print(f"✓ Num classes: {metadata['num_classes'][:3]}...")
        print(f"✓ Total classes: {metadata['total_classes']}")
        print(f"✓ Offsets: {metadata['offsets'][:4]}...")  # Show first 4
        
        # Verify all expected keys are present
        expected_keys = ['task_names', 'num_classes', 'total_tasks', 'total_classes', 
                        'offsets', 'class_names', 'label_to_idx', 'idx_to_label']
        for key in expected_keys:
            assert key in metadata, f"Missing key: {key}"
        print(f"✓ All expected keys present: {expected_keys}")
        
        # Test concatenation and splitting
        batch_size = 4
        dummy_logits = {}
        for task_name, num_cls in zip(metadata['task_names'], metadata['num_classes']):
            dummy_logits[task_name] = torch.randn(batch_size, num_cls)
        
        # Test concatenation
        concat_logits = concat_task_logits(dummy_logits)
        expected_shape = (batch_size, metadata['total_classes'])
        assert concat_logits.shape == expected_shape, f"Expected {expected_shape}, got {concat_logits.shape}"
        print(f"✓ Concatenation: {concat_logits.shape}")
        
        # Test splitting
        split_logits = split_concat_logits(concat_logits)
        assert len(split_logits) == metadata['total_tasks'], f"Expected {metadata['total_tasks']} tasks in split"
        for task_name, expected_cls in zip(metadata['task_names'], metadata['num_classes']):
            actual_shape = split_logits[task_name].shape
            expected_shape = (batch_size, expected_cls)
            assert actual_shape == expected_shape, f"Task {task_name}: expected {expected_shape}, got {actual_shape}"
        print(f"✓ Splitting: recovered {len(split_logits)} tasks")
        
        # Test class weights
        task_weights = get_task_class_weights()
        assert len(task_weights) == metadata['total_tasks'], f"Expected {metadata['total_tasks']} weight tensors"
        for task_name, expected_cls in zip(metadata['task_names'], metadata['num_classes']):
            actual_len = len(task_weights[task_name])
            assert actual_len == expected_cls, f"Task {task_name}: expected {expected_cls} weights, got {actual_len}"
        print(f"✓ Class weights: {len(task_weights)} tasks configured")
        
        # Test with dataset statistics (mock)
        mock_stats = {
            metadata['task_names'][0]: {  # First task
                'class_counts': [50, 30, 20]  # Mock counts
            }
        }
        weighted_task_weights = get_task_class_weights(mock_stats)
        first_task = metadata['task_names'][0]
        print(f"✓ Dataset stats integration: {first_task} weights = {weighted_task_weights[first_task]}")
        
        print("✓ All task metadata tests passed!")
        
    except Exception as e:
        print(f"✗ Task metadata test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def compute_task_metrics(
    logits_dict: Dict[str, torch.Tensor], 
    targets_dict: Dict[str, torch.Tensor],
    task_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive metrics for each task.
    
    Args:
        logits_dict: Dict mapping task names to logits tensors (B, num_classes)
        targets_dict: Dict mapping task names to target tensors (B,)
        task_names: Optional list of task names to compute metrics for
        
    Returns:
        Dict mapping task names to metrics dict with keys:
        - accuracy, precision, recall, f1_score, top1_acc, top3_acc (if applicable)
    """
    if task_names is None:
        task_names = list(logits_dict.keys())
    
    metrics = {}
    
    for task_name in task_names:
        if task_name not in logits_dict or task_name not in targets_dict:
            continue
            
        logits = logits_dict[task_name]  # (B, num_classes)
        targets = targets_dict[task_name]  # (B,)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)  # (B,)
        probs = torch.softmax(logits, dim=1)  # (B, num_classes)
        
        num_classes = logits.shape[1]
        batch_size = logits.shape[0]
        
        # Basic accuracy
        correct = (preds == targets).float()
        accuracy = correct.mean().item()
        
        # Per-class metrics
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        for cls in range(num_classes):
            # True positives, false positives, false negatives
            tp = ((preds == cls) & (targets == cls)).float().sum().item()
            fp = ((preds == cls) & (targets != cls)).float().sum().item()
            fn = ((preds != cls) & (targets == cls)).float().sum().item()
            
            # Precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)
        
        # Macro averages
        macro_precision = sum(precision_per_class) / len(precision_per_class)
        macro_recall = sum(recall_per_class) / len(recall_per_class)
        macro_f1 = sum(f1_per_class) / len(f1_per_class)
        
        # Top-k accuracy (if more than 3 classes)
        top1_acc = accuracy
        top3_acc = None
        if num_classes >= 3:
            _, top3_preds = torch.topk(probs, min(3, num_classes), dim=1)
            top3_correct = torch.any(top3_preds == targets.unsqueeze(1), dim=1).float()
            top3_acc = top3_correct.mean().item()
        
        # Store metrics
        task_metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'top1_acc': top1_acc,
        }
        
        if top3_acc is not None:
            task_metrics['top3_acc'] = top3_acc
            
        # Add per-class metrics
        task_metrics['precision_per_class'] = precision_per_class
        task_metrics['recall_per_class'] = recall_per_class
        task_metrics['f1_per_class'] = f1_per_class
        
        metrics[task_name] = task_metrics
    
    return metrics


def compute_confusion_matrices(
    logits_dict: Dict[str, torch.Tensor],
    targets_dict: Dict[str, torch.Tensor],
    task_names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute confusion matrices for each task.
    
    Args:
        logits_dict: Dict mapping task names to logits tensors
        targets_dict: Dict mapping task names to target tensors
        task_names: Optional list of task names
        
    Returns:
        Dict mapping task names to confusion matrices
    """
    if task_names is None:
        task_names = list(logits_dict.keys())
    
    confusion_matrices = {}
    
    for task_name in task_names:
        if task_name not in logits_dict or task_name not in targets_dict:
            continue
            
        logits = logits_dict[task_name]
        targets = targets_dict[task_name]
        
        preds = torch.argmax(logits, dim=1)
        num_classes = logits.shape[1]
        
        # Compute confusion matrix
        cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for t, p in zip(targets.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
            
        confusion_matrices[task_name] = cm
    
    return confusion_matrices


def compute_task_weights_from_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    weighting_strategy: str = 'inverse_accuracy'
) -> Dict[str, float]:
    """
    Compute task weights based on performance metrics.
    
    Args:
        metrics_dict: Dict from compute_task_metrics
        weighting_strategy: Strategy for computing weights
            - 'inverse_accuracy': Weight inversely proportional to accuracy
            - 'inverse_f1': Weight inversely proportional to F1 score
            - 'uniform': Equal weights for all tasks
            - 'difficulty': Based on number of classes and performance
            
    Returns:
        Dict mapping task names to weights
    """
    task_names = list(metrics_dict.keys())
    
    if weighting_strategy == 'uniform':
        return {task: 1.0 for task in task_names}
    
    weights = {}
    
    if weighting_strategy == 'inverse_accuracy':
        for task_name in task_names:
            acc = metrics_dict[task_name]['accuracy']
            # Inverse accuracy with smoothing
            weights[task_name] = 1.0 / (acc + 0.1)  # Add small epsilon
            
    elif weighting_strategy == 'inverse_f1':
        for task_name in task_names:
            f1 = metrics_dict[task_name]['macro_f1']
            weights[task_name] = 1.0 / (f1 + 0.1)
            
    elif weighting_strategy == 'difficulty':
        # Combine number of classes and performance
        metadata = get_task_metadata()
        for i, task_name in enumerate(task_names):
            if task_name in metadata['task_names']:
                task_idx = metadata['task_names'].index(task_name)
                num_classes = metadata['num_classes'][task_idx]
                acc = metrics_dict[task_name]['accuracy']
                
                # Weight based on difficulty (more classes = harder, lower acc = harder)
                class_difficulty = num_classes / 10.0  # Normalize by max classes
                acc_difficulty = 1.0 - acc
                weights[task_name] = class_difficulty + acc_difficulty
            else:
                weights[task_name] = 1.0
    else:
        raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")
    
    # Normalize weights to sum to number of tasks
    total_weight = sum(weights.values())
    num_tasks = len(weights)
    for task_name in weights:
        weights[task_name] = weights[task_name] / total_weight * num_tasks
    
    return weights


def format_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    precision: int = 4
) -> str:
    """
    Format metrics as a readable table string.
    
    Args:
        metrics_dict: Dict from compute_task_metrics
        precision: Number of decimal places
        
    Returns:
        Formatted table string
    """
    if not metrics_dict:
        return "No metrics available"
    
    # Get all metric keys
    all_keys = set()
    for task_metrics in metrics_dict.values():
        all_keys.update(task_metrics.keys())
    
    # Remove per-class metrics from main table
    main_keys = [k for k in all_keys if not k.endswith('_per_class')]
    main_keys = sorted(main_keys)
    
    # Create table
    lines = []
    
    # Header
    header = f"{'Task':<20} " + " ".join(f"{key:>12}" for key in main_keys)
    lines.append(header)
    lines.append("-" * len(header))
    
    # Rows
    for task_name in sorted(metrics_dict.keys()):
        task_metrics = metrics_dict[task_name]
        row = f"{task_name:<20} "
        
        for key in main_keys:
            if key in task_metrics:
                value = task_metrics[key]
                if value is None:
                    row += f"{'N/A':>12} "
                else:
                    row += f"{value:>12.{precision}f} "
            else:
                row += f"{'N/A':>12} "
        
        lines.append(row.rstrip())
    
    return "\n".join(lines)


def compute_overall_metrics(metrics_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Compute overall metrics across all tasks.
    
    Args:
        metrics_dict: Dict from compute_task_metrics
        
    Returns:
        Dict with overall metrics (mean, std, etc.)
    """
    if not metrics_dict:
        return {}
    
    # Collect all metric values
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    for task_metrics in metrics_dict.values():
        all_accuracies.append(task_metrics['accuracy'])
        all_precisions.append(task_metrics['macro_precision'])
        all_recalls.append(task_metrics['macro_recall'])
        all_f1s.append(task_metrics['macro_f1'])
    
    # Convert to tensors for easy computation
    accuracies = torch.tensor(all_accuracies)
    precisions = torch.tensor(all_precisions)
    recalls = torch.tensor(all_recalls)
    f1s = torch.tensor(all_f1s)
    
    overall_metrics = {
        'mean_accuracy': accuracies.mean().item(),
        'std_accuracy': accuracies.std().item(),
        'mean_precision': precisions.mean().item(),
        'std_precision': precisions.std().item(),
        'mean_recall': recalls.mean().item(),
        'std_recall': recalls.std().item(),
        'mean_f1': f1s.mean().item(),
        'std_f1': f1s.std().item(),
        'min_accuracy': accuracies.min().item(),
        'max_accuracy': accuracies.max().item(),
    }
    
    return overall_metrics


def make_weighted_sampler_from_metrics(dataset, metrics, task='action_class',
                                       power=1.0, min_weight=0.1, max_weight=10.0):
    """
    Build per-sample weights inversely proportional to per-class recall.
    `power` controls aggressiveness (1 = inverse, 0.5 = sqrt-inverse, …).
    """
    if task not in metrics:
        print(f"Warning: Task '{task}' not in metrics. Cannot create weighted sampler.")
        return None

    recall_per_class = metrics[task].get('recall_per_class')
    if recall_per_class is None:
        print(f"Warning: Recall not found for task '{task}'. Cannot create weighted sampler.")
        return None

    class_weights = {}
    for i, recall in enumerate(recall_per_class):
        weight = (1 / (recall + 1e-6)) ** power
        weight = max(min_weight, min(max_weight, weight))
        class_weights[i] = weight
        print(f"Dynamic sampler: class-{i} weight = {weight:.4f}")

    # Assumes dataset has a 'get_labels_for_task' method
    try:
        labels = dataset.get_labels_for_task(task)
    except (ValueError, RuntimeError) as e:
        print(f"Warning: Could not get labels for task '{task}'. Sampler not updated. Error: {e}")
        return None
        
    sample_weights = torch.tensor([class_weights[label] for label in labels])

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


if __name__ == "__main__":
    test_class_weights()
    print()
    test_task_metadata()
