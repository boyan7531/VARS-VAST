#!/usr/bin/env python3
"""
Demonstration of Adaptive Loss Features for MVFouls Training

This script shows how to use the new adaptive loss configuration features:
1. Per-task loss types (CE, Focal, BCE)
2. Effective number class weights
3. Adaptive task weighting
4. Unified loss computation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.mvfouls_model import build_multi_task_model
from utils import get_task_metadata, compute_class_weights, compute_task_weights_from_metrics
import torch
import numpy as np


def demo_loss_types():
    """Demonstrate different loss type configurations."""
    print("🎯 Demo: Per-Task Loss Types")
    print("=" * 50)
    
    # Example 1: Mixed loss types
    print("\n1. Mixed Loss Configuration:")
    print("   - action_class: Cross-Entropy (balanced classes)")
    print("   - severity: Focal Loss (imbalanced, hard examples)")
    print("   - offence: Cross-Entropy (straightforward)")
    
    model = build_multi_task_model(
        backbone_pretrained=False,
        loss_types_per_task=['ce', 'focal', 'ce']
    )
    print(f"   ✓ Model configured with: {model.head.loss_types_per_task}")
    
    # Example 2: All focal loss
    print("\n2. All Focal Loss (for very imbalanced dataset):")
    model_focal = build_multi_task_model(
        backbone_pretrained=False,
        loss_types_per_task=['focal', 'focal', 'focal']
    )
    print(f"   ✓ Model configured with: {model_focal.head.loss_types_per_task}")


def demo_effective_weights():
    """Demonstrate effective number class weights."""
    print("\n\n🎯 Demo: Effective Number Class Weights")
    print("=" * 50)
    
    # Simulate extreme imbalance scenario
    print("\n1. Extreme Imbalance Scenario:")
    print("   Class distribution: [10000, 500, 50, 5] samples")
    
    # Create synthetic imbalanced data
    class_counts = np.array([10000, 500, 50, 5])
    synthetic_labels = np.repeat(np.arange(4), class_counts)
    
    # Compare different weighting methods
    balanced_weights = compute_class_weights(synthetic_labels, method='balanced')
    effective_weights = compute_class_weights(synthetic_labels, method='effective')
    
    print(f"\n   Balanced weights:  {balanced_weights.numpy()}")
    print(f"   Effective weights: {effective_weights.numpy()}")
    print(f"   Ratio (effective/balanced): {(effective_weights / balanced_weights).numpy()}")
    
    print("\n   🔍 Notice how effective weights provide more extreme weighting")
    print("       for the rarest classes, helping with severe imbalance.")


def demo_adaptive_weighting():
    """Demonstrate adaptive task weighting strategies."""
    print("\n\n🎯 Demo: Adaptive Task Weighting")
    print("=" * 50)
    
    # Simulate validation metrics for 3 tasks
    print("\n1. Simulated Validation Metrics:")
    mock_metrics = {
        'action_class': {
            'accuracy': 0.85,
            'macro_f1': 0.82,
            'macro_precision': 0.84,
            'macro_recall': 0.81
        },
        'severity': {
            'accuracy': 0.45,  # Performing poorly
            'macro_f1': 0.38,
            'macro_precision': 0.42,
            'macro_recall': 0.35
        },
        'offence': {
            'accuracy': 0.72,
            'macro_f1': 0.69,
            'macro_precision': 0.71,
            'macro_recall': 0.67
        }
    }
    
    # Print metrics table
    print("   Task Performance:")
    for task, metrics in mock_metrics.items():
        print(f"     {task:12}: Acc={metrics['accuracy']:.3f}, F1={metrics['macro_f1']:.3f}")
    
    # Demonstrate different weighting strategies
    strategies = ['uniform', 'inverse_accuracy', 'inverse_f1', 'difficulty']
    
    print(f"\n2. Adaptive Weighting Strategies:")
    for strategy in strategies:
        weights = compute_task_weights_from_metrics(mock_metrics, strategy)
        print(f"   {strategy:20}: {dict(weights)}")
    
    print(f"\n   🔍 Notice how 'severity' gets higher weight due to poor performance")


def demo_unified_loss():
    """Demonstrate unified loss computation."""
    print("\n\n🎯 Demo: Unified Loss Computation")
    print("=" * 50)
    
    # Create a model with unified loss capability
    model = build_multi_task_model(
        backbone_pretrained=False,
        loss_types_per_task=['ce', 'focal', 'ce']
    )
    
    if hasattr(model.head, 'compute_unified_loss'):
        print("\n1. Standard vs Unified Loss:")
        
        # Create dummy data
        batch_size = 4
        metadata = get_task_metadata()
        
        dummy_logits = {}
        dummy_targets = {}
        for task_name, num_classes in zip(metadata['task_names'], metadata['num_classes']):
            dummy_logits[task_name] = torch.randn(batch_size, num_classes)
            dummy_targets[task_name] = torch.randint(0, num_classes, (batch_size,))
        
        # Standard loss computation
        standard_loss = model.compute_loss(dummy_logits, dummy_targets, return_dict=True)
        print(f"   Standard loss: {standard_loss['total_loss']:.4f}")
        
        # Unified loss with adaptive weighting
        unified_loss = model.head.compute_unified_loss(
            dummy_logits,
            dummy_targets,
            weighting_strategy='uniform',
            focal_gamma=2.0,
            adaptive_weights=False
        )
        print(f"   Unified loss:  {unified_loss['total_loss']:.4f}")
        
        # Show per-task losses
        print(f"\n   Per-task losses:")
        for task in metadata['task_names']:
            if f'{task}_loss' in unified_loss:
                print(f"     {task:12}: {unified_loss[f'{task}_loss']:.4f}")
    else:
        print("   ⚠️  Unified loss not available in this model")


def demo_command_line_usage():
    """Show command line usage examples."""
    print("\n\n🎯 Demo: Command Line Usage Examples")
    print("=" * 50)
    
    examples = [
        {
            "name": "Basic Mixed Loss Types",
            "command": "python train_with_class_weights.py --multi-task --loss-types action_class ce severity focal offence ce [other args...]",
            "description": "Use CE for action_class/offence, focal for severity"
        },
        {
            "name": "Effective Weights + Adaptive",
            "command": "python train_with_class_weights.py --multi-task --effective-class-weights --adaptive-weights --weighting-strategy inverse_accuracy [other args...]",
            "description": "Use effective number weights with adaptive task weighting"
        },
        {
            "name": "Full Advanced Configuration",
            "command": "python train_with_class_weights.py --multi-task --loss-types action_class ce severity focal offence ce --effective-class-weights --adaptive-weights --weighting-strategy difficulty [other args...]",
            "description": "Complete adaptive configuration with per-task losses"
        },
        {
            "name": "Simple Balanced Sampling",
            "command": "python train_with_class_weights.py --multi-task --balanced-sampling --disable-class-weights [other args...]",
            "description": "Use only balanced sampling, no loss weights"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}:")
        print(f"   {example['description']}")
        print(f"   Command:")
        print(f"   {example['command']}")


def main():
    """Run all demonstrations."""
    print("🚀 MVFouls Adaptive Loss Features Demo")
    print("=" * 60)
    print("This demo shows the new adaptive loss capabilities:")
    print("• Per-task loss type configuration")
    print("• Effective number class weights for extreme imbalance")
    print("• Adaptive task weighting based on performance")
    print("• Unified loss computation with flexible strategies")
    
    try:
        demo_loss_types()
        demo_effective_weights()
        demo_adaptive_weighting()
        demo_unified_loss()
        demo_command_line_usage()
        
        print("\n\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("• Try the new features with your dataset")
        print("• Run the test suite: python tests/test_adaptive_loss.py") 
        print("• Check TRAINING_GUIDE.md for detailed usage instructions")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 