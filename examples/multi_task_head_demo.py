"""
Demonstration of multi-task MVFoulsHead functionality.

This script shows how to:
1. Create a multi-task head
2. Perform forward passes
3. Compute multi-task losses
4. Switch between single and multi-task modes
"""

import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.head import build_multi_task_head, MVFoulsHead
from utils import get_task_metadata


def demo_multi_task_head():
    """Demonstrate multi-task head functionality."""
    
    print("🚀 Multi-Task MVFoulsHead Demo")
    print("=" * 50)
    
    # 1. Create multi-task head
    print("\n1. Creating multi-task head...")
    head = build_multi_task_head(
        in_dim=1024,
        dropout=0.3,
        pooling='avg',
        loss_type='focal'
    )
    
    print(f"   ✓ Created head with {len(head.task_names)} tasks")
    print(f"   ✓ Task names: {head.task_names}")
    print(f"   ✓ Classes per task: {head.num_classes_per_task}")
    print(f"   ✓ Total classes: {head.num_classes}")
    
    # 2. Prepare sample data
    print("\n2. Preparing sample data...")
    batch_size = 3
    x = torch.randn(batch_size, 1024)
    print(f"   ✓ Input shape: {x.shape}")
    
    # 3. Multi-task forward pass
    print("\n3. Multi-task forward pass...")
    logits_dict, extras = head.forward_multi(x)
    
    print(f"   ✓ Got logits for {len(logits_dict)} tasks:")
    for task_name, logits in logits_dict.items():
        print(f"     - {task_name}: {logits.shape}")
    
    print(f"   ✓ Features shape: {extras['feat'].shape}")
    
    # 4. Single-tensor forward pass (backward compatibility)
    print("\n4. Single-tensor forward pass...")
    logits_concat, extras = head.forward_single(x)
    print(f"   ✓ Concatenated logits shape: {logits_concat.shape}")
    
    # 5. Create dummy targets for loss computation
    print("\n5. Creating dummy targets...")
    targets_dict = {}
    metadata = get_task_metadata()
    
    for task_name, num_classes in zip(metadata['task_names'], metadata['num_classes']):
        targets_dict[task_name] = torch.randint(0, num_classes, (batch_size,))
    
    print(f"   ✓ Created targets for {len(targets_dict)} tasks")
    
    # 6. Compute multi-task loss
    print("\n6. Computing multi-task loss...")
    loss_dict = head.compute_multi_task_loss(logits_dict, targets_dict)
    
    print(f"   ✓ Total loss: {loss_dict['total_loss']:.4f}")
    print("   ✓ Per-task losses:")
    for task_name in head.task_names:
        if f'{task_name}_loss' in loss_dict:
            print(f"     - {task_name}: {loss_dict[f'{task_name}_loss']:.4f}")
    
    # 7. Update metrics
    print("\n7. Updating metrics...")
    head.update_multi_task_metrics(logits_dict, targets_dict)
    print("   ✓ Metrics updated for all tasks")
    
    # 8. Demonstrate temporal input
    print("\n8. Testing temporal input...")
    x_temporal = torch.randn(batch_size, 8, 1024)  # (B, T, C)
    logits_dict_temporal, _ = head.forward_multi(x_temporal)
    
    print(f"   ✓ Temporal input shape: {x_temporal.shape}")
    print(f"   ✓ Output shapes same as before:")
    for task_name, logits in logits_dict_temporal.items():
        print(f"     - {task_name}: {logits.shape}")
    
    # 9. Compare with single-task head
    print("\n9. Comparing with single-task head...")
    single_head = MVFoulsHead(
        in_dim=1024,
        num_classes=5,
        multi_task=False
    )
    
    logits_single, _ = single_head.forward(x)
    print(f"   ✓ Single-task head output: {logits_single.shape}")
    print(f"   ✓ Multi-task head has {sum(p.numel() for p in head.parameters()):,} parameters")
    print(f"   ✓ Single-task head has {sum(p.numel() for p in single_head.parameters()):,} parameters")
    
    # 10. Show structure
    print("\n10. Multi-task head structure:")
    head.print_structure()
    
    print("\n🎉 Demo completed successfully!")


def demo_task_specific_losses():
    """Demonstrate different loss types per task."""
    
    print("\n" + "=" * 50)
    print("🎯 Task-Specific Loss Types Demo")
    print("=" * 50)
    
    metadata = get_task_metadata()
    
    # Create head with mixed loss types
    loss_types = ['focal'] * len(metadata['task_names'])
    loss_types[0] = 'ce'      # Cross-entropy for action_class
    loss_types[1] = 'bce'     # Binary cross-entropy for severity
    
    head = build_multi_task_head(
        in_dim=1024,
        loss_types_per_task=loss_types
    )
    
    print(f"✓ Created head with mixed loss types:")
    for task_name, loss_type in zip(head.task_names, head.loss_types_per_task):
        print(f"  - {task_name}: {loss_type}")
    
    # Test forward pass and loss computation
    batch_size = 2
    x = torch.randn(batch_size, 1024)
    logits_dict, _ = head.forward_multi(x)
    
    # Create targets
    targets_dict = {}
    for task_name, num_classes in zip(metadata['task_names'], metadata['num_classes']):
        targets_dict[task_name] = torch.randint(0, num_classes, (batch_size,))
    
    # Compute loss
    loss_dict = head.compute_multi_task_loss(logits_dict, targets_dict)
    
    print(f"\n✓ Successfully computed losses:")
    print(f"  - Total loss: {loss_dict['total_loss']:.4f}")
    for task_name in head.task_names[:3]:  # Show first 3 tasks
        if f'{task_name}_loss' in loss_dict:
            print(f"  - {task_name}: {loss_dict[f'{task_name}_loss']:.4f}")


if __name__ == '__main__':
    demo_multi_task_head()
    demo_task_specific_losses() 