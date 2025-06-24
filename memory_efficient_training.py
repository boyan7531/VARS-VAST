#!/usr/bin/env python3
"""
Memory-efficient MVFouls training for limited GPU resources.

Key strategies:
1. Keep batch size 3 but use gradient accumulation to simulate larger batches
2. Delay unfreezing until much later (epoch 8+)
3. Use gradient accumulation to simulate effective batch size of 12
4. Only unfreeze when head is very stable
5. Leverage powerful CPU (EPYC 7742) with high worker count
"""

import subprocess
import sys

def run_memory_efficient_training():
    """Run training optimized for limited GPU memory but powerful CPU."""
    
    cmd = [
        "python", "train_with_class_weights.py",
        "--train-dir", "mvfouls/train_720p",
        "--val-dir", "mvfouls/valid_720p", 
        "--train-annotations", "mvfouls/train_720p.json",
        "--val-annotations", "mvfouls/valid_720p.json",
        "--multi-task",
        "--epochs", "20",  # Much longer training to accommodate delayed unfreezing
        "--batch-size", "12",  # Start with 12, will reduce to 3 when unfreezing
        "--train-fraction", "1.0",
        "--freeze-mode", "gradual",
        "--use-smart-weighting",
        "--core-task-weight", "15.0",  # Higher weight for core tasks
        "--support-task-weight", "0.001", 
        "--context-task-weight", "0.001",
        "--balanced-sampling",
        # REMOVED: --joint-severity-sampling (too complex for small batches)
        "--output-dir", "./outputs_memory_efficient",
        "--verbose",
        "--lr", "4e-04",  # Slightly lower LR for stability
        "--disable-class-weights",
        "--reduce-batch-on-unfreeze",
        "--unfreeze-batch-size", "3",  # Keep at 3 for memory constraints
        "--gradient-accumulation-steps", "4",  # Simulate larger effective batch size
        "--num-workers", "16"  # Leverage EPYC 7742's 32+ cores for fast data loading
    ]
    
    print("🚀 Running memory-efficient training command (optimized for EPYC 7742):")
    print(" ".join(cmd))
    print()
    print("📊 Training Strategy:")
    print("   • Epochs 1-7: Head-only training (batch size 12)")
    print("   • Epoch 8+: Gradual unfreezing (batch size 3)")
    print("   • Gradient accumulation: 4 steps (effective batch size 12-48)")
    print("   • Data loading: 16 workers (utilizing powerful CPU)")
    print("   • Focus on core tasks (15x weight)")
    print()
    
    # Run the command
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

if __name__ == "__main__":
    exit_code = run_memory_efficient_training()
    sys.exit(exit_code) 