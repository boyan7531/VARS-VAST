#!/usr/bin/env python3
"""
Quick fix for MVFouls training issues.

Key changes:
1. Delay unfreezing until epoch 5 (let head stabilize first)
2. Keep larger batch size (8 instead of 3)
3. Use single-task sampling instead of joint sampling
4. Increase learning rate slightly
"""

import subprocess
import sys

def run_optimized_training():
    """Run training with optimized parameters for better learning."""
    
    cmd = [
        "python", "train_with_class_weights.py",
        "--train-dir", "mvfouls/train_720p",
        "--val-dir", "mvfouls/valid_720p", 
        "--train-annotations", "mvfouls/train_720p.json",
        "--val-annotations", "mvfouls/valid_720p.json",
        "--multi-task",
        "--epochs", "15",  # More epochs since we're delaying unfreezing
        "--batch-size", "12",
        "--train-fraction", "1.0",
        "--freeze-mode", "gradual",
        "--use-smart-weighting",
        "--core-task-weight", "10.0",
        "--support-task-weight", "0.001", 
        "--context-task-weight", "0.001",
        "--balanced-sampling",
        # REMOVED: --joint-severity-sampling (too complex)
        "--output-dir", "./outputs_optimized",
        "--verbose",
        "--lr", "5e-04",  # Slightly higher LR
        "--disable-class-weights",
        "--reduce-batch-on-unfreeze",
        "--unfreeze-batch-size", "8"  # Larger than 3
    ]
    
    print("🚀 Running optimized training command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

if __name__ == "__main__":
    exit_code = run_optimized_training()
    sys.exit(exit_code) 