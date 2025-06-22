#!/usr/bin/env python3
"""
Adaptive Learning Rate Demo for MVFouls Training

This script demonstrates how to use the new adaptive learning rate scaling feature
that automatically adjusts learning rates when major backbone stages are unfrozen.

Example usage:
    python examples/adaptive_lr_demo.py --help
    
    # Basic usage with adaptive LR
    python train_with_class_weights.py \
        --train-dir mvfouls/train_720p \
        --val-dir mvfouls/valid_720p \
        --train-annotations mvfouls/train_720p.json \
        --val-annotations mvfouls/valid_720p.json \
        --multi-task \
        --epochs 20 \
        --batch-size 8 \
        --freeze-mode gradual \
        --output-dir ./outputs_adaptive_lr \
        --reduce-batch-on-unfreeze \
        --unfreeze-batch-size 4 \
        --adaptive-lr \
        --primary-task-weight 3.0 \
        --verbose
        
    # Custom LR scaling factors
    python train_with_class_weights.py \
        --train-dir mvfouls/train_720p \
        --val-dir mvfouls/valid_720p \
        --train-annotations mvfouls/train_720p.json \
        --val-annotations mvfouls/valid_720p.json \
        --multi-task \
        --epochs 20 \
        --batch-size 8 \
        --freeze-mode gradual \
        --output-dir ./outputs_custom_scaling \
        --reduce-batch-on-unfreeze \
        --unfreeze-batch-size 4 \
        --adaptive-lr \
        --lr-scale-minor 2.0 \
        --lr-scale-major 4.0 \
        --lr-scale-massive 8.0 \
        --primary-task-weight 3.0 \
        --verbose
"""

import argparse

def show_adaptive_lr_explanation():
    """Show detailed explanation of the adaptive learning rate feature."""
    
    print("🔧 Adaptive Learning Rate Scaling Feature")
    print("=" * 50)
    print()
    
    print("📖 What it does:")
    print("   Automatically scales learning rate when backbone stages are unfrozen")
    print("   Solves the problem of learning rate being too low for newly unfrozen parameters")
    print()
    
    print("🎯 Why it's needed:")
    print("   • When backbone is frozen: ~100K trainable parameters")
    print("   • When stage_2 unfrozen: ~60M trainable parameters (600x increase!)")
    print("   • Original LR becomes too small for the massive parameter increase")
    print("   • Result: Loss plateaus and training stagnates")
    print()
    
    print("⚙️  How it works:")
    print("   1. Monitors unfreezing events during training")
    print("   2. Categorizes unfreezing by parameter count:")
    print("      • MINOR: patch_embed, stage_0 (< 1M params)")
    print("      • MAJOR: stage_1 (1-10M params)")  
    print("      • MASSIVE: stage_2, stage_3 (> 10M params)")
    print("   3. Applies appropriate LR scaling factor")
    print("   4. Logs all changes for transparency")
    print()
    
    print("🔢 Default scaling factors:")
    print("   • Minor unfreezing: 1.5x LR boost")
    print("   • Major unfreezing: 3.0x LR boost")
    print("   • Massive unfreezing: 5.0x LR boost")
    print()
    
    print("📊 Expected behavior:")
    print("   • Epoch 1-3: Base LR (frozen backbone)")
    print("   • Epoch 4: 1.5x LR (patch_embed unfrozen)")
    print("   • Epoch 7: 1.5x LR (stage_0 unfrozen)")
    print("   • Epoch 10: 3.0x LR (stage_1 unfrozen)")
    print("   • Epoch 13: 5.0x LR (stage_2 unfrozen - MASSIVE)")
    print("   • Epoch 16: 5.0x LR (stage_3 unfrozen - MASSIVE)")
    print()
    
    print("💡 Benefits:")
    print("   • Prevents loss plateaus during unfreezing")
    print("   • Maintains training momentum")
    print("   • Automatic - no manual LR scheduling needed")
    print("   • Customizable scaling factors")
    print()
    
    print("⚠️  Important notes:")
    print("   • Only works with --freeze-mode gradual")
    print("   • Must enable with --adaptive-lr flag")
    print("   • Can be combined with primary task weighting")
    print("   • Monitor tensorboard for LR changes")
    print()

def show_command_examples():
    """Show practical command examples."""
    
    print("🚀 Command Examples")
    print("=" * 50)
    print()
    
    print("1️⃣  Basic adaptive LR (recommended):")
    print("   python train_with_class_weights.py \\")
    print("     --train-dir mvfouls/train_720p \\")
    print("     --val-dir mvfouls/valid_720p \\")
    print("     --train-annotations mvfouls/train_720p.json \\")
    print("     --val-annotations mvfouls/valid_720p.json \\")
    print("     --multi-task \\")
    print("     --epochs 20 \\")
    print("     --batch-size 8 \\")
    print("     --freeze-mode gradual \\")
    print("     --output-dir ./outputs_adaptive_lr \\")
    print("     --reduce-batch-on-unfreeze \\")
    print("     --unfreeze-batch-size 4 \\")
    print("     --adaptive-lr \\")
    print("     --primary-task-weight 3.0 \\")
    print("     --verbose")
    print()
    
    print("2️⃣  Conservative scaling (for stability):")
    print("   python train_with_class_weights.py \\")
    print("     [... same args as above ...] \\")
    print("     --adaptive-lr \\")
    print("     --lr-scale-minor 1.2 \\")
    print("     --lr-scale-major 2.0 \\")
    print("     --lr-scale-massive 3.0")
    print()
    
    print("3️⃣  Aggressive scaling (for fast convergence):")
    print("   python train_with_class_weights.py \\")
    print("     [... same args as above ...] \\")
    print("     --adaptive-lr \\")
    print("     --lr-scale-minor 2.0 \\")
    print("     --lr-scale-major 5.0 \\")
    print("     --lr-scale-massive 10.0")
    print()
    
    print("4️⃣  Combined with primary task weighting:")
    print("   python train_with_class_weights.py \\")
    print("     [... same args as above ...] \\")
    print("     --adaptive-lr \\")
    print("     --primary-task-weight 4.0 \\")
    print("     --auxiliary-task-weight 1.0 \\")
    print("     --primary-tasks action_class severity")
    print()

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Adaptive Learning Rate Demo')
    parser.add_argument('--explanation', action='store_true', 
                       help='Show detailed explanation of adaptive LR')
    parser.add_argument('--examples', action='store_true',
                       help='Show command examples')
    parser.add_argument('--all', action='store_true',
                       help='Show everything')
    
    args = parser.parse_args()
    
    if args.all or args.explanation:
        show_adaptive_lr_explanation()
        
    if args.all or args.examples:
        show_command_examples()
        
    if not any([args.explanation, args.examples, args.all]):
        print("🔧 Adaptive Learning Rate Demo")
        print("Use --help to see available options")
        print("Use --all to see complete documentation")

if __name__ == '__main__':
    main() 