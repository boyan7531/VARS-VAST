#!/usr/bin/env python3
"""
Debug script to analyze why macro F1 scores are poor.
This script will examine prediction patterns to understand the issue.
"""

import torch
import numpy as np
from collections import Counter, defaultdict
from dataset import MVFoulsDataset
from model.mvfouls_model import build_mvfouls_model
from utils import get_task_metadata, compute_task_metrics
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_predictions_and_targets(logits_dict, targets_dict, task_name):
    """Analyze prediction patterns for a specific task."""
    logits = logits_dict[task_name]
    targets = targets_dict[task_name]
    
    # Get predictions
    preds = torch.argmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    
    num_classes = logits.shape[1]
    
    print(f"\n🔍 ANALYSIS FOR {task_name.upper()}")
    print("=" * 50)
    
    # Class distribution in targets
    target_counts = Counter(targets.cpu().numpy())
    pred_counts = Counter(preds.cpu().numpy())
    
    print(f"📊 Target distribution:")
    for cls in range(num_classes):
        count = target_counts.get(cls, 0)
        percentage = count / len(targets) * 100
        print(f"   Class {cls}: {count:4d} samples ({percentage:5.1f}%)")
    
    print(f"\n📊 Prediction distribution:")
    for cls in range(num_classes):
        count = pred_counts.get(cls, 0)
        percentage = count / len(preds) * 100
        print(f"   Class {cls}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Per-class precision, recall, F1
    print(f"\n📊 Per-class metrics:")
    print(f"{'Class':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>9} {'Recall':>7} {'F1':>7}")
    print("-" * 50)
    
    f1_scores = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).float().sum().item()
        fp = ((preds == cls) & (targets != cls)).float().sum().item()
        fn = ((preds != cls) & (targets == cls)).float().sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        f1_scores.append(f1)
        
        print(f"{cls:5d} {tp:4.0f} {fp:4.0f} {fn:4.0f} {precision:9.3f} {recall:7.3f} {f1:7.3f}")
    
    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n📊 Macro F1: {macro_f1:.4f}")
    
    # Check if model is biased toward specific classes
    print(f"\n🎯 Prediction bias analysis:")
    max_prob_classes = torch.argmax(probs, dim=1)
    avg_max_probs = []
    for cls in range(num_classes):
        mask = max_prob_classes == cls
        if mask.sum() > 0:
            avg_prob = probs[mask, cls].mean().item()
            avg_max_probs.append(avg_prob)
            print(f"   Class {cls}: Avg confidence = {avg_prob:.3f} (predicted {mask.sum()} times)")
        else:
            avg_max_probs.append(0.0)
            print(f"   Class {cls}: Never predicted")
    
    return {
        'macro_f1': macro_f1,
        'f1_per_class': f1_scores,
        'target_counts': target_counts,
        'pred_counts': pred_counts,
        'avg_confidences': avg_max_probs
    }

def debug_model_on_validation():
    """Load model and analyze predictions on validation set."""
    
    # Load model
    print("🔧 Loading model...")
    model_path = "outputs/smoke_test_memory_opt/best_model_latest.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    # Build model
    metadata = get_task_metadata()
    model = build_mvfouls_model(
        backbone='video_swin_b',
        num_classes=metadata['num_classes'],
        multi_task=True,
        task_names=metadata['task_names'],
        pretrained=False,  # Don't load pretrained weights since we're loading our checkpoint
        clip_pooling_type='mean',  # Match training config
        clip_pooling_temperature=1.0
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"✅ Model loaded successfully")
    
    # Load validation dataset
    print("📂 Loading validation dataset...")
    val_dataset = MVFoulsDataset(
        root_dir="mvfouls",
        split="valid",
        bag_of_clips=True,
        max_clips_per_action=4,
        min_clips_per_action=2,
        clip_sampling_strategy="random"
    )
    
    # Get a small batch for analysis
    print("🔍 Analyzing predictions on validation samples...")
    
    batch_size = 32
    all_logits = {task: [] for task in metadata['task_names']}
    all_targets = {task: [] for task in metadata['task_names']}
    
    with torch.no_grad():
        for i in range(min(batch_size, len(val_dataset))):
            videos, targets = val_dataset[i]
            
            # Handle bag-of-clips format
            if isinstance(videos, list):
                # Convert list of video tensors to batch tensor
                videos = torch.stack(videos, dim=0)  # Shape: [num_clips, C, T, H, W]
                videos = videos.unsqueeze(0)  # Add batch dimension: [1, num_clips, C, T, H, W]
            else:
                videos = videos.unsqueeze(0)  # Add batch dimension
            
            # Convert to float and normalize if needed
            videos = videos.float()
            if videos.max() > 1.0:
                videos = videos / 255.0  # Normalize from [0, 255] to [0, 1]
            
            videos = videos.to(device)
            
            # Forward pass (bag-of-clips mode)
            logits_dict, _ = model(videos, return_dict=True)
            
            # Store results
            for task_name in metadata['task_names']:
                all_logits[task_name].append(logits_dict[task_name].cpu())
                # Handle different target formats
                if isinstance(targets, dict):
                    target_value = targets[task_name]
                else:
                    # If targets is a tensor, assume task order matches metadata
                    task_idx = metadata['task_names'].index(task_name)
                    target_value = targets[task_idx] if targets.dim() > 0 else targets.item()
                
                all_targets[task_name].append(torch.tensor([target_value]))
    
    # Concatenate all results
    combined_logits = {}
    combined_targets = {}
    for task_name in metadata['task_names']:
        combined_logits[task_name] = torch.cat(all_logits[task_name], dim=0)
        combined_targets[task_name] = torch.cat(all_targets[task_name], dim=0)
    
    print(f"\n📊 Analyzed {batch_size} validation samples")
    
    # Analyze each task
    analysis_results = {}
    for task_name in metadata['task_names']:
        analysis_results[task_name] = analyze_predictions_and_targets(
            combined_logits, combined_targets, task_name
        )
    
    # Summary
    print(f"\n🎯 SUMMARY")
    print("=" * 50)
    for task_name in metadata['task_names']:
        macro_f1 = analysis_results[task_name]['macro_f1']
        print(f"{task_name:15}: Macro F1 = {macro_f1:.4f}")
    
    # Specific insights
    print(f"\n💡 INSIGHTS")
    print("=" * 50)
    
    for task_name in metadata['task_names']:
        result = analysis_results[task_name]
        f1_scores = result['f1_per_class']
        
        # Find classes with zero F1
        zero_f1_classes = [i for i, f1 in enumerate(f1_scores) if f1 == 0.0]
        if zero_f1_classes:
            print(f"🚨 {task_name}: Classes with F1=0: {zero_f1_classes}")
        
        # Find dominant prediction class
        pred_counts = result['pred_counts']
        if pred_counts:
            dominant_class = max(pred_counts.keys(), key=lambda k: pred_counts[k])
            dominant_percentage = pred_counts[dominant_class] / sum(pred_counts.values()) * 100
            if dominant_percentage > 70:
                print(f"⚠️  {task_name}: Heavily biased toward class {dominant_class} ({dominant_percentage:.1f}%)")

if __name__ == "__main__":
    debug_model_on_validation() 