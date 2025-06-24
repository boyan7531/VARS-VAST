#!/usr/bin/env python3
"""
Debug script to analyze model predictions and understand why loss is low but accuracy is poor.
Focus on the 3 core tasks: action_class, severity, offence
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from dataset import MVFoulsDataset
from model.mvfouls_model import MVFoulsModel
from transforms import get_video_transforms
from utils import get_task_metadata
from torch.utils.data import DataLoader

def analyze_predictions(model_path: str, data_dir: str, annotations_path: str, num_samples: int = 100):
    """Analyze what the model is actually predicting."""
    
    print("🔍 Loading model and data...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For now, let's create a fresh model to see initial predictions
    # (since we don't have a saved checkpoint yet)
    from model.mvfouls_model import build_multi_task_model
    model = build_multi_task_model(
        backbone_pretrained=True,
        backbone_freeze_mode='gradual',
        loss_types_per_task=['focal'] * 3,  # Only 3 tasks now
    )
    model.to(device)
    model.eval()
    
    # Load dataset
    transforms = get_video_transforms(image_size=224, augment_train=False)
    dataset = MVFoulsDataset(
        root_dir=str(Path(data_dir).parent),
        split=Path(data_dir).name.replace('_720p', ''),
        transform=transforms['val'],
        load_annotations=True,
        num_frames=32
    )
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Get task metadata
    metadata = get_task_metadata()
    task_names = metadata['task_names']  # Should now be ['action_class', 'severity', 'offence']
    
    print(f"📊 Active tasks: {task_names}")
    print(f"📊 Analyzing {min(num_samples, len(dataset))} samples...")
    
    all_predictions = {task: [] for task in task_names}
    all_targets = {task: [] for task in task_names}
    all_confidences = {task: [] for task in task_names}
    
    samples_processed = 0
    
    with torch.no_grad():
        for videos, targets in dataloader:
            if samples_processed >= num_samples:
                break
                
            videos = videos.to(device)
            batch_size = videos.shape[0]
            
            # Forward pass
            outputs = model(videos)
            
            # Handle different output formats
            if isinstance(outputs, tuple) and len(outputs) == 2:
                logits_dict, _ = outputs
            elif isinstance(outputs, dict):
                logits_dict = outputs
            else:
                # Handle tuple of logits for each task
                logits_dict = {}
                for i, task_name in enumerate(task_names):
                    if i < len(outputs):
                        logits_dict[task_name] = outputs[i]
            
            # Process each task
            for task_idx, task_name in enumerate(task_names):
                if task_name not in logits_dict:
                    print(f"⚠️ Task {task_name} not found in model outputs")
                    continue
                    
                task_logits = logits_dict[task_name]
                task_targets = targets[:, task_idx] if targets.dim() > 1 else targets
                
                # Get predictions and confidences
                probs = F.softmax(task_logits, dim=1)
                predictions = torch.argmax(task_logits, dim=1)
                confidences = torch.max(probs, dim=1)[0]
                
                # Store results
                all_predictions[task_name].extend(predictions.cpu().numpy())
                all_targets[task_name].extend(task_targets.cpu().numpy())
                all_confidences[task_name].extend(confidences.cpu().numpy())
            
            samples_processed += batch_size
    
    # Analyze each task
    for task_name in task_names:
        if not all_predictions[task_name]:
            continue
            
        predictions = np.array(all_predictions[task_name])
        targets = np.array(all_targets[task_name])
        confidences = np.array(all_confidences[task_name])
        
        # Get class names
        class_names = metadata['class_names'][task_name]
        num_classes = len(class_names)
        
        print(f"\n🎯 {task_name.upper()}:")
        print(f"   Classes: {num_classes}")
        print(f"   Class names: {class_names}")
        
        # Prediction distribution
        pred_counts = Counter(predictions)
        target_counts = Counter(targets)
        
        print(f"   Target distribution:")
        for class_idx in range(num_classes):
            class_name = class_names[class_idx]
            count = target_counts.get(class_idx, 0)
            percentage = (count / len(targets)) * 100 if len(targets) > 0 else 0
            print(f"     {class_idx}: {class_name} -> {count} ({percentage:.1f}%)")
        
        print(f"   Prediction distribution:")
        for class_idx in range(num_classes):
            class_name = class_names[class_idx]
            count = pred_counts.get(class_idx, 0)
            percentage = (count / len(predictions)) * 100 if len(predictions) > 0 else 0
            print(f"     {class_idx}: {class_name} -> {count} ({percentage:.1f}%)")
        
        # Accuracy
        accuracy = np.mean(predictions == targets) if len(targets) > 0 else 0
        avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0
        
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Avg Confidence: {avg_confidence:.3f}")
        
        # Check if model is just predicting one class
        if len(pred_counts) > 0:
            most_predicted_class = max(pred_counts, key=pred_counts.get)
            most_predicted_percentage = (pred_counts[most_predicted_class] / len(predictions)) * 100
            
            if most_predicted_percentage > 80:
                print(f"   ⚠️  MODEL IS MOSTLY PREDICTING: {class_names[most_predicted_class]} (index {most_predicted_class}) ({most_predicted_percentage:.1f}%)")
                
                # Special analysis for severity collapse
                if task_name == 'severity' and most_predicted_class == 2:
                    print(f"   🚨 SEVERITY COLLAPSE DETECTED: Model is predicting severity '2' for {most_predicted_percentage:.1f}% of examples")
                    print(f"      This suggests the model learned a degenerate solution")
                    print(f"      Possible causes: extreme class imbalance, poor loss weighting, focal loss gamma too high")
        
        # Confidence analysis for correct vs incorrect predictions
        if len(targets) > 0 and len(predictions) > 0:
            correct_mask = predictions == targets
            if np.any(correct_mask) and np.any(~correct_mask):
                correct_conf = np.mean(confidences[correct_mask])
                incorrect_conf = np.mean(confidences[~correct_mask])
                print(f"   Confidence - Correct: {correct_conf:.3f}, Incorrect: {incorrect_conf:.3f}")
                
                if incorrect_conf > 0.8:
                    print(f"   🚨 HIGH CONFIDENCE ON WRONG PREDICTIONS: {incorrect_conf:.3f}")
                    print(f"      This indicates severe overfitting or model collapse")
    
    return all_predictions, all_targets, all_confidences

def analyze_focal_loss_behavior():
    """Analyze how focal loss behaves with different confidence levels."""
    print("\n🔬 Focal Loss Behavior Analysis:")
    print("=" * 50)
    
    # Simulate different prediction scenarios
    scenarios = [
        ("Confident Correct", 0.95, True),
        ("Confident Wrong", 0.95, False),
        ("Uncertain Correct", 0.6, True),
        ("Uncertain Wrong", 0.4, False),
    ]
    
    gamma = 2.0
    
    for name, confidence, is_correct in scenarios:
        if is_correct:
            # Probability of correct class
            p = confidence
        else:
            # Probability of correct class when predicting wrong
            p = 1.0 - confidence
        
        # Standard cross-entropy loss
        ce_loss = -np.log(p)
        
        # Focal loss
        focal_loss = -(1 - p) ** gamma * np.log(p)
        
        print(f"   {name:18} | CE Loss: {ce_loss:.4f} | Focal Loss: {focal_loss:.4f} | Ratio: {focal_loss/ce_loss:.3f}")

def analyze_class_imbalance():
    """Analyze class imbalance in the dataset."""
    print("\n📊 Class Imbalance Analysis:")
    print("=" * 50)
    
    try:
        # Load dataset statistics
        transforms = get_video_transforms(image_size=224, augment_train=False)
        dataset = MVFoulsDataset(
            root_dir="mvfouls",
            split="train",
            transform=transforms['val'],
            load_annotations=True,
            num_frames=32
        )
        
        stats = dataset.get_task_statistics()
        metadata = get_task_metadata()
        
        for task_name in metadata['task_names']:
            if task_name in stats:
                class_dist = stats[task_name]['class_distribution']
                class_names = metadata['class_names'][task_name]
                
                print(f"\n  {task_name}:")
                total_samples = sum(class_dist.values())
                
                for class_idx, count in class_dist.items():
                    class_name = class_names[class_idx] if class_idx < len(class_names) else f"Unknown_{class_idx}"
                    percentage = (count / total_samples) * 100
                    print(f"    {class_idx}: {class_name} -> {count} ({percentage:.1f}%)")
                
                # Calculate imbalance ratio
                max_count = max(class_dist.values())
                min_count = min([c for c in class_dist.values() if c > 0])
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                print(f"    Imbalance ratio: {imbalance_ratio:.1f}:1")
                
                if imbalance_ratio > 20:
                    print(f"    🚨 SEVERE IMBALANCE - Consider strong class weights or resampling")
                elif imbalance_ratio > 5:
                    print(f"    ⚠️ MODERATE IMBALANCE - Consider class weights")
                    
    except Exception as e:
        print(f"   Could not analyze dataset: {e}")

if __name__ == "__main__":
    # Analyze focal loss behavior first
    analyze_focal_loss_behavior()
    
    # Analyze class imbalance
    analyze_class_imbalance()
    
    # Then analyze actual predictions
    try:
        predictions, targets, confidences = analyze_predictions(
            model_path=None,  # Using fresh model for now
            data_dir="mvfouls/valid_720p",
            annotations_path="mvfouls/valid_720p.json",
            num_samples=200
        )
        
        print("\n✅ Analysis complete!")
        print("\n💡 Recommendations:")
        print("   1. Check if model is predicting only majority classes")
        print("   2. Consider stronger class weights or different loss functions")
        print("   3. Try curriculum learning starting with balanced subsets")
        print("   4. Consider oversampling minority classes")
        print("   5. For severity collapse, try reducing focal gamma or using CE loss")
        print("   6. Ensure offence task gets proper gradient weights in training")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("   This is expected if model checkpoint doesn't exist yet.")
        print("   Run this script after training for a few epochs.") 