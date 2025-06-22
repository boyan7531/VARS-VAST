#!/usr/bin/env python3
"""
Debug script to analyze model predictions and understand why loss is low but accuracy is poor.
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
        loss_types_per_task=['focal'] * 11,
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
    task_names = metadata['task_names']
    
    print(f"📊 Analyzing {min(num_samples, len(dataset))} samples...")
    
    all_predictions = {task: [] for task in task_names}
    all_targets = {task: [] for task in task_names}
    all_confidences = {task: [] for task in task_names}
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, (videos, targets) in enumerate(dataloader):
            if samples_processed >= num_samples:
                break
                
            videos = videos.to(device)
            batch_size = videos.shape[0]
            
            # Forward pass
            logits_dict, _ = model(videos, return_dict=True)
            
            # Process each task
            for task_idx, task_name in enumerate(task_names):
                if task_name in logits_dict:
                    logits = logits_dict[task_name]
                    task_targets = targets[:, task_idx]
                    
                    # Get predictions and confidences
                    probs = F.softmax(logits, dim=1)
                    predictions = torch.argmax(logits, dim=1)
                    confidences = torch.max(probs, dim=1)[0]
                    
                    # Store results
                    all_predictions[task_name].extend(predictions.cpu().numpy())
                    all_targets[task_name].extend(task_targets.numpy())
                    all_confidences[task_name].extend(confidences.cpu().numpy())
            
            samples_processed += batch_size
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {samples_processed} samples...")
    
    print("\n📈 Analysis Results:")
    print("=" * 80)
    
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
        
        # Prediction distribution
        pred_counts = Counter(predictions)
        target_counts = Counter(targets)
        
        print(f"   Target distribution:")
        for class_idx in range(num_classes):
            class_name = class_names[class_idx]
            count = target_counts.get(class_idx, 0)
            percentage = (count / len(targets)) * 100
            print(f"     {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"   Prediction distribution:")
        for class_idx in range(num_classes):
            class_name = class_names[class_idx]
            count = pred_counts.get(class_idx, 0)
            percentage = (count / len(predictions)) * 100
            print(f"     {class_name}: {count} ({percentage:.1f}%)")
        
        # Accuracy
        accuracy = np.mean(predictions == targets)
        avg_confidence = np.mean(confidences)
        
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Avg Confidence: {avg_confidence:.3f}")
        
        # Check if model is just predicting one class
        most_predicted_class = max(pred_counts, key=pred_counts.get)
        most_predicted_percentage = (pred_counts[most_predicted_class] / len(predictions)) * 100
        
        if most_predicted_percentage > 80:
            print(f"   ⚠️  MODEL IS MOSTLY PREDICTING: {class_names[most_predicted_class]} ({most_predicted_percentage:.1f}%)")
        
        # Confidence analysis for correct vs incorrect predictions
        correct_mask = predictions == targets
        if np.any(correct_mask) and np.any(~correct_mask):
            correct_conf = np.mean(confidences[correct_mask])
            incorrect_conf = np.mean(confidences[~correct_mask])
            print(f"   Confidence - Correct: {correct_conf:.3f}, Incorrect: {incorrect_conf:.3f}")
    
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

if __name__ == "__main__":
    # Analyze focal loss behavior first
    analyze_focal_loss_behavior()
    
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
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("   This is expected if model checkpoint doesn't exist yet.")
        print("   Run this script after training for a few epochs.") 