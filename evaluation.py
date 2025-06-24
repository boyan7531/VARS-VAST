#!/usr/bin/env python3
"""
MVFouls Model Evaluation Script

This script loads a trained MVFouls model and generates action-level predictions 
for the test set by aggregating clip-level predictions.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model.mvfouls_model import MVFoulsModel
from dataset import MVFoulsDataset
from transforms import get_video_transforms
from utils import get_task_metadata


def setup_logging(level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_trained_model(checkpoint_path: str, device: torch.device) -> MVFoulsModel:
    """Load a trained MVFouls model from checkpoint."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"📥 Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    if 'config' in checkpoint:
        model_config = checkpoint['config'].get('model_config', {})
        logger.info(f"📋 Model config: {model_config}")
    else:
        logger.warning("⚠️  No config found in checkpoint, using defaults")
        model_config = {}
    
    # Reconstruct model (assuming multi-task model)
    from model.mvfouls_model import build_multi_task_model
    
    model = build_multi_task_model(
        backbone_pretrained=True,
        backbone_freeze_mode='none',  # For inference, we don't need gradual unfreezing
        backbone_checkpointing=False  # Disable for inference
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"✅ Model loaded successfully!")
    logger.info(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"   Best metric: {checkpoint.get('best_metric', 'unknown')}")
    
    return model


def predict_batch(model: MVFoulsModel, videos: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    """Generate predictions for a batch of videos."""
    videos = videos.to(device)
    
    with torch.no_grad():
        # Get model predictions
        outputs = model(videos)
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            # If model returns tuple, convert to dict format
            # Assuming order: action_class, severity, offence, contact, bodypart, etc.
            task_names = ['action_class', 'severity', 'offence', 'contact', 'bodypart', 
                         'upper_body_part', 'multiple_fouls', 'try_to_play', 'touch_ball', 
                         'handball', 'handball_offence']
            outputs_dict = {}
            for i, task_name in enumerate(task_names):
                if i < len(outputs):
                    outputs_dict[task_name] = outputs[i]
            outputs = outputs_dict
        
        # Convert logits to probabilities
        predictions = {}
        for task_name, task_output in outputs.items():
            # Handle nested dict structure (e.g., {'logits': tensor, 'loss': tensor})
            if isinstance(task_output, dict):
                if 'logits' in task_output:
                    logits = task_output['logits']
                elif 'predictions' in task_output:
                    logits = task_output['predictions']
                else:
                    # Take the first tensor value if no standard key found
                    logits = next(v for v in task_output.values() if isinstance(v, torch.Tensor))
            else:
                # Direct tensor output
                logits = task_output
            
            probabilities = F.softmax(logits, dim=1)
            predictions[task_name] = probabilities
    
    return predictions


def aggregate_clip_predictions_to_action(
    clip_predictions: List[Dict[str, torch.Tensor]], 
    action_ids: List[str]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Aggregate clip-level predictions to action-level predictions."""
    logger = logging.getLogger(__name__)
    
    # Group predictions by action_id
    action_groups = defaultdict(list)
    for i, action_id in enumerate(action_ids):
        action_groups[action_id].append(i)
    
    logger.info(f"📊 Aggregating {len(clip_predictions)} clip predictions into {len(action_groups)} actions")
    
    action_predictions = {}
    
    for action_id, clip_indices in action_groups.items():
        # Aggregate predictions for this action
        aggregated = {}
        
        for task_name in clip_predictions[0].keys():
            # Collect all clip predictions for this task
            task_preds = []
            for clip_idx in clip_indices:
                task_preds.append(clip_predictions[clip_idx][task_name])
            
            # Stack and average the predictions
            stacked_preds = torch.stack(task_preds, dim=0)
            averaged_preds = torch.mean(stacked_preds, dim=0, keepdim=True)
            aggregated[task_name] = averaged_preds
        
        action_predictions[action_id] = aggregated
    
    return action_predictions


def convert_action_predictions_to_submission_format(
    action_predictions: Dict[str, Dict[str, torch.Tensor]], 
    metadata: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Convert action-level predictions to the required submission JSON format."""
    
    results = {}
    class_names = metadata['class_names']
    
    for action_id, predictions in action_predictions.items():
        # Get action_class prediction
        action_probs = predictions['action_class'][0]  # Remove batch dimension
        action_class_idx = torch.argmax(action_probs).item()
        action_confidence = torch.max(action_probs).item()
        action_class_name = class_names['action_class'][action_class_idx]
        
        # Get severity prediction
        severity_probs = predictions['severity'][0]
        severity_idx = torch.argmax(severity_probs).item()
        severity_confidence = torch.max(severity_probs).item()
        # Convert severity index to string (0->Missing, 1->1.0, 2->2.0, etc.)
        if severity_idx == 0:
            severity_value = "Missing"
        else:
            severity_value = f"{severity_idx}.0"
        
        # Get offence prediction
        offence_probs = predictions['offence'][0]
        offence_idx = torch.argmax(offence_probs).item()
        offence_name = class_names['offence'][offence_idx]
        
        # Create action entry
        results[action_id] = {
            "Action class": action_class_name,
            "Offence": offence_name,
            "Severity": severity_value,
            "Severity_confidence": float(severity_confidence),
            "Action_confidence": float(action_confidence)
        }
    
    return results


def evaluate_test_set(
    model: MVFoulsModel,
    test_dataset: MVFoulsDataset,
    test_loader: DataLoader,
    device: torch.device,
    output_file: str
) -> None:
    """Evaluate model on test set and save results in JSON format."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔮 Starting evaluation on test set...")
    logger.info(f"   Test clips: {len(test_dataset)}")
    logger.info(f"   Test batches: {len(test_loader)}")
    logger.info(f"   Output file: {output_file}")
    
    # Get metadata for class name mapping
    metadata = get_task_metadata()
    
    # Collect all clip predictions and their corresponding action IDs
    all_clip_predictions = []
    all_action_ids = []
    
    # Process test set in batches
    with tqdm(test_loader, desc="Evaluating clips", unit="batch") as pbar:
        for batch_idx, (videos, targets) in enumerate(pbar):
            # Generate predictions
            batch_predictions = predict_batch(model, videos, device)
            
            # Get action IDs for this batch
            batch_start_idx = batch_idx * test_loader.batch_size
            batch_end_idx = min(batch_start_idx + len(videos), len(test_dataset))
            
            for i in range(len(videos)):
                clip_idx = batch_start_idx + i
                if clip_idx < len(test_dataset):
                    # Get action ID from dataset
                    clip_info = test_dataset.dataset_index[clip_idx]
                    action_id = clip_info.action_id
                    
                    # Extract single clip predictions
                    clip_pred = {}
                    for task_name, task_preds in batch_predictions.items():
                        clip_pred[task_name] = task_preds[i:i+1]  # Keep batch dimension
                    
                    all_clip_predictions.append(clip_pred)
                    all_action_ids.append(action_id)
            
            # Update progress bar
            pbar.set_postfix({
                'Clips': len(all_clip_predictions),
                'Batch': f'{batch_idx+1}/{len(test_loader)}'
            })
    
    logger.info(f"📊 Collected predictions for {len(all_clip_predictions)} clips")
    
    # Aggregate clip predictions to action predictions
    action_predictions = aggregate_clip_predictions_to_action(all_clip_predictions, all_action_ids)
    
    # Convert to submission format
    submission_actions = convert_action_predictions_to_submission_format(action_predictions, metadata)
    
    # Create final submission data
    submission_data = {
        "Set": "test",
        "Actions": submission_actions
    }
    
    # Save results to JSON file
    logger.info(f"💾 Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(submission_data, f, indent=2)
    
    logger.info(f"✅ Evaluation complete!")
    logger.info(f"   Total clips processed: {len(all_clip_predictions)}")
    logger.info(f"   Total actions predicted: {len(submission_actions)}")
    logger.info(f"   Results saved to: {output_file}")
    
    # Print sample of results
    logger.info(f"📋 Sample predictions:")
    for i, (action_id, prediction) in enumerate(submission_data["Actions"].items()):
        if i < 3:  # Show first 3 predictions
            logger.info(f"   Action {action_id}: {prediction['Action class']} "
                       f"(conf: {prediction['Action_confidence']:.3f}), "
                       f"Severity {prediction['Severity']} "
                       f"(conf: {prediction['Severity_confidence']:.3f})")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate MVFouls Model on Test Set')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--test-dir', type=str, required=True, 
                       help='Test data directory')
    parser.add_argument('--test-annotations', type=str, required=True, 
                       help='Test annotations file')
    
    # Optional arguments
    parser.add_argument('--output-file', type=str, default='test_predictions.json',
                       help='Output JSON file for predictions (default: test_predictions.json)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for evaluation (default: 8)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu, default: auto)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging(log_level)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"🚀 Starting MVFouls model evaluation")
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test dir: {args.test_dir}")
    
    try:
        # Load trained model
        model = load_trained_model(args.checkpoint, device)
        
        # Create test dataset
        logger.info("📂 Creating test dataset...")
        
        # Get transforms (no augmentation for test)
        transforms = get_video_transforms(image_size=224, augment_train=False)
        
        # Extract root directory and split name
        root_dir = str(Path(args.test_dir).parent)
        test_split = Path(args.test_dir).name.replace('_720p', '')
        
        test_dataset = MVFoulsDataset(
            root_dir=root_dir,
            split=test_split,
            transform=transforms['val'],  # Use validation transforms (no augmentation)
            load_annotations=True,
            num_frames=32
        )
        
        logger.info(f"📊 Test dataset size: {len(test_dataset)} clips")
        
        # Create test dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Important: don't shuffle test set
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False  # Important: include all test samples
        )
        
        # Run evaluation
        evaluate_test_set(model, test_dataset, test_loader, device, args.output_file)
        
    except Exception as e:
        logger.error(f"💥 Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()
