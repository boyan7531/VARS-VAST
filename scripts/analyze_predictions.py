#!/usr/bin/env python3
"""Analyse trained MVFouls model predictions on a test set.

Loads a checkpoint, runs inference on the test split (clip-level or bag-of-clips),
and prints per-task class distribution so you can quickly see whether the
network collapsed to predicting a single class.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from dataset import MVFoulsDataset, bag_of_clips_collate_fn  # type: ignore
from transforms import get_val_transforms  # type: ignore
from utils import get_task_metadata  # type: ignore
from model.mvfouls_model import MVFoulsModel, build_multi_task_model  # type: ignore

# Import thresholding utilities
try:
    from thresholding import load_thresholds, extract_all_task_thresholds  # type: ignore
except ImportError:
    load_thresholds = None
    extract_all_task_thresholds = None


LOGGER = logging.getLogger("analyse_preds")

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_model(ckpt_path: Path, device: torch.device) -> MVFoulsModel:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}).get("model_config", {})

    model = build_multi_task_model(
        backbone_arch=cfg.get("backbone_arch", "swin"),
        backbone_pretrained=False,
        backbone_freeze_mode="none",
        backbone_checkpointing=False,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    LOGGER.info("Model loaded (epoch %s, best %s)", ckpt.get("epoch"), ckpt.get("best_metric"))
    return model


def predict_distribution(
    model: MVFoulsModel,
    loader: DataLoader,
    device: torch.device,
    thresholds_dict: Dict[str, list] = None,
) -> Dict[str, np.ndarray]:
    meta = get_task_metadata()
    n_classes = meta["num_classes"]
    task_names = meta["task_names"]
    counts = {t: np.zeros(nc, dtype=int) for t, nc in zip(task_names, n_classes)}

    # Log threshold usage
    if thresholds_dict:
        LOGGER.info("🎯 Using threshold-based predictions for %d tasks", len(thresholds_dict))
        for task_name, task_thresholds in thresholds_dict.items():
            LOGGER.info("   %s: %d thresholds (range: [%.4f, %.4f])", 
                       task_name, len(task_thresholds), min(task_thresholds), max(task_thresholds))
    else:
        LOGGER.info("📊 Using standard argmax predictions")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            # Support both single-clip (videos, targets) and bag-of-clips
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                videos = batch[0]
            else:
                videos = batch  # fallback
            videos = videos.to(device)
            
            # Get predictions using thresholds if available
            if thresholds_dict:
                # Use model.predict() with thresholds
                result = model.predict(videos, thresholds=thresholds_dict, return_probs=False)
                predictions_dict = result['predictions']
                for t in task_names:
                    if t in predictions_dict:
                        preds = predictions_dict[t].cpu().numpy()
                        for p in preds:
                            if 0 <= p < len(counts[t]):  # Valid prediction
                                counts[t][p] += 1
            else:
                # Standard argmax predictions
                logits_dict, _ = model(videos, return_dict=True)
                for t in task_names:
                    preds = torch.argmax(logits_dict[t], dim=1).cpu().numpy()
                    for p in preds:
                        counts[t][p] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Analyse MVFouls model predictions distribution")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to model checkpoint (.pth)")
    parser.add_argument("--test-dir", required=True, type=str, help="Directory with test split videos")
    parser.add_argument("--test-annotations", required=True, type=str, help="Path to test annotations.json")
    parser.add_argument("--thresholds", type=str, default=None, help="Path to thresholds.json file for threshold-based predictions")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--bag-of-clips", action="store_true", help="Enable bag-of-clips mode")
    parser.add_argument("--max-clips-per-action", type=int, default=8)
    parser.add_argument("--min-clips-per-action", type=int, default=1)
    args = parser.parse_args()

    setup_logging()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)

    # Build test dataset & loader
    transforms = get_val_transforms()
    test_ds = MVFoulsDataset(
        root_dir=args.test_dir,
        split="test",
        transform=transforms,
        bag_of_clips=args.bag_of_clips,
        max_clips_per_action=args.max_clips_per_action,
        min_clips_per_action=args.min_clips_per_action,
        clip_sampling_strategy="uniform",
        load_annotations=True,
        annotations_dict=None,
        cache_mode="none",
    )

    collate_fn = bag_of_clips_collate_fn if args.bag_of_clips else None
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = load_model(Path(args.checkpoint), device)

    # Load thresholds if provided
    thresholds_dict = None
    if args.thresholds:
        if load_thresholds is None or extract_all_task_thresholds is None:
            LOGGER.warning("⚠️ Thresholding utilities not available, falling back to argmax predictions")
        else:
            try:
                thresholds_data = load_thresholds(args.thresholds)
                thresholds_dict = extract_all_task_thresholds(thresholds_data)
                LOGGER.info("✅ Loaded thresholds for %d tasks from %s", len(thresholds_dict), args.thresholds)
            except Exception as e:
                LOGGER.error("❌ Failed to load thresholds: %s", e)
                LOGGER.warning("⚠️ Falling back to argmax predictions")

    counts = predict_distribution(model, test_loader, device, thresholds_dict)

    meta = get_task_metadata()
    for task, arr in counts.items():
        LOGGER.info("\n=== %s ===", task)
        labels = meta["class_names"][task]
        total = arr.sum()
        for idx, c in enumerate(arr):
            LOGGER.info("%-25s : %5d  (%.2f%%)", labels[idx], c, 100 * c / max(1, total))

    LOGGER.info("Done.")


if __name__ == "__main__":
    main() 