# LDAM Loss Usage Guide

## Overview

LDAM (Label-Distribution-Aware Margin) loss has been successfully integrated into the MVFouls training pipeline. This loss function is particularly effective for imbalanced datasets as it applies larger margins to minority classes, improving their recall.

## How LDAM Works

LDAM modifies the standard cross-entropy loss by applying class-dependent margins to the logits:

- **Minority classes** get larger margins → harder to classify → better recall
- **Majority classes** get smaller margins → easier to classify → maintains precision
- Margins are computed as: `m_c = C / n_c^(1/4)` where `n_c` is the training count for class `c`

## Usage Examples

### Basic LDAM Usage

To use LDAM loss for the `offence` task (which is typically most imbalanced):

```bash
python train_with_class_weights.py \
  --train-dir mvfouls/train \
  --val-dir mvfouls/valid \
  --train-annotations mvfouls/train/annotations.json \
  --val-annotations mvfouls/valid/annotations.json \
  --multi-task \
  --epochs 20 \
  --batch-size 3 \
  --lr 5e-5 \
  --backbone-arch mvitv2_b \
  --loss-types action_class ce severity focal offence ldam \
  --output-dir outputs/ldam_experiment
```

### LDAM with Custom Parameters

You can adjust the LDAM parameters:

```bash
python train_with_class_weights.py \
  --train-dir mvfouls/train \
  --val-dir mvfouls/valid \
  --train-annotations mvfouls/train/annotations.json \
  --val-annotations mvfouls/valid/annotations.json \
  --multi-task \
  --loss-types action_class ldam severity ldam offence ldam \
  --ldam-max-m 0.3 \
  --ldam-s 20.0 \
  --epochs 20 \
  --output-dir outputs/all_ldam_experiment
```

### LDAM Combined with Deep Task-Specific Heads

For maximum effectiveness on rare classes like `offence`:

```bash
python train_with_class_weights.py \
  --train-dir mvfouls/train \
  --val-dir mvfouls/valid \
  --train-annotations mvfouls/train/annotations.json \
  --val-annotations mvfouls/valid/annotations.json \
  --multi-task \
  --loss-types action_class ce severity focal offence ldam \
  --task-head offence deep_mlp depth=3 hidden=1024 dropout=0.2 \
  --ldam-max-m 0.5 \
  --ldam-s 30.0 \
  --epochs 20 \
  --output-dir outputs/ldam_plus_deep_heads
```

## Parameters

### LDAM Loss Parameters

- `--ldam-max-m`: Maximum margin value (default: 0.5)
  - Higher values create larger margins but may destabilize training
  - Typical range: 0.1 - 1.0

- `--ldam-s`: Scale parameter (default: 30.0)  
  - Controls the overall scaling of the logits
  - Higher values make the loss more confident
  - Typical range: 10.0 - 50.0

### Loss Type Options

The `--loss-types` argument now accepts `ldam` in addition to:
- `ce`: Standard cross-entropy
- `focal`: Focal loss (good for moderate imbalance)
- `bce`: Binary cross-entropy
- `ldam`: LDAM loss (best for severe imbalance)

## Expected Benefits

Using LDAM loss, especially for the `offence` task, should provide:

1. **Higher macro-recall**: Better detection of rare classes
2. **Improved minority class performance**: Especially for "Between" and rare offence types
3. **Maintained overall accuracy**: LDAM preserves majority class performance

## Monitoring Training

When using LDAM, monitor these metrics:
- Per-class recall for each task (especially rare classes)
- Overall macro-recall and macro-F1
- Training stability (loss should converge smoothly)

## Troubleshooting

### Training Instability
If training becomes unstable with LDAM:
- Reduce `--ldam-max-m` (try 0.3 or 0.2)
- Reduce `--ldam-s` (try 20.0 or 15.0)
- Lower learning rate

### Poor Convergence
If loss doesn't converge well:
- Increase `--ldam-s` (try 40.0 or 50.0)
- Ensure class counts are computed correctly
- Check that you're not using both LDAM and class weights simultaneously

## Implementation Details

- LDAM automatically computes class-specific margins from training data imbalance
- Margins are computed once during model initialization
- Each task has its own LDAM instance with task-specific class distributions
- LDAM is compatible with all existing features (gradual unfreezing, adaptive LR, etc.) 