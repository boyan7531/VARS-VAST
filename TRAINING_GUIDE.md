# MVFouls Model Training Guide 🚀

This guide shows you how to train the MVFouls model using the complete training pipeline.

## Quick Start

### 0. Test Before Training (Recommended!)

**Before starting actual training, test if everything will work:**

```bash
# Quick pipeline test (fastest - ~30 seconds)
python test_training_pipeline.py \
  --train-dir ./mvfouls/train \
  --val-dir ./mvfouls/val \
  --train-annotations ./mvfouls/train.csv \
  --val-annotations ./mvfouls/val.csv \
  --multi-task

# Dry run mode (tests full pipeline without training)
python train.py \
  --train-dir ./mvfouls/train \
  --val-dir ./mvfouls/val \
  --train-annotations ./mvfouls/train.csv \
  --val-annotations ./mvfouls/val.csv \
  --multi-task \
  --dry-run

# Minimal training test (1 epoch, small batch)
python train.py \
  --train-dir ./mvfouls/train \
  --val-dir ./mvfouls/val \
  --train-annotations ./mvfouls/train.csv \
  --val-annotations ./mvfouls/val.csv \
  --multi-task \
  --epochs 1 \
  --batch-size 2 \
  --max-frames 8 \
  --num-workers 0
```

### 1. Basic Training Commands

**Multi-task training (recommended):**
```bash
python train.py --train-dir ./mvfouls/train --val-dir ./mvfouls/val --train-annotations ./mvfouls/train.csv --val-annotations ./mvfouls/val.csv --multi-task --epochs 50
```

**Single-task training:**
```bash
python train.py --train-dir ./mvfouls/train --val-dir ./mvfouls/val --train-annotations ./mvfouls/train.csv --val-annotations ./mvfouls/val.csv --num-classes 2 --epochs 50
```

**Using a config file:**
```bash
python train.py --config config_example.yaml
```

### 2. Common Training Scenarios

**Quick test run (small batch, few epochs):**
```bash
python train.py --train-dir ./mvfouls/train --val-dir ./mvfouls/val --train-annotations ./mvfouls/train.csv --val-annotations ./mvfouls/val.csv --multi-task --epochs 5 --batch-size 4
```

**Production training (with all optimizations):**
```bash
python train.py --config config_example.yaml --gradient-accumulation-steps 4 --max-grad-norm 1.0
```

**Resume from checkpoint:**
```bash
python train.py --config config_example.yaml --resume ./outputs/mvfouls_multi_20241201_123456/best_model.pth
```

## Configuration Options

### Data Settings
- `--train-dir`: Path to training video directory
- `--val-dir`: Path to validation video directory
- `--test-dir`: Path to test video directory (optional)
- `--challenge-dir`: Path to challenge video directory (optional)
- `--train-annotations`: Path to training CSV annotations file
- `--val-annotations`: Path to validation CSV annotations file
- `--test-annotations`: Path to test CSV annotations file (optional)
- `--challenge-annotations`: Path to challenge CSV annotations file (optional)
- `--max-frames`: Maximum frames per video (default: 16)
- `--fps`: Target FPS for processing (default: 8)
- `--image-size`: Input image size (default: 224)

### Model Settings
- `--multi-task`: Enable multi-task learning
- `--num-classes`: Number of classes (single-task only)
- `--backbone-arch`: Backbone architecture (swin, mvit, mvitv2_s, mvitv2_b)
- `--pretrained`: Use pretrained backbone (recommended)
- `--freeze-backbone`: Freeze backbone during training
- `--freeze-mode`: Freeze policy (none, freeze_all, gradual, freeze_stages{k})
- `--backbone-checkpointing`: Enable gradient checkpointing for memory efficiency

### Training Settings
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size (adjust based on GPU memory)
- `--lr`: Learning rate (default: 1e-4)
- `--optimizer`: Optimizer type (adamw, adam, sgd)
- `--scheduler`: Learning rate scheduler (cosine, step, plateau, warmup_cosine)
- `--gradient-accumulation-steps`: For effective larger batch sizes
- `--max-grad-norm`: Gradient clipping value

## Output Structure

After training, you'll find the following in your output directory:

```
outputs/mvfouls_multi_20241201_123456/
├── config.json              # Training configuration
├── training.log             # Detailed training logs
├── tensorboard/             # TensorBoard logs
├── best_model.pth          # Best model checkpoint
├── final_model.pth         # Final model checkpoint
└── checkpoint_epoch_*.pth  # Regular checkpoints
```

## Monitoring Training

### 1. Console Output
The script provides real-time training progress:
```
2024-12-01 12:34:56 - Starting training experiment: mvfouls_multi_20241201_123456
2024-12-01 12:34:57 - Device: cuda
2024-12-01 12:35:00 - Dataset created: 800 train, 200 val samples
Epoch 1/50 - Loss: 0.8234, Time: 45.2s, LR: 1.00e-04
```

### 2. TensorBoard
Launch TensorBoard to visualize training:
```bash
tensorboard --logdir ./outputs/mvfouls_multi_20241201_123456/tensorboard
```

### 3. Log Files
Check detailed logs in `training.log` for debugging and analysis.

## Memory and Performance Tips

### GPU Memory Optimization
- **Reduce batch size**: Start with `--batch-size 4` and increase as memory allows
- **Use gradient accumulation**: `--gradient-accumulation-steps 4` for effective batch size of 16
- **Reduce max frames**: `--max-frames 16` if videos are long
- **Enable gradient checkpointing**: `--backbone-checkpointing` to reduce VRAM usage by ~40%

### Training Speed
- **Increase num_workers**: `--num-workers 8` (match your CPU cores)
- **Use mixed precision**: The model automatically uses efficient operations
- **Freeze backbone initially**: `--freeze-backbone` for faster initial training

### Gradient Checkpointing 🔄
Gradient checkpointing trades compute for memory by recomputing activations during the backward pass instead of storing them.

**Benefits:**
- Reduces VRAM usage by ~40%
- Enables larger batch sizes or longer sequences
- Compatible with mixed-precision training

**Trade-offs:**
- Increases training time by ~10-20%
- Uses more compute (recomputes activations)

**When to use:**
- Limited GPU memory (< 12GB VRAM)
- Want to increase batch size for better training
- Training with long video sequences

**Recommended hyperparameter adjustments:**
- Increase batch size by 50-100% (e.g., 8 → 12-16)
- Increase gradient accumulation steps (e.g., 2 → 4)
- Consider using smaller, more efficient architectures (mvitv2_s vs swin)

**Usage:**
```bash
# Enable gradient checkpointing
python train.py --backbone-checkpointing [other args]

# Combine with other memory optimizations
python train.py \
  --backbone-arch mvitv2_s \
  --backbone-checkpointing \
  --batch-size 12 \
  --gradient-accumulation-steps 4
```

### Example Memory-Efficient Config
```bash
python train.py \
  --train-dir ./mvfouls/train \
  --val-dir ./mvfouls/val \
  --train-annotations ./mvfouls/train.csv \
  --val-annotations ./mvfouls/val.csv \
  --multi-task \
  --backbone-arch mvitv2_s \
  --backbone-checkpointing \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --max-frames 16 \
  --num-workers 4
```

## Advanced Features

### Adaptive Loss Configuration 🎯

The MVFouls training pipeline now supports rich, adaptive loss strategies to handle class imbalance better:

#### Per-Task Loss Types
Configure different loss functions for each task:
```bash
# Use Cross-Entropy for action_class, Focal Loss for severity, Cross-Entropy for offence
python train_with_class_weights.py \
  --train-dir ./mvfouls/train \
  --val-dir ./mvfouls/val \
  --train-annotations ./mvfouls/train.csv \
  --val-annotations ./mvfouls/val.csv \
  --multi-task \
  --loss-types action_class ce severity focal offence ce
```

#### Adaptive Task Weighting
Automatically adjust task weights based on validation performance:
```bash
# Tasks performing worse get higher weights during training
python train_with_class_weights.py \
  --multi-task \
  --adaptive-weights \
  --weighting-strategy inverse_accuracy \
  [... other args ...]
```

#### Effective Number Class Weights
For extreme class imbalance, use effective number weighting:
```bash
python train_with_class_weights.py \
  --multi-task \
  --effective-class-weights \
  [... other args ...]
```

#### Class Imbalance Strategies

| Strategy | Description | When to Use | CLI Flags |
|----------|-------------|-------------|-----------|
| **Balanced Sampling** | Equalizes class representation in batches | Moderate imbalance | `--balanced-sampling` |
| **Effective Class Weights** | Uses effective number of samples | Extreme imbalance | `--effective-class-weights` |
| **Focal Loss** | Focuses on hard examples | Many easy examples | `--loss-types task focal` |
| **Adaptive Weighting** | Adjusts task weights based on performance | Multi-task scenarios | `--adaptive-weights` |

#### Recommended Recipes

**1. Balanced Sampling Only (Simple):**
```bash
python train_with_class_weights.py \
  --multi-task \
  --balanced-sampling \
  --disable-class-weights \
  [... other args ...]
```

**2. Effective Weights + Focal Loss (Advanced):**
```bash
python train_with_class_weights.py \
  --multi-task \
  --effective-class-weights \
  --loss-types action_class ce severity focal offence ce \
  [... other args ...]
```

**3. Full Adaptive Strategy (Research):**
```bash
python train_with_class_weights.py \
  --multi-task \
  --effective-class-weights \
  --adaptive-weights \
  --weighting-strategy difficulty \
  --loss-types action_class ce severity focal offence ce \
  [... other args ...]
```

### Curriculum Learning
The trainer supports automatic curriculum learning for multi-task scenarios. Tasks are gradually introduced based on performance.

### Model Export
After training, export to ONNX for deployment:
```python
from model.mvfouls_model import MVFoulsModel

# Load trained model
model = MVFoulsModel(multi_task=True)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Export to ONNX
model.export_onnx('mvfouls_model.onnx')
```

### Custom Configurations
Create your own YAML config file based on `config_example.yaml` for reproducible experiments.

## Troubleshooting

### Testing Strategies

**🧪 Before reporting issues, run these tests:**

1. **Quick smoke test** (30 seconds):
   ```bash
   python test_training_pipeline.py --train-dir ./train --val-dir ./val --train-annotations train.csv --val-annotations val.csv --multi-task
   ```

2. **Dry run test** (2-3 minutes):
   ```bash
   python train.py --train-dir ./train --val-dir ./val --train-annotations train.csv --val-annotations val.csv --multi-task --dry-run
   ```

3. **Mini training test** (5-10 minutes):
   ```bash
   python train.py --train-dir ./train --val-dir ./val --train-annotations train.csv --val-annotations val.csv --multi-task --epochs 1 --batch-size 2 --max-frames 8
   ```

### Common Issues

**1. CUDA out of memory**
```bash
# Solution: Reduce batch size and use gradient accumulation
python train.py --batch-size 2 --gradient-accumulation-steps 8
```

**2. Dataset loading errors**
```bash
# Check paths and file formats
ls ./mvfouls/train/
ls ./mvfouls/val/
head ./mvfouls/train.csv
head ./mvfouls/val.csv
```

**3. Import errors**
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt
```

**4. Slow training**
```bash
# Increase workers and check GPU utilization
python train.py --num-workers 8
nvidia-smi  # Check GPU usage
```

## Results and Evaluation

The training script automatically:
- ✅ Saves the best model based on validation accuracy
- ✅ Computes comprehensive metrics for all tasks
- ✅ Generates confusion matrices
- ✅ Provides detailed performance tables
- ✅ Logs everything to TensorBoard

Check the validation results in the console output and TensorBoard for detailed analysis.

## Next Steps

After training:
1. **Evaluate**: Use the saved model for inference
2. **Deploy**: Export to ONNX for production use
3. **Fine-tune**: Resume training with different hyperparameters
4. **Analyze**: Review TensorBoard logs and confusion matrices

Happy training! 🎯 