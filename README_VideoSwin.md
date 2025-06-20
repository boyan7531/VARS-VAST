# Video Swin Transformer for MVFouls Dataset

This implementation provides a **Video Swin Transformer Base (Swin-B)** with custom classification heads for the MVFouls video classification dataset.

## 🏗️ Architecture Overview

The implementation consists of three main components:

1. **Video Swin Transformer Backbone** (`model/video_swin_transformer.py`)
   - 3D patch embedding for video data
   - Simplified transformer layers with multi-head attention
   - Configurable depth and embedding dimensions
   - Support for pre-trained weight loading

2. **Classification Heads** (`model/classification_heads.py`)
   - Simple classification head for single-task learning
   - Multi-task classification head for multiple tasks
   - Severity regression head for continuous predictions

3. **Complete Model** (`model/mvfouls_model.py`)
   - Combines backbone and heads
   - Supports multi-task learning
   - Backbone freezing/unfreezing capabilities
   - Factory functions for easy model creation

## 🚀 Quick Start

### 1. Test the Implementation

```bash
# Run the test suite to verify everything works
python test_model.py
```

### 2. Basic Usage

```python
from model import create_mvfouls_model
import torch

# Create a Video Swin Transformer model
task_configs = {
    'action_class': 8,
    'offence': 4,
    'severity': 1  # Regression task
}

model = create_mvfouls_model(
    task_configs=task_configs,
    pretrained=False,  # Set to True for pre-trained weights
    freeze_backbone=False
)

# Forward pass
batch_size = 2
video_input = torch.randn(batch_size, 3, 32, 224, 224)  # (B, C, T, H, W)

with torch.no_grad():
    outputs = model(video_input)

print("Model outputs:")
for task_name, output in outputs.items():
    print(f"  {task_name}: {output.shape}")
```

### 3. Using with MVFouls Dataset

```python
from dataset import MVFoulsDataset
from transforms import get_train_transforms, get_val_transforms
from model import create_mvfouls_model_from_dataset

# Create dataset with transforms
train_transform = get_train_transforms(size=224)
train_dataset = MVFoulsDataset(
    root_dir="mvfouls",
    split='train',
    clip_selection='first',
    transform=train_transform
)

# Create model based on dataset
model = create_mvfouls_model_from_dataset(
    dataset=train_dataset,
    pretrained=True,
    freeze_backbone=False
)

print(f"Model created with {model.get_trainable_parameters()['total']:,} parameters")
```

## 📊 Model Configuration

### Default Configuration (Memory Efficient)

- **Embedding Dimension**: 96
- **Depths**: (2, 2, 6, 2)
- **Number of Heads**: (3, 6, 12, 24)
- **Patch Size**: (4, 8, 8) - Temporal, Height, Width
- **Input Size**: 32 frames × 224×224 pixels
- **Parameters**: ~10M

### Full Swin-B Configuration

For production use with more GPU memory:

```python
from model.video_swin_transformer import VideoSwinConfig
from model import MVFoulsModelConfig, MVFoulsVideoModel

# Full Swin-B configuration
backbone_config = VideoSwinConfig(
    embed_dim=128,
    depths=(2, 2, 18, 2),  # Full Swin-B
    num_heads=(4, 8, 16, 32),
    patch_size=(2, 4, 4),  # Smaller patches for better accuracy
    pretrained_2d=True
)

model_config = MVFoulsModelConfig(
    backbone_config=backbone_config,
    task_configs={'action_class': 10},
    freeze_backbone=False
)

model = MVFoulsVideoModel(model_config)
```

## 🎯 Supported Tasks

The model supports various MVFouls dataset tasks:

- **Action Classification**: Main action type classification
- **Offence Detection**: Type of offence classification
- **Contact Analysis**: Contact type classification
- **Body Part Detection**: Which body part was involved
- **Binary Classifications**: Try to play, Touch ball, Handball
- **Severity Regression**: Continuous severity score (0-1)

## 🔧 Training Features

### Backbone Freezing

```python
# Freeze backbone for transfer learning
model.freeze_backbone()

# Unfreeze for fine-tuning
model.unfreeze_backbone()

# Check trainable parameters
params = model.get_trainable_parameters()
print(f"Backbone: {params['backbone']:,}")
print(f"Heads: {params['heads']:,}")
print(f"Total: {params['total']:,}")
```

### Multi-Task Learning

The model automatically handles multi-task learning when multiple tasks are specified:

```python
task_configs = {
    'action_class': 8,
    'offence': 4,
    'contact': 3,
    'severity': 1
}

model = create_mvfouls_model(task_configs=task_configs)

# Forward pass returns outputs for all tasks
outputs = model(video_input)
# outputs = {
#     'action_class': tensor([batch_size, 8]),
#     'offence': tensor([batch_size, 4]),
#     'contact': tensor([batch_size, 3]),
#     'severity': tensor([batch_size, 1])
# }
```

## 📝 Data Format

The model expects video data in the following format:

- **Input Shape**: `(B, C, T, H, W)`
  - B: Batch size
  - C: Channels (3 for RGB)
  - T: Temporal frames (32)
  - H, W: Height, Width (224, 224)
- **Data Type**: `torch.float32`
- **Value Range**: Normalized [0, 1] or ImageNet normalized

Your transforms handle the conversion from the dataset format `(T, H, W, C)` to model format `(C, T, H, W)`.

## 🔄 Transform Pipeline

The transform pipeline is designed for Video Swin Transformer:

```python
from transforms import get_train_transforms, get_val_transforms

# Training transforms (with augmentation)
train_transform = get_train_transforms(size=224)
# 1. Resize to ~256
# 2. Random crop to 224
# 3. Random horizontal flip
# 4. Convert to tensor (C, T, H, W)
# 5. ImageNet normalization

# Validation transforms (no augmentation)
val_transform = get_val_transforms(size=224)
# 1. Resize to ~256
# 2. Center crop to 224
# 3. Convert to tensor (C, T, H, W)
# 4. ImageNet normalization
```

## 💡 Tips for Usage

### 1. Memory Management

- Start with the default configuration (memory efficient)
- Use smaller batch sizes for larger models
- Consider gradient checkpointing for very deep models

### 2. Transfer Learning Strategy

```python
# Stage 1: Freeze backbone, train heads only
model = create_mvfouls_model(freeze_backbone=True)
# Train for a few epochs...

# Stage 2: Unfreeze and fine-tune end-to-end
model.unfreeze_backbone()
# Continue training with lower learning rate...
```

### 3. Multi-Task Loss Weighting

```python
# Example loss computation for multi-task learning
losses = {}
loss_weights = {
    'action_class': 1.0,
    'offence': 0.5,
    'severity': 2.0
}

for task, pred in predictions.items():
    if task == 'severity':
        loss = nn.MSELoss()(pred, targets[task])
    else:
        loss = nn.CrossEntropyLoss()(pred, targets[task])
    
    losses[task] = loss * loss_weights.get(task, 1.0)

total_loss = sum(losses.values())
```

## 📁 File Structure

```
model/
├── __init__.py                 # Package initialization
├── video_swin_transformer.py   # Video Swin Transformer backbone
├── classification_heads.py     # Classification and regression heads
└── mvfouls_model.py            # Complete model implementation

transforms.py                   # Video transforms for preprocessing
dataset.py                      # MVFouls dataset loader
test_model.py                   # Test suite
example_usage.py               # Usage examples
requirements.txt               # Dependencies
```

## 🔧 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
numpy>=1.21.0
tqdm>=4.64.0
Pillow>=8.3.0
timm>=0.9.0
transformers>=4.20.0
```

## 🎯 Next Steps

1. **Download the MVFouls dataset** using your `download_dataset.py` script
2. **Run the example** with `python example_usage.py`
3. **Experiment** with different configurations and hyperparameters
4. **Implement actual pre-trained weight loading** from Kinetics-600
5. **Add more sophisticated augmentations** for better performance

## 🔬 Advanced Usage

### Custom Classification Head

```python
from model.classification_heads import SimpleClassificationHead

# Create custom head
custom_head = SimpleClassificationHead(
    in_features=model.backbone_features,
    num_classes=your_num_classes,
    dropout=0.5,
    hidden_dim=512
)

# Replace existing head
model.heads['custom_task'] = custom_head
```

### Model Ensemble

```python
from model.classification_heads import EnsembleHead

ensemble_head = EnsembleHead(
    in_features=model.backbone_features,
    num_classes=10,
    num_heads=3
)
```

This implementation provides a solid foundation for video classification with the MVFouls dataset using Video Swin Transformer architecture! 