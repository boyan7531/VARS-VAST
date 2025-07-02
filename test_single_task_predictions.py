import torch
import torch.nn.functional as F
import os
from model.mvfouls_model import build_single_task_model

print('🧪 Testing single-task model predictions...')

# Create model
model = build_single_task_model(
    num_classes=2, 
    backbone_arch='mvitv2_b',
    freeze_mode='none'
)

# Try to load trained weights with proper security settings
checkpoint_path = 'outputs/debug_single_task/best_model_latest.pth'
trained = False

if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('✅ Loaded trained weights from epoch', checkpoint.get('epoch', '?'))
        print('📊 Best metric:', checkpoint.get('best_metric', '?'))
        trained = True
    except Exception as e:
        print('⚠️  Failed to load trained weights:', str(e))
        trained = False
else:
    print('⚠️  No checkpoint found - using random weights')
    trained = False

model.eval()

# Test with multiple dummy inputs to see prediction diversity
print('\n🎯 Testing prediction diversity...')
dummy_input = torch.randn(8, 3, 32, 224, 224)

with torch.no_grad():
    # Single-task model returns (logits, extras) tuple
    outputs, extras = model(dummy_input)
    probs = F.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    print(f'Batch predictions: {preds.tolist()}')
    print(f'Unique predictions: {torch.unique(preds).tolist()}')
    print(f'Class distribution: Class 0: {(preds == 0).sum().item()}/8, Class 1: {(preds == 1).sum().item()}/8')
    
    print('\nDetailed per-sample predictions:')
    for i in range(8):
        pred = preds[i]
        prob_0, prob_1 = probs[i][0], probs[i][1]
        confidence = max(prob_0, prob_1)
        print(f'  Sample {i}: Pred={pred.item()}, Class 0: {prob_0:.3f}, Class 1: {prob_1:.3f} (conf: {confidence:.1%})')

# Test with different random seeds to ensure variety
print('\n🔄 Testing with different random seeds...')
unique_preds_across_seeds = set()

for seed in [42, 123, 999]:
    torch.manual_seed(seed)
    test_input = torch.randn(4, 3, 32, 224, 224)
    
    with torch.no_grad():
        outputs, extras = model(test_input)
        preds = torch.argmax(outputs, dim=1)
        unique_preds_across_seeds.update(preds.tolist())
        print(f'  Seed {seed}: {preds.tolist()}')

print(f'\n📈 Summary:')
print(f'  Model status: {"Trained" if trained else "Random weights"}')
print(f'  Unique predictions across all tests: {sorted(unique_preds_across_seeds)}')
print(f'  Can predict multiple classes: {"✅ YES" if len(unique_preds_across_seeds) > 1 else "❌ NO - One-class prediction issue"}')

if len(unique_preds_across_seeds) == 1:
    print(f'  ⚠️  Model always predicts class {list(unique_preds_across_seeds)[0]}')
else:
    print(f'  🎉 Model successfully predicts both classes!')

# Additional test: Check if model weights are actually different from random initialization
if trained:
    print('\n🔍 Verifying model was actually trained...')
    # Create a fresh random model for comparison
    random_model = build_single_task_model(num_classes=2, backbone_arch='mvitv2_b', freeze_mode='none')
    random_model.eval()
    
    with torch.no_grad():
        trained_out, _ = model(dummy_input[:1])  # Single sample
        random_out, _ = random_model(dummy_input[:1])
        
        diff = torch.abs(trained_out - random_out).mean().item()
        print(f'  Output difference from random model: {diff:.6f}')
        print(f'  Model appears trained: {"✅ YES" if diff > 0.01 else "⚠️  MAYBE - very small difference"}') 