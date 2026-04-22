# Apple Detection Project - Parameter Tuning Guide

## Overview
This document explains the tuned parameters created for the apple detection project. Three configuration files have been generated:
- **config.yaml** - Human-readable YAML format with detailed comments
- **config.json** - JSON format for programmatic access
- **config.py** - Python module for direct import

## Key Tuning Decisions

### Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 0.005 | Reduced from 0.02 for better convergence and stability with limited data (20 training samples) |
| **Epochs** | 100 | Increased from 50 to allow more thorough training with limited data |
| **Batch Size** | 2 | Kept small to prevent overfitting and due to memory constraints |
| **LR Schedule** | [30, 60, 80] | Gradual learning rate decay at reasonable intervals for 100 epochs |
| **Weight Decay** | 1e-4 | Standard L2 regularization to prevent overfitting |
| **Momentum** | 0.9 | Standard SGD momentum for convergence acceleration |

### Model Architecture

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Model** | Mask-RCNN | Better for small objects (apples) with mask generation capability |
| **Backbone** | ResNet50 | Good balance between performance and speed |
| **Pretrained** | True | Critical with limited data - transfer learning from COCO dataset |
| **FPN** | Enabled | Feature Pyramid Network improves detection of objects at different scales |

### Regularization

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Dropout** | 0.5 | Prevents co-adaptation in dense layers |
| **L2 Regularization** | 1e-4 | Prevents weight explosion and overfitting |
| **Data Augmentation** | Enabled | Critical with only 20 training samples |

### Data Augmentation Settings

- **Horizontal Flip**: 50% probability
- **Random Rotation**: ±15 degrees
- **Color Jitter**: Brightness 0.2, Contrast 0.2, Saturation 0.2
- Reason: Increases effective training data size

### Inference Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Confidence Threshold** | 0.5 | Balanced precision/recall for apple detection |
| **NMS Threshold** | 0.5 | Standard for removing duplicate detections |
| **Max Detections** | 100 | Sufficient for typical image |

## How to Use the Configuration Files

### Option 1: Using Python Config Module
```python
from config import load_config, TRAINING, MODEL

# Load all config
all_config = load_config("all")

# Load specific section
training_config = load_config("training")

# Or access directly
learning_rate = TRAINING["learning_rate"]
model_type = MODEL["backbone"]
```

### Option 2: Using YAML Config
```python
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

learning_rate = config['training']['learning_rate']
```

### Option 3: Using JSON Config
```python
import json

with open('config.json', 'r') as f:
    config = json.load(f)

learning_rate = config['training']['learning_rate']
```

### Option 4: Command Line Arguments
You can still override config values via command line:
```bash
python train_apples.py --epochs 100 --learning-rate 0.005 --batch-size 2
```

## Recommended Training Command

```bash
python train_apples.py \
    --model maskrcnn \
    --epochs 100 \
    --batch-size 2 \
    --learning-rate 0.005 \
    --lr-steps 30 60 80 \
    --lr-gamma 0.1 \
    --data-path ./detection \
    --output-dir ./checkpoints \
    --device cuda
```

## Important Notes

1. **Limited Training Data**: With only 20 training samples, focus on:
   - Data augmentation (already configured)
   - Transfer learning (using pre-trained COCO weights)
   - Early stopping to prevent overfitting

2. **Memory Considerations**:
   - Batch size is set to 2 to fit on most GPUs
   - Adjust if you have more/less VRAM available

3. **Validation**:
   - Validation frequency set to every 5 epochs
   - Monitor validation metrics to avoid overfitting

4. **Checkpoint Saving**:
   - Models saved every 5 epochs
   - Early stopping after 15 epochs without improvement

## Performance Tuning Tips

### If Model is Overfitting:
- Increase weight_decay to 5e-4
- Increase dropout_rate to 0.7
- Reduce learning_rate to 0.002
- Enable label_smoothing (0.1)

### If Model is Underfitting:
- Increase learning_rate to 0.01
- Reduce weight_decay to 5e-5
- Reduce dropout_rate to 0.3
- Increase epochs to 150

### For Faster Inference:
- Reduce image_max_size to 768
- Increase confidence_threshold to 0.6
- Increase nms_threshold to 0.6

### For Better Accuracy:
- Increase image_max_size to 1280
- Decrease confidence_threshold to 0.4
- Decrease nms_threshold to 0.4
- Increase epochs to 150

## Configuration Files Location

All configuration files are saved in the project root:
```
apple_detector/
├── config.py       # Python configuration module
├── config.json     # JSON format config
├── config.yaml     # YAML format config (with comments)
├── train_apples.py
├── predict_apples.py
└── ...
```

## Next Steps

1. Review and adjust parameters based on your specific needs
2. Run training with the recommended command above
3. Monitor training/validation curves
4. Adjust parameters if needed based on performance
5. Use the trained model for inference
