#!/usr/bin/env python
"""
Quick Start Guide - Apple Detection with Tuned Parameters
This script provides ready-to-run examples using the tuned configuration.
"""

import os
import sys
import subprocess

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import load_config, TRAINING, INFERENCE, SYSTEM

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def run_command(cmd, description):
    """Run a command and print feedback"""
    print(f"\n{description}")
    print(f"Command: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    print_section("APPLE DETECTION - QUICK START GUIDE")
    
    print("\nThis guide shows you how to train and run predictions with the tuned")
    print("parameters from config.py, config.json, and config.yaml")
    
    # Show current config
    print_section("CURRENT TUNED CONFIGURATION")
    print(f"Model Type: {TRAINING['model_type'].upper()}")
    print(f"Learning Rate: {TRAINING['learning_rate']}")
    print(f"Epochs: {TRAINING['epochs']}")
    print(f"Batch Size: {TRAINING['batch_size']}")
    print(f"LR Schedule: {TRAINING['lr_steps']}")
    print(f"Device: {SYSTEM['device']}")
    
    # Training examples
    print_section("EXAMPLE 1: BASIC TRAINING")
    print("Run training with default tuned parameters:")
    train_cmd_basic = (
        "python train_apples.py"
    )
    print(f"  {train_cmd_basic}\n")
    
    print_section("EXAMPLE 2: CUSTOM TRAINING")
    print("Run training with custom checkpoint output:")
    train_cmd_custom = (
        "python train_apples.py "
        "--epochs 100 "
        "--batch-size 2 "
        "--learning-rate 0.005 "
        "--output-dir ./tuned_checkpoints"
    )
    print(f"  {train_cmd_custom}\n")
    
    print_section("EXAMPLE 3: TRAINING ON GPU")
    print("Run training on GPU (default):")
    train_cmd_gpu = (
        "python train_apples.py "
        "--device cuda "
        "--epochs 100"
    )
    print(f"  {train_cmd_gpu}\n")
    
    print_section("EXAMPLE 4: PREDICTION")
    print("Run predictions on test images:")
    predict_cmd = (
        "python predict_apples.py "
        "--weight-file ./checkpoints/model_99.pth "
        "--data-path ./detection/test "
        "--output-file ./predictions.csv"
    )
    print(f"  {predict_cmd}\n")
    
    print_section("CONFIGURATION FILE FORMATS")
    print("""
Three configuration formats are available:

1. PYTHON MODULE (config.py)
   - Import directly: from config import TRAINING, MODEL
   - Best for: Integration into Python code
   - Example:
       from config import load_config
       config = load_config('training')
       lr = config['learning_rate']

2. YAML FORMAT (config.yaml)
   - Human-readable with comments
   - Best for: Manual review and editing
   - Example:
       import yaml
       with open('config.yaml') as f:
           config = yaml.safe_load(f)

3. JSON FORMAT (config.json)
   - Machine-readable, language-independent
   - Best for: Sharing between systems
   - Example:
       import json
       with open('config.json') as f:
           config = json.load(f)
    """)
    
    print_section("USING CONFIG IN YOUR CODE")
    print("""
Method 1: Load specific section
    from config import load_config
    training = load_config('training')
    
Method 2: Load all config
    from config import get_all_config
    config = get_all_config()
    
Method 3: Access directly
    from config import TRAINING, MODEL, INFERENCE
    lr = TRAINING['learning_rate']
    model = MODEL['backbone']
    threshold = INFERENCE['confidence_threshold']
    """)
    
    print_section("KEY TUNING DECISIONS")
    print("""
✓ Learning Rate (0.005)
  - Reduced from default 0.02 for better convergence
  - Essential for training with limited data

✓ Epochs (100)
  - Increased to 100 for thorough training
  - With early stopping after 15 epochs without improvement

✓ Data Augmentation
  - Horizontal flips, rotation, color jittering
  - Critical with only 20 training samples

✓ Model: Mask-RCNN
  - Better for small objects like apples
  - Pre-trained on COCO for transfer learning

✓ Regularization
  - Weight decay and dropout to prevent overfitting
  - Conservative settings for small dataset
    """)
    
    print_section("TUNING RECOMMENDATIONS")
    print("""
If OVERFITTING:
  - Increase weight_decay to 5e-4
  - Increase dropout_rate to 0.7
  - Reduce learning_rate to 0.002
  - Edit config.py and modify REGULARIZATION dict

If UNDERFITTING:
  - Increase learning_rate to 0.01
  - Reduce weight_decay to 5e-5
  - Increase epochs to 150
  - Edit config.py and modify TRAINING dict

For FASTER TRAINING:
  - Reduce epochs to 50
  - Reduce workers to 2
  - Use batch_size 4 (if GPU memory allows)

For BETTER ACCURACY:
  - Increase epochs to 150
  - Reduce learning_rate to 0.002
  - Increase workers to 8
  - Increase batch_size (if memory allows)
    """)
    
    print_section("NEXT STEPS")
    print("""
1. Review the configuration:
   - Open config.py to see all parameters
   - Open PARAMETERS_README.md for detailed explanations

2. Start training:
   python train_apples.py

3. Monitor training:
   - Check losses in terminal output
   - Losses should decrease over time

4. Run predictions:
   python predict_apples.py --weight-file ./checkpoints/model_99.pth

5. Evaluate results:
   - Check predictions.csv for results
   - Compare with ground truth
    """)
    
    print_section("IMPORTANT FILES")
    print("""
Configuration Files:
  • config.py           - Python configuration module (primary)
  • config.yaml         - YAML format with comments
  • config.json         - JSON format
  
Documentation:
  • PARAMETERS_README.md  - Detailed parameter explanations
  • QUICK_START.py        - This file
  
Training/Prediction:
  • train_apples.py       - Updated to use config defaults
  • predict_apples.py     - Updated to use config
  • MinneApple/train_rcnn.py   - Core training logic
  • MinneApple/predict_rcnn.py - Core prediction logic
    """)
    
    print_section("HELP & DOCUMENTATION")
    print("""
For detailed parameter information:
  • Read PARAMETERS_README.md
  • See inline comments in config.py
  • Check config.yaml for organized sections

For training help:
  python train_apples.py --help

For prediction help:
  python predict_apples.py --help
    """)
    
    print_section("END OF QUICK START")
    print("\nYou're all set! Start training with:\n  python train_apples.py\n")

if __name__ == "__main__":
    main()
