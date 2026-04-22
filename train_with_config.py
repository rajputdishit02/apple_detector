#!/usr/bin/env python
"""
One-Command Training Script using Tuned Parameters
Simply run: python train_with_config.py
"""

import os
import sys
import argparse

# Add MinneApple to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MinneApple'))

from MinneApple.train_rcnn import main as train_main
from config import TRAINING, SYSTEM

def create_args_from_config():
    """Create argparse-like object from config"""
    class Args:
        pass
    
    args = Args()
    args.model = TRAINING['model_type']
    args.epochs = TRAINING['epochs']
    args.batch_size = TRAINING['batch_size']
    args.learning_rate = TRAINING['learning_rate']
    args.lr = TRAINING['learning_rate']  # Alias used by train_rcnn
    args.momentum = TRAINING['momentum']
    args.weight_decay = TRAINING['weight_decay']
    args.lr_steps = TRAINING['lr_steps']
    args.lr_gamma = TRAINING['lr_gamma']
    args.device = SYSTEM['device']
    args.workers = SYSTEM['num_workers']
    args.print_freq = TRAINING['print_freq']
    args.data_path = TRAINING['data_path']
    args.output_dir = TRAINING['output_dir']
    args.resume = ''
    
    return args

def main():
    print("=" * 70)
    print("  APPLE DETECTION TRAINING - USING TUNED PARAMETERS")
    print("=" * 70)
    print(f"\nConfiguration loaded from config.py:")
    print(f"  Model: {TRAINING['model_type'].upper()}")
    print(f"  Learning Rate: {TRAINING['learning_rate']}")
    print(f"  Epochs: {TRAINING['epochs']}")
    print(f"  Batch Size: {TRAINING['batch_size']}")
    print(f"  LR Steps: {TRAINING['lr_steps']}")
    print(f"  Device: {SYSTEM['device']}")
    
    print(f"\nStarting training with these parameters...")
    print(f"Output directory: {TRAINING['output_dir']}\n")
    
    # Create output directory
    os.makedirs(TRAINING['output_dir'], exist_ok=True)
    
    # Create args from config
    args = create_args_from_config()
    
    try:
        # Call training directly
        train_main(args)
        
        print("\n" + "=" * 70)
        print("  TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nCheckpoints saved to: {TRAINING['output_dir']}")
        print("\nTo run predictions, use:")
        print(f"  python predict_apples.py --weight-file {TRAINING['output_dir']}/model_<epoch>.pth")
    except Exception as e:
        print("\n" + "=" * 70)
        print("  TRAINING FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
