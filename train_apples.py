#!/usr/bin/env python
"""
Simplified training wrapper for apple detection on local dataset.
Uses Mask-RCNN by default on the detection/train folder.
"""
import os
import sys
import argparse

# Add MinneApple to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MinneApple'))

from MinneApple.train_rcnn import main

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train apple detector on local dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and output
    parser.add_argument(
        '--data-path',
        default='./detection',
        help='Path to dataset folder (should contain train/test subdirs)'
    )
    parser.add_argument(
        '--output-dir',
        default='./checkpoints',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--model',
        choices=['maskrcnn', 'frcnn', 'mrcnn'],
        default='maskrcnn',
        help='Model type: maskrcnn (Mask-RCNN), frcnn (Faster-RCNN), mrcnn (alias for maskrcnn)'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        '-b',
        type=int,
        default=2,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.02,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGD momentum'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay for L2 regularization'
    )
    parser.add_argument(
        '--lr-steps',
        nargs='+',
        type=int,
        default=[8, 11],
        help='Epochs at which to decay learning rate'
    )
    parser.add_argument(
        '--lr-gamma',
        type=float,
        default=0.1,
        help='Learning rate decay factor'
    )
    
    # System
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to use for training'
    )
    parser.add_argument(
        '--workers',
        '-j',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--print-freq',
        type=int,
        default=20,
        help='Print frequency (iterations)'
    )
    parser.add_argument(
        '--resume',
        default='',
        help='Path to checkpoint to resume training from'
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert aliases
    if args.model == 'mrcnn':
        args.model = 'maskrcnn'
    
    # Validate model choice
    assert args.model in ['maskrcnn', 'frcnn'], \
        f"Model must be 'maskrcnn' or 'frcnn', got '{args.model}'"
    
    # Map argument names to expected format
    args.lr = args.learning_rate
    
    print(f"Training on {args.data_path} with {args.model}")
    print(f"Saving checkpoints to {args.output_dir}")
    
    main(args)