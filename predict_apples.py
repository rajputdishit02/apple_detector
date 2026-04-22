#!/usr/bin/env python
"""
Simplified prediction wrapper for apple detection on local dataset.
Uses local test images and generates predictions.
Configuration is loaded from config.py with tuned parameters.
"""
import os
import sys
import argparse

# Add MinneApple to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MinneApple'))

from MinneApple.predict_rcnn import main
from config import INFERENCE, SYSTEM

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run apple detector predictions on test dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and model
    parser.add_argument(
        '--data-path',
        default='./detection/test',
        help='Path to test images folder'
    )
    parser.add_argument(
        '--weight-file',
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--output-file',
        default='./predictions.csv',
        help='Output file for predictions (CSV format)'
    )
    parser.add_argument(
        '--model',
        choices=['maskrcnn', 'frcnn', 'mrcnn'],
        default='maskrcnn',
        help='Model type used: maskrcnn (Mask-RCNN), frcnn (Faster-RCNN), mrcnn (alias for maskrcnn)'
    )
    
    # System
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to use for inference'
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Validate weight file exists
    if not os.path.exists(args.weight_file):
        raise FileNotFoundError(f"Weight file not found: {args.weight_file}")
    
    # Validate data path exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")
    
    # Convert model aliases to expected format
    if args.model == 'mrcnn':
        args.mrcnn = True
        args.frcnn = False
    elif args.model == 'frcnn':
        args.frcnn = True
        args.mrcnn = False
    else:  # maskrcnn
        args.mrcnn = True
        args.frcnn = False
    
    print(f"Loading model from {args.weight_file}")
    print(f"Running predictions on {args.data_path}")
    print(f"Saving results to {args.output_file}")
    
    main(args)