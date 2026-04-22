"""
Apple Detection Project - Configuration Module
Tuned parameters for Mask-RCNN and Faster-RCNN training and inference
"""

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING = {
    "model_type": "maskrcnn",  # Options: "maskrcnn", "frcnn"
    "pretrained": True,
    "data_path": "./detection",
    "output_dir": "./checkpoints",
    "epochs": 5,  # Reduced from 100 for CPU training (test quickly first)
    "lr_steps": [2, 3, 4],  # Adjusted for 5 epochs
    "lr_gamma": 0.1,
    "batch_size": 1,  # Reduced from 2 for CPU training
    "workers": 2,  # Reduced from 4
    "optimizer": "sgd",
    "learning_rate": 0.005,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "print_freq": 1,  # Print every iteration for feedback
    "save_freq": 1,  # Save checkpoint every epoch
    "early_stopping_patience": 10,
    "gradient_clip_norm": 1.0,
}

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
AUGMENTATION = {
    "horizontal_flip_prob": 0.5,
    "random_rotation": True,
    "rotation_degrees": 15,
    "color_jitter": True,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================
MODEL = {
    "backbone": "resnet50",
    "fpn_enabled": True,
    "num_classes": 2,
    "roi_batch_size_per_image": 512,
    "pos_fraction": 0.25,
    "score_thresh": 0.05,
    "nms_thresh": 0.5,
    "detections_per_image": 100,
    "rpn_fg_iou_thresh": 0.7,
    "rpn_bg_iou_thresh": 0.3,
    "rpn_num_samples": 256,
    "rpn_positive_fraction": 0.5,
    "rpn_pre_nms_top_n": {
        "train": 2000,
        "test": 1000,
    },
    "rpn_post_nms_top_n": {
        "train": 2000,
        "test": 1000,
    },
}

# ============================================================================
# INFERENCE/PREDICTION CONFIGURATION
# ============================================================================
INFERENCE = {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.5,
    "max_detections_per_image": 100,
    "device": "cuda",
    "batch_size": 1,
    "mask_threshold": 0.5,
    "mask_area_min": 10,
}

# ============================================================================
# REGULARIZATION
# ============================================================================
REGULARIZATION = {
    "dropout_rate": 0.5,
    "l2_regularization": 1e-4,
    "use_mixup": False,
    "mixup_alpha": 0.2,
    "label_smoothing": 0.0,
    "stochastic_depth": False,
}

# ============================================================================
# EVALUATION
# ============================================================================
EVALUATION = {
    "val_freq": 5,
    "metrics": ["mAP", "precision", "recall", "f1_score"],
    "iou_thresholds": [0.5, 0.75],
    "use_coco_metrics": True,
}

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================
SYSTEM = {
    "device": "cpu",  # Changed from "cuda" - PyTorch not compiled with CUDA
    "num_workers": 2,  # Reduced from 4 for CPU training
    "pin_memory": False,  # False for CPU training
    "mixed_precision": False,
    "seed": 42,
    "cuda_benchmark": False,  # False for CPU training
}

# ============================================================================
# DATA SUBSET SETTINGS (for quick testing)
# ============================================================================
DATA_SUBSET = {
    "use_subset": False,
    "train_subset_size": 20,
    "test_subset_size": 10,
}

# ============================================================================
# DATASET PARAMETERS
# ============================================================================
DATASET = {
    "dataset_type": "detection",
    "image_format": "jpg",
    "mask_format": "png",
    "image_min_size": 512,
    "image_max_size": 1024,
    "aspect_ratio_group_factor": 3,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_all_config():
    """Return all configuration as a single dictionary"""
    return {
        "training": TRAINING,
        "augmentation": AUGMENTATION,
        "model": MODEL,
        "inference": INFERENCE,
        "regularization": REGULARIZATION,
        "evaluation": EVALUATION,
        "system": SYSTEM,
        "data_subset": DATA_SUBSET,
        "dataset": DATASET,
    }


def load_config(config_type="all"):
    """
    Load specific configuration
    
    Args:
        config_type: "training", "model", "inference", "all", etc.
    
    Returns:
        Configuration dictionary
    """
    all_config = get_all_config()
    
    if config_type == "all":
        return all_config
    elif config_type in all_config:
        return all_config[config_type]
    else:
        raise ValueError(f"Unknown config type: {config_type}")


if __name__ == "__main__":
    # Print all configurations
    import json
    config = get_all_config()
    print(json.dumps(config, indent=2))
