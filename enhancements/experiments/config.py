"""
Experimental Configuration for Enhanced nnFormer
==============================================

Configuration files for comparing baseline vs enhanced nnFormer with 
multi-scale cross-attention on the BraTS dataset.

Author: 210353V
Date: October 2025
"""

# Baseline nnFormer Configuration
BASELINE_CONFIG = {
    "model_type": "baseline_nnformer",
    "architecture": {
        "crop_size": [64, 128, 128],
        "embedding_dim": 192,
        "input_channels": 4,  # FLAIR, T1w, T1Gd, T2w
        "num_classes": 4,     # Background, edema, non-enhancing tumor, enhancing tumor
        "depths": [2, 2, 2, 2],
        "num_heads": [6, 12, 24, 48],
        "patch_size": [2, 4, 4],
        "window_size": [4, 4, 8, 4],
        "deep_supervision": True
    },
    "training": {
        "max_epochs": 1000,
        "initial_lr": 0.01,
        "weight_decay": 3e-5,
        "momentum": 0.99,
        "batch_size": 2,
        "patch_size": [64, 128, 128],
        "num_threads_for_batchgen_train": 8,
        "num_threads_for_batchgen_val": 4
    },
    "data_augmentation": {
        "do_elastic": True,
        "elastic_deform_alpha": (0., 900.),
        "elastic_deform_sigma": (9., 13.),
        "do_scaling": True,
        "scale_range": (0.85, 1.25),
        "do_rotation": True,
        "rotation_x": (-15. / 360 * 2. * 3.14159, 15. / 360 * 2. * 3.14159),
        "rotation_y": (-15. / 360 * 2. * 3.14159, 15. / 360 * 2. * 3.14159),
        "rotation_z": (-15. / 360 * 2. * 3.14159, 15. / 360 * 2. * 3.14159),
        "do_gamma": True,
        "gamma_range": (0.7, 1.4),
        "do_mirror": True,
        "mirror_axes": (0, 1, 2)
    },
    "evaluation": {
        "val_freq": 50,
        "save_freq": 50,
        "metrics": ["Dice", "HD95", "Sensitivity", "Specificity"],
        "classes": ["WT", "TC", "ET"]  # Whole Tumor, Tumor Core, Enhancing Tumor
    }
}

# Enhanced nnFormer Configuration
ENHANCED_CONFIG = {
    "model_type": "enhanced_nnformer",
    "architecture": {
        "crop_size": [64, 128, 128],
        "embedding_dim": 192,
        "input_channels": 4,
        "num_classes": 4,
        "depths": [2, 2, 2, 2],
        "num_heads": [6, 12, 24, 48],
        "patch_size": [2, 4, 4],
        "window_size": [4, 4, 8, 4],
        "deep_supervision": True,
        "enable_cross_attention": True  # Key enhancement
    },
    "training": {
        "max_epochs": 1000,
        "initial_lr": 0.01,
        "weight_decay": 3e-5,
        "momentum": 0.99,
        "batch_size": 2,
        "patch_size": [64, 128, 128],
        "num_threads_for_batchgen_train": 8,
        "num_threads_for_batchgen_val": 4,
        # Enhanced training parameters
        "cross_attention_warmup_epochs": 50,
        "cross_attention_lr_factor": 0.5,
        "boundary_loss_weight": 0.1,
        "consistency_loss_weight": 0.05
    },
    "data_augmentation": {
        "do_elastic": True,
        "elastic_deform_alpha": (0., 900.),
        "elastic_deform_sigma": (9., 13.),
        "do_scaling": True,
        "scale_range": (0.85, 1.25),
        "do_rotation": True,
        "rotation_x": (-15. / 360 * 2. * 3.14159, 15. / 360 * 2. * 3.14159),
        "rotation_y": (-15. / 360 * 2. * 3.14159, 15. / 360 * 2. * 3.14159),
        "rotation_z": (-15. / 360 * 2. * 3.14159, 15. / 360 * 2. * 3.14159),
        "do_gamma": True,
        "gamma_range": (0.7, 1.4),
        "do_mirror": True,
        "mirror_axes": (0, 1, 2),
        # Enhanced augmentation for cross-attention
        "do_additive_brightness": True,
        "additive_brightness_mu": 0.0,
        "additive_brightness_sigma": 0.1
    },
    "evaluation": {
        "val_freq": 50,
        "save_freq": 50,
        "metrics": ["Dice", "HD95", "Sensitivity", "Specificity", "Precision", "Recall"],
        "classes": ["WT", "TC", "ET"],
        # Enhanced evaluation
        "boundary_evaluation": True,
        "attention_visualization": True
    }
}

# Ablation Study Configurations
ABLATION_CONFIGS = {
    "no_cross_attention": {
        **ENHANCED_CONFIG,
        "model_type": "ablation_no_cross_attention",
        "architecture": {
            **ENHANCED_CONFIG["architecture"],
            "enable_cross_attention": False
        }
    },
    "cross_attention_single_scale": {
        **ENHANCED_CONFIG,
        "model_type": "ablation_single_scale",
        "architecture": {
            **ENHANCED_CONFIG["architecture"],
            "cross_attention_scales": [2]  # Only between scale 1 and 2
        }
    },
    "different_lr_factors": [
        {
            **ENHANCED_CONFIG,
            "model_type": f"ablation_lr_factor_{factor}",
            "training": {
                **ENHANCED_CONFIG["training"],
                "cross_attention_lr_factor": factor
            }
        } for factor in [0.1, 0.25, 0.5, 0.75, 1.0]
    ]
}

# Experiment Planning
EXPERIMENT_PLAN = {
    "phase_1_baseline": {
        "description": "Establish baseline performance",
        "config": BASELINE_CONFIG,
        "duration_epochs": 1000,
        "priority": "HIGH",
        "success_criteria": {
            "dice_wt": 0.85,
            "dice_tc": 0.75,
            "dice_et": 0.70
        }
    },
    "phase_2_enhanced": {
        "description": "Full enhanced model with cross-attention",
        "config": ENHANCED_CONFIG,
        "duration_epochs": 1000,
        "priority": "HIGH",
        "success_criteria": {
            "dice_wt_improvement": 0.02,  # 2% improvement over baseline
            "dice_tc_improvement": 0.03,  # 3% improvement over baseline
            "dice_et_improvement": 0.05   # 5% improvement over baseline
        }
    },
    "phase_3_ablation": {
        "description": "Ablation studies to validate components",
        "configs": ABLATION_CONFIGS,
        "duration_epochs": 500,
        "priority": "MEDIUM"
    }
}

# Dataset Configuration
DATASET_CONFIG = {
    "name": "BraTS2021",
    "task_id": "Task01_BrainTumour",
    "modalities": {
        0: "FLAIR",
        1: "T1w", 
        2: "T1Gd",
        3: "T2w"
    },
    "labels": {
        0: "background",
        1: "edema", 
        2: "non-enhancing tumor",
        3: "enhancing tumor"
    },
    "evaluation_regions": {
        "WT": [1, 2, 3],  # Whole Tumor: all tumor classes
        "TC": [2, 3],     # Tumor Core: non-enhancing + enhancing
        "ET": [3]         # Enhancing Tumor: only enhancing
    },
    "num_training": 484,
    "num_test": 266,
    "spacing": [1.0, 1.0, 1.0],  # Target spacing
    "intensity_properties": {
        "normalization": "z_score_per_case",
        "clip_values": [-5, 5]
    }
}

# Evaluation Metrics Configuration
METRICS_CONFIG = {
    "primary_metrics": {
        "dice": {
            "description": "Dice Similarity Coefficient",
            "better": "higher",
            "range": [0, 1],
            "threshold": 0.5
        },
        "hd95": {
            "description": "95th Percentile Hausdorff Distance",
            "better": "lower", 
            "unit": "mm"
        }
    },
    "secondary_metrics": {
        "sensitivity": {
            "description": "True Positive Rate",
            "better": "higher",
            "range": [0, 1]
        },
        "specificity": {
            "description": "True Negative Rate", 
            "better": "higher",
            "range": [0, 1]
        },
        "precision": {
            "description": "Positive Predictive Value",
            "better": "higher",
            "range": [0, 1]
        }
    },
    "efficiency_metrics": {
        "training_time": {
            "description": "Time per epoch",
            "unit": "minutes",
            "better": "lower"
        },
        "memory_usage": {
            "description": "Peak GPU memory usage",
            "unit": "GB",
            "better": "lower"
        },
        "inference_time": {
            "description": "Time per case inference",
            "unit": "seconds",
            "better": "lower"
        }
    }
}

# Hardware and Environment Configuration
ENVIRONMENT_CONFIG = {
    "hardware": {
        "min_gpu_memory": "8GB",
        "recommended_gpu": "RTX 2080 Ti or better",
        "cpu_cores": 8,
        "ram": "32GB"
    },
    "software": {
        "python_version": "3.6+",
        "pytorch_version": "1.8.1+",
        "cuda_version": "10.1+",
        "additional_packages": [
            "nnformer",
            "batchgenerators", 
            "SimpleITK",
            "scikit-image",
            "matplotlib",
            "tensorboard"
        ]
    },
    "paths": {
        "base_dir": "./",
        "data_dir": "../DATASET/nnFormer_raw/nnFormer_raw_data/Task01_BrainTumour",
        "preprocessed_dir": "../DATASET/nnFormer_preprocessed/Task001_ACDC",
        "results_dir": "./enhancements/experiments/results",
        "models_dir": "../DATASET/nnFormer_trained_models"
    }
}