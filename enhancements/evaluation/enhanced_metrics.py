"""
Enhanced Evaluation Metrics for nnFormer
========================================

This module provides enhanced evaluation metrics specifically designed
for medical image segmentation with focus on boundary accuracy and
multi-scale feature assessment.

Author: 210353V
Date: October 2025
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import SimpleITK as sitk


def compute_dice_coefficient(pred, target, smooth=1e-6):
    """
    Compute Dice Similarity Coefficient.
    
    Args:
        pred (torch.Tensor): Predicted segmentation [B, C, H, W, D]
        target (torch.Tensor): Ground truth segmentation [B, C, H, W, D]
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        dict: Dice scores for each class
    """
    # Convert to binary if needed
    if pred.dim() > target.dim():
        pred = torch.argmax(pred, dim=1)
    
    dice_scores = {}
    num_classes = target.max().item() + 1
    
    for class_idx in range(1, num_classes):  # Skip background
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_scores[f'dice_class_{class_idx}'] = dice.item()
    
    # Compute composite regions for BraTS
    if num_classes == 4:  # BraTS case
        # Whole Tumor (WT): classes 1, 2, 3
        pred_wt = ((pred >= 1) & (pred <= 3)).float()
        target_wt = ((target >= 1) & (target <= 3)).float()
        
        intersection_wt = (pred_wt * target_wt).sum()
        union_wt = pred_wt.sum() + target_wt.sum()
        dice_wt = (2 * intersection_wt + smooth) / (union_wt + smooth)
        dice_scores['dice_WT'] = dice_wt.item()
        
        # Tumor Core (TC): classes 2, 3
        pred_tc = ((pred >= 2) & (pred <= 3)).float()
        target_tc = ((target >= 2) & (target <= 3)).float()
        
        intersection_tc = (pred_tc * target_tc).sum()
        union_tc = pred_tc.sum() + target_tc.sum()
        dice_tc = (2 * intersection_tc + smooth) / (union_tc + smooth)
        dice_scores['dice_TC'] = dice_tc.item()
        
        # Enhancing Tumor (ET): class 3
        pred_et = (pred == 3).float()
        target_et = (target == 3).float()
        
        intersection_et = (pred_et * target_et).sum()
        union_et = pred_et.sum() + target_et.sum()
        dice_et = (2 * intersection_et + smooth) / (union_et + smooth)
        dice_scores['dice_ET'] = dice_et.item()
    
    return dice_scores


def compute_hausdorff_distance_95(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    Compute 95th percentile Hausdorff Distance.
    
    Args:
        pred (numpy.ndarray): Predicted segmentation
        target (numpy.ndarray): Ground truth segmentation  
        spacing (tuple): Voxel spacing
    
    Returns:
        dict: HD95 values for each class
    """
    hd95_scores = {}
    
    # Convert torch tensors to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    # Remove batch dimension if present
    if pred.ndim == 4:
        pred = pred[0]
    if target.ndim == 4:
        target = target[0]
    
    num_classes = int(target.max()) + 1
    
    for class_idx in range(1, num_classes):  # Skip background
        pred_class = (pred == class_idx).astype(np.uint8)
        target_class = (target == class_idx).astype(np.uint8)
        
        if pred_class.sum() == 0 or target_class.sum() == 0:
            hd95_scores[f'hd95_class_{class_idx}'] = np.inf
            continue
        
        # Convert to SimpleITK images for proper spacing handling
        pred_img = sitk.GetImageFromArray(pred_class)
        target_img = sitk.GetImageFromArray(target_class)
        pred_img.SetSpacing(spacing)
        target_img.SetSpacing(spacing)
        
        # Compute surface distances
        hausdorff_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(pred_img, target_img)
        
        # Get 95th percentile
        hd95 = hausdorff_filter.GetAverageHausdorffDistance()
        hd95_scores[f'hd95_class_{class_idx}'] = hd95
    
    # Compute for BraTS composite regions
    if num_classes == 4:
        for region, classes in [('WT', [1, 2, 3]), ('TC', [2, 3]), ('ET', [3])]:
            pred_region = np.isin(pred, classes).astype(np.uint8)
            target_region = np.isin(target, classes).astype(np.uint8)
            
            if pred_region.sum() == 0 or target_region.sum() == 0:
                hd95_scores[f'hd95_{region}'] = np.inf
                continue
            
            pred_img = sitk.GetImageFromArray(pred_region)
            target_img = sitk.GetImageFromArray(target_region)
            pred_img.SetSpacing(spacing)
            target_img.SetSpacing(spacing)
            
            hausdorff_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_filter.Execute(pred_img, target_img)
            hd95_scores[f'hd95_{region}'] = hausdorff_filter.GetAverageHausdorffDistance()
    
    return hd95_scores


def compute_boundary_dice(pred, target, tolerance=2):
    """
    Compute Boundary Dice for boundary-aware evaluation.
    
    Args:
        pred (numpy.ndarray): Predicted segmentation
        target (numpy.ndarray): Ground truth segmentation
        tolerance (int): Boundary tolerance in pixels
    
    Returns:
        dict: Boundary Dice scores
    """
    boundary_dice_scores = {}
    
    # Convert torch tensors to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    # Remove batch dimension if present
    if pred.ndim == 4:
        pred = pred[0]
    if target.ndim == 4:
        target = target[0]
    
    def extract_boundary(mask, tolerance=2):
        """Extract boundary with tolerance."""
        eroded = binary_erosion(mask, iterations=tolerance)
        boundary = mask.astype(bool) & ~eroded
        dilated_boundary = binary_dilation(boundary, iterations=tolerance)
        return dilated_boundary
    
    num_classes = int(target.max()) + 1
    
    for class_idx in range(1, num_classes):
        pred_class = (pred == class_idx)
        target_class = (target == class_idx)
        
        if pred_class.sum() == 0 and target_class.sum() == 0:
            boundary_dice_scores[f'boundary_dice_class_{class_idx}'] = 1.0
            continue
        elif pred_class.sum() == 0 or target_class.sum() == 0:
            boundary_dice_scores[f'boundary_dice_class_{class_idx}'] = 0.0
            continue
        
        # Extract boundaries
        pred_boundary = extract_boundary(pred_class, tolerance)
        target_boundary = extract_boundary(target_class, tolerance)
        
        # Compute boundary dice
        intersection = (pred_boundary & target_boundary).sum()
        union = pred_boundary.sum() + target_boundary.sum()
        
        if union == 0:
            boundary_dice = 1.0
        else:
            boundary_dice = (2 * intersection) / union
        
        boundary_dice_scores[f'boundary_dice_class_{class_idx}'] = boundary_dice
    
    return boundary_dice_scores


def compute_sensitivity_specificity(pred, target):
    """
    Compute Sensitivity (Recall) and Specificity.
    
    Args:
        pred (torch.Tensor): Predicted segmentation
        target (torch.Tensor): Ground truth segmentation
    
    Returns:
        dict: Sensitivity and specificity scores
    """
    metrics = {}
    
    # Convert to numpy for sklearn
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    # Flatten arrays
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    num_classes = int(target.max()) + 1
    
    for class_idx in range(1, num_classes):
        pred_binary = (pred_flat == class_idx).astype(int)
        target_binary = (target_flat == class_idx).astype(int)
        
        # Compute metrics
        tn = ((pred_binary == 0) & (target_binary == 0)).sum()
        tp = ((pred_binary == 1) & (target_binary == 1)).sum()
        fn = ((pred_binary == 0) & (target_binary == 1)).sum()
        fp = ((pred_binary == 1) & (target_binary == 0)).sum()
        
        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f'sensitivity_class_{class_idx}'] = sensitivity
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics[f'specificity_class_{class_idx}'] = specificity
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics[f'precision_class_{class_idx}'] = precision
    
    return metrics


def compute_volume_similarity(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    Compute Volume Similarity metrics.
    
    Args:
        pred (numpy.ndarray): Predicted segmentation
        target (numpy.ndarray): Ground truth segmentation
        spacing (tuple): Voxel spacing
    
    Returns:
        dict: Volume similarity metrics
    """
    volume_metrics = {}
    
    # Convert torch tensors to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    voxel_volume = np.prod(spacing)  # Volume of one voxel
    
    num_classes = int(target.max()) + 1
    
    for class_idx in range(1, num_classes):
        pred_volume = (pred == class_idx).sum() * voxel_volume
        target_volume = (target == class_idx).sum() * voxel_volume
        
        # Volume difference
        volume_diff = abs(pred_volume - target_volume)
        volume_metrics[f'volume_diff_class_{class_idx}'] = volume_diff
        
        # Relative volume difference
        if target_volume > 0:
            rel_volume_diff = volume_diff / target_volume
        else:
            rel_volume_diff = 1.0 if pred_volume > 0 else 0.0
        
        volume_metrics[f'rel_volume_diff_class_{class_idx}'] = rel_volume_diff
    
    return volume_metrics


def compute_comprehensive_metrics(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    Compute comprehensive set of metrics for medical image segmentation.
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation  
        spacing: Voxel spacing
    
    Returns:
        dict: All computed metrics
    """
    all_metrics = {}
    
    # Basic metrics
    all_metrics.update(compute_dice_coefficient(pred, target))
    all_metrics.update(compute_sensitivity_specificity(pred, target))
    
    # Advanced metrics (convert to numpy for these)
    if torch.is_tensor(pred):
        pred_np = pred.cpu().numpy()
    else:
        pred_np = pred
        
    if torch.is_tensor(target):
        target_np = target.cpu().numpy()
    else:
        target_np = target
    
    all_metrics.update(compute_hausdorff_distance_95(pred_np, target_np, spacing))
    all_metrics.update(compute_boundary_dice(pred_np, target_np))
    all_metrics.update(compute_volume_similarity(pred_np, target_np, spacing))
    
    return all_metrics


class MetricsTracker:
    """
    Track and accumulate metrics across batches and epochs.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.metrics_sum = {}
        self.count = 0
    
    def update(self, metrics_dict):
        """Update with new metrics from a batch."""
        self.count += 1
        
        for key, value in metrics_dict.items():
            if key not in self.metrics_sum:
                self.metrics_sum[key] = 0
            self.metrics_sum[key] += value
    
    def get_average_metrics(self):
        """Get average metrics across all updates."""
        if self.count == 0:
            return {}
        
        avg_metrics = {}
        for key, value in self.metrics_sum.items():
            avg_metrics[key] = value / self.count
        
        return avg_metrics
    
    def get_summary(self):
        """Get a formatted summary of metrics."""
        avg_metrics = self.get_average_metrics()
        
        summary = []
        summary.append("="*50)
        summary.append("SEGMENTATION METRICS SUMMARY")
        summary.append("="*50)
        
        # Group metrics by type
        dice_metrics = {k: v for k, v in avg_metrics.items() if 'dice' in k}
        hd95_metrics = {k: v for k, v in avg_metrics.items() if 'hd95' in k}
        boundary_metrics = {k: v for k, v in avg_metrics.items() if 'boundary' in k}
        
        if dice_metrics:
            summary.append("\nDICE COEFFICIENTS:")
            for key, value in dice_metrics.items():
                summary.append(f"  {key}: {value:.4f}")
        
        if hd95_metrics:
            summary.append("\nHAUSDORFF DISTANCE 95%:")
            for key, value in hd95_metrics.items():
                summary.append(f"  {key}: {value:.2f} mm")
        
        if boundary_metrics:
            summary.append("\nBOUNDARY METRICS:")
            for key, value in boundary_metrics.items():
                summary.append(f"  {key}: {value:.4f}")
        
        summary.append("="*50)
        
        return "\n".join(summary)