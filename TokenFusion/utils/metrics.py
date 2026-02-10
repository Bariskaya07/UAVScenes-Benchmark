"""
Evaluation Metrics for Semantic Segmentation

Implements:
- Confusion Matrix
- mIoU (mean Intersection over Union)
- Static/Dynamic mIoU for UAVScenes
- Pixel Accuracy
- Class-wise IoU
"""

import numpy as np
import torch


class ConfusionMatrix:
    """
    Confusion Matrix for semantic segmentation evaluation.

    Accumulates predictions across batches and computes metrics.
    """

    def __init__(self, num_classes, ignore_label=255):
        """
        Initialize confusion matrix.

        Args:
            num_classes: Number of classes (19 for UAVScenes)
            ignore_label: Label to ignore in evaluation (255)
        """
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        """
        Update confusion matrix with batch predictions.

        Args:
            pred: Predicted labels [B, H, W] or [H, W]
            target: Ground truth labels [B, H, W] or [H, W]
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        pred = pred.flatten()
        target = target.flatten()

        # Mask out ignore labels
        mask = target != self.ignore_label
        pred = pred[mask]
        target = target[mask]

        # Clip predictions to valid range
        pred = np.clip(pred, 0, self.num_classes - 1)

        # Update confusion matrix
        for t, p in zip(target, pred):
            self.mat[t, p] += 1

    def get_metrics(self):
        """
        Compute all metrics from confusion matrix.

        Returns:
            Dictionary with metrics:
            - miou: Mean IoU
            - static_miou: mIoU for static classes (0-16)
            - dynamic_miou: mIoU for dynamic classes (17-18)
            - pixel_acc: Pixel accuracy
            - class_iou: Per-class IoU
            - class_acc: Per-class accuracy
        """
        # Per-class metrics
        tp = np.diag(self.mat)
        fp = self.mat.sum(axis=0) - tp
        fn = self.mat.sum(axis=1) - tp

        # IoU per class
        union = tp + fp + fn
        iou = np.where(union > 0, tp / union, 0)

        # Accuracy per class
        class_total = self.mat.sum(axis=1)
        class_acc = np.where(class_total > 0, tp / class_total, 0)

        # Mean IoU (only for classes with samples)
        valid_mask = self.mat.sum(axis=1) > 0
        miou = iou[valid_mask].mean() if valid_mask.any() else 0.0

        # Static mIoU (classes 0-16)
        static_mask = np.zeros(self.num_classes, dtype=bool)
        static_mask[:17] = True
        static_valid = valid_mask & static_mask
        static_miou = iou[static_valid].mean() if static_valid.any() else 0.0

        # Dynamic mIoU (classes 17-18: sedan, truck)
        dynamic_mask = np.zeros(self.num_classes, dtype=bool)
        if self.num_classes > 17:
            dynamic_mask[17:] = True
        dynamic_valid = valid_mask & dynamic_mask
        dynamic_miou = iou[dynamic_valid].mean() if dynamic_valid.any() else 0.0

        # Pixel accuracy
        total_correct = tp.sum()
        total_pixels = self.mat.sum()
        pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0.0

        # Mean accuracy
        mean_acc = class_acc[valid_mask].mean() if valid_mask.any() else 0.0

        return {
            'miou': miou,
            'static_miou': static_miou,
            'dynamic_miou': dynamic_miou,
            'pixel_acc': pixel_acc,
            'mean_acc': mean_acc,
            'class_iou': iou,
            'class_acc': class_acc
        }

    def reset(self):
        """Reset confusion matrix."""
        self.mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


def compute_metrics(pred, target, num_classes=19, ignore_label=255):
    """
    Compute segmentation metrics for a batch.

    Args:
        pred: Predictions [B, H, W]
        target: Ground truth [B, H, W]
        num_classes: Number of classes
        ignore_label: Label to ignore

    Returns:
        Dictionary with metrics
    """
    cm = ConfusionMatrix(num_classes, ignore_label)
    cm.update(pred, target)
    return cm.get_metrics()


def print_metrics(metrics, class_names=None):
    """
    Print metrics in a formatted table.

    Args:
        metrics: Dictionary from ConfusionMatrix.get_metrics()
        class_names: List of class names
    """
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  mIoU:         {metrics['miou'] * 100:.2f}%")
    print(f"  Static mIoU:  {metrics['static_miou'] * 100:.2f}%")
    print(f"  Dynamic mIoU: {metrics['dynamic_miou'] * 100:.2f}%")
    print(f"  Pixel Acc:    {metrics['pixel_acc'] * 100:.2f}%")
    print(f"  Mean Acc:     {metrics['mean_acc'] * 100:.2f}%")

    if class_names is not None and 'class_iou' in metrics:
        print(f"\nPer-class IoU:")
        for i, (name, iou) in enumerate(zip(class_names, metrics['class_iou'])):
            marker = " (dynamic)" if i >= 17 else ""
            print(f"  {i:2d}. {name:20s}: {iou * 100:5.2f}%{marker}")

    print("=" * 60 + "\n")


# UAVScenes class names
UAVSCENES_CLASSES = [
    "background",       # 0
    "roof",             # 1
    "dirt_road",        # 2
    "paved_road",       # 3
    "river",            # 4
    "pool",             # 5
    "bridge",           # 6
    "container",        # 7
    "airstrip",         # 8
    "traffic_barrier",  # 9
    "green_field",      # 10
    "wild_field",       # 11
    "solar_panel",      # 12
    "umbrella",         # 13
    "transparent_roof", # 14
    "car_park",         # 15
    "paved_walk",       # 16
    "sedan",            # 17 (dynamic)
    "truck",            # 18 (dynamic)
]


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersect_and_union(pred, target, num_classes, ignore_label):
    """
    Calculate intersection and union for IoU computation.

    Args:
        pred: Predictions [H, W]
        target: Ground truth [H, W]
        num_classes: Number of classes
        ignore_label: Label to ignore

    Returns:
        area_intersect: Intersection per class
        area_union: Union per class
        area_pred: Prediction area per class
        area_target: Target area per class
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    # Mask out ignore labels
    mask = target != ignore_label
    pred = pred[mask]
    target = target[mask]

    # Clip predictions
    pred = np.clip(pred, 0, num_classes - 1)

    # Compute areas
    area_intersect = np.zeros(num_classes)
    area_union = np.zeros(num_classes)
    area_pred = np.zeros(num_classes)
    area_target = np.zeros(num_classes)

    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c

        area_intersect[c] = np.logical_and(pred_c, target_c).sum()
        area_union[c] = np.logical_or(pred_c, target_c).sum()
        area_pred[c] = pred_c.sum()
        area_target[c] = target_c.sum()

    return area_intersect, area_union, area_pred, area_target
