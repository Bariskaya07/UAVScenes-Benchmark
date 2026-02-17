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


class UAVScenesMetrics:
    """Metrics calculator for UAVScenes segmentation with detailed output."""

    CLASS_NAMES = [
        'background', 'roof', 'dirt_road', 'paved_road', 'river', 'pool',
        'bridge', 'container', 'airstrip', 'traffic_barrier', 'green_field',
        'wild_field', 'solar_panel', 'umbrella', 'transparent_roof', 'car_park',
        'paved_walk', 'sedan', 'truck'
    ]

    STATIC_CLASSES = list(range(17))
    DYNAMIC_CLASSES = [17, 18]

    def __init__(self, num_classes=19, ignore_label=255, class_names=None):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.class_names = class_names or self.CLASS_NAMES
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        pred = pred.flatten()
        target = target.flatten()

        mask = target != self.ignore_label
        pred = pred[mask]
        target = target[mask]

        valid_mask = (pred >= 0) & (pred < self.num_classes) & \
                     (target >= 0) & (target < self.num_classes)
        pred = pred[valid_mask]
        target = target[valid_mask]

        indices = target * self.num_classes + pred
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute_iou(self):
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) +
                 self.confusion_matrix.sum(axis=0) - intersection)
        iou = intersection / (union + 1e-10)
        valid_classes = self.confusion_matrix.sum(axis=1) > 0
        miou = iou[valid_classes].mean() if valid_classes.any() else 0.0
        return iou, miou

    def compute_static_dynamic_iou(self):
        iou, _ = self.compute_iou()

        static_iou = iou[self.STATIC_CLASSES]
        static_valid = self.confusion_matrix.sum(axis=1)[self.STATIC_CLASSES] > 0
        static_miou = static_iou[static_valid].mean() if static_valid.any() else 0.0

        dynamic_iou = iou[self.DYNAMIC_CLASSES]
        dynamic_valid = self.confusion_matrix.sum(axis=1)[self.DYNAMIC_CLASSES] > 0
        dynamic_miou = dynamic_iou[dynamic_valid].mean() if dynamic_valid.any() else 0.0

        return static_miou, dynamic_miou

    def compute_pixel_accuracy(self):
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / (total + 1e-10)

    def compute_class_accuracy(self):
        correct = np.diag(self.confusion_matrix)
        total = self.confusion_matrix.sum(axis=1)
        return correct / (total + 1e-10)

    def compute_precision_recall_f1(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        return precision, recall, f1

    def get_results(self):
        iou, miou = self.compute_iou()
        static_miou, dynamic_miou = self.compute_static_dynamic_iou()
        pixel_acc = self.compute_pixel_accuracy()
        class_acc = self.compute_class_accuracy()
        precision, recall, f1 = self.compute_precision_recall_f1()

        return {
            'mIoU': miou,
            'static_mIoU': static_miou,
            'dynamic_mIoU': dynamic_miou,
            'pixel_accuracy': pixel_acc,
            'per_class_iou': iou,
            'per_class_accuracy': class_acc,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'mean_precision': precision.mean(),
            'mean_recall': recall.mean(),
            'mean_f1': f1.mean(),
        }

    def print_results(self, logger=None):
        results = self.get_results()
        support = self.confusion_matrix.sum(axis=1)

        def log(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)

        log(f"\nmIoU: {results['mIoU']*100:.2f}% | "
            f"Static: {results['static_mIoU']*100:.2f}% | "
            f"Dynamic: {results['dynamic_mIoU']*100:.2f}% | "
            f"Acc: {results['pixel_accuracy']*100:.2f}%")

        log("\n" + "=" * 100)
        log(f"{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Support':>12}")
        log("-" * 100)

        for i, cls_name in enumerate(self.class_names):
            if support[i] > 0:
                cls_type = "[D]" if i in self.DYNAMIC_CLASSES else "[S]"
                log(f"  {cls_type} {cls_name:<15} "
                    f"{results['per_class_iou'][i]*100:>7.2f}% "
                    f"{results['per_class_precision'][i]*100:>7.2f}% "
                    f"{results['per_class_recall'][i]*100:>7.2f}% "
                    f"{results['per_class_f1'][i]*100:>7.2f}% "
                    f"{results['per_class_accuracy'][i]*100:>7.2f}% "
                    f"{support[i]:>11,}")

        log("-" * 100)
        log(f"  {'mIoU':<18} {results['mIoU']*100:>7.2f}%")
        log(f"  {'Static mIoU':<18} {results['static_mIoU']*100:>7.2f}%  (17 classes: background-paved_walk)")
        log(f"  {'Dynamic mIoU':<18} {results['dynamic_mIoU']*100:>7.2f}%  (2 classes: sedan, truck)")
        log(f"  {'Pixel Accuracy':<18} {results['pixel_accuracy']*100:>7.2f}%")
        log(f"  {'Mean Precision':<18} {results['mean_precision']*100:>7.2f}%")
        log(f"  {'Mean Recall':<18} {results['mean_recall']*100:>7.2f}%")
        log(f"  {'Mean F1':<18} {results['mean_f1']*100:>7.2f}%")
        log("=" * 100)

        log("\nMost Confused Class Pairs (Top 5):")
        log("-" * 60)

        cm_copy = self.confusion_matrix.copy()
        np.fill_diagonal(cm_copy, 0)

        flat_indices = np.argsort(cm_copy.ravel())[::-1][:5]
        for idx in flat_indices:
            true_cls = idx // self.num_classes
            pred_cls = idx % self.num_classes
            count = cm_copy[true_cls, pred_cls]
            if count > 0:
                log(f"  {self.class_names[true_cls]:<18} -> {self.class_names[pred_cls]:<18}: {count:>12,} pixels")

        log("=" * 100 + "\n")


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
