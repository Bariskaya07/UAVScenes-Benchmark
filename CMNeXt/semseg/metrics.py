"""
Evaluation Metrics for UAVScenes Semantic Segmentation

Computes:
- Per-class IoU
- mIoU (mean Intersection over Union)
- Static mIoU (classes 0-16)
- Dynamic mIoU (classes 17-18: sedan, truck)
- Pixel Accuracy
"""

import numpy as np
import torch
from typing import Dict, List, Tuple


class UAVScenesMetrics:
    """Metrics calculator for UAVScenes segmentation.

    Uses confusion matrix for efficient IoU computation.

    Args:
        num_classes: Number of classes (19 for UAVScenes)
        ignore_label: Label to ignore in evaluation (255)
        class_names: Optional list of class names for reporting
    """

    # Default class names
    CLASS_NAMES = [
        'background',       # 0  - static
        'roof',             # 1  - static
        'dirt_road',        # 2  - static
        'paved_road',       # 3  - static
        'river',            # 4  - static
        'pool',             # 5  - static
        'bridge',           # 6  - static
        'container',        # 7  - static
        'airstrip',         # 8  - static
        'traffic_barrier',  # 9  - static
        'green_field',      # 10 - static
        'wild_field',       # 11 - static
        'solar_panel',      # 12 - static
        'umbrella',         # 13 - static
        'transparent_roof', # 14 - static
        'car_park',         # 15 - static
        'paved_walk',       # 16 - static
        'sedan',            # 17 - DYNAMIC
        'truck'             # 18 - DYNAMIC
    ]

    # Static vs Dynamic class indices
    STATIC_CLASSES = list(range(17))   # 0-16
    DYNAMIC_CLASSES = [17, 18]         # sedan, truck

    def __init__(self, num_classes=19, ignore_label=255, class_names=None):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.class_names = class_names or self.CLASS_NAMES

        # Confusion matrix: rows=gt, cols=pred
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        """Reset confusion matrix for new evaluation."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray):
        """Update confusion matrix with batch of predictions.

        Args:
            pred: Predicted labels [B, H, W] or [H, W]
            target: Ground truth labels [B, H, W] or [H, W]
        """
        # Flatten arrays
        pred = pred.flatten()
        target = target.flatten()

        # Filter out ignore label
        mask = target != self.ignore_label
        pred = pred[mask]
        target = target[mask]

        # Filter out invalid predictions (should be 0 to num_classes-1)
        valid_mask = (pred >= 0) & (pred < self.num_classes) & \
                     (target >= 0) & (target < self.num_classes)
        pred = pred[valid_mask]
        target = target[valid_mask]

        # Update confusion matrix using bincount (fast!)
        indices = target * self.num_classes + pred
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def update_batch(self, pred: torch.Tensor, target: torch.Tensor):
        """Update with PyTorch tensors.

        Args:
            pred: Predicted logits [B, C, H, W] or labels [B, H, W]
            target: Ground truth labels [B, H, W]
        """
        # Handle logits
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)

        # Convert to numpy
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()

        self.update(pred_np, target_np)

    def compute_iou(self) -> Tuple[np.ndarray, float]:
        """Compute per-class IoU and mIoU.

        Returns:
            iou: Per-class IoU array [num_classes]
            miou: Mean IoU across all classes
        """
        # IoU = TP / (TP + FP + FN)
        # TP = diagonal
        # FP = column sum - diagonal
        # FN = row sum - diagonal
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) +  # row sum (GT)
                 self.confusion_matrix.sum(axis=0) -  # col sum (Pred)
                 intersection)

        # Avoid division by zero
        iou = intersection / (union + 1e-10)

        # Only consider classes that appear in ground truth
        valid_classes = self.confusion_matrix.sum(axis=1) > 0
        miou = iou[valid_classes].mean() if valid_classes.any() else 0.0

        return iou, miou

    def compute_static_dynamic_iou(self) -> Tuple[float, float]:
        """Compute separate mIoU for static and dynamic classes.

        Returns:
            static_miou: mIoU for static classes (0-16)
            dynamic_miou: mIoU for dynamic classes (17-18)
        """
        iou, _ = self.compute_iou()

        # Static classes mIoU
        static_iou = iou[self.STATIC_CLASSES]
        static_valid = self.confusion_matrix.sum(axis=1)[self.STATIC_CLASSES] > 0
        static_miou = static_iou[static_valid].mean() if static_valid.any() else 0.0

        # Dynamic classes mIoU
        dynamic_iou = iou[self.DYNAMIC_CLASSES]
        dynamic_valid = self.confusion_matrix.sum(axis=1)[self.DYNAMIC_CLASSES] > 0
        dynamic_miou = dynamic_iou[dynamic_valid].mean() if dynamic_valid.any() else 0.0

        return static_miou, dynamic_miou

    def compute_pixel_accuracy(self) -> float:
        """Compute overall pixel accuracy.

        Returns:
            accuracy: Pixel accuracy (0-1)
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / (total + 1e-10)

    def compute_class_accuracy(self) -> np.ndarray:
        """Compute per-class accuracy.

        Returns:
            accuracy: Per-class accuracy array [num_classes]
        """
        correct = np.diag(self.confusion_matrix)
        total = self.confusion_matrix.sum(axis=1)
        return correct / (total + 1e-10)

    def compute_precision_recall_f1(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-class Precision, Recall, and F1-Score.
        
        Precision = TP / (TP + FP) = "Tahmin ettiğim X'lerin kaçı gerçekten X?"
        Recall    = TP / (TP + FN) = "Gerçek X'lerin kaçını buldum?"
        F1        = 2 * (Precision * Recall) / (Precision + Recall)
        
        Returns:
            precision: [num_classes] array
            recall: [num_classes] array  
            f1: [num_classes] array
        """
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp  # Column sum - TP
        fn = self.confusion_matrix.sum(axis=1) - tp  # Row sum - TP
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        return precision, recall, f1

    def get_confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix.
        
        Returns:
            confusion_matrix: [num_classes, num_classes] array
        """
        return self.confusion_matrix.copy()

    def get_results(self) -> Dict:
        """Get all metrics as a dictionary.

        Returns:
            Dictionary with all metrics
        """
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
            'confusion_matrix': self.confusion_matrix,
        }

    def print_results(self, logger=None):
        """Print formatted results table.

        Args:
            logger: Optional logger to use instead of print
        """
        results = self.get_results()
        support = self.confusion_matrix.sum(axis=1)  # Per-class pixel count

        def log(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)

        # =====================================================================
        # ÖZET SATIRI (Her evaluation'da görünecek)
        # =====================================================================
        log(f"\nmIoU: {results['mIoU']*100:.2f}% | "
            f"Static: {results['static_mIoU']*100:.2f}% | "
            f"Dynamic: {results['dynamic_mIoU']*100:.2f}% | "
            f"Acc: {results['pixel_accuracy']*100:.2f}%")
        
        # =====================================================================
        # DETAYLI TABLO
        # =====================================================================
        log("\n" + "=" * 100)
        log(f"{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Support':>12}")
        log("-" * 100)
        
        # Per-class results
        for i, cls_name in enumerate(self.class_names):
            if support[i] > 0:  # Only show classes with samples
                cls_type = "[D]" if i in self.DYNAMIC_CLASSES else "[S]"
                log(f"  {cls_type} {cls_name:<15} "
                    f"{results['per_class_iou'][i]*100:>7.2f}% "
                    f"{results['per_class_precision'][i]*100:>7.2f}% "
                    f"{results['per_class_recall'][i]*100:>7.2f}% "
                    f"{results['per_class_f1'][i]*100:>7.2f}% "
                    f"{results['per_class_accuracy'][i]*100:>7.2f}% "
                    f"{support[i]:>11,}")
        
        # =====================================================================
        # ÖZET İSTATİSTİKLER
        # =====================================================================
        log("-" * 100)
        log(f"  {'mIoU':<18} {results['mIoU']*100:>7.2f}%")
        log(f"  {'Static mIoU':<18} {results['static_mIoU']*100:>7.2f}%  (17 classes: background-paved_walk)")
        log(f"  {'Dynamic mIoU':<18} {results['dynamic_mIoU']*100:>7.2f}%  (2 classes: sedan, truck)")
        log(f"  {'Pixel Accuracy':<18} {results['pixel_accuracy']*100:>7.2f}%")
        log(f"  {'Mean Precision':<18} {results['mean_precision']*100:>7.2f}%")
        log(f"  {'Mean Recall':<18} {results['mean_recall']*100:>7.2f}%")
        log(f"  {'Mean F1':<18} {results['mean_f1']*100:>7.2f}%")
        log("=" * 100)
        
        # =====================================================================
        # EN ÇOK KARIŞTIRILAN SINIFLAR (Top 5)
        # =====================================================================
        log("\n📊 En Çok Karıştırılan Sınıf Çiftleri (Top 5):")
        log("-" * 60)
        
        # Diagonal'i sıfırla (doğru tahminleri hariç tut)
        cm_copy = self.confusion_matrix.copy()
        np.fill_diagonal(cm_copy, 0)
        
        # En yüksek 5 karışıklığı bul
        flat_indices = np.argsort(cm_copy.ravel())[::-1][:5]
        for idx in flat_indices:
            true_cls = idx // self.num_classes
            pred_cls = idx % self.num_classes
            count = cm_copy[true_cls, pred_cls]
            if count > 0:
                log(f"  {self.class_names[true_cls]:<18} → {self.class_names[pred_cls]:<18}: {count:>12,} pixels")
        
        log("=" * 100 + "\n")

    def save_confusion_matrix(self, save_path: str):
        """Save confusion matrix as image.
        
        Args:
            save_path: Path to save the confusion matrix image
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Normalize confusion matrix (row-wise)
            cm_normalized = self.confusion_matrix.astype('float') / (
                self.confusion_matrix.sum(axis=1, keepdims=True) + 1e-10
            )
            
            plt.figure(figsize=(16, 14))
            sns.heatmap(
                cm_normalized,
                annot=False,
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                square=True,
                cbar_kws={'label': 'Ratio'}
            )
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('True', fontsize=12)
            plt.title('Confusion Matrix (Normalized)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to: {save_path}")
        except ImportError:
            print("Warning: matplotlib/seaborn not installed, skipping confusion matrix plot")

    def __str__(self) -> str:
        """String representation with key metrics."""
        results = self.get_results()
        return (f"mIoU: {results['mIoU']*100:.2f}% | "
                f"Static: {results['static_mIoU']*100:.2f}% | "
                f"Dynamic: {results['dynamic_mIoU']*100:.2f}% | "
                f"Acc: {results['pixel_accuracy']*100:.2f}%")


class EarlyStopping:
    """Early stopping to stop training when metric stops improving.

    Args:
        patience: Number of evaluations to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'max' for metrics like mIoU, 'min' for loss
    """
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score) -> bool:
        """Check if training should stop.

        Args:
            current_score: Current metric value (e.g., mIoU)

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        # Check improvement
        if self.mode == 'max':
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping: {self.counter}/{self.patience} '
                  f'(best: {self.best_score:.4f}, current: {current_score:.4f})')

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset the early stopping counter."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


# Test code
if __name__ == '__main__':
    # Test metrics
    metrics = UAVScenesMetrics(num_classes=19, ignore_label=255)

    # Create random predictions and targets
    np.random.seed(42)
    for _ in range(10):
        pred = np.random.randint(0, 19, (4, 256, 256))
        target = np.random.randint(0, 19, (4, 256, 256))
        # Add some ignore labels
        target[target == 5] = 255

        metrics.update(pred, target)

    # Print results
    print("\nTest Results:")
    metrics.print_results()

    # Test early stopping
    print("\n\nTesting Early Stopping:")
    early_stop = EarlyStopping(patience=3, min_delta=0.01)

    mious = [0.50, 0.55, 0.58, 0.58, 0.585, 0.58, 0.57]  # Stops improving after 0.58
    for epoch, miou in enumerate(mious):
        stop = early_stop(miou)
        print(f"Epoch {epoch}: mIoU={miou:.3f}, Stop={stop}")
        if stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
