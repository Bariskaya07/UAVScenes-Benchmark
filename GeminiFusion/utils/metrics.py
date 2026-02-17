"""
Evaluation Metrics for UAVScenes Semantic Segmentation
"""

import numpy as np
import torch
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional


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

    def compute_iou(self) -> Tuple[np.ndarray, float]:
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) +
                 self.confusion_matrix.sum(axis=0) - intersection)
        iou = intersection / (union + 1e-10)
        valid_classes = self.confusion_matrix.sum(axis=1) > 0
        miou = iou[valid_classes].mean() if valid_classes.any() else 0.0
        return iou, miou

    def compute_static_dynamic_iou(self) -> Tuple[float, float]:
        iou, _ = self.compute_iou()

        static_iou = iou[self.STATIC_CLASSES]
        static_valid = self.confusion_matrix.sum(axis=1)[self.STATIC_CLASSES] > 0
        static_miou = static_iou[static_valid].mean() if static_valid.any() else 0.0

        dynamic_iou = iou[self.DYNAMIC_CLASSES]
        dynamic_valid = self.confusion_matrix.sum(axis=1)[self.DYNAMIC_CLASSES] > 0
        dynamic_miou = dynamic_iou[dynamic_valid].mean() if dynamic_valid.any() else 0.0

        return static_miou, dynamic_miou

    def compute_pixel_accuracy(self) -> float:
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / (total + 1e-10)

    def compute_class_accuracy(self) -> np.ndarray:
        correct = np.diag(self.confusion_matrix)
        total = self.confusion_matrix.sum(axis=1)
        return correct / (total + 1e-10)

    def compute_precision_recall_f1(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        return precision, recall, f1

    def get_results(self) -> Dict:
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

    def save_results(self, save_dir: str, model_name: str,
                     inference_time_ms: Optional[float] = None,
                     fps: Optional[float] = None,
                     num_images: Optional[int] = None,
                     logger=None):
        """Save results to JSON and text files."""
        os.makedirs(save_dir, exist_ok=True)
        results = self.get_results()
        support = self.confusion_matrix.sum(axis=1)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_results = {
            'model': model_name, 'timestamp': timestamp,
            'metrics': {
                'mIoU': float(results['mIoU'] * 100),
                'static_mIoU': float(results['static_mIoU'] * 100),
                'dynamic_mIoU': float(results['dynamic_mIoU'] * 100),
                'pixel_accuracy': float(results['pixel_accuracy'] * 100),
                'mean_precision': float(results['mean_precision'] * 100),
                'mean_recall': float(results['mean_recall'] * 100),
                'mean_f1': float(results['mean_f1'] * 100),
            },
            'per_class': {}, 'inference': {}
        }

        if inference_time_ms is not None:
            json_results['inference']['time_per_image_ms'] = round(inference_time_ms, 2)
        if fps is not None:
            json_results['inference']['fps'] = round(fps, 2)
        if num_images is not None:
            json_results['inference']['num_images'] = num_images

        for i, cls_name in enumerate(self.class_names):
            json_results['per_class'][cls_name] = {
                'iou': float(results['per_class_iou'][i] * 100),
                'precision': float(results['per_class_precision'][i] * 100),
                'recall': float(results['per_class_recall'][i] * 100),
                'f1': float(results['per_class_f1'][i] * 100),
                'accuracy': float(results['per_class_accuracy'][i] * 100),
                'support': int(support[i]),
                'type': 'dynamic' if i in self.DYNAMIC_CLASSES else 'static'
            }

        json_path = os.path.join(save_dir, f'{model_name}_results.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        txt_path = os.path.join(save_dir, f'{model_name}_results.txt')
        with open(txt_path, 'w') as f:
            f.write(f"{'='*100}\nUAVScenes Benchmark Results - {model_name}\nTimestamp: {timestamp}\n{'='*100}\n\n")
            f.write(f"SUMMARY\n{'-'*50}\n")
            f.write(f"mIoU:           {results['mIoU']*100:>7.2f}%\n")
            f.write(f"Static mIoU:    {results['static_mIoU']*100:>7.2f}%  (17 classes)\n")
            f.write(f"Dynamic mIoU:   {results['dynamic_mIoU']*100:>7.2f}%  (2 classes)\n")
            f.write(f"Pixel Accuracy: {results['pixel_accuracy']*100:>7.2f}%\n")
            f.write(f"Mean F1:        {results['mean_f1']*100:>7.2f}%\n")
            if inference_time_ms is not None:
                f.write(f"\nINFERENCE SPEED\n{'-'*50}\n")
                if num_images: f.write(f"Test Images:    {num_images}\n")
                f.write(f"Time/Image:     {inference_time_ms:.2f} ms\n")
                if fps: f.write(f"FPS:            {fps:.2f}\n")
            f.write(f"\nPER-CLASS RESULTS\n{'='*100}\n")
            f.write(f"{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>12}\n{'-'*100}\n")
            for i, cls_name in enumerate(self.class_names):
                if support[i] > 0:
                    t = "[D]" if i in self.DYNAMIC_CLASSES else "[S]"
                    f.write(f"  {t} {cls_name:<15} {results['per_class_iou'][i]*100:>7.2f}% "
                           f"{results['per_class_precision'][i]*100:>7.2f}% {results['per_class_recall'][i]*100:>7.2f}% "
                           f"{results['per_class_f1'][i]*100:>7.2f}% {support[i]:>11,}\n")
            f.write(f"{'='*100}\n")

        if logger:
            logger.info(f"Results saved to: {save_dir}/{model_name}_results.[json|txt]")
        return json_path, txt_path
