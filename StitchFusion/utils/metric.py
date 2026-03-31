# encoding: utf-8

import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))
    confusionMatrix = np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                        minlength=n_cl ** 2).reshape(n_cl, n_cl)
    return confusionMatrix, labeled, correct

def compute_score(hist, correct, labeled):
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:]) # useless for NYUDv2

    freq = hist.sum(1) / hist.sum()
    freq_IoU = (iou[freq > 0] * freq[freq > 0]).sum()

    classAcc = np.diag(hist) / hist.sum(axis=1)
    mean_pixel_acc = np.nanmean(classAcc)

    pixel_acc = correct / labeled

    return iou, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc


STATIC_CLASS_INDICES = list(range(17))
DYNAMIC_CLASS_INDICES = [17, 18]


def compute_detailed_score(hist):
    support = hist.sum(axis=1)
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp

    iou = tp / (tp + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    class_acc = recall

    valid = support > 0
    miou = iou[valid].mean() if valid.any() else 0.0
    pixel_acc = tp.sum() / (hist.sum() + 1e-10)

    static_valid = support[STATIC_CLASS_INDICES] > 0
    dynamic_valid = support[DYNAMIC_CLASS_INDICES] > 0
    static_miou = (
        iou[STATIC_CLASS_INDICES][static_valid].mean() if static_valid.any() else 0.0
    )
    dynamic_miou = (
        iou[DYNAMIC_CLASS_INDICES][dynamic_valid].mean() if dynamic_valid.any() else 0.0
    )

    mean_precision = precision[valid].mean() if valid.any() else 0.0
    mean_recall = recall[valid].mean() if valid.any() else 0.0
    mean_f1 = f1[valid].mean() if valid.any() else 0.0

    return {
        'support': support,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_acc': class_acc,
        'mIoU': miou,
        'static_mIoU': static_miou,
        'dynamic_mIoU': dynamic_miou,
        'pixel_accuracy': pixel_acc,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'confusion_matrix': hist,
    }


def format_detailed_report(hist, class_names, total_time=None, num_images=None):
    results = compute_detailed_score(hist)
    support = results['support']
    lines = []

    lines.append(
        f"mIoU: {results['mIoU']*100:.2f}% | "
        f"Static: {results['static_mIoU']*100:.2f}% | "
        f"Dynamic: {results['dynamic_mIoU']*100:.2f}% | "
        f"Acc: {results['pixel_accuracy']*100:.2f}%"
    )
    lines.append("")
    lines.append("=" * 100)
    lines.append(f"{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Support':>12}")
    lines.append("-" * 100)

    for idx, cls_name in enumerate(class_names):
        if support[idx] <= 0:
            continue
        cls_type = "[D]" if idx in DYNAMIC_CLASS_INDICES else "[S]"
        lines.append(
            f"  {cls_type} {cls_name:<15} "
            f"{results['iou'][idx]*100:>7.2f}% "
            f"{results['precision'][idx]*100:>7.2f}% "
            f"{results['recall'][idx]*100:>7.2f}% "
            f"{results['f1'][idx]*100:>7.2f}% "
            f"{results['class_acc'][idx]*100:>7.2f}% "
            f"{int(support[idx]):>11,}"
        )

    lines.append("-" * 100)
    lines.append(f"  {'mIoU':<18} {results['mIoU']*100:>7.2f}%")
    lines.append(f"  {'Static mIoU':<18} {results['static_mIoU']*100:>7.2f}%  (17 classes: background-paved_walk)")
    lines.append(f"  {'Dynamic mIoU':<18} {results['dynamic_mIoU']*100:>7.2f}%  (2 classes: sedan, truck)")
    lines.append(f"  {'Pixel Accuracy':<18} {results['pixel_accuracy']*100:>7.2f}%")
    lines.append(f"  {'Mean Precision':<18} {results['mean_precision']*100:>7.2f}%")
    lines.append(f"  {'Mean Recall':<18} {results['mean_recall']*100:>7.2f}%")
    lines.append(f"  {'Mean F1':<18} {results['mean_f1']*100:>7.2f}%")
    lines.append("=" * 100)
    lines.append("")
    lines.append("En Cok Karistirilan Sinif Ciftleri (Top 5):")
    lines.append("-" * 60)

    cm_copy = hist.copy()
    np.fill_diagonal(cm_copy, 0)
    flat_indices = np.argsort(cm_copy.ravel())[::-1][:5]
    for flat_idx in flat_indices:
        true_cls = flat_idx // hist.shape[1]
        pred_cls = flat_idx % hist.shape[1]
        count = cm_copy[true_cls, pred_cls]
        if count > 0:
            lines.append(
                f"  {class_names[true_cls]:<18} -> {class_names[pred_cls]:<18}: {int(count):>12,} pixels"
            )
    lines.append("=" * 100)

    if total_time is not None and total_time > 0:
        lines.append("")
        lines.append("Inference speed:")
        if num_images is not None and num_images > 0:
            avg_ms = (total_time / num_images) * 1000.0
            fps = num_images / total_time
            lines.append(f"  Average time per image: {avg_ms:.1f}ms")
            lines.append(f"  FPS: {fps:.2f}")
        else:
            lines.append(f"  Total time: {total_time:.2f}s")

    return "\n".join(lines)
