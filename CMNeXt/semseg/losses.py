"""
Loss Functions for Semantic Segmentation

Includes:
- CrossEntropyLoss
- OhemCrossEntropyLoss (Online Hard Example Mining)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard Cross Entropy Loss with ignore index support."""

    def __init__(self, ignore_index=255, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction='mean'
        )

    def forward(self, logits, target):
        """
        Args:
            logits: [B, C, H, W] predicted logits
            target: [B, H, W] ground truth labels

        Returns:
            Scalar loss value
        """
        return self.criterion(logits, target)


class OhemCrossEntropyLoss(nn.Module):
    """Online Hard Example Mining Cross Entropy Loss.

    Focuses on hard-to-classify pixels by keeping only pixels
    with probability below a threshold.

    Args:
        ignore_index: Label to ignore in loss computation
        thresh: Probability threshold for hard examples
        min_kept: Minimum number of pixels to keep
        weight: Optional class weights
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = thresh
        self.min_kept = min_kept
        self.weight = weight

        # Base loss for weight support
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction='none'
        )

    def forward(self, logits, target):
        """
        Args:
            logits: [B, C, H, W] predicted logits
            target: [B, H, W] ground truth labels

        Returns:
            Scalar loss value
        """
        B, C, H, W = logits.shape

        # Compute per-pixel loss
        loss = self.criterion(logits, target)  # [B, H, W]

        # Compute probabilities
        prob = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Get probability of correct class
        # Flatten target and loss
        target_flat = target.view(-1)  # [B*H*W]
        loss_flat = loss.view(-1)  # [B*H*W]

        # CRITICAL: Correct way to flatten prob tensor!
        # PyTorch memory layout: [B, C, H, W] (contiguous in this order)
        # WRONG: prob.view(C, -1).t() -> mixes pixels across batches!
        # RIGHT: permute channels to last, then flatten
        # [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        prob_flat = prob.permute(0, 2, 3, 1).contiguous().view(-1, C)

        # Mask for valid pixels (not ignore_index)
        valid_mask = target_flat != self.ignore_index

        # Get probability of correct class for valid pixels
        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)

        if len(valid_indices) == 0:
            return loss.mean()

        # Get correct class probabilities
        target_valid = target_flat[valid_mask]
        prob_valid = prob_flat[valid_mask]
        loss_valid = loss_flat[valid_mask]

        # Get probability of correct class
        correct_prob = prob_valid.gather(1, target_valid.unsqueeze(1)).squeeze(1)

        # OHEM: select hard examples
        # Keep pixels with prob < thresh
        hard_mask = correct_prob < self.thresh
        num_hard = hard_mask.sum().item()

        # Ensure minimum number of pixels
        if num_hard < self.min_kept:
            # Sort by loss (descending) and keep top min_kept
            _, sorted_indices = loss_valid.sort(descending=True)
            selected_indices = sorted_indices[:self.min_kept]
            ohem_loss = loss_valid[selected_indices].mean()
        else:
            # Use hard examples
            ohem_loss = loss_valid[hard_mask].mean()

        return ohem_loss


def get_loss(cfg):
    """Build loss function from config.

    Args:
        cfg: Configuration dictionary

    Returns:
        Loss function module
    """
    loss_cfg = cfg.get('LOSS', {})
    loss_name = loss_cfg.get('NAME', 'CrossEntropy')
    ignore_index = loss_cfg.get('IGNORE_INDEX', 255)

    if loss_name == 'OhemCrossEntropy':
        return OhemCrossEntropyLoss(
            ignore_index=ignore_index,
            thresh=loss_cfg.get('THRESH', 0.7),
            min_kept=loss_cfg.get('MIN_KEPT', 100000),
            weight=None  # Can add class weights here
        )
    else:
        return CrossEntropyLoss(ignore_index=ignore_index)


# Test code
if __name__ == '__main__':
    # Test OHEM loss
    loss_fn = OhemCrossEntropyLoss(ignore_index=255, thresh=0.7, min_kept=1000)

    # Create random data
    logits = torch.randn(2, 19, 256, 256)
    target = torch.randint(0, 19, (2, 256, 256))
    target[0, 100:150, 100:150] = 255  # Add ignore region

    loss = loss_fn(logits, target)
    print(f"OHEM Loss: {loss.item():.4f}")

    # Test standard CE loss
    ce_loss_fn = CrossEntropyLoss(ignore_index=255)
    ce_loss = ce_loss_fn(logits, target)
    print(f"CE Loss: {ce_loss.item():.4f}")
