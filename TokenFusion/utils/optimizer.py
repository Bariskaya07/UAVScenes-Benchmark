"""
Optimizer and Learning Rate Scheduler for TokenFusion

Implements:
- PolyWarmupAdamW: AdamW with polynomial learning rate decay and warmup
- Parameter group utilities for encoder/decoder

Reference:
    TokenFusion paper uses PolyWarmupAdamW with:
    - Base LR: 6e-5
    - Warmup: 1500 iterations
    - Polynomial power: 1.0
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math


class PolyWarmupAdamW:
    """
    AdamW optimizer with polynomial learning rate decay and linear warmup.

    Learning rate schedule:
        - Warmup phase: linear increase from 0 to base_lr
        - Decay phase: polynomial decay lr = base_lr * (1 - iter/max_iter)^power

    Args:
        params: Model parameters or parameter groups
        lr: Base learning rate
        weight_decay: Weight decay coefficient
        betas: Adam betas
        warmup_iter: Number of warmup iterations
        max_iter: Maximum training iterations
        power: Polynomial power for decay (1.0 = linear)
    """

    def __init__(
        self,
        params,
        lr=6e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_iter=1500,
        max_iter=40000,
        power=1.0
    ):
        self.base_lr = lr
        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.power = power

        # Create AdamW optimizer
        self.optimizer = AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )

        # Create LR scheduler
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda
        )

        self.current_iter = 0

    def _lr_lambda(self, current_iter):
        """
        Compute learning rate multiplier.

        Args:
            current_iter: Current training iteration

        Returns:
            Learning rate multiplier
        """
        if current_iter < self.warmup_iter:
            # Linear warmup from warmup_ratio to 1.0 (matching CMNeXt)
            warmup_ratio = 0.1  # CMNeXt paper setting
            alpha = float(current_iter) / float(max(1, self.warmup_iter))
            return warmup_ratio + (1 - warmup_ratio) * alpha
        else:
            # Polynomial decay
            return (1 - (current_iter - self.warmup_iter) / (self.max_iter - self.warmup_iter)) ** self.power

    def step(self):
        """Perform optimization step and update LR."""
        self.optimizer.step()
        self.scheduler.step()
        self.current_iter += 1

    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        """Get state dict for checkpointing."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'current_iter': self.current_iter
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.current_iter = state_dict['current_iter']


class WarmupPolyLR:
    """
    Polynomial LR scheduler with warmup (standalone, for use with any optimizer).

    Args:
        optimizer: PyTorch optimizer
        warmup_iter: Number of warmup iterations
        max_iter: Maximum training iterations
        warmup_ratio: Initial LR ratio during warmup
        power: Polynomial decay power
    """

    def __init__(
        self,
        optimizer,
        warmup_iter=1500,
        max_iter=40000,
        warmup_ratio=1e-6,
        power=1.0
    ):
        self.optimizer = optimizer
        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.warmup_ratio = warmup_ratio
        self.power = power

        # Store base learning rates
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_iter = 0

    def step(self):
        """Update learning rate."""
        self.current_iter += 1
        lr_mult = self._get_lr_mult()

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_mult

    def _get_lr_mult(self):
        """Get learning rate multiplier."""
        if self.current_iter <= self.warmup_iter:
            # Linear warmup
            k = (1 - self.warmup_ratio) * self.current_iter / self.warmup_iter + self.warmup_ratio
            return k
        else:
            # Polynomial decay
            return (1 - (self.current_iter - self.warmup_iter) / (self.max_iter - self.warmup_iter)) ** self.power

    def get_lr(self):
        """Get current learning rates."""
        return [pg['lr'] for pg in self.optimizer.param_groups]


def get_optimizer(model, cfg):
    """
    Create optimizer from config.

    Args:
        model: WeTr model instance
        cfg: Config with optimizer settings

    Returns:
        PolyWarmupAdamW optimizer
    """
    # Get parameter groups
    param_groups = model.get_param_groups()

    # Different LR for different parameter groups
    # encoder non-norm: base_lr
    # encoder norm: base_lr * 10
    # decoder: base_lr * 10
    base_lr = getattr(cfg, 'lr', 6e-5)
    lr_mult = getattr(cfg, 'lr_mult', 10.0)

    params = [
        {'params': param_groups[0], 'lr': base_lr, 'weight_decay': cfg.weight_decay},
        {'params': param_groups[1], 'lr': base_lr * lr_mult, 'weight_decay': 0.0},
        {'params': param_groups[2], 'lr': base_lr * lr_mult, 'weight_decay': cfg.weight_decay}
    ]

    optimizer = PolyWarmupAdamW(
        params,
        lr=base_lr,
        weight_decay=getattr(cfg, 'weight_decay', 0.01),
        warmup_iter=getattr(cfg, 'warmup_iter', 1500),
        max_iter=getattr(cfg, 'max_iter', 40000),
        power=getattr(cfg, 'power', 1.0)
    )

    return optimizer


def create_optimizer(model, lr=6e-5, weight_decay=0.01, warmup_iter=1500, max_iter=40000):
    """
    Create optimizer with default settings.

    Args:
        model: WeTr model
        lr: Base learning rate
        weight_decay: Weight decay
        warmup_iter: Warmup iterations
        max_iter: Max iterations

    Returns:
        PolyWarmupAdamW optimizer
    """
    # Get parameter groups from model
    if hasattr(model, 'get_param_groups'):
        param_groups = model.get_param_groups()
        params = [
            {'params': param_groups[0], 'lr': lr, 'weight_decay': weight_decay},
            {'params': param_groups[1], 'lr': lr * 10, 'weight_decay': 0.0},
            {'params': param_groups[2], 'lr': lr * 10, 'weight_decay': weight_decay}
        ]
    else:
        params = model.parameters()

    return PolyWarmupAdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        warmup_iter=warmup_iter,
        max_iter=max_iter
    )
