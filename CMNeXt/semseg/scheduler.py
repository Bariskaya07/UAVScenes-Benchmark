"""
Learning Rate Schedulers for Training

Includes:
- WarmupPolyLR: Warmup + Polynomial decay (CMNeXt default)
- WarmupCosineLR: Warmup + Cosine annealing
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupPolyLR(_LRScheduler):
    """Polynomial decay with warmup.

    Learning rate schedule:
    - Warmup phase: linear increase from warmup_ratio * lr to lr
    - Decay phase: polynomial decay from lr to 0

    Args:
        optimizer: Optimizer instance
        max_iters: Total number of training iterations
        warmup_iters: Number of warmup iterations
        warmup_ratio: Initial LR = base_lr * warmup_ratio
        power: Polynomial decay power
        last_epoch: Last epoch number
    """

    def __init__(self, optimizer, max_iters, warmup_iters=10, warmup_ratio=0.1,
                 power=0.9, last_epoch=-1):
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Warmup phase: linear increase
            alpha = self.last_epoch / self.warmup_iters
            factor = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        else:
            # Decay phase: polynomial decay
            progress = (self.last_epoch - self.warmup_iters) / \
                       (self.max_iters - self.warmup_iters)
            factor = (1 - progress) ** self.power

        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupCosineLR(_LRScheduler):
    """Cosine annealing with warmup.

    Learning rate schedule:
    - Warmup phase: linear increase from warmup_ratio * lr to lr
    - Decay phase: cosine annealing from lr to min_lr

    Args:
        optimizer: Optimizer instance
        max_iters: Total number of training iterations
        warmup_iters: Number of warmup iterations
        warmup_ratio: Initial LR = base_lr * warmup_ratio
        min_lr: Minimum learning rate
        last_epoch: Last epoch number
    """

    def __init__(self, optimizer, max_iters, warmup_iters=10, warmup_ratio=0.1,
                 min_lr=1e-6, last_epoch=-1):
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Warmup phase: linear increase
            alpha = self.last_epoch / self.warmup_iters
            factor = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Decay phase: cosine annealing
            progress = (self.last_epoch - self.warmup_iters) / \
                       (self.max_iters - self.warmup_iters)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor
                    for base_lr in self.base_lrs]


def get_scheduler(optimizer, cfg, iters_per_epoch):
    """Build scheduler from config.

    Args:
        optimizer: Optimizer instance
        cfg: Configuration dictionary
        iters_per_epoch: Number of iterations per epoch

    Returns:
        Scheduler instance
    """
    scheduler_cfg = cfg.get('SCHEDULER', {})
    scheduler_name = scheduler_cfg.get('NAME', 'warmuppolylr').lower()

    epochs = cfg.get('TRAIN', {}).get('EPOCHS', 200)
    
    # GRAD_ACCUM düzeltmesi: Scheduler step sayısı = batch sayısı / grad_accum
    # Çünkü train_mm.py'de scheduler.step() her grad_accum batch'te 1 kez çağrılıyor
    grad_accum = cfg.get('TRAIN', {}).get('GRAD_ACCUM', 1)
    effective_iters_per_epoch = iters_per_epoch // grad_accum
    
    max_iters = epochs * effective_iters_per_epoch

    warmup_epochs = scheduler_cfg.get('WARMUP', 10)
    warmup_iters = warmup_epochs * effective_iters_per_epoch
    warmup_ratio = scheduler_cfg.get('WARMUP_RATIO', 0.1)

    if scheduler_name == 'warmuppolylr':
        power = scheduler_cfg.get('POWER', 0.9)
        return WarmupPolyLR(
            optimizer,
            max_iters=max_iters,
            warmup_iters=warmup_iters,
            warmup_ratio=warmup_ratio,
            power=power
        )
    elif scheduler_name == 'warmupcosinelr':
        min_lr = scheduler_cfg.get('MIN_LR', 1e-6)
        return WarmupCosineLR(
            optimizer,
            max_iters=max_iters,
            warmup_iters=warmup_iters,
            warmup_ratio=warmup_ratio,
            min_lr=min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


# Test code
if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5)

    # Test schedulers
    max_iters = 1000
    warmup_iters = 100

    # WarmupPolyLR
    scheduler_poly = WarmupPolyLR(optimizer, max_iters, warmup_iters, power=0.9)

    lrs_poly = []
    for i in range(max_iters):
        lrs_poly.append(optimizer.param_groups[0]['lr'])
        scheduler_poly.step()

    # Reset optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5)

    # WarmupCosineLR
    scheduler_cosine = WarmupCosineLR(optimizer, max_iters, warmup_iters)

    lrs_cosine = []
    for i in range(max_iters):
        lrs_cosine.append(optimizer.param_groups[0]['lr'])
        scheduler_cosine.step()

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(lrs_poly)
    plt.axvline(x=warmup_iters, color='r', linestyle='--', label='Warmup end')
    plt.title('WarmupPolyLR')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(lrs_cosine)
    plt.axvline(x=warmup_iters, color='r', linestyle='--', label='Warmup end')
    plt.title('WarmupCosineLR')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig('scheduler_test.png', dpi=150)
    print("Scheduler test saved to 'scheduler_test.png'")
