import torch


def get_fair_param_groups(model, lr, weight_decay):
    """Fair benchmark param grouping shared across models.

    Policy:
    - Same LR for all params
    - No weight decay for bias / normalization params / 1D params
    - Standard weight decay for the rest
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = (
            param.ndim == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "bn" in name.lower()
        )
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
    ]

class PolyWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None):
        super().__init__(params, lr=lr, betas=betas,weight_decay=weight_decay, eps=1e-8)

        self.global_step = 0
        self.warmup_iter = 0 if warmup_iter is None else int(warmup_iter)
        self.warmup_ratio = 0.1 if warmup_ratio is None else float(warmup_ratio)
        self.max_iter = 0 if max_iter is None else int(max_iter)
        self.power = 0.9 if power is None else float(power)

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.warmup_iter > 0 and self.global_step < self.warmup_iter:

            # CMNeXt-consistent linear warmup: warmup_ratio -> 1.0
            alpha = self.global_step / max(1, self.warmup_iter)
            lr_mult = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        elif self.max_iter > 0 and self.global_step < self.max_iter:

            # CMNeXt/TokenFusion-style post-warmup normalized polynomial decay
            decay_start = self.warmup_iter
            decay_den = max(1, self.max_iter - decay_start)
            progress = (self.global_step - decay_start) / decay_den
            progress = min(max(progress, 0.0), 1.0)
            lr_mult = (1 - progress) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1
