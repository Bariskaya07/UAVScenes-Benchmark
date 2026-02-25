"""
Optimizers for Training

Includes factory function to build optimizer from config.
"""

import torch
import torch.optim as optim


def get_optimizer(model, cfg):
    """Build optimizer from config.

    Args:
        model: Model to optimize
        cfg: Configuration dictionary

    Returns:
        Optimizer instance
    """
    opt_cfg = cfg.get('OPTIMIZER', {})
    opt_name = opt_cfg.get('NAME', 'adamw').lower()

    lr = opt_cfg.get('LR', 6e-5)
    weight_decay = opt_cfg.get('WEIGHT_DECAY', 0.01)

    # Fair benchmark policy: same LR for all params, no weight decay on norm/bias/1D params
    params = get_param_groups(model, lr=lr, weight_decay=weight_decay)

    if opt_name == 'adamw':
        betas = opt_cfg.get('BETAS', [0.9, 0.999])
        eps = opt_cfg.get('EPS', 1e-8)
        return optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(betas),
            eps=eps
        )
    elif opt_name == 'adam':
        betas = opt_cfg.get('BETAS', [0.9, 0.999])
        eps = opt_cfg.get('EPS', 1e-8)
        return optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(betas),
            eps=eps
        )
    elif opt_name == 'sgd':
        momentum = opt_cfg.get('MOMENTUM', 0.9)
        return optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_param_groups(model, lr, weight_decay, lr_decay_layers=None, lr_decay_rate=0.1):
    """Get parameter groups with optional layer-wise learning rate decay.

    Args:
        model: Model
        lr: Base learning rate
        weight_decay: Weight decay
        lr_decay_layers: List of layer names to apply decay
        lr_decay_rate: Decay rate for specified layers

    Returns:
        List of parameter group dictionaries
    """
    param_groups = []

    # Separate parameters by whether they should have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for bias, normalization layers, and 1D params (e.g., LN/BN weights)
        if param.ndim == 1 or 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups.append({
        'params': decay_params,
        'lr': lr,
        'weight_decay': weight_decay
    })

    param_groups.append({
        'params': no_decay_params,
        'lr': lr,
        'weight_decay': 0.0
    })

    return param_groups


# Test code
if __name__ == '__main__':
    import torch.nn as nn

    # Create dummy model
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.BatchNorm1d(10),
        nn.Linear(10, 10)
    )

    # Test optimizer creation
    cfg = {
        'OPTIMIZER': {
            'NAME': 'adamw',
            'LR': 6e-5,
            'WEIGHT_DECAY': 0.01,
            'BETAS': [0.9, 0.999]
        }
    }

    optimizer = get_optimizer(model, cfg)
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"  LR: {optimizer.param_groups[0]['lr']}")
    print(f"  Weight Decay: {optimizer.param_groups[0]['weight_decay']}")

    # Test with param groups
    param_groups = get_param_groups(model, lr=6e-5, weight_decay=0.01)
    print(f"\nParam groups: {len(param_groups)}")
    for i, pg in enumerate(param_groups):
        print(f"  Group {i}: {len(pg['params'])} params, "
              f"lr={pg['lr']}, wd={pg['weight_decay']}")
