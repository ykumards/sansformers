"""Learning rate policies."""

import numpy as np
import torch


def lr_sched_steps(cfg, optimizer):
    """Steps schedule (cfg.OPTIM.LR_POLICY = 'steps')."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: min(i / (cfg.OPTIM.LR_WARMUP / cfg.TRAIN.BATCH_SIZE), 1.0)
    )


def lr_sched_const(cfg, optimizer):
    """Steps schedule (cfg.OPTIM.LR_POLICY = 'const')."""
    return torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lambda ep: 1.0, last_epoch=-1
    )


def lr_sched_exp(cfg, optimizer):
    """Exponential schedule (cfg.OPTIM.LR_POLICY = 'exp')."""
    return torch.optim.lr_scheduler.ExponentialLR(
        optimizer, cfg.OPTIM.LR_GAMMA, last_epoch=-1
    )


def lr_sched_cos(cfg, optimizer):
    """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.OPTIM.LR_WARMUP, T_mult=1, eta_min=1e-5, last_epoch=-1
    )


def lr_sched_1cycle(cfg, optimizer):
    """Cosine schedule (cfg.OPTIM.LR_POLICY = '1cycle')."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.OPTIM.BASE_LR,
        anneal_strategy="linear",
        epochs=cfg.OPTIM.MAX_EPOCHS,
        steps_per_epoch=cfg.OPTIM.STEPS_PER_EPOCH,
    )


def get_lr_sched(cfg, optimizer):
    """Retrieves the specified lr policy function"""
    lr_fun = "lr_sched_" + cfg.OPTIM.LR_POLICY
    if lr_fun not in globals():
        raise NotImplementedError("Unknown LR policy:" + cfg.OPTIM.LR_POLICY)
    return globals()[lr_fun](cfg, optimizer)
