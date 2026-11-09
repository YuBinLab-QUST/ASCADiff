import copy
import math
import warnings

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.warmup import GradualWarmupScheduler


# customize exp lr scheduler with min lr
class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


def repeat_data(data: Data, num_repeat) -> Batch:
    datas = [copy.deepcopy(data) for i in range(num_repeat)]
    return Batch.from_data_list(datas)


def repeat_batch(batch: Batch, num_repeat) -> Batch:
    datas = batch.to_data_list()
    new_data = []
    for i in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)


# def inf_iterator(iterable):
#     iterator = iterable.__iter__()
#     while True:
#         try:
#             yield iterator.__next__()
#         except StopIteration:
#             iterator = iterable.__iter__()

def inf_iterator(iterable):
    """改进的无限迭代器,避免内存泄漏"""
    while True:
        for item in iterable:  # 让 Python 自动管理迭代器生命周期
            yield item
# def inf_iterator(iterable):
#     iterator = iterable.__iter__()
#     while True:
#         try:
#             yield iterator.__next__()
#         except StopIteration:
#             iterator = iterable.__iter__()




def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2,)
        )
    elif cfg.type == 'adamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2,)
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr
        )
    elif cfg.type == 'warmup_plateau':
        return GradualWarmupScheduler(
            optimizer,
            multiplier=cfg.multiplier,
            total_epoch=cfg.total_epoch,
            after_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=cfg.factor,
                patience=cfg.patience,
                min_lr=cfg.min_lr
            )
        )
    elif cfg.type == 'expmin':
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=cfg.factor,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'expmin_milestone':
        gamma = np.exp(np.log(cfg.factor) / cfg.milestone)
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=gamma,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'cosine_with_warmup':
        # 如果有transformers库，使用其实现
        try:
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=getattr(cfg, 'warmup_steps', 1000),
                num_training_steps=getattr(cfg, 'total_steps', 10000),
                # eta_min=getattr(cfg, 'eta_min', 0)
            )
        except ImportError:
            # 如果没有transformers库，使用自定义实现
            return CosineAnnealingWarmupScheduler(
                optimizer,
                warmup_steps=getattr(cfg, 'warmup_steps', 1000),
                total_steps=getattr(cfg, 'total_steps', 10000),
                eta_min=getattr(cfg, 'eta_min', 0)
            )





    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


class CosineAnnealingWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(CosineAnnealingWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]