from torch.optim.lr_scheduler import _LRScheduler
import math


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, warmup_steps: int,
                 max_steps: int, cycles: float = .5, current_step: int = None):
        
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.cycles = cycles
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        
        if current_step < self.warmup_steps:
            progress = float(current_step) / float(max(1., self.warmup_steps))
            new_lr = self.initial_lr * progress
        else:
            progress = float(current_step - self.warmup_steps) / float(max(1., self.max_steps - self.warmup_steps))
            new_lr = self.initial_lr * (1. - progress) ** 0.9
            #new_lr = self.initial_lr * (1 + math.cos(math.pi * float(self.cycles) * 2. * progress)) / 2.

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr