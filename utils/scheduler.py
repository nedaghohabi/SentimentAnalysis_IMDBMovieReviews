from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    MultiStepLR,
    CyclicLR,
    ExponentialLR,
    OneCycleLR,
    SequentialLR,
)
import numpy as np

class BaseScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        raise NotImplementedError("step method must be implemented in child class")
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    

class WarmupCosineAnnealingScheduler(BaseScheduler):
    def __init__(self, optimizer, warmup_epochs, cosine_epochs, total_epochs, min_lr):
        super().__init__(optimizer)

        if warmup_epochs + cosine_epochs > total_epochs:
            raise ValueError("Warmup + Cosine epochs must be less than total epochs")
        
        warmup_scheduler = LinearLR(optimizer, start_factor=.1, total_iters=warmup_epochs)
        fixed_scheduler = LinearLR(optimizer, start_factor=1, total_iters=total_epochs - warmup_epochs - cosine_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=min_lr)
        schedulers = [warmup_scheduler, fixed_scheduler, cosine_scheduler]
        milestones = [warmup_epochs, total_epochs - cosine_epochs]
        self.scheduler = SequentialLR(optimizer, schedulers, milestones)

    def step(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_last_lr()