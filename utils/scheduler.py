# Adapted from https://github.com/chemprop/chemprop/blob/master/chemprop/nn_utils.py
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        total_epochs,
        steps_per_epoch,
        init_lr,
        max_lr,
        final_lr,
        fine_tune_coff=1.0,
        fine_tune_param_idx=0,
    ):

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array([warmup_epochs] * self.num_lrs)
        self.total_epochs = np.array([total_epochs] * self.num_lrs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array([init_lr] * self.num_lrs)
        self.max_lr = np.array([max_lr] * self.num_lrs)
        self.final_lr = np.array([final_lr] * self.num_lrs)
        self.lr_coff = np.array([1] * self.num_lrs)
        self.fine_tune_param_idx = fine_tune_param_idx
        self.lr_coff[self.fine_tune_param_idx] = fine_tune_coff

        self.current_step = 0
        self.lr = [init_lr] * self.num_lrs
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))
        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        return list(self.lr)

    def step(self, current_step=None):

        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1
        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:
                self.lr[i] = self.final_lr[i]
            self.lr[i] *= self.lr_coff[i]
            self.optimizer.param_groups[i]["lr"] = self.lr[i]
