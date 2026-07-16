# This file is covered by the LICENSE file in the root of this project.

import torch.optim.lr_scheduler as toptim


class ConstantLearningRate:
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr
        self.last_epoch = 0
        self.max_steps = None
        self._apply_lr()

    def _apply_lr(self):
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = int(epoch)
        self._apply_lr()

    def state_dict(self):
        return {
            "lr": self.lr,
            "last_epoch": self.last_epoch,
            "max_steps": self.max_steps,
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = int(state_dict.get("last_epoch", 0))
        self.max_steps = state_dict.get("max_steps")
        self._apply_lr()

    def set_max_steps(self, max_steps):
        max_steps = int(max_steps)
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps


class WarmupConstantLearningRate:
    """Linearly warm up to the configured LR, then keep it constant."""

    def __init__(self, optimizer, lr, warmup_steps, start_lr=0.0):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if start_lr < 0 or start_lr > lr:
            raise ValueError("start_lr must be between 0 and lr")
        self.optimizer = optimizer
        self.lr = lr
        self.start_lr = start_lr
        self.warmup_steps = max(int(warmup_steps), 1)
        self.last_epoch = 0
        self.max_steps = None
        self._apply_lr()

    def _current_lr(self):
        warmup_progress = min(self.last_epoch / self.warmup_steps, 1.0)
        return self.start_lr + warmup_progress * (self.lr - self.start_lr)

    def _apply_lr(self):
        current_lr = self._current_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = current_lr

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = int(epoch)
        self._apply_lr()

    def state_dict(self):
        return {
            "lr": self.lr,
            "start_lr": self.start_lr,
            "warmup_steps": self.warmup_steps,
            "last_epoch": self.last_epoch,
            "max_steps": self.max_steps,
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = int(state_dict.get("last_epoch", 0))
        self.max_steps = state_dict.get("max_steps")
        self._apply_lr()

    def set_max_steps(self, max_steps):
        max_steps = int(max_steps)
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps


class WarmupLR(toptim._LRScheduler):
    """ Warmup learning rate scheduler.
        Initially, increases the learning rate from 0 to the final value, in a
        certain number of steps. After this number of steps, each step decreases
        LR exponentially.
    """

    def __init__(self, optimizer, lr, warmup_steps, momentum, decay):
        # cyclic params
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.momentum = momentum
        self.decay = decay

        # cap to one
        if self.warmup_steps < 1:
            self.warmup_steps = 1

        # cyclic lr
        self.initial_scheduler = toptim.CyclicLR(self.optimizer,
                                                 base_lr=0,
                                                 max_lr=self.lr,
                                                 step_size_up=self.warmup_steps,
                                                 step_size_down=self.warmup_steps,
                                                 cycle_momentum=False,
                                                 base_momentum=self.momentum,
                                                 max_momentum=self.momentum)

        # our params
        self.last_epoch = -1  # fix for pytorch 1.1 and below
        self.finished = False  # am i done
        super().__init__(optimizer)

    def get_lr(self):
        return [self.lr * (self.decay ** self.last_epoch) for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished or self.initial_scheduler.last_epoch >= self.warmup_steps:
            if not self.finished:
                self.base_lrs = [self.lr for lr in self.base_lrs]
                self.finished = True
            return super(WarmupLR, self).step(epoch)
        else:
            return self.initial_scheduler.step(epoch)


class WarmupCosineLR(toptim._LRScheduler):
    """ Warmup learning rate scheduler.
        Initially, increases the learning rate from 0 to the final value, in a
        certain number of steps. After this number of steps, each step decreases
        LR exponentially.
    """

    def __init__(self, optimizer, lr, warmup_steps, momentum, max_steps):
        # cyclic params
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.momentum = momentum

        # cap to one
        if self.warmup_steps < 1:
            self.warmup_steps = 1

        # cyclic lr
        self.cosine_scheduler = toptim.CosineAnnealingLR(
            self.optimizer, T_max=max_steps)

        self.initial_scheduler = toptim.CyclicLR(self.optimizer,
                                                 base_lr=0,
                                                 max_lr=self.lr,
                                                 step_size_up=self.warmup_steps,
                                                 step_size_down=self.warmup_steps,
                                                 cycle_momentum=False,
                                                 base_momentum=self.momentum,
                                                 max_momentum=self.momentum)

        # our params
        self.last_epoch = -1  # fix for pytorch 1.1 and below
        self.finished = False  # am i done
        super().__init__(optimizer)

    def step(self, epoch=None):
        if self.finished or self.initial_scheduler.last_epoch >= self.warmup_steps:
            if not self.finished:
                self.base_lrs = [self.lr for lr in self.base_lrs]
                self.finished = True
            return self.cosine_scheduler.step(epoch)
        else:
            return self.initial_scheduler.step(epoch)
