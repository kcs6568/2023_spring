import warnings
import torch


def get_warmup_scheduler(optimizer, warmup_ratio, largest_size):
    if warmup_ratio == -1:
        return None
    elif warmup_ratio < 1:
        warmup_iters = largest_size * warmup_ratio
    elif warmup_ratio == 1:
        warmup_iters = largest_size
    elif warmup_ratio > 1:
        warmup_iters = warmup_ratio
    else:
        assert isinstance(warmup_ratio, float) or isinstance(warmup_ratio, int), \
            "Warmup ratio must be integer of floating number"
        raise ValueError("Warmup ratio must be entered in the initial phrase.")
        
    start_factor = 0.001

    lr_scheduler = LinearLR(
        optimizer, start_factor=start_factor, total_iters=warmup_iters
    )

    return lr_scheduler


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1,
                 verbose=False):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.total_iters):
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor +
                (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]
        
        
class PolynomialDecay:
    def __init__(self, temperature, max_iters, min_temp, power=0.9):
        super(PolynomialDecay, self).__init__()
        self.power = power
        self.start_temp = temperature
        self.max_iters = max_iters  # avoid zero lr
        self.min_temp = min_temp
        
    
    def decay_temp(self, cur_iter):
        return (self.start_temp * (1 - cur_iter/self.max_iters)**self.power) 



class SimpleDecay:
    def __init__(self, temperature, max_iter, decay_gamma):
        super(SimpleDecay, self).__init__()
        self.temperature = temperature
        self.decay_gamma = decay_gamma
        self.max_iter = max_iter
        
    
    def decay_temp(self, cur_iter):
        if cur_iter % self.max_iter == 0 :
            self.temperature *= self.decay_gamma
        return self.temperature


class ExponentialDecay:
    def __init__(self, temperature, decay_gamma, max_iter, power=0.9):
        super(ExponentialDecay, self).__init__()
        self.power = power
        self.start_temp = temperature
        self.max_iter = max_iter # avoid zero lr
        self.decay_gamma = decay_gamma
        
    
    def decay_temp(self, cur_iter):
        return self.start_temp * self.decay_gamma ** (cur_iter/self.max_iter)
    
    
def set_decay_fucntion(hyp):
    if 'exp' in hyp['decay_type']:
        return ExponentialDecay(hyp['temperature'], hyp['max_iter'], hyp['gamma'])
        
    elif 'simple' in hyp['decay_type']:
        return SimpleDecay(hyp['temperature'], hyp['max_iter'], hyp['gamma'])