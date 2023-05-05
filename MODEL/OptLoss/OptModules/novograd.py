"""NovoGrad Optimizer.
Original impl by Masashi Kimura (Convergence Lab): https://github.com/convergence-lab/novograd
Paper: `Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks`
    - https://arxiv.org/abs/1905.11286
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class NovoGrad(Optimizer):
    def __init__(self, params, grad_averaging=False, lr=0.1, betas=(0.95, 0.98), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(NovoGrad, self).__init__(params, defaults)
        self._lr = lr
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._eps = eps
        self._wd = weight_decay
        self._grad_averaging = grad_averaging

        self._momentum_initialized = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if not self._momentum_initialized:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('NovoGrad does not support sparse gradients')

                    v = torch.norm(grad)**2
                    m = grad/(torch.sqrt(v) + self._eps) + self._wd * p.data
                    state['step'] = 0
                    state['v'] = v
                    state['m'] = m
                    state['grad_ema'] = None
            self._momentum_initialized = True

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['step'] += 1

                step, v, m = state['step'], state['v'], state['m']
                grad_ema = state['grad_ema']

                grad = p.grad.data
                g2 = torch.norm(grad)**2
                grad_ema = g2 if grad_ema is None else grad_ema * \
                    self._beta2 + g2 * (1. - self._beta2)
                grad *= 1.0 / (torch.sqrt(grad_ema) + self._eps)

                if self._grad_averaging:
                    grad *= (1. - self._beta1)

                g2 = torch.norm(grad)**2
                v = self._beta2*v + (1. - self._beta2)*g2
                m = self._beta1*m + (grad / (torch.sqrt(v) + self._eps) + self._wd * p.data)
                bias_correction1 = 1 - self._beta1 ** step
                bias_correction2 = 1 - self._beta2 ** step
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                state['v'], state['m']  = v, m
                state['grad_ema'] = grad_ema
                p.data.add_(-step_size, m)
        return loss

""" Nvidia NovoGrad Optimizer.
Original impl by Nvidia from Jasper example:
    - https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper
Paper: `Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks`
    - https://arxiv.org/abs/1905.11286
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class NvNovoGrad(Optimizer):
    """
    Implements Novograd algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0.98))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging: gradient averaging
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.95, 0.98), eps=1e-8,
                 weight_decay=0, grad_averaging=False, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        grad_averaging=grad_averaging,
                        amsgrad=amsgrad)

        super(NvNovoGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NvNovoGrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported.')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                norm = torch.sum(torch.pow(grad, 2))

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, norm)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                grad.div_(denom)
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)
                if group['grad_averaging']:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                p.data.add_(-group['lr'], exp_avg)

        return loss