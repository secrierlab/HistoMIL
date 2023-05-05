"""
modules for SSL pipeline

optimizer:
- LARS

batch renorm:
- BatchRenorm1d
- BatchRenorm2d
- BatchRenorm3d
"""
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
################################################################
#     lars
################################################################
"""
Layer-wise adaptive rate scaling for SGD in PyTorch!
Based on https://github.com/noahgolmant/pytorch-lars
"""



class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
        max_epoch: maximum training epoch to determine polynomial LR decay.

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=1.0, momentum=0.9, weight_decay=0.0005, eta=0.001, max_epoch=200, warmup_epochs=1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            max_epoch=max_epoch,
            warmup_epochs=warmup_epochs,
            use_lars=True,
        )
        super().__init__(params, defaults)

    def step(self, epoch=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            lr = group["lr"]
            warmup_epochs = group["warmup_epochs"]
            use_lars = group["use_lars"]
            group["lars_lrs"] = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)

                # Global LR computed on polynomial decay schedule
                warmup = min((1 + float(epoch)) / warmup_epochs, 1)
                global_lr = lr * warmup

                # Update the momentum term
                if use_lars:
                    # Compute local learning rate for this layer
                    local_lr = eta * weight_norm / (grad_norm + weight_decay * weight_norm)
                    actual_lr = local_lr * global_lr
                    group["lars_lrs"].append(actual_lr.item())
                else:
                    actual_lr = global_lr
                    group["lars_lrs"].append(global_lr)

                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                else:
                    buf = param_state["momentum_buffer"]

                buf.mul_(momentum).add_(d_p + weight_decay * p.data, alpha=actual_lr)
                p.data.add_(-buf)

        return loss

################################################################
#        batch renorm
################################################################
"""
From https://github.com/ludvb/batchrenorm
@article{batchrenomalization,
  author    = {Sergey Ioffe},
  title     = {Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models},
  journal   = {arXiv preprint arXiv:1702.03275},
  year      = {2017},
}
"""
__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]


class BatchRenorm(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.float))
        self.register_buffer("running_std", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.weight = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(1.0, 3.0)

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(0.0, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            batch_mean = x.mean(dims)
            batch_std = x.std(dims, unbiased=False) + self.eps
            r = (batch_std.detach() / self.running_std.view_as(batch_std)).clamp_(
                1 / self.rmax.item(), self.rmax.item()
            )
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean)) / self.running_std.view_as(batch_std)
            ).clamp_(-self.dmax.item(), self.dmax.item())
            x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (batch_mean.detach() - self.running_mean)
            self.running_std += self.momentum * (batch_std.detach() - self.running_std)
            self.num_batches_tracked += 1
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")
        
################################################################
#
################################################################
class NormW_Linear(nn.Linear):
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=1, keepdim=True) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.linear(x, weight, self.bias)

class MLP(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, num_layers, weight_standardization=False, normalization=None
    ):
        super().__init__()

        if normalization is not None:
            assert callable(normalization), "normalization must be callable"

        if num_layers == 0:
            self.net = torch.nn.Identity()
        elif num_layers == 1:
            self.net = torch.nn.Linear(input_dim, output_dim)
            return
        elif num_layers >= 2:
            linear_net = NormW_Linear if weight_standardization else torch.nn.Linear

            layers = []
            prev_dim = input_dim
            for _ in range(num_layers - 1):
                layers.append(linear_net(prev_dim, hidden_dim))
                if normalization is not None:
                    layers.append(normalization())
                layers.append(torch.nn.ReLU())
                prev_dim = hidden_dim

            layers.append(torch.nn.Linear(hidden_dim, output_dim))

            self.net = torch.nn.Sequential(*layers)
        else:
            raise ValueError("num_layers must be >= 0")

    def forward(self, x):
        return self.net(x)
