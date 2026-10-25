"""A-HiSD: momentum-accelerated high-index saddle dynamics in PyTorch."""

import itertools
from functools import reduce

import torch
from torch.optim.optimizer import Optimizer, required


class AcceleratedHiSD(Optimizer):
    """A-HiSD: reflected-gradient steps with a momentum buffer."""

    def __init__(self, params, lr=0.0001, gamma=0.9, v=required, v_lr=0.0001):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma value: {}".format(gamma))
        params, params_copy = itertools.tee(params, 2)
        nn_params = sum((p.numel() for p in params_copy))
        if not v.shape[0] == nn_params:
            raise ValueError(
                "Invalid dimenson of direction vector v: {}".format(v.shape[0])
                + ". Expected dimension: {}".format(nn_params)
            )
        defaults = dict(lr=lr, gamma=gamma, v=v, v_lr=v_lr)
        super(AcceleratedHiSD, self).__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError(
                "AcceleratedHiSD does not support per-parameter options (parameter groups)"
            )
        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0
            )
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single hisd step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and return loss.
        """
        assert len(self.param_groups) == 1
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group = self.param_groups[0]
        lr = group["lr"]
        gamma = group["gamma"]
        v = group["v"]
        state = self.state[self._params[0]]
        if gamma != 0:
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros(
                    (self._numel(),), dtype=torch.float, device=self._params[0].device
                )
        d_p = self._gather_flat_grad()
        proj_onto_v = v.matmul(v.t().matmul(d_p))
        d_p.add_(proj_onto_v, alpha=-2.0)
        d_p.mul_(-lr)
        if gamma != 0:
            d_p.add_(state["momentum_buffer"], alpha=gamma)
        self._add_grad(step_size=1.0, update=d_p)
        if gamma != 0:
            state["momentum_buffer"] = d_p
        return loss
