"""HiSD-BB: the NumPy BB update used in the neural-network comparison."""

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required


class HiSDBB(Optimizer):
    """Apply the reflected-gradient BB2-like rule with clipped step sizes."""

    def __init__(self, params, lr=0.1, v=required):
        super().__init__(params, dict(lr=lr, v=v))
        if len(self.param_groups) != 1:
            raise ValueError("HiSD-BB requires one parameter group")
        self._params = self.param_groups[0]["params"]
        self._previous_point = None
        self._previous_direction = None
        self._beta = lr

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        theta = (
            torch.nn.utils.parameters_to_vector(self._params).detach().numpy().copy()
        )
        grad = (
            torch.nn.utils.parameters_to_vector([p.grad for p in self._params])
            .detach()
            .numpy()
            .copy()
        )
        v = self.param_groups[0]["v"].detach().numpy()
        G = -(grad - 2.0 * v @ (v.T @ grad))
        if self._previous_direction is not None:
            s = theta - self._previous_point
            diff = G - self._previous_direction
            yy = float(np.dot(diff, diff))
            sy = float(np.dot(s, diff))
            if yy > 1e-14 and np.isfinite(yy) and np.isfinite(sy):
                beta_raw = abs(sy / yy)
                base_step = self.param_groups[0]["lr"]
                self._beta = float(np.clip(beta_raw, 0.2 * base_step, 5.0 * base_step))
        new_theta = theta + self._beta * G
        torch.nn.utils.vector_to_parameters(torch.as_tensor(new_theta), self._params)
        self._previous_point = theta
        self._previous_direction = G.copy()
        return loss
