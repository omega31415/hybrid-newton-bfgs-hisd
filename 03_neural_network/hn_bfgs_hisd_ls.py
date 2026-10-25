"""Full-memory LS: hybrid Newton-BFGS HiSD for neural-network parameters."""

import math
import torch
from torch.optim.optimizer import required
from _hisd_base import NewtonHiSDBase


class HNBFGSHiSDLS(NewtonHiSDBase):
    """Full-memory LS: PyTorch hybrid Newton-BFGS HiSD with decoupled line search.

    Call step(closure), then refresh eigenpairs and call update_bfgs()
    unless the stopping tolerance has been met. The closure must
    compute a fresh full-batch loss and call backward()."""

    def __init__(
        self,
        params,
        lr1=1.0,
        lr2=1.0,
        v=required,
        v_lr=0.0001,
        H_init_scale=1.0,
        *,
        eigenvalues=None,
        epsilon_eig=1e-10,
        tol_curv=1e-08,
        max_ls=12,
        ls_c=0.0001,
        ls_rho=0.5,
        epsilon_g=1e-12
    ):
        if not math.isfinite(H_init_scale) or H_init_scale <= 0:
            raise ValueError("H_init_scale must be finite and positive")
        super().__init__(
            params,
            lr1,
            lr2,
            v,
            v_lr,
            eigenvalues,
            epsilon_eig,
            tol_curv,
            max_ls,
            ls_c,
            ls_rho,
            epsilon_g,
        )
        p = self._params[0]
        self._H = H_init_scale * torch.eye(
            self._numel(), dtype=p.dtype, device=p.device
        )

    def _stable_direction(self, g_perp, v):
        return self._proj_perp(self._H @ g_perp, v)

    def _update_memory(self, s, y):
        status, curvature = self._curvature_status(s, y)
        if status != "updated":
            return status
        rho = 1.0 / curvature
        Hy = self._H @ y
        H_new = (
            self._H
            - rho * (torch.outer(s, Hy) + torch.outer(Hy, s))
            + (rho + rho * rho * torch.dot(y, Hy)) * torch.outer(s, s)
        )
        if not torch.isfinite(H_new).all():
            return "nonfinite"
        self._H = 0.5 * (H_new + H_new.T)
        return "updated"

    @property
    def H_matrix(self):
        return self._H.detach().clone()
