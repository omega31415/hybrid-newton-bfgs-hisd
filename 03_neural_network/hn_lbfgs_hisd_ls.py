"""Limited-memory LS: hybrid Newton-BFGS HiSD for neural-network parameters."""

from collections import deque
import math
import torch
from torch.optim.optimizer import required
from _hisd_base import NewtonHiSDBase


class HNLBFGSHiSDLS(NewtonHiSDBase):
    """Limited-memory LS: PyTorch hybrid Newton-BFGS HiSD with decoupled line search.

    Call step(closure), then refresh eigenpairs and call update_bfgs()
    unless the stopping tolerance has been met. The closure must
    compute a fresh full-batch loss and call backward()."""

    def __init__(
        self,
        params,
        m=10,
        lr1=1.0,
        lr2=1.0,
        v=required,
        v_lr=0.0001,
        *,
        eigenvalues=None,
        epsilon_eig=1e-10,
        tol_curv=1e-12,
        max_ls=10,
        ls_c=0.0001,
        ls_rho=0.5,
        epsilon_g=1e-12
    ):
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise ValueError("m must be a positive integer")
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
        self.m = m
        self._s_history = deque(maxlen=m)
        self._y_history = deque(maxlen=m)
        self._rho_history = deque(maxlen=m)
        self._gamma = 1.0

    def _two_loop_recursion(self, g):
        q = g.clone()
        alphas = []
        pairs = list(zip(self._s_history, self._y_history, self._rho_history))
        for s, y, rho in reversed(pairs):
            alpha = rho * torch.dot(s, q)
            q = q - alpha * y
            alphas.append(alpha)
        r = self._gamma * q
        for (s, y, rho), alpha in zip(pairs, reversed(alphas)):
            beta = rho * torch.dot(y, r)
            r = r + (alpha - beta) * s
        return r

    def _stable_direction(self, g_perp, v):
        return self._proj_perp(self._two_loop_recursion(g_perp), v)

    def _update_memory(self, s, y):
        status, curvature = self._curvature_status(s, y)
        if status != "updated":
            return status
        y_sq = torch.dot(y, y).item()
        gamma = curvature / y_sq if y_sq > 0 else float("nan")
        rho = 1.0 / curvature
        if not math.isfinite(gamma) or gamma <= 0 or (not math.isfinite(rho)):
            return "nonfinite"
        self._s_history.append(s.detach().clone())
        self._y_history.append(y.detach().clone())
        self._rho_history.append(rho)
        self._gamma = gamma
        return "updated"
