"""Shared gradient, eigenpair and line-search operations for NN Newton HiSD."""

import math
import torch
from torch.optim import Optimizer


@torch.enable_grad()
def hessian_vector_product(f, net, v):
    params = tuple(net.parameters())
    grads = torch.autograd.grad(
        f, params, create_graph=True, retain_graph=True, allow_unused=True
    )
    flat_grad = torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for p, g in zip(params, grads)
        ]
    )
    dot = torch.dot(flat_grad, v.to(flat_grad).reshape(-1))
    if not dot.requires_grad:
        return torch.zeros_like(flat_grad)
    grads2 = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for p, g in zip(params, grads2)
        ]
    ).detach()


def hessian_mat_product(f, net, V):
    if V.ndim != 2:
        raise ValueError("V must have shape (number of parameters, k)")
    if V.shape[1] == 0:
        return V.clone()
    return torch.stack([hessian_vector_product(f, net, v) for v in V.T], dim=1)


class NewtonHiSDBase(Optimizer):

    def __init__(
        self,
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
    ):
        for name, value in (
            ("lr1", lr1),
            ("lr2", lr2),
            ("v_lr", v_lr),
            ("epsilon_eig", epsilon_eig),
            ("tol_curv", tol_curv),
            ("epsilon_g", epsilon_g),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(max_ls, bool) or not isinstance(max_ls, int) or max_ls < 1:
            raise ValueError("max_ls must be a positive integer")
        if not 0 < ls_c < 1 or not 0 < ls_rho < 1:
            raise ValueError("ls_c and ls_rho must lie strictly between 0 and 1")
        super().__init__(params, dict(lr1=lr1, lr2=lr2, v_lr=v_lr))
        if len(self.param_groups) != 1:
            raise ValueError("NewtonHiSD supports exactly one parameter group")
        self._params = self.param_groups[0]["params"]
        p0 = self._params[0]
        if p0.dtype not in (torch.float32, torch.float64):
            raise ValueError("NewtonHiSD requires float32 or float64 parameters")
        if any(
            (
                p.dtype != p0.dtype or p.device != p0.device or (not p.requires_grad)
                for p in self._params
            )
        ):
            raise ValueError("Parameters must share dtype/device and require gradients")
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            raise ValueError("v must be an orthonormal tensor of shape (d, k)")
        if v.shape[0] != self._numel() or v.shape[1] > v.shape[0]:
            raise ValueError("v has incompatible shape")
        v = v.detach().to(p0).clone()
        self._validate_basis(v)
        self.param_groups[0]["v"] = v
        self.epsilon_eig, self.tol_curv = (epsilon_eig, tol_curv)
        self.max_ls, self.ls_c, self.ls_rho = (max_ls, ls_c, ls_rho)
        self.epsilon_g = epsilon_g
        self._eigenvalues = None
        self._eigen_x = None
        self._step_pair = None
        self._grad_after_step = None
        self.last_beta1 = self.last_beta2 = 0.0
        self.last_ls_satisfied = False
        self.last_ls_evaluations = 0
        self.last_bfgs_status = "not_updated"
        self.last_curvature = None
        if eigenvalues is not None:
            self._set_eigenpairs(v, eigenvalues)
        elif v.shape[1] == 0:
            self._set_eigenpairs(v, v.new_empty(0))

    @staticmethod
    def _validate_basis(v):
        eye = torch.eye(v.shape[1], dtype=v.dtype, device=v.device)
        if not torch.isfinite(v).all() or not torch.allclose(
            v.T @ v, eye, atol=1e-06, rtol=1e-05
        ):
            raise ValueError("Columns of v must be orthonormal")

    def _numel(self):
        return sum((p.numel() for p in self._params))

    def _gather_flat_params(self):
        return torch.cat([p.detach().reshape(-1) for p in self._params])

    def _gather_flat_grad(self):
        if any((p.grad is None for p in self._params)):
            raise RuntimeError(
                "closure must call backward() for all optimized parameters"
            )
        return torch.cat(
            [
                (p.grad.to_dense() if p.grad.is_sparse else p.grad).detach().reshape(-1)
                for p in self._params
            ]
        ).clone()

    @torch.no_grad()
    def _set_flat_params(self, x):
        offset = 0
        for p in self._params:
            p.copy_(x[offset : offset + p.numel()].view_as(p))
            offset += p.numel()

    @torch.no_grad()
    def _set_flat_grad(self, g):
        offset = 0
        for p in self._params:
            p.grad = g[offset : offset + p.numel()].view_as(p).clone()
            offset += p.numel()

    @staticmethod
    def _proj_perp(u, v):
        return u - v @ (v.T @ u)

    def _evaluate(self, closure):
        self.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = closure()
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise RuntimeError("closure must return a scalar loss tensor")
        return (loss, self._gather_flat_grad())

    def _check_eigenpairs(self):
        if self._eigenvalues is None:
            raise RuntimeError(
                "Initialize eigenpairs before step(): pass eigenvalues= or call update_eigenpairs() with a fresh loss"
            )
        if self.param_groups[0]["v"].shape[1] and (
            not torch.equal(self._gather_flat_params(), self._eigen_x)
        ):
            raise RuntimeError(
                "Refresh eigenpairs at the current parameters before step() or update_bfgs()"
            )

    def _negative_direction(self, g, v):
        alpha = self._eigenvalues.abs().clamp_min(self.epsilon_eig)
        return v @ (v.T @ g / alpha)

    def _line_search(self, closure, x, g, v, neg_step, bfgs_step):
        beta1, beta2 = (self.param_groups[0]["lr1"], self.param_groups[0]["lr2"])
        g_perp = self._proj_perp(g, v)
        norm_perp, norm_v = (g_perp.norm().item(), (g - g_perp).norm().item())

        def trial():
            self._set_flat_params(x + beta1 * neg_step - beta2 * bfgs_step)
            loss, g_trial = self._evaluate(closure)
            self.last_ls_evaluations += 1
            finite = bool(torch.isfinite(loss).all() and torch.isfinite(g_trial).all())
            trial_perp = self._proj_perp(g_trial, v)
            v_ok = finite and (
                norm_v < self.epsilon_g
                or (g_trial - trial_perp).norm().item()
                <= (1 - self.ls_c * beta1) * norm_v
            )
            perp_ok = finite and (
                norm_perp < self.epsilon_g
                or trial_perp.norm().item() <= (1 - self.ls_c * beta2) * norm_perp
            )
            return (loss, g_trial, v_ok, perp_ok, finite)

        self.last_ls_evaluations = 0
        for _ in range(self.max_ls):
            loss, g_trial, v_ok, perp_ok, finite = trial()
            if v_ok and perp_ok:
                break
            if not v_ok:
                beta1 *= self.ls_rho
            if not perp_ok:
                beta2 *= self.ls_rho
        else:
            loss, g_trial, v_ok, perp_ok, finite = trial()
        if not finite:
            raise FloatingPointError("The final line-search trial is nonfinite")
        self.last_beta1, self.last_beta2 = (beta1, beta2)
        self.last_ls_satisfied = v_ok and perp_ok
        return (loss, g_trial)

    @torch.no_grad()
    def step(self, closure=None):
        """Take one step; refresh eigenpairs then update BFGS unless stopping.

        The returned loss has already been used for backward(). Recompute a
        forward loss before any Hessian-vector product or autograd diagnostic.
        """
        if closure is None:
            raise RuntimeError("NewtonHiSD.step() requires a loss/backward closure")
        self._check_eigenpairs()
        self._step_pair = None
        x = self._gather_flat_params()
        loss, g = self._evaluate(closure)
        if not torch.isfinite(loss).all() or not torch.isfinite(g).all():
            raise FloatingPointError("The starting loss or gradient is nonfinite")
        v = self.param_groups[0]["v"]
        neg_step = self._negative_direction(g, v)
        bfgs_step = self._stable_direction(self._proj_perp(g, v), v)
        try:
            loss, g_new = self._line_search(closure, x, g, v, neg_step, bfgs_step)
        except Exception:
            self._set_flat_params(x)
            self._set_flat_grad(g)
            raise
        x_new = self._gather_flat_params()
        self._grad_after_step = g_new.detach().clone()
        self._step_pair = (x.clone(), g.clone(), x_new.clone(), g_new.clone())
        self.last_bfgs_status = "pending"
        return loss

    def _curvature_status(self, s, y):
        curvature = torch.dot(s, y).item()
        s_sq = torch.dot(s, s).item()
        self.last_curvature = curvature
        if not math.isfinite(curvature) or not math.isfinite(s_sq):
            return ("nonfinite", curvature)
        if curvature < 0:
            return ("negative", curvature)
        if curvature <= self.tol_curv * s_sq:
            return ("small_positive", curvature)
        return ("updated", curvature)

    @torch.no_grad()
    def update_bfgs(self):
        """Consume the last step once, projecting both differences with V_new."""
        if self._step_pair is None:
            return "no_pair"
        x, g, x_new, g_new = self._step_pair
        if not torch.equal(self._gather_flat_params(), x_new):
            raise RuntimeError(
                "Parameters changed after step(); cached secants are stale"
            )
        self._check_eigenpairs()
        v = self.param_groups[0]["v"]
        s = self._proj_perp(x_new - x, v)
        y = self._proj_perp(g_new - g, v)
        self.last_bfgs_status = self._update_memory(s, y)
        self._step_pair = None
        return self.last_bfgs_status

    def _set_eigenpairs(self, v, eigenvalues):
        v = v.detach().to(self._params[0]).clone()
        eigenvalues = (
            torch.as_tensor(eigenvalues, dtype=v.dtype, device=v.device)
            .detach()
            .clone()
        )
        if eigenvalues.shape != (v.shape[1],) or not torch.isfinite(eigenvalues).all():
            raise ValueError("eigenvalues must be finite and match the columns of v")
        self._validate_basis(v)
        self.param_groups[0]["v"] = v
        self._eigenvalues = eigenvalues
        self._eigen_x = self._gather_flat_params().clone()

    def update_eigenpairs(self, loss, net):
        """Refresh the lowest k Hessian eigenpairs by a dense symmetric solve."""
        v = self.param_groups[0]["v"]
        d, k = v.shape
        if k == 0:
            self._set_eigenpairs(v, v.new_empty(0))
            return
        H = hessian_mat_product(loss, net, torch.eye(d, dtype=v.dtype, device=v.device))
        eigenvalues, vectors = torch.linalg.eigh(0.5 * (H + H.T))
        self._set_eigenpairs(vectors[:, :k], eigenvalues[:k])

    @property
    def eigenvalues(self):
        return None if self._eigenvalues is None else self._eigenvalues.detach().clone()

    @property
    def grad_after_step(self):
        """New gradient snapshot; independent of later zero_grad/diagnostics."""
        if self._grad_after_step is None:
            raise RuntimeError("No completed step is available")
        return self._grad_after_step.clone()
