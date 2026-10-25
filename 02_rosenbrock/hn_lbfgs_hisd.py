"""Limited-memory: hybrid Newton-L-BFGS HiSD with fixed step sizes."""

import numpy as np
from collections import deque


class HNLBFGSHiSD:
    """Limited-memory: hybrid Newton-L-BFGS HiSD with fixed step sizes."""

    def __init__(
        self, m=10, beta1=1.0, beta2=1.0, tol=1e-12, epsilon_eig=1e-10, tol_curv=None
    ):
        self.m = m
        self.beta1 = beta1
        self.beta2 = beta2
        self.tol = tol
        self.tol_curv = tol if tol_curv is None else tol_curv
        self.epsilon_eig = epsilon_eig
        if not np.isfinite(self.epsilon_eig) or self.epsilon_eig <= 0:
            raise ValueError("epsilon_eig must be finite and positive")
        if not np.isfinite(self.tol_curv) or self.tol_curv <= 0:
            raise ValueError("tol_curv must be finite and positive")
        self.s_history = deque(maxlen=m)
        self.y_history = deque(maxlen=m)
        self.rho_history = deque(maxlen=m)

    def _project_perp(self, v, V):
        """Project onto the orthogonal complement of the columns of V."""
        return v - V @ (V.T @ v)

    def _two_loop_recursion(self, g, gamma):
        """Apply the stored L-BFGS pairs with initial inverse gamma * I."""
        q = g.copy()
        alphas = []
        s_list = list(self.s_history)
        y_list = list(self.y_history)
        rho_list = list(self.rho_history)
        n_mem = len(s_list)
        for i in range(n_mem - 1, -1, -1):
            alpha = rho_list[i] * (s_list[i] @ q)
            q -= alpha * y_list[i]
            alphas.append(alpha)
        r = gamma * q
        alphas.reverse()
        for i in range(n_mem):
            beta = rho_list[i] * (y_list[i] @ r)
            r += s_list[i] * (alphas[i] - beta)
        return r

    def run(
        self, f, grad_f, x0, V0, eigen_solver, lambdas_solver, max_iter=2000, tol=1e-12
    ):
        """Return (final_point, history). The gradient tolerance is set by tol here.

        x_history includes the initial point; grad_norm records updated points.
        eigen_solver(x, V) supplies the unstable subspace, and
        lambdas_solver(x, V) supplies its corresponding curvatures."""
        d = len(x0)
        x, V = (x0.copy(), V0.copy())
        history = dict(
            energy=[],
            grad_norm=[],
            step_norm=[],
            x_history=[x.copy()],
            bfgs_updates=[],
            curvature=[],
            phase=[],
        )
        self.s_history.clear()
        self.y_history.clear()
        self.rho_history.clear()
        gamma = 1.0
        g = grad_f(x)
        E = f(x)
        gnorm = np.linalg.norm(g)
        for n in range(max_iter):
            lams = lambdas_solver(x, V)
            neg_step = np.zeros_like(x)
            for i, lam in enumerate(lams):
                alpha = max(abs(lam), self.epsilon_eig)
                neg_step += 1.0 / alpha * (V[:, i] @ g) * V[:, i]
            g_perp = self._project_perp(g, V)
            if len(self.s_history) > 0:
                lbfgs_dir = self._two_loop_recursion(g_perp, gamma)
            else:
                lbfgs_dir = g_perp
            bfgs_step = self._project_perp(lbfgs_dir, V)
            beta1 = self.beta1
            beta2 = self.beta2
            x_new = x + beta1 * neg_step - beta2 * bfgs_step
            g_new = grad_f(x_new)
            E_new = f(x_new)
            step_norm = np.linalg.norm(x_new - x)
            gnorm = np.linalg.norm(g_new)
            if gnorm <= tol:
                history["energy"].append(E_new)
                history["grad_norm"].append(gnorm)
                history["step_norm"].append(step_norm)
                history["x_history"].append(x_new.copy())
                history["bfgs_updates"].append("stopped")
                history["curvature"].append(0.0)
                x, g = (x_new, g_new)
                break
            V_new = eigen_solver(x_new, V)
            s_vec = self._project_perp(x_new - x, V_new)
            y_vec = self._project_perp(g_new - g, V_new)
            status = "skipped"
            curv = float(s_vec @ y_vec)
            s_sq = float(s_vec @ s_vec)
            if np.isfinite(curv) and curv > self.tol_curv * s_sq:
                rho = 1.0 / curv
                self.s_history.append(s_vec)
                self.y_history.append(y_vec)
                self.rho_history.append(rho)
                y_sq = float(y_vec @ y_vec)
                gamma = curv / y_sq if y_sq > 0 else 1.0
                status = "updated"
            step_norm = np.linalg.norm(x_new - x)
            gnorm = np.linalg.norm(g_new)
            history["energy"].append(E_new)
            history["grad_norm"].append(gnorm)
            history["step_norm"].append(step_norm)
            history["bfgs_updates"].append(status)
            history["curvature"].append(curv)
            history["x_history"].append(x_new.copy())
            x, V, g, E = (x_new, V_new, g_new, E_new)
        return (x, history)
