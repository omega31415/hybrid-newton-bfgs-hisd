"""HiSD-BB: reflected-gradient BB steps; the paper uses gamma=0."""

import numpy as np


class HiSDBB:
    """HiSD-BB: reflected-gradient BB steps; the paper uses gamma=0."""

    def __init__(
        self,
        beta=0.001,
        gamma=0.5,
        use_bb=True,
        bb_lower=0.2,
        bb_upper=5.0,
        bb_eps=1e-14,
    ):
        self.beta0 = beta
        self.beta = beta
        self.gamma = gamma
        self.use_bb = use_bb
        self.bb_lower = bb_lower
        self.bb_upper = bb_upper
        self.bb_eps = bb_eps

    def run(
        self,
        f,
        grad_f,
        x0,
        V0,
        eigen_solver,
        lambdas_solver=None,
        max_iter=2000,
        tol=1e-10,
    ):
        """Return (final_point, history). The gradient tolerance is set by tol here.

        x_history includes the initial point; grad_norm records updated points.
        eigen_solver(x, V) supplies the unstable subspace, and
        lambdas_solver(x, V) supplies its corresponding curvatures."""
        x = x0.copy()
        x_prev = x0.copy()
        V = V0.copy()
        g = grad_f(x)
        G_prev = None
        beta_n = self.beta0
        history = dict(
            energy=[],
            grad_norm=[],
            step_norm=[],
            x_history=[x.copy()],
            momentum_norm=[],
            beta=[],
            beta_raw=[],
        )
        if lambdas_solver is not None:
            history["lambdas"] = []
        for n in range(max_iter):
            VVT_g = V @ (V.T @ g)
            direction = g - 2.0 * VVT_g
            G = -direction
            beta_raw = beta_n
            if self.use_bb and G_prev is not None:
                s = x - x_prev
                y = G - G_prev
                yy = np.dot(y, y)
                sy = np.dot(s, y)
                if yy > self.bb_eps and np.isfinite(yy) and np.isfinite(sy):
                    beta_raw = abs(sy / yy)
                    beta_min = self.bb_lower * self.beta0
                    beta_max = self.bb_upper * self.beta0
                    beta_n = np.clip(beta_raw, beta_min, beta_max)
                else:
                    beta_raw = beta_n
            momentum = x - x_prev
            x_new = x + beta_n * G + self.gamma * momentum
            g_new = grad_f(x_new)
            step_norm = np.linalg.norm(x_new - x)
            gnorm = np.linalg.norm(g_new)
            history["energy"].append(f(x))
            history["grad_norm"].append(gnorm)
            history["step_norm"].append(step_norm)
            history["x_history"].append(x_new.copy())
            history["momentum_norm"].append(np.linalg.norm(momentum))
            history["beta"].append(beta_n)
            history["beta_raw"].append(beta_raw)
            if lambdas_solver is not None:
                history["lambdas"].append(lambdas_solver(x, V))
            G_prev = G.copy()
            x_prev = x
            x = x_new
            g = g_new
            V = eigen_solver(x, V)
            if gnorm < tol:
                break
        return (x, history)
