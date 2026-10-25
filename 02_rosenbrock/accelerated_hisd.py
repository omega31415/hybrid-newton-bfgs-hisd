"""A-HiSD: momentum-accelerated high-index saddle dynamics."""

import numpy as np


class AcceleratedHiSD:
    """A-HiSD: momentum-accelerated high-index saddle dynamics."""

    def __init__(self, beta=0.001, gamma=0.5):
        self.beta = beta
        self.gamma = gamma

    def run(
        self, f, grad_f, x0, V0, eigen_solver, lambdas_solver, max_iter=2000, tol=1e-10
    ):
        """Return (final_point, history). The gradient tolerance is set by tol here.

        x_history includes the initial point; grad_norm records updated points.
        eigen_solver(x, V) supplies the unstable subspace, and
        lambdas_solver(x, V) supplies its corresponding curvatures."""
        x = x0.copy()
        x_prev = x0.copy()
        V = V0.copy()
        history = dict(
            energy=[],
            grad_norm=[],
            step_norm=[],
            x_history=[x.copy()],
            momentum_norm=[],
        )
        g = grad_f(x)
        for n in range(max_iter):
            VVT_g = V @ (V.T @ g)
            direction = g - 2 * VVT_g
            momentum = x - x_prev
            x_new = x - self.beta * direction + self.gamma * momentum
            g_new = grad_f(x_new)
            step_norm = np.linalg.norm(x_new - x)
            gnorm = np.linalg.norm(g_new)
            history["energy"].append(f(x))
            history["grad_norm"].append(gnorm)
            history["step_norm"].append(step_norm)
            history["x_history"].append(x_new.copy())
            history["momentum_norm"].append(np.linalg.norm(momentum))
            x_prev = x
            x = x_new
            g = g_new
            V = eigen_solver(x, V)
            if np.linalg.norm(g) < tol:
                break
        return (x, history)
