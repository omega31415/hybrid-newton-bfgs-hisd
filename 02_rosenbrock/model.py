"""Modified Rosenbrock model used in the paper."""

import numpy as np
from scipy.linalg import eigh


class RosenbrockModified:
    """Rosenbrock energy plus sum(s * arctan(x - x_star)**2)."""

    def __init__(self, d, s_vals, x_star=None):
        self.d = d
        self.s = np.array(s_vals, dtype=float)
        self.x_star = np.ones(d) if x_star is None else np.array(x_star)

    def energy(self, x):
        x = np.asarray(x, dtype=float)
        R = np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
        A = np.sum(self.s * np.arctan(x - self.x_star) ** 2)
        return R + A

    def gradient(self, x):
        x = np.asarray(x, dtype=float)
        d = self.d
        g = np.zeros(d)
        for i in range(d - 1):
            g[i] += -2 * (1 - x[i])
            g[i] += -400 * x[i] * (x[i + 1] - x[i] ** 2)
        for i in range(1, d):
            g[i] += 200 * (x[i] - x[i - 1] ** 2)
        g += 2 * self.s * np.arctan(x - self.x_star) / (1 + (x - self.x_star) ** 2)
        return g

    def hessian_clean(self, x):
        """Evaluate the analytic Hessian."""
        x = np.asarray(x, dtype=float)
        d = self.d
        H = np.zeros((d, d))
        for i in range(d - 1):
            H[i, i] += 2 + 1200 * x[i] ** 2 - 400 * x[i + 1]
            H[i, i + 1] += -400 * x[i]
            H[i + 1, i] += -400 * x[i]
        for i in range(1, d):
            H[i, i] += 200
        u = x - self.x_star
        for i in range(d):
            H[i, i] += (
                2
                * self.s[i]
                * (
                    1.0 / (1 + u[i] ** 2) ** 2
                    - 2 * u[i] * np.arctan(u[i]) / (1 + u[i] ** 2) ** 2
                )
            )
        return H

    def compute_saddle_info(self):
        """Return saddle index, absolute spectral condition number and eigenvalues."""
        H = self.hessian_clean(self.x_star)
        eigs = np.linalg.eigvalsh(H)
        k = int(np.sum(eigs < 0))
        nonzero = eigs[np.abs(eigs) > 1e-14]
        kappa = np.max(np.abs(nonzero)) / np.min(np.abs(nonzero))
        return (k, kappa, eigs)


def make_eigen_solver(func, k):
    """Return the dense lowest-k eigenspace and eigenvalue solvers."""

    def eigen_solver(x, V_prev):
        H = func.hessian_clean(x)
        eigs, vecs = eigh(H, subset_by_index=[0, k - 1])
        return vecs

    def lambdas_solver(x, V):
        H = func.hessian_clean(x)
        eigs, _ = eigh(H, subset_by_index=[0, k - 1])
        return eigs

    return (eigen_solver, lambdas_solver)
