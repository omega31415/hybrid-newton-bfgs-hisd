import numpy as np


class DynamicEigenspaceSaddle:
    """Nonlinear polynomial energy with a moving unstable subspace."""

    def __init__(self, d=10, k=2, gamma=0.1):
        self.d = d
        self.k = k
        self.gamma = gamma
        self.a = np.linspace(1.0, 101.0, k)
        self.b = np.linspace(1.0, 101.0, d - k)
        np.random.seed(42)
        self.alpha = np.random.uniform(0.5, 20.5, size=(d - k, k))

    def _split(self, x_full):
        return (x_full[: self.k], x_full[self.k :])

    def f(self, x_full):
        """Evaluate the energy."""
        x, y = self._split(x_full)
        Z = y - self.alpha @ x**2
        term_x = np.sum(-0.5 * self.a * x**2 + self.gamma / 4.0 * x**4)
        term_y = 0.5 * np.sum(self.b * Z**2)
        return term_x + term_y

    def grad_f(self, x_full):
        """Evaluate its analytic gradient."""
        x, y = self._split(x_full)
        Z = y - self.alpha @ x**2
        g = np.zeros_like(x_full)
        g_y = self.b * Z
        coupling_force = self.alpha.T @ g_y
        g_x = -self.a * x + self.gamma * x**3 - 2 * x * coupling_force
        g[: self.k] = g_x
        g[self.k :] = g_y
        return g

    def hessian(self, x_full):
        """Evaluate its analytic Hessian."""
        x, y = self._split(x_full)
        Z = y - self.alpha @ x**2
        H = np.zeros((self.d, self.d))
        H_yy = np.diag(self.b)
        H_yx = -2 * self.b[:, None] * self.alpha * x[None, :]
        H_xy = H_yx.T
        diag_xx = -self.a + 3 * self.gamma * x**2 - 2 * (self.alpha.T @ (self.b * Z))
        B_mat = np.diag(self.b)
        X_mat = np.diag(x)
        outer_coupling = 4 * X_mat @ self.alpha.T @ B_mat @ self.alpha @ X_mat
        H_xx = np.diag(diag_xx) + outer_coupling
        H[: self.k, : self.k] = H_xx
        H[: self.k, self.k :] = H_xy
        H[self.k :, : self.k] = H_yx
        H[self.k :, self.k :] = H_yy
        return H

    def eigen_solver(self, x_full, V_old=None):
        """Compute the k lowest Hessian eigenvectors by dense diagonalization."""
        from scipy.linalg import eigh

        H = self.hessian(x_full)
        evals, evecs = eigh(H)
        return evecs[:, : self.k]

    def lambdas_solver(self, x_full, V):
        """Return the Rayleigh curvatures associated with V."""
        H = self.hessian(x_full)
        sub_matrix = V.T @ H @ V
        return np.diag(sub_matrix)
