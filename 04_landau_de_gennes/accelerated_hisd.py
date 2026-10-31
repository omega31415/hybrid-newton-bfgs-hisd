"""A-HiSD with the discrete L2 metric and one LOBPCG update."""

import numpy as np
import time
from typing import Tuple, Dict


class AcceleratedHiSD:
    """A-HiSD with the discrete L2 metric and one LOBPCG update."""

    def __init__(self, beta=0.01, gamma=0.99, tol=1e-06, h_eps=1e-06, minl=1e-10):
        self.beta = beta
        self.gamma = gamma
        self.tol = tol
        self.h_eps = h_eps
        self.minl = minl

    def _innp(self, model, a: np.ndarray, b: np.ndarray):
        if a.ndim == 1 and b.ndim == 1:
            return np.dot(a, model.M_weights * b)
        elif a.ndim == 2 and b.ndim == 1:
            return a.T @ (model.M_weights * b)
        elif a.ndim == 2 and b.ndim == 2:
            return a.T @ (model.M_weights[:, None] * b)
        raise ValueError("Unsupported inner-product dimensions.")

    def _project_perp(self, model, v: np.ndarray, V: np.ndarray):
        if V.shape[1] == 0:
            return v.copy()
        return v - V @ self._innp(model, V, v)

    def _project(self, model, v: np.ndarray, V: np.ndarray):
        if V.shape[1] == 0:
            return np.zeros_like(v)
        return V @ self._innp(model, V, v)

    def run(
        self, model, x0: np.ndarray, V0: np.ndarray, max_iter=10000
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        n = len(x0)
        k = V0.shape[1]
        x = x0.copy()
        x_prev = x0.copy()
        V = V0.copy()
        if k > 0:
            for i in range(k):
                if i > 0:
                    V[:, i] -= V[:, :i] @ self._innp(model, V[:, :i], V[:, i])
                V[:, i] /= np.sqrt(self._innp(model, V[:, i], V[:, i]))
        alpha_eig = np.zeros(k)
        U, W, Y = (np.zeros((n, k)), np.zeros((n, k)), np.zeros((n, k)))
        Vp, Up = (np.zeros((n, k)), np.zeros((n, k)))
        l_h = self.h_eps
        history = {
            "grad_norm": [],
            "curvature": [],
            "final_eigvals": None,
            "converged": False,
            "n_iter": 0,
            "final_grad_norm": None,
            "wall_time_total": None,
            "cpu_time_total": None,
            "wall_time_iter": [],
            "cpu_time_iter": [],
            "wall_time_cumulative": [],
            "cpu_time_cumulative": [],
        }
        g = model.gradient(x)
        if k > 0:
            for i in range(k):
                U[:, i] = model.hvp(x, V[:, i], eps=l_h)
                alpha_eig[i] = self._innp(model, V[:, i], U[:, i])
        print(
            f"=== [A-HiSD + LOBPCG] beta={self.beta:.4e}, gamma={self.gamma:.4e}, tol={self.tol} ==="
        )
        wall_t0 = time.perf_counter()
        cpu_t0 = time.process_time()
        for iter_n in range(max_iter):
            iter_wall_t0 = time.perf_counter()
            iter_cpu_t0 = time.process_time()
            gnorm = model.norm(g)
            if gnorm < self.tol:
                wall_now = time.perf_counter()
                cpu_now = time.process_time()
                history["converged"] = True
                history["n_iter"] = int(iter_n)
                history["final_grad_norm"] = float(gnorm)
                history["wall_time_total"] = float(wall_now - wall_t0)
                history["cpu_time_total"] = float(cpu_now - cpu_t0)
                print(
                    f"Converged after {iter_n} updates | gradient = {gnorm:.4e} | wall = {history['wall_time_total']:.3f}s | cpu = {history['cpu_time_total']:.3f}s"
                )
                break
            g_v = self._project(model, g, V)
            reflected_g = g - 2.0 * g_v
            momentum = self.gamma * (x - x_prev)
            step = -self.beta * reflected_g + momentum
            x_new = x + step
            g_new = model.gradient(x_new)
            V_new = V.copy()
            if k > 0:
                emp = np.ones(3 * k, dtype=bool)
                for i in range(k):
                    U[:, i] = model.hvp(x_new, V[:, i], eps=l_h)
                    alpha_eig[i] = self._innp(model, V[:, i], U[:, i])
                    W[:, i] = U[:, i] - alpha_eig[i] * V[:, i]
                    W[:, i] -= V @ self._innp(model, V, W[:, i])
                    if i > 0:
                        W[:, i] -= W[:, :i] @ self._innp(model, W[:, :i], W[:, i])
                    nrmW = np.sqrt(max(0, self._innp(model, W[:, i], W[:, i])))
                    if nrmW > 1e-08:
                        W[:, i] /= nrmW
                        Y[:, i] = model.hvp(x_new, W[:, i], eps=l_h)
                    else:
                        W[:, i], Y[:, i], emp[k + i] = (0, 0, False)
                for i in range(k):
                    valid_bases = [V, W] if i == 0 else [V, W, Vp[:, :i]]
                    VWprev = np.concatenate(valid_bases, axis=1)
                    Vp[:, i] -= VWprev @ self._innp(model, VWprev, Vp[:, i])
                    nrmVpi = np.sqrt(max(0, self._innp(model, Vp[:, i], Vp[:, i])))
                    if nrmVpi > 1e-08:
                        Vp[:, i] /= nrmVpi
                        Vp[:, i] -= VWprev @ self._innp(model, VWprev, Vp[:, i])
                        nrmVpi = np.sqrt(max(0, self._innp(model, Vp[:, i], Vp[:, i])))
                        if nrmVpi > 1e-08:
                            Vp[:, i] /= nrmVpi
                            Up[:, i] = model.hvp(x_new, Vp[:, i], eps=l_h)
                        else:
                            Vp[:, i], Up[:, i], emp[2 * k + i] = (0, 0, False)
                    else:
                        Vp[:, i], Up[:, i], emp[2 * k + i] = (0, 0, False)
                UU = np.concatenate((V, W, Vp), axis=1)[:, emp]
                YY = np.concatenate((U, Y, Up), axis=1)[:, emp]
                Pn = self._innp(model, UU, YY)
                Pn = (Pn + Pn.T) / 2.0
                eigvals, eigvecs = np.linalg.eigh(Pn)
                idx = np.argsort(eigvals)
                eigvals, eigvecs = (eigvals[idx], eigvecs[:, idx])
                Vp = V.copy()
                V_new = UU @ eigvecs[:, :k]
                alpha_eig = eigvals[:k]
            l_h = max(l_h / 1.05, self.minl)
            history["grad_norm"].append(gnorm)
            history["curvature"].append(alpha_eig.copy())
            iter_wall_dt = time.perf_counter() - iter_wall_t0
            iter_cpu_dt = time.process_time() - iter_cpu_t0
            history["wall_time_iter"].append(float(iter_wall_dt))
            history["cpu_time_iter"].append(float(iter_cpu_dt))
            history["wall_time_cumulative"].append(float(time.perf_counter() - wall_t0))
            history["cpu_time_cumulative"].append(float(time.process_time() - cpu_t0))
            if iter_n % 10 == 0:
                eig_str = np.array2string(alpha_eig, precision=3) if k > 0 else "[]"
                print(f"n={iter_n:>4} | ||g||={gnorm:.3e} | Ritz={eig_str} | ")
            x_prev = x
            x = x_new
            V = V_new
            g = g_new
        if not history["converged"]:
            final_gnorm = model.norm(g)
            history["n_iter"] = max_iter
            history["final_grad_norm"] = float(final_gnorm)
        if history["wall_time_total"] is None:
            history["wall_time_total"] = float(time.perf_counter() - wall_t0)
        if history["cpu_time_total"] is None:
            history["cpu_time_total"] = float(time.process_time() - cpu_t0)
        history["final_eigvals"] = alpha_eig.copy()
        return (x, V, history)
