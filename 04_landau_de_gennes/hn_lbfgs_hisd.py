"""Hybrid method: limited-memory BFGS with fixed steps and one LOBPCG update."""

import numpy as np
import time
from collections import deque
from typing import Tuple, Dict


class HNLBFGSHiSD:
    """Hybrid method: limited-memory BFGS with fixed steps and one LOBPCG update."""

    def __init__(
        self,
        m=10,
        beta1=1.0,
        beta2=1.0,
        tol=1e-06,
        h_eps=1e-06,
        minl=1e-10,
        epsilon_eig=1e-06,
        tol_curv=None,
        final_eigen_diagnostics=True,
    ):
        self.m = m
        self.beta1 = beta1
        self.beta2 = beta2
        self.tol = tol
        self.h_eps = h_eps
        self.minl = minl
        self.epsilon_eig = epsilon_eig
        self.tol_curv = tol if tol_curv is None else tol_curv
        self.final_eigen_diagnostics = final_eigen_diagnostics
        if not np.isfinite(epsilon_eig) or epsilon_eig <= 0:
            raise ValueError("epsilon_eig must be finite and positive")
        if not np.isfinite(self.tol_curv) or self.tol_curv <= 0:
            raise ValueError("tol_curv must be finite and positive")
        self.s_history = deque(maxlen=m)
        self.y_history = deque(maxlen=m)
        self.rho_history = deque(maxlen=m)

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

    def _two_loop_recursion(self, model, g: np.ndarray, gamma: float):
        q = g.copy()
        alphas = []
        n_mem = len(self.s_history)
        s_list = list(self.s_history)
        y_list = list(self.y_history)
        rho_list = list(self.rho_history)
        for i in range(n_mem - 1, -1, -1):
            alpha = rho_list[i] * self._innp(model, s_list[i], q)
            q -= alpha * y_list[i]
            alphas.append(alpha)
        r = gamma * q
        alphas.reverse()
        for i in range(n_mem):
            beta = rho_list[i] * self._innp(model, y_list[i], r)
            r += s_list[i] * (alphas[i] - beta)
        return r

    def run(
        self, model, x0: np.ndarray, V0: np.ndarray, max_iter=10000
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        n = len(x0)
        k = V0.shape[1]
        x = x0.copy()
        V = V0.copy()
        if k > 0:
            for i in range(k):
                if i > 0:
                    V[:, i] -= V[:, :i] @ self._innp(model, V[:, :i], V[:, i])
                V[:, i] /= np.sqrt(self._innp(model, V[:, i], V[:, i]))
        self.s_history.clear()
        self.y_history.clear()
        self.rho_history.clear()
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
        for key in ("step_norm", "phase", "bfgs_updates", "beta1", "beta2"):
            history.setdefault(key, [])
        g = model.gradient(x)
        gamma = 1.0
        if k > 0:
            for i in range(k):
                U[:, i] = model.hvp(x, V[:, i], eps=l_h)
                alpha_eig[i] = self._innp(model, V[:, i], U[:, i])
        print(f"=== [Hybrid method / Limited-memory + LOBPCG] tolerance {self.tol} ===")
        wall_t0 = time.perf_counter()
        cpu_t0 = time.process_time()
        for iter_n in range(max_iter):
            iter_wall_t0 = time.perf_counter()
            iter_cpu_t0 = time.process_time()
            gnorm = model.norm(g)
            if gnorm <= self.tol:
                break
            raw_step_v = np.zeros_like(x)
            if k > 0:
                for i in range(k):
                    lam = alpha_eig[i]
                    alpha = max(abs(lam), self.epsilon_eig)
                    raw_step_v += 1.0 / alpha * self._innp(model, V[:, i], g) * V[:, i]
            g_perp = self._project_perp(model, g, V)
            if len(self.s_history) > 0:
                raw_step_vperp = self._two_loop_recursion(model, g_perp, gamma)
            else:
                raw_step_vperp = g_perp
            raw_step_vperp = self._project_perp(model, raw_step_vperp, V)
            step_v_eff = raw_step_v * self.beta1
            step_vperp_eff = raw_step_vperp * self.beta2
            x_new = x + step_v_eff - step_vperp_eff
            g_new = model.gradient(x_new)
            b1_log, b2_log = (self.beta1, self.beta2)
            step_norm = model.norm(x_new - x)
            new_gnorm = model.norm(g_new)
            if new_gnorm <= self.tol:
                history["grad_norm"].append(float(new_gnorm))
                history["step_norm"].append(float(step_norm))
                history["curvature"].append(np.full(k, np.nan))
                history["bfgs_updates"].append("stopped")
                history["phase"].append("fixed")
                history["beta1"].append(float(b1_log))
                history["beta2"].append(float(b2_log))
                history["wall_time_iter"].append(
                    float(time.perf_counter() - iter_wall_t0)
                )
                history["cpu_time_iter"].append(
                    float(time.process_time() - iter_cpu_t0)
                )
                history["wall_time_cumulative"].append(
                    float(time.perf_counter() - wall_t0)
                )
                history["cpu_time_cumulative"].append(
                    float(time.process_time() - cpu_t0)
                )
                x, g = (x_new, g_new)
                break
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
                UU, YY = (
                    np.concatenate((V, W, Vp), axis=1)[:, emp],
                    np.concatenate((U, Y, Up), axis=1)[:, emp],
                )
                Pn = self._innp(model, UU, YY)
                Pn = (Pn + Pn.T) / 2.0
                eigvals, eigvecs = np.linalg.eigh(Pn)
                idx = np.argsort(eigvals)
                eigvals, eigvecs = (eigvals[idx], eigvecs[:, idx])
                Vp = V.copy()
                V_new = UU @ eigvecs[:, :k]
                alpha_eig = eigvals[:k]
            l_h = max(l_h / 1.05, self.minl)
            curv = 0.0
            status = "skip"
            s_vec = self._project_perp(model, x_new - x, V_new)
            y_vec = self._project_perp(model, g_new - g, V_new)
            curv = float(self._innp(model, s_vec, y_vec))
            s_sq = float(self._innp(model, s_vec, s_vec))
            if np.isfinite(curv) and curv > self.tol_curv * s_sq:
                self.s_history.append(s_vec)
                self.y_history.append(y_vec)
                self.rho_history.append(1.0 / curv)
                y_sq = float(self._innp(model, y_vec, y_vec))
                gamma = curv / y_sq if y_sq > 0 else 1.0
                status = "add"
            step_norm = model.norm(x_new - x)
            x, V, g = (x_new, V_new, g_new)
            gnorm = model.norm(g)
            history["grad_norm"].append(gnorm)
            history["curvature"].append(alpha_eig.copy())
            history["step_norm"].append(float(step_norm))
            history["phase"].append("fixed")
            history["bfgs_updates"].append(status)
            history["beta1"].append(float(b1_log))
            history["beta2"].append(float(b2_log))
            iter_wall_dt = time.perf_counter() - iter_wall_t0
            iter_cpu_dt = time.process_time() - iter_cpu_t0
            history["wall_time_iter"].append(float(iter_wall_dt))
            history["cpu_time_iter"].append(float(iter_cpu_dt))
            history["wall_time_cumulative"].append(float(time.perf_counter() - wall_t0))
            history["cpu_time_cumulative"].append(float(time.process_time() - cpu_t0))
            if iter_n % 10 == 0:
                eig_str = np.array2string(alpha_eig, precision=3) if k > 0 else "[]"
                print(
                    f"n={iter_n:>4} | ||g||={gnorm:.3e} | L-BFGS={status:<4} | b1/b2={b1_log:.4f},{b2_log:.4f} | Ritz={eig_str}|"
                )
        final_gnorm = float(model.norm(g))
        history["converged"] = final_gnorm <= self.tol
        history["n_iter"] = len(history["grad_norm"])
        history["final_grad_norm"] = final_gnorm
        history["final_eigen_diagnostics_performed"] = False
        if history["converged"] and self.final_eigen_diagnostics and (k > 0):
            HV = np.column_stack([model.hvp(x, V[:, i], eps=l_h) for i in range(k)])
            K = self._innp(model, V, HV)
            alpha_eig, rotation = np.linalg.eigh((K + K.T) / 2.0)
            V = V @ rotation
            history["final_eigen_diagnostics_performed"] = True
            if history["curvature"]:
                history["curvature"][-1] = alpha_eig.copy()
        history["final_eigvals"] = alpha_eig.copy()
        history["final_eigvals_at_final_point"] = (
            k == 0
            or not history["converged"]
            or history["n_iter"] == 0
            or history["final_eigen_diagnostics_performed"]
        )
        history["wall_time_total"] = float(time.perf_counter() - wall_t0)
        history["cpu_time_total"] = float(time.process_time() - cpu_t0)
        if history["converged"]:
            print(
                f"Converged after {history['n_iter']} updates | gradient = {history['final_grad_norm']:.4e} | wall = {history['wall_time_total']:.3f}s | cpu = {history['cpu_time_total']:.3f}s"
            )
        return (x, V, history)
