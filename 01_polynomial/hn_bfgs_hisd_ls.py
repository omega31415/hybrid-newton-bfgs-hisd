"""Full-memory LS: hybrid Newton-BFGS HiSD with subspace line searches."""

import numpy as np


class HNBFGSHiSDLS:
    """Full-memory LS: hybrid Newton-BFGS HiSD with subspace line searches."""

    def __init__(
        self,
        beta1=1.0,
        beta2=1.0,
        tol=1e-08,
        H_in=None,
        epsilon_eig=1e-10,
        tol_curv=None,
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.tol = tol
        self.tol_curv = tol if tol_curv is None else tol_curv
        self.epsilon_eig = epsilon_eig
        if not np.isfinite(self.epsilon_eig) or self.epsilon_eig <= 0:
            raise ValueError("epsilon_eig must be finite and positive")
        if not np.isfinite(self.tol_curv) or self.tol_curv <= 0:
            raise ValueError("tol_curv must be finite and positive")
        self.H_in = H_in

    def _project_perp(self, v, V):
        """Project onto the orthogonal complement of the columns of V."""
        return v - V @ (V.T @ v)

    def _bfgs_update(self, H, s, y):
        """Apply the curvature-filtered inverse BFGS update."""
        curvature = float(s @ y)
        s_sq = float(s @ s)
        if curvature < 0:
            return (H, "negative", curvature)
        if not np.isfinite(curvature) or curvature <= self.tol_curv * s_sq:
            return (H, "small_positive", curvature)
        rho = 1.0 / curvature
        n = len(s)
        I = np.eye(n)
        A = I - rho * np.outer(s, y)
        H_new = A @ H @ A.T + rho * np.outer(s, s)
        return (H_new, "updated", curvature)

    def run(
        self, f, grad_f, x0, V0, eigen_solver, lambdas_solver, max_iter=2000, tol=1e-10
    ):
        """Return (final_point, history). The gradient tolerance is set by tol here.

        x_history includes the initial point; grad_norm records updated points.
        eigen_solver(x, V) supplies the unstable subspace, and
        lambdas_solver(x, V) supplies its corresponding curvatures."""
        d = len(x0)
        x = x0.copy()
        V = V0.copy()
        if self.H_in is not None:
            H = self.H_in
        else:
            H = np.eye(d)
        history = dict(
            energy=[],
            grad_norm=[],
            step_norm=[],
            x_history=[x.copy()],
            bfgs_updates=[],
            curvature=[],
            phase=[],
            beta1=[],
            beta2=[],
        )
        g = grad_f(x)
        for n in range(max_iter):
            lams = lambdas_solver(x, V)
            neg_step = np.zeros_like(x)
            for i, lam in enumerate(lams):
                alpha = max(abs(lam), self.epsilon_eig)
                neg_step += 1.0 / alpha * (V[:, i] @ g) * V[:, i]
            g_perp = self._project_perp(g, V)
            bfgs_step = self._project_perp(H @ g_perp, V)
            beta1_eff = self.beta1
            beta2_eff = self.beta2
            c = 0.0001
            rho = 0.5
            g_v = g - g_perp
            norm_g_perp = np.linalg.norm(g_perp)
            norm_g_v = np.linalg.norm(g_v)
            for ls_iter in range(12):
                x_trial = x + beta1_eff * neg_step - beta2_eff * bfgs_step
                g_trial = grad_f(x_trial)
                g_trial_perp = self._project_perp(g_trial, V)
                g_trial_v = g_trial - g_trial_perp
                norm_g_trial_perp = np.linalg.norm(g_trial_perp)
                norm_g_trial_v = np.linalg.norm(g_trial_v)
                v_ok = (
                    norm_g_v < 1e-12
                    or norm_g_trial_v <= (1.0 - c * beta1_eff) * norm_g_v
                )
                perp_ok = (
                    norm_g_perp < 1e-12
                    or norm_g_trial_perp <= (1.0 - c * beta2_eff) * norm_g_perp
                )
                if v_ok and perp_ok:
                    break
                if not v_ok:
                    beta1_eff *= rho
                if not perp_ok:
                    beta2_eff *= rho
            else:
                x_trial = x + beta1_eff * neg_step - beta2_eff * bfgs_step
                g_trial = grad_f(x_trial)
            x_new = x_trial
            g_new = g_trial
            E_new = f(x_new)
            history["beta1"].append(beta1_eff)
            history["beta2"].append(beta2_eff)
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
            s = self._project_perp(x_new - x, V_new)
            y = self._project_perp(g_new - g, V_new)
            H, status, curv = self._bfgs_update(H, s, y)
            step_norm = np.linalg.norm(x_new - x)
            gnorm = np.linalg.norm(g_new)
            history["energy"].append(E_new)
            history["grad_norm"].append(gnorm)
            history["step_norm"].append(step_norm)
            history["x_history"].append(x_new.copy())
            history["bfgs_updates"].append(status)
            history["curvature"].append(curv)
            x = x_new
            V = V_new
            g = g_new
        return (x, history)
