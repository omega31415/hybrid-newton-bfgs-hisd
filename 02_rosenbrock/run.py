"""Run all six modified Rosenbrock methods with the paper parameters."""

import numpy as np
from threadpoolctl import threadpool_limits

from model import RosenbrockModified, make_eigen_solver
from hisd import HiSD
from hisd_bb import HiSDBB
from accelerated_hisd import AcceleratedHiSD
from hn_bfgs_hisd_ls import HNBFGSHiSDLS
from hn_lbfgs_hisd import HNLBFGSHiSD
from hn_lbfgs_hisd_ls import HNLBFGSHiSDLS


DIMENSION = 1000
SEED = 42
RHO = 0.1
MEMORY = 5
MAX_ITER = 500
STOP_TOL = 1e-8
BLAS_THREADS = 4


def main():
    s_vals = np.ones(DIMENSION)
    s_vals[:5] = -50000.0
    model = RosenbrockModified(DIMENSION, s_vals)
    x_star = model.x_star.copy()
    direction = np.random.RandomState(SEED).randn(DIMENSION)
    direction /= np.linalg.norm(direction)
    x0 = x_star + RHO * direction
    solvers = {
        "Full-memory LS": HNBFGSHiSDLS(beta1=1, beta2=1, tol_curv=1e-24),
        "HiSD": HiSD(beta=2e-5),
        "HiSD-BB": HiSDBB(beta=2e-5, gamma=0.0),
        "A-HiSD": AcceleratedHiSD(beta=2e-5, gamma=0.95),
        "Limited-memory": HNLBFGSHiSD(m=MEMORY, beta1=0.1, beta2=0.1, tol_curv=1e-18),
        "Limited-memory LS": HNLBFGSHiSDLS(m=MEMORY, beta1=1, beta2=1, tol_curv=1e-18),
    }
    with threadpool_limits(limits=BLAS_THREADS):
        index, condition_number, _ = model.compute_saddle_info()
        eigen_solver, lambdas_solver = make_eigen_solver(model, index)
        V0 = eigen_solver(x0, None)
        print(
            f"Modified Rosenbrock: d={DIMENSION}, index={index}, seed={SEED}, m={MEMORY}"
        )
        print(f"Absolute spectral condition number: {condition_number:.6e}", flush=True)
        for name, solver in solvers.items():
            print(f"Running {name} ...", flush=True)
            final, history = solver.run(
                model.energy,
                model.gradient,
                x0.copy(),
                V0.copy(),
                eigen_solver,
                lambdas_solver,
                max_iter=MAX_ITER,
                tol=STOP_TOL,
            )
            error = np.linalg.norm(final - x_star)
            gradient = np.linalg.norm(model.gradient(final))
            if not np.all(np.isfinite(final)) or not np.isfinite(gradient):
                raise RuntimeError(f"{name}: non-finite iterate or gradient.")
            status = "converged" if gradient <= STOP_TOL else "max_iter"
            print(
                f"  {len(history['x_history']) - 1} updates; "
                f"error={error:.4e}; gradient={gradient:.4e}; {status}",
                flush=True,
            )


if __name__ == "__main__":
    main()
