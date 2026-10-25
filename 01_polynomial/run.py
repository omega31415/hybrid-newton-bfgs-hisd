"""Run the nonlinear polynomial experiment with the paper parameters."""

import numpy as np
from threadpoolctl import threadpool_limits

from model import DynamicEigenspaceSaddle
from hn_bfgs_hisd_ls import HNBFGSHiSDLS
from hn_lbfgs_hisd_ls import HNLBFGSHiSDLS


DIMENSION = 60
INDEX = 6
GAMMA = 0.2
SEED = 42
PERTURBATION_SCALE = 5e-3
MEMORY = 5
MAX_ITER = 200
STOP_TOL = 1e-11
BLAS_THREADS = 4


def main():
    model = DynamicEigenspaceSaddle(d=DIMENSION, k=INDEX, gamma=GAMMA)
    x0 = np.random.RandomState(SEED).uniform(-1, 1, DIMENSION) * PERTURBATION_SCALE
    solvers = {
        "Full-memory LS": HNBFGSHiSDLS(beta1=1.0, beta2=1.0, tol_curv=1e-24),
        "Limited-memory LS": HNLBFGSHiSDLS(
            m=MEMORY, beta1=1.0, beta2=1.0, tol_curv=1e-18
        ),
    }
    print(
        f"Nonlinear polynomial: d={DIMENSION}, index={INDEX}, seed={SEED}, m={MEMORY}"
    )
    print(
        f"{'Method':<24} {'Updates':>8} {'Final error':>14} {'Final gradient':>16} {'Status':>12}"
    )
    with threadpool_limits(limits=BLAS_THREADS):
        V0 = model.eigen_solver(x0)
        for name, solver in solvers.items():
            final, history = solver.run(
                f=model.f,
                grad_f=model.grad_f,
                x0=x0.copy(),
                V0=V0.copy(),
                eigen_solver=model.eigen_solver,
                lambdas_solver=model.lambdas_solver,
                max_iter=MAX_ITER,
                tol=STOP_TOL,
            )
            error = np.linalg.norm(final)
            gradient = np.linalg.norm(model.grad_f(final))
            if not np.all(np.isfinite(final)) or not np.isfinite(gradient):
                raise RuntimeError(f"{name}: non-finite iterate or gradient.")
            status = "converged" if gradient <= STOP_TOL else "max_iter"
            print(
                f"{name:<24} {len(history['x_history']) - 1:>8d} "
                f"{error:>14.4e} {gradient:>16.4e} {status:>12}",
                flush=True,
            )


if __name__ == "__main__":
    main()
