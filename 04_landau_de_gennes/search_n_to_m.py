"""Search N -> M (downward) with the limited-memory hybrid method."""

from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from model import LdGModel
from hn_lbfgs_hisd import HNLBFGSHiSD


LAMBDA = 50.0
MESH_SPACING = 1 / 50
TARGET_INDEX = 3
PERTURBATION_COLUMN = 3  # Zero-based: eigenvector 4.
PERTURBATION = 1e-3
MEMORY = 30
STOP_TOL = 1e-6
HVP_STEP = 1e-6
MIN_HVP_STEP = 1e-9
MAX_ITER = 10_000
BLAS_THREADS = 4


def main():
    model = LdGModel(lambda_val=LAMBDA, h=MESH_SPACING)
    input_path = Path(__file__).resolve().parent / "input_data" / "n_state.npz"
    with np.load(input_path, allow_pickle=False) as data:
        u = data["u"]
        V = data["V"]
    required_columns = max(TARGET_INDEX, PERTURBATION_COLUMN + 1)
    if u.shape != (model.state_size,) or V.ndim != 2 or V.shape[0] != model.state_size:
        raise ValueError("Input data do not match the model dimension.")
    if V.shape[1] < required_columns:
        raise ValueError("Input data contain too few initial eigenvectors.")
    x0 = u + PERTURBATION * V[:, PERTURBATION_COLUMN]
    V0 = V[:, :TARGET_INDEX]

    # Hybrid method (Limited-memory): active configuration.
    solver = HNLBFGSHiSD(
        m=MEMORY,
        beta1=1e-2,
        beta2=1e-2,
        tol=STOP_TOL,
        h_eps=HVP_STEP,
        minl=MIN_HVP_STEP,
        epsilon_eig=1e-6,
        tol_curv=1e-6,
    )
    method_name = "Hybrid method (Limited-memory)"
    max_iter = MAX_ITER

    # To switch methods, replace the active configuration with one block below.
    # HiSD-BB:
    # from hisd_bb import HiSDBB
    # solver = HiSDBB(
    #     beta=2e-4, gamma=0.0, tol=STOP_TOL,
    #     h_eps=HVP_STEP, minl=MIN_HVP_STEP,
    #     use_bb=True, bb_lower=0.2, bb_upper=5.0, bb_eps=1e-30,
    # )
    # method_name = "HiSD-BB"
    # max_iter = MAX_ITER

    # A-HiSD:
    # from accelerated_hisd import AcceleratedHiSD
    # solver = AcceleratedHiSD(
    #     beta=8e-5, gamma=0.99, tol=STOP_TOL,
    #     h_eps=HVP_STEP, minl=MIN_HVP_STEP,
    # )
    # method_name = "A-HiSD"
    # max_iter = MAX_ITER

    print(
        f"N -> M (downward); target index={TARGET_INDEX}; "
        f"degrees of freedom={model.state_size}; method={method_name}",
        flush=True,
    )
    with threadpool_limits(limits=BLAS_THREADS):
        final, final_vectors, history = solver.run(
            model=model, x0=x0, V0=V0, max_iter=max_iter
        )
        gradient_norm = float(model.norm(model.gradient(final)))
        energy = model.energy(final)
    if not np.all(np.isfinite(final)) or not np.isfinite(gradient_norm):
        raise RuntimeError("The search produced a non-finite point or gradient.")
    status = "converged" if gradient_norm <= STOP_TOL else "max_iter"
    print(
        f"{method_name}: {history['n_iter']} updates; "
        f"gradient={gradient_norm:.6e}; energy={energy:.10e}; "
        f"status={status}; wall={history['wall_time_total']:.3f}s",
        flush=True,
    )
    return final, final_vectors, history


if __name__ == "__main__":
    main()
