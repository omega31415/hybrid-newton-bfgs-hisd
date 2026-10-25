"""Run the six neural-network saddle-search methods with the paper settings."""

from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from model import FiveLayerNet
from embedding import build_reference
from _hisd_base import hessian_mat_product
from hisd import HiSD
from hisd_bb import HiSDBB
from accelerated_hisd import AcceleratedHiSD
from hn_bfgs_hisd_ls import HNBFGSHiSDLS
from hn_lbfgs_hisd_ls import HNLBFGSHiSDLS


INPUT_PATH = Path(__file__).resolve().parent / "input_data.npz"
SEED = 2026
RADIUS = 0.01
INDEX = 31
MEMORY = 10
STEP = 0.1
MAX_ITER = 2000
STOP_TOL = 1e-10
METHODS = (
    "Full-memory LS",
    "Limited-memory LS",
    "HiSD",
    "A-HiSD (gamma=0.9)",
    "A-HiSD (gamma=0.95)",
    "HiSD-BB",
)


def configure_runtime():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)


def prepare_experiment():
    with np.load(INPUT_PATH, allow_pickle=False) as data:
        x, y, theta_small = (data[key].copy() for key in ("x", "y", "theta_small"))
    if x.shape != (400, 2) or y.shape != (400,):
        raise ValueError("Unexpected dataset dimensions")
    center, widths, normal = build_reference(theta_small)
    perturbation = np.random.default_rng(SEED).normal(size=len(center))
    perturbation *= RADIUS / np.linalg.norm(perturbation)
    return dict(
        x=x,
        y=y,
        center=center,
        widths=widths,
        normal=normal,
        initial=center + perturbation,
    )


def run_method(name, experiment, max_iter=MAX_ITER):
    """Return (final_parameter_vector, summary) for one independent run."""
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}")
    x = torch.as_tensor(experiment["x"])
    y = torch.as_tensor(experiment["y"][:, None])
    initial = experiment["initial"]
    net = FiveLayerNet(experiment["widths"])
    torch.nn.utils.vector_to_parameters(torch.tensor(initial.copy()), net.parameters())
    eye = torch.eye(len(initial), dtype=x.dtype)
    initial_basis = eye[:, :INDEX].clone()

    def flat():
        return (
            torch.nn.utils.parameters_to_vector(net.parameters())
            .detach()
            .numpy()
            .copy()
        )

    def loss_tensor():
        return ((net(x) - y) ** 2).mean()

    def closure():
        net.zero_grad(set_to_none=True)
        loss = loss_tensor()
        loss.backward()
        return loss

    def gradient_and_loss():
        loss = closure()
        grad = torch.nn.utils.parameters_to_vector([p.grad for p in net.parameters()])
        return float(loss.detach()), grad.detach().numpy().copy()

    def unstable_basis():
        H = hessian_mat_product(loss_tensor(), net, eye)
        _, vectors = torch.linalg.eigh(0.5 * (H + H.T))
        return vectors[:, :INDEX].detach()

    quasi_newton = name in METHODS[:2]
    if quasi_newton:
        optimizer_class = HNBFGSHiSDLS if name == METHODS[0] else HNLBFGSHiSDLS
        kwargs = dict(
            lr1=STEP, lr2=STEP, v=initial_basis, epsilon_eig=1e-10, tol_curv=1e-12
        )
        if name == METHODS[1]:
            kwargs["m"] = MEMORY
        optimizer = optimizer_class(net.parameters(), **kwargs)
        optimizer.update_eigenpairs(loss_tensor(), net)
    else:
        if name == "HiSD":
            optimizer = HiSD(net.parameters(), lr=STEP, v=initial_basis)
        elif name == "HiSD-BB":
            optimizer = HiSDBB(net.parameters(), lr=STEP, v=initial_basis)
        else:
            gamma = 0.9 if name == METHODS[3] else 0.95
            optimizer = AcceleratedHiSD(
                net.parameters(), lr=STEP, gamma=gamma, v=initial_basis, v_lr=0.0
            )
        optimizer.param_groups[0]["v"] = unstable_basis()
        gradient_and_loss()

    status = "max_iter"
    for iteration in range(1, max_iter + 1):
        if quasi_newton:
            optimizer.step(closure)
            gradient = float(optimizer.grad_after_step.norm())
            loss = float(loss_tensor().detach())
        else:
            optimizer.step()
            loss, grad = gradient_and_loss()
            gradient = float(np.linalg.norm(grad))
        if not np.isfinite(loss) or not np.isfinite(gradient):
            raise FloatingPointError(f"{name}: non-finite loss or gradient")
        if gradient < STOP_TOL:
            status = "converged"
            break
        if quasi_newton:
            optimizer.update_eigenpairs(loss_tensor(), net)
            optimizer.update_bfgs()
        else:
            optimizer.param_groups[0]["v"] = unstable_basis()

    final = flat()
    distance = np.linalg.norm(experiment["normal"].T @ (final - experiment["center"]))
    summary = dict(
        method=name,
        iterations=iteration,
        loss=loss,
        gradient=gradient,
        distance=float(distance),
        status=status,
    )
    return final, summary


def main():
    configure_runtime()
    with threadpool_limits(limits=1):
        experiment = prepare_experiment()
        print(
            f"Neural network: N=400, parameters=100, index={INDEX}, "
            f"manifold dimension=13, seed={SEED}, m={MEMORY}",
            flush=True,
        )
        for name in METHODS:
            print(f"Running {name} ...", flush=True)
            _, result = run_method(name, experiment)
            print(
                f"  {result['iterations']} updates; loss={result['loss']:.6e}; "
                f"gradient={result['gradient']:.6e}; "
                f"distance={result['distance']:.6e}; {result['status']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
