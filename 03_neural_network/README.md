# Neural-network saddle manifold

This is the third experiment in the paper. It compares Full-memory LS,
Limited-memory LS, HiSD, HiSD-BB, and A-HiSD with momentum 0.9 and 0.95
near a fixed degenerate saddle manifold.

## Installation and run

Tested with Python 3.11.14, NumPy 2.3.5, SciPy 1.17.1, PyTorch 2.10.0 and
threadpoolctl 3.6.0 on Windows. Computations use CPU float64 and one thread;
a GPU is not required. From this directory:

```bash
python -m pip install -r requirements.txt
python run.py
```

The dependency versions for this PyTorch experiment are recorded separately
from those of the first two NumPy examples.

The script runs all six methods and prints the number of updates, final
loss, gradient norm, distance to the saddle manifold and termination status.
A complete 2000-step rerun of each method was checked in the tested environment:
the final parameter vectors, gradient norms and manifold distances matched
the original experiment records exactly.
A `max_iter` status means that the 2000-step budget was exhausted before
the gradient tolerance was met. Repeated dense Hessian calculations are
performed for all six methods, so allow time for the complete comparison.

## Fixed input and embedding

`input_data.npz` contains only three float64 input arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `x` | (400, 2) | Fixed training inputs, sampled uniformly from [-2, 2]^2 |
| `y` | (400,) | Fixed target values used in the paper |
| `theta_small` | (30,) | Optimized small-network parameters |

The target values were generated from a fixed teacher network plus eight
sine modes with amplitude 0.15 after RMS normalization. The supplied
small-network checkpoint is a numerically stationary local minimum with
loss approximately 0.00371494. Using this fixed checkpoint reproduces the
selected saddle-search problem without retraining the small network.

The small architecture is (2, 2, 2, 2, 3, 1), with 30 parameters.
`embedding.py` splits the first hidden-layer neurons into 5 and 5 copies,
and the third hidden-layer neurons into 4 and 3 copies. Equal outgoing
weight allocation gives the reference point in the large architecture
(2, 10, 2, 7, 3, 1), with 100 parameters. Both networks have four tanh hidden
layers with biases and a bias-free linear output. Parameters are ordered
by layer: row-major weight matrix, then bias when present.

The splitting coefficients generate a 13-dimensional affine family.
Its tangent and orthonormal normal basis are recomputed from the small
checkpoint. Distance is evaluated by orthogonal projection onto this
normal space. The reference Hessian has numerical inertia (31, 13, 56)
using a zero threshold of 1e-10. No saved Hessians or eigenvectors are used.

## Search parameters

| Parameter | Value |
| --- | --- |
| Initial perturbation | `default_rng(2026).normal(size=100)`, normalized to norm 0.01 |
| Unstable subspace dimension | 31 |
| HiSD / A-HiSD fixed step size | 0.1 |
| A-HiSD momentum | 0.9 and 0.95 |
| HiSD-BB initial step size | 0.1 |
| HiSD-BB bounds / denominator threshold | [0.02, 0.5] / 1e-14 |
| Hybrid initial trial step sizes | 0.1 / 0.1 |
| L-BFGS memory length | 10 |
| Eigenvalue safeguard / curvature filter | 1e-10 / 1e-12 |
| Backtracking budget: full / limited | 12 / 10 |
| Backtracking decrease / contraction constants | 1e-4 / 0.5 |
| Small projected-gradient threshold | 1e-12 |
| Gradient termination tolerance | 1e-10 |
| Maximum updates per method | 2000 |

Every method refreshes the unstable subspace by the same dense symmetric
PyTorch Hessian eigendecomposition. The hybrid methods project secant
differences using the eigenspace at the new point. Bounded backtracking
uses the final reduced trial when its search budget is exhausted, as in
the original implementation. The first-order HiSD-BB vector update uses
NumPy, preserving the arithmetic of the paper experiment.

## Files

- `model.py`: network definition and parameter layout.
- `embedding.py`: neuron splitting, tangent directions and normal basis.
- `run.py`: fixed parameters, initial perturbation and six-method execution.
- `hisd.py`: HiSD.
- `hisd_bb.py`: HiSD-BB.
- `accelerated_hisd.py`: both A-HiSD variants.
- `hn_bfgs_hisd_ls.py`: Full-memory LS.
- `hn_lbfgs_hisd_ls.py`: Limited-memory LS (m=10).
- `_hisd_base.py`: shared gradient, dense eigenpair and line-search operations.

Each hybrid iteration calls `step(closure)`, checks the new gradient, then
refreshes eigenpairs and calls `update_bfgs()` when continuing. The closure
evaluates the full-batch loss and calls `backward()`.

Floating-point differences between library versions can affect terminal
values. This example studies local saddle search from the fixed perturbation.
