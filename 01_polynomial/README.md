# Nonlinear polynomial

The first experiment compares **Full-memory LS** and **Limited-memory LS**
near the known index-6 saddle at the origin.

## Run

From this directory:

```bash
python run.py
```

The script prints final numerical summaries. The experiment requires no
external input files.

## Parameters

| Parameter | Value |
| --- | --- |
| Dimension / saddle index | 60 / 6 |
| Quartic coefficient | 0.2 |
| Negative / positive curvature coefficients | Uniform grids from 1 to 101 |
| Coupling coefficients | Uniform on [0.5, 20.5], NumPy seed 42 |
| Initial point | `RandomState(42).uniform(-1, 1, 60) * 5e-3` |
| Initial trial step sizes | 1 / 1 |
| L-BFGS memory length | 5 |
| Curvature filter: full / limited | 1e-24 / 1e-18 |
| Gradient tolerance | 1e-11 |
| Maximum updates | 200 |

Both methods start from the same point and Hessian eigenspace. The model
computes its Hessian and derivatives analytically, and obtains the unstable
subspace by dense symmetric diagonalization.

## Files

- `model.py`: polynomial energy, derivatives and eigensolver.
- `run.py`: parameters, initial point and execution of both methods.
- `hn_bfgs_hisd_ls.py`: Full-memory LS.
- `hn_lbfgs_hisd_ls.py`: Limited-memory LS.

The solvers return `(final_point, history)`. Histories are held in memory
during a run; no precomputed histories, images or cache files are needed.
