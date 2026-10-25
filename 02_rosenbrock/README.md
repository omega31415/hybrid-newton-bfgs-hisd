# Modified Rosenbrock

The second experiment compares six saddle-search methods on the modified
Rosenbrock energy. The known saddle is the all-ones vector.

## Run

From this directory:

```bash
python run.py
```

The script runs all six methods and prints a numerical summary for each.
It uses dense 1000-by-1000 Hessian eigenproblems repeatedly, so allow time
for the full experiment. No external input files are required.

## Parameters

| Parameter | Value |
| --- | --- |
| Dimension / saddle index | 1000 / 5 |
| Modification coefficients | First 5: -50000; remaining: 1 |
| Initial perturbation | Normal vector from `RandomState(42)`, normalized to norm 0.1 |
| HiSD step size | 2e-5 |
| HiSD-BB initial step size | 2e-5 |
| HiSD-BB step-size bounds | [0.2, 5] times the initial step size |
| A-HiSD step size / momentum | 2e-5 / 0.95 |
| Full-memory / limited-memory LS initial trial steps | 1 / 1 |
| Limited-memory fixed steps | 0.1 / 0.1 |
| L-BFGS memory length | 5 |
| Curvature filter: full / limited | 1e-24 / 1e-18 |
| Gradient tolerance | 1e-8 |
| Maximum updates per method | 500 |
| BLAS threads | 4 |

All methods use the same initial point and the same analytic Hessian
eigensolver. HiSD-BB uses zero momentum. HiSD is the zero-momentum case of
the update in `accelerated_hisd.py`, exposed as the separate `HiSD` class.
Baseline methods that reach 500 updates without meeting the gradient
tolerance are reported as `max_iter`.

## Files

- `model.py`: energy, derivatives, saddle information and eigensolver.
- `run.py`: parameters, initial point and execution of all six methods.
- `hisd.py`: HiSD.
- `hisd_bb.py`: HiSD-BB.
- `accelerated_hisd.py`: A-HiSD.
- `hn_bfgs_hisd_ls.py`: Full-memory LS.
- `hn_lbfgs_hisd.py`: Limited-memory.
- `hn_lbfgs_hisd_ls.py`: Limited-memory LS.

The solvers return `(final_point, history)`. Histories are held in memory
during a run; no precomputed histories, images or cache files are needed.
