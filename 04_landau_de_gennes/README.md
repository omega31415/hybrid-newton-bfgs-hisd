# Landau-de Gennes saddle searches

Four independent examples search between the C, M and N states, whose
numerical Morse indices are 2, 3 and 4. Each entry uses the **Hybrid method
(Limited-memory)** with fixed step sizes.

## Run

Use the NumPy environment described in the repository README. From the
repository root:

```bash
python 04_landau_de_gennes/search_c_to_m.py
python 04_landau_de_gennes/search_m_to_n.py
python 04_landau_de_gennes/search_n_to_m.py
python 04_landau_de_gennes/search_m_to_c.py
```

Each command runs one search and prints its update count, final gradient
norm, energy, elapsed time and termination status. `converged` means that
the gradient tolerance was met; `max_iter` means that the iteration budget
was exhausted. The target index specifies the search dynamics.

All four hybrid searches were checked on Windows with Python 3.13.5 and
the pinned NumPy environment in the repository: each reached the stated
gradient tolerance. The alternative method configurations were checked
over short runs.

## Model and search parameters

The five independent tensor components are discretized on the square
[-1, 1] x [-1, 1], with the prescribed Dirichlet boundary conditions in
`model.py`. The model uses B = 6400, C = 3500 and A = -B^2/(3C).
All computations use float64 arrays.

| Parameter | Value |
| --- | --- |
| Lambda / mesh spacing | 50 / 1/50 |
| Degrees of freedom | 49005 |
| Initial perturbation magnitude | 1e-3 |
| L-BFGS memory length | 30 |
| Fixed steps beta1 / beta2 | 0.01 / 0.01 |
| Gradient tolerance in the discrete L2 metric | 1e-6 |
| Eigenvalue safeguard / curvature filter | 1e-6 / 1e-6 |
| Initial / minimum central-difference probe length | 1e-6 / 1e-9 |
| Eigenspace update | One embedded LOBPCG step per iteration |
| BLAS threads | 4 |

Eigenvectors below are numbered from 1, in increasing eigenvalue order.
Each initial point is formed by adding 1e-3 times the specified supplied
eigenvector to its starting state. The initial search subspace consists
of the first `target index` supplied eigenvectors.

| Entry | Direction | Target index | Perturbation eigenvector | Maximum updates |
| --- | --- | --- | --- | --- |
| `search_c_to_m.py` | C to M, upward | 3 | C: 3 | 10000 |
| `search_m_to_n.py` | M to N, upward | 4 | M: 4 | 10000 |
| `search_n_to_m.py` | N to M, downward | 3 | N: 4 | 10000 |
| `search_m_to_c.py` | M to C, downward | 2 | M: 3 | 10000 |

## Other methods

Each entry contains commented configurations for **HiSD-BB** and
**A-HiSD**. Replace the active solver configuration with one of these
blocks to run that method on the same input. All four entries and all
method configurations share the default limit `MAX_ITER = 10000`.

HiSD-BB uses initial step 2e-4, zero momentum and BB step bounds
[0.2, 5] times the initial step. A-HiSD uses step 8e-5 and momentum 0.99
in all four searches, matching the server runs reported in the paper.
The gradient tolerance and finite-difference settings are shared.
HiSD-BB and A-HiSD may need more than 10000 updates to converge; some
searches can require millions. Increase `MAX_ITER` in the selected entry
when running these methods to convergence. Reaching the supplied limit
is reported as `max_iter`.

## Files and input data

- `model.py`: energy, metric gradient, Hessian-vector products and metric norm.
- `hn_lbfgs_hisd.py`: the limited-memory hybrid method with fixed steps.
- `hisd_bb.py`: HiSD-BB.
- `accelerated_hisd.py`: A-HiSD.
- `search_*.py`: four separate search examples and commented alternatives.
- `input_data/c_state.npz`: C starting state and 3 initial eigenvectors.
- `input_data/m_state.npz`: M starting state and 4 initial eigenvectors.
- `input_data/n_state.npz`: N starting state and 4 initial eigenvectors.

Each input archive contains only `u` (the starting state) and `V`
(the initial eigenvectors). These are fixed inputs extracted from the
experiment. Their vector signs are retained to specify the perturbation
direction. Both M searches share the same input archive.

The solvers return the final point, search subspace and an in-memory
history. The entry scripts print the result without writing output files.

These examples provide the model, fixed inputs and search settings used
for the experiment. The paper runs were performed on a server. Differences
in floating-point arithmetic, BLAS/LAPACK implementations and initial-data
precision can change the iteration counts and timings. Exact agreement
with the reported counts is not required; check the final gradient against
the stated tolerance.
