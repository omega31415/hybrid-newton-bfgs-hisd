# Hybrid Newton-BFGS HiSD

Experiment code for **Hybrid Newton--BFGS High-index Saddle Dynamics for
Accelerating Saddle-point Searches**.

This directory contains the four experiments from the paper.

| Directory | Experiment |
| --- | --- |
| `01_polynomial` | Nonlinear polynomial: local convergence behavior |
| `02_rosenbrock` | Modified Rosenbrock: an ill-conditioned problem |
| `03_neural_network` | Neural-network loss landscape: degenerate saddle manifolds |
| `04_landau_de_gennes` | Landau-de Gennes model: four saddle searches |

## Installation

The NumPy examples (01, 02 and 04) use the environment tested on Windows
with Python 3.13.5.
Install their dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

Their dependencies are NumPy, SciPy and threadpoolctl. The first two examples
generate their model coefficients and initial points using fixed random seeds.
The Landau-de Gennes examples include three fixed starting states and their
initial eigenvectors.

The neural-network example uses PyTorch and has its own tested environment
and `03_neural_network/requirements.txt`. In a Python 3.11 environment, install:

```bash
python -m pip install -r 03_neural_network/requirements.txt
```

No notebook environment is required. The neural-network example includes
the fixed dataset and small-network checkpoint as `input_data.npz`, and
reconstructs the reference saddle by neuron splitting. See each experiment's
README for its setup and tested dependency versions.

## Run

From this directory:

```bash
python 01_polynomial/run.py
python 02_rosenbrock/run.py
python 03_neural_network/run.py
python 04_landau_de_gennes/search_c_to_m.py
python 04_landau_de_gennes/search_m_to_n.py
python 04_landau_de_gennes/search_n_to_m.py
python 04_landau_de_gennes/search_m_to_c.py
```

Each script uses the experiment parameters defined near its top and prints
the iteration count, final gradient norm and termination status. The first
three examples also report errors or distances to the reference saddle.
A `max_iter` status means that the method exhausted the iteration budget;
it does not indicate convergence. The Rosenbrock baseline methods are also
run for their full 500-iteration budget.

Each Landau-de Gennes entry runs the limited-memory hybrid method with fixed
steps. Its commented configurations provide the parameters for HiSD-BB and
A-HiSD. See `04_landau_de_gennes/README.md` for the four search directions.
All LdG configurations default to 10000 updates; the baseline methods may
need a larger limit to converge.

## Algorithm names

| Name in the paper / figures | File | Python class |
| --- | --- | --- |
| HiSD | `hisd.py` | `HiSD` |
| HiSD-BB | `hisd_bb.py` | `HiSDBB` |
| A-HiSD | `accelerated_hisd.py` | `AcceleratedHiSD` |
| Full-memory LS | `hn_bfgs_hisd_ls.py` | `HNBFGSHiSDLS` |
| Limited-memory | `hn_lbfgs_hisd.py` | `HNLBFGSHiSD` |
| Limited-memory LS | `hn_lbfgs_hisd_ls.py` | `HNLBFGSHiSDLS` |

`HN` denotes Hybrid Newton, `LBFGS` denotes the limited-memory BFGS
approximation, and `LS` denotes line search. Both limited-memory methods
use memory length 5 in the first two experiments; the neural-network
limited-memory version uses 10, and the Landau-de Gennes version uses 30.
The LdG table's Hybrid method corresponds to `hn_lbfgs_hisd.py`.
A-HiSD is run with momentum 0.95 in the Rosenbrock example, with both 0.9
and 0.95 in the neural-network example, and with 0.9 in the LdG example.
Each experiment contains its own algorithm files and can be run independently.

The model definitions and numerical updates are extracted from the research
code and notebooks. `tol_curv` controls the curvature filter, and the
experiment's stopping tolerance controls gradient termination. The first
two examples and the Landau-de Gennes scripts limit BLAS to four threads.
The neural-network example uses PyTorch CPU float64 computations and one
thread. Floating-point differences across operating systems and BLAS/LAPACK
implementations can affect terminal iteration counts, particularly near the
stopping tolerance. The LdG examples provide runnable searches; their iteration
counts need not exactly match the server runs reported in the paper.
