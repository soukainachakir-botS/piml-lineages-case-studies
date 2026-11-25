# Supplementary code: What Physics-Informed Machine Learning Inherits from Numerical Analysis

Code and results for the three case studies (Section 10) of the review by S. Chakir and S. B. Belhaouari, submitted to Engineering with Computers.
Everything runs on a CPU. Run each script from inside `code/`. Figures are written to `fig/`.

## Requirements

Python 3.10 or newer, plus the packages listed in `requirements.txt`:

    pip install -r requirements.txt

## Case study 1: forward boundary-layer problem (Fig. 4, Tables 7 and 8)

| Script | Purpose | Output |
|---|---|---|
| `case_forward.py` | Runs FEM (P1/P2), ELM and PINN on the advection-diffusion boundary-layer problem for each value of epsilon | `forward_results.json`, `forward_curves.json` |
| `run_ladder.py B` and `run_ladder.py C` | Runs the PINN remedy ladders (B and C) | `ladder_B.json`, `ladder_C.json` |
| `case_forward_pinn_fix.py` | Runs the graded-points remedy, reported as a generalization-error illustration | `forward_pinnfix.json` |
| `pinn_try.py` | Short exploratory PINN runs | none |

## Case study 2: parameter identification with and without identifiability (Fig. 5, Table 9)

| Script | Purpose | Output |
|---|---|---|
| `case_inverse.py` | Least squares from twelve starts, five PINN runs, Fisher information and Cramér–Rao bound, for case I (non-identifiable) and case II (identifiable) | `inverse_case1.json`, `inverse_case2.json` |
| `case_inverse_bias.py` | Bias test: PINN runs for case I started from large D | `inverse_case1_bias.json` |

## Case study 3: Fourier neural operator outside its training distribution (Fig. 6)

| Script | Purpose | Output |
|---|---|---|
| `case_operator.py` | Generates 1D Darcy data with a finite-volume solver, trains an FNO and tests it on four distributions, recording a per-query residual for each prediction | `operator_results.json` |

`case_operator.py` takes about 10 to 15 minutes on a laptop CPU.

## Figures

| Script | Output |
|---|---|
| `make_figs.py` | `fig/fig_case1.pdf`, `fig/fig_case2.pdf` |
| `make_fig_operator.py` | `fig/fig_case3.pdf` |

Every JSON file used in the paper is included, so the figures can be rebuilt without rerunning the experiments.
Random seeds are fixed inside each script.

## Licence

MIT. Copyright (c) 2026 Soukaina Chakir and Samir B. Belhaouari.

## Contact

Soukaina Chakir, soukaina.chakir@edu.umi.ac.ma
