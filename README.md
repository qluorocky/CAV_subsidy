# Python experiments for AV adoption subsidies

This is a new Python implementation based on the MATLAB files supplied with
Qi Luo, Romesh Saigal, Zhibin Chen, and Yafeng Yin, “Accelerating the adoption
of automated vehicles by subsidies: A dynamic games approach,”
*Transportation Research Part B* 129 (2019), 226–243,
[doi:10.1016/j.trb.2019.09.011](https://doi.org/10.1016/j.trb.2019.09.011).

It runs without MATLAB. The supplied PDF and MATLAB files are **inputs for
interpretation**, not files included in this package. No result in this
repository should be described as an exact numerical reproduction of all
figures in the published paper. The table below states what each output is.

## Run

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m av_subsidy.figures --output results --figures all
```

The last command writes PNG figures, CSV data, `hjb_baseline.npz`, and
`run_metadata.json` to `results/`. To run only the core experiments, use
`--figures core`. `--grid 25` is the default for the two-state HJB solver.
The checked-in `results/` directory was generated with `--figures all` and
grid size 25. `run_metadata.json` records the parameters and convergence.

## What is reproduced

| Output | Basis | Status |
| --- | --- | --- |
| Figure 1 benefit curve | `Numerical_TR_B/benefit.m`, `marketsize.m`; paper §3.4 | Direct formula reproduction. The source uses \(-6000(0.33)^2\), approximately −653.4, while the paper rounds this to −650. |
| Figure 2 market paths | `Numerical_TR_B/plot_samplepath.m`; paper Eq. (1) | Same SDE and source parameters, with seeded NumPy Euler–Maruyama. Individual paths differ from MATLAB's `twister` generator. |
| Figure 6 value and error | `Numerical_TR_B/main_effort.m`; paper Theorem 2 | Independent finite-difference implementation of the two-state HJB with fixed sensitivity \(Y=1\). It converges under the stated numerical assumptions; no pixel or numeric match to the published figure is claimed. |
| Figure 7 subsidy | HJB result sampled at \(W\approx4.2\) | **Analogue only.** This fixed-\(Y\) solver produces a declining subsidy, not the paper's claimed two-threshold profile. Do not use it as evidence for that claim. |
| Figures 8 and 9 sweeps | Repeated HJB calculations at \(X\approx0.8,W\approx4.2\) | Model sensitivity analogues for benefit magnitude, worst-case penetration, and contagion. The supplied files omit the full published experiment settings. |
| Figure 10 source data | `Numerical_TR_B/plot_sensitivity.m` | Plots the five hard-coded points in the MATLAB file. These points are **not independently derived** from the HJB. The second x array is market potential (50–400), despite its MATLAB axis label saying volatility. |
| Figure 12 response examples | `Numerical_TR_B/subsidy_1.m`, `subsidy_2.m` | Deterministic Bass trajectories from the source parameters. |
| Figure 13 consumer/manufacturer comparison | Paper §4 and separate root-folder scripts | **Not reproduced.** The provided main 3D/4D MATLAB workflow is incomplete and does not give a reliable numeric target. |

Figures 3, 4, 5, and 11 in the article are explanatory diagrams, not outputs
of the supplied numerical scripts.

## Numerical assumptions and limitations

The article does not publish all the inputs necessary for exact numerical
replication. The two-state solver uses the utility implied by `main_effort.m`,
\(h(S,a)=\sqrt{S}-10a^2/2\), bounds \(0.05\leq a\leq0.20\), a continuation-value
range \(0\leq W\leq10\), and an explicit subsidy cap \(0\leq S\leq1\). The
subsidy cap is a numerical assumption chosen for a stable baseline; it is not
reported as a paper parameter. The control sensitivity is held at \(Y=1\), as
in the MATLAB main experiment. An optional `optimize_sensitivity=True` switch
exists in `solve_hjb`, but its local first-order update did **not** converge
on the baseline grid and is not used for the figures.

The HJB is evaluated for a frozen policy with a sparse linear solve and
upwind first derivatives, then updated with damping. This is a documented
numerical variant of the paper's stated finite-difference approach. It is not
a line-by-line MATLAB translation. A convergence flag means policy iterations
met the maximum-change tolerance; it is **not** proof of convergence under
mesh refinement or agreement with the published curves. For example, at
\(X\approx0.8,W\approx4.2\), the computed subsidy changes visibly between
20, 25, 30, and 40 grid points. The value is more stable. Treat subsidy
magnitudes and thresholds as exploratory until the missing experiment settings
are recovered and a grid convergence study is completed.

The supplied `Numerical_TR_B/main.m` contains an empty update loop.
`main_effort.m` fixes \(Y=1\), ends after at most 50 iterations, and refers to
`S(30, ...)` on a 25-row array in an optional plot. `compare_1.m` uses an
effort denominator from the other policy in one update. The root-folder
`main.m` and `solve_principal.m` belong to a broader contract-design
experiment, with different parameter values and state variables. They cannot
serve as an unambiguous specification of paper Figure 13.

## Validation performed

`unittest` checks the benefit formula, deterministic Bass behavior, seeded
stochastic paths, HJB boundary values, control bounds, finite results, and
convergence on a 20×20 grid. The checked-in 25×25 baseline converged in 152
policy iterations with maximum update `8.53e-6`. Every sensitivity run in the
checked-in `results/` has its own convergence flag in its CSV file. Inspect
those flags before interpreting a sweep.

## Code layout

- `av_subsidy/model.py`: Bass process, benefit function, and HJB solver.
- `av_subsidy/figures.py`: figure and CSV generator.
- `tests/test_model.py`: numerical checks.
- `results/`: generated example outputs and run metadata.

The source MATLAB, figures, and article PDF are not copied into the package.
