"""Command-line figure and data generation for the supplied paper."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import Parameters, benefit, deterministic_path, simulate_paths, solve_hjb


def _save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=180)
    plt.close(fig)


def _csv(folder, name, header, rows):
    with (folder / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def fig1(folder, p):
    x = np.linspace(0, 1, 201)
    g = benefit(x, p)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, g, color="black", lw=2)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(xlabel="Market penetration rate X", ylabel="Efficiency benefit g(X)", title="Figure 1: efficiency benefit")
    _save(fig, folder, "fig1_benefit")
    _csv(folder, "fig1_benefit", ["X", "benefit"], zip(x, g))


def fig2(folder, p):
    # The paper's sample-path script uses a=.05, b=.38, M=10, sigma=.1.
    t, paths = simulate_paths(p, a=.05, b=.38, sigma=.1, x0=0, seed=142857)
    _, mean = deterministic_path(.05, .38)
    sd = paths.std(axis=1, ddof=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, paths[:, :100], color="black", alpha=.06, lw=.6)
    ax.plot(t, mean, color="firebrick", lw=2, label="deterministic path")
    ax.plot(t, mean+2*sd, "k--", lw=1, label="mean ± 2 SD")
    ax.plot(t, mean-2*sd, "k--", lw=1)
    ax.set(xlabel="Period t", ylabel="Market penetration X", title="Figure 2: diffusion paths")
    ax.legend()
    _save(fig, folder, "fig2_paths")
    _csv(folder, "fig2_paths", ["time", "deterministic", "sample_mean", "sample_sd"],
         zip(t, mean, paths.mean(axis=1), sd))


def fig6_7(folder, p, nx, nw):
    result = solve_hjb(p, nx=nx, nw=nw)
    if not result.converged:
        raise RuntimeError(f"Baseline HJB did not converge: final error {result.errors[-1]:.6g}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im = axes[0].pcolormesh(result.x, result.w, (result.value / benefit(1, p)).T,
                            shading="auto", cmap="viridis")
    fig.colorbar(im, ax=axes[0], label="F / g(1)")
    axes[0].set(xlabel="Market penetration X", ylabel="Continuation value W", title="Figure 6a: government value")
    axes[1].semilogy(np.arange(1, len(result.errors)+1), result.errors)
    axes[1].set(xlabel="Policy iteration", ylabel="Maximum change in F", title="Figure 6b: convergence")
    _save(fig, folder, "fig6_value_convergence")
    wi = int(np.argmin(abs(result.w - 4.2)))
    # The controls are defined only at interior mesh nodes.
    wi_control = max(0, min(nw-3, wi-1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.x[1:-1], result.subsidy[:, wi_control], marker="o", ms=3)
    ax.set(xlabel="Market penetration X", ylabel="Per-AV subsidy", title=f"Figure 7 analogue: W={result.w[wi]:.2f}")
    _save(fig, folder, "fig7_subsidy")
    np.savez_compressed(folder / "hjb_baseline.npz", x=result.x, w=result.w,
                        value=result.value, subsidy=result.subsidy,
                        effort=result.effort, sensitivity=result.sensitivity,
                        errors=result.errors)
    _csv(folder, "fig7_subsidy", ["X", "subsidy"],
         zip(result.x[1:-1], result.subsidy[:, wi_control]))
    return {"iterations": len(result.errors), "final_error": float(result.errors[-1]),
            "converged": result.converged, "fixed_sensitivity": 1.0,
            "subsidy_cap": p.subsidy_max}


def _measure(result, x0=.8, w0=4.2):
    i = int(np.argmin(abs(result.x[1:-1] - x0)))
    j = int(np.argmin(abs(result.w[1:-1] - w0)))
    return result.value[i+1, j+1], result.subsidy[i, j]


def fig8_9(folder, p, nx, nw):
    """Parameter sweeps using the documented fixed-Y HJB approximation."""
    groups = {
        "fig8_benefit_scale": [("benefit_scale", v) for v in (3000, 4500, 6000, 7500, 9000)],
        "fig8_beta": [("beta", v) for v in (.25, .29, .33, .37, .41)],
        "fig9_contagion": [("b", v) for v in (.55, .75, .957, 1.15, 1.35)],
    }
    baseline = solve_hjb(p, nx=nx, nw=nw)
    if not baseline.converged:
        raise RuntimeError("Baseline HJB failed during sensitivity sweep")
    base_f, base_s = _measure(baseline)
    metadata = {}
    for name, changes in groups.items():
        rows = []
        for key, value in changes:
            result = solve_hjb(replace(p, **{key: value}), nx=nx, nw=nw)
            f, s = _measure(result)
            rows.append((value, f/base_f, s/base_s, int(result.converged), result.errors[-1]))
        _csv(folder, name, [changes[0][0], "relative_value", "relative_subsidy", "converged", "final_error"], rows)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        values = np.asarray(rows)
        axes[0].plot(values[:, 0], values[:, 1], "o-")
        axes[1].plot(values[:, 0], values[:, 2], "o-")
        axes[0].set(xlabel=changes[0][0], ylabel="Value / baseline")
        axes[1].set(xlabel=changes[0][0], ylabel="Subsidy / baseline")
        fig.suptitle(name.replace("_", " "))
        _save(fig, folder, name)
        metadata[name] = {"all_converged": all(row[3] for row in rows)}
    return metadata


def fig10(folder):
    # These five-point series are copied from Numerical_TR_B/plot_sensitivity.m.
    # They are preserved as source data, not recomputed from the HJB.
    volatility = np.array([.3, .5, .975, 1.2, 1.5])
    value = np.array([2.19e-4, 1.27e-4, 1.27e-5, 7.60e-6, 2.07e-6])
    market = np.array([50, 100, 200, 300, 400])
    subsidy = np.array([1.26e-4, 7.54e-5, 1.27e-5, 5.52e-7, 3.37e-6])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(volatility, value/value[1], "ko-")
    axes[1].plot(market, subsidy/subsidy[1], "ko-")
    axes[0].set(xlabel="Volatility σ", ylabel="Relative continuation value")
    axes[1].set(xlabel="Market potential M", ylabel="Relative subsidy")
    fig.suptitle("MATLAB sensitivity data (plot_sensitivity.m)")
    _save(fig, folder, "fig10_matlab_source_data")
    _csv(folder, "fig10_value_source", ["sigma", "value"], zip(volatility, value))
    _csv(folder, "fig10_subsidy_source", ["market", "subsidy"], zip(market, subsidy))


def fig12(folder):
    """Deterministic manufacturer-response examples from subsidy_1/2.m."""
    t, baseline = deterministic_path(.1, .85)
    _, medium = deterministic_path(.3, .85)
    _, high = deterministic_path(.5, .85)
    _, baseline_other = deterministic_path(.05, .38)
    _, changed_market = deterministic_path(.1, .95, x0=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for y, label in ((baseline, "a=0.1"), (medium, "a=0.3"), (high, "a=0.5")):
        axes[0].plot(t, 200*y, label=label)
    axes[0].set(xlabel="Period t", ylabel="Cumulative AV sales", title="Innovation response")
    axes[0].legend()
    axes[1].plot(t, 270*baseline_other, label="baseline: M=270")
    axes[1].plot(t, 108*changed_market, label="modified: M=108")
    axes[1].set(xlabel="Period t", ylabel="Cumulative AV sales", title="Market-potential response")
    axes[1].legend()
    _save(fig, folder, "fig12_response_examples")
    _csv(folder, "fig12_innovation", ["time", "a_0.1", "a_0.3", "a_0.5"],
         zip(t, 200*baseline, 200*medium, 200*high))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--figures", choices=("core", "all"), default="core")
    parser.add_argument("--grid", type=int, default=25)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    p = Parameters()
    fig1(args.output, p)
    fig2(args.output, p)
    hjb = fig6_7(args.output, p, args.grid, args.grid)
    fig12(args.output)
    status = {"parameters": asdict(p), "grid": args.grid, "hjb": hjb,
              "scope": args.figures,
              "notes": ["HJB fixes Y=1 as in main_effort.m.",
                        "Figure 7 is an analogue at W≈4.2; source lacks exact figure settings.",
                        "Figure 12 follows the parameter alternatives in subsidy_1.m and subsidy_2.m."]}
    if args.figures == "all":
        status["sensitivity"] = fig8_9(args.output, p, args.grid, args.grid)
        fig10(args.output)
    (args.output / "run_metadata.json").write_text(json.dumps(status, indent=2))
    print(json.dumps({"output": str(args.output), "hjb": hjb}, indent=2))


if __name__ == "__main__":
    main()
