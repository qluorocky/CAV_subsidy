"""Numerical model for Luo et al. (2019), Transportation Research B 129."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class Parameters:
    a: float = 0.108
    b: float = 0.957
    market: float = 200.0
    benefit_scale: float = 6000.0
    beta: float = 0.33
    sigma: float = 1.1
    r_government: float = 0.1
    r_agent: float = 0.1
    effort_cost: float = 10.0
    w_max: float = 10.0
    a_min: float = 0.05
    a_max: float = 0.20
    subsidy_max: float = 1.0
    sensitivity_max: float = 2.0


def benefit(x, p: Parameters = Parameters()):
    """The supplied benefit.m function, with price P=0."""
    x = np.asarray(x, dtype=float)
    return p.benefit_scale * ((x - p.beta) ** 2 - p.beta ** 2)


def bass_drift(x, a, b):
    x = np.asarray(x, dtype=float)
    return (a + b * x) * (1 - x)


def deterministic_path(a, b, *, horizon=10.0, steps=100, x0=0.0):
    t = np.linspace(0, horizon, steps + 1)
    result = solve_ivp(lambda _, x: bass_drift(x, a, b), (0, horizon), [x0],
                       t_eval=t, rtol=1e-10, atol=1e-12)
    if not result.success:
        raise RuntimeError(result.message)
    return t, result.y[0]


def simulate_paths(p: Parameters = Parameters(), *, a=None, b=None, sigma=None,
                   horizon=10.0, steps=100, trials=300, seed=142857, x0=0.0):
    """Euler-Maruyama simulation. NumPy's RNG differs from MATLAB twister."""
    a = p.a if a is None else a
    b = p.b if b is None else b
    sigma = p.sigma if sigma is None else sigma
    dt = horizon / steps
    t = np.linspace(0, horizon, steps + 1)
    paths = np.empty((steps + 1, trials))
    paths[0] = x0
    shocks = np.random.default_rng(seed).standard_normal((steps, trials))
    for k in range(steps):
        x = paths[k]
        paths[k + 1] = x + bass_drift(x, a, b) * dt + sigma * x * np.sqrt(dt) * shocks[k]
    return t, paths


@dataclass
class HJBResult:
    x: np.ndarray
    w: np.ndarray
    value: np.ndarray
    subsidy: np.ndarray
    effort: np.ndarray
    sensitivity: np.ndarray
    errors: np.ndarray
    converged: bool


def _boundary(nx, nw, p):
    f = np.zeros((nx, nw))
    f[-1, :] = float(benefit(1, p))
    f[:, 0] = 0
    f[:, -1] = -float(benefit(1, p))
    return f


def _controls(f, x, dx, dw, p, old_s, optimize_sensitivity):
    fx = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dx)
    fw = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dw)
    fww = (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dw**2
    fxw = (f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] + f[:-2, :-2]) / (4 * dx * dw)
    xx = x[1:-1, None]
    numerator = (1 - xx) * (fx - p.r_government * old_s * p.market)
    denominator = -p.r_agent * p.effort_cost * fw
    a = np.clip(np.divide(numerator, denominator, out=np.full_like(fx, p.a),
                          where=np.abs(denominator) > 1e-10), p.a_min, p.a_max)
    growth = (a + p.b * xx) * (1 - xx)
    # h(s,a)=sqrt(s)-effort_cost*a^2/2, as used in main_effort.m.
    # Match the normalization in Numerical_TR_B/main_effort.m.
    root_s = np.abs(p.r_government * fw / (2 * p.market * growth))
    s = np.clip(root_s**2, 0, p.subsidy_max)
    if optimize_sensitivity:
        y = np.clip(np.divide(-xx * fxw, p.r_agent * fww,
                              out=np.ones_like(fww), where=np.abs(fww) > 1e-10),
                    0, p.sensitivity_max)
    else:
        y = np.ones_like(fww)
    return a, s, y


def _policy_value(x, w, p, a, s, y):
    """Evaluate a frozen HJB policy by sparse finite differences."""
    nx, nw = len(x), len(w)
    dx, dw = x[1] - x[0], w[1] - w[0]
    f = _boundary(nx, nw, p)
    matrix = lil_matrix(((nx - 2) * (nw - 2), (nx - 2) * (nw - 2)))
    rhs = np.empty(matrix.shape[0])
    index = lambda i, j: (i - 1) * (nw - 2) + j - 1
    for i in range(1, nx - 1):
        for j in range(1, nw - 1):
            row = index(i, j)
            ai, si, yi = a[i - 1, j - 1], s[i - 1, j - 1], y[i - 1, j - 1]
            bx = float(bass_drift(x[i], ai, p.b))
            bw = p.r_agent * (w[j] - np.sqrt(si) + 0.5 * p.effort_cost * ai**2)
            qx = 0.5 * p.sigma**2 * x[i]**2 / dx**2
            qw = 0.5 * p.sigma**2 * p.r_agent**2 * yi**2 / dw**2
            cross = p.sigma**2 * p.r_agent * x[i] * yi / (4 * dx * dw)
            coeff = {(i, j): -p.r_government - 2 * qx - 2 * qw}
            def add(ii, jj, v):
                coeff[ii, jj] = coeff.get((ii, jj), 0) + v
            add(i-1, j, qx)
            add(i+1, j, qx)
            add(i, j-1, qw)
            add(i, j+1, qw)
            if bx >= 0:
                add(i+1, j, bx/dx)
                add(i, j, -bx/dx)
            else:
                add(i-1, j, -bx/dx)
                add(i, j, bx/dx)
            if bw >= 0:
                add(i, j+1, bw/dw)
                add(i, j, -bw/dw)
            else:
                add(i, j-1, -bw/dw)
                add(i, j, bw/dw)
            for ii, jj, sign in ((i+1,j+1,1), (i-1,j-1,1),
                                 (i+1,j-1,-1), (i-1,j+1,-1)):
                add(ii, jj, sign*cross)
            rhs[row] = -p.r_government * (float(benefit(x[i], p)) - si*p.market*bx)
            for (ii, jj), v in coeff.items():
                if ii in (0, nx-1) or jj in (0, nw-1):
                    rhs[row] -= v*f[ii, jj]
                else:
                    matrix[row, index(ii, jj)] += v
    interior = spsolve(matrix.tocsr(), rhs)
    if not np.all(np.isfinite(interior)):
        raise FloatingPointError("Nonfinite HJB policy value")
    f[1:-1, 1:-1] = interior.reshape(nx-2, nw-2)
    return f


def solve_hjb(p: Parameters = Parameters(), *, nx=25, nw=25, max_iterations=300,
              tolerance=1e-5, optimize_sensitivity=False, damping=0.2):
    """Solve the X,W HJB. Default fixed Y=1 matches main_effort.m."""
    if min(nx, nw) < 4 or max_iterations < 1 or not 0 < damping <= 1:
        raise ValueError("Invalid HJB grid or iteration settings")
    x, w = np.linspace(0, 1, nx), np.linspace(0, p.w_max, nw)
    a = np.full((nx-2, nw-2), p.a)
    s = np.zeros_like(a)
    y = np.ones_like(a)
    f = _policy_value(x, w, p, a, s, y)
    errors = []
    for _ in range(max_iterations):
        an, sn, yn = _controls(f, x, x[1]-x[0], w[1]-w[0], p, s,
                               optimize_sensitivity)
        a = damping*an + (1-damping)*a
        s = damping*sn + (1-damping)*s
        y = damping*yn + (1-damping)*y
        candidate = _policy_value(x, w, p, a, s, y)
        updated = damping*candidate + (1-damping)*f
        error = float(np.max(np.abs(updated-f)))
        errors.append(error)
        f = updated
        if error < tolerance:
            break
    return HJBResult(x, w, f, s, a, y, np.asarray(errors), errors[-1] < tolerance)
