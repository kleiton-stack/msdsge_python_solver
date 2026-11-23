"""Optimisers for the posterior mode (bounded minimisation).

Equivalents of the ``optimizer`` options seen in RISE:

=================== =========================================================================
``bee_gate``/``abc`` Artificial bee colony (Karaboga and Basturk, 2007), the algorithm of RISE's
                     ``bee`` class ("global optimization of constrained functions using
                     bees"). Own implementation, from the paper.
``blockwise``        Optimisation by blocks of parameters (RISE's ``blockwise_optimization`` and
                     ``estim_blocks`` option).
``fmincon``          Bounded quasi-Newton: SciPy's ``L-BFGS-B``.
``fminunc``,         Unbounded quasi-Newton with numerical gradient: SciPy's ``BFGS``, applied
``csminwel``         to parameters transformed to respect the bounds. Not a port of Sims's
                     csminwel.
``fminsearch``       Nelder-Mead.
``differential_evolution``, ``dual_annealing``: SciPy global optimisers (not in RISE).
=================== =========================================================================

They are functional equivalents, not replicas: the search path differs from MATLAB's.
``multistart`` repeats any of them from several starting points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

__all__ = ["OPTIMIZERS", "OptimResult", "minimize", "abc", "blockwise", "multistart"]

OPTIMIZERS = ("bee_gate", "abc", "blockwise", "fmincon", "fminunc", "csminwel", "fminsearch",
              "differential_evolution", "dual_annealing")


@dataclass
class OptimResult:
    x: np.ndarray
    fun: float
    nfev: int
    message: str
    extras: dict = field(default_factory=dict)


def _bounds(n, lower, upper):
    lo = np.full(n, -np.inf) if lower is None else np.asarray(lower, float)
    up = np.full(n, np.inf) if upper is None else np.asarray(upper, float)
    return lo, up


def _box(x0, lo, up, width: float = 1.0):
    """Finite box for the population methods: the bounds, or x0 ± width (1 + |x0|) where they are infinite."""
    span = width * (1 + np.abs(x0))
    return np.where(np.isfinite(lo), lo, x0 - span), np.where(np.isfinite(up), up, x0 + span)


class _Counter:
    def __init__(self, fun, penalty=1e12):
        self.fun, self.n, self.penalty = fun, 0, penalty

    def __call__(self, x):
        self.n += 1
        v = self.fun(x)
        return float(v) if np.isfinite(v) else self.penalty


# ---------------------------------------------------------------- bee colony
def abc(fun, x0, lower=None, upper=None, colony: int = 20, maxiter: int = 500, limit: int | None = None,
        maxfev: int | None = None, tol: float = 1e-8, stall: int = 100, seed: int = 0, width: float = 1.0) -> OptimResult:
    """Artificial bee colony (Karaboga and Basturk, 2007).

    ``colony`` food sources. In each cycle: employed-bee phase (each source i generates
    v_ij = x_ij + φ (x_ij - x_kj), with j and k ≠ i drawn at random and φ ~ U(-1, 1), and keeps the
    better of the two); onlooker phase (the same move, with the source chosen with probability
    proportional to fitness); scout phase (a source with no improvement for ``limit`` trials is
    replaced by a random point). ``x0`` enters as one of the sources. Stops if there is no
    improvement larger than ``tol`` in ``stall`` cycles.
    """
    x0 = np.asarray(x0, float)
    n = x0.size
    lo, up = _bounds(n, lower, upper)
    blo, bup = _box(x0, lo, up, width)
    f = _Counter(fun)
    rng = np.random.default_rng(seed)
    limit = colony * n if limit is None else limit
    X = blo + rng.uniform(size=(colony, n)) * (bup - blo)
    X[0] = x0
    F = np.array([f(x) for x in X])
    trials = np.zeros(colony, int)
    best, fbest = X[np.argmin(F)].copy(), F.min()
    last_improve, it = 0, 0

    def move(i):
        nonlocal best, fbest
        k = int(rng.integers(colony - 1))
        k += k >= i
        j = int(rng.integers(n))
        v = X[i].copy()
        v[j] = np.clip(X[i, j] + rng.uniform(-1, 1) * (X[i, j] - X[k, j]), max(lo[j], -np.inf), up[j])
        fv = f(v)
        if fv < F[i]:
            X[i], F[i], trials[i] = v, fv, 0
            if fv < fbest:
                best, fbest = v.copy(), fv
        else:
            trials[i] += 1

    for it in range(1, maxiter + 1):
        prev = fbest
        for i in range(colony):
            move(i)
        fit = 1.0 / (1.0 + (F - F.min()))            # Karaboga's fitness, with the cost shifted to ≥ 0
        p = fit / fit.sum()
        for i in rng.choice(colony, colony, p=p):
            move(int(i))
        worst = int(np.argmax(trials))
        if trials[worst] > limit:
            X[worst] = blo + rng.uniform(size=n) * (bup - blo)
            F[worst], trials[worst] = f(X[worst]), 0
            if F[worst] < fbest:
                best, fbest = X[worst].copy(), F[worst]
        if prev - fbest > tol:
            last_improve = it
        if it - last_improve >= stall or (maxfev is not None and f.n >= maxfev):
            break
    return OptimResult(best, float(fbest), f.n, f"abc: {it} cycles", {"population": X, "values": F})


# ---------------------------------------------------------------- blocks
def blockwise(fun, x0, lower=None, upper=None, blocks=None, inner: str = "L-BFGS-B", sweeps: int = 20,
              tol: float = 1e-8, maxiter: int = 500) -> OptimResult:
    """Optimise one block of parameters at a time, in successive sweeps, until the improvement of
    a whole sweep falls below ``tol``. ``blocks`` is a list of lists of indices (by default, one
    parameter per block)."""
    x = np.asarray(x0, float).copy()
    lo, up = _bounds(x.size, lower, upper)
    f = _Counter(fun)
    blocks = [[i] for i in range(x.size)] if blocks is None else [list(b) for b in blocks]
    fx = f(x)
    s = 0
    for s in range(1, sweeps + 1):
        start = fx
        for b in blocks:
            def sub(v, b=b):
                y = x.copy()
                y[b] = v
                return f(y)
            bnds = [(None if not np.isfinite(lo[i]) else lo[i], None if not np.isfinite(up[i]) else up[i]) for i in b]
            r = optimize.minimize(sub, x[b], method=inner, bounds=bnds, options={"maxiter": maxiter})
            if r.fun < fx:
                x[b], fx = r.x, float(r.fun)
        if start - fx < tol:
            break
    return OptimResult(x, float(fx), f.n, f"blockwise: {s} sweeps")


# ---------------------------------------------------------------- dispatch
def _unbounded(fun, x0, lo, up, method, maxiter):
    """Unbounded quasi-Newton on transformed parameters (logit on intervals, log on half-lines)."""
    both, low, high = np.isfinite(lo) & np.isfinite(up), np.isfinite(lo) & ~np.isfinite(up), ~np.isfinite(lo) & np.isfinite(up)

    def to_x(z):
        x = z.copy()
        x[both] = lo[both] + (up[both] - lo[both]) / (1 + np.exp(-z[both]))
        x[low] = lo[low] + np.exp(z[low])
        x[high] = up[high] - np.exp(z[high])
        return x
    z0 = np.asarray(x0, float).copy()
    u = (x0[both] - lo[both]) / (up[both] - lo[both])
    z0[both] = np.log(u / (1 - u))
    z0[low] = np.log(x0[low] - lo[low])
    z0[high] = np.log(up[high] - x0[high])
    with np.errstate(over="ignore"):
        r = optimize.minimize(lambda z: fun(to_x(z)), z0, method=method, options={"maxiter": maxiter})
    return to_x(r.x), float(r.fun), str(r.message)


def minimize(fun, x0, lower=None, upper=None, method: str = "bee_gate", maxiter: int = 2000, seed: int = 0,
             **options) -> OptimResult:
    """Minimise ``fun`` with the requested optimiser (see the module table)."""
    x0 = np.asarray(x0, float)
    lo, up = _bounds(x0.size, lower, upper)
    if method in ("bee_gate", "abc"):
        return abc(fun, x0, lo, up, maxiter=options.pop("cycles", min(maxiter, 500)), seed=seed, **options)
    if method == "blockwise":
        return blockwise(fun, x0, lo, up, maxiter=maxiter, **options)
    f = _Counter(fun)
    if method in ("fmincon", "fminsearch"):
        bnds = [(None if not np.isfinite(a) else a, None if not np.isfinite(b) else b) for a, b in zip(lo, up)]
        opts = {"maxiter": maxiter, **options}
        if method == "fminsearch":
            opts = {"xatol": 1e-7, "fatol": 1e-9, **opts}
        r = optimize.minimize(f, x0, method="L-BFGS-B" if method == "fmincon" else "Nelder-Mead", bounds=bnds, options=opts)
        return OptimResult(r.x, float(r.fun), f.n, f"{method}: {r.message}")
    if method in ("fminunc", "csminwel"):
        x, fx, msg = _unbounded(f, x0, lo, up, "BFGS", maxiter)
        return OptimResult(x, fx, f.n, f"{method}: {msg}")
    if method in ("differential_evolution", "dual_annealing"):
        blo, bup = _box(x0, lo, up, options.pop("width", 1.0))
        bnds = list(zip(blo, bup))
        if method == "differential_evolution":
            r = optimize.differential_evolution(f, bnds, x0=np.clip(x0, blo, bup), seed=seed,
                                                maxiter=options.pop("generations", 200), polish=False, **options)
        else:
            r = optimize.dual_annealing(f, bnds, x0=np.clip(x0, blo, bup), seed=seed,
                                        maxiter=options.pop("annealing_iter", 300), no_local_search=True, **options)
        return OptimResult(r.x, float(r.fun), f.n, f"{method}: {r.message}")
    raise ValueError(f"unknown optimizer: {method!r}; options: {OPTIMIZERS}")


def multistart(fun, x0, lower=None, upper=None, starts: int = 5, method: str = "fmincon", seed: int = 0,
               draw=None, spread: float = 0.25, optimizer=None, **options) -> OptimResult:
    """Multiple restarts: the first one starts from ``x0``; the others, from ``draw(rng)`` (for
    example, a prior draw) or, without ``draw``, from uniform points in a box around ``x0`` that
    covers the fraction ``spread`` of the distance to each bound. Only points with a finite function
    value are used. Returns the best; all local optima are kept in ``extras["runs"]``.

    ``optimizer`` (optional) is a function ``(fun, x0) -> object with x and fun``, used instead of ``method``.
    """
    x0 = np.asarray(x0, float)
    lo, up = _bounds(x0.size, lower, upper)
    blo, bup = _box(x0, lo, up)
    rng = np.random.default_rng(seed)
    runs, nfev = [], 0
    for k in range(starts):
        if k == 0:
            s = x0
        else:
            s = None
            for _ in range(200):
                c = np.asarray(draw(rng), float) if draw is not None else \
                    x0 + spread * (rng.uniform(size=x0.size) * (bup - blo) + blo - x0)
                v = fun(c)
                nfev += 1
                if np.all(c >= lo) and np.all(c <= up) and np.isfinite(v) and v < 1e11:
                    s = c
                    break
            if s is None:
                continue
        r = optimizer(fun, s) if optimizer is not None else minimize(fun, s, lo, up, method=method, seed=seed + k, **options)
        nfev += getattr(r, "nfev", 0)
        runs.append({"start": s, "x": np.asarray(r.x, float), "fun": float(r.fun)})
    best = min(runs, key=lambda d: d["fun"])
    return OptimResult(best["x"], best["fun"], nfev, f"multistart: best of {len(runs)} starts", {"runs": runs})
