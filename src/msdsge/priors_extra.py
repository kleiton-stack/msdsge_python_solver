"""Additional priors: univariate ones from ``rise.def``, Dirichlet and endogenous priors.

Univariate (``make_prior``), parameterised by (mean, standard deviation) unless stated otherwise:

- ``cauchy``: no moments; the two numbers are location and scale.
- ``laplace``, ``logistic``: mean and standard deviation.
- ``pareto``: mean and standard deviation (requires a finite standard deviation, i.e. shape > 2).
- ``left_triang``: triangular with mode at the left end (decreasing density);
  ``right_triang``: mode at the right end (increasing density).
- ``truncated_normal``: mean and standard deviation of the normal before truncation, truncated at
  the parameter bounds.
- ``exponential``: only the mean is used.

RISE does not document the parameterisation of these shapes (internal functions in p-code); the
conventions above are this package's and have not been compared with it.

``DirichletPrior`` follows the format in the help of RISE's ``setup_priors``:
``{sd_ii, name1, m1, ..., namen, mn}``, with the off-diagonal probabilities of one row of the
transition matrix and the standard deviation of the diagonal element (the remainder).

``EndogenousPrior`` reproduces ``estim_endogenous_priors``: priors on model moments
(Del Negro and Schorfheide, 2008; Christiano, Trabandt and Walentin, 2011), added to the log
posterior without a Jacobian.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

EXTRA_SHAPES = ("cauchy", "laplace", "logistic", "pareto", "left_triang", "right_triang",
                "truncated_normal", "exponential")


def make_prior(shape: str, m: float, s: float, bounds=None):
    """Frozen scipy distribution for the additional shapes."""
    if shape == "cauchy":
        return stats.cauchy(m, s)
    if shape == "laplace":
        return stats.laplace(m, s / np.sqrt(2))
    if shape == "logistic":
        return stats.logistic(m, s * np.sqrt(3) / np.pi)
    if shape == "pareto":
        if m <= 0:
            raise ValueError("pareto prior requires a positive mean")
        a = 1 + np.sqrt(1 + (m / s) ** 2)
        return stats.pareto(b=a, scale=m * (a - 1) / a)
    if shape in ("left_triang", "right_triang"):
        w = s * np.sqrt(18.0)
        if shape == "left_triang":
            return stats.triang(c=0.0, loc=m - w / 3, scale=w)
        return stats.triang(c=1.0, loc=m - 2 * w / 3, scale=w)
    if shape == "truncated_normal":
        lo, up = (-np.inf, np.inf) if bounds is None else bounds
        if not lo < up:
            raise ValueError("truncated_normal prior requires lower bound < upper bound")
        return stats.truncnorm((lo - m) / s, (up - m) / s, loc=m, scale=s)
    if shape == "exponential":
        if m <= 0:
            raise ValueError("exponential prior requires a positive mean")
        return stats.expon(scale=m)
    raise ValueError(f"unknown prior: {shape!r}")


def from_quantiles(shape: str, lo: float, up: float, p: float, bounds=None):
    """Distribution with mass ``p`` between ``lo`` and ``up`` and equal tails."""
    hi = 1 - (1 - p) / 2
    mid, half = (lo + up) / 2, (up - lo) / 2
    if shape == "cauchy":
        return stats.cauchy(mid, half / stats.cauchy.ppf(hi))
    if shape == "laplace":
        return stats.laplace(mid, half / stats.laplace.ppf(hi))
    if shape == "logistic":
        return stats.logistic(mid, half / stats.logistic.ppf(hi))
    if shape == "left_triang":      # F(x) = 1 - (1 - u)^2, u = (x - a) / w
        u1, u2 = 1 - np.sqrt(1 - (1 - hi)), 1 - np.sqrt(1 - hi)
        w = (up - lo) / (u2 - u1)
        return stats.triang(c=0.0, loc=lo - u1 * w, scale=w)
    if shape == "right_triang":     # F(x) = u^2
        u1, u2 = np.sqrt(1 - hi), np.sqrt(hi)
        w = (up - lo) / (u2 - u1)
        return stats.triang(c=1.0, loc=lo - u1 * w, scale=w)
    if shape == "pareto":           # F(x) = 1 - (xm / x)^a
        if lo <= 0:
            raise ValueError("pareto prior requires positive quantiles")
        a = np.log(hi / (1 - hi)) / np.log(up / lo)
        return stats.pareto(b=a, scale=lo * hi ** (1 / a))
    if shape == "truncated_normal":
        blo, bup = (-np.inf, np.inf) if bounds is None else bounds

        def make(h):
            return stats.truncnorm((blo - h[0]) / np.exp(h[1]), (bup - h[0]) / np.exp(h[1]), loc=h[0], scale=np.exp(h[1]))
        z = stats.norm.ppf(hi)
        sol = optimize.least_squares(lambda h: (make(h).ppf([1 - hi, hi]) - [lo, up]) / (up - lo),
                                     [mid, np.log(half / z)], xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if np.max(np.abs(sol.fun)) > 1e-6:
            raise ValueError(f"could not find a truncated_normal for the quantiles ({lo}, {up}) with mass {p}")
        return make(sol.x)
    raise ValueError(f"quantile-based prior not available for {shape!r}")


def location_scale(dist) -> tuple[float, float]:
    """Mean and standard deviation, or median and half the interquartile range when the moments do not exist."""
    m, s = float(dist.mean()), float(dist.std())
    if not np.isfinite(m):
        m = float(dist.median())
    if not np.isfinite(s):
        s = float(dist.ppf(0.75) - dist.ppf(0.25)) / 2
    return m, s


# ---------------------------------------------------------------- Dirichlet
@dataclass
class DirichletPrior:
    """Joint Dirichlet for the off-diagonal probabilities of one row of the transition matrix.

    ``index`` holds the positions of the parameters in the estimated vector; ``alpha`` has one
    extra element, the last, for the diagonal probability (1 minus the sum of the others).
    """
    index: np.ndarray
    alpha: np.ndarray
    names: list = field(default_factory=list)

    @classmethod
    def from_rise(cls, all_names, spec) -> "DirichletPrior":
        """``spec = [sd_ii, name1, m1, name2, m2, ...]`` (RISE format) or
        ``{"names": [...], "alpha": [...]}`` with the hyperparameters (the last is the diagonal one)."""
        all_names = list(all_names)
        if isinstance(spec, dict):
            names, alpha = list(spec["names"]), np.asarray(spec["alpha"], float)
        else:
            sd, rest = float(spec[0]), list(spec[1:])
            names, means = rest[0::2], np.array(rest[1::2], float)
            mii = 1 - means.sum()
            if not (0 < mii < 1 and np.all(means > 0)) or sd <= 0:
                raise ValueError("Dirichlet: the means must be positive and sum to less than 1")
            a0 = mii * (1 - mii) / sd ** 2 - 1
            if a0 <= 0:
                raise ValueError("Dirichlet: diagonal standard deviation too large for the given means")
            alpha = np.append(means, mii) * a0
        if alpha.size != len(names) + 1 or np.any(alpha <= 0):
            raise ValueError("Dirichlet: 'alpha' must have one positive element per name plus the diagonal one")
        missing = [n for n in names if n not in all_names]
        if missing:
            raise ValueError(f"Dirichlet: parameters not estimated: {missing}")
        return cls(np.array([all_names.index(n) for n in names]), alpha, names)

    @property
    def mean(self) -> np.ndarray:
        return self.alpha[:-1] / self.alpha.sum()

    @property
    def std(self) -> np.ndarray:
        a0 = self.alpha.sum()
        m = self.alpha[:-1] / a0
        return np.sqrt(m * (1 - m) / (a0 + 1))

    def logpdf(self, theta) -> float:
        p = np.asarray(theta, float)[self.index]
        rest = 1 - p.sum()
        if np.any(p <= 0) or rest <= 0:
            return -np.inf
        x = np.append(p, rest)
        return float(special.gammaln(self.alpha.sum()) - special.gammaln(self.alpha).sum()
                     + np.sum((self.alpha - 1) * np.log(x)))

    def rvs(self, rng) -> np.ndarray:
        return rng.dirichlet(self.alpha)[:-1]


# ---------------------------------------------------------------- endogenous priors
class EndogenousPrior:
    """Priors on model moments (RISE's ``estim_endogenous_priors``).

    ``moments(solution)`` returns the vector of moments; ``priors`` is the list, in the same order,
    of ``(mean, standard deviation, shape)`` tuples (as in RISE) or ``Prior`` objects. If the
    computation of the moments fails, the log density is ``-inf``.
    """

    def __init__(self, moments, priors):
        from .estimation import Prior
        self.moments = moments
        self.priors = [p if isinstance(p, Prior) else Prior(str(p[2]).replace("_pdf", ""), float(p[0]), float(p[1]))
                       for p in priors]

    def logpdf_solution(self, solution) -> float:
        try:
            v = np.atleast_1d(np.asarray(self.moments(solution), float))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, ArithmeticError):
            return -np.inf
        if v.size != len(self.priors):
            raise ValueError(f"endogenous priors: {v.size} moments for {len(self.priors)} priors")
        if not np.all(np.isfinite(v)):
            return -np.inf
        return float(sum(p.logpdf(x) for p, x in zip(self.priors, v)))

    def logpdf(self, problem, theta) -> float:
        from .model import SteadyStateError
        from .solve import SolveError
        try:
            sol = problem.solve(np.asarray(theta, float))
        except (SolveError, SteadyStateError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            return -np.inf
        return self.logpdf_solution(sol)


def stdev_moments(variables, regime: int = 1):
    """Moment function: theoretical standard deviations of the variables, conditional on the regime
    (analogous to RISE's ``cov_endo_priors`` example)."""
    def moments(solution):
        from .analysis import theoretical_autocovariances
        cov = theoretical_autocovariances(solution, lags=0, regime=regime)[:, :, 0]
        pos = [solution.model.endogenous.index(v) for v in variables]
        return np.sqrt(np.clip(np.diag(cov)[pos], 0, None))
    return moments
