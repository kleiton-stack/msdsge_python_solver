"""Global sensitivity analysis: quasi-random sampling, HDMR and Monte Carlo filtering.

Replaces RISE's ``quasi_monte_carlo``, ``hdmr`` and ``mcf`` classes (examples in
``examples/HighDimensionalModelRepresentation`` and ``examples/MonteCarloFiltering``).

- ``sample``: Sobol, Halton, Latin hypercube or uniform (``scipy.stats.qmc``).
- ``HDMR``: RS-HDMR with orthonormal polynomials (Li, Rosenthal and Rabitz, 2001; Li, Wang and
  Rabitz, 2002). ``f(x) = f0 + sum_i f_i(x_i) + sum_{i<j} f_ij(x_i, x_j) + ...``, each component
  expanded in orthonormal polynomials on the unit cube; the partial variance of each component is
  the sum of the squared coefficients and the sensitivity index is its ratio to the total
  variance (Sobol indices).
- ``MCF``: Monte Carlo filtering: samples the parameter space, separates the samples that satisfy
  a behaviour and compares the distributions (Smirnov test) and the correlations within each
  group.
- ``stability_map``: maps, in the parameter space, where the model has a solution, where it is
  mean-square stable and, optionally, where there is more than one stable solution.

Convention: samples are (n, n_par) matrices, one row per draw (RISE uses columns).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

PROCEDURES = ("sobol", "halton", "latin_hypercube", "uniform")


def sample(lb, ub, n: int, procedure: str = "sobol", scramble: bool = True, seed=0) -> np.ndarray:
    """Draw ``n`` points in ``[lb, ub]``. Returns an (n, n_par) matrix.

    ``procedure``: 'sobol' (2^m points are generated and the first ``n`` returned; use a power
    of 2 to keep the balance), 'halton', 'latin_hypercube' or 'uniform'. RISE's Sobol and Halton
    sequences are not scipy's (different order and direction numbers); comparisons with RISE
    use its samples.
    """
    lb, ub = np.atleast_1d(np.asarray(lb, float)), np.atleast_1d(np.asarray(ub, float))
    if lb.shape != ub.shape or np.any(ub < lb):
        raise ValueError("inconsistent bounds")
    d = lb.size
    if procedure == "sobol":
        eng = qmc.Sobol(d, scramble=scramble, seed=seed)
        m = int(np.ceil(np.log2(max(n, 1))))
        u = eng.random_base2(m)[:n] if n > 0 else np.zeros((0, d))
    elif procedure == "halton":
        u = qmc.Halton(d, scramble=scramble, seed=seed).random(n)
    elif procedure == "latin_hypercube":
        u = qmc.LatinHypercube(d, seed=seed).random(n)
    elif procedure == "uniform":
        u = np.random.default_rng(seed).uniform(size=(n, d))
    else:
        raise ValueError(f"unknown procedure: {procedure!r}; options: {PROCEDURES}")
    return lb + u * (ub - lb)


# --------------------------------------------------------------------------- test functions
def ishigami(theta, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    """Ishigami function on [-pi, pi]^3 (the same as ``ishigami.m`` in the RISE example)."""
    x = np.atleast_2d(theta)
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def ishigami_indices(a: float = 7.0, b: float = 0.1) -> dict:
    """Analytical variance and Sobol indices of the Ishigami function."""
    pi = np.pi
    V = a ** 2 / 8 + b * pi ** 4 / 5 + b ** 2 * pi ** 8 / 18 + 0.5
    V1 = 0.5 * (1 + b * pi ** 4 / 5) ** 2
    V2 = a ** 2 / 8
    V13 = b ** 2 * pi ** 8 * (1 / 18 - 1 / 50)
    return {"mean": a / 2, "variance": V, "S": {(0,): V1 / V, (1,): V2 / V, (2,): 0.0,
                                               (0, 1): 0.0, (0, 2): V13 / V, (1, 2): 0.0}}


SOBOL_G_A = (4.5, 4.5, 1.0, 0.0, 1.0, 9.0, 0.0, 9.0)   # ``satelli_sobol95.m`` in the RISE example


def sobol_g(theta, a=SOBOL_G_A) -> np.ndarray:
    """Sobol g function on [0, 1]^d (the one in ``satelli_sobol95.m`` in the RISE example)."""
    x = np.atleast_2d(theta)
    a = np.asarray(a, float)
    return np.prod((np.abs(4 * x - 2) + a) / (1 + a), axis=1)


def sobol_g_indices(a=SOBOL_G_A) -> dict:
    """Analytical variance and indices of the g function: ``V_i = (1/3)/(1+a_i)^2``, ``V = prod(1+V_i) - 1``."""
    a = np.asarray(a, float)
    Vi = (1 / 3) / (1 + a) ** 2
    V = np.prod(1 + Vi) - 1
    S = {(i,): Vi[i] / V for i in range(a.size)}
    S.update({(i, j): Vi[i] * Vi[j] / V for i, j in itertools.combinations(range(a.size), 2)})
    total = np.array([Vi[i] * np.prod(1 + np.delete(Vi, i)) / V for i in range(a.size)])
    return {"mean": 1.0, "variance": V, "S": S, "total": total}


# --------------------------------------------------------------------------- HDMR
def _legendre_orthonormal(x, order):
    """Shifted Legendre polynomials, orthonormal on [0, 1]: columns of degree 1 to ``order``."""
    z = 2 * np.asarray(x, float) - 1
    P = np.polynomial.legendre.legvander(z, order)[:, 1:]
    return P * np.sqrt(2 * np.arange(1, order + 1) + 1)


class HDMR:
    """High-dimensional model representation (RS-HDMR) of a scalar function of the parameters.

    ``objective``: function ``f(theta)`` that takes the (n, n_par) matrix and returns the (n,)
    vector, or the pair ``(f, theta)`` already computed. ``bounds``: (n_par, 2) matrix with lower
    and upper bounds. ``expansion_order``: 1, 2 or 3 (highest interaction order). ``pol_order``:
    maximum degree of the polynomials. ``optimal=True`` orthonormalises the polynomials with
    respect to the sample itself (RISE's "optimal polynomials"); with ``False`` it uses the
    Legendre ones, orthonormal under the uniform distribution.

    ``estimate(method)``: 'regression' (least squares on all terms), 'projection' (each
    coefficient is the sample mean of ``f * phi``, the Monte Carlo estimator of RS-HDMR) or
    'auto' (regression when there are at least two observations per coefficient).
    """

    def __init__(self, objective, bounds, names=None, expansion_order: int = 2, pol_order: int = 4,
                 optimal: bool = True, n: int = 1024, procedure: str = "sobol", seed=0,
                 sample_percentage: float = 1.0):
        self.bounds = np.asarray(bounds, float).reshape(-1, 2)
        self.lb, self.ub = self.bounds[:, 0], self.bounds[:, 1]
        self.npar = self.lb.size
        if isinstance(objective, (tuple, list)):
            f, theta = objective
            self.theta = np.asarray(theta, float)
            self.f = np.asarray(f, float).ravel()
        else:
            self.theta = sample(self.lb, self.ub, n, procedure, seed=seed)
            self.f = np.asarray(objective(self.theta), float).ravel()
        if self.theta.shape != (self.f.size, self.npar):
            raise ValueError("theta must be (n, n_par), one row per draw, and f must have n elements")
        if not 1 <= expansion_order <= 3:
            raise ValueError("expansion_order must be 1, 2 or 3")
        self.names = list(names) if names is not None else [f"p{i + 1}" for i in range(self.npar)]
        self.expansion_order = min(expansion_order, self.npar)
        self.pol_order = pol_order
        self.optimal = optimal
        # as in RISE (which uses 0.75), only the first fraction of the sample enters the estimation;
        # the rest is kept for the out-of-sample check (``fit(outofsample=True)``)
        cut = max(1, int(round(sample_percentage * self.f.size)))
        self.theta_out, self.f_out = self.theta[cut:], self.f[cut:]
        self.theta, self.f = self.theta[:cut], self.f[:cut]
        self.N = self.f.size
        self._R = None            # (n_par, pol+1, pol+1): change of basis of the optimal polynomials
        self.coefficients = None

    # ---- polynomial basis
    def _unit(self, theta):
        return (np.asarray(theta, float) - self.lb) / (self.ub - self.lb)

    def _fit_basis(self, x):
        if not self.optimal:
            return
        self._R = np.zeros((self.npar, self.pol_order + 1, self.pol_order + 1))
        for i in range(self.npar):
            V = np.vander(2 * x[:, i] - 1, self.pol_order + 1, increasing=True)
            _, R = np.linalg.qr(V)
            R = R * np.sign(np.diag(R))[:, None]
            self._R[i] = R / np.sqrt(x.shape[0])

    def _phi(self, x):
        """List with, for each parameter, the (n, pol_order) matrix of the polynomials evaluated at x."""
        out = []
        for i in range(self.npar):
            if self.optimal:
                V = np.vander(2 * x[:, i] - 1, self.pol_order + 1, increasing=True)
                out.append(np.linalg.solve(self._R[i].T, V.T).T[:, 1:])
            else:
                out.append(_legendre_orthonormal(x[:, i], self.pol_order))
        return out

    def _design(self, x):
        phi = self._phi(x)
        cols, terms = [], []
        for order in range(1, self.expansion_order + 1):
            for combo in itertools.combinations(range(self.npar), order):
                block = phi[combo[0]]
                for i in combo[1:]:
                    block = (block[:, :, None] * phi[i][:, None, :]).reshape(x.shape[0], -1)
                cols.append(block)
                terms.append(combo)
        return cols, terms

    # ---- estimation
    def estimate(self, method: str = "auto") -> "HDMR":
        x = self._unit(self.theta)
        self._fit_basis(x)
        cols, terms = self._design(x)
        X = np.hstack(cols)
        self.f0 = float(self.f.mean())
        self.D = float(self.f.var())
        y = self.f - self.f0
        if method == "auto":
            method = "regression" if self.N >= 2 * (X.shape[1] + 1) else "projection"
        if method == "regression":
            Xc = np.column_stack([np.ones(self.N), X])
            beta = np.linalg.lstsq(Xc, self.f, rcond=None)[0]
            self.f0, coef = float(beta[0]), beta[1:]
        elif method == "projection":
            coef = X.T @ y / self.N
        else:
            raise ValueError("method must be 'auto', 'regression' or 'projection'")
        self.method = method
        self.terms = terms
        self.coefficients, pos = {}, 0
        for t, block in zip(terms, cols):
            self.coefficients[t] = coef[pos:pos + block.shape[1]]
            pos += block.shape[1]
        fitted = self.f0 + X @ coef
        self.r2 = float(1 - np.sum((self.f - fitted) ** 2) / np.sum((self.f - self.f.mean()) ** 2))
        return self

    def _check(self):
        if self.coefficients is None:
            raise RuntimeError("call estimate() first")

    def label(self, term) -> str:
        return ",".join(self.names[i] for i in term)

    @property
    def indices(self) -> pd.DataFrame:
        """Partial variance and sensitivity index of each component (rows sum to the total ``S``)."""
        self._check()
        rows = [(self.label(t), len(t), float(np.sum(c ** 2)), float(np.sum(c ** 2)) / self.D)
                for t, c in self.coefficients.items()]
        return pd.DataFrame(rows, columns=["term", "order", "partial_variance", "sensitivity"]).set_index("term")

    def sensitivity(self, *params) -> float:
        """Index of the component formed by the given parameters (names or 0-based positions)."""
        self._check()
        t = tuple(sorted(self.names.index(p) if isinstance(p, str) else int(p) for p in params))
        return float(np.sum(self.coefficients[t] ** 2)) / self.D

    def total_effects(self) -> pd.Series:
        """Sum of the indices of all components in which each parameter appears (up to the expanded order)."""
        self._check()
        out = np.zeros(self.npar)
        for t, c in self.coefficients.items():
            for i in t:
                out[i] += np.sum(c ** 2) / self.D
        return pd.Series(out, index=self.names, name="total")

    def metamodel(self, theta) -> np.ndarray:
        """Evaluate the approximation at new points (n, n_par)."""
        self._check()
        cols, _ = self._design(self._unit(np.atleast_2d(theta)))
        return self.f0 + np.hstack(cols) @ np.concatenate(list(self.coefficients.values()))

    def first_order_effect(self, param, grid=None) -> pd.Series:
        """First-order component ``f_i(theta_i)`` on a grid of the parameter (in original units)."""
        self._check()
        i = self.names.index(param) if isinstance(param, str) else int(param)
        g = np.linspace(self.lb[i], self.ub[i], 101) if grid is None else np.asarray(grid, float)
        x = np.tile(0.5, (g.size, self.npar))
        x[:, i] = (g - self.lb[i]) / (self.ub[i] - self.lb[i])
        return pd.Series(self._phi(x)[i] @ self.coefficients[(i,)], index=pd.Index(g, name=self.names[i]),
                         name=f"f_{self.names[i]}")

    def fit(self, theta=None, f=None, outofsample: bool = False) -> dict:
        """R² and root mean squared error: in sample, on the held-out part (``outofsample``) or at
        new points (``theta`` and ``f``)."""
        self._check()
        if outofsample:
            if not self.f_out.size:
                raise ValueError("there is no held-out sample; use sample_percentage < 1")
            theta, f = self.theta_out, self.f_out
        theta = self.theta if theta is None else np.atleast_2d(theta)
        f = self.f if f is None else np.asarray(f, float).ravel()
        e = f - self.metamodel(theta)
        return {"r2": float(1 - np.sum(e ** 2) / np.sum((f - f.mean()) ** 2)), "rmse": float(np.sqrt(np.mean(e ** 2)))}


def sobol_indices(func, bounds, n: int = 1024, seed=0) -> pd.DataFrame:
    """First-order and total Sobol indices by the estimator of Saltelli et al. (2010).

    ``func`` takes the (n, n_par) matrix. Uses ``scipy.stats.sobol_indices`` with scrambled Sobol
    samples; cost of ``n (n_par + 2)`` evaluations. Serves as a check on the HDMR.
    """
    b = np.asarray(bounds, float).reshape(-1, 2)
    dists = [stats.uniform(loc=lo, scale=hi - lo) for lo, hi in b]
    res = stats.sobol_indices(func=lambda x: func(x.T), n=n, dists=dists, rng=np.random.default_rng(seed))
    return pd.DataFrame({"first_order": np.ravel(res.first_order), "total_order": np.ravel(res.total_order)})


# --------------------------------------------------------------------------- Monte Carlo filtering
class MCF:
    """Monte Carlo filtering.

    ``check_behavior(x)`` takes a parameter vector and returns true or false (or a tuple whose
    first element is the indicator; the others are stored in ``user_outputs``).
    ``nsim_or_draws``: number of draws or an (n, n_par) matrix of ready-made draws.
    """

    def __init__(self, check_behavior, nsim_or_draws, lb=None, ub=None, names=None,
                 procedure: str = "uniform", seed=0):
        if np.ndim(nsim_or_draws) == 2:
            self.samples = np.asarray(nsim_or_draws, float)
            self.lb = self.samples.min(0) if lb is None else np.asarray(lb, float)
            self.ub = self.samples.max(0) if ub is None else np.asarray(ub, float)
        else:
            self.lb, self.ub = np.asarray(lb, float), np.asarray(ub, float)
            self.samples = sample(self.lb, self.ub, int(nsim_or_draws), procedure, seed=seed)
        self.nsim, self.nparam = self.samples.shape
        self.names = list(names) if names is not None else [f"p{i + 1}" for i in range(self.nparam)]
        self.procedure = procedure
        flags, extra = np.zeros(self.nsim, bool), []
        for k, x in enumerate(self.samples):
            res = check_behavior(x)
            if isinstance(res, tuple):
                flags[k] = bool(res[0])
                extra.append(res[1:])
            else:
                flags[k] = bool(res)
        self.is_behaved = flags
        self.user_outputs = extra

    @property
    def share_behaved(self) -> float:
        return float(self.is_behaved.mean())

    def _group(self, which):
        if which in ("behave", "behaved", True):
            return self.samples[self.is_behaved]
        if which in ("non-behave", "non-behaved", False):
            return self.samples[~self.is_behaved]
        raise ValueError("use 'behave' or 'non-behave'")

    def smirnov(self) -> pd.DataFrame:
        """Smirnov test (two samples) of equality of the distributions of each parameter in the two groups.

        A small p-value indicates that the parameter matters for the behaviour.
        """
        a, b = self._group("behave"), self._group("non-behave")
        rows = []
        for i, nm in enumerate(self.names):
            if a.shape[0] and b.shape[0]:
                r = stats.ks_2samp(a[:, i], b[:, i])
                rows.append((nm, r.statistic, r.pvalue))
            else:
                rows.append((nm, np.nan, np.nan))
        return pd.DataFrame(rows, columns=["parameter", "statistic", "pvalue"]).set_index("parameter")

    def cdf(self, param) -> pd.DataFrame:
        """Empirical cumulative distributions of the parameter in the two groups, on a common grid."""
        i = self.names.index(param) if isinstance(param, str) else int(param)
        grid = np.sort(self.samples[:, i])
        out = {}
        for nm in ("behave", "non-behave"):
            g = np.sort(self._group(nm)[:, i])
            out[nm] = np.searchsorted(g, grid, side="right") / g.size if g.size else np.full(grid.size, np.nan)
        return pd.DataFrame(out, index=pd.Index(grid, name=self.names[i]))

    def correlation_patterns(self, which="behave", cutoff: float | None = None) -> pd.DataFrame:
        """Correlations between parameters within a group, with p-value.

        ``cutoff``: keeps only the pairs with p-value below the cutoff (significant correlations,
        which indicate combinations of parameters that generate the behaviour).
        """
        g = self._group(which)
        rows = []
        for i, j in itertools.combinations(range(self.nparam), 2):
            if g.shape[0] > 2:
                r, p = stats.pearsonr(g[:, i], g[:, j])
            else:
                r, p = np.nan, np.nan
            if cutoff is None or (np.isfinite(p) and p < cutoff):
                rows.append((self.names[i], self.names[j], r, p))
        return pd.DataFrame(rows, columns=["param_1", "param_2", "correlation", "pvalue"])

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.samples, columns=self.names)
        df["behaved"] = self.is_behaved
        return df


# --------------------------------------------------------------------------- stability map
@dataclass
class StabilityMap:
    draws: pd.DataFrame      # drawn parameters and columns 'status', 'mss_radius', 'n_stable'
    mcf: MCF                 # behaviour = status 'stable' (or 'determinate' with alternatives)

    def shares(self) -> pd.Series:
        return self.draws["status"].value_counts(normalize=True)

    def smirnov(self) -> pd.DataFrame:
        return self.mcf.smirnov()


def stability_map(model, bounds: dict, n: int = 500, procedure: str = "latin_hypercube", seed=0,
                  alternatives: int = 0, solve_options: dict | None = None) -> StabilityMap:
    """Map the region of solution, stability and determinacy in the parameter space.

    ``bounds``: ``{"name": (min, max)}``; for a regime-specific parameter,
    ``{"phi(pol,1)": (min, max)}`` or the key ``("phi", 1)``. Each draw is classified as:

    - ``fail``: steady state or solution not found;
    - ``unstable``: solution found, but not mean-square stable;
    - ``stable``: mean-square stable solution (with the solver's default guess);
    - with ``alternatives=k > 0`` (k random guesses per point, see ``solve_alternatives``):
      ``determinate`` if there is exactly one stable solution, ``indeterminate`` if there is more
      than one, ``unstable`` if none.

    In regime-switching models, the algorithms return one solution per guess; the count of
    stable solutions is a lower bound.
    """
    from .model import SteadyStateError
    from .solve import SolveError

    keys = list(bounds)
    lb = np.array([bounds[k][0] for k in keys], float)
    ub = np.array([bounds[k][1] for k in keys], float)
    draws = sample(lb, ub, n, procedure, seed=seed)
    labels = [_key_label(k) for k in keys]
    opts = dict(solve_options or {})
    status, radius, nst = [], np.full(n, np.nan), np.zeros(n, int)

    def overrides(x):
        ov: dict = {}
        for k, v in zip(keys, x):
            name, reg = _key_split(k)
            if reg is None:
                ov[name] = float(v)
            else:
                base = dict(model.spec.calibration.get(name) or {}) if isinstance(model.spec.calibration.get(name), dict) else {}
                ov.setdefault(name, base)[reg] = float(v)
        return ov

    for k, x in enumerate(draws):
        try:
            if alternatives > 0:
                sols = model.solve_alternatives(overrides(x), starts=alternatives, **opts)
                if not sols:
                    status.append("fail")
                    continue
                stable = [s for s in sols if s.is_stable]
                nst[k] = len(stable)
                radius[k] = min(s.mss_radius for s in sols)
                status.append("determinate" if len(stable) == 1 else "indeterminate" if stable else "unstable")
            else:
                s = model.solve(overrides(x), **opts)
                radius[k] = s.mss_radius
                nst[k] = int(s.is_stable)
                status.append("stable" if s.is_stable else "unstable")
        except (SolveError, SteadyStateError, np.linalg.LinAlgError, ValueError, FloatingPointError):
            status.append("fail")
    df = pd.DataFrame(draws, columns=labels)
    df["status"], df["mss_radius"], df["n_stable"] = status, radius, nst
    good = "determinate" if alternatives > 0 else "stable"
    flags = iter(df["status"].eq(good).to_numpy())
    mcf = MCF(lambda _x: next(flags), draws, lb, ub, labels, procedure)
    return StabilityMap(draws=df, mcf=mcf)


def _key_split(key):
    if isinstance(key, tuple):
        return key[0], int(key[-1])
    if "(" in key:
        name, rest = key.split("(", 1)
        return name.strip(), int(rest.rstrip(")").split(",")[-1])
    return key, None


def _key_label(key) -> str:
    name, reg = _key_split(key)
    return name if reg is None else f"{name}_{reg}"
