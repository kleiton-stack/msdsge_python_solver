"""Estimation: priors, posterior mode, Metropolis-Hastings and marginal data density.

The estimated parameters come from the ``estimated_params`` block (same field order as Dynare).
With no prior on any parameter, estimation is by maximum likelihood.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .model import SteadyStateError
from .solve import SolveError, solve_first_order


# ---------------------------------------------------------------- priors
def _inv_gamma1_hyper(mean: float, std: float) -> tuple[float, float]:
    """(s, nu) of the type-1 inverse gamma (X = sqrt(Z), Z ~ IG2(s, nu)) with the given mean and std."""
    def ratio(nu):  # E[X]^2 / E[X^2], increasing in nu
        return np.exp(2 * (special.gammaln((nu - 1) / 2) - special.gammaln(nu / 2))) * (nu - 2) / 2
    target = mean ** 2 / (mean ** 2 + std ** 2)
    nu = optimize.brentq(lambda v: ratio(v) - target, 2 + 1e-8, 1e6)
    return (mean ** 2 + std ** 2) * (nu - 2), nu


_TWO_PAR = {
    # shape -> (function returning the distribution from 2 positive hyperparameters, guess from mean and std)
    "normal": lambda a, b: stats.norm(a, b),
    "beta": lambda a, b: stats.beta(a, b),
    "gamma": lambda a, b: stats.gamma(a=a, scale=b),
    "inv_gamma": lambda a, b: stats.invgamma(a=a, scale=b),
    "inv_gamma2": lambda a, b: stats.invgamma(a=a, scale=b),
    "lognormal": lambda a, b: stats.lognorm(s=b, scale=np.exp(a)),
    "weibull": lambda a, b: stats.weibull_min(c=a, scale=b),
    "uniform": lambda a, b: stats.uniform(a, b - a),
}


@dataclass
class Prior:
    """Prior of a parameter, defined by mean and std or by two quantiles (``prob``).

    In the quantile form (the RISE one), ``mean`` and ``std`` hold the lower and upper quantiles,
    with mass ``prob`` between them and equal tails. In that form, ``inv_gamma`` is the usual
    inverse gamma applied directly to the parameter.
    """
    shape: str
    mean: float
    std: float
    prob: float | None = None
    _dist: object = field(init=False, repr=False)
    _ig1: tuple | None = field(init=False, default=None, repr=False)
    bounds: tuple | None = None            # parameter bounds (used by truncated_normal)

    def __post_init__(self):
        if self.prob is not None:
            self._from_quantiles()
            return
        m, s = self.mean, self.std
        if s <= 0:
            raise ValueError("the prior standard deviation must be positive")
        if self.shape == "normal":
            self._dist = stats.norm(m, s)
        elif self.shape == "beta":
            if not 0 < m < 1 or s ** 2 >= m * (1 - m):
                raise ValueError(f"infeasible beta prior with mean {m} and std {s}")
            k = m * (1 - m) / s ** 2 - 1
            self._dist = stats.beta(m * k, (1 - m) * k)
        elif self.shape == "gamma":
            self._dist = stats.gamma(a=(m / s) ** 2, scale=s ** 2 / m)
        elif self.shape == "uniform":
            self._dist = stats.uniform(m - np.sqrt(3) * s, 2 * np.sqrt(3) * s)
        elif self.shape == "lognormal":
            v = np.log(1 + (s / m) ** 2)
            self._dist = stats.lognorm(s=np.sqrt(v), scale=m * np.exp(-v / 2))
        elif self.shape == "weibull":
            k = optimize.brentq(lambda c: special.gamma(1 + 2 / c) / special.gamma(1 + 1 / c) ** 2 - 1 - (s / m) ** 2, 0.05, 500)
            self._dist = stats.weibull_min(c=k, scale=m / special.gamma(1 + 1 / k))
        elif self.shape == "inv_gamma2":      # distribution of a variance
            a = 2 + (m / s) ** 2
            self._dist = stats.invgamma(a=a, scale=m * (a - 1))
        elif self.shape == "inv_gamma":       # type 1: distribution of a standard deviation
            sc, nu = _inv_gamma1_hyper(m, s)
            self._ig1 = (sc, nu)
            self._dist = stats.invgamma(a=nu / 2, scale=sc / 2)   # distribution of X^2
        else:                                 # cauchy, laplace, logistic, pareto, triangular, truncated_normal
            from .priors_extra import make_prior
            self._dist = make_prior(self.shape, m, s, self.bounds)

    def _from_quantiles(self):
        lo, up, p = self.mean, self.std, self.prob
        if self.shape not in _TWO_PAR:
            from . import priors_extra
            if self.shape in priors_extra.EXTRA_SHAPES:
                self._dist = priors_extra.from_quantiles(self.shape, lo, up, p, self.bounds)
                self.mean, self.std = priors_extra.location_scale(self._dist)
                return
            raise ValueError(f"quantile prior not available for {self.shape!r}")
        make = _TWO_PAR[self.shape]
        tails = np.array([(1 - p) / 2, 1 - (1 - p) / 2])
        if self.shape == "uniform":
            w = (up - lo) / p
            self._dist = make(lo - w * (1 - p) / 2, up + w * (1 - p) / 2)
        elif self.shape == "normal":
            z = stats.norm.ppf(tails[1])
            self._dist = make((lo + up) / 2, (up - lo) / (2 * z))
        elif self.shape == "lognormal":
            z = stats.norm.ppf(tails[1])
            self._dist = make((np.log(lo) + np.log(up)) / 2, (np.log(up) - np.log(lo)) / (2 * z))
        else:
            mid, sd = (lo + up) / 2, (up - lo) / (2 * stats.norm.ppf(tails[1]))
            if self.shape == "beta":
                k = max(mid * (1 - mid) / sd ** 2 - 1, 1e-3)
                x0 = np.log([mid * k, (1 - mid) * k])
            elif self.shape == "gamma":
                x0 = np.log([(mid / sd) ** 2, sd ** 2 / mid])
            elif self.shape == "weibull":
                x0 = np.log([max(mid / sd, 0.5), mid])
            else:  # inverse gamma
                a0 = 2 + (mid / sd) ** 2
                x0 = np.log([a0, mid * (a0 - 1)])
            sol = optimize.least_squares(lambda h: (make(*np.exp(h)).ppf(tails) - [lo, up]) / (up - lo), x0,
                                         xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if np.max(np.abs(sol.fun)) > 1e-6:
                raise ValueError(f"could not find {self.shape} hyperparameters for the quantiles ({lo}, {up}) with mass {p}")
            self._dist = make(*np.exp(sol.x))
        self.mean, self.std = float(self._dist.mean()), float(self._dist.std())

    def logpdf(self, x: float) -> float:
        if self._ig1:
            return float(self._dist.logpdf(x * x) + np.log(2 * x)) if x > 0 else -np.inf
        return float(self._dist.logpdf(x))

    def support(self) -> tuple[float, float]:
        if self._ig1:
            return 0.0, np.inf
        return tuple(float(v) for v in self._dist.support())

    def rvs(self, rng) -> float:
        v = float(self._dist.rvs(random_state=rng))
        return np.sqrt(v) if self._ig1 else v


# ---------------------------------------------------------------- problem
class EstimationProblem:
    """Ties together parameter vector, solution, filter and prior."""

    def __init__(self, model, data: pd.DataFrame, filter: str = "rise", solver: str = "mfi",
                 check_stability: bool = True, measurement_errors=None, presample: int = 0,
                 solve_options: dict | None = None, dirichlet=None, endogenous_priors=None):
        self.model, self.data = model, data
        self.items = list(model.spec.estimated)
        if not self.items:
            raise ValueError("the model has no estimated_params block")
        self.names = [e.label for e in self.items]
        self.priors = [Prior(e.prior, e.mean, e.std, e.prob, bounds=(e.lower, e.upper)) if e.prior else None for e in self.items]
        self.filter, self.solver, self.check_stability = filter, solver, check_stability
        self.measurement_errors, self.presample = measurement_errors, presample
        self.solve_options = solve_options or {}
        if filter in ("ukf", "ckf", "ddf"):   # sigma-point filters (nonlinear_filters.py): 2nd-order solution
            self.solve_options = {"order": 2, **self.solve_options}
        lo, up = [], []
        for e, p in zip(self.items, self.priors):
            a, b = p.support() if p else (-np.inf, np.inf)
            lo.append(max(e.lower, a))
            up.append(min(e.upper, b))
        self.lower, self.upper = np.array(lo), np.array(up)
        self.bayesian = any(p is not None for p in self.priors)
        # joint priors (Dirichlet on the transition probabilities) and endogenous priors: see priors_extra.py
        self.joint_priors, self.endogenous_priors = [], endogenous_priors
        if dirichlet:
            from .priors_extra import DirichletPrior
            specs = dirichlet if isinstance(dirichlet[0], (list, tuple, dict)) else [dirichlet]
            self.joint_priors = [DirichletPrior.from_rise(self.names, d) for d in specs]
        self.bayesian = self.bayesian or bool(self.joint_priors) or endogenous_priors is not None

    def initial(self) -> np.ndarray:
        x = []
        for e, p in zip(self.items, self.priors):
            if e.init is not None:
                x.append(e.init)
            elif p is not None:
                x.append(p.mean)
            else:
                raise ValueError(f"'{e.label}' has neither an initial value nor a prior")
        return np.array(x, float)

    def _overrides(self, theta):
        params, stderr = {}, {}
        for e, v in zip(self.items, theta):
            if e.kind == "stderr":
                stderr[e.name] = float(v)
            elif e.regime:
                cur = params.get(e.name)
                if cur is None:
                    cal = self.model.spec.calibration.get(e.name, {})
                    cur = params[e.name] = dict(cal) if isinstance(cal, dict) else {}
                cur[e.regime[1]] = float(v)
            else:
                params[e.name] = float(v)
        return params, stderr

    def solve(self, theta):
        params, stderr = self._overrides(theta)
        return solve_first_order(self.model, self.model.parameter_values(params), solver=self.solver,
                                 shock_stderr=stderr, **self.solve_options)

    def loglik(self, theta) -> float:
        theta = np.asarray(theta, float)
        if np.any(theta < self.lower) or np.any(theta > self.upper) or not np.all(np.isfinite(theta)):
            return -np.inf
        try:
            sol = self.solve(theta)
        except (SolveError, SteadyStateError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            return -np.inf
        if self.check_stability and not sol.is_stable:
            return -np.inf
        res = sol.filter(self.data, filter=self.filter, measurement_errors=self.measurement_errors, smooth=False)
        ll = float(np.sum(res.loglik_t[self.presample:]))
        return ll if np.isfinite(ll) else -np.inf

    def logprior(self, theta) -> float:
        if np.any(theta < self.lower) or np.any(theta > self.upper):
            return -np.inf
        if self.model.spec.restrictions:   # parameter_restrictions block: zero prior outside the region
            try:
                if not self.model.check_restrictions(self.model.parameter_values(self._overrides(theta)[0])):
                    return -np.inf
            except (ValueError, SteadyStateError):
                return -np.inf
        return float(sum(p.logpdf(v) for p, v in zip(self.priors, theta) if p is not None)
                     + sum(j.logpdf(theta) for j in self.joint_priors))

    def logpost(self, theta) -> float:
        lp = self.logprior(np.asarray(theta, float))
        if not np.isfinite(lp):
            return -np.inf
        ll = self.loglik(theta)
        if self.endogenous_priors is not None and np.isfinite(ll):
            ll += self.endogenous_priors.logpdf(self, theta)
        return ll + lp if np.isfinite(ll) else -np.inf


# ---------------------------------------------------------------- mode
def numerical_hessian(f, x, rel_step: float = 1e-4) -> np.ndarray:
    """Hessian by central differences."""
    x = np.asarray(x, float)
    n = x.size
    hstep = rel_step * np.maximum(np.abs(x), 1e-2)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n); ei[i] = hstep[i]
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / hstep[i] ** 2
        for j in range(i):
            ej = np.zeros(n); ej[j] = hstep[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * hstep[i] * hstep[j])
    return H


@dataclass
class EstimationResult:
    problem: EstimationProblem
    mode: np.ndarray
    logpost_mode: float
    loglik_mode: float
    hessian: np.ndarray             # of -log posterior at the mode
    cov: np.ndarray
    hessian_pd: bool
    optimizer_message: str
    draws: np.ndarray | None = None        # (chains, iterations, n)
    acceptance: np.ndarray | None = None
    logpost_draws: np.ndarray | None = None

    @property
    def log_mdd_laplace(self) -> float:
        n = self.mode.size
        sign, logdet = np.linalg.slogdet(self.hessian)
        if sign <= 0:
            return np.nan
        return float(self.logpost_mode + 0.5 * n * np.log(2 * np.pi) - 0.5 * logdet)

    def log_mdd_mhm(self, p: float = 0.9) -> float:
        """Modified harmonic mean of Geweke (1999), with a normal truncated at probability p."""
        if self.draws is None:
            raise ValueError("run the MCMC first")
        x = self.draws.reshape(-1, self.mode.size)
        lp = self.logpost_draws.ravel()
        mu, S = x.mean(0), np.cov(x.T).reshape(self.mode.size, -1)
        Si = np.linalg.inv(S)
        d = np.einsum("ti,ij,tj->t", x - mu, Si, x - mu)
        inside = d <= stats.chi2.ppf(p, x.shape[1])
        logf = -np.log(p) - 0.5 * (x.shape[1] * np.log(2 * np.pi) + np.linalg.slogdet(S)[1] + d)
        terms = np.where(inside, logf - lp, -np.inf)
        return float(-(special.logsumexp(terms) - np.log(len(lp))))

    def sample(self, sampler: str = "mh", **options) -> "EstimationResult":
        """Samples the posterior: ``mh`` (adaptive), ``apt``, ``dsmh``, ``rrf`` or ``smc`` (see samplers.py)."""
        from .samplers import posterior_sample
        return posterior_sample(self, sampler, **options)

    def log_mdd(self, algorithm: str = "mhm", L: int = 5000, seed: int = 0, **options) -> float:
        """Marginal data density by any estimator in mdd.py (``mhm``, ``swz``, ``mueller``, ``bridge``,
        ``is``, ``ris``, ``cj``, ``laplace``, ``laplace_mcmc``, ``ce``)."""
        from .mdd import from_result
        return from_result(self, L=L, seed=seed).estimate(algorithm, **options)

    def rhat(self) -> pd.Series:
        """Gelman-Rubin diagnostic per parameter (needs 2 or more chains)."""
        d = self.draws
        m, n = d.shape[0], d.shape[1]
        W = d.var(axis=1, ddof=1).mean(0)
        B = n * d.mean(axis=1).var(axis=0, ddof=1)
        return pd.Series(np.sqrt(((n - 1) / n * W + B / n) / W), index=self.problem.names)

    def summary(self, level: float = 0.90) -> pd.DataFrame:
        pr = self.problem
        rows = []
        sd = np.sqrt(np.clip(np.diag(self.cov), 0, None))
        for k, (e, p) in enumerate(zip(pr.items, pr.priors)):
            row = {"parameter": pr.names[k], "prior": p.shape if p else "", "prior_mean": p.mean if p else np.nan,
                   "prior_std": p.std if p else np.nan, "mode": self.mode[k], "std_hessian": sd[k]}
            if self.draws is not None:
                x = self.draws[:, :, k].ravel()
                q = np.quantile(x, [(1 - level) / 2, 0.5, 1 - (1 - level) / 2])
                row.update(post_mean=x.mean(), post_median=q[1], ci_low=q[0], ci_high=q[2])
            rows.append(row)
        return pd.DataFrame(rows).set_index("parameter")


def estimate(model, data: pd.DataFrame, x0=None, optimizer: str = "L-BFGS-B", maxiter: int = 2000,
             restarts: int = 0, seed: int = 0, optimizer_options: dict | None = None,
             **problem_options) -> EstimationResult:
    """Posterior mode (or maximum likelihood, without priors) and Hessian at the mode.

    ``optimizer`` accepts the methods of ``scipy.optimize.minimize`` and those in ``optimizers.OPTIMIZERS``
    (``bee_gate``, ``blockwise``, ``fmincon``, ``fminunc``, ``csminwel``, ``fminsearch``,
    ``differential_evolution``, ``dual_annealing``). ``restarts`` > 0 repeats the optimisation from
    points drawn from the prior (or around ``x0``, without a prior) and keeps the best.
    """
    # problem_class: subclass of EstimationProblem with a different likelihood (e.g. dsge_var.DSGEVARProblem)
    pr = problem_options.pop("problem_class", EstimationProblem)(model, data, **problem_options)
    x0 = pr.initial() if x0 is None else np.asarray(x0, float)

    def obj(th):
        v = pr.logpost(th)
        return -v if np.isfinite(v) else 1e12
    f0 = pr.logpost(x0)
    if not np.isfinite(f0):
        raise ValueError("the posterior is not finite at the initial point (solution, stability or prior)")
    eps = 1e-8
    bounds = [(lo + eps * (1 + abs(lo)) if np.isfinite(lo) else None, up - eps * (1 + abs(up)) if np.isfinite(up) else None)
              for lo, up in zip(pr.lower, pr.upper)]
    from . import optimizers as _opt
    blo = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
    bup = np.array([np.inf if b[1] is None else b[1] for b in bounds])

    def _run(start, k=0):
        if optimizer in _opt.OPTIMIZERS:
            return _opt.minimize(obj, start, blo, bup, method=optimizer, maxiter=maxiter, seed=seed + k,
                                 **(optimizer_options or {}))
        return optimize.minimize(obj, start, method=optimizer, bounds=bounds,
                                 options={"maxiter": maxiter, **(optimizer_options or {})})
    res = _run(x0)
    if restarts > 0:
        draw = None
        if all(p is not None for p in pr.priors) and not pr.joint_priors:
            draw = lambda rng: np.clip([p.rvs(rng) for p in pr.priors], blo, bup)
        counter = iter(range(1, restarts + 2))
        ms = _opt.multistart(obj, res.x, blo, bup, starts=restarts + 1, seed=seed, draw=draw,
                             optimizer=lambda f, s: _run(s, next(counter)))
        if ms.fun < res.fun:
            res = ms
    # derivative-free polishing, which copes better with regions where the solution fails
    res2 = optimize.minimize(obj, res.x, method="Nelder-Mead", bounds=bounds,
                             options={"maxiter": maxiter, "xatol": 1e-7, "fatol": 1e-9})
    best = res2 if res2.fun <= res.fun else res
    H = numerical_hessian(obj, best.x)
    H = (H + H.T) / 2
    pd_ok = bool(np.all(np.isfinite(H)) and np.all(np.linalg.eigvalsh(H) > 0))
    cov = np.linalg.inv(H) if pd_ok else np.full_like(H, np.nan)
    return EstimationResult(problem=pr, mode=best.x, logpost_mode=-float(best.fun), loglik_mode=pr.loglik(best.x),
                            hessian=H, cov=cov, hessian_pd=pd_ok, optimizer_message=str(best.message))


def mcmc(result: EstimationResult, draws: int = 5000, chains: int = 2, burn: int = 1000, scale: float = 0.3,
         seed: int = 0, cov=None) -> EstimationResult:
    """Random-walk Metropolis-Hastings, starting from the mode, with proposal N(0, scale² Σ)."""
    pr = result.problem
    S = result.cov if cov is None else np.asarray(cov, float)
    if not np.all(np.isfinite(S)):
        raise ValueError("invalid proposal covariance (the Hessian is not positive definite); provide 'cov'")
    L = np.linalg.cholesky(S) * scale
    n = result.mode.size
    out = np.zeros((chains, draws, n))
    lps = np.zeros((chains, draws))
    acc = np.zeros(chains)
    for c, ss in enumerate(np.random.SeedSequence(seed).spawn(chains)):
        rng = np.random.default_rng(ss)
        x, lp = result.mode.copy(), result.logpost_mode
        for it in range(burn + draws):
            cand = x + L @ rng.standard_normal(n)
            lpc = pr.logpost(cand)
            if np.log(rng.uniform()) < lpc - lp:
                x, lp = cand, lpc
                acc[c] += it >= burn
            if it >= burn:
                out[c, it - burn], lps[c, it - burn] = x, lp
    result.draws, result.logpost_draws, result.acceptance = out, lps, acc / draws
    return result
