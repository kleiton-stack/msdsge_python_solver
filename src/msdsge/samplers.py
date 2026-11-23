"""Posterior samplers.

All of them take ``logpost(theta) -> float`` (log of the posterior kernel, ``-inf`` outside the
support), optional bounds and a fixed seed, and return a ``SamplerResult`` with ``draws`` of shape
(chains, iterations, n).

- ``mh_sampler``: adaptive random-walk Metropolis-Hastings, with the options of RISE's
  ``mh_sampler``. Scale and covariance are updated with formulas 20, 23 and 24 of
  Miasojedow, Moulines and Vihola (2013), the same ones cited in the help of RISE's
  ``utils.mcmc.update_scaling`` and ``update_moments``.
- ``apt_sampler``: adaptive parallel tempering (Miasojedow, Moulines and Vihola, 2013).
- ``dsmh_sampler``: dynamic striated Metropolis-Hastings (Waggoner, Wu and Zha, 2016).
- ``rrf_sampler``: ``dsmh_sampler`` with the defaults of RISE's "Rapid Reaction Force Sampler",
  which has neither a paper nor readable code; it is equivalent in family (staged tempering with
  bins by density level), not a replica.
- ``smc_sampler``: sequential Monte Carlo with likelihood tempering (Herbst and Schorfheide, 2014).

``dsmh``, ``rrf`` and ``smc`` also return the log marginal data density.
``posterior_sample`` connects any of them to an ``EstimationResult``.

Parallelism (``parallel=True`` or a number of processes) uses ``multiprocessing`` with ``fork``;
where ``fork`` does not exist, it runs serially. Results do not depend on the number of processes.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import special, stats

__all__ = ["SamplerResult", "mh_sampler", "apt_sampler", "dsmh_sampler", "rrf_sampler", "smc_sampler",
           "smc_from_kernel", "posterior_sample", "InitialDensity"]


@dataclass
class SamplerResult:
    draws: np.ndarray               # (chains, iterations, n)
    logpost: np.ndarray             # (chains, iterations)
    acceptance: np.ndarray          # per chain
    sampler: str
    log_mdd: float | None = None
    extras: dict = field(default_factory=dict)

    @property
    def flat(self) -> np.ndarray:
        return self.draws.reshape(-1, self.draws.shape[-1])


# ---------------------------------------------------------------- utilities
_TASK = None


def _call(i):
    return _TASK(i)


def _workers(parallel) -> int:
    if not parallel:
        return 1
    return (os.cpu_count() or 1) if parallel is True else int(parallel)


def pmap(fn, n: int, parallel=False) -> list:
    """``[fn(0), ..., fn(n-1)]``, in separate processes if ``parallel``. ``fn`` may be a closure."""
    global _TASK
    w = min(_workers(parallel), n)
    if w <= 1 or "fork" not in mp.get_all_start_methods() or _TASK is not None:
        return [fn(i) for i in range(n)]
    _TASK = fn
    try:
        with mp.get_context("fork").Pool(w) as pool:
            return pool.map(_call, range(n), chunksize=max(1, n // (4 * w)))
    finally:
        _TASK = None


def _bounds(n, lower, upper):
    lo = np.full(n, -np.inf) if lower is None else np.asarray(lower, float)
    up = np.full(n, np.inf) if upper is None else np.asarray(upper, float)
    return lo, up


def _guard(logpost, lo, up):
    def f(x):
        if np.any(x < lo) or np.any(x > up):
            return -np.inf
        v = logpost(x)
        return float(v) if np.isfinite(v) else -np.inf
    return f


def _chol(S, jitter: float = 1e-10):
    S = (S + S.T) / 2
    d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
    for k in range(12):
        try:
            return np.linalg.cholesky(S + jitter * 10.0 ** k * np.outer(d, d) * np.eye(len(d)))
        except np.linalg.LinAlgError:
            continue
    return np.diag(d)


def _target(alpha) -> float:
    return float(np.mean(alpha))


def _start(f, x0, L, rng, dispersion):
    """Starting point with finite posterior: x0, or x0 plus a perturbation N(0, dispersion² Σ)."""
    if dispersion > 0:
        for _ in range(1000):
            x = x0 + dispersion * (L @ rng.standard_normal(x0.size))
            lp = f(x)
            if np.isfinite(lp):
                return x, lp
    lp = f(x0)
    if not np.isfinite(lp):
        raise ValueError("posterior is not finite at the starting point")
    return x0.copy(), lp


def weighted_moments(x, w):
    w = w / w.sum()
    mu = w @ x
    d = x - mu
    return mu, (d * w[:, None]).T @ d


def systematic_resample(w, rng, size=None) -> np.ndarray:
    n = len(w) if size is None else size
    u = (rng.uniform() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w / w.sum()), u), len(w) - 1)


class InitialDensity:
    """Multivariate Student t centred at ``mean``, truncated at the bounds (normalised density).

    It is the initial distribution of ``dsmh_sampler`` and of ``smc_from_kernel``.
    """

    def __init__(self, mean, cov, lower=None, upper=None, df: float = 5.0, seed: int = 0, mass_draws: int = 20000):
        self.mean = np.asarray(mean, float)
        n = self.mean.size
        self.lo, self.up = _bounds(n, lower, upper)
        self.df = df
        self.L = _chol(np.asarray(cov, float))
        self._logdet = 2 * np.sum(np.log(np.diag(self.L)))
        self.log_mass = 0.0
        if np.any(np.isfinite(self.lo)) or np.any(np.isfinite(self.up)):
            x = self._raw(np.random.default_rng([seed, 991]), mass_draws)
            frac = np.mean(np.all((x >= self.lo) & (x <= self.up), axis=1))
            if frac <= 0:
                raise ValueError("the initial distribution has no mass inside the bounds")
            self.log_mass = float(np.log(frac))

    def _raw(self, rng, size):
        n = self.mean.size
        z = rng.standard_normal((size, n))
        g = rng.chisquare(self.df, size) / self.df if np.isfinite(self.df) else np.ones(size)
        return self.mean + (z / np.sqrt(g)[:, None]) @ self.L.T

    def rvs(self, rng, size: int) -> np.ndarray:
        out = np.empty((0, self.mean.size))
        while len(out) < size:
            x = self._raw(rng, max(2 * (size - len(out)), 64))
            out = np.vstack([out, x[np.all((x >= self.lo) & (x <= self.up), axis=1)]])
        return out[:size]

    def logpdf(self, x) -> np.ndarray:
        x = np.atleast_2d(x)
        n = self.mean.size
        if not hasattr(self, "_Li"):
            self._Li = np.linalg.inv(self.L)
        z = (x - self.mean) @ self._Li.T
        q = np.sum(z * z, axis=1)
        if np.isfinite(self.df):
            v = self.df
            lp = (special.gammaln((v + n) / 2) - special.gammaln(v / 2) - 0.5 * n * np.log(v * np.pi)
                  - 0.5 * self._logdet - 0.5 * (v + n) * np.log1p(q / v))
        else:
            lp = -0.5 * (n * np.log(2 * np.pi) + self._logdet + q)
        inside = np.all((x >= self.lo) & (x <= self.up), axis=1)
        return np.where(inside, lp - self.log_mass, -np.inf)


# ---------------------------------------------------------------- adaptive Metropolis-Hastings
def _mh_chain(f, x0, cov, n_keep, burn, thin, rng, target, c, c_range, fixed_scaling, adapt_covariance,
              use_true_moments, rwm_exp, retune_cov_every, dispersion, temper: float = 1.0):
    n = x0.size
    L0 = _chol(cov)
    x, lp = _start(f, x0, L0, rng, dispersion)
    mu, S, L = x.copy(), np.array(cov, float), L0
    log_c = np.log(c)
    lcr = np.log(c_range)
    out, lps = np.empty((n_keep, n)), np.empty(n_keep)
    acc = 0
    total = burn + n_keep * thin
    for it in range(total):
        cand = x + np.exp(0.5 * log_c) * (L @ rng.standard_normal(n))
        lpc = f(cand)
        a = float(np.exp(min(0.0, temper * (lpc - lp)))) if np.isfinite(lpc) else 0.0
        if rng.uniform() < a:
            x, lp = cand, lpc
            acc += it >= burn
        gam = (it + 1.0) ** (-rwm_exp)
        if not fixed_scaling:
            log_c = float(np.clip(log_c + gam * (a - target), *lcr))
        if adapt_covariance:
            g2 = 1.0 / (it + 2.0) if use_true_moments else gam
            d = x - mu
            mu = mu + g2 * d
            S = S + g2 * (np.outer(d, d) - S)
            if (it + 1) % retune_cov_every == 0:
                L = _chol(S)
        if it >= burn and (it - burn) % thin == thin - 1:
            k = (it - burn) // thin
            out[k], lps[k] = x, lp
    return out, lps, acc / max(total - burn, 1), np.exp(log_c), S


def mh_sampler(logpost, x0, cov, lower=None, upper=None, draws: int = 20000, burn: int = 0, chains: int = 1,
               thin: int = 1, seed: int = 0, alpha=(0.25, 0.45), c: float = 0.25,
               c_range=(np.sqrt(np.finfo(float).eps), 100.0), fixed_scaling: bool = False,
               adapt_covariance: bool = False, use_true_moments: bool = False, rwm_exp: float = 0.6,
               retune_cov_every: int = 100, dispersion: float = 0.0, parallel=False) -> SamplerResult:
    """Adaptive random-walk Metropolis-Hastings, with proposal N(x, c Σ).

    Option names and defaults are those of RISE's ``mh_sampler`` (``alpha``, ``c``, ``c_range``,
    ``thin``, ``rwm_exp``, ``fixed_scaling``, ``adapt_covariance``, ``use_true_moments``,
    ``retune_cov_every``). At each iteration n, with γ_n = (n+1)^(-rwm_exp):

    - scale: log c ← log c + γ_n (a_n - α*), where a_n is the acceptance probability and α* the
      midpoint of ``alpha`` (RISE accepts an interval; its rule inside the interval is not
      documented);
    - covariance (if ``adapt_covariance``): μ ← μ + γ_n (x - μ), Σ ← Σ + γ_n ((x - μ)(x - μ)' - Σ),
      with γ_n = 1/(n+1) if ``use_true_moments``.

    Adaptation continues after burn-in with a decreasing step (vanishing adaptation, which
    preserves ergodicity). ``draws`` is the number of draws kept per chain;
    ``dispersion`` > 0 disperses the chains' starting points as N(x0, dispersion² Σ).
    """
    x0, cov = np.asarray(x0, float), np.atleast_2d(np.asarray(cov, float))
    lo, up = _bounds(x0.size, lower, upper)
    f = _guard(logpost, lo, up)
    seeds = np.random.SeedSequence(seed).spawn(chains)

    def run(k):
        return _mh_chain(f, x0, cov, draws, burn, thin, np.random.default_rng(seeds[k]), _target(alpha), c, c_range,
                         fixed_scaling, adapt_covariance, use_true_moments, rwm_exp, retune_cov_every, dispersion)
    res = pmap(run, chains, parallel)
    return SamplerResult(np.stack([r[0] for r in res]), np.stack([r[1] for r in res]), np.array([r[2] for r in res]),
                         "mh", extras={"c": np.array([r[3] for r in res]), "cov": [r[4] for r in res]})


# ---------------------------------------------------------------- adaptive parallel tempering
def _apt_chain(f, x0, cov, n_keep, burn, thin, rng, levels, target, swap_target, c, c_range, rwm_exp,
               retune_cov_every, adapt_temperatures, dispersion):
    n = x0.size
    L0 = _chol(cov)
    xs, lps = [], []
    for _ in range(levels):
        x, lp = _start(f, x0, L0, rng, dispersion)
        xs.append(x); lps.append(lp)
    xs, lps = np.array(xs), np.array(lps)
    mu, S = xs.copy(), np.array([cov.copy() for _ in range(levels)])
    Ls = np.array([L0.copy() for _ in range(levels)])
    log_c = np.full(levels, np.log(c))
    rho = np.zeros(levels - 1)                           # log of the temperature increments
    lam = 1.0 / np.concatenate([[1.0], 1.0 + np.cumsum(np.exp(rho))])
    out, olp = np.empty((n_keep, n)), np.empty(n_keep)
    acc, swaps, tries = 0, np.zeros(max(levels - 1, 1)), np.zeros(max(levels - 1, 1))
    total = burn + n_keep * thin
    lcr = np.log(c_range)
    for it in range(total):
        gam = (it + 1.0) ** (-rwm_exp)
        for h in range(levels):
            cand = xs[h] + np.exp(0.5 * log_c[h]) * (Ls[h] @ rng.standard_normal(n))
            lpc = f(cand)
            a = float(np.exp(min(0.0, lam[h] * (lpc - lps[h])))) if np.isfinite(lpc) else 0.0
            if rng.uniform() < a:
                xs[h], lps[h] = cand, lpc
                acc += (h == 0) and it >= burn
            log_c[h] = np.clip(log_c[h] + gam * (a - target), *lcr)
            d = xs[h] - mu[h]
            mu[h] += gam * d
            S[h] += gam * (np.outer(d, d) - S[h])
            if (it + 1) % retune_cov_every == 0:
                Ls[h] = _chol(S[h])
        if levels > 1:
            h = int(rng.integers(levels - 1))
            a = float(np.exp(min(0.0, (lam[h] - lam[h + 1]) * (lps[h + 1] - lps[h]))))
            tries[h] += 1
            if rng.uniform() < a:
                xs[[h, h + 1]], lps[[h, h + 1]] = xs[[h + 1, h]], lps[[h + 1, h]]
                swaps[h] += 1
            if adapt_temperatures:
                rho[h] = np.clip(rho[h] + gam * (a - swap_target), -20, 20)
                lam = 1.0 / np.concatenate([[1.0], 1.0 + np.cumsum(np.exp(rho))])
        if it >= burn and (it - burn) % thin == thin - 1:
            k = (it - burn) // thin
            out[k], olp[k] = xs[0], lps[0]
    return out, olp, acc / max(total - burn, 1), lam, swaps / np.maximum(tries, 1), np.exp(log_c)


def apt_sampler(logpost, x0, cov, lower=None, upper=None, draws: int = 20000, burn: int = 0, chains: int = 1,
                thin: int = 1, levels: int = 5, seed: int = 0, alpha=(0.25, 0.45), swap_alpha: float = 0.234,
                c: float = 0.25, c_range=(np.sqrt(np.finfo(float).eps), 100.0), rwm_exp: float = 0.6,
                retune_cov_every: int = 100, adapt_temperatures: bool = True, dispersion: float = 0.0,
                parallel=False) -> SamplerResult:
    """Adaptive parallel tempering (Miasojedow, Moulines and Vihola, 2013).

    ``levels`` replicas with targets π(θ)^λ_h, λ_1 = 1 > λ_2 > ... At each iteration, one
    random-walk Metropolis step per level (own scale and covariance, adapted as in
    ``mh_sampler``) and one swap proposal between a randomly drawn pair of neighbouring levels.
    The temperatures T_h = 1/λ_h satisfy T_{h+1} = T_h + exp(ρ_h), and ρ_h goes up or down as the
    swap probability is above or below ``swap_alpha``. Returns the cold replica.
    """
    x0, cov = np.asarray(x0, float), np.atleast_2d(np.asarray(cov, float))
    lo, up = _bounds(x0.size, lower, upper)
    f = _guard(logpost, lo, up)
    seeds = np.random.SeedSequence(seed).spawn(chains)

    def run(k):
        return _apt_chain(f, x0, cov, draws, burn, thin, np.random.default_rng(seeds[k]), levels, _target(alpha),
                          swap_alpha, c, c_range, rwm_exp, retune_cov_every, adapt_temperatures, dispersion)
    res = pmap(run, chains, parallel)
    return SamplerResult(np.stack([r[0] for r in res]), np.stack([r[1] for r in res]), np.array([r[2] for r in res]),
                         "apt", extras={"lambda": np.array([r[3] for r in res]),
                                        "swap_rate": np.array([r[4] for r in res]),
                                        "c": np.array([r[5] for r in res])})


# ---------------------------------------------------------------- DSMH
def _next_lambda(lp, lam_prev, ess_target):
    """Largest λ in (lam_prev, 1] whose relative ESS of the weights f^(λ - lam_prev) is at least ess_target."""
    def rel_ess(lam):
        lw = (lam - lam_prev) * lp
        w = np.exp(lw - lw.max())
        return w.sum() ** 2 / (w @ w) / len(w)
    if rel_ess(1.0) >= ess_target:
        return 1.0
    a, b = lam_prev, 1.0
    for _ in range(60):
        m = 0.5 * (a + b)
        a, b = (m, b) if rel_ess(m) >= ess_target else (a, m)
    return max(a, lam_prev * (1 + 1e-6) + 1e-12)


def dsmh_sampler(logpost, x0, cov, lower=None, upper=None, draws: int = 5000, groups: int = 10, stages: int = 20,
                 lambda_1: float | None = None, striations: int = 20, p_striated: float = 0.1, burn: int = 50,
                 thin: int = 1, alpha=(0.25, 0.45), c: float = 0.5, schedule: str = "geometric",
                 ess_min: float = 0.1, init: InitialDensity | None = None, init_scale: float = 3.0,
                 init_df: float = 5.0, seed: int = 0, parallel=False) -> SamplerResult:
    """Dynamic striated Metropolis-Hastings (Waggoner, Wu and Zha, 2016).

    Stages with targets f(θ)^λ_i, λ_1 < ... < λ_H = 1 (f is the posterior kernel). At each stage:

    1. the sample from the previous stage gets importance weights f^(λ_i - λ_{i-1}), which give the
       proposal covariance, the starting points and the increment of the normalising constant;
    2. that sample is sorted by the value of f and cut into ``striations`` striations of equal size;
    3. ``groups`` independent chains, each with ``draws/groups`` kept draws. With probability
       ``p_striated`` the proposal is a point drawn, in the previous sample, from the striation
       the current point belongs to, accepted with probability min{1, (f*/f)^(λ_i - λ_{i-1})}
       if it falls in the same striation; otherwise, a random walk N(θ, c Σ_i) with the scale
       tuned to the target acceptance rate.

    Stage zero consists of draws from the normalised initial density ``init`` (by default, a
    Student t centred at ``x0`` with covariance ``init_scale² cov``, truncated at the bounds). Since
    it is normalised, the product of the weight means estimates the marginal data density
    (``log_mdd``).

    ``schedule="geometric"`` uses λ_i = λ_1^((H-i)/(H-1)); ``"adaptive"`` picks each λ so that the
    relative effective sample size of the weights is ``ess_min`` (the number of stages becomes
    endogenous). In either case, if the first-stage weights under the requested ``lambda_1`` have
    relative effective sample size below ``ess_min``, ``lambda_1`` is raised until it is reached
    (the value used is in ``extras["lambda"][0]``).
    """
    x0, cov = np.asarray(x0, float), np.atleast_2d(np.asarray(cov, float))
    n = x0.size
    lo, up = _bounds(n, lower, upper)
    f = _guard(logpost, lo, up)
    rng = np.random.default_rng([seed, 0])
    if init is None:
        init = InitialDensity(x0, init_scale ** 2 * cov, lo, up, df=init_df, seed=seed)
    per = int(np.ceil(draws / groups))
    N = per * groups
    X = init.rvs(rng, N)
    lp = np.array(pmap(lambda i: f(X[i]), N, parallel))
    logq = init.logpdf(X)                       # log of the density the current sample was drawn from
    ok = np.isfinite(lp)
    if ok.sum() < max(10, 2 * n):
        raise ValueError("too few starting points with finite posterior; reduce init_scale")
    # A λ_1 that is too small leaves f^λ_1 almost flat: on unbounded support, the initial sample
    # does not cover it and the first-stage constant comes out biased. λ_1 is raised until the
    # first-stage weights have relative effective sample size of at least ess_min.
    lambda_1_asked = lambda_1
    lambda_1 = _first_lambda(lp, logq, ess_min, 1e-6 if lambda_1 is None else lambda_1)
    target = _target(alpha)
    log_c, lam_prev, log_mdd = np.log(c), 0.0, 0.0
    lams, accs, esss = [], [], []
    stage = 0
    while lam_prev < 1.0:
        stage += 1
        if schedule == "geometric":
            lam = 1.0 if stages == 1 or stage >= stages else float(lambda_1 ** ((stages - stage) / (stages - 1)))
        elif stage == 1:
            lam = lambda_1
        else:
            lam = _next_lambda(lp, lam_prev, ess_min)
        lw = np.where(ok, lam * lp - logq, -np.inf)             # weights relative to the previous sample
        log_mdd += float(special.logsumexp(lw) - np.log(N))
        w = np.exp(lw - lw.max())
        esss.append(float(w.sum() ** 2 / (w @ w)))
        mu, S = weighted_moments(X, w)
        L = _chol(S)
        order = np.argsort(np.where(ok, lp, -np.inf))
        order = order[np.isfinite(lw[order])]
        edges = np.linspace(0, len(order), striations + 1).astype(int)
        cuts = lp[order[edges[1:-1]]]                            # lower limits of striations 2..M
        starts = systematic_resample(w, rng, groups)
        rng.shuffle(starts)
        seeds = np.random.SeedSequence([seed, stage]).spawn(groups)
        Xp, lpp, logqp, c_now = X, lp, logq, np.exp(log_c)

        def run(g, lam=lam, L=L, Xp=Xp, lpp=lpp, logqp=logqp, order=order, edges=edges, cuts=cuts, starts=starts,
                seeds=seeds, c_now=c_now):
            r = np.random.default_rng(seeds[g])
            x, l = Xp[starts[g]].copy(), lpp[starts[g]]
            lq = logqp[starts[g]]            # log of the previous sample's density at the current point
            lc = np.log(c_now)
            ox, ol = np.empty((per, n)), np.empty(per)
            acc_rw = n_rw = 0
            total = burn + per * thin
            for it in range(total):
                if r.uniform() < p_striated:
                    s = int(np.searchsorted(cuts, l, side="right"))
                    j = order[r.integers(edges[s], edges[s + 1])] if edges[s + 1] > edges[s] else None
                    if j is not None and int(np.searchsorted(cuts, lpp[j], side="right")) == s:
                        lq_c = logqp[j]
                        # for a point outside the previous sample, its density is evaluated by the stage rule
                        if np.log(r.uniform()) < (lam * lpp[j] - lq_c) - (lam * l - lq):
                            x, l, lq = Xp[j].copy(), lpp[j], lq_c
                else:
                    cand = x + np.exp(0.5 * lc) * (L @ r.standard_normal(n))
                    lc_ = f(cand)
                    a = float(np.exp(min(0.0, lam * (lc_ - l)))) if np.isfinite(lc_) else 0.0
                    if r.uniform() < a:
                        x, l = cand, lc_
                        lq = prev_logq(cand, lc_)
                        acc_rw += it >= burn
                    n_rw += it >= burn
                    if it < burn:
                        lc += (it + 1.0) ** (-0.6) * (a - target)
                if it >= burn and (it - burn) % thin == thin - 1:
                    k = (it - burn) // thin
                    ox[k], ol[k] = x, l
            return ox, ol, acc_rw / max(n_rw, 1), lc

        if stage == 1:
            prev_logq = lambda th, l: float(init.logpdf(th)[0])
        else:
            prev_logq = lambda th, l, lam_prev=lam_prev: lam_prev * l
        res = pmap(run, groups, parallel)
        X = np.concatenate([r_[0] for r_ in res])
        lp = np.concatenate([r_[1] for r_ in res])
        ok = np.isfinite(lp)
        logq = lam * lp
        accs.append(float(np.mean([r_[2] for r_ in res])))
        log_c = float(np.mean([r_[3] for r_ in res]))
        lams.append(lam)
        lam_prev = lam
    return SamplerResult(X.reshape(groups, per, n), lp.reshape(groups, per), np.full(groups, accs[-1]), "dsmh",
                         log_mdd=log_mdd, extras={"lambda": np.array(lams), "acceptance_by_stage": np.array(accs),
                                                  "ess_by_stage": np.array(esss), "c": float(np.exp(log_c)),
                                                  "lambda_1_requested": lambda_1_asked})


def _first_lambda(lp, logq, ess_target, start):
    """Smallest λ ≥ start, on a geometric grid up to 1, whose weights f^λ / g have relative
    ESS ≥ ess_target (or the one with the largest ESS, if none reaches it)."""
    ok = np.isfinite(lp)
    grid = np.unique(np.append(np.geomspace(min(start, 1.0), 1.0, 80), 1.0))
    best, best_ess = 1.0, -1.0
    for lam in grid:
        lw = lam * lp[ok] - logq[ok]
        w = np.exp(lw - lw.max())
        e = w.sum() ** 2 / (w @ w) / len(lp)
        if e >= ess_target:
            return float(lam)
        if e > best_ess:
            best, best_ess = float(lam), e
    return best


def rrf_sampler(logpost, x0, cov, lower=None, upper=None, draws: int = 20000, stages: int = 40,
                lambda_1: float = 2.5e-5, M_over_N: float = 0.05, ess_min: float = 0.1,
                geometric_lambda: bool = True, p: float | None = None, **options) -> SamplerResult:
    """``dsmh_sampler`` with the option names and defaults of RISE's ``rrf_sampler``.

    ``H`` → ``stages`` (40), ``lambda_1`` (2.5e-5), ``M_over_N`` (fraction of draws per bin, 5%,
    which here sets the number of striations to 1/``M_over_N``), ``geometric_lambda``, ``ess_min``,
    ``p`` (probability of drawing from the previous distribution). RISE's algorithm is not
    documented; this function has not been compared with it draw by draw.
    """
    res = dsmh_sampler(logpost, x0, cov, lower, upper, draws=draws, stages=stages, lambda_1=lambda_1,
                       striations=max(int(round(1 / M_over_N)), 1), ess_min=ess_min,
                       schedule="geometric" if geometric_lambda else "adaptive",
                       **({} if p is None else {"p_striated": p}), **options)
    res.sampler = "rrf"
    return res


# ---------------------------------------------------------------- SMC
def smc_sampler(loglik, logprior, prior_rvs, particles: int = 1000, stages: int = 100, bend: float = 2.0,
                mh_steps: int = 1, blocks: int = 1, c: float = 0.5, target: float = 0.25, mixture: float = 0.9,
                adaptive: bool = False, ess_target: float = 0.5, resample_threshold: float = 0.5, seed: int = 0,
                parallel=False) -> SamplerResult:
    """Sequential Monte Carlo with likelihood tempering (Herbst and Schorfheide, 2014, algorithms 1 and 2).

    Targets π_n(θ) ∝ p(Y|θ)^φ_n p(θ), φ_n = ((n-1)/(stages-1))^bend. At each stage:
    correction (incremental weights p(Y|θ)^(φ_n - φ_{n-1})), selection (systematic resampling when
    the effective sample size falls below ``resample_threshold`` N) and mutation (``mh_steps``
    Metropolis steps in ``blocks`` random blocks, with a mixture proposal: N(θ, c² Σ_n) with
    probability ``mixture``, and the rest split between the diagonal version and an independent
    proposal N(μ_n, c² Σ_n)). The scale follows
    c ← c (0.95 + 0.10 e^{16(a - target)} / (1 + e^{16(a - target)})).
    The marginal data density is the product of the weighted means of the incremental weights.

    ``prior_rvs(rng, N)`` returns N draws from the prior, (N, n). With ``adaptive=True`` the φ_n
    are chosen so that the relative effective sample size after correction is ``ess_target`` (an
    extension that is not in the 2014 paper). Returns the final resampled particles (equal weights).
    """
    rng = np.random.default_rng([seed, 0])
    X = np.atleast_2d(np.asarray(prior_rvs(rng, particles), float))
    N, n = X.shape

    def both(x):
        lpr = logprior(x)
        if not np.isfinite(lpr):
            return -np.inf, -np.inf
        ll = loglik(x)
        return (float(ll), float(lpr)) if np.isfinite(ll) else (-np.inf, float(lpr))
    v = np.array(pmap(lambda i: both(X[i]), N, parallel))
    ll, lpr = v[:, 0], v[:, 1]
    W = np.full(N, 1.0 / N)
    phi_prev, log_mdd, stage = 0.0, 0.0, 1
    phis, esss, accs, cs = [0.0], [float(N)], [], []
    while phi_prev < 1.0:
        stage += 1
        if adaptive:
            phi = _next_phi(ll, W, phi_prev, ess_target)
        else:
            phi = 1.0 if stage >= stages else ((stage - 1) / (stages - 1)) ** bend
        with np.errstate(invalid="ignore"):
            lw = np.where(np.isfinite(ll), (phi - phi_prev) * ll, -np.inf)
        lW = np.log(W) + lw
        inc = special.logsumexp(lW)
        if not np.isfinite(inc):
            raise ValueError("all particles have zero likelihood; increase the number of particles")
        log_mdd += float(inc)
        W = np.exp(lW - inc)
        ess = 1.0 / (W @ W)
        mu, S = weighted_moments(X, W)
        if ess < resample_threshold * N or phi >= 1.0:
            idx = systematic_resample(W, rng)
            X, ll, lpr, W = X[idx], ll[idx], lpr[idx], np.full(N, 1.0 / N)
        L, sd = _chol(S), np.sqrt(np.clip(np.diag(S), 1e-300, None))
        Si = np.linalg.inv(L @ L.T)
        ldr = -0.5 * np.linalg.slogdet(Si)[1] - np.sum(np.log(sd))      # log |Σ|^(1/2) / prod(sd)
        # all random numbers of the mutation step come from the main generator: the result does not
        # depend on the number of processes
        parts = np.array_split(rng.permutation(n), min(blocks, n))
        nb = len(parts)
        Z = rng.standard_normal((N, mh_steps, nb, n))
        U = rng.uniform(size=(N, mh_steps, nb, 2))
        Xc, llc, lprc = X, ll, lpr

        def mutate(i, phi=phi, L=L, sd=sd, Si=Si, mu=mu, Xc=Xc, llc=llc, lprc=lprc, c=c, Z=Z, U=U, parts=parts, ldr=ldr):
            x, l, p = Xc[i].copy(), llc[i], lprc[i]
            acc = tot = 0
            for s_ in range(mh_steps):
                for k, b in enumerate(parts):
                    u, z = U[i, s_, k, 0], Z[i, s_, k]
                    cand = x.copy()
                    corr = 0.0
                    if u < mixture:
                        step = c * (L @ z)
                    elif u < mixture + (1 - mixture) / 2:
                        step = c * sd * z
                    else:                       # independent around the mean; only with one block
                        step = mu + c * (L @ z) - x if nb == 1 else c * (L @ z)
                    cand[b] = x[b] + step[b]
                    if mixture < 1.0 and nb == 1:          # Hastings ratio of the mixture proposal
                        corr = _mix_logq(x, cand, mu, Si, sd, c, mixture, ldr) - _mix_logq(cand, x, mu, Si, sd, c, mixture, ldr)
                    lc, pc = both(cand)
                    tot += 1
                    if np.isfinite(lc) and np.log(U[i, s_, k, 1]) < phi * (lc - l) + pc - p + corr:
                        x, l, p = cand, lc, pc
                        acc += 1
            return x, l, p, acc / tot
        res = pmap(mutate, N, parallel)
        X = np.array([r_[0] for r_ in res])
        ll = np.array([r_[1] for r_ in res])
        lpr = np.array([r_[2] for r_ in res])
        a = float(np.mean([r_[3] for r_ in res]))
        e = np.exp(16 * (a - target))
        c = c * (0.95 + 0.10 * e / (1 + e))
        phis.append(phi); esss.append(float(ess)); accs.append(a); cs.append(c)
        phi_prev = phi
    return SamplerResult(X[None], (ll + lpr)[None], np.array([accs[-1]]), "smc", log_mdd=log_mdd,
                         extras={"phi": np.array(phis), "ess": np.array(esss), "acceptance_by_stage": np.array(accs),
                                 "c": np.array(cs), "loglik": ll, "logprior": lpr})


def _mix_logq(new, old, mu, Si, sd, c, mixture, ldr):
    """log q(new | old) of the mixture proposal, up to the common constant of the normal N(·, c² Σ)."""
    d = new - old
    a = -0.5 * d @ Si @ d / c ** 2
    b = -0.5 * np.sum((d / (c * sd)) ** 2) + ldr
    e = new - mu
    g = -0.5 * e @ Si @ e / c ** 2
    m = max(a, b, g)
    return m + np.log(mixture * np.exp(a - m) + (1 - mixture) / 2 * (np.exp(b - m) + np.exp(g - m)))


def _next_phi(ll, W, phi_prev, ess_target):
    ok = np.isfinite(ll)

    def rel_ess(phi):
        lw = np.where(ok, (phi - phi_prev) * np.where(ok, ll, 0.0), -np.inf) + np.log(W)
        w = np.exp(lw - lw.max())
        return w.sum() ** 2 / (w @ w) / len(W)
    if rel_ess(1.0) >= ess_target:
        return 1.0
    a, b = phi_prev, 1.0
    for _ in range(60):
        m = 0.5 * (a + b)
        a, b = (m, b) if rel_ess(m) >= ess_target else (a, m)
    return max(a, phi_prev + 1e-10)


def smc_from_kernel(logpost, init: InitialDensity, **options) -> SamplerResult:
    """SMC for an arbitrary kernel: geometric bridge g^(1-φ) f^φ between the normalised initial
    density ``init`` and the kernel ``f``. Useful when there is no prior to draw from."""
    last = [None, 0.0]

    def logprior(x):
        last[0], last[1] = x, float(init.logpdf(x)[0])
        return last[1]

    def loglik(x):                      # only called inside the support of init, right after logprior(x)
        v = logpost(x)
        lg = last[1] if last[0] is x else float(init.logpdf(x)[0])
        return float(v) - lg if np.isfinite(v) else -np.inf
    res = smc_sampler(loglik, logprior, init.rvs, **options)
    return res


# ---------------------------------------------------------------- link with estimation
def posterior_sample(result, sampler: str = "mh", cov=None, **options):
    """Run the sampler from an ``EstimationResult`` and store the draws in it.

    ``sampler``: ``"mh"``, ``"apt"``, ``"dsmh"``, ``"rrf"`` or ``"smc"``. SMC starts from the prior
    when every parameter has a proper prior and there is no joint prior; otherwise, from a Student t
    around the mode (``smc_from_kernel``). The full sampler result is kept in
    ``result.sampler_result`` (it includes ``log_mdd`` when the sampler provides it).
    """
    pr = result.problem
    S = result.cov if cov is None else np.asarray(cov, float)
    needs_cov = not (sampler == "smc" and _prior_is_samplable(pr))
    if needs_cov and not np.all(np.isfinite(S)):
        raise ValueError("invalid proposal covariance (the Hessian is not positive definite); pass 'cov'")
    if sampler == "mh":
        out = mh_sampler(pr.logpost, result.mode, S, pr.lower, pr.upper, **options)
    elif sampler == "apt":
        out = apt_sampler(pr.logpost, result.mode, S, pr.lower, pr.upper, **options)
    elif sampler == "dsmh":
        out = dsmh_sampler(pr.logpost, result.mode, S, pr.lower, pr.upper, **options)
    elif sampler == "rrf":
        out = rrf_sampler(pr.logpost, result.mode, S, pr.lower, pr.upper, **options)
    elif sampler == "smc":
        if _prior_is_samplable(pr):
            endo = getattr(pr, "endogenous_priors", None)

            def loglik(x):
                ll = pr.loglik(x)
                return ll + endo.logpdf(pr, x) if endo is not None and np.isfinite(ll) else ll

            def prior_rvs(rng, size):
                out_ = np.empty((size, len(pr.priors)))
                for i in range(size):
                    for _ in range(10000):
                        x = np.array([p.rvs(rng) for p in pr.priors])
                        if np.all(x >= pr.lower) and np.all(x <= pr.upper):
                            break
                    out_[i] = x
                return out_
            out = smc_sampler(loglik, pr.logprior, prior_rvs, **options)
            # SMC starts from the prior truncated at the bounds (normalised); the kernel of pr.logpost
            # uses the prior without renormalising. The log of the prior mass inside the bounds is added.
            rng_m = np.random.default_rng([options.get("seed", 0), 77])
            xm = np.array([[p.rvs(rng_m) for p in pr.priors] for _ in range(20000)])
            mass = np.mean(np.all((xm >= pr.lower) & (xm <= pr.upper), axis=1))
            out.extras["log_prior_mass"] = float(np.log(mass))
            out.log_mdd += float(np.log(mass))
        else:
            scale = options.pop("init_scale", 3.0)
            init = InitialDensity(result.mode, scale ** 2 * S, pr.lower, pr.upper, seed=options.get("seed", 0))
            out = smc_from_kernel(pr.logpost, init, **options)
    else:
        raise ValueError(f"unknown sampler: {sampler!r}")
    result.draws, result.logpost_draws, result.acceptance = out.draws, out.logpost, out.acceptance
    result.sampler_result = out
    return result


def _prior_is_samplable(pr) -> bool:
    return all(p is not None for p in pr.priors) and not getattr(pr, "joint_priors", None)
