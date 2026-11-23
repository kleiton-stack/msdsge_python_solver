"""Estimators of the log marginal data density (MDD) from posterior draws.

The nine algorithms of RISE's ``mdd`` class (list in ``LWZ_Econometrica2013/master_switch.m``),
with the same names, plus ``ce``:

=============== ==========================================================================
``laplace``      Laplace approximation at the mode (Tierney and Kadane, 1986).
``laplace_mcmc`` Laplace with mode and covariance taken from the draws (Lewis and Raftery, 1997).
``mhm``          Modified harmonic mean (Gelfand and Dey, 1994; Geweke, 1999).
``swz``          Harmonic mean with truncated elliptical density (Sims, Waggoner and Zha, 2008).
``mueller``      Mueller's method, as described in Liu, Waggoner and Zha (2011, appendix).
``bridge``       Bridge sampling with the optimal bridge (Meng and Wong, 1996).
``is``           Importance sampling (Frühwirth-Schnatter, 2004).
``ris``          Reciprocal importance sampling (Frühwirth-Schnatter, 2004).
``cj``           Chib and Jeliazkov (2001), from the Metropolis-Hastings output.
``ce``           Importance sampling with minimum cross-entropy density (Chan and Eisenstat, 2015).
                 Not in RISE.
=============== ==========================================================================

``mhm``, ``ris`` and ``laplace_mcmc`` use only the draws and their log posterior values; the others
evaluate the posterior kernel at ``L`` new points (RISE's ``L`` option). The ``swz`` and
``mueller`` methods are undocumented stubs in RISE; the implementation follows the papers.

All of them handle parameter bounds: the weighting density is truncated to the support and
renormalised (mass estimated by simulation when there are bounds).
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

from .samplers import InitialDensity, _chol

ALGORITHMS = ("mhm", "swz", "mueller", "bridge", "is", "ris", "cj", "laplace", "laplace_mcmc", "ce")

REFERENCES = {
    "laplace": "Tierney and Kadane (1986), JASA 81(393), 82-86, doi:10.1080/01621459.1986.10478240",
    "laplace_mcmc": "Lewis and Raftery (1997), JASA 92(438), 648-655, doi:10.1080/01621459.1997.10474016",
    "mhm": "Gelfand and Dey (1994), JRSS B 56(3), 501-514, doi:10.1111/j.2517-6161.1994.tb01996.x; "
           "Geweke (1999), Econometric Reviews 18(1), 1-73, doi:10.1080/07474939908800428",
    "swz": "Sims, Waggoner and Zha (2008), J. Econometrics 146(2), 255-274, doi:10.1016/j.jeconom.2008.08.023",
    "mueller": "Liu, Waggoner and Zha (2011), Quantitative Economics 2(2), 251-301, doi:10.3982/QE71 (appendix)",
    "bridge": "Meng and Wong (1996), Statistica Sinica 6, 831-860 (no DOI in Crossref)",
    "is": "Frühwirth-Schnatter (2004), Econometrics Journal 7(1), 143-167, doi:10.1111/j.1368-423X.2004.00125.x",
    "ris": "Frühwirth-Schnatter (2004), Econometrics Journal 7(1), 143-167, doi:10.1111/j.1368-423X.2004.00125.x",
    "cj": "Chib and Jeliazkov (2001), JASA 96(453), 270-281, doi:10.1198/016214501750332848",
    "ce": "Chan and Eisenstat (2015), Econometric Reviews 34(3), 256-285, doi:10.1080/07474938.2014.944474",
}


class _Mixture:
    """Mixture of normals fitted by EM, truncated at the bounds."""

    def __init__(self, x, k, lo, up, rng, iters: int = 200):
        N, n = x.shape
        mu = x[rng.choice(N, k, replace=False)]
        S = np.array([np.cov(x.T).reshape(n, n) for _ in range(k)])
        w = np.full(k, 1.0 / k)
        prev = -np.inf
        for _ in range(iters):
            lr = np.stack([np.log(w[j]) + stats.multivariate_normal(mu[j], S[j], allow_singular=True).logpdf(x)
                           for j in range(k)], axis=1).reshape(N, k)
            tot = special.logsumexp(lr, axis=1)
            r = np.exp(lr - tot[:, None])
            nk = r.sum(0) + 1e-300
            w = nk / N
            mu = (r.T @ x) / nk[:, None]
            for j in range(k):
                d = x - mu[j]
                S[j] = (d * r[:, j:j + 1]).T @ d / nk[j] + 1e-10 * np.eye(n)
            if abs(tot.sum() - prev) < 1e-8 * N:
                break
            prev = tot.sum()
        self.w, self.mu, self.S, self.lo, self.up = w, mu, S, lo, up
        self.L = [_chol(s) for s in S]
        self.log_mass = 0.0
        if np.any(np.isfinite(lo)) or np.any(np.isfinite(up)):
            y = self._raw(rng, 20000)
            self.log_mass = float(np.log(max(np.mean(np.all((y >= lo) & (y <= up), axis=1)), 1e-300)))

    def _raw(self, rng, size):
        comp = rng.choice(len(self.w), size, p=self.w)
        z = rng.standard_normal((size, self.mu.shape[1]))
        return np.array([self.mu[c] + self.L[c] @ zi for c, zi in zip(comp, z)])

    def rvs(self, rng, size):
        out = np.empty((0, self.mu.shape[1]))
        while len(out) < size:
            y = self._raw(rng, max(2 * (size - len(out)), 64))
            out = np.vstack([out, y[np.all((y >= self.lo) & (y <= self.up), axis=1)]])
        return out[:size]

    def logpdf(self, x):
        x = np.atleast_2d(x)
        lr = np.stack([np.log(self.w[j]) + np.atleast_1d(stats.multivariate_normal(self.mu[j], self.S[j], allow_singular=True).logpdf(x))
                       for j in range(len(self.w))], axis=1)
        inside = np.all((x >= self.lo) & (x <= self.up), axis=1)
        return np.where(inside, special.logsumexp(lr, axis=1) - self.log_mass, -np.inf)


class MDD:
    """Marginal data density estimators from draws.

    Parameters
    ----------
    draws : (N, n) or (chains, N, n)
    logpost_draws : log of the posterior kernel at the draws, same shape
    logpost : kernel function (needed for swz, mueller, bridge, is, cj, ce)
    lower, upper : parameter bounds
    mode, hessian : mode and Hessian of -log posterior (for ``laplace``)
    L : number of new draws from the weighting density (RISE's ``L`` option)
    """

    def __init__(self, draws, logpost_draws, logpost=None, lower=None, upper=None, mode=None, hessian=None,
                 L: int = 5000, seed: int = 0):
        d = np.asarray(draws, float)
        self.chains = d.shape[0] if d.ndim == 3 else 1
        self.x = d.reshape(-1, d.shape[-1])
        self.lp = np.asarray(logpost_draws, float).ravel()
        if self.lp.size != len(self.x):
            raise ValueError("draws and logpost_draws have different sizes")
        self.N, self.n = self.x.shape
        self.logpost = logpost
        self.lo = np.full(self.n, -np.inf) if lower is None else np.asarray(lower, float)
        self.up = np.full(self.n, np.inf) if upper is None else np.asarray(upper, float)
        self.mode = None if mode is None else np.asarray(mode, float)
        self.hessian = None if hessian is None else np.asarray(hessian, float)
        self.L, self.seed = L, seed
        self.mean = self.x.mean(0)
        self.cov = np.atleast_2d(np.cov(self.x.T))
        self._cache = {}

    # ------------------------------------------------------------ helpers
    def _rng(self, tag: int):
        return np.random.default_rng([self.seed, tag])

    def _f(self, X) -> np.ndarray:
        if self.logpost is None:
            raise ValueError("this estimator needs the posterior kernel function (logpost)")
        out = np.empty(len(X))
        for i, x in enumerate(X):
            if np.any(x < self.lo) or np.any(x > self.up):
                out[i] = -np.inf
            else:
                v = self.logpost(x)
                out[i] = v if np.isfinite(v) else -np.inf
        return out

    def _weighting(self, df=np.inf, scale: float = 1.0):
        """Normal (or t) with the posterior moments, truncated at the bounds, and its L evaluated draws."""
        key = ("w", df, scale)
        if key not in self._cache:
            h = InitialDensity(self.mean, scale ** 2 * self.cov, self.lo, self.up, df=df, seed=self.seed)
            xh = h.rvs(self._rng(1), self.L)
            self._cache[key] = (h, xh, self._f(xh), h.logpdf(xh))
        return self._cache[key]

    def _neff(self) -> float:
        from .diagnostics import ess
        per = self.N // self.chains
        return float(min(ess(self.lp[:per * self.chains].reshape(self.chains, per)[:, :, None])[0], self.N))

    # ------------------------------------------------------------ estimators
    def laplace(self) -> float:
        if self.mode is None or self.hessian is None:
            raise ValueError("laplace needs 'mode' and 'hessian'")
        sign, logdet = np.linalg.slogdet(self.hessian)
        if sign <= 0:
            return np.nan
        lp = self.logpost(self.mode) if self.logpost is not None else self.lp.max()
        return float(lp + 0.5 * self.n * np.log(2 * np.pi) - 0.5 * logdet)

    def laplace_mcmc(self) -> float:
        """Laplace-Metropolis: maximum of the kernel among the draws and sample covariance."""
        return float(self.lp.max() + 0.5 * self.n * np.log(2 * np.pi) + 0.5 * np.linalg.slogdet(self.cov)[1])

    def mhm(self, tau=0.9) -> float:
        """Modified harmonic mean with a normal truncated to the ``tau`` probability region.

        ``tau`` may be a list; in that case the mean of the estimates is returned.
        """
        taus = np.atleast_1d(tau)
        Li = _chol(self.cov)
        z = np.linalg.solve(Li, (self.x - self.mean).T).T
        d = np.sum(z * z, axis=1)
        logdet = 2 * np.sum(np.log(np.diag(Li)))
        vals = []
        for p in taus:
            crit = stats.chi2.ppf(p, self.n)
            log_mass = np.log(p)
            if np.any(np.isfinite(self.lo)) or np.any(np.isfinite(self.up)):
                # part of the ellipse outside the support: correct the mass by simulation
                zz = self._rng(2).standard_normal((20000, self.n))
                zz = zz[np.sum(zz * zz, axis=1) <= crit]
                y = self.mean + zz @ Li.T
                log_mass += np.log(max(np.mean(np.all((y >= self.lo) & (y <= self.up), axis=1)), 1e-300))
            logf = -log_mass - 0.5 * (self.n * np.log(2 * np.pi) + logdet + d)
            terms = np.where(d <= crit, logf - self.lp, -np.inf)
            vals.append(-(special.logsumexp(terms) - np.log(self.N)))
        return float(np.mean(vals))

    def ris(self, df=np.inf) -> float:
        """Reciprocal importance: 1/p(Y) = E_post[h(θ)/f(θ)], h normal with the posterior moments."""
        h = InitialDensity(self.mean, self.cov, self.lo, self.up, df=df, seed=self.seed)
        return float(-(special.logsumexp(h.logpdf(self.x) - self.lp) - np.log(self.N)))

    def is_(self, df=np.inf, scale: float = 1.0) -> float:
        """Importance: p(Y) = E_h[f(θ)/h(θ)]."""
        _, _, fh, lh = self._weighting(df, scale)
        return float(special.logsumexp(fh - lh) - np.log(len(fh)))

    def bridge(self, df=np.inf, tol: float = 1e-10, maxiter: int = 1000) -> float:
        """Bridge sampling with the optimal bridge of Meng and Wong (1996), by fixed-point iteration.

        The posterior sample size enters through the effective size (autocorrelation of the draws).
        """
        h, _, fh, lh = self._weighting(df)
        l1 = self.lp - h.logpdf(self.x)            # posterior draws
        l2 = fh - lh                                # draws from the weighting density
        shift = np.median(l1)
        l1, l2 = l1 - shift, l2 - shift
        n1, n2 = self._neff(), len(l2)
        s1, s2 = n1 / (n1 + n2), n2 / (n1 + n2)
        logr = 0.0
        for _ in range(maxiter):
            num = special.logsumexp(l2 - np.logaddexp(np.log(s1) + l2, np.log(s2) + logr)) - np.log(len(l2))
            den = special.logsumexp(-np.logaddexp(np.log(s1) + l1, np.log(s2) + logr)) - np.log(len(l1))
            new = num - den
            if abs(new - logr) < tol:
                logr = new
                break
            logr = new
        return float(logr + shift)

    def mueller(self, df=np.inf) -> float:
        """Mueller's method: the constant c that equates the masses of (π - h)⁺ and (h - π)⁺, π = f/c.

        Solves E_post[(1 - c h/f)⁺] = E_h[(1 - f/(c h))⁺] in log c by bisection.
        """
        h, _, fh, lh = self._weighting(df)
        r1 = self.lp - h.logpdf(self.x)            # log f/h under the posterior
        r2 = fh - lh                                # log f/h under the weighting density

        def gap(logc):
            a = np.mean(np.clip(1 - np.exp(np.minimum(logc - r1, 0.0)), 0, None))
            with np.errstate(over="ignore"):
                b = np.mean(np.clip(1 - np.exp(np.minimum(r2 - logc, 0.0)), 0, None))
            return a - b
        fin = np.concatenate([r1[np.isfinite(r1)], r2[np.isfinite(r2)]])
        lo, up = fin.min() - 1.0, fin.max() + 1.0
        if gap(lo) <= 0 or gap(up) >= 0:
            return np.nan
        return float(optimize.brentq(gap, lo, up, xtol=1e-12))

    def swz(self, pvalue: float = 0.9) -> float:
        """Sims, Waggoner and Zha (2008): elliptical density with power-law radius, truncated to the
        region {f(θ) > L}, where L leaves the fraction ``pvalue`` of the draws above it."""
        center = self.x[np.argmax(self.lp)] if self.mode is None else self.mode
        d = self.x - center
        Om = d.T @ d / self.N
        Lc = _chol(Om)
        logdet = 2 * np.sum(np.log(np.diag(Lc)))
        r = np.sqrt(np.sum(np.linalg.solve(Lc, d.T).T ** 2, axis=1))
        c1, c10, c90 = np.quantile(r, [0.01, 0.10, 0.90])
        v = np.log(1 / 9) / np.log(c10 / c90)
        a, b = c1, c90 / 0.9 ** (1 / v)
        level = np.quantile(self.lp, 1 - pvalue)
        n = self.n

        def logg(rr):
            lf = np.log(v) + (v - 1) * np.log(rr) - np.log(b ** v - a ** v)
            return special.gammaln(n / 2) - np.log(2) - 0.5 * n * np.log(np.pi) - 0.5 * logdet + lf - (n - 1) * np.log(rr)
        # mass of g in the truncated region, from iid draws of g
        rng = self._rng(3)
        u = rng.standard_normal((self.L, n))
        u /= np.linalg.norm(u, axis=1)[:, None]
        rr = (a ** v + rng.uniform(size=self.L) * (b ** v - a ** v)) ** (1 / v)
        y = center + (rr[:, None] * u) @ Lc.T
        q = np.mean(self._f(y) > level)
        if q <= 0:
            return np.nan
        inside = (r >= a) & (r <= b) & (self.lp > level)
        with np.errstate(divide="ignore"):
            terms = np.where(inside, logg(np.where(inside, r, 1.0)) - np.log(q) - self.lp, -np.inf)
        return float(-(special.logsumexp(terms) - np.log(self.N)))

    def cj(self, scale: float | None = None) -> float:
        """Chib and Jeliazkov (2001): log p(Y) = log f(θ*) - log π(θ*|Y), with

        π(θ*|Y) = E_post[α(θ, θ*) q(θ, θ*)] / E_{q(θ*, ·)}[α(θ*, θ)],

        θ* the draw with the highest density and q the normal random walk with covariance
        ``scale`` Σ (default 2.38²/n). The identity holds for any q, not only the one used in the
        sampler.
        """
        k = int(np.argmax(self.lp))
        xs, ls = self.x[k], self.lp[k]
        sc = 2.38 ** 2 / self.n if scale is None else scale
        Lq = _chol(sc * self.cov)
        logdet = 2 * np.sum(np.log(np.diag(Lq)))
        z = np.linalg.solve(Lq, (self.x - xs).T).T
        logq = -0.5 * (self.n * np.log(2 * np.pi) + logdet + np.sum(z * z, axis=1))
        num = special.logsumexp(np.minimum(0.0, ls - self.lp) + logq) - np.log(self.N)
        y = xs + self._rng(4).standard_normal((self.L, self.n)) @ Lq.T
        fy = self._f(y)
        den = special.logsumexp(np.minimum(0.0, fy - ls)) - np.log(self.L)
        return float(ls - (num - den))

    def ce(self, components: int = 2) -> float:
        """Importance sampling with the mixture of normals of minimum cross-entropy relative to the
        posterior (maximum likelihood of the mixture on the draws, by EM)."""
        rng = self._rng(5)
        sub = self.x if self.N <= 20000 else self.x[rng.choice(self.N, 20000, replace=False)]
        g = _Mixture(sub, components, self.lo, self.up, rng)
        y = g.rvs(rng, self.L)
        return float(special.logsumexp(self._f(y) - g.logpdf(y)) - np.log(self.L))

    # ------------------------------------------------------------ interface
    def estimate(self, algorithm: str = "mhm", **options) -> float:
        name = {"is": "is_"}.get(algorithm, algorithm)
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm: {algorithm!r}; options: {ALGORITHMS}")
        return getattr(self, name)(**options)

    def all(self, algorithms=None) -> dict:
        """All applicable estimators (``nan`` for those that cannot be computed)."""
        out = {}
        for a in (ALGORITHMS if algorithms is None else algorithms):
            try:
                out[a] = self.estimate(a)
            except ValueError:
                out[a] = np.nan
        return out


def mcmc_mdd(draws, logpost_draws, logpost=None, lower=None, upper=None, algorithm: str = "mhm", L: int = 5000,
             seed: int = 0, mode=None, hessian=None, **options) -> float:
    """Shortcut in the style of RISE's ``mcmc_mdd``."""
    return MDD(draws, logpost_draws, logpost, lower, upper, mode, hessian, L=L, seed=seed).estimate(algorithm, **options)


def from_result(result, L: int = 5000, seed: int = 0) -> MDD:
    """``MDD`` from an ``EstimationResult`` with draws."""
    if result.draws is None:
        raise ValueError("run a sampler first")
    pr = result.problem
    return MDD(result.draws, result.logpost_draws, pr.logpost, pr.lower, pr.upper, result.mode, result.hessian,
               L=L, seed=seed)
