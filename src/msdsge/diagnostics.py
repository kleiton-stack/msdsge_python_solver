"""Convergence, mode and identification diagnostics.

- ``gelman_rubin``: univariate (Gelman and Rubin, 1992) and multivariate (Brooks and Gelman, 1998)
  potential scale reduction factor, with the outputs of ``gelman_rubin`` in RISE's ``mcmc`` class
  (R, W, B, Rp).
- ``ess``: effective sample size by Geyer's (1992) initial positive sequence, with several
  chains combined as in Vehtari et al. (2021).
- ``geweke``: test of equal means between the start and the end of the chain (Geweke, 1992).
- ``trace_data``, ``cumulative_means``, ``autocorrelations``, ``psrf_path``,
  ``prior_posterior_data``: data behind the plots of RISE's ``mcmc`` class (``traceplot``,
  ``meanplot``, ``autocorrplot``, ``psrf_plot``, ``densplot``). ``trace_plot`` and
  ``prior_posterior_plot`` draw with matplotlib if it is installed; otherwise they return the data.
- ``check_optimum`` (``mode_curvature`` in RISE): profile of the posterior around the mode.
- ``identification``: rank of the Hessian or of the Jacobian of the moments (Iskrev, 2010).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["gelman_rubin", "psrf_path", "ess", "geweke", "autocorrelations", "cumulative_means", "trace_data",
           "prior_posterior_data", "trace_plot", "prior_posterior_plot", "check_optimum", "mode_curvature",
           "identification", "summary"]


def _as3d(draws) -> np.ndarray:
    d = np.asarray(draws, float)
    if d.ndim == 1:
        d = d[None, :, None]
    elif d.ndim == 2:
        d = d[None]
    return d


def _names(draws, names):
    return list(names) if names is not None else [f"p{i + 1}" for i in range(_as3d(draws).shape[2])]


# ---------------------------------------------------------------- convergence
def gelman_rubin(draws, names=None, split: bool = False) -> dict:
    """PSRF per parameter and multivariate.

    Returns ``R`` (series, square root of ((n-1)/n W + B/n) / W), ``W`` (within variance), ``B``
    (between variance, B/n in the notation of Brooks and Gelman multiplied by n) and ``Rp``
    (multivariate factor, equation 4.1 of Brooks and Gelman, 1998). Needs at least two chains, or
    ``split=True`` to split each chain in half.
    """
    d = _as3d(draws)
    if split:
        h = d.shape[1] // 2
        d = np.concatenate([d[:, :h], d[:, h:2 * h]])
    m, n, k = d.shape
    if m < 2:
        raise ValueError("Gelman-Rubin needs two or more chains (or split=True)")
    nm = _names(d, names)
    means = d.mean(1)
    W = d.var(axis=1, ddof=1).mean(0)
    B = n * means.var(axis=0, ddof=1)
    R = np.sqrt(((n - 1) / n * W + B / n) / W)
    Wm = np.mean([np.atleast_2d(np.cov(d[c].T)) for c in range(m)], axis=0)
    Bn = np.atleast_2d(np.cov(means.T))                       # B/n
    try:
        lam = np.max(np.real(np.linalg.eigvals(np.linalg.solve(Wm, Bn))))
        Rp = float(np.sqrt((n - 1) / n + (m + 1) / m * lam))
    except np.linalg.LinAlgError:
        Rp = np.nan
    return {"R": pd.Series(R, index=nm), "W": pd.Series(W, index=nm), "B": pd.Series(B, index=nm), "Rp": Rp}


def psrf_path(draws, names=None, points: int = 20) -> pd.DataFrame:
    """PSRF computed with the first n iterations, for increasing n (data of ``psrf_plot``)."""
    d = _as3d(draws)
    grid = np.unique(np.linspace(max(20, d.shape[1] // points), d.shape[1], points).astype(int))
    return pd.DataFrame({n: gelman_rubin(d[:, :n], names)["R"] for n in grid}).T.rename_axis("iterations")


def _autocov(x):
    n = len(x)
    x = x - x.mean()
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, size)
    return np.fft.irfft(f * np.conj(f), size)[:n] / n


def ess(draws) -> np.ndarray:
    """Effective sample size per parameter, adding up the chains."""
    d = _as3d(draws)
    m, n, k = d.shape
    out = np.empty(k)
    for j in range(k):
        acov = np.array([_autocov(d[c, :, j]) for c in range(m)])
        w = np.mean(acov[:, 0]) * n / (n - 1)
        var_plus = w * (n - 1) / n
        if m > 1:
            var_plus += d[:, :, j].mean(1).var(ddof=1)
        if var_plus <= 0 or not np.isfinite(var_plus):
            out[j] = np.nan
            continue
        rho = 1 - (w - acov.mean(0)) / var_plus
        rho[0] = 1.0
        tau, t = -1.0, 0
        prev = np.inf
        while t + 1 < n:
            pair = rho[t] + rho[t + 1]
            if pair < 0:
                break
            pair = min(pair, prev)               # monotone sequence
            tau += 2 * pair
            prev = pair
            t += 2
        out[j] = m * n / max(tau, 1.0 / np.log10(max(m * n, 10)))
    return out


def _spectral0(x) -> float:
    """Spectral density at zero (long-run variance) with a Bartlett window."""
    n = len(x)
    ac = _autocov(x)
    bw = int(min(n - 1, max(1, np.floor(4 * (n / 100.0) ** (2 / 9) * 5))))
    w = 1 - np.arange(1, bw + 1) / (bw + 1)
    return float(ac[0] + 2 * np.sum(w * ac[1:bw + 1]))


def geweke(draws, names=None, first: float = 0.1, last: float = 0.5) -> pd.DataFrame:
    """Geweke's (1992) z statistic per chain and parameter: mean of the first ``first`` fraction
    against that of the last ``last`` fraction of the chain, with long-run variance (Bartlett window)."""
    d = _as3d(draws)
    m, n, k = d.shape
    nm = _names(d, names)
    na, nb = int(first * n), int(last * n)
    rows = []
    for c in range(m):
        for j in range(k):
            a, b = d[c, :na, j], d[c, n - nb:, j]
            se = np.sqrt(_spectral0(a) / na + _spectral0(b) / nb)
            z = (a.mean() - b.mean()) / se if se > 0 else np.nan
            rows.append({"chain": c + 1, "parameter": nm[j], "z": z, "pvalue": 2 * stats.norm.sf(abs(z))})
    return pd.DataFrame(rows)


def autocorrelations(draws, lags: int = 40) -> np.ndarray:
    """(chains, lags+1, n)."""
    d = _as3d(draws)
    out = np.empty((d.shape[0], lags + 1, d.shape[2]))
    for c in range(d.shape[0]):
        for j in range(d.shape[2]):
            ac = _autocov(d[c, :, j])
            out[c, :, j] = ac[:lags + 1] / ac[0] if ac[0] > 0 else np.nan
    return out


def cumulative_means(draws) -> np.ndarray:
    d = _as3d(draws)
    return np.cumsum(d, axis=1) / np.arange(1, d.shape[1] + 1)[None, :, None]


def trace_data(draws, names=None) -> dict:
    d = _as3d(draws)
    return {nm: d[:, :, j] for j, nm in enumerate(_names(d, names))}


def summary(draws, names=None, level: float = 0.90) -> pd.DataFrame:
    """Mean, standard deviation, quantiles, ESS and, with two or more chains, PSRF."""
    d = _as3d(draws)
    nm = _names(d, names)
    x = d.reshape(-1, d.shape[2])
    q = np.quantile(x, [(1 - level) / 2, 0.5, 1 - (1 - level) / 2], axis=0)
    out = pd.DataFrame({"mean": x.mean(0), "std": x.std(0, ddof=1), "ci_low": q[0], "median": q[1], "ci_high": q[2],
                        "ess": ess(d)}, index=nm)
    if d.shape[0] > 1:
        out["rhat"] = gelman_rubin(d, nm)["R"]
    return out


# ---------------------------------------------------------------- prior x posterior
def prior_posterior_data(result, points: int = 200) -> dict:
    """Per parameter: grid, prior density and kernel density of the posterior (with the mode)."""
    pr = result.problem
    out = {}
    for j, (name, p) in enumerate(zip(pr.names, pr.priors)):
        post = None if result.draws is None else result.draws[:, :, j].ravel()
        lo, up = [], []
        if post is not None:
            lo.append(post.min()); up.append(post.max())
        if p is not None:
            a, b = (p._dist.ppf(0.001), p._dist.ppf(0.999))
            if p._ig1:
                a, b = np.sqrt(a), np.sqrt(b)
            lo.append(a); up.append(b)
        if not lo:
            lo, up = [result.mode[j] - 1], [result.mode[j] + 1]
        grid = np.linspace(max(min(lo), pr.lower[j]), min(max(up), pr.upper[j]), points)
        item = {"grid": grid, "mode": float(result.mode[j])}
        if p is not None:
            item["prior"] = np.exp([p.logpdf(v) for v in grid])
        if post is not None and np.ptp(post) > 0:
            item["posterior"] = stats.gaussian_kde(post)(grid)
        out[name] = item
    return out


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _grid(plt, k):
    cols = min(3, k)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel()[k:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def trace_plot(draws, names=None):
    """Trace plot (matplotlib figure) or, without matplotlib, the data of ``trace_data``."""
    data = trace_data(draws, names)
    plt = _pyplot()
    if plt is None:
        return data
    fig, axes = _grid(plt, len(data))
    for ax, (nm, v) in zip(axes, data.items()):
        for c in range(v.shape[0]):
            ax.plot(v[c], lw=0.6)
        ax.set_title(nm)
    fig.tight_layout()
    return fig


def prior_posterior_plot(result, points: int = 200):
    """Prior against posterior (matplotlib figure) or, without matplotlib, the data."""
    data = prior_posterior_data(result, points)
    plt = _pyplot()
    if plt is None:
        return data
    fig, axes = _grid(plt, len(data))
    for ax, (nm, v) in zip(axes, data.items()):
        if "prior" in v:
            ax.plot(v["grid"], v["prior"], color="0.5", label="prior")
        if "posterior" in v:
            ax.plot(v["grid"], v["posterior"], label="posterior")
        ax.axvline(v["mode"], ls="--", lw=0.8, color="k")
        ax.set_title(nm)
    axes[0].legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- mode
def check_optimum(result, points: int = 21, width: float = 3.0, relative: float | None = None) -> dict:
    """Profile of the log posterior, likelihood and prior around the mode, one parameter at a time.

    The grid of each parameter covers ``width`` Hessian standard deviations on each side (or
    ``relative`` x |mode| when the standard deviation does not exist or ``relative`` is given), cut
    at the bounds. In each table, the ``is_max`` attribute tells whether no grid point beats the
    mode. Equivalent to RISE's ``mode_curvature`` (``check_optimum`` is listed, but does not exist
    in the local copy).
    """
    pr = result.problem
    sd = np.sqrt(np.clip(np.diag(result.cov), 0, None)) if np.all(np.isfinite(result.cov)) else np.full(result.mode.size, np.nan)
    out = {}
    for j, name in enumerate(pr.names):
        half = width * sd[j] if relative is None and np.isfinite(sd[j]) and sd[j] > 0 else (relative or 0.1) * max(abs(result.mode[j]), 1e-2)
        lo, up = max(result.mode[j] - half, pr.lower[j]), min(result.mode[j] + half, pr.upper[j])
        grid = np.unique(np.append(np.linspace(lo, up, points), result.mode[j]))
        rows = []
        for v in grid:
            th = result.mode.copy()
            th[j] = v
            lpr = pr.logprior(th)
            ll = pr.loglik(th) if np.isfinite(lpr) else -np.inf
            lpost = pr.logpost(th)
            rows.append({"value": v, "log_post": lpost, "log_lik": ll, "log_prior": lpr})
        df = pd.DataFrame(rows)
        df.attrs["mode"] = float(result.mode[j])
        df.attrs["is_max"] = bool(df["log_post"].max() <= result.logpost_mode + 1e-6)
        out[name] = df
    return out


mode_curvature = check_optimum


# ---------------------------------------------------------------- identification
def _jacobian(fun, x, rel_step=1e-5):
    x = np.asarray(x, float)
    f0 = np.asarray(fun(x), float).ravel()
    J = np.empty((f0.size, x.size))
    for i in range(x.size):
        h = rel_step * max(abs(x[i]), 1e-2)
        e = np.zeros(x.size); e[i] = h
        J[:, i] = (np.asarray(fun(x + e), float).ravel() - np.asarray(fun(x - e), float).ravel()) / (2 * h)
    return J


def default_moments(problem, lags: int = 4):
    """Moments the sample identifies in the linear model: mean and autocovariances of the observables
    per regime, plus the transition matrix. With one regime, it is the vector of Iskrev (2010)."""
    from .analysis import theoretical_autocovariances
    m = problem.model
    obs = [m.endogenous.index(v) for v in m.spec.observables] if hasattr(m.spec, "observables") else None

    def moments(theta):
        sol = problem.solve(np.asarray(theta, float))
        idx = obs if obs is not None else list(range(sol.T.shape[1]))
        parts = [np.asarray(sol.Q, float).ravel()]
        for r in range(sol.T.shape[0]):
            ac = theoretical_autocovariances(sol, lags=lags, regime=r + 1)
            parts.append((sol.ss[r] + sol.c[r])[idx])
            parts.append(ac[np.ix_(idx, idx)].ravel())
        return np.concatenate(parts)
    return moments


def identification(problem_or_result, theta=None, method: str = "moments", moments=None, lags: int = 4,
                   tol: float = 1e-6) -> dict:
    """Simple local identification at the point ``theta``.

    ``method="moments"``: rank of the Jacobian of the moments with respect to the estimated
    parameters (Iskrev, 2010); ``moments`` may be a function ``theta -> vector`` (by default,
    ``default_moments``). ``method="hessian"``: rank of the Hessian of minus the log likelihood
    (observed information matrix).

    The rank is computed on the matrix with normalised columns; singular values below ``tol``
    times the largest count as zero. Returns the rank, the singular values, the condition number,
    the "strength" of each parameter (norm of the column after projecting out the others, relative
    to the norm of the column) and, if the rank is deficient, the unidentified parameter
    combinations (null-space vectors). With regime switching, the regime-conditional moments are
    not observed directly; full rank here is supporting evidence, not proof.
    """
    result = problem_or_result if hasattr(problem_or_result, "problem") else None
    pr = result.problem if result is not None else problem_or_result
    theta = np.asarray(result.mode if theta is None and result is not None else theta, float)
    if method == "moments":
        J = _jacobian(moments or default_moments(pr, lags), theta)
    elif method == "hessian":
        from .estimation import numerical_hessian
        J = numerical_hessian(lambda th: -pr.loglik(th), theta)
    else:
        raise ValueError("method must be 'moments' or 'hessian'")
    norms = np.linalg.norm(J, axis=0)
    Jn = J / np.where(norms > 0, norms, 1.0)
    u, s, vt = np.linalg.svd(Jn, full_matrices=False)
    rank = int(np.sum(s > tol * s[0])) if s.size and s[0] > 0 else 0
    strength = np.empty(theta.size)
    for i in range(theta.size):
        others = np.delete(Jn, i, axis=1)
        if norms[i] == 0:
            strength[i] = 0.0
            continue
        resid = Jn[:, i] - others @ np.linalg.lstsq(others, Jn[:, i], rcond=None)[0] if others.size else Jn[:, i]
        strength[i] = np.linalg.norm(resid)
    null = vt[rank:].T / np.where(norms > 0, norms, 1.0)[:, None] if rank < theta.size else np.empty((theta.size, 0))
    return {"rank": rank, "n_parameters": int(theta.size), "identified": rank == theta.size,
            "singular_values": s, "condition_number": float(s[0] / s[-1]) if s[-1] > 0 else np.inf,
            "strength": pd.Series(strength, index=pr.names),
            "null_space": pd.DataFrame(null, index=pr.names), "jacobian": J}
