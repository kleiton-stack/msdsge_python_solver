"""Analyses based on the 1st-order solution: moments, decompositions, forecasting and generalised IRF."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg

from .filtering import ergodic_distribution

VARDEC_PERIODS = (1, 4, 8, 16, 40, 100, 200, 400)   # RISE default (vardec_periods)


def _visible(solution, variables):
    m = solution.model
    names = [v for v in m.endogenous if not v.startswith("AUX_")] if variables is None else list(variables)
    return names, [m.endogenous.index(v) for v in names]


def ergodic_matrices(solution):
    """T and R (already scaled by the standard deviations) averaged over the ergodic distribution of the regimes.

    These are the matrices RISE (commit 1b2edfa) uses to compute the "aggregate" theoretical moments of
    a switching model: the initial covariance of the filter, ``theoretical_autocovariances`` and the
    variance decomposition with ``vardec_ergodic=true`` (identified by numerical comparison,
    validation/matlab/ref_tools.m). It is not the second moment of the ergodic distribution of the model
    (which would weight T kron T), only that of a linear model with the averaged matrices.
    """
    pi = ergodic_distribution(solution.Q)
    return np.einsum("r,rij->ij", pi, solution.T), np.einsum("r,rij->ij", pi, solution.R * solution.sigma)


def theoretical_autocovariances(solution, lags: int = 5, regime: int | None = None) -> np.ndarray:
    """Theoretical autocovariances. Returns (n, n, lags+1), with ``out[:, :, k] = Cov(y_t, y_{t-k})``.

    ``regime=None`` (default): RISE rule, a single matrix computed from T and R averaged over the
    ergodic distribution (see ``ergodic_matrices``); with one regime it is the usual covariance.
    ``regime=j``: conditional on staying in regime j. Variables follow ``model.endogenous``
    (RISE returns them in alphabetical order).
    """
    if regime is None:
        T, RS = ergodic_matrices(solution)
        label = "the model with averaged matrices"
    else:
        T, RS = solution.T[regime - 1], solution.R[regime - 1] * solution.sigma
        label = f"regime {regime} taken alone"
    if np.max(np.abs(np.linalg.eigvals(T))) >= 1 - 1e-10:
        raise ValueError(f"{label} is not stationary; there is no unconditional covariance")
    V = linalg.solve_discrete_lyapunov(T, RS @ RS.T)
    out = np.zeros(V.shape + (lags + 1,))
    out[:, :, 0] = (V + V.T) / 2
    for k in range(1, lags + 1):
        out[:, :, k] = T @ out[:, :, k - 1]
    return out


def theoretical_autocorrelations(solution, lags: int = 5, regime: int | None = None) -> np.ndarray:
    """Theoretical autocorrelations: ``out[i, j, k] = Corr(y_i,t, y_j,t-k)``. Same options as the autocovariances.

    Variables with zero variance get zero correlation (and one on their own diagonal at k = 0).
    """
    ac = theoretical_autocovariances(solution, lags, regime)
    sd = np.sqrt(np.clip(np.diag(ac[:, :, 0]), 0, None))
    scale = np.outer(sd, sd)
    out = np.divide(ac, scale[:, :, None], out=np.zeros_like(ac), where=scale[:, :, None] > 0)
    zero = sd == 0
    out[zero, zero, 0] = 1.0
    return out


def _vardec_table(T, RS, periods, names, idx, shocks):
    n, ne = RS.shape
    contrib = np.zeros((max(periods), n, ne))       # cumulative variance per shock
    imp = RS.copy()
    acc = np.zeros((n, ne))
    for k in range(max(periods)):
        acc = acc + imp ** 2
        contrib[k] = acc
        imp = T @ imp
    rows, keys = [], []
    stable = np.max(np.abs(np.linalg.eigvals(T))) < 1 - 1e-10
    inf = None
    if stable:
        inf = np.column_stack([np.diag(linalg.solve_discrete_lyapunov(T, np.outer(RS[:, e], RS[:, e]))) for e in range(ne)])
    for v, i in zip(names, idx):
        for k in periods:
            tot = contrib[k - 1, i].sum()
            rows.append(contrib[k - 1, i] / tot if tot > 0 else np.zeros(ne))
            keys.append((v, k))
        if inf is not None:
            tot = inf[i].sum()
            rows.append(inf[i] / tot if tot > 0 else np.zeros(ne))
            keys.append((v, np.inf))
    return pd.DataFrame(rows, columns=shocks, index=pd.MultiIndex.from_tuples(keys, names=["variable", "horizon"]))


def _vardec_simulated(solution, periods, names, idx, regime, draws, long_run, seed):
    """Simulation-based decomposition, with randomly drawn regime switches (see ``variance_decomposition``)."""
    m = solution.model
    rng = np.random.default_rng(seed)
    h, n = solution.T.shape[0], solution.T.shape[1]
    ne, H = len(m.exogenous), max(periods)
    RS = solution.R * solution.sigma
    pi = ergodic_distribution(solution.Q)
    cum = np.cumsum(solution.Q, axis=1)

    def next_regime(s):
        if regime is not None or h == 1:
            return s
        return int(min(np.searchsorted(cum[s], rng.uniform(), side="right"), h - 1))

    # k-step-ahead forecast error: given the regime path it is linear in the shocks, and the
    # part due to each shock is independent of the others
    acc = np.zeros((H, n, ne))
    for _ in range(draws):
        s = (regime - 1) if regime is not None else int(rng.choice(h, p=pi))
        part = np.zeros((n, ne))
        for k in range(H):
            s = next_regime(s)
            part = solution.T[s] @ part + RS[s] * rng.standard_normal(ne)
            acc[k] += part ** 2
    # infinite horizon: variance of each part in a long simulation
    s = (regime - 1) if regime is not None else int(rng.choice(h, p=pi))
    part, burn = np.zeros((n, ne)), min(1000, long_run // 10)
    s1, s2 = np.zeros((n, ne)), np.zeros((n, ne))
    for t in range(long_run + burn):
        s = next_regime(s)
        part = solution.T[s] @ part + RS[s] * rng.standard_normal(ne)
        if t >= burn:
            s1 += part
            s2 += part ** 2
    inf = s2 / long_run - (s1 / long_run) ** 2
    rows, keys = [], []
    for v, i in zip(names, idx):
        for k in list(periods) + [np.inf]:
            x = inf[i] if k == np.inf else acc[k - 1, i]
            tot = x.sum()
            rows.append(x / tot if tot > 0 and np.all(np.isfinite(x)) else np.zeros(ne))
            keys.append((v, k))
    return pd.DataFrame(rows, columns=m.exogenous, index=pd.MultiIndex.from_tuples(keys, names=["variable", "horizon"]))


def variance_decomposition(solution, periods=VARDEC_PERIODS, regime: int | None = None, variables=None,
                           ergodic: bool = False, theoretical: bool = True, draws: int = 2000,
                           long_run: int = 100000, seed: int = 0) -> dict:
    """Forecast-error variance decomposition by shock.

    Returns ``{key: DataFrame}`` with index (variable, horizon) and one column per shock, in
    shares that sum to one. The ``inf`` horizon is the unconditional decomposition (when it exists).

    - Default: one table per regime (key = regime number), conditional on staying in the regime.
    - ``ergodic=True`` (RISE's ``vardec_ergodic``): one table, key ``"ergodic"``, computed
      from T and R averaged over the ergodic distribution (RISE rule, see ``ergodic_matrices``).
    - ``theoretical=False``: by simulation. Regime paths (chain with the steady-state Q, ergodic
      start) and shock paths are drawn; given the regime path the forecast error is the sum of
      the parts due to each shock, and the share is the variance of each part over the
      sum. With ``ergodic=True`` (or without ``regime``) the regimes switch, which gives the
      decomposition of the actual ergodic distribution (key ``"ergodic"``); with ``regime=j`` it
      stays in the regime. RISE (commit 1b2edfa) has no simulated version ("not yet implemented").
    """
    m = solution.model
    names, idx = _visible(solution, variables)
    if not theoretical:
        if ergodic or regime is None:
            return {"ergodic": _vardec_simulated(solution, periods, names, idx, None, draws, long_run, seed)}
        return {regime: _vardec_simulated(solution, periods, names, idx, regime, draws, long_run, seed)}
    if ergodic:
        T, RS = ergodic_matrices(solution)
        return {"ergodic": _vardec_table(T, RS, periods, names, idx, m.exogenous)}
    out = {}
    for r in (range(solution.T.shape[0]) if regime is None else [regime - 1]):
        out[r + 1] = _vardec_table(solution.T[r], solution.R[r] * solution.sigma, periods, names, idx, m.exogenous)
    return out


def _hd_rise(solution, xs, eps, sm0, draws, seed):
    """RISE rule with regime switching: contribution = history - E[simulation without the shock]."""
    h, n = solution.T.shape[0], solution.T.shape[1]
    nT, ne = eps.shape
    RS = solution.R * solution.sigma
    Q = solution.Q
    const = np.stack([solution.ss[j] + solution.c[j] - solution.T[j] @ solution.ss[j] for j in range(h)])
    C = np.zeros((nT, n, ne))
    if draws is None:
        # exact expectation: m_j(t) = E[y_t 1(s_t = j)], Markov chain with Q and s_0 ~ sm0
        for k in range(ne):
            e = eps.copy()
            e[:, k] = 0.0
            p = sm0.copy()
            mom = p[:, None] * xs[0][None, :]
            for t in range(1, nT):
                pn = p @ Q
                prev = Q.T @ mom                         # sum_i Q[i, j] m_i
                mom = np.stack([solution.T[j] @ prev[j] + pn[j] * (const[j] + RS[j] @ e[t]) for j in range(h)])
                p = pn
                C[t, :, k] = xs[t] - mom.sum(0)
        return C
    rng = np.random.default_rng(seed)
    cum = np.cumsum(Q, axis=1)
    for k in range(ne):
        e = eps.copy()
        e[:, k] = 0.0
        acc = np.zeros((nT, n))
        for _ in range(draws):
            s = int(rng.choice(h, p=sm0))
            y = xs[0].copy()
            u = rng.uniform(size=nT)
            for t in range(1, nT):
                s = int(min(np.searchsorted(cum[s], u[t], side="right"), h - 1))
                y = const[s] + solution.T[s] @ y + RS[s] @ e[t]
                acc[t] += y
        C[1:, :, k] = xs[1:] - acc[1:] / draws
    return C


def historical_decomposition(solution, filtration, variables=None, method: str = "weighted",
                             draws: int | None = None, seed: int = 0) -> dict:
    """Historical decomposition of the smoothed variables into the contribution of each shock.

    With one regime it is exact and equal to RISE's under both methods. With regime switching it is approximate:

    - ``method="weighted"`` (default): the matrices of each period are the average of the regime
      matrices weighted by the smoothed probabilities, and the shocks are the expected smoothed ones.
    - ``method="rise"``: RISE rule (commit 1b2edfa), identified by numerical comparison
      (validation/matlab/ref_tools.m). The contribution of shock k is the smoothed history minus the
      mean of simulations run with all smoothed shocks except k, starting from the smoothed
      state of the first period, with the regime path drawn from the Markov chain
      (initial regime drawn from the smoothed probabilities of the first period). RISE uses 100
      simulations per shock, so its result changes on every call; here
      ``draws=None`` computes the exact expectation of those simulations and ``draws=100`` mimics RISE.
      This rule ignores the information in the data about the regimes after the first period and
      attributes to a shock differences that come only from regime switching (a shock that does not
      affect the variable gets a non-zero contribution). With endogenous probabilities it uses the
      steady-state Q.

    Under both methods the ``init`` column gathers the initial condition, the steady state and the
    approximation error, so that the columns sum to the smoothed variable. Returns ``{variable: DataFrame}``.
    """
    if filtration.smoothed_states is None:
        raise ValueError("the decomposition needs the smoothed states ('rise' filter)")
    m = solution.model
    names, idx = _visible(solution, variables)
    sm = filtration.smoothed_probs.to_numpy()
    eps = filtration.smoothed_shocks.to_numpy()
    xs = filtration.smoothed_states.to_numpy()
    RS = solution.R * solution.sigma
    nT, n, ne = xs.shape[0], solution.T.shape[1], RS.shape[2]
    if method == "rise":
        C = _hd_rise(solution, xs, eps, sm[0] / sm[0].sum(), draws, seed)
    elif method == "weighted":
        C = np.zeros((nT, n, ne))
        prev = np.zeros((n, ne))
        for t in range(nT):
            Tt = np.einsum("j,jik->ik", sm[t], solution.T)
            Rt = np.einsum("j,jik->ik", sm[t], RS)
            prev = Tt @ prev + Rt * eps[t]
            C[t] = prev
    else:
        raise ValueError("method must be 'weighted' or 'rise'")
    out = {}
    for v, i in zip(names, idx):
        df = pd.DataFrame(C[:, i, :], index=filtration.smoothed_states.index, columns=m.exogenous)
        df.insert(0, "init", xs[:, i] - C[:, i, :].sum(1))
        out[v] = df
    return out


def forecast(solution, filtration, steps: int = 12, variables=None) -> pd.DataFrame:
    """Point forecast from the last updated state, propagating the regime probabilities.

    At each step, the forecast of each regime starts from the expected state of the previous step and the
    result is the average weighted by the predicted probabilities (with endogenous probabilities, the
    transition matrix is kept at its steady-state value).
    """
    names, idx = _visible(solution, variables)
    x = filtration.filtered_states.to_numpy()[-1]
    mu = filtration.filtered_probs.to_numpy()[-1]
    h = solution.T.shape[0]
    rows, probs = [], []
    for _ in range(steps):
        mu = mu @ solution.Q
        xr = np.stack([solution.ss[k] + solution.c[k] + solution.T[k] @ (x - solution.ss[k]) for k in range(h)])
        x = mu @ xr
        rows.append(x[idx])
        probs.append(mu)
    df = pd.DataFrame(rows, columns=names, index=pd.RangeIndex(1, steps + 1, name="step"))
    for k in range(h):
        df[f"prob_regime_{k + 1}"] = np.array(probs)[:, k]
    return df


def girf(solution, shock: str, periods: int = 40, draws: int = 500, size: float | None = None,
         regime0: int | None = None, seed: int = 0, variables=None) -> pd.DataFrame:
    """Generalised IRF: average, over regime and shock paths, of the difference between the
    simulation with and without the impulse in period 0 (same random numbers in both).

    ``regime0`` fixes the initial regime; without it, it is drawn from the ergodic distribution. With
    endogenous probabilities the regime paths of the two simulations may diverge.
    """
    m = solution.model
    names, idx = _visible(solution, variables)
    k = m.exogenous.index(shock)
    amp = solution.sigma[k] if size is None else size
    h, n = solution.T.shape[0], solution.T.shape[1]
    ne = len(m.exogenous)
    rng = np.random.default_rng(seed)
    pi = ergodic_distribution(solution.Q)
    acc = np.zeros((periods, n))

    def step_regime(r, y, y_prev, u):
        row = solution.Q[r]
        if solution.endogenous_switching:
            row = np.clip(m.transition_matrix(y, y, y_prev, solution.params[r], solution.ss[r], regime=r), 0, 1)
            row = row / row.sum()
        return int(np.searchsorted(np.cumsum(row), u, side="right").clip(0, h - 1))

    for _ in range(draws):
        r0 = (regime0 - 1) if regime0 else int(rng.choice(h, p=pi))
        e = rng.standard_normal((periods, ne)) * solution.sigma
        u = rng.uniform(size=periods)
        paths = []
        for impulse in (0.0, amp):
            r, y_prev = r0, solution.ss[r0].copy()
            path = np.zeros((periods, n))
            for t in range(periods):
                et = e[t].copy()
                if t == 0:
                    et[k] += impulse
                y = solution.ss[r] + solution.c[r] + solution.T[r] @ (y_prev - solution.ss[r]) + solution.R[r] @ et
                path[t] = y
                r_next = step_regime(r, y, y_prev, u[t]) if h > 1 else 0
                y_prev, r = y, r_next
            paths.append(path)
        acc += paths[1] - paths[0]
    return pd.DataFrame(acc[:, idx] / draws, columns=names, index=pd.RangeIndex(periods, name="period"))
