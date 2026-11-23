"""Forecasting with uncertainty, conditional forecasting, counterfactuals and simulation with imposed paths.

Everything is based on the first-order solution ``y_t = ss_r + c_r + T_r (y_{t-1} - ss_r) + R_r sigma e_t``
(plus the anticipated-shock terms, if the model was solved with ``shock_horizon > 0``).
Shocks are always measured in standard deviations (``e ~ N(0, 1)``), as in the RISE outputs.

- ``simulate_path``: deterministic simulation given the path of regimes and of shocks.
- ``forecast``: bands by simulation with shock and regime uncertainty; hard or interval
  conditions on endogenous variables and imposed values for shocks.
- ``counterfactual``: history rebuilt with a subset of the smoothed shocks, with another
  solution (another policy) or with another regime path.

Conditional forecasting (Waggoner and Zha, 1999; Maih, 2010): given the regime path, the path of
the variables is linear in the stacked vector of shocks, ``y = b + M e``. The hard conditions give
``D e = r``; under ``e ~ N(0, I)``, the conditional distribution is normal with mean ``D^+ r``
(minimum-norm solution) and covariance ``I - D^+ D``. Interval conditions ("soft", Maih, 2010) are
drawn from a truncated multivariate normal and then treated as hard. Validated against RISE's
``forecast`` (validation/matlab/ref_tools.m, tests/test_forecasting.py).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

SAMPLERS = ("rejection", "gibbs")


# --------------------------------------------------------------------------- basic blocks
def _step(sol, r, y_prev, e, future=None):
    """One step in regime r (0-based). ``future``: shocks of t+1, t+2, ... already known at t."""
    y = sol.ss[r] + sol.c[r] + sol.T[r] @ (y_prev - sol.ss[r]) + sol.R[r] @ (sol.sigma * e)
    if future is not None:
        for k in range(min(sol.R_ant.shape[0], len(future))):
            y = y + sol.R_ant[k][r] @ (sol.sigma * future[k])
    return y


def simulate_path(solution, y0, shocks, regimes, anticipated: bool = False, variables=None) -> pd.DataFrame:
    """Simulate with imposed shocks and regimes.

    ``y0``: vector of the endogenous variables in the period before the first one (or a
    dictionary/Series by name; whatever is missing stays at the steady state of the first regime of
    the path). ``shocks``: matrix (H, n_exo) in standard deviations (or a DataFrame with shock
    names). ``regimes``: vector of length H, 1-based (or an integer). ``anticipated=True``: in each
    period agents already know the shocks of the next ``shock_horizon`` periods of the ``shocks``
    matrix itself.
    """
    m = solution.model
    E = _shock_matrix(solution, shocks)
    H = E.shape[0]
    reg = _regime_path(solution, regimes, H)
    y = _initial_state(solution, y0, reg[0])
    out = np.zeros((H, m.n))
    K = solution.R_ant.shape[0] if anticipated else 0
    for t in range(H):
        y = _step(solution, reg[t], y, E[t], E[t + 1:t + 1 + K] if K else None)
        out[t] = y
    names = [v for v in m.endogenous if not v.startswith("AUX_")] if variables is None else list(variables)
    df = pd.DataFrame(out[:, [m.endogenous.index(v) for v in names]], columns=names,
                      index=pd.RangeIndex(1, H + 1, name="step"))
    df.insert(0, "regime", reg + 1)
    return df


def _shock_matrix(solution, shocks):
    exo = solution.model.exogenous
    if isinstance(shocks, pd.DataFrame):
        unknown = [c for c in shocks.columns if c not in exo]
        if unknown:
            raise ValueError(f"unknown shocks: {unknown}")
        E = np.zeros((len(shocks), len(exo)))
        for c in shocks.columns:
            E[:, exo.index(c)] = shocks[c].to_numpy(float)
        return np.nan_to_num(E)
    E = np.atleast_2d(np.asarray(shocks, float))
    if E.shape[1] != len(exo):
        raise ValueError(f"shocks must have {len(exo)} columns, in the order {exo}")
    return E


def _regime_path(solution, regimes, H):
    h = solution.T.shape[0]
    reg = np.broadcast_to(np.asarray(regimes, int), (H,)) if np.ndim(regimes) == 0 else np.asarray(regimes, int)
    if reg.size != H:
        raise ValueError(f"the regime path needs {H} elements")
    if reg.min() < 1 or reg.max() > h:
        raise ValueError(f"regimes must be between 1 and {h}")
    return reg - 1


def _initial_state(solution, y0, r):
    m = solution.model
    if isinstance(y0, (dict, pd.Series)):
        y = solution.ss[r].copy()
        for k, v in dict(y0).items():
            if k in m.endogenous and np.isfinite(v):
                y[m.endogenous.index(k)] = float(v)
        return y
    y = np.asarray(y0, float).ravel()
    if y.size != m.n:
        raise ValueError(f"y0 needs {m.n} elements, in the order of model.endogenous")
    return y.copy()


def _linear_map(solution, y0, reg, anticipated):
    """Path without shocks ``b`` (H, n) and response ``M`` (H, n, H, n_exo) to the stacked shocks."""
    H, n, ne = reg.size, solution.T.shape[1], solution.R.shape[2]
    K = solution.R_ant.shape[0] if anticipated else 0
    b, M = np.zeros((H, n)), np.zeros((H, n, H, ne))
    y, prev = y0, np.zeros((n, H, ne))
    for t in range(H):
        r = reg[t]
        y = solution.ss[r] + solution.c[r] + solution.T[r] @ (y - solution.ss[r])
        b[t] = y
        cur = np.einsum("ij,jke->ike", solution.T[r], prev)
        cur[:, t, :] += solution.R[r] * solution.sigma
        for k in range(K):
            if t + 1 + k < H:
                cur[:, t + 1 + k, :] += solution.R_ant[k][r] * solution.sigma
        M[t] = cur
        prev = cur
    return b, M


def _draw_regimes(solution, probs0, H, rng, regime0=None):
    """Draw s_0 from ``probs0`` (or use ``regime0``) and then the chain with the steady-state Q."""
    h = solution.T.shape[0]
    reg = np.zeros(H, int)
    s = int(rng.choice(h, p=probs0)) if regime0 is None else regime0 - 1
    for t in range(H):
        s = int(rng.choice(h, p=solution.Q[s]))
        reg[t] = s
    return reg


# --------------------------------------------------------------------------- truncated normal
def truncated_mvn(mean, cov, lower, upper, rng, algo: str = "rejection", burnin: int = 100,
                  thinning: int = 1, max_attempts: int = 500) -> np.ndarray:
    """One draw from ``N(mean, cov)`` truncated to ``[lower, upper]``.

    'rejection': draws from the normal until it falls in the region (at most ``max_attempts``; if
    it fails, it warns and uses Gibbs). 'gibbs': coordinate-by-coordinate Gibbs sampler in the
    whitened space, with ``burnin`` and ``thinning``.
    """
    mean, lower, upper = (np.asarray(a, float) for a in (mean, lower, upper))
    cov = np.atleast_2d(cov)
    w, V = np.linalg.eigh((cov + cov.T) / 2)
    keep = w > 1e-12 * max(w.max(), 1e-300)
    L = V[:, keep] * np.sqrt(w[keep])                       # x = mean + L z
    if algo == "rejection":
        for _ in range(max_attempts):
            x = mean + L @ rng.standard_normal(L.shape[1])
            if np.all(x >= lower - 1e-12) and np.all(x <= upper + 1e-12):
                return x
        warnings.warn("rejection sampling found no point in the region; using Gibbs", RuntimeWarning)
    elif algo != "gibbs":
        raise ValueError(f"unknown sampler: {algo!r}; options: {SAMPLERS}")
    k = L.shape[1]
    # feasible starting point: projection of the centre of the region
    target = np.clip(mean, lower, upper)
    z = np.linalg.lstsq(L, target - mean, rcond=None)[0]
    lo, hi = lower - mean, upper - mean
    for it in range(burnin + thinning):
        for j in range(k):
            rest = L @ z - L[:, j] * z[j]
            a, b = -np.inf, np.inf
            for i in range(L.shape[0]):
                lij = L[i, j]
                if abs(lij) < 1e-14:
                    continue
                u1, u2 = (lo[i] - rest[i]) / lij, (hi[i] - rest[i]) / lij
                a, b = max(a, min(u1, u2)), min(b, max(u1, u2))
            if a > b:               # empty region in this coordinate: keep the point
                continue
            z[j] = stats.truncnorm.rvs(a, b, random_state=rng)
    return mean + L @ z


# --------------------------------------------------------------------------- forecast
@dataclass
class ForecastResult:
    mean: pd.DataFrame            # (H, variables)
    quantiles: pd.DataFrame       # index (quantile, step)
    shocks: pd.DataFrame          # mean of the shocks (standard deviations), (H, shocks)
    regime_probs: pd.DataFrame    # frequency of each regime per step
    draws: np.ndarray             # (draws, H, variables)
    shock_draws: np.ndarray       # (draws, H, n_exo)
    regime_draws: np.ndarray      # (draws, H), 1-based
    variables: list

    def band(self, variable: str, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
        """Mean and two quantiles of a variable (computed from the draws)."""
        k = self.variables.index(variable)
        q = np.quantile(self.draws[:, :, k], [lower, upper], axis=0)
        return pd.DataFrame({"mean": self.mean[variable].to_numpy(), "lower": q[0], "upper": q[1]},
                            index=self.mean.index)


def _origin(solution, origin, start):
    """Return (y0, probs0, future index or None, position of the origin)."""
    h = solution.T.shape[0]
    if hasattr(origin, "filtered_states"):
        f = origin
        idx = f.filtered_states.index
        if start is None:
            return f.filtered_states.to_numpy()[-1], f.filtered_probs.to_numpy()[-1], idx, len(idx) - 1
        pos = idx.get_loc(start) if not isinstance(start, (int, np.integer)) else int(start)
        if pos < 1:
            raise ValueError("the forecast needs at least one period before 'start'")
        states = f.smoothed_states if f.smoothed_states is not None else f.filtered_states
        return states.to_numpy()[pos - 1], f.smoothed_probs.to_numpy()[pos - 1], idx, pos - 1
    if isinstance(origin, tuple):
        y0, p0 = origin
        return y0, np.asarray(p0, float), None, None
    from .filtering import ergodic_distribution
    return origin, (ergodic_distribution(solution.Q) if h > 1 else np.ones(1)), None, None


def _future_index(idx, pos, H):
    if idx is None:
        return pd.RangeIndex(1, H + 1, name="step")
    if isinstance(idx, pd.PeriodIndex):
        return pd.period_range(idx[pos] + 1, periods=H, freq=idx.freq)
    if isinstance(idx, pd.RangeIndex) or pd.api.types.is_integer_dtype(idx):
        return pd.RangeIndex(int(idx[pos]) + 1, int(idx[pos]) + 1 + H, name=idx.name)
    return pd.RangeIndex(1, H + 1, name="step")


def _cond_array(spec, names, H, what):
    """Convert conditions (DataFrame, dict of vectors) into an (H, len(names)) matrix with NaN where there is none."""
    out = np.full((H, len(names)), np.nan)
    if spec is None:
        return out
    items = spec.items() if isinstance(spec, dict) else ((c, spec[c].to_numpy()) for c in spec.columns)
    for k, v in items:
        if k not in names:
            raise ValueError(f"unknown {what} in the conditions: {k!r}")
        v = np.atleast_1d(np.asarray(v, float))[:H]
        out[:v.size, names.index(k)] = v
    return out


def forecast(solution, origin, steps: int = 12, draws: int = 1000, shock_uncertainty: bool = True,
             regime_uncertainty: bool = True, regimes=None, cond_endo=None, cond_exo=None,
             cond_lower=None, cond_upper=None, free_shocks=None, start=None, anticipated: bool = False,
             quantiles=(0.05, 0.16, 0.5, 0.84, 0.95), sampling_algo: str = "rejection",
             sampling_burnin: int = 100, sampling_thinning: int = 1, max_rejection_attempts: int = 500,
             seed: int = 0, variables=None) -> ForecastResult:
    """Forecast by simulation, with or without conditions.

    ``origin``: filter result (starts from the last updated state and the last filtered
    probabilities; with ``start``, label or position of the first forecast period, it starts from
    the smoothed state and probabilities of the previous period, like ``forecast_start_date`` in
    RISE), the pair ``(y0, probs0)`` or just ``y0`` (ergodic probabilities).

    Uncertainty: ``shock_uncertainty`` draws future shocks; ``regime_uncertainty`` draws the
    regime path (starting regime drawn from the probabilities at the origin; steady-state Q).
    Without regime uncertainty and without ``regimes``, it uses the most likely regime at the
    origin in every step. ``regimes`` imposes the path (1-based), like ``simul_regime``.

    Conditions (dictionary ``{name: vector}`` or DataFrame, one row per step, NaN = free):
    ``cond_endo`` exact values of endogenous variables; ``cond_lower``/``cond_upper`` intervals;
    ``cond_exo`` values of shocks in standard deviations. ``free_shocks``: shocks that may adjust
    to meet the conditions (default: all those not in ``cond_exo``). Shocks outside
    ``free_shocks`` and ``cond_exo`` stay at zero when there are conditions on endogenous variables.
    ``anticipated=True`` uses the anticipated-shock terms of the solution (Maih, 2010): the shocks
    that meet the conditions are known ``shock_horizon`` periods ahead.

    With endogenous probabilities the regime path is drawn with the steady-state Q.
    """
    m = solution.model
    exo, n, ne, h = m.exogenous, m.n, len(m.exogenous), solution.T.shape[0]
    H = int(steps)
    rng = np.random.default_rng(seed)
    y0_raw, probs0, idx, pos = _origin(solution, origin, start)
    probs0 = np.clip(np.asarray(probs0, float), 0, None)
    probs0 = probs0 / probs0.sum()
    names = [v for v in m.endogenous if not v.startswith("AUX_")] if variables is None else list(variables)
    sel = [m.endogenous.index(v) for v in names]

    CE = _cond_array(cond_endo, m.endogenous, H, "variable")
    LO = _cond_array(cond_lower, m.endogenous, H, "variable")
    UP = _cond_array(cond_upper, m.endogenous, H, "variable")
    CX = _cond_array(cond_exo, exo, H, "shock")
    hard, soft = np.isfinite(CE), (np.isfinite(LO) | np.isfinite(UP)) & ~np.isfinite(CE)
    fixed = np.isfinite(CX)
    has_endo = hard.any() or soft.any()
    free_cols = np.ones(ne, bool)
    if free_shocks is not None:
        free_cols = np.array([s in free_shocks for s in exo])
    free = np.broadcast_to(free_cols, (H, ne)) & ~fixed
    if anticipated and solution.R_ant.shape[0] == 0:
        raise ValueError("solve with shock_horizon > 0 to use anticipated=True")
    stochastic = shock_uncertainty or soft.any() or (regime_uncertainty and regimes is None and h > 1)
    nd = int(draws) if stochastic else 1

    Y, E, S = np.zeros((nd, H, n)), np.zeros((nd, H, ne)), np.zeros((nd, H), int)
    fixed_path = None if regimes is None else _regime_path(solution, regimes, H)
    cache = {}
    for d in range(nd):
        if fixed_path is not None:
            reg = fixed_path
        elif regime_uncertainty and h > 1:
            reg = _draw_regimes(solution, probs0, H, rng)
        else:
            reg = np.full(H, int(np.argmax(probs0)))
        key = reg.tobytes()
        if key not in cache:
            y0 = _initial_state(solution, y0_raw, reg[0])
            cache[key] = _linear_map(solution, y0, reg, anticipated)
        b, M = cache[key]
        e = np.zeros((H, ne))
        e[fixed] = CX[fixed]
        if not has_endo:
            if shock_uncertainty:
                z = rng.standard_normal((H, ne))
                e[~fixed] = z[~fixed]
        else:
            Mf = M[:, :, free]                                   # (H, n, n_free)
            base = b + np.einsum("tike,ke->ti", M, e)
            Dh, rh = Mf[hard], (CE - base)[hard]
            if hard.any():
                Dh_pinv = np.linalg.pinv(Dh)
                mu = Dh_pinv @ rh
                if np.max(np.abs(Dh @ mu - rh)) > 1e-8 * (1 + np.max(np.abs(rh))):
                    raise ValueError("the hard conditions cannot be met with the free shocks")
                Pn = np.eye(Mf.shape[2]) - Dh_pinv @ Dh          # projector onto the null space
            else:
                mu, Pn = np.zeros(Mf.shape[2]), np.eye(Mf.shape[2])
            if soft.any():
                Ds = Mf[soft]
                lo = np.where(np.isfinite(LO), LO, -np.inf)[soft] - base[soft]
                up = np.where(np.isfinite(UP), UP, np.inf)[soft] - base[soft]
                vs = truncated_mvn(Ds @ mu, Ds @ Pn @ Ds.T, lo, up, rng, sampling_algo, sampling_burnin,
                                   sampling_thinning, max_rejection_attempts)
                Dall, rall = np.vstack([Dh, Ds]), np.concatenate([rh, vs])
                Dp = np.linalg.pinv(Dall)
                mu, Pn = Dp @ rall, np.eye(Mf.shape[2]) - Dp @ Dall
            ef = mu + (Pn @ rng.standard_normal(mu.size) if shock_uncertainty else 0.0)
            e[free] = ef
        Y[d] = b + np.einsum("tike,ke->ti", M, e)
        E[d], S[d] = e, reg + 1

    fidx = _future_index(idx, pos, H)
    mean = pd.DataFrame(Y[:, :, sel].mean(0), index=fidx, columns=names)
    qs = np.quantile(Y[:, :, sel], list(quantiles), axis=0)
    qdf = pd.concat({q: pd.DataFrame(qs[k], index=fidx, columns=names) for k, q in enumerate(quantiles)},
                    names=["quantile"])
    probs = pd.DataFrame({f"regime_{j + 1}": (S == j + 1).mean(0) for j in range(h)}, index=fidx)
    return ForecastResult(mean=mean, quantiles=qdf, shocks=pd.DataFrame(E.mean(0), index=fidx, columns=exo),
                          regime_probs=probs, draws=Y[:, :, sel], shock_draws=E, regime_draws=S, variables=names)


# --------------------------------------------------------------------------- counterfactual
def counterfactual(solution, filtration, shocks=None, alt_solution=None, regimes="mode", draws: int = 100,
                   seed: int = 0, variables=None) -> pd.DataFrame:
    """Counterfactual history from the smoothed states and shocks.

    Starts from the smoothed state of the first period and reapplies the smoothed shocks.

    - ``shocks``: list of the shocks that are kept (the others are set to zero). Default: all,
      which, with one regime, reproduces the smoothed history.
    - ``alt_solution``: another solution (other parameters or another policy rule) fed with the
      same shocks.
    - ``regimes``: 'mode' (most likely regime according to the smoothed probabilities, as in the
      tutorial of Maih et al., 2026, §6), 'draws' (mean of ``draws`` paths drawn, period by
      period, from the smoothed probabilities; it is the Monte Carlo integration that RISE
      describes for switching models), an integer or a 1-based vector.

    With regime switching an exact replica of history is not possible, because the regime is latent.
    Returns a DataFrame with the counterfactual variables, on the index of the data.
    """
    if filtration.smoothed_states is None or filtration.smoothed_shocks is None:
        raise ValueError("the counterfactual needs smoothed states and shocks ('rise' filter)")
    sol = solution if alt_solution is None else alt_solution
    m = solution.model
    xs = filtration.smoothed_states.to_numpy()
    es = filtration.smoothed_shocks.to_numpy().copy()
    if shocks is not None:
        unknown = [s for s in shocks if s not in m.exogenous]
        if unknown:
            raise ValueError(f"unknown shocks: {unknown}")
        for k, s in enumerate(m.exogenous):
            if s not in shocks:
                es[:, k] = 0.0
    sm = filtration.smoothed_probs.to_numpy()
    nT, h = xs.shape[0], sol.T.shape[0]
    if sol.model.endogenous != m.endogenous or sol.model.exogenous != m.exogenous:
        raise ValueError("alt_solution must have the same variables and shocks")
    if isinstance(regimes, str):
        if regimes == "mode" or h == 1:
            paths = [np.argmax(sm, axis=1)] if sm.shape[1] == h else [np.zeros(nT, int)]
        elif regimes == "draws":
            rng = np.random.default_rng(seed)
            cum = np.cumsum(sm, axis=1)
            paths = [np.minimum((rng.uniform(size=(nT, 1)) > cum).sum(1), h - 1) for _ in range(draws)]
        else:
            raise ValueError("regimes must be 'mode', 'draws', an integer or a vector")
    else:
        paths = [_regime_path(sol, regimes, nT)]
    acc = np.zeros_like(xs)
    for reg in paths:
        y = xs[0].copy()
        out = np.zeros_like(xs)
        out[0] = y
        for t in range(1, nT):
            y = _step(sol, reg[t], y, es[t])
            out[t] = y
        acc += out
    acc /= len(paths)
    names = [v for v in m.endogenous if not v.startswith("AUX_")] if variables is None else list(variables)
    return pd.DataFrame(acc[:, [m.endogenous.index(v) for v in names]], columns=names,
                        index=filtration.smoothed_states.index)
