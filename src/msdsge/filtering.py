"""Filters and likelihood for linear regime-switching models.

- ``rise`` (default): recursion that reproduces the RISE filter (commit 1b2edfa, 2019) with
  ``kf_riccati_tol=0``, identified by numerical comparison (validation/explore_filter.py):
  initial state and covariance common to all regimes, with the Lyapunov covariance computed from
  T and R averaged over the ergodic distribution; the first observation uses the initial state as the
  prediction; means and covariances are mixed across regimes without the dispersion-of-means term.
- ``imm``: Interacting Multiple Model (Blom and Bar-Shalom, 1988), the default in recent RISE versions
  according to Maih et al. (2026, §4.1): mixing before prediction, h Kalman steps per period.
- ``kim``: Kim (1994) / Kim and Nelson (1999), GPB(2): h² Kalman steps, then collapse.
- With a single regime they all reduce to the Kalman filter.

State in levels: ``x_t = ss_j + T_j (x_{t-1} - ss_j) + R_j diag(sigma) e_t`` when ``s_t = j``.
Measurement: ``y_t = x_t[obs] + v_t``, ``v_t ~ N(0, H)`` (H = 0 by default). Missing data (NaN)
are handled by dropping the corresponding observation row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg

from ._kernels import HAVE_NUMBA, rise_filter_kernel

FILTERS = ("rise", "imm", "kim")
_LOG2PI = np.log(2 * np.pi)


@dataclass
class FilterResult:
    loglik: float
    loglik_t: np.ndarray            # (n_obs,)
    filtered_probs: pd.DataFrame    # Pr(s_t | Y_t)
    predicted_probs: pd.DataFrame   # Pr(s_t | Y_{t-1})
    smoothed_probs: pd.DataFrame    # Pr(s_t | Y_T), Kim (1994) recursion
    filtered_states: pd.DataFrame   # E[x_t | Y_t], combined across regimes
    filter: str
    smoothed_states: pd.DataFrame | None = None   # E[x_t | Y_T]
    smoothed_shocks: pd.DataFrame | None = None   # E[e_t | Y_T], in standard-deviation units
    transition_matrices: np.ndarray | None = None   # (n_obs, h, h): Q used in the transition from t-1 to t


def ergodic_distribution(Q: np.ndarray) -> np.ndarray:
    h = Q.shape[0]
    A = np.vstack([Q.T - np.eye(h), np.ones(h)])
    pi, *_ = np.linalg.lstsq(A, np.r_[np.zeros(h), 1.0], rcond=None)
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


def _initial_cov(T, RS, diffuse=10.0):
    """Unconditional covariance of the regime (Lyapunov); if it does not exist, diag(diffuse)."""
    n = T.shape[0]
    if np.max(np.abs(np.linalg.eigvals(T))) < 1 - 1e-9:
        try:
            P = linalg.solve_discrete_lyapunov(T, RS @ RS.T)
            return (P + P.T) / 2
        except (linalg.LinAlgError, ValueError):
            pass
    try:   # unit roots: diffuse Lyapunov, like RISE's diffuse_lyapunov_equation
        from .filter_extras import diffuse_lyapunov
        return diffuse_lyapunov(T, RS @ RS.T, diffuse)
    except (linalg.LinAlgError, ValueError):
        return np.eye(n) * diffuse


def _kalman_step(a0, P0, ss, T, RRt, y, Z, H, c=0.0):
    """One prediction and update step. Returns (a, P, log predictive density)."""
    a = ss + c + T @ (a0 - ss)
    P = T @ P0 @ T.T + RRt
    ok = np.isfinite(y)
    if not ok.any():
        return a, P, 0.0
    Zt = Z[ok]
    v = y[ok] - a[Zt]
    PZ = P[:, Zt]
    F = PZ[Zt] + H[np.ix_(ok, ok)]
    F = (F + F.T) / 2
    try:
        c, low = linalg.cho_factor(F)
    except linalg.LinAlgError:
        return a, P, -np.inf
    Fv = linalg.cho_solve((c, low), v)
    ll = -0.5 * (v.size * _LOG2PI + 2 * np.sum(np.log(np.diag(c))) + v @ Fv)
    a = a + PZ @ Fv
    P = P - PZ @ linalg.cho_solve((c, low), PZ.T)
    return a, (P + P.T) / 2, ll


def _collapse(w, a, P):
    """Collapses a Gaussian mixture (weights w, means a[k], covariances P[k]) into a single moment."""
    m = w @ a
    d = a - m
    return m, np.einsum("k,kij->ij", w, P) + np.einsum("k,ki,kj->ij", w, d, d)


def run_filter(solution, data: pd.DataFrame, filter: str = "rise", measurement_errors=None,
               init_probs=None, diffuse: float = 10.0, smooth: bool = True, riccati_tol: float = 0.0,
               presample: int = 0, init_variance: float | None = None, user_init=None) -> FilterResult:
    """RISE-related options: ``riccati_tol`` (kf_riccati_tol, "rise" filter only), ``presample``
    (kf_presample: initial increments left out of ``loglik``), ``init_variance`` (kf_init_variance),
    ``user_init`` (kf_user_init), ``diffuse`` (lyapunov_diffuse_factor); see filter_extras.py.
    Filters "ukf", "ckf" and "ddf" (higher-order solutions): see nonlinear_filters.py."""
    if filter in ("ukf", "ckf", "ddf"):
        from .nonlinear_filters import run_nonlinear_filter
        return run_nonlinear_filter(solution, data, filter=filter, measurement_errors=measurement_errors,
                                    init_probs=init_probs, diffuse=diffuse, presample=presample,
                                    init_variance=init_variance, user_init=user_init)
    if filter not in FILTERS:
        raise ValueError(f"unknown filter: {filter!r}; options: {FILTERS}")
    m = solution.model
    obs = list(m.spec.observables)
    if not obs:
        raise ValueError("the model declares no observables (varobs)")
    missing = [v for v in obs if v not in data.columns]
    if missing:
        raise ValueError(f"observables missing from the data: {missing}")
    # observed exogenous variables (e.g. a deterministic trend): they enter as data, not as shocks
    xobs = [v for v in obs if v not in m.endogenous]
    obs = [v for v in obs if v in m.endogenous]
    Y = data[obs].to_numpy(float)
    Z = np.array([m.endogenous.index(v) for v in obs])
    H = np.zeros((len(obs), len(obs)))
    if measurement_errors is None:
        # RISE convention: parameter stderr_X is the std of the measurement error of observable X
        measurement_errors = {v: solution.params[0, m.parameters.index(f"stderr_{v}")]
                              for v in obs if f"stderr_{v}" in m.parameters}
    for k, v in (measurement_errors or {}).items():
        H[obs.index(k), obs.index(k)] = float(v) ** 2
    h, n = solution.T.shape[0], solution.T.shape[1]
    T, ss, cst = solution.T, solution.ss, solution.c
    RS = solution.R * solution.sigma
    Dx = np.zeros((Y.shape[0] + 1, h, n))     # contribution of the observed exogenous variables to the state at t
    if xobs:
        kx = [m.exogenous.index(v) for v in xobs]
        X = data[xobs].to_numpy(float)
        if not np.all(np.isfinite(X)):
            raise ValueError(f"observed exogenous variables cannot have missing data: {xobs}")
        Dx[:-1] = np.einsum("rik,tk->tri", solution.R[:, :, kx], X)
        RS = RS.copy()
        RS[:, :, kx] = 0.0
    RRt = np.einsum("rik,rjk->rij", RS, RS)
    Q = solution.Q
    endo = solution.endogenous_switching and h > 1

    mu = ergodic_distribution(Q) if init_probs is None else np.asarray(init_probs, float)
    if filter == "rise":
        P0 = _initial_cov(np.einsum("r,rij->ij", mu, T), np.einsum("r,rij->ij", mu, RS), diffuse)
        a, P = np.tile(mu @ ss, (h, 1)) + Dx[0], np.stack([P0] * h)   # predictions for t = 0
    else:
        a = ss.copy()
        P = np.stack([_initial_cov(T[j], RS[j], diffuse) for j in range(h)])
    if init_variance is not None or user_init is not None:
        from .filter_extras import initial_conditions
        a, P, mu = initial_conditions(a, P, mu, list(m.endogenous), init_variance, user_init)
    from .filter_extras import HistoryStore, RiccatiShortcut, smooth_histories
    ric = RiccatiShortcut(riccati_tol, Y, RS) if riccati_tol > 0 and filter == "rise" else None
    hist = HistoryStore(Y.shape[0], h, n, filter == "kim") if smooth and filter != "rise" else None
    a_prev = mu @ a

    nT = Y.shape[0]
    store_a, store_P = np.zeros((nT, h, n)), np.zeros((nT, h, n, n))   # regime-specific predictions ("rise" filter)
    ll_t = np.zeros(nT)
    filt, pred = np.zeros((nT, h)), np.zeros((nT, h))
    Qs = np.zeros((nT, h, h))
    Qs[0] = Q
    states = np.zeros((nT, n))
    fast = filter == "rise" and not endo and HAVE_NUMBA and np.all(np.isfinite(Y)) and not xobs and ric is None
    if fast:
        ll_t, filt, pred, states, store_a, store_P, ok = rise_filter_kernel(
            np.ascontiguousarray(Y), Z, H, np.ascontiguousarray(T), np.ascontiguousarray(RRt),
            np.ascontiguousarray(ss), np.ascontiguousarray(cst), np.ascontiguousarray(Q), mu, a, P)
        if not ok:
            return _failed(filter, data, nT, h, n, m)
        Qs[:] = Q
    for t in range(nT if not fast else 0):
        if filter == "rise":
            # a, P are the predictions of each regime; mu = Pr(s_t | Y_{t-1})
            pred[t] = mu
            store_a[t], store_P[t] = a, P
            au, Pu, lam = np.zeros_like(a), np.zeros_like(P), np.zeros(h)
            for j in range(h):
                au[j], Pu[j], lam[j] = _kalman_step(a[j], P[j], a[j] * 0, np.eye(n), 0 * P[j], Y[t], Z, H)
            if ric is not None:
                au, Pu, lam = ric.step(t, a, P, au, Pu, lam, Y[t], Z, H)
            mx = np.max(lam)
            if not np.isfinite(mx):
                return _failed(filter, data, nT, h, n, m)
            w = mu * np.exp(lam - mx)
            ll_t[t] = mx + np.log(w.sum())
            post = w / w.sum()
            filt[t], states[t] = post, post @ au
            if endo:
                prev = states[t - 1] if t else states[t]
                # As in RISE (commit 1b2edfa): the whole Q matrix is evaluated at the updated state of the
                # FIRST regime, not at that of each origin regime nor at the expected state. This was
                # identified by comparing the predicted probabilities (validation/matlab/ref_B2.m).
                # The "imm" and "kim" filters use the state of the origin regime.
                att = au[0]
                Q = np.stack([np.clip(m.transition_matrix(att, att, prev, solution.params[i], ss[i], regime=i), 0, 1)
                              for i in range(h)])
                Q = Q / Q.sum(1, keepdims=True)
            if t + 1 < nT:
                Qs[t + 1] = Q
            joint = Q * post[:, None]
            mu = joint.sum(0)
            for j in range(h):
                wj = joint[:, j] / mu[j] if mu[j] > 0 else post
                a[j] = ss[j] + cst[j] + T[j] @ (wj @ au - ss[j]) + Dx[t + 1, j]
                P[j] = T[j] @ np.einsum("k,kij->ij", wj, Pu) @ T[j].T + RRt[j]
            if ric is not None and ric.steady:
                P = ric.P.copy()
            continue
        if endo:  # transition from t-1 to t evaluated at the filtered state of each origin regime
            Q = np.stack([np.clip(m.transition_matrix(a[i], a[i], a_prev, solution.params[i], ss[i], regime=i), 0, 1)
                          for i in range(h)])
            Q = Q / Q.sum(1, keepdims=True)
        Qs[t] = Q
        joint = Q * mu[:, None]            # Pr(s_{t-1}=i, s_t=j | Y_{t-1})
        cj = joint.sum(0)                  # Pr(s_t=j | Y_{t-1})
        pred[t] = cj
        a_new, P_new, lam = np.zeros_like(a), np.zeros_like(P), np.full(h, -np.inf)
        if filter == "imm":
            for j in range(h):
                if cj[j] <= 0:
                    a_new[j], P_new[j] = a[j], P[j]
                    continue
                a0, P0 = _collapse(joint[:, j] / cj[j], a, P)
                if hist is not None:
                    hist.put(t, 0, j, ss[j] + cst[j] + Dx[t, j] + T[j] @ (a0 - ss[j]), T[j] @ P0 @ T[j].T + RRt[j])
                a_new[j], P_new[j], lam[j] = _kalman_step(a0, P0, ss[j], T[j], RRt[j], Y[t], Z, H, cst[j] + Dx[t, j])
            mx = np.max(lam)
            if not np.isfinite(mx):
                return _failed(filter, data, nT, h, n, m)
            w = cj * np.exp(lam - mx)
            ll_t[t] = mx + np.log(w.sum())
            mu = w / w.sum()
        else:  # kim
            aij, Pij, lij = np.zeros((h, h, n)), np.zeros((h, h, n, n)), np.full((h, h), -np.inf)
            for i in range(h):
                for j in range(h):
                    if joint[i, j] > 0 and hist is not None:
                        hist.put(t, i, j, ss[j] + cst[j] + Dx[t, j] + T[j] @ (a[i] - ss[j]), T[j] @ P[i] @ T[j].T + RRt[j])
                    if joint[i, j] > 0:
                        aij[i, j], Pij[i, j], lij[i, j] = _kalman_step(a[i], P[i], ss[j], T[j], RRt[j], Y[t], Z, H, cst[j] + Dx[t, j])
            mx = np.max(lij)
            if not np.isfinite(mx):
                return _failed(filter, data, nT, h, n, m)
            w = joint * np.exp(lij - mx)
            ll_t[t] = mx + np.log(w.sum())
            post = w / w.sum()
            if hist is not None:
                hist.post[t] = post
            mu = post.sum(0)
            for j in range(h):
                if mu[j] > 0:
                    a_new[j], P_new[j] = _collapse(post[:, j] / mu[j], aij[:, j], Pij[:, j])
                else:
                    a_new[j], P_new[j] = a[j], P[j]
        a_prev = states[t - 1] if t else a_prev
        a, P = a_new, P_new
        filt[t] = mu
        states[t] = mu @ a

    # smoothed probabilities: Kim (1994) backward recursion
    sm = np.zeros_like(filt)
    sm[-1] = filt[-1]
    for t in range(nT - 2, -1, -1):
        ratio = np.divide(sm[t + 1], pred[t + 1], out=np.zeros(h), where=pred[t + 1] > 0)
        # RISE (commit 1b2edfa) uses here the Q of the transition arriving at t, not the one leaving t;
        # it only matters with endogenous probabilities. "imm" and "kim" use the one leaving t.
        sm[t] = filt[t] * (Qs[t if filter == "rise" else t + 1] @ ratio)
        sm[t] /= sm[t].sum()

    cols = [f"regime_{j + 1}" for j in range(h)]
    df = lambda x, c: pd.DataFrame(x, index=data.index, columns=c)
    sm_states = sm_shocks = None
    if filter == "rise" and smooth:
        xs, es = _smooth_states(store_a, store_P, Y, Z, H, T, RS, sm)
        sm_states, sm_shocks = df(xs, m.endogenous), df(es, m.exogenous)
    elif hist is not None:   # "imm" and "kim": Hashimzade et al. (2024), Algorithm 4
        xs, es = smooth_histories(hist, Y, Z, H, T, RS, sm, pred, Qs)
        sm_states, sm_shocks = df(xs, m.endogenous), df(es, m.exogenous)
    return FilterResult(smoothed_states=sm_states, smoothed_shocks=sm_shocks, loglik=float(ll_t[presample:].sum()), loglik_t=ll_t, filtered_probs=df(filt, cols),
                        predicted_probs=df(pred, cols), smoothed_probs=df(sm, cols),
                        filtered_states=df(states, m.endogenous), filter=filter,
                        transition_matrices=Qs)


def _smooth_states(a, P, Y, Z, H, T, RS, sm):
    """Regime-by-regime state and shock smoother, in the form of Durbin and Koopman (2012).

    As in RISE (commit 1b2edfa; validation/explore_smoother.py): each regime is smoothed
    separately, r_j = Z'F_j^{-1}v_j + (I - K_j Z)' T_j' r_j(t+1), without mixing regimes in the
    backward recursion. Smoothed state of the regime: a_j + P_j r_j; shock: (R_j sigma)' r_j. The
    expected result weights by the smoothed probabilities. The first-period shock is zero, because
    the first observation is explained by the initial state. With one regime it is the usual smoother.
    """
    nT, h, n = a.shape
    xs, es = np.zeros((nT, n)), np.zeros((nT, RS.shape[2]))
    r_next = np.zeros((h, n))
    for t in range(nT - 1, -1, -1):
        ok = np.isfinite(Y[t])
        Zt = Z[ok]
        r_now = np.zeros((h, n))
        for j in range(h):
            mj = T[j].T @ r_next[j] if t + 1 < nT else np.zeros(n)
            if ok.any():
                PZ = P[t, j][:, Zt]
                F = PZ[Zt] + H[np.ix_(ok, ok)]
                Fi = np.linalg.inv((F + F.T) / 2)
                v = Y[t][ok] - a[t, j][Zt]
                K = PZ @ Fi
                ZF = np.zeros(n); ZF[Zt] = Fi @ v
                KZm = np.zeros(n); KZm[Zt] = K.T @ mj
                rj = ZF + mj - KZm
            else:
                rj = mj
            r_now[j] = rj
            xs[t] += sm[t, j] * (a[t, j] + P[t, j] @ rj)
            es[t] += sm[t, j] * (RS[j].T @ rj)
        r_next = r_now
    es[0] = 0.0
    return xs, es


def _failed(filter, data, nT, h, n, m):
    nan = lambda k, c: pd.DataFrame(np.full((nT, k), np.nan), index=data.index, columns=c)
    cols = [f"regime_{j + 1}" for j in range(h)]
    return FilterResult(-np.inf, np.full(nT, -np.inf), nan(h, cols), nan(h, cols), nan(h, cols),
                        nan(n, m.endogenous), filter)
