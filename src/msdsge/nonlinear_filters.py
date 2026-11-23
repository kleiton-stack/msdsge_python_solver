"""Sigma-point filters with regime switching for higher-order solutions (Binning and Maih, 2015).

Three integration rules on the same skeleton (Algorithm 2 of the paper), as in RISE:

- ``ukf``: unscented transform with ``n + lambda = 3`` (``switching_unscented_kalman_filter``);
- ``ckf``: spherical-radial cubature rule (``switching_cubature_kalman_filter``);
- ``ddf``: Stirling interpolation with step ``delta = sqrt(3)`` (``switching_divided_difference_filter``).

Skeleton, per period: (1) in each regime, forecast of the observable, marginal likelihood and
Kalman update (the measurement is linear: a selection of variables); (2) update of the
probabilities; (3) collapse of the updated states for each destination regime, with weights
``Q[i, j] p_t|t(i) / p_t+1|t(j)`` and without the term for the dispersion of the means; (4) sigma
points over ``X = [x_t|t, eta_{t+1}]`` and prediction with the nonlinear law of motion of the
destination regime.

RISE details (commit 1b2edfa) identified by numerical comparison (validation/explore_nlf.py),
with the handles that ``dsge.filter`` hands to ``kf_user_algo`` and ``utils.filtering.sigma_points``:

- the state vector is ``[y_t in RISE order; shocks at t]`` (the shocks enter the state so that
  they are recovered in the update); the variable order is static, predetermined, "both",
  forward-looking, alphabetical within each group. The order matters because the square root of
  the covariance is the lower Cholesky factor;
- eigenvalues of the covariance below ``sqrt(eps)`` are raised to that floor, both in the
  predicted covariance and before the factorisation that generates the sigma points;
- the initial state and covariance are those of the linear ``rise`` filter (common to the regimes,
  1st-order Lyapunov with average T and R), with an identity block for the shocks; the first
  observation uses the initial state as the predicted one;
- law of motion without pruning: ``y_t = ss_r + G_r z + 1/2 G2_r (z kron z)``,
  ``z = [y_{t-1}[states] - ss_r[states], 1, eta_t]``.

RISE's divided difference filter does not run in MATLAB R2024a (syntax error in the p-code),
so ``ddf`` follows the paper and was not compared number by number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg

from .filtering import FilterResult, _failed, _initial_cov, ergodic_distribution

NONLINEAR_FILTERS = ("ukf", "ckf", "ddf")
_LOG2PI = np.log(2 * np.pi)
_FLOOR = np.sqrt(np.finfo(float).eps)


@dataclass
class NonlinearFilterResult(FilterResult):
    updated_shocks: pd.DataFrame | None = None       # E[eta_t | Y_t], in standard-deviation units
    predicted_states: pd.DataFrame | None = None     # E[x_t | Y_{t-1}], combined across regimes


def rise_variable_order(solution) -> np.ndarray:
    """Indices of the endogenous variables in RISE order: static, predetermined, "both", forward-looking."""
    m = solution.model
    sys = m.first_order_system(solution.params, solution.ss)
    lag = np.any(sys.Am != 0, axis=(0, 1))
    lead = np.any(sys.Ap != 0, axis=(0, 1))
    group = np.where(lag & lead, 2, np.where(lag, 1, np.where(lead, 3, 0)))
    return np.array(sorted(range(m.n), key=lambda i: (group[i], m.endogenous[i])), int)


def sigma_weights(n: int, rule: str) -> np.ndarray:
    """Mean weights: ``ukf`` and ``ddf`` have 2n+1 points (the first is the centre), ``ckf`` has 2n."""
    if rule == "ckf":
        return np.full(2 * n, 1 / (2 * n))
    if rule in ("ukf", "ddf"):
        return np.r_[1 - n / 3, np.full(2 * n, 1 / 6)]
    raise ValueError(f"unknown rule: {rule!r}; options: {NONLINEAR_FILTERS}")


def _floor_cov(P: np.ndarray) -> np.ndarray:
    """Covariance with eigenvalues below sqrt(eps) raised to that floor (as in RISE)."""
    P = (P + P.T) / 2
    w, V = np.linalg.eigh(P)
    if w.min() < _FLOOR:
        P = (V * np.maximum(w, _FLOOR)) @ V.T
        P = (P + P.T) / 2
    return P


def _sqrt_cov(P: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of the covariance with a floor on the eigenvalues."""
    return np.linalg.cholesky(_floor_cov(P))


def sigma_points(mean: np.ndarray, P: np.ndarray, rule: str) -> np.ndarray:
    """Sigma points in rows: centre (except ``ckf``), then ``+`` and ``-`` each column of the square root."""
    n = mean.size
    S = _sqrt_cov(P) * np.sqrt(n if rule == "ckf" else 3.0)
    pts = np.vstack([mean + S.T, mean - S.T])
    return pts if rule == "ckf" else np.vstack([mean, pts])


def _transition(solution, order: np.ndarray):
    """Law of motion of the augmented state ``[y (given order); eta]`` in each regime."""
    h, n = solution.T.shape[0], solution.T.shape[1]
    ne = solution.R.shape[2]
    st = np.flatnonzero(np.abs(solution.T).sum((0, 1)) > 0)
    k = st.size
    G = np.zeros((h, n, k + 1 + ne))
    G[:, :, :k] = solution.T[:, :, st]
    G[:, :, k] = solution.c
    G[:, :, k + 1:] = solution.R * solution.sigma
    G2 = solution.G2 if solution.G2 is not None else np.zeros((h, n, k + 1 + ne, k + 1 + ne))
    if G2.shape[2] != k + 1 + ne:
        raise ValueError("G2 is not compatible with the state variables of the solution")
    pos = np.empty(n, int)
    pos[order] = np.arange(n)            # position of each endogenous variable in the reordered vector
    G, G2, ss = G[:, order], G2[:, order], solution.ss[:, order]
    stp = pos[st]

    def f(r: int, X: np.ndarray) -> np.ndarray:
        """X: (N, n + ne) with [y_{t-1}, eta_t]; returns (N, n + ne) with [y_t, eta_t]."""
        z = np.concatenate([X[:, stp] - ss[r, stp], np.ones((X.shape[0], 1)), X[:, n:]], axis=1)
        y = ss[r] + z @ G[r].T + 0.5 * np.einsum("nij,pi,pj->pn", G2[r], z, z)
        return np.concatenate([y, X[:, n:]], axis=1)

    return f


def _predict(f, r, mean, P, ne, rule):
    """Predicted mean and covariance of ``[y_t, eta_t]`` from the collapsed state of t-1.

    ``mean`` and ``P`` have dimension ``n + ne`` (endogenous variables and shocks of t-1); the sigma
    points are generated over that vector augmented with the shocks of t, with identity covariance.
    """
    M = mean.size
    n = M - ne
    N = M + ne
    S = np.zeros((N, N))
    S[:M, :M] = P
    S[M:, M:] = np.eye(ne)
    X = sigma_points(np.r_[mean, np.zeros(ne)], S, rule)
    FX = f(r, np.concatenate([X[:, :n], X[:, M:]], axis=1))
    w = sigma_weights(N, rule)
    a = w @ FX
    if rule == "ddf":
        d = np.sqrt(3.0)
        S1 = (FX[1:N + 1] - FX[N + 1:]) / (2 * d)
        S2 = (FX[1:N + 1] + FX[N + 1:] - 2 * FX[0]) * np.sqrt(d * d - 1) / (2 * d * d)
        Pn = S1.T @ S1 + S2.T @ S2
    else:
        D = FX - a
        Pn = (D * w[:, None]).T @ D
    return a, _floor_cov(Pn)


def sigma_filter_core(f, Y, Z, H, Q, mu, a, P, ne, rule):
    """Recursion of Algorithm 2 of Binning and Maih (2015) over the augmented state.

    ``f(r, X)``: law of motion (see ``_transition``); ``a`` (h, M) and ``P`` (h, M, M): predicted
    values for the first observation; ``Z``: positions of the observables in the state. Returns
    ``(ll_t, filtered, predicted, smoothed, updated state, predicted state)`` or ``None`` on failure.
    """
    a, P, mu = a.copy(), P.copy(), np.asarray(mu, float).copy()
    h, M = a.shape
    nT = Y.shape[0]
    ll_t = np.zeros(nT)
    filt, pred = np.zeros((nT, h)), np.zeros((nT, h))
    upd, prd = np.zeros((nT, M)), np.zeros((nT, M))
    for t in range(nT):
        pred[t] = mu
        prd[t] = mu @ a
        ok = np.isfinite(Y[t])
        Zt = Z[ok]
        au, Pu, lam = a.copy(), P.copy(), np.zeros(h)
        if ok.any():
            for j in range(h):
                v = Y[t][ok] - a[j][Zt]
                PZ = P[j][:, Zt]
                F = PZ[Zt] + H[np.ix_(ok, ok)]
                F = (F + F.T) / 2
                try:
                    cf = linalg.cho_factor(F)
                except linalg.LinAlgError:
                    return None
                Fv = linalg.cho_solve(cf, v)
                lam[j] = -0.5 * (v.size * _LOG2PI + 2 * np.sum(np.log(np.diag(cf[0]))) + v @ Fv)
                au[j] = a[j] + PZ @ Fv
                Pu[j] = P[j] - PZ @ linalg.cho_solve(cf, PZ.T)
                Pu[j] = (Pu[j] + Pu[j].T) / 2
        mx = np.max(lam)
        if not np.isfinite(mx):
            return None
        w = mu * np.exp(lam - mx)
        if not w.sum() > 0:
            return None
        ll_t[t] = mx + np.log(w.sum()) if ok.any() else 0.0
        post = w / w.sum()
        filt[t], upd[t] = post, post @ au
        joint = Q * post[:, None]
        mu = joint.sum(0)
        try:
            for j in range(h):
                wj = joint[:, j] / mu[j] if mu[j] > 0 else post
                a[j], P[j] = _predict(f, j, wj @ au, np.einsum("k,kij->ij", wj, Pu), ne, rule)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(a)):
            return None

    sm = np.zeros_like(filt)
    sm[-1] = filt[-1]
    for t in range(nT - 2, -1, -1):
        ratio = np.divide(sm[t + 1], pred[t + 1], out=np.zeros(h), where=pred[t + 1] > 0)
        sm[t] = filt[t] * (Q @ ratio)
        sm[t] /= sm[t].sum()

    return ll_t, filt, pred, sm, upd, prd


def run_nonlinear_filter(solution, data: pd.DataFrame, filter: str = "ukf", measurement_errors=None,
                         init_probs=None, diffuse: float = 10.0, user_init=None, init_variance=None,
                         presample: int = 0) -> NonlinearFilterResult:
    """Sigma-point filter with regime switching; see the module docstring.

    ``presample``, ``init_variance`` and ``user_init`` as in ``filtering.run_filter``.
    """
    if filter not in NONLINEAR_FILTERS:
        raise ValueError(f"unknown filter: {filter!r}; options: {NONLINEAR_FILTERS}")
    if solution.endogenous_switching:
        raise NotImplementedError("sigma-point filters with endogenous probabilities are not supported yet")
    m = solution.model
    obs = list(m.spec.observables)
    if not obs:
        raise ValueError("the model declares no observables (varobs)")
    missing = [v for v in obs if v not in data.columns]
    if missing:
        raise ValueError(f"observables missing from the data: {missing}")
    order = rise_variable_order(solution)
    pos = np.empty(m.n, int)
    pos[order] = np.arange(m.n)
    Y = data[obs].to_numpy(float)
    Z = pos[[m.endogenous.index(v) for v in obs]]
    H = np.zeros((len(obs), len(obs)))
    for k, v in (measurement_errors or {}).items():
        H[obs.index(k), obs.index(k)] = float(v) ** 2
    h, n, ne = solution.T.shape[0], solution.T.shape[1], solution.R.shape[2]
    M = n + ne
    Q = solution.Q
    f = _transition(solution, order)
    RS = solution.R * solution.sigma

    mu = ergodic_distribution(Q) if init_probs is None else np.asarray(init_probs, float)
    a0 = mu @ solution.ss
    P0 = _initial_cov(np.einsum("r,rij->ij", mu, solution.T), np.einsum("r,rij->ij", mu, RS), diffuse)
    if init_variance is not None or user_init is not None:
        from .filter_extras import initial_conditions
        a1, P1, mu = initial_conditions(a0[None], P0[None], np.ones(1), list(m.endogenous),
                                        init_variance, user_init[:2] if user_init is not None else None)
        a0, P0 = a1[0], P1[0]
        mu = ergodic_distribution(Q) if init_probs is None else np.asarray(init_probs, float)
        if user_init is not None and len(user_init) > 2 and user_init[2] is not None:
            mu = np.asarray(user_init[2], float) / np.sum(user_init[2])
    a = np.zeros((h, M))
    a[:, :n] = a0[order]
    P = np.zeros((h, M, M))
    P[:, :n, :n] = P0[np.ix_(order, order)]
    P[:, n:, n:] = np.eye(ne)

    out = sigma_filter_core(f, Y, Z, H, Q, mu, a, P, ne, filter)
    nT = Y.shape[0]
    if out is None:
        return _failed_nl(filter, data, nT, h, n, m)
    ll_t, filt, pred, sm, upd, prd = out

    cols = [f"regime_{j + 1}" for j in range(h)]
    df = lambda x, c: pd.DataFrame(x, index=data.index, columns=c)
    return NonlinearFilterResult(
        loglik=float(ll_t[presample:].sum()), loglik_t=ll_t, filtered_probs=df(filt, cols), predicted_probs=df(pred, cols),
        smoothed_probs=df(sm, cols), filtered_states=df(upd[:, pos], m.endogenous), filter=filter,
        transition_matrices=np.broadcast_to(Q, (nT, h, h)).copy(),
        updated_shocks=df(upd[:, n:], m.exogenous), predicted_states=df(prd[:, pos], m.endogenous))


def _failed_nl(filter, data, nT, h, n, m):
    r = _failed(filter, data, nT, h, n, m)
    return NonlinearFilterResult(**{k: getattr(r, k) for k in r.__dataclass_fields__})
