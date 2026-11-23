"""Complements to the linear regime-switching filter (used by filtering.run_filter).

- ``diffuse_lyapunov``: initial covariance with unit roots (RISE's ``diffuse_lyapunov_equation``);
- ``initial_conditions``: equivalents of ``kf_init_variance`` and ``kf_user_init``;
- ``RiccatiShortcut``: RISE's Riccati shortcut (``kf_riccati_tol``);
- ``HistoryStore`` and ``smooth_histories``: smoother of states and shocks for the ``imm`` and
  ``kim`` filters (Hashimzade, Kirsanov, Kirsanova and Maih, 2024, Algorithm 4).

The RISE rules were identified by numerical comparison (validation/explore_riccati.py,
validation/matlab/ref_riccati.m and ref_init.m).
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

_LOG2PI = np.log(2 * np.pi)


def diffuse_lyapunov(T: np.ndarray, RQR: np.ndarray, diffuse: float = 10.0, tol: float = 1e-9) -> np.ndarray:
    """Solution of ``V = T V T' + RQR`` when T has unit (or explosive) roots.

    Ordered Schur decomposition, ``T = U S U'``, with the roots of modulus >= 1 first. The stable
    block is autonomous in the Schur coordinates and gets its covariance from the Lyapunov
    equation; the nonstationary block receives variance ``diffuse`` and zero covariance with the
    rest: ``V = diffuse * U1 U1' + U2 X U2'``. Reproduces RISE's
    ``diffuse_lyapunov_equation(T, RQR, diffuse)``. Without a unit root it is the usual solution.
    """
    n = T.shape[0]
    S, U, k = linalg.schur(T, output="real", sort=lambda re, im=0.0: abs(complex(re, im)) >= 1 - tol)
    if k == 0:
        V = linalg.solve_discrete_lyapunov(T, RQR)
        return (V + V.T) / 2
    U1, U2 = U[:, :k], U[:, k:]
    V = diffuse * U1 @ U1.T
    if k < n:
        X = linalg.solve_discrete_lyapunov(S[k:, k:], U2.T @ RQR @ U2)
        V = V + U2 @ X @ U2.T
    return (V + V.T) / 2


def initial_conditions(a, P, mu, names, init_variance=None, user_init=None):
    """Applies ``init_variance`` (``kf_init_variance``) and ``user_init`` (``kf_user_init``).

    ``a`` (h, n), ``P`` (h, n, n) and ``mu`` (h,) are the default values of the filter.
    ``init_variance`` replaces the covariance by ``init_variance * I`` (the information in T and R
    is ignored). ``user_init = (a0,)``, ``(a0, P0)`` or ``(a0, P0, probabilities)``: ``a0`` is a
    dictionary name -> value (missing variables keep the default) or a vector in the order of
    ``model.endogenous``; ``P0`` is the covariance (matrix) or the variances (vector) in that order.
    Values common to all regimes, as in RISE.
    """
    a, P, mu = a.copy(), P.copy(), np.asarray(mu, float).copy()
    h, n = a.shape
    if init_variance is not None:
        if not init_variance > 0:
            raise ValueError("init_variance must be positive")
        P[:] = np.eye(n) * float(init_variance)
    if user_init is not None:
        user_init = tuple(user_init)
        if not 1 <= len(user_init) <= 3:
            raise ValueError("user_init must have 1 to 3 elements: (a0, P0, probabilities)")
        a0 = user_init[0]
        if isinstance(a0, dict):
            unknown = [k for k in a0 if k not in names]
            if unknown:
                raise ValueError(f"unknown variables in user_init: {unknown}")
            for k, v in a0.items():
                a[:, names.index(k)] = float(v)
        elif a0 is not None:
            a[:] = np.asarray(a0, float).reshape(n)
        if len(user_init) > 1 and user_init[1] is not None:
            P0 = np.asarray(user_init[1], float)
            P[:] = np.diag(P0) if P0.ndim == 1 else P0.reshape(n, n)
        if len(user_init) > 2 and user_init[2] is not None:
            mu = np.asarray(user_init[2], float).reshape(h)
            mu = mu / mu.sum()
    return a, P, mu


class RiccatiShortcut:
    """Riccati shortcut of the ``rise`` filter (RISE's ``kf_riccati_tol``).

    Rule identified by numerical comparison: each period the Kalman gain of all regimes is compared
    with that of the previous period, including the rows of the shocks (the RISE state is
    ``[y; shocks]``, so the gain has the block ``(R_j sigma)' Z' F_j^{-1}``; in the first period
    that block is zero). When the largest absolute difference falls below the tolerance and there
    are no more missing observations ahead, the gain, F, the updated covariance and the predicted
    covariance of each regime are frozen at the values of that period; only the means and the
    probabilities keep being updated. With regime switching the gain does not actually converge
    (it depends on the probabilities), so the shortcut changes the likelihood.
    """

    def __init__(self, tol: float, Y: np.ndarray, RS: np.ndarray):
        self.tol, self.RS = float(tol), RS
        miss = np.flatnonzero(~np.all(np.isfinite(Y), axis=1))
        self.first = int(miss[-1]) + 1 if miss.size else 0     # first period with no missing observations ahead
        self.steady, self.frozen_at, self.Kold = False, None, None

    def step(self, t, a, P, au, Pu, lam, y, Z, H):
        """Called after the update of period t; returns the (au, Pu, lam) to use."""
        h = a.shape[0]
        if self.steady:
            au, lam = au.copy(), lam.copy()
            for j in range(h):
                v = y - a[j][Z]
                Fv = self.Fi[j] @ v
                au[j] = a[j] + self.PZ[j] @ Fv
                lam[j] = -0.5 * (v.size * _LOG2PI + self.logdet[j] + v @ Fv)
            return au, self.Pu, lam
        if t < self.first:
            return au, Pu, lam
        PZ = np.stack([P[j][:, Z] for j in range(h)])
        Fi, logdet, K = [], [], []
        for j in range(h):
            F = PZ[j][Z] + H
            F = (F + F.T) / 2
            Fi.append(np.linalg.inv(F))
            logdet.append(np.linalg.slogdet(F)[1])
            Ks = self.RS[j][Z].T @ Fi[j] if t > 0 else np.zeros((self.RS.shape[2], Z.size))
            K.append(np.vstack([PZ[j] @ Fi[j], Ks]))
        K = np.stack(K)
        if self.Kold is not None and np.max(np.abs(K - self.Kold)) < self.tol:
            self.steady, self.frozen_at = True, t
            self.PZ, self.Fi, self.logdet = PZ, np.stack(Fi), np.array(logdet)
            self.Pu, self.P = Pu.copy(), P.copy()
        self.Kold = K
        return au, Pu, lam


class HistoryStore:
    """Predicted values per history for the smoother: ``imm`` stores per regime, ``kim`` per pair (i, j)."""

    def __init__(self, nT: int, h: int, n: int, pairs: bool):
        g = h if pairs else 1
        self.pairs = pairs
        self.a = np.zeros((nT, g, h, n))
        self.P = np.zeros((nT, g, h, n, n))
        self.used = np.zeros((nT, g, h), bool)
        self.post = np.zeros((nT, g, h))        # Pr(history | Y_t)

    def put(self, t, i, j, a, P):
        self.a[t, i, j], self.P[t, i, j], self.used[t, i, j] = a, P, True


def smooth_histories(store: HistoryStore, Y, Z, H, T, RS, sm, pred, Qs):
    """Smoothed states and shocks of the ``imm`` and ``kim`` filters (Hashimzade et al., 2024, Algorithm 4).

    For each history ``H_t`` (current regime in the IMM; pair ``(s_{t-1}, s_t)`` in the Kim filter):

        L = T_{s_{t+1}} (I - K Z),
        r(H_t) = Z' F^{-1} v + sum_{s_{t+1}} Q(s_t, s_{t+1}) L' r(H_{t+1}),
        alpha_{t|n}(H_t) = alpha_{t|t-1}(H_t) + P_{t|t-1}(H_t) r(H_t),

    and the result is weighted by the smoothed probabilities of the histories (eq. 14). The smoothed
    shock of the history is ``(R_j sigma)' r(H_t)``. It uses only the inverses of F from the filter
    itself. Returns (states (nT, n), shocks (nT, ne)).
    """
    nT, g, h, n = store.a.shape
    ne = RS.shape[2]
    xs, es = np.zeros((nT, n)), np.zeros((nT, ne))
    r_next = np.zeros((g, h, n))
    for t in range(nT - 1, -1, -1):
        ok = np.isfinite(Y[t])
        Zt = Z[ok]
        # smoothed probabilities of the histories
        if store.pairs:
            if t + 1 < nT:
                ratio = np.divide(sm[t + 1], pred[t + 1], out=np.zeros(h), where=pred[t + 1] > 0)
                w = store.post[t] * (Qs[t + 1] @ ratio)[None, :]
            else:
                w = store.post[t].copy()
            w = w / w.sum() if w.sum() > 0 else w
        else:
            w = sm[t][None, :]
        r_now = np.zeros((g, h, n))
        for i in range(g):
            for j in range(h):
                if not store.used[t, i, j]:
                    continue
                a, P = store.a[t, i, j], store.P[t, i, j]
                mj = np.zeros(n)
                if t + 1 < nT:
                    for k in range(h):
                        q = Qs[t + 1][j, k]
                        if q > 0:
                            mj += q * (T[k].T @ r_next[j if store.pairs else 0, k])
                if ok.any():
                    PZ = P[:, Zt]
                    F = PZ[Zt] + H[np.ix_(ok, ok)]
                    Fi = np.linalg.inv((F + F.T) / 2)
                    v = Y[t][ok] - a[Zt]
                    r = mj.copy()
                    r[Zt] += Fi @ v - (PZ @ Fi).T @ mj
                else:
                    r = mj
                r_now[i, j] = r
                xs[t] += w[i, j] * (a + P @ r)
                es[t] += w[i, j] * (RS[j].T @ r)
        r_next = r_now
    return xs, es
