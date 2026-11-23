"""Compiled numerical kernels (numba, optional). Same recursion as filtering.run_filter, "rise" filter."""
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # numba missing: the package uses the pure-Python path
    HAVE_NUMBA = False

    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def rise_filter_kernel(Y, Z, H, T, RRt, ss, cst, Q, mu0, a0, P0):
    """"rise" filter with constant Q and no missing data.

    Returns (ll_t, filt, pred, states, store_a, store_P, ok); ok=False if some F is not positive definite.
    """
    nT, ny = Y.shape
    h, n = ss.shape
    ll_t = np.zeros(nT)
    filt = np.zeros((nT, h))
    pred = np.zeros((nT, h))
    states = np.zeros((nT, n))
    store_a = np.zeros((nT, h, n))
    store_P = np.zeros((nT, h, n, n))
    a = a0.copy()
    P = P0.copy()
    mu = mu0.copy()
    au = np.zeros((h, n))
    Pu = np.zeros((h, n, n))
    lam = np.zeros(h)
    log2pi = np.log(2 * np.pi)
    for t in range(nT):
        pred[t] = mu
        store_a[t] = a
        store_P[t] = P
        for j in range(h):
            PZ = np.ascontiguousarray(P[j][:, Z])
            F = PZ[Z, :] + H
            F = (F + F.T) / 2
            v = Y[t] - a[j][Z]
            sign, logdet = np.linalg.slogdet(F)
            if sign <= 0 or not np.isfinite(logdet):
                return ll_t, filt, pred, states, store_a, store_P, False
            Fi = np.linalg.inv(F)
            Fv = Fi @ v
            lam[j] = -0.5 * (ny * log2pi + logdet + v @ Fv)
            au[j] = a[j] + PZ @ Fv
            Pj = P[j] - PZ @ Fi @ PZ.T
            Pu[j] = (Pj + Pj.T) / 2
        mx = np.max(lam)
        w = mu * np.exp(lam - mx)
        sw = w.sum()
        ll_t[t] = mx + np.log(sw)
        post = w / sw
        filt[t] = post
        st = np.zeros(n)
        for j in range(h):
            st += post[j] * au[j]
        states[t] = st
        mu = np.zeros(h)
        for i in range(h):
            for j in range(h):
                mu[j] += Q[i, j] * post[i]
        for j in range(h):
            am = np.zeros(n)
            Pm = np.zeros((n, n))
            for i in range(h):
                wij = Q[i, j] * post[i] / mu[j] if mu[j] > 0 else post[i]
                am += wij * au[i]
                Pm += wij * Pu[i]
            Tj = np.ascontiguousarray(T[j])
            a[j] = ss[j] + cst[j] + Tj @ (am - ss[j])
            P[j] = Tj @ Pm @ Tj.T + RRt[j]
    return ll_t, filt, pred, states, store_a, store_P, True
