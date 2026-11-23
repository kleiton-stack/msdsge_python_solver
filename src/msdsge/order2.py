"""2nd-order perturbation with regime switching (Maih, 2015, §3.3, eq. 20).

Notation. State ``z = [x, sigma, e]``: ``x`` are the deviations of the k state variables at t-1,
``sigma`` the perturbation parameter and ``e`` the shocks (unit variance; the scale enters through R).
Policy function of regime r:

    y_t - ss_r = G_r z + 1/2 G2_r (z kron z),      G_r = [X_r, c_r, R_r diag(sigma_e)].

With ``z' = (H_r + u) z`` (``H_r = [L G_r; e_sigma'; 0]``, ``u`` random with ``E[u kron u] = sum_i
E_i kron E_i``) and ``v = [y_{t+1}; y_t; y_{t-1}; e_t]``, the 2nd-order condition is linear in G2:

    U_r G2_r + sum_r' Q[r,r'] Ap_r G2_r' C_rr' + A_r = 0,
    C_rr' = H_rr' kron H_rr' + sum_i E_i kron E_i,
    A_r = sum_r' Q[r,r'] f_vv^r (V0 kron V0 + sum_i V1_i kron V1_i).

Regime-specific steady state. Since the policy function of regime r' is written in deviations from
the steady state of r', the sigma column of ``H_rr'`` carries the difference between the steady states
of the state variables, ``x_ss(r) - x_ss(r')``. This is how the sigma terms of RISE were
reproduced (validation/explore_order2.py); at 1st order RISE does not include this term.

Limitations of this version: exogenous transition probabilities; direct (Kronecker) solution or
matrix-free GMRES for large systems.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.sparse.linalg import LinearOperator, gmres

from .parser import timed
from .solve import SolveError, _U, _states


def _hessian_entries(model):
    """Non-zero second derivatives of the equations: list (eq, i, j) and a function that evaluates them."""
    if getattr(model, "_hess", None) is not None:
        return model._hess
    y0 = [timed(v) for v in model.endogenous]
    ym = [timed(v, -1) for v in model.endogenous]
    yp = [timed(v, 1) for v in model.endogenous]
    e = [timed(v) for v in model.exogenous]
    p = [timed(v) for v in model.parameters]
    ss_syms = [sp.Symbol(f"{v}__SS", real=True) for v in model.endogenous]
    from .parser import _SS
    ss_map = {_SS(timed(v)): s for v, s in zip(model.endogenous, ss_syms)}
    v = yp + y0 + ym + e
    pos = {s: i for i, s in enumerate(v)}
    idx, exprs = [], []
    for q, eq in enumerate(model._equations):
        eq = eq.xreplace(ss_map)
        syms = sorted((s for s in eq.free_symbols if s in pos), key=lambda s: pos[s])
        for a, sa in enumerate(syms):
            da = sp.diff(eq, sa)
            if da.is_number and da == 0:
                continue
            for sb in syms[a:]:
                d2 = sp.diff(da, sb)
                if d2 != 0:
                    idx.append((q, pos[sa], pos[sb]))
                    exprs.append(d2)
    fn = sp.lambdify((yp, y0, ym, e, ss_syms, p), exprs, modules="numpy", cse=True) if exprs else None
    model._hess = (np.array(idx, int).reshape(-1, 3), fn)
    return model._hess


def solve_second_order(solution, dense_limit: int = 6000, tol: float = 1e-12) -> np.ndarray:
    """Returns G2 (h, n, nz, nz) for the given 1st-order solution."""
    model = solution.model
    if solution.endogenous_switching:
        raise NotImplementedError("2nd order with endogenous transition probabilities is not supported yet")
    params, ss = solution.params, solution.ss
    sys = model.first_order_system(params, ss)
    h, n = model.h, model.n
    ne = len(model.exogenous)
    st = _states(sys)
    k = st.size
    nz = k + 1 + ne
    sig = solution.sigma
    # 1st-order policy function in z
    G = np.zeros((h, n, nz))
    G[:, :, :k] = solution.T[:, :, st]
    G[:, :, k] = solution.c
    G[:, :, k + 1:] = solution.R * sig
    Hm = np.zeros((h, h, nz, nz))            # H[r, q]: transition from regime r to regime q
    for r in range(h):
        for q in range(h):
            Hm[r, q, :k, :] = G[r][st, :]
            Hm[r, q, k, k] = 1.0
            Hm[r, q, :k, k] += (ss[r] - ss[q])[st]
    E = np.zeros((ne, nz, nz))
    for i in range(ne):
        E[i, k + 1 + i, k] = 1.0
    EE = sum(np.kron(E[i], E[i]) for i in range(ne)) if ne else np.zeros((nz * nz, nz * nz))
    U = _U(sys, solution.T)

    idx, fn = _hessian_entries(model)
    nv = 3 * n + ne
    e0 = np.zeros(ne)
    A = np.zeros((h, n, nz * nz))
    C = np.zeros((h, h, nz * nz, nz * nz))
    for r in range(h):
        for q in range(h):
            C[r, q] = np.kron(Hm[r, q], Hm[r, q]) + EE
        if fn is None or len(idx) == 0:
            continue
        vals = np.asarray(fn(ss[r], ss[r], ss[r], e0, ss[r], params[r]), float)
        for q in range(h):
            if sys.Q[r, q] == 0:
                continue
            V0 = np.zeros((nv, nz))
            V0[:n] = G[q] @ Hm[r, q]
            V0[n:2 * n] = G[r]
            V0[2 * n + st, np.arange(k)] = 1.0
            V0[3 * n + np.arange(ne), k + 1 + np.arange(ne)] = sig     # e_t = sigma_e * unit shock
            V1 = np.zeros((ne, nv, nz))
            for i in range(ne):
                V1[i, :n] = G[q] @ E[i]
            for (eq, a, b), val in zip(idx, vals):
                if val == 0:
                    continue
                term = np.outer(V0[a], V0[b])
                for i in range(ne):
                    term += np.outer(V1[i, a], V1[i, b])
                if a != b:
                    term = term + term.T
                A[r, eq] += sys.Q[r, q] * val * term.ravel()

    # linear system: G2_r + sum_q Lp[r,q] G2_q C_rq = -U_r^{-1} A_r
    Lp = np.stack([[sys.Q[r, q] * np.linalg.solve(U[r], sys.Ap[r]) for q in range(h)] for r in range(h)])
    rhs = -np.linalg.solve(U, A)
    m2 = nz * nz
    size = h * n * m2
    if not np.any(rhs):
        return np.zeros((h, n, nz, nz))

    def apply(xvec):
        X = xvec.reshape(h, n, m2)
        out = X.copy()
        for r in range(h):
            for q in range(h):
                if sys.Q[r, q] != 0:
                    out[r] += Lp[r, q] @ X[q] @ C[r, q]
        return out.ravel()

    if size <= dense_limit:
        big = np.zeros((size, size))
        eye = np.eye(size)
        for j in range(size):
            big[:, j] = apply(eye[j])
        sol = np.linalg.solve(big, rhs.ravel())
    else:
        sol, info = gmres(LinearOperator((size, size), matvec=apply), rhs.ravel(), rtol=tol, atol=0.0,
                          restart=200, maxiter=2000)
        if info != 0:
            raise SolveError(f"GMRES did not converge at 2nd order (info={info})")
    G2 = sol.reshape(h, n, nz, nz)
    return (G2 + G2.transpose(0, 1, 3, 2)) / 2
