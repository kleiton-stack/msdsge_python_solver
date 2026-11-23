"""Order-p perturbation with regime switching (Maih, 2015, §3.3 to §3.5; Levintal, 2017).

Notation. State ``z = [x, sigma, e]``: ``x`` holds the deviations of the k state variables at t-1
from the steady state of the current regime, ``sigma`` is the perturbation parameter and ``e`` the
unit-variance shocks. Policy of regime r:

    y_t - ss_r = sum_{j=1..p} (1/j!) G^j_r z^{(x) j}.

Rather than writing the chain rule order by order, the module works with truncated polynomials in
``w = (z, u)``, with ``u = sigma * eps_{t+1}`` (a single block, as in Levintal, 2017). The state at
t+1 seen from regime r, with destination q, is

    z' = [L G_r(z) + (x_ss(r) - x_ss(q)) sigma ; sigma ; u],

and ``v = [G_q(z') ; G_r(z) ; x ; sigma_e e]``. The Taylor series of f around the steady state of
regime r is composed with these polynomials, the expectation over ``eps`` is taken (normal moments)
and the coefficient of each degree-j monomial is set to zero. That coefficient is affine in G^j:

    U_r X_r + sum_q Q[r,q] Ap_r X_q C_rq + A_r = 0,

the same structure as at 2nd order (order2.py): ``A_r`` is the coefficient computed with ``G^j = 0``
and ``C_rq`` maps the degree-j monomials of z' into those of z. Coefficients are stored per monomial
(one per combination with repetition), which exploits the symmetry of the tensors.

Endogenous transition probabilities: ``Q_rq`` becomes a polynomial in z (composition with v) and
multiplies the expansion of f before the expectation. The new terms of order j use only derivatives
of Q up to order j-1 and lower-order policies, so the linear system is the same.

RISE convention for regime-specific steady states: the difference between steady states enters the
sigma column of z' (see docs/VALIDATION.md), and the 1st-order sigma column is zero.

Simulation: ``simulate`` (with and without pruning), ``irf`` and ``girf``. Pruning follows Andreasen,
Fernández-Villaverde and Rubio-Ramírez (2018) generalised to order p: the state is decomposed into
components of order 1 to p and only products whose total order does not exceed p enter.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sps
import sympy as sp
from scipy.sparse.linalg import LinearOperator, gmres, tfqmr

from .parser import _SS, timed
from .solve import SolveError, _U, _states

MAX_ORDER = 5


# --------------------------------------------------------------------------------------------
# monomial basis and truncated polynomial algebra
# --------------------------------------------------------------------------------------------
_WEIGHTS = np.random.default_rng(20261122).integers(1, 2 ** 40, size=4096).astype(np.int64)


class _Basis:
    """Monomials of degree 0 to p in m variables; each one is a sorted tuple of indices."""

    def __init__(self, m: int, p: int):
        self.m, self.p = m, p
        self.tup, self.key, self._order, self._skey = [], [], [], []
        w = _WEIGHTS[:m]
        for d in range(p + 1):
            t = (np.array(list(itertools.combinations_with_replacement(range(m), d)), dtype=np.int64).reshape(-1, d)
                 if d else np.zeros((1, 0), np.int64))
            key = w[t].sum(1) if d else np.zeros(1, np.int64)
            order = np.argsort(key, kind="stable")
            if np.any(np.diff(key[order]) == 0):
                raise RuntimeError("key collision in the monomial basis")
            self.tup.append(t), self.key.append(key), self._order.append(order), self._skey.append(key[order])
        self.M = [len(t) for t in self.tup]
        self.off = np.concatenate([[0], np.cumsum(self.M)])
        self.N = int(self.off[-1])
        self._tables: dict = {}

    def block(self, d: int) -> slice:
        return slice(int(self.off[d]), int(self.off[d + 1]))

    def lookup(self, d: int, keys) -> np.ndarray:
        """Position, within degree d, of the monomials with the given keys."""
        keys = np.asarray(keys, np.int64)
        pos = np.searchsorted(self._skey[d], keys)
        pos = np.clip(pos, 0, self.M[d] - 1)
        if np.any(self._skey[d][pos] != keys):
            raise RuntimeError("monomial outside the basis")
        return self._order[d][pos]

    def exponents(self, d: int) -> np.ndarray:
        e = np.zeros((self.M[d], self.m), np.int64)
        for c in range(d):
            np.add.at(e, (np.arange(self.M[d]), self.tup[d][:, c]), 1)
        return e

    def table(self, a: int, b: int):
        """Sparse matrix (M_{a+b} x M_a M_b) that sums the products of monomials of degrees a and b."""
        if (a, b) not in self._tables:
            keys = (self.key[a][:, None] + self.key[b][None, :]).ravel()
            tgt = self.lookup(a + b, keys)
            self._tables[a, b] = sps.csr_matrix((np.ones(tgt.size), (tgt, np.arange(tgt.size))),
                                                shape=(self.M[a + b], tgt.size))
        return self._tables[a, b]

    def mul(self, A: np.ndarray, b: np.ndarray, amin: int = 0, bmin: int = 0) -> np.ndarray:
        """Truncated product of each row of A (K x N) with the polynomial b (N,).

        ``amin`` and ``bmin``: lowest degree with a nonzero coefficient in A and in b.
        """
        out = np.zeros_like(A)
        K = A.shape[0]
        for db in range(bmin, self.p + 1):
            Bb = b[self.block(db)]
            nzb = np.flatnonzero(Bb)
            if nzb.size == 0:
                continue
            for da in range(amin, self.p - db + 1):
                Aa = A[:, self.block(da)]
                if not Aa.any():
                    continue
                S = self.table(da, db)
                if nzb.size * 4 < Bb.size:
                    # sparse b (typical case: a single monomial): avoids the full outer product
                    cols = (np.arange(self.M[da])[:, None] * self.M[db] + nzb[None, :]).ravel()
                    prod = (Aa[:, :, None] * Bb[nzb][None, None, :]).reshape(K, -1)
                    out[:, self.block(da + db)] += (S[:, cols] @ prod.T).T
                else:
                    prod = (Aa[:, :, None] * Bb[None, None, :]).reshape(K, -1)
                    out[:, self.block(da + db)] += (S @ prod.T).T
        return out


def _normal_moment(k: int) -> float:
    return 0.0 if k % 2 else float(np.prod(np.arange(k - 1, 0, -2))) if k else 1.0


@dataclass
class _Spaces:
    """Bases in w = (z, u) and in z for a truncation order, with embedding and expectation."""
    B: _Basis
    Bz: _Basis
    zin: np.ndarray           # position in B of each monomial of Bz
    Ex: sps.csr_matrix        # (N_w x N_z): expectation over eps of each monomial of w
    parents: list             # per degree d >= 1 of Bz: (global index of the parent monomial, last variable)


_SPACE_CACHE: dict = {}


def _spaces(nz: int, ne: int, k: int, p: int) -> _Spaces:
    key = (nz, ne, k, p)
    if key in _SPACE_CACHE:
        return _SPACE_CACHE[key]
    B, Bz = _Basis(nz + ne, p), _Basis(nz, p)
    zin = np.concatenate([B.off[d] + B.lookup(d, Bz.key[d]) for d in range(p + 1)])
    rows, cols, vals = [], [], []
    w = _WEIGHTS
    for d in range(p + 1):
        ex = B.exponents(d)
        nu = ex[:, nz:].sum(1)
        wt = np.ones(B.M[d])
        for i in range(ne):
            wt *= np.array([_normal_moment(int(c)) for c in ex[:, nz + i]])
        keep = np.flatnonzero(wt)
        tk = ex[keep, :nz] @ w[:nz] + nu[keep] * w[k]              # u = sigma * eps
        rows.append(B.off[d] + keep), cols.append(Bz.off[d] + Bz.lookup(d, tk)), vals.append(wt[keep])
    Ex = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(B.N, Bz.N))
    parents = [None]
    for d in range(1, p + 1):
        last = Bz.tup[d][:, -1]
        par = Bz.off[d - 1] + Bz.lookup(d - 1, Bz.key[d] - w[last])
        parents.append((par, last))
    sp_ = _Spaces(B, Bz, zin, Ex, parents)
    if len(_SPACE_CACHE) > 16:
        _SPACE_CACHE.clear()
    _SPACE_CACHE[key] = sp_
    return sp_


# --------------------------------------------------------------------------------------------
# higher-order symbolic derivatives (nonzero entries only, one representative per symmetry class)
# --------------------------------------------------------------------------------------------
@dataclass
class _Taylor:
    """Taylor series of a list of expressions in v = [y(+1), y, y(-1), e]."""
    n_out: int
    order: int
    levels: list     # per length L: dict(tuples (K, L), parent (K,), rows, cols, scale, sl (slice into vals))
    fn: object       # evaluates all stored derivatives
    f0: object       # evaluates the expressions

    def compose(self, vals: np.ndarray, V: np.ndarray, basis: _Basis, upto: int) -> np.ndarray:
        """Sum_beta (D^beta f / beta!) v^beta, truncated, with v given by the rows of V (nv x N)."""
        out = np.zeros((self.n_out, basis.N))
        prev = None
        for L, lev in enumerate(self.levels[:upto], start=1):
            tup = lev["tuples"]
            if tup.shape[0] == 0:
                break
            if L == 1:
                P = V[tup[:, 0]]
            else:
                P = np.zeros((tup.shape[0], basis.N))
                last = tup[:, -1]
                for i in np.unique(last):
                    sel = np.flatnonzero(last == i)
                    if V[i].any():
                        P[sel] = basis.mul(prev[lev["parent"][sel]], V[i], amin=L - 1, bmin=1)
            coef = sps.csr_matrix((vals[lev["sl"]] * lev["scale"], (lev["rows"], lev["cols"])),
                                  shape=(self.n_out, tup.shape[0]))
            out += coef @ P
            prev = P
        return out


def _args(model):
    y0 = [timed(v) for v in model.endogenous]
    ym = [timed(v, -1) for v in model.endogenous]
    yp = [timed(v, 1) for v in model.endogenous]
    e = [timed(v) for v in model.exogenous]
    p = [timed(v) for v in model.parameters]
    ss_syms = [sp.Symbol(f"{v}__SS", real=True) for v in model.endogenous]
    ss_map = {_SS(timed(v)): s for v, s in zip(model.endogenous, ss_syms)}
    return yp, y0, ym, e, ss_syms, p, ss_map


def _taylor(model, exprs, order: int, tag: str) -> _Taylor:
    cache = model.__dict__.setdefault("_pert_cache", {})
    if (tag, order) in cache:
        return cache[tag, order]
    yp, y0, ym, e, ss_syms, p, ss_map = _args(model)
    v = yp + y0 + ym + e
    pos = {s: i for i, s in enumerate(v)}
    exprs = [sp.sympify(x).xreplace(ss_map) for x in exprs]
    per_level = [dict() for _ in range(order)]      # tuple -> list of (output expression, derivative)
    for q, eq in enumerate(exprs):
        syms = sorted((s for s in eq.free_symbols if s in pos), key=lambda s: pos[s])
        level = {(): eq}
        for L in range(order):
            new = {}
            for t, ex in level.items():
                for s in syms:
                    if t and pos[s] < t[-1]:
                        continue
                    d = sp.diff(ex, s)
                    if d != 0:
                        new[t + (pos[s],)] = d
            for t, d in new.items():
                per_level[L].setdefault(t, []).append((q, d))
            level = new
    levels, flat, index_prev = [], [], {}
    for L in range(order):
        tuples = sorted(per_level[L])
        index = {t: i for i, t in enumerate(tuples)}
        rows, cols, scale, start = [], [], [], len(flat)
        for t in tuples:
            mult = math.prod(math.factorial(c) for c in np.unique(t, return_counts=True)[1])
            for q, d in per_level[L][t]:
                rows.append(q), cols.append(index[t]), scale.append(1.0 / mult), flat.append(d)
        levels.append(dict(
            tuples=np.array(tuples, np.int64).reshape(-1, L + 1),
            parent=np.array([index_prev[t[:-1]] for t in tuples], np.int64) if L else np.zeros(len(tuples), np.int64),
            rows=np.array(rows, np.int64), cols=np.array(cols, np.int64), scale=np.array(scale),
            sl=slice(start, len(flat))))
        index_prev = index
    a = (yp, y0, ym, e, ss_syms, p)
    fn = sp.lambdify(a, flat, modules="numpy", cse=True) if flat else (lambda *x: [])
    f0 = sp.lambdify(a, exprs, modules="numpy", cse=True)
    cache[tag, order] = _Taylor(len(exprs), order, levels, fn, f0)
    return cache[tag, order]


# --------------------------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------------------------
@dataclass
class Perturbation:
    """Policy coefficients per monomial: ``coef[j]`` has shape (h, n, M_j), with
    ``y - ss_r = sum_j coef[j][r] @ monomials_j(z)`` and ``z = [states, sigma, unit shocks]``."""
    order: int
    states: np.ndarray              # indices of the state variables
    n_exo: int
    coef: dict
    ss: np.ndarray
    seconds: dict = field(default_factory=dict)      # solution time per order
    method: dict = field(default_factory=dict)       # "kron", "gmres" or "tfqmr" per order
    lower_residual: dict = field(default_factory=dict)
    transition_point: np.ndarray | None = None       # point where the probabilities are evaluated (default: ss)

    @property
    def nz(self) -> int:
        return self.states.size + 1 + self.n_exo

    def tensor(self, j: int) -> np.ndarray:
        """Derivative tensor G^j, shape (h, n, nz, ..., nz), like RISE's ``Tz``, ``Tzz``, ..."""
        Bz = _Basis(self.nz, j)
        idx = np.array(list(itertools.product(range(self.nz), repeat=j)), np.int64).reshape(-1, j)
        pos = Bz.lookup(j, _WEIGHTS[idx].sum(1))
        fact = np.array([math.prod(math.factorial(int(c)) for c in row) for row in Bz.exponents(j)], float)
        full = (self.coef[j] * fact)[:, :, pos]
        return full.reshape(full.shape[:2] + (self.nz,) * j)

    def stacked(self, order: int | None = None) -> tuple[np.ndarray, _Basis]:
        """Coefficients of all degrees up to ``order`` side by side (h, n, N_z) and the basis used."""
        p = self.order if order is None else order
        Bz = _Basis(self.nz, p)
        h, n = self.coef[1].shape[:2]
        out = np.zeros((h, n, Bz.N))
        for j in range(1, p + 1):
            out[:, :, Bz.block(j)] = self.coef[j]
        return out, Bz


# --------------------------------------------------------------------------------------------
# solution
# --------------------------------------------------------------------------------------------
def _transition_taylor(model, order):
    """Taylor series of the off-diagonal probabilities, per chain."""
    out = {}
    for c in model.chain_names:
        items = sorted(model._transitions[c].items())
        out[c] = ([ij for ij, _ in items], _taylor(model, [ex for _, ex in items], order, f"trans_{c}"))
    return out


def _q_polynomials(model, trans, r, point, V, basis, upto):
    """Polynomials Q_rq(w), q = 1..h, for the origin regime r (h x N)."""
    n = model.n
    Vq = V.copy()
    Vq[:n] = 0.0
    rows = []                                   # per chain: polynomials of the row of the current state
    for ci, c in enumerate(model.chain_names):
        ij, ty = trans[c]
        kc = model.spec.chains[c].regimes
        i_now = model.regimes[r][ci]
        P = np.zeros((kc, basis.N))
        if ij:
            vals = np.asarray(ty.fn(*point), float) if ty.levels and ty.levels[0]["tuples"].size else np.zeros(0)
            for lev in ty.levels:
                if lev["tuples"].size and np.any(lev["tuples"] < n):
                    lead = {int(q) for q, col in zip(lev["rows"], lev["cols"]) if np.any(lev["tuples"][col] < n)}
                    if any(ij[q][0] == i_now for q in lead):
                        raise NotImplementedError("transition probability that depends on a variable at t+1")
            comp = ty.compose(vals, Vq, basis, upto) if vals.size else np.zeros((len(ij), basis.N))
            const = np.asarray(ty.f0(*point), float).ravel()
            for q, (i, j) in enumerate(ij):
                if i == i_now and j != i:
                    P[j - 1] = comp[q]
                    P[j - 1, 0] += const[q]
        P[i_now - 1] = -P.sum(0)
        P[i_now - 1, 0] += 1.0
        rows.append(P)
    out = np.zeros((model.h, basis.N))
    for q, state in enumerate(model.regimes):
        poly = np.zeros((1, basis.N))
        poly[0, 0] = 1.0
        for ci in range(len(model.chain_names)):
            poly = basis.mul(poly, rows[ci][state[ci] - 1])
        out[q] = poly[0]
    return out


def _expansion(model, sys, ty, trans, params, ss, sig, st, coefs, j, endogenous, tpoint=None):
    """Expansion of E[sum_q Q_rq f(v_rq)] up to degree j with the given policies.

    Returns ``F`` (h, n, N_z) and ``C`` (h, h, M_j, M_j).
    """
    h, n = model.h, model.n
    ne, k = len(model.exogenous), st.size
    nz = k + 1 + ne
    S = _spaces(nz, ne, k, j)
    B, Bz = S.B, S.Bz
    G = np.zeros((h, n, Bz.N))
    for d, c in coefs.items():
        if d <= j:
            G[:, :, Bz.block(d)] = c
    Gw = np.zeros((h, n, B.N))
    Gw[:, :, S.zin] = G
    e0 = np.zeros(ne)
    F = np.zeros((h, n, Bz.N))
    C = np.zeros((h, h, Bz.M[j], Bz.M[j]))
    Exj = S.Ex[B.block(j)][:, Bz.block(j)]
    for r in range(h):
        point = (ss[r], ss[r], ss[r], e0, ss[r], params[r])
        vals = np.asarray(ty.fn(*point), float)
        V = np.zeros((3 * n + ne, B.N))
        V[n:2 * n] = Gw[r]
        V[2 * n + st, 1 + np.arange(k)] = 1.0                       # degree 1: monomial i is variable i
        V[3 * n + np.arange(ne), 1 + k + 1 + np.arange(ne)] = sig
        if endogenous:
            tp = ss[r] if tpoint is None else tpoint[r]
            Qp = _q_polynomials(model, trans, r, (tp, tp, tp, e0, ss[r], params[r]), V, B, j)
        acc = np.zeros((n, B.N))
        for q in range(h):
            if not endogenous and sys.Q[r, q] == 0:
                continue
            zp = np.zeros((nz, B.N))
            zp[:k] = Gw[r][st]
            zp[:k, 1 + k] += (ss[r] - ss[q])[st]
            zp[k, 1 + k] = 1.0
            zp[k + 1 + np.arange(ne), 1 + nz + np.arange(ne)] = 1.0
            Mz = np.zeros((Bz.N, B.N))
            Mz[0, 0] = 1.0
            for d in range(1, j + 1):
                par, last = S.parents[d]
                for i in np.unique(last):
                    sel = np.flatnonzero(last == i)
                    Mz[Bz.off[d] + sel] = B.mul(Mz[par[sel]], zp[i], amin=d - 1, bmin=1)
            C[r, q] = (Exj.T @ Mz[Bz.block(j)][:, B.block(j)].T).T
            V[:n] = G[q] @ Mz
            comp = ty.compose(vals, V, B, j)
            if endogenous:
                for i in range(n):
                    acc[i] += B.mul(comp[i:i + 1], Qp[q])[0]
            else:
                acc += sys.Q[r, q] * comp
        F[r] = (S.Ex.T @ acc.T).T
    return F, C


def _linear_solve(U, Ap, Q, C, A, dense_limit, tol, method):
    """Solves U_r X_r + sum_q Q[r,q] Ap_r X_q C_rq + A_r = 0."""
    h, n, M = A.shape
    rhs = -np.linalg.solve(U, A)
    if not np.any(rhs):
        return np.zeros_like(A), "kron"
    Lp = np.linalg.solve(U, Ap)
    size = h * n * M
    used = method if method != "auto" else ("kron" if size <= dense_limit else "gmres")
    if used == "kron":
        big = np.eye(size)
        for r in range(h):
            for q in range(h):
                if Q[r, q] != 0:
                    big[r * n * M:(r + 1) * n * M, q * n * M:(q + 1) * n * M] += Q[r, q] * np.kron(Lp[r], C[r, q].T)
        return np.linalg.solve(big, rhs.ravel()).reshape(h, n, M), used
    Cs = {(r, q): sps.csr_matrix(C[r, q]) for r in range(h) for q in range(h) if Q[r, q] != 0}

    def apply(x):
        X = x.reshape(h, n, M)
        out = X.copy()
        for (r, q), Crq in Cs.items():
            out[r] += Q[r, q] * (Lp[r] @ (Crq.T @ X[q].T).T)
        return out.ravel()

    op = LinearOperator((size, size), matvec=apply, dtype=float)
    if used == "gmres":
        sol, info = gmres(op, rhs.ravel(), rtol=tol, atol=0.0, restart=200, maxiter=2000)
        if info != 0 and method == "auto":            # fallback: TFQMR (the iterative solver used by RISE)
            used = "tfqmr"
            sol, info = tfqmr(op, rhs.ravel(), rtol=tol, atol=0.0, maxiter=20000)
    elif used == "tfqmr":
        sol, info = tfqmr(op, rhs.ravel(), rtol=tol, atol=0.0, maxiter=20000)
    else:
        raise ValueError("method must be 'auto', 'kron', 'gmres' or 'tfqmr'")
    if info != 0:
        raise SolveError(f"{used} did not converge on the perturbation linear system (info={info})")
    return sol.reshape(h, n, M), used


def solve_perturbation(solution, order: int = 3, dense_limit: int = 1500, tol: float = 1e-12,
                       method: str = "auto", transition_point=None) -> Perturbation:
    """Perturbation up to ``order`` starting from the 1st-order solution.

    ``method``: ``"kron"`` (dense system), ``"gmres"`` or ``"tfqmr"`` (iterative, without building
    the matrix); ``"auto"`` uses the dense one up to ``dense_limit`` unknowns.
    ``transition_point``: (h, n) matrix with the point where the endogenous probabilities and their
    derivatives are evaluated (default: the steady state of the regime). It serves to reproduce RISE,
    which evaluates the probabilities at another point when solving (see docs/VALIDATION.md).
    """
    if not 1 <= order <= MAX_ORDER:
        raise ValueError(f"order must be between 1 and {MAX_ORDER}")
    model = solution.model
    params, ss = solution.params, solution.ss
    sys = model.first_order_system(params, ss)
    if np.max(np.abs(sys.resid)) > 1e-8 or np.any(solution.c):
        raise NotImplementedError("higher order with a single approximation point (steady_state='unique') "
                                  "is not supported by this module")
    endogenous = bool(solution.endogenous_switching)
    h, n, ne = model.h, model.n, len(model.exogenous)
    st = _states(sys)
    k = st.size
    sig = solution.sigma
    G1 = np.zeros((h, n, k + 1 + ne))
    G1[:, :, :k] = solution.T[:, :, st]
    G1[:, :, k + 1:] = solution.R * sig
    pert = Perturbation(order=order, states=st, n_exo=ne, coef={1: G1}, ss=ss, transition_point=transition_point)
    if order == 1:
        return pert
    t0 = time.perf_counter()
    ty = _taylor(model, model._equations, order, "eqs")
    trans = _transition_taylor(model, order) if endogenous else None
    pert.seconds["derivatives"] = time.perf_counter() - t0
    sys.Q = solution.Q            # the matrix used at 1st order
    U = _U(sys, solution.T)
    for j in range(2, order + 1):
        t0 = time.perf_counter()
        F, C = _expansion(model, sys, ty, trans, params, ss, sig, st, pert.coef, j, endogenous, transition_point)
        Bz = _Basis(k + 1 + ne, j)
        A = F[:, :, Bz.block(j)]
        pert.lower_residual[j] = float(np.max(np.abs(F[:, :, Bz.off[2]:Bz.off[j]]))) if j > 2 else 0.0
        X, used = _linear_solve(U, sys.Ap, sys.Q, C, A, dense_limit, tol, method)
        pert.coef[j] = X
        pert.method[j] = used
        pert.seconds[j] = time.perf_counter() - t0
    return pert


def equation_residual(solution, order: int | None = None) -> np.ndarray:
    """Coefficients of the expansion of the equilibrium conditions under the full policy, (h, n, N_z).

    The blocks of degree 2 to ``order`` must be zero. At degree 1, the sigma column is zero only when
    the steady states do not differ across regimes (RISE convention).
    """
    pert = solution.pert
    order = pert.order if order is None else order
    model = solution.model
    sys = model.first_order_system(solution.params, solution.ss)
    sys.Q = solution.Q
    endogenous = bool(solution.endogenous_switching)
    ty = _taylor(model, model._equations, order, "eqs")
    trans = _transition_taylor(model, order) if endogenous else None
    F, _ = _expansion(model, sys, ty, trans, solution.params, solution.ss, solution.sigma, pert.states,
                      pert.coef, order, endogenous, pert.transition_point)
    return F


# --------------------------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------------------------
class _Stepper:
    """Evaluates the policy of order ``order`` with or without pruning."""

    def __init__(self, solution, order: int | None = None, pruned: bool = True, regime_switch: str = "first"):
        if regime_switch not in ("first", "rise"):
            raise ValueError("regime_switch must be 'first' or 'rise'")
        self.regime_switch = regime_switch
        pert = solution.pert
        if pert is None:
            raise ValueError("solve with order >= 2 to simulate at higher order")
        self.order = pert.order if order is None else order
        if self.order > pert.order:
            raise ValueError(f"the model was solved up to order {pert.order}")
        self.pruned, self.sol, self.pert = pruned, solution, pert
        self.G, self.Bz = pert.stacked(self.order)
        self.st, self.k, self.nz = pert.states, pert.states.size, pert.nz
        w = _WEIGHTS
        self.parents = [None]
        for d in range(1, self.order + 1):
            last = self.Bz.tup[d][:, -1]
            self.parents.append((self.Bz.off[d - 1] + self.Bz.lookup(d - 1, self.Bz.key[d] - w[last]), last))

    def start(self, x_dev: np.ndarray | None = None) -> np.ndarray:
        """State components (k, order); the initial deviation enters the 1st-order one."""
        X = np.zeros((self.k, self.order if self.pruned else 1))
        if x_dev is not None:
            X[:, 0] = x_dev
        return X

    def shift(self, X: np.ndarray, r_old: int, r_new: int) -> np.ndarray:
        """Regime change: deviations are now measured from the new steady state.

        ``regime_switch="first"`` (default): the difference between steady states enters only the
        1st-order component, and the sum of the components remains equal to the actual deviation.
        ``"rise"``: it enters every component, as RISE does in pruned simulation.
        """
        if r_old != r_new:
            X = X.copy()
            gap = (self.sol.ss[r_old] - self.sol.ss[r_new])[self.st]
            if self.regime_switch == "rise":
                X += gap[:, None]
            else:
                X[:, 0] += gap
        return X

    def step(self, r: int, X: np.ndarray, eta: np.ndarray):
        """One period in regime r. Returns y (level) and the components of next period's state."""
        p = self.order
        if not self.pruned:
            z = np.concatenate([X[:, 0], [1.0], eta])
            mon = np.zeros(self.Bz.N)
            mon[0] = 1.0
            for d in range(1, p + 1):
                par, last = self.parents[d]
                mon[self.Bz.block(d)] = mon[par] * z[last]
            dev = self.G[r] @ mon
            return self.sol.ss[r] + dev, dev[self.st][:, None]
        # series in lambda: z = lambda [x1, 1, eta] + sum_{j>=2} lambda^j [x_j, 0, 0]
        Z = np.zeros((self.nz, p + 1))
        Z[:self.k, 1:] = X
        Z[self.k, 1] = 1.0
        Z[self.k + 1:, 1] = eta
        mon = np.zeros((self.Bz.N, p + 1))
        mon[0, 0] = 1.0
        for d in range(1, p + 1):
            par, last = self.parents[d]
            A, Zl = mon[par], Z[last]
            blk = np.zeros((par.size, p + 1))
            for a in range(d - 1, p):
                for b in range(1, p - a + 1):
                    blk[:, a + b] += A[:, a] * Zl[:, b]
            mon[self.Bz.block(d)] = blk
        Y = self.G[r] @ mon                       # (n, p+1): component of each order
        return self.sol.ss[r] + Y.sum(1), Y[self.st, 1:]


def _next_regime(solution, r, y, y_prev, u):
    h = solution.T.shape[0]
    if h == 1:
        return 0
    row = solution.Q[r]
    if solution.endogenous_switching:
        row = np.clip(solution.model.transition_matrix(y, y, y_prev, solution.params[r], solution.ss[r], regime=r), 0, 1)
        row = row / row.sum()
    return int(np.searchsorted(np.cumsum(row), u, side="right").clip(0, h - 1))


def simulate(solution, periods: int = 200, seed: int = 0, regime0: int = 1, burn: int = 0, regimes=None,
             shocks=None, variables=None, order: int | None = None, pruned: bool = True,
             initial=None, regime_switch: str = "first") -> pd.DataFrame:
    """Simulates in levels with the policy of order ``order`` (default: the solved order).

    ``initial``: vector (n) with the level of the variables in the period before the first simulated
    one (default: steady state of the initial regime).

    ``shocks``: (periods, n_exo) matrix of unit-variance shocks to impose (otherwise drawn);
    ``regimes``: imposed path (1-based). ``pruned=False`` uses the unpruned policy.
    On a regime change the deviations are re-anchored at the steady state of the new regime
    (``regime_switch``: see ``_Stepper.shift``).
    """
    m = solution.model
    stp = _Stepper(solution, order, pruned, regime_switch)
    names, idx = solution._names(variables)
    rng = np.random.default_rng(seed)
    ne, total = len(m.exogenous), periods + burn
    if shocks is not None:
        shocks = np.asarray(shocks, float).reshape(-1, ne)
        if shocks.shape[0] < total:
            raise ValueError("not enough shocks for the number of periods (include the burn)")
    r = (int(regimes[0]) if regimes is not None else regime0) - 1
    y_prev = solution.ss[r].copy() if initial is None else np.asarray(initial, float)
    X = stp.start((y_prev - solution.ss[r])[stp.st])
    out, path = np.zeros((total, len(idx))), np.zeros(total, int)
    for t in range(total):
        if regimes is not None:
            r_new = int(regimes[min(t, len(regimes) - 1)]) - 1
            X, r = stp.shift(X, r, r_new), r_new
        eta = rng.standard_normal(ne) if shocks is None else shocks[t]      # same draw order as Solution.simulate
        y, Xn = stp.step(r, X, eta)
        out[t], path[t] = y[idx], r + 1
        X = Xn
        if regimes is None and m.h > 1:
            row = solution.Q[r]
            if solution.endogenous_switching:
                row = np.clip(m.transition_matrix(y, y, y_prev, solution.params[r], solution.ss[r], regime=r), 0, 1)
                row = row / row.sum()
            r_new = int(rng.choice(len(row), p=row))
            X, r = stp.shift(X, r, r_new), r_new
        y_prev = y
    df = pd.DataFrame(out[burn:], columns=names)
    df.insert(0, "regime", path[burn:])
    return df


def irf(solution, shock: str | None = None, periods: int = 40, regime: int | None = None,
        size: float = 1.0, variables=None, order: int | None = None, pruned: bool = True) -> pd.DataFrame:
    """Regime-specific IRF at higher order: path with a shock of ``size`` standard deviations in
    period 0 minus the path without shocks, both starting from the steady state of the regime and
    staying in it. Index: (shock, regime, period)."""
    m = solution.model
    stp = _Stepper(solution, order, pruned)
    names, idx = solution._names(variables)
    ne = len(m.exogenous)
    rows, keys = [], []
    for s in (m.exogenous if shock is None else [shock]):
        ks = m.exogenous.index(s)
        for r in (range(m.h) if regime is None else [regime - 1]):
            paths = []
            for amp in (0.0, size):
                X, path = stp.start(), np.zeros((periods, m.n))
                for t in range(periods):
                    eta = np.zeros(ne)
                    if t == 0:
                        eta[ks] = amp
                    path[t], X = stp.step(r, X, eta)
                paths.append(path)
            for t in range(periods):
                rows.append((paths[1] - paths[0])[t, idx])
                keys.append((s, r + 1, t))
    return pd.DataFrame(rows, columns=names, index=pd.MultiIndex.from_tuples(keys, names=["shock", "regime", "period"]))


def girf(solution, shock: str, periods: int = 40, draws: int = 500, size: float = 1.0, regime0: int | None = None,
         seed: int = 0, variables=None, order: int | None = None, pruned: bool = True, burn: int = 0) -> pd.DataFrame:
    """Nonlinear generalised IRF (Koop, Pesaran and Potter, 1996): average, over drawn shocks and
    regimes, of the difference between the simulation with and without the impulse (same random
    numbers).

    ``burn`` warm-up periods take the initial state to the ergodic distribution; with zero, it
    starts from the steady state of the initial regime (drawn from the ergodic distribution of Q if
    ``regime0`` is not given).
    """
    from .filtering import ergodic_distribution
    m = solution.model
    stp = _Stepper(solution, order, pruned)
    names, idx = solution._names(variables)
    ks, ne = m.exogenous.index(shock), len(m.exogenous)
    rng = np.random.default_rng(seed)
    pi = ergodic_distribution(solution.Q)
    acc = np.zeros((periods, m.n))
    for _ in range(draws):
        r0 = (regime0 - 1) if regime0 else int(rng.choice(m.h, p=pi))
        eta = rng.standard_normal((burn + periods, ne))
        u = rng.uniform(size=burn + periods)
        X0, y_prev0 = stp.start(), solution.ss[r0].copy()
        for t in range(burn):
            y, X0 = stp.step(r0, X0, eta[t])
            r_new = _next_regime(solution, r0, y, y_prev0, u[t])
            X0, r0, y_prev0 = stp.shift(X0, r0, r_new), r_new, y
        paths = []
        for amp in (0.0, size):
            r, X, y_prev = r0, X0, y_prev0
            path = np.zeros((periods, m.n))
            for t in range(periods):
                et = eta[burn + t].copy()
                if t == 0:
                    et[ks] += amp
                y, X = stp.step(r, X, et)
                path[t] = y
                r_new = _next_regime(solution, r, y, y_prev, u[burn + t])
                X, r, y_prev = stp.shift(X, r, r_new), r_new, y
            paths.append(path)
        acc += paths[1] - paths[0]
    return pd.DataFrame(acc[:, idx] / draws, columns=names, index=pd.RangeIndex(periods, name="period"))


def simulated_moments(solution, periods: int = 20000, burn: int = 500, seed: int = 0, variables=None,
                      order: int | None = None, pruned: bool = True, lags: int = 1) -> dict:
    """Mean, standard deviation, skewness, kurtosis and autocorrelations of a long simulation."""
    df = simulate(solution, periods=periods, burn=burn, seed=seed, variables=variables, order=order, pruned=pruned)
    x = df.drop(columns="regime")
    d = x - x.mean()
    sd = x.std(ddof=0)
    with np.errstate(all="ignore"):
        out = {"mean": x.mean(), "std": sd, "skewness": (d ** 3).mean() / sd ** 3, "kurtosis": (d ** 4).mean() / sd ** 4,
               "autocorr": pd.DataFrame({L: [x[c].autocorr(L) for c in x.columns] for L in range(1, lags + 1)},
                                        index=x.columns),
               "regime_frequency": df["regime"].value_counts(normalize=True).sort_index()}
    return out
