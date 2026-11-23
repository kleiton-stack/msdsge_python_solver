"""Reduced-form and structural VAR models, with and without Markov regime switching.

Reimplements RISE's ``rfvar``/``svar`` classes (``@abstvar``, ``+vartools``):

- ``VAR``: reduced form ``y_t = C d_t + B_1 y_{t-1} + ... + B_p y_{t-p} + u_t``, ``u_t ~ N(0, Sigma)``.
- ``SVAR``: structural form ``A0 y_t = C d_t + A_1 y_{t-1} + ... + A_p y_{t-p} + diag(s) e_t``,
  with ``diag(A0) = 1`` and ``e_t ~ N(0, I)``.

In both, ``d_t`` collects the constant, trend and exogenous variables, and any block of parameters
can be controlled by Markov chains (``MarkovChain``). Package conventions: ``Q[i, j] = Pr(s_{t+1}=j |
s_t=i)`` and 1-based regimes in the interface. Data matrices have time in the rows (``Y`` is T x n,
``X`` is T x K) and coefficients have equations in the rows (``B`` is n x K), so that ``Y ~ X B'``.
The columns of ``X`` follow the RISE order: deterministic terms, then the lags.
As in RISE, the first ``nlags`` observations of the sample serve as initial conditions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import linalg, optimize, special, stats

PRIOR_TYPES = ("minnesota", "jeffrey", "nw", "normal-wishart", "inw", "sz")


# ================================================================ utilities
def vech(S: np.ndarray) -> np.ndarray:
    """Stack the lower triangle by columns (order of RISE's ``s_i_j`` parameters)."""
    n = S.shape[0]
    return np.concatenate([S[j:, j] for j in range(n)])


def ivech(v: np.ndarray, n: int) -> np.ndarray:
    S = np.zeros((n, n))
    k = 0
    for j in range(n):
        S[j:, j] = v[k:k + n - j]
        S[j, j:] = v[k:k + n - j]
        k += n - j
    return S


def companion(B_lags: np.ndarray, n: int, p: int) -> np.ndarray:
    """Companion matrix from ``[B_1 ... B_p]`` (n x n p)."""
    if p == 0:
        return np.zeros((n, n))
    C = np.zeros((n * p, n * p))
    C[:n] = B_lags
    C[n:, :-n] = np.eye(n * (p - 1))
    return C


def ma_coefficients(B_lags: np.ndarray, n: int, p: int, periods: int) -> np.ndarray:
    """Coefficients of the MA representation: ``Theta[h]`` (n x n), h = 0..periods-1."""
    out = np.zeros((periods, n, n))
    if p == 0:
        out[0] = np.eye(n)
        return out
    C = companion(B_lags, n, p)
    P = np.eye(n * p)
    for h in range(periods):
        out[h] = P[:n, :n]
        P = C @ P
    return out


def long_run_multiplier(B_lags: np.ndarray, n: int, p: int) -> np.ndarray:
    """``(I - B_1 - ... - B_p)^{-1}``: cumulative long-run effect."""
    A1 = sum(B_lags[:, l * n:(l + 1) * n] for l in range(p)) if p else np.zeros((n, n))
    return np.linalg.inv(np.eye(n) - A1)


def gaussian_loglik(U: np.ndarray, Sigma: np.ndarray) -> float:
    T, n = U.shape
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        return -np.inf
    return float(-0.5 * T * n * np.log(2 * np.pi) - 0.5 * T * logdet - 0.5 * np.sum(U * np.linalg.solve(Sigma, U.T).T))


def ergodic(Q: np.ndarray) -> np.ndarray:
    h = Q.shape[0]
    A = np.vstack([np.eye(h) - Q.T, np.ones((1, h))])
    pi = np.linalg.lstsq(A, np.r_[np.zeros(h), 1.0], rcond=None)[0]
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


def _draw_inverse_wishart(S: np.ndarray, nu: float, rng) -> np.ndarray:
    return np.atleast_2d(stats.invwishart.rvs(df=nu, scale=S, random_state=rng))


# ================================================================ data
@dataclass
class VARData:
    Y: np.ndarray            # (T, n)
    X: np.ndarray            # (T, K)
    index: pd.Index          # dates of Y
    levels: np.ndarray       # (T + p, n): whole sample of the endogenous variables, including the initial conditions
    det: np.ndarray          # (T + p, nd): deterministic terms of the whole sample
    full_index: pd.Index


def _slice(data: pd.DataFrame, sample):
    if sample is None:
        return data
    start, end = sample
    return data.loc[start:end]


# ================================================================ Markov chains
@dataclass
class MarkovChain:
    """Markov chain that controls blocks of parameters.

    ``controls`` uses the RISE names: ``'c'`` (deterministic terms), ``'a0'``, ``'a1'``, ... (or
    ``'b1'``, ... in the reduced form), ``'s'`` (standard deviations or covariances). ``'a'``/``'b'``
    stand for all lags. An index in parentheses restricts it to one equation: ``'a0(1)'``, ``'s(1)'``.
    """
    name: str
    states: int
    controls: tuple = ()

    def __post_init__(self):
        self.controls = tuple(self.controls)
        if self.states < 2:
            raise ValueError("a chain needs at least 2 states")


class _Layout:
    """Map between the parameter vector (RISE names) and the matrices of each regime."""

    def __init__(self, n, nd, p, chains, structural: bool, endog):
        self.n, self.nd, self.p, self.structural = n, nd, p, structural
        self.chains = list(chains)
        self.endog = list(endog)
        self.lag_prefix = "a" if structural else "b"
        sizes = [c.states for c in self.chains]
        self.h = int(np.prod(sizes)) if sizes else 1
        # regime -> state of each chain; the first chain varies most slowly
        self.regime_states = np.array(list(np.ndindex(*sizes)), dtype=int).reshape(self.h, len(sizes))
        self.names: list[str] = []
        self._index: dict[tuple, int] = {}       # (block, eq, col, chain, state) -> position
        self._chain_of: dict[tuple, int | None] = {}
        blocks = [("c", nd)] + ([("a0", n)] if structural else []) + [(f"{self.lag_prefix}{l}", n) for l in range(1, p + 1)]
        for blk, ncol in blocks:
            for col in range(ncol):
                for eq in range(n):
                    self._add(blk, eq, col)
        if structural:
            for eq in range(n):
                self._add("s", eq, eq)
        else:
            for col in range(n):
                for row in range(col, n):
                    self._add("s", row, col)
        self.n_var = len(self.names)
        self.tp_names: list[str] = []
        self.tp_index: dict[tuple, int] = {}
        for ci, c in enumerate(self.chains):
            for i in range(c.states):
                for j in range(c.states):
                    if i != j:
                        self.tp_index[(ci, i, j)] = self.n_var + len(self.tp_names)
                        self.tp_names.append(f"{c.name}_tp_{i + 1}_{j + 1}")
        self.names += self.tp_names
        self.size = len(self.names)

    def _controller(self, blk, eq):
        """Index of the chain that controls block ``blk`` in equation ``eq`` (or None)."""
        is_lag = blk[0] == self.lag_prefix and blk[1:].isdigit() and blk != "a0"
        for ci, c in enumerate(self.chains):
            for item in c.controls:
                m = re.fullmatch(r"\s*([a-z]+\d*)\s*(?:\(\s*(\d+)\s*\))?\s*", item)
                if not m:
                    raise ValueError(f"invalid controlled parameter: {item!r}")
                name, which = m.group(1), m.group(2)
                hit = is_lag if name in ("a", "b") else name == blk
                if blk == "s" and not self.structural:
                    which = None            # full covariance: the chain controls the whole matrix
                if hit and (which is None or int(which) == eq + 1):
                    return ci
        return None

    def _add(self, blk, eq, col):
        ci = self._controller(blk, eq)
        self._chain_of[(blk, eq, col)] = ci
        base = f"{blk}_{eq + 1}_{col + 1}"
        if ci is None:
            self._index[(blk, eq, col, None, 0)] = len(self.names)
            self.names.append(base)
        else:
            c = self.chains[ci]
            for st in range(c.states):
                self._index[(blk, eq, col, ci, st)] = len(self.names)
                self.names.append(f"{base}_{c.name}_{st + 1}")

    def position(self, blk, eq, col, regime: int = 0, chain: str | None = None, state: int | None = None) -> int:
        ci = self._chain_of[(blk, eq, col)]
        if ci is None:
            if chain is not None:
                raise ValueError(f"{blk}({eq + 1},{col + 1}) is not controlled by chain {chain!r}")
            return self._index[(blk, eq, col, None, 0)]
        if chain is not None:
            if self.chains[ci].name != chain:
                raise ValueError(f"{blk}({eq + 1},{col + 1}) is controlled by {self.chains[ci].name!r}, not by {chain!r}")
            return self._index[(blk, eq, col, ci, state)]
        return self._index[(blk, eq, col, ci, int(self.regime_states[regime, ci]))]

    def positions_all_states(self, blk, eq, col):
        ci = self._chain_of[(blk, eq, col)]
        if ci is None:
            return [self._index[(blk, eq, col, None, 0)]]
        return [self._index[(blk, eq, col, ci, st)] for st in range(self.chains[ci].states)]

    # ---- matrices per regime
    def unpack(self, theta: np.ndarray):
        """Return (A0 (h,n,n), Bx (h,n,K), S (h,n) or (h,n,n), Q (h,h)). In the reduced form A0 = I."""
        n, nd, p, h = self.n, self.nd, self.p, self.h
        K = nd + n * p
        A0 = np.tile(np.eye(n), (h, 1, 1))
        Bx = np.zeros((h, n, K))
        S = np.zeros((h, n)) if self.structural else np.zeros((h, n, n))
        for r in range(h):
            for (blk, eq, col), ci in self._chain_of.items():
                v = theta[self.position(blk, eq, col, r)]
                if blk == "c":
                    Bx[r, eq, col] = v
                elif blk == "a0":
                    A0[r, eq, col] = v
                elif blk == "s":
                    if self.structural:
                        S[r, eq] = v
                    else:
                        S[r, eq, col] = S[r, col, eq] = v
                else:
                    Bx[r, eq, nd + (int(blk[1:]) - 1) * n + col] = v
        return A0, Bx, S, self.transition(theta)

    def chain_matrices(self, theta):
        out = []
        for ci, c in enumerate(self.chains):
            P = np.zeros((c.states, c.states))
            for i in range(c.states):
                for j in range(c.states):
                    if i != j:
                        P[i, j] = theta[self.tp_index[(ci, i, j)]]
                P[i, i] = 1 - P[i].sum()
            out.append(P)
        return out

    def transition(self, theta) -> np.ndarray:
        Q = np.ones((1, 1))
        for P in self.chain_matrices(theta):
            Q = np.kron(Q, P)
        return Q

    def pack(self, A0, Bx, S, chain_Ps=None) -> np.ndarray:
        """Inverse of ``unpack`` (uses the first regime in which each state appears)."""
        theta = np.zeros(self.size)
        n, nd = self.n, self.nd
        for r in reversed(range(self.h)):
            for (blk, eq, col), ci in self._chain_of.items():
                k = self.position(blk, eq, col, r)
                if blk == "c":
                    theta[k] = Bx[r, eq, col]
                elif blk == "a0":
                    theta[k] = A0[r, eq, col]
                elif blk == "s":
                    theta[k] = S[r, eq] if self.structural else S[r, eq, col]
                else:
                    theta[k] = Bx[r, eq, nd + (int(blk[1:]) - 1) * n + col]
        if chain_Ps is not None:
            for (ci, i, j), k in self.tp_index.items():
                theta[k] = chain_Ps[ci][i, j]
        return theta

    # ---- restrictions in the RISE format
    _TOKEN = re.compile(r"\b([a-z]+\d*)\s*\(([^()]*)\)")

    def parse_restriction(self, text: str):
        """Parse ``'a1(1,pi)+a0(3,FFR,chain,2)=0'``. Returns (coefficients, constant, operator):
        the restriction is ``coef @ theta + const (op) 0`` with op in ``'=', '>=', '<='``."""
        m = re.search(r"(>=|<=|=)", text)
        if not m:
            raise ValueError(f"restriction without '=', '>=' or '<=': {text!r}")
        op = m.group(1)
        lhs, rhs = text[:m.start()], text[m.end():]
        coef, const = np.zeros(self.size), 0.0
        for side, sgn in ((lhs, 1.0), (rhs, -1.0)):
            c1, k1 = self._linear(side)
            coef += sgn * c1
            const += sgn * k1
        return coef, const, op

    def _linear(self, expr: str):
        import sympy as sp
        syms, order = {}, []

        def repl(m):
            pos = self._lookup(m.group(1), [a.strip() for a in m.group(2).split(",")])
            name = f"THETA{len(order)}X"
            syms[name] = sp.Symbol(name)
            order.append(pos)
            return name
        clean = self._TOKEN.sub(repl, expr)
        for nm in self.tp_names:               # transition probabilities by name
            if re.search(rf"\b{re.escape(nm)}\b", clean):
                name = f"THETA{len(order)}X"
                syms[name] = sp.Symbol(name)
                order.append([self.names.index(nm)])
                clean = re.sub(rf"\b{re.escape(nm)}\b", name, clean)
        e = sp.sympify(clean, locals=syms) if clean.strip() else sp.Integer(0)
        coef = np.zeros(self.size)
        const = e
        for k, pos in enumerate(order):
            s = syms[f"THETA{k}X"]
            c = sp.diff(e, s)
            if c.free_symbols:
                raise ValueError(f"restriction is not linear: {expr!r}")
            if len(pos) != 1:
                raise ValueError(f"in {expr!r}: a regime-switching parameter requires chain and state")
            coef[pos[0]] += float(c)
            const = const.subs(s, 0)
        return coef, float(const)

    def _lookup(self, name, args):
        if name == "s" and not self.structural:
            eq, col = int(args[0]) - 1, self._var(args[1]) if len(args) > 1 else int(args[0]) - 1
            eq, col = max(eq, col), min(eq, col)
        elif name == "s":
            eq = col = int(args[0]) - 1
            args = [args[0], args[0]] + list(args[1:])
        elif name == "c":
            eq, col = int(args[0]) - 1, (int(args[1]) - 1 if len(args) > 1 and args[1].isdigit() else 0)
            if len(args) == 1 or not args[1].isdigit():
                args = [args[0], "1"] + list(args[1:])
        else:
            if len(args) < 2:
                raise ValueError(f"{name}({','.join(args)}): give equation and variable")
            eq, col = int(args[0]) - 1, self._var(args[1])
        if (name, eq, col) not in self._chain_of:
            raise ValueError(f"nonexistent parameter: {name}({','.join(args)})")
        if len(args) >= 4:
            return [self.position(name, eq, col, chain=args[2], state=int(args[3]) - 1)]
        return self.positions_all_states(name, eq, col)

    def _var(self, a):
        return int(a) - 1 if a.isdigit() else self.endog.index(a)


# ================================================================ priors
@dataclass
class VARPrior:
    """Hyperparameters of the BVAR prior (same fields as RISE's ``rfvar.prior_template()``).

    - ``L1``: overall tightness; ``L2``: cross-lag tightness (minnesota, inw); ``L3``:
      lag decay; ``L4``: tightness of the deterministic terms (minnesota, nw, inw).
    - ``sz`` prior (Sims-Zha by dummy observations, as RISE implements it): ``L5`` is the number
      of repetitions of the covariance dummy observation, ``L6`` the weight of the dummy initial
      observation (co-persistence) and ``L7`` the weight of the sum of coefficients.
    - ``coefprior``: prior mean of the coefficient on the first own lag (default 1).
    """
    type: str = "minnesota"
    L1: float = 0.1
    L2: float = 0.5
    L3: float = 1.0
    L4: float = 100.0
    L5: float = 3.0
    L6: float = 5.0
    L7: float = 2.0
    coefprior: float | None = None
    normal_wishart_eta: float = 0.01
    independent_normal_wishart_eta: float = 0.01

    def __post_init__(self):
        if self.type not in PRIOR_TYPES:
            raise ValueError(f"unknown prior: {self.type!r}; options: {PRIOR_TYPES}")


def sims_zha_dummies(levels: np.ndarray, nd: int, p: int, prior: VARPrior):
    """Dummy observations of RISE's ``sz`` prior. Returns (Yd, Xd), with time in the rows.

    Identified numerically from RISE (see docs/VALIDATION.md): mean and standard deviation
    (with T-1) computed on the whole sample, including the initial conditions. Blocks, in this
    order: n p Minnesota-type observations (``sigma_i l^L3 / L1``), ``L5`` repetitions of
    ``diag(sigma)``, n sum-of-coefficients observations (``L7 ybar_i``) and one co-persistence
    observation (``L6``). The constant is the first deterministic column; the other deterministic
    terms get zero.
    """
    n = levels.shape[1]
    ybar, sig = levels.mean(0), levels.std(0, ddof=1)
    K = nd + n * p
    yd, xd = [], []
    for l in range(1, p + 1):
        for i in range(n):
            y, x = np.zeros(n), np.zeros(K)
            if l == 1:
                y[i] = sig[i] / prior.L1
            x[nd + (l - 1) * n + i] = sig[i] * l ** prior.L3 / prior.L1
            yd.append(y), xd.append(x)
    for _ in range(int(round(prior.L5))):
        for i in range(n):
            y = np.zeros(n)
            y[i] = sig[i]
            yd.append(y), xd.append(np.zeros(K))
    if prior.L7 > 0:
        for i in range(n):
            y, x = np.zeros(n), np.zeros(K)
            y[i] = prior.L7 * ybar[i]
            for l in range(p):
                x[nd + l * n + i] = prior.L7 * ybar[i]
            yd.append(y), xd.append(x)
    if prior.L6 > 0:
        x = np.zeros(K)
        if nd:
            x[0] = prior.L6
        x[nd:] = np.tile(prior.L6 * ybar, p)
        yd.append(prior.L6 * ybar), xd.append(x)
    return np.array(yd).reshape(-1, n), np.array(xd).reshape(-1, K)


def _ar_sigmas(levels: np.ndarray, p: int) -> np.ndarray:
    """Residual standard deviation of an AR(p) with constant for each variable (Minnesota scale)."""
    out = []
    for i in range(levels.shape[1]):
        y = levels[:, i]
        q = max(p, 1)
        Z = np.column_stack([np.ones(len(y) - q)] + [y[q - l:len(y) - l] for l in range(1, q + 1)])
        e = y[q:] - Z @ np.linalg.lstsq(Z, y[q:], rcond=None)[0]
        out.append(np.sqrt(e @ e / max(len(e) - Z.shape[1], 1)))
    return np.array(out)


# ================================================================ identification
_RESTR = re.compile(r"^\s*(\w+)\s*(?:\{(.*)\})?\s*@\s*(\w+)\s*$")


def _parse_lags(text):
    if text is None or not text.strip():
        return [0]
    t = text.replace("[", "").replace("]", "")
    out = []
    for part in re.split(r"[,\s]+", t.strip()):
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":")
            out += list(range(int(a), int(b) + 1))
        elif part.lower() == "inf":
            out.append(np.inf)
        else:
            out.append(int(part))
    return out


@dataclass
class Identification:
    """Identification scheme: returns the impact matrix ``R`` (column = shock) with ``R R' = Sigma``.

    Zeros (at any horizon, including ``inf``) are imposed as in Rubio-Ramírez, Waggoner and
    Zha (2010): the shocks are ordered by the number of zeros and each column of the rotation is
    taken from the null space of its restrictions and of the previous columns. When there is slack
    (underidentification), the column is drawn in that space and the sign restrictions are checked
    by accept-reject (Arias, Rubio-Ramírez and Waggoner, 2018, without the importance weights).
    Without a sign restriction, the column is normalised so that the variable in the same position
    responds positively.
    """
    endog: list
    shocks: list
    zeros: list            # (variable, horizon, shock)
    signs: list            # (variable, horizon, shock, sign)
    n: int
    p: int
    nd: int
    max_trials: int = 6000
    seed: int = 0
    choleski_order: list | None = None
    trials_used: int = 0

    def _counts(self):
        c = np.zeros(self.n, int)
        for (_, _, j) in self.zeros:
            c[j] += 1
        return c

    @property
    def is_over(self) -> bool:
        c = np.sort(self._counts())[::-1]
        return bool(np.any(c > self.n - 1 - np.arange(self.n)))

    @property
    def is_exact(self) -> bool:
        c = np.sort(self._counts())[::-1]
        return bool(np.all(c == self.n - 1 - np.arange(self.n)))

    @property
    def is_under(self) -> bool:
        return not self.is_exact and not self.is_over

    @property
    def shock_names(self) -> list:
        return list(self.shocks) + [f"shock_{k + 1}" for k in range(len(self.shocks), self.n)]

    def _responses(self, B, L, horizons):
        n, p, nd = self.n, self.p, self.nd
        lags = B[:, nd:]
        finite = [int(h) for h in horizons if np.isfinite(h)]
        out = {}
        if finite:
            Th = ma_coefficients(lags, n, p, max(finite) + 1)
            for h in finite:
                out[h] = Th[h] @ L
        if any(not np.isfinite(h) for h in horizons):
            out[np.inf] = long_run_multiplier(lags, n, p) @ L
        return out

    def __call__(self, B: np.ndarray, Sigma: np.ndarray, rng=None) -> np.ndarray:
        n = self.n
        if self.choleski_order is not None:
            o = [self.endog.index(v) for v in self.choleski_order]
            L = np.zeros((n, n))
            L[np.ix_(o, o)] = np.linalg.cholesky(Sigma[np.ix_(o, o)])
            return L
        rng = np.random.default_rng(self.seed) if rng is None else rng
        L = np.linalg.cholesky(Sigma)
        F = self._responses(B, L, {h for (_, h, _) in self.zeros} | {h for (_, h, _, _) in self.signs} | {0})
        counts = self._counts()
        order = sorted(range(n), key=lambda j: -counts[j])
        for trial in range(1, self.max_trials + 1):
            Qm = np.zeros((n, n))
            done, random_used, ok = [], False, True
            for j in order:
                rows = [F[h][i] for (i, h, jj) in self.zeros if jj == j] + [Qm[:, k] for k in done]
                N = linalg.null_space(np.array(rows), rcond=1e-10) if rows else np.eye(n)
                if N.shape[1] == 0:
                    raise ValueError("incompatible zero restrictions (overidentified model)")
                if N.shape[1] == 1:
                    q = N[:, 0]
                else:
                    x = N.T @ rng.standard_normal(n)
                    q = N @ x / np.linalg.norm(x)
                    random_used = True
                sg = [s * (F[h][i] @ q) for (i, h, jj, s) in self.signs if jj == j]
                if sg:
                    if all(v < 0 for v in sg):
                        q = -q
                    elif not all(v > 0 for v in sg):
                        ok = False
                elif j < n and (L @ q)[j] < 0:
                    q = -q
                Qm[:, j] = q
                done.append(j)
                if not ok:
                    break
            self.trials_used = trial
            if ok:
                return L @ Qm
            if not random_used:
                raise ValueError("the sign restrictions are not satisfied by the exactly identified rotation")
        raise ValueError(f"no rotation satisfied the sign restrictions in {self.max_trials} trials")

    def check(self, B, R, tol: float = 1e-8) -> bool:
        """Check whether ``R`` satisfies all zero and sign restrictions."""
        F = self._responses(B, R, {h for (_, h, _) in self.zeros} | {h for (_, h, _, _) in self.signs} | {0})
        return (all(abs(F[h][i, j]) < tol for (i, h, j) in self.zeros)
                and all(s * F[h][i, j] > 0 for (i, h, j, s) in self.signs))


def make_identification(endog, n, p, nd, restrictions="choleski", shocks=None, max_trials: int = 6000,
                        seed: int = 0, agnostic: bool = True) -> Identification:
    """Build the scheme from the RISE syntax.

    ``restrictions`` is ``'choleski'`` (``shocks`` then becomes the ordering of the variables) or a
    list of pairs ``('variable{horizons}@shock', '+' | '-' | 0)``; ``{inf}`` is the long run and
    ``{0:3}``/``{1,4}`` list horizons. ``agnostic`` exists for compatibility: the algorithm is the
    same in both cases.
    """
    endog = list(endog)
    if isinstance(restrictions, str):
        if restrictions.lower() not in ("choleski", "cholesky"):
            raise ValueError("restrictions: 'choleski' or a list of pairs")
        order = list(shocks) if shocks is not None else endog
        if sorted(order) != sorted(endog):
            raise ValueError("the Cholesky ordering must contain all endogenous variables")
        return Identification(endog, [f"shock_{k + 1}" for k in range(n)], [], [], n, p, nd, choleski_order=order)
    shocks = list(shocks) if shocks is not None else []
    zeros, signs = [], []
    for target, kind in restrictions:
        m = _RESTR.match(target)
        if not m:
            raise ValueError(f"invalid restriction: {target!r}")
        v, lags, s = m.group(1), _parse_lags(m.group(2)), m.group(3)
        if v not in endog:
            raise ValueError(f"unknown variable in restriction {target!r}")
        if s not in shocks:
            shocks.append(s)
        i, j = endog.index(v), shocks.index(s)
        for h in lags:
            if kind in ("+", "-"):
                signs.append((i, h, j, 1.0 if kind == "+" else -1.0))
            elif kind == 0:
                zeros.append((i, h, j))
            else:
                raise ValueError(f"invalid restriction type in {target!r}: use '+', '-' or 0")
    if len(shocks) > n:
        raise ValueError("there are more named shocks than variables")
    return Identification(endog, shocks, zeros, signs, n, p, nd, max_trials=max_trials, seed=seed)


# ================================================================ linear restrictions
def _restricted_ls(blocks, Kmat, d):
    """Generalised least squares with ``vec(B) = d + K x``.

    ``blocks`` is a list of (Sxx, Syx, W): weighted moments of the regressors and the weight matrix
    (inverse of the covariance) of each regime, all relative to the same ``vec(B)`` stacked by regime.
    """
    nb = len(blocks)
    n, K = blocks[0][1].shape
    size = n * K
    H = np.zeros((nb * size, nb * size))
    g = np.zeros(nb * size)
    for b, (Sxx, Syx, W) in enumerate(blocks):
        sl = slice(b * size, (b + 1) * size)
        H[sl, sl] = np.kron(Sxx, W)
        g[sl] = (W @ Syx).reshape(-1, order="F")
    A = Kmat.T @ H @ Kmat
    x = np.linalg.lstsq(A, Kmat.T @ (g - H @ d), rcond=None)[0]
    return d + Kmat @ x


# ================================================================ model
class VAR:
    """Reduced-form VAR (RISE's ``rfvar``).

    ``VAR(endog, exog=(), nlags=1, constant=True, trend=False, markov_chains=())``. Estimate with
    ``fit(data, sample=(start, end), prior=..., restrictions=[...])``.
    """
    structural = False

    def __init__(self, endog, exog=(), nlags: int = 1, constant: bool = True, trend: bool = False,
                 markov_chains=()):
        self.endog, self.exog = list(endog), list(exog or ())
        self.nlags, self.constant, self.trend = int(nlags), bool(constant), bool(trend)
        if isinstance(markov_chains, MarkovChain):
            markov_chains = [markov_chains]
        self.chains = list(markov_chains or ())
        self.n = len(self.endog)
        self.det_names = (["const"] if constant else []) + (["trend"] if trend else []) + self.exog
        self.nd = len(self.det_names)
        self.K = self.nd + self.n * self.nlags
        self.layout = _Layout(self.n, self.nd, self.nlags, self.chains, self.structural, self.endog)
        self.h = self.layout.h

    # ---- names
    @property
    def parameter_names(self) -> list:
        return list(self.layout.names)

    @property
    def regressor_names(self) -> list:
        return self.det_names + [f"{v}{{-{l}}}" for l in range(1, self.nlags + 1) for v in self.endog]

    @property
    def is_switching(self) -> bool:
        return self.h > 1

    # ---- data
    def _deterministics(self, df: pd.DataFrame, offset: int = 0) -> np.ndarray:
        cols = []
        if self.constant:
            cols.append(np.ones(len(df)))
        if self.trend:
            cols.append(np.arange(1 + offset, len(df) + 1 + offset, dtype=float))
        for x in self.exog:
            cols.append(df[x].to_numpy(float))
        return np.column_stack(cols) if cols else np.zeros((len(df), 0))

    def make_data(self, data: pd.DataFrame, sample=None) -> VARData:
        df = _slice(data, sample)
        cols = self.endog + self.exog
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"columns missing from the data: {missing}")
        df = df[cols]
        if df.isna().any().any():
            raise ValueError("there are missing values in the sample; handle them before estimating the VAR")
        lev = df[self.endog].to_numpy(float)
        det = self._deterministics(df)
        p, T = self.nlags, len(df) - self.nlags
        if T <= self.K:
            raise ValueError("sample too short for the number of regressors")
        X = np.column_stack([det[p:]] + [lev[p - l:p - l + T] for l in range(1, p + 1)])
        return VARData(Y=lev[p:], X=X, index=df.index[p:], levels=lev, det=det, full_index=df.index)

    # ---- restrictions
    def _split_restrictions(self, restrictions):
        eqR, eqr, ineq = [], [], []
        for text in restrictions or ():
            coef, const, op = self.layout.parse_restriction(text)
            if op == "=":
                eqR.append(coef), eqr.append(-const)
            else:
                ineq.append((coef, const) if op == ">=" else (-coef, -const))
        return np.array(eqR).reshape(-1, self.layout.size), np.array(eqr), ineq

    def _automatic_equalities(self):
        return np.zeros((0, self.layout.size)), np.zeros(0)

    # ---- estimation
    def fit(self, data: pd.DataFrame, sample=None, prior: VARPrior | str | None = None, restrictions=None,
            method: str | None = None, **options) -> "VARResults":
        """Estimate the VAR.

        - Without Markov chains: OLS (``method='ols'``, default, covariance with T-K as in RISE) or
          maximum likelihood (``'ml'``); with ``prior`` (type or ``VARPrior``), the BVAR posterior.
        - With Markov chains: ``method='em'`` (default without prior or inequalities, reduced
          form) or ``'optimize'`` (posterior mode or maximum likelihood by direct optimisation).
          Options: ``transition_priors``, ``regimes`` (known regimes, 1-based), ``x0``,
          ``starts``, ``seed``, ``maxiter``, ``tol``.
        """
        d = self.make_data(data, sample)
        if isinstance(prior, str):
            prior = VARPrior(type=prior)
        if not self.is_switching and not self.structural:
            return self._fit_constant(d, prior, restrictions, method or "ols", **options)
        return _fit_switching(self, d, prior, restrictions, method, **options)

    def _fit_constant(self, d: VARData, prior, restrictions, method, draws: int = 2000, burn: int = 500,
                      seed: int = 0) -> "VARResults":
        n, K, lay = self.n, self.K, self.layout
        ncoef = n * K
        R, r, ineq = self._split_restrictions(restrictions)
        if ineq:
            raise ValueError("inequalities are only accepted in estimation by optimisation (switching models or SVAR)")
        if R.size and np.any(np.abs(R[:, ncoef:]) > 0):
            raise ValueError("in the constant-parameter VAR the linear restrictions apply to the coefficients")
        Kmat, dvec, _ = _rref_parametrization(R[:, :ncoef], r, ncoef)
        restricted = R.size > 0
        Y, X, T = d.Y, d.X, d.Y.shape[0]
        post = None

        def ls(Ya, Xa, W=None):
            if not restricted and W is None:
                return np.linalg.solve(Xa.T @ Xa, Xa.T @ Ya).T
            b = _restricted_ls([(Xa.T @ Xa, Ya.T @ Xa, np.eye(n) if W is None else W)], Kmat, dvec)
            return b.reshape(n, K, order="F")

        ptype = prior.type if prior is not None else None
        if ptype is None:
            if method not in ("ols", "ml"):
                raise ValueError("method: 'ols' or 'ml'")
            B = ls(Y, X)
            U = Y - X @ B.T
            Sigma = U.T @ U / (T - K if method == "ols" else T)
            if method == "ml" and restricted:
                for _ in range(500):          # iterated GLS up to the likelihood maximum
                    Bn = ls(Y, X, np.linalg.inv(Sigma))
                    U = Y - X @ Bn.T
                    Sn = U.T @ U / T
                    done = max(np.max(np.abs(Bn - B)), np.max(np.abs(Sn - Sigma))) < 1e-12
                    B, Sigma = Bn, Sn
                    if done:
                        break
        elif ptype in ("sz", "jeffrey"):
            Yd, Xd = sims_zha_dummies(d.levels, self.nd, self.nlags, prior) if ptype == "sz" else (np.zeros((0, n)), np.zeros((0, K)))
            Ya, Xa = np.vstack([Y, Yd]), np.vstack([X, Xd])
            B = ls(Ya, Xa)
            Ua = Ya - Xa @ B.T
            Ta = Ya.shape[0]
            Sigma = Ua.T @ Ua / (Ta if ptype == "sz" else T - K)
            post = dict(kind="niw", B=B, Phi=np.linalg.inv(Xa.T @ Xa), S=Ua.T @ Ua, nu=Ta - K, restricted=restricted)
        else:
            if restricted:
                raise ValueError(f"linear restrictions not implemented with the {ptype!r} prior")
            sig = _ar_sigmas(d.levels, self.nlags)
            ar = 1.0 if prior.coefprior is None else float(prior.coefprior)
            B0 = np.zeros((n, K))
            for i in range(n):
                if self.nlags:
                    B0[i, self.nd + i] = ar
            if ptype in ("nw", "normal-wishart"):
                phi = np.zeros(K)
                phi[:self.nd] = (prior.L1 * prior.L4) ** 2
                for l in range(1, self.nlags + 1):
                    phi[self.nd + (l - 1) * n:self.nd + l * n] = (prior.L1 / l ** prior.L3) ** 2 / sig ** 2
                alpha0 = n + 2
                S0 = (alpha0 - n - 1) * np.diag(sig ** 2)
                Pinv = np.diag(1 / phi)
                Phi = np.linalg.inv(Pinv + X.T @ X)
                B = (Phi @ (Pinv @ B0.T + X.T @ Y)).T
                S = Y.T @ Y + S0 + B0 @ Pinv @ B0.T - B @ (Pinv + X.T @ X) @ B.T
                S = (S + S.T) / 2
                nu = T + alpha0
                Sigma = S / (nu - n - 1)
                post = dict(kind="niw", B=B, Phi=Phi, S=S, nu=nu, restricted=False)
            else:
                # Minnesota-type prior variance for vec(B) (by columns)
                V = np.zeros((n, K))
                for i in range(n):
                    V[i, :self.nd] = sig[i] ** 2 * (prior.L1 * prior.L4) ** 2
                    for l in range(1, self.nlags + 1):
                        for j in range(n):
                            V[i, self.nd + (l - 1) * n + j] = ((prior.L1 / l ** prior.L3) ** 2 if i == j else
                                                              (sig[i] / sig[j]) ** 2 * (prior.L1 * prior.L2 / l ** prior.L3) ** 2)
                Oinv = np.diag(1 / V.reshape(-1, order="F"))
                b0 = B0.reshape(-1, order="F")
                XX, XY = X.T @ X, X.T @ Y

                def cond_beta(Sig):
                    Si = np.linalg.inv(Sig)
                    Om = np.linalg.inv(Oinv + np.kron(XX, Si))
                    Om = (Om + Om.T) / 2
                    return Om @ (Oinv @ b0 + (Si @ XY.T).reshape(-1, order="F")), Om
                if ptype == "minnesota":
                    Sigma = np.diag(sig ** 2)
                    bbar, Om = cond_beta(Sigma)
                    B = bbar.reshape(n, K, order="F")
                    post = dict(kind="normal", b=bbar, Omega=Om, Sigma=Sigma)
                else:                       # inw: Gibbs sampler
                    rng = np.random.default_rng(seed)
                    alpha0 = n + 2
                    S0 = (alpha0 - n - 1) * np.diag(sig ** 2)
                    Bc = np.linalg.solve(XX, XY).T
                    Sig = (Y - X @ Bc.T).T @ (Y - X @ Bc.T) / T
                    Bs, Ss = np.zeros((draws, n, K)), np.zeros((draws, n, n))
                    for it in range(burn + draws):
                        bbar, Om = cond_beta(Sig)
                        Bc = rng.multivariate_normal(bbar, Om).reshape(n, K, order="F")
                        U = Y - X @ Bc.T
                        Sig = _draw_inverse_wishart(S0 + U.T @ U, T + alpha0, rng)
                        if it >= burn:
                            Bs[it - burn], Ss[it - burn] = Bc, Sig
                    B, Sigma = Bs.mean(0), Ss.mean(0)
                    post = dict(kind="draws", B=Bs, Sigma=Ss)
        U = Y - X @ B.T
        theta = lay.pack(np.eye(n)[None], B[None], Sigma[None])
        return VARResults(model=self, data=d, theta=theta, A0=np.eye(n)[None], Bs=B[None], Sigmas=Sigma[None],
                          Q=np.ones((1, 1)), loglik=gaussian_loglik(U, Sigma), prior=prior, posterior=post,
                          method=ptype or method)


# ================================================================ results
@dataclass
class VARResults:
    """Estimates of a VAR and the derived analyses (IRF, decompositions, forecast).

    The matrices per regime are in ``Bs`` (h, n, K), ``Sigmas`` (h, n, n) and ``A0`` (h, n, n);
    ``B`` and ``Sigma`` return those of regime 1 (the only ones when there is no switching).
    ``theta`` is the parameter vector in the RISE order and with the RISE names (``names``).
    """
    model: VAR
    data: VARData
    theta: np.ndarray
    A0: np.ndarray
    Bs: np.ndarray
    Sigmas: np.ndarray
    Q: np.ndarray
    loglik: float
    prior: VARPrior | None = None
    posterior: dict | None = None
    method: str = "ols"
    logprior: float = 0.0
    S0: np.ndarray | None = None            # (h, n) standard deviations of the structural shocks (SVAR)
    filtered_probs: pd.DataFrame | None = None     # Pr(s_t | data up to t-1)
    updated_probs: pd.DataFrame | None = None      # Pr(s_t | data up to t)
    smoothed_probs: pd.DataFrame | None = None
    smoothed_state_probs: dict | None = None       # per chain
    loglik_t: np.ndarray | None = None
    message: str = ""
    iterations: int = 0
    free: np.ndarray | None = None          # x in the parameterisation theta = d + K x
    restriction_map: tuple | None = None    # (K, d)
    objective: object = None                # -log posterior as a function of x

    # ---- access
    @property
    def names(self) -> list:
        return self.model.parameter_names

    @property
    def params(self) -> pd.Series:
        return pd.Series(self.theta, index=self.names)

    @property
    def B(self) -> np.ndarray:
        return self.Bs[0]

    @property
    def Sigma(self) -> np.ndarray:
        return self.Sigmas[0]

    @property
    def h(self) -> int:
        return self.Bs.shape[0]

    @property
    def logpost(self) -> float:
        return self.loglik + self.logprior

    def coefficients(self, regime: int = 1) -> pd.DataFrame:
        return pd.DataFrame(self.Bs[regime - 1], index=self.model.endog, columns=self.model.regressor_names)

    def _lags(self, regime):
        return self.Bs[regime - 1][:, self.model.nd:]

    def is_stationary(self, regime: int = 1) -> bool:
        m = self.model
        return bool(np.max(np.abs(np.linalg.eigvals(companion(self._lags(regime), m.n, m.nlags)))) < 1) if m.nlags else True

    def residuals(self, regime: int = 1) -> pd.DataFrame:
        """Reduced-form residuals with the parameters of the given regime."""
        U = self.data.Y - self.data.X @ self.Bs[regime - 1].T
        return pd.DataFrame(U, index=self.data.index, columns=self.model.endog)

    # ---- identification
    def identify(self, restrictions="choleski", shocks=None, agnostic: bool = True, max_trials: int = 6000,
                 seed: int = 0) -> Identification:
        m = self.model
        return make_identification(m.endog, m.n, m.nlags, m.nd, restrictions, shocks, max_trials, seed, agnostic)

    def impact(self, ident=None, regime: int = 1, rng=None) -> np.ndarray:
        """Impact matrix of the structural shocks (column = shock).

        ``ident``: ``None`` (as in RISE: Cholesky in the order of the variables in the reduced-form
        VAR and ``A0^{-1} diag(s)`` in the SVAR), ``'choleski'``, ``'structural'``, an
        ``Identification`` or an n x n matrix.
        """
        r = regime - 1
        if ident is None:
            ident = "structural" if self.S0 is not None else "choleski"
        if isinstance(ident, str):
            if ident in ("choleski", "cholesky"):
                return np.linalg.cholesky(self.Sigmas[r])
            if ident != "structural" or self.S0 is None:
                raise ValueError("ident='structural' requires an SVAR")
            return np.linalg.solve(self.A0[r], np.diag(self.S0[r]))
        if isinstance(ident, Identification):
            return ident(self.Bs[r], self.Sigmas[r], rng)
        return np.asarray(ident, float)

    def _shock_names(self, ident):
        return ident.shock_names if isinstance(ident, Identification) else [f"shock_{k + 1}" for k in range(self.model.n)]

    def _regimes(self, regime):
        return list(range(1, self.h + 1)) if regime is None else [regime]

    # ---- IRF
    def irf(self, periods: int = 40, ident=None, shocks=None, regime: int | None = None) -> pd.DataFrame:
        """Impulse responses (period 0 = impact), conditional on staying in the regime.

        Index (shock, period), or (shock, regime, period) when there is regime switching.
        """
        m = self.model
        names = self._shock_names(ident)
        sel = names if shocks is None else list(shocks)
        rows, keys = [], []
        impacts = {r: self.impact(ident, r) for r in self._regimes(regime)}
        for s in sel:
            k = names.index(s)
            for r, R in impacts.items():
                Th = ma_coefficients(self._lags(r), m.n, m.nlags, periods)
                for t in range(periods):
                    rows.append(Th[t] @ R[:, k])
                    keys.append((s, r, t) if self.h > 1 else (s, t))
        idx = pd.MultiIndex.from_tuples(keys, names=["shock", "regime", "period"] if self.h > 1 else ["shock", "period"])
        return pd.DataFrame(rows, columns=m.endog, index=idx)

    def girf(self, periods: int = 40, ident=None, draws: int = 300, seed: int = 0, regime0: int | None = None) -> pd.DataFrame:
        """Generalised IRF with regime switching: mean, over drawn regime paths, of the response
        to a unit structural shock in period 0 (the initial regime comes from the ergodic
        distribution, unless ``regime0`` is given)."""
        m, rng = self.model, np.random.default_rng(seed)
        n, p, nd = m.n, m.nlags, m.nd
        names = self._shock_names(ident)
        impacts = [self.impact(ident, r + 1) for r in range(self.h)]
        pi = ergodic(self.Q)
        acc = np.zeros((n, periods, n))
        for _ in range(draws):
            r = regime0 - 1 if regime0 else int(rng.choice(self.h, p=pi))
            state = np.zeros((n * max(p, 1), n))          # one column per shock
            for t in range(periods):
                if t == 0:
                    y = impacts[r].copy()
                else:
                    y = self.Bs[r][:, nd:] @ state[:n * p] if p else np.zeros((n, n))
                acc[:, t, :] += y.T
                if p:
                    state = np.vstack([y, state[:n * (p - 1)]])
                r = int(rng.choice(self.h, p=self.Q[r]))
        acc /= draws
        idx = pd.MultiIndex.from_product([names, range(periods)], names=["shock", "period"])
        return pd.DataFrame(acc.reshape(n * periods, n), columns=m.endog, index=idx)

    # ---- moments
    def autocov(self, max_periods: int = 5, regime: int = 1):
        """Theoretical autocovariances and autocorrelations: ``C[:, :, k] = Cov(y_t, y_{t-k})``."""
        m = self.model
        n, p = m.n, m.nlags
        C = companion(self._lags(regime), n, p)
        if np.max(np.abs(np.linalg.eigvals(C))) >= 1:
            raise ValueError("non-stationary VAR: the autocovariances do not exist")
        Om = np.zeros_like(C)
        Om[:n, :n] = self.Sigmas[regime - 1]
        V = linalg.solve_discrete_lyapunov(C, Om)
        out = np.zeros((n, n, max_periods + 1))
        P = V.copy()
        for k in range(max_periods + 1):
            out[:, :, k] = P[:n, :n]
            P = C @ P
        sd = np.sqrt(np.diag(out[:, :, 0]))
        return out, out / np.outer(sd, sd)[:, :, None]

    def variance_decomposition(self, periods=400, ident=None, regime: int = 1) -> pd.DataFrame:
        """Forecast error variance decomposition, in fractions that sum to one.

        Index (variable, horizon), with horizons 1..``periods`` (or the given list) and ``inf``
        (when the VAR is stationary); horizon 1 is the impact, as in RISE.
        """
        m = self.model
        hs = list(range(1, periods + 1)) if np.isscalar(periods) else list(periods)
        R = self.impact(ident, regime)
        Th = ma_coefficients(self._lags(regime), m.n, m.nlags, max(hs))
        contrib = np.cumsum((Th @ R) ** 2, axis=0)            # (H, n, shocks)
        blocks = {k: contrib[k - 1] for k in hs}
        if self.is_stationary(regime):
            C = companion(self._lags(regime), m.n, m.nlags) if m.nlags else np.zeros((m.n, m.n))
            inf = np.zeros((m.n, m.n))
            for k in range(m.n):
                Om = np.zeros_like(C)
                Om[:m.n, :m.n] = np.outer(R[:, k], R[:, k])
                inf[:, k] = np.diag(linalg.solve_discrete_lyapunov(C, Om))[:m.n]
            blocks[np.inf] = inf
        rows, keys = [], []
        for i, v in enumerate(m.endog):
            for k, blk in blocks.items():
                rows.append(blk[i] / blk[i].sum())
                keys.append((v, k))
        return pd.DataFrame(rows, columns=self._shock_names(ident),
                            index=pd.MultiIndex.from_tuples(keys, names=["variable", "horizon"]))

    def structural_shocks(self, ident=None, regime: int = 1) -> pd.DataFrame:
        """Structural shocks ``e_t = R^{-1} u_t`` (unit variance)."""
        R = self.impact(ident, regime)
        E = np.linalg.solve(R, self.residuals(regime).to_numpy().T).T
        return pd.DataFrame(E, index=self.data.index, columns=self._shock_names(ident))

    def historical_decomposition(self, ident=None, regime: int = 1) -> dict:
        """Historical decomposition: ``{variable: DataFrame}`` with the columns ``y0`` (initial
        conditions), one per deterministic term and one per shock; the columns sum to the observed
        series.

        With regime switching it uses the parameters of the given regime (the residuals are those
        of that regime).
        """
        m = self.model
        n, p, nd = m.n, m.nlags, m.nd
        B = self.Bs[regime - 1]
        R = self.impact(ident, regime)
        E = self.structural_shocks(R, regime).to_numpy()
        T = self.data.Y.shape[0]
        ncomp = 1 + nd + n
        comp = np.zeros((T, n, ncomp))
        state = np.zeros((n * max(p, 1), ncomp))
        if p:
            state[:n * p, 0] = self.data.X[0, nd:]
        for t in range(T):
            y = B[:, nd:] @ state[:n * p] if p else np.zeros((n, ncomp))
            y[:, 1:1 + nd] += B[:, :nd] * self.data.X[t, :nd]
            y[:, 1 + nd:] += R * E[t]
            comp[t] = y
            if p:
                state = np.vstack([y, state[:n * (p - 1)]])
        cols = ["y0"] + m.det_names + self._shock_names(ident)
        return {v: pd.DataFrame(comp[:, i, :], index=self.data.index, columns=cols) for i, v in enumerate(m.endog)}

    # ---- forecast
    def _future_det(self, steps, start_pos, exog_future):
        m = self.model
        cols = []
        if m.constant:
            cols.append(np.ones(steps))
        if m.trend:
            cols.append(np.arange(start_pos + 1, start_pos + steps + 1, dtype=float))
        if m.exog:
            if exog_future is None:
                raise ValueError("pass 'exog_future' (DataFrame with the exogenous variables over the forecast horizon)")
            ex = np.asarray(exog_future[m.exog] if isinstance(exog_future, pd.DataFrame) else exog_future, float)
            ex = ex.reshape(-1, len(m.exog))
            if len(ex) < steps:
                raise ValueError("'exog_future' is shorter than the horizon")
            cols += [ex[:steps, k] for k in range(len(m.exog))]
        return np.column_stack(cols) if cols else np.zeros((steps, 0))

    def forecast(self, steps: int = 12, start=None, conditions: pd.DataFrame | None = None, ident=None,
                 shock_uncertainty: bool = False, draws: int = 1000, seed: int = 0, exog_future=None,
                 regime: int | None = None, params=None):
        """Forecast, unconditional or conditional.

        - ``start``: label of the first forecast date (default: right after the end of the sample).
          As in RISE, when ``start`` falls inside the sample the forecast starts from the preceding
          observations.
        - ``conditions``: DataFrame (rows = steps 1.., columns = conditioned endogenous variables,
          NaN = free). The conditions are hard and met by the combination of shocks with the
          smallest norm (Waggoner and Zha, 1999); the result does not depend on the rotation.
        - ``shock_uncertainty=True`` returns the simulations (draws, steps, n) instead of the mean.
        - ``params``: list of (B, Sigma) (for example, posterior draws); returns an array with
          one forecast per element.
        - With regime switching, the forecast is the mean of the regimes weighted by the
          probabilities predicted from the last updated probability (or fixed at ``regime``).
        """
        m = self.model
        n, p, nd = m.n, m.nlags, m.nd
        full = self.data.full_index
        if start is None:
            pos = len(full)
        else:
            pos = full.get_loc(start) if start in full else len(full)
        if pos < p:
            raise ValueError("observations before 'start' are missing for the initial conditions")
        hist = self.data.levels[pos - p:pos] if p else np.zeros((0, n))
        if pos < len(full) and m.exog and exog_future is None:
            det = self.data.det[pos:pos + steps]
            if len(det) < steps:
                raise ValueError("pass 'exog_future' for the stretch beyond the sample")
        else:
            det = self._future_det(steps, pos, exog_future)
        index = pd.RangeIndex(1, steps + 1, name="step")
        if pos + steps <= len(full):
            index = full[pos:pos + steps]
        if conditions is not None and isinstance(conditions, pd.DataFrame) is False:
            raise ValueError("'conditions' must be a DataFrame")

        def run(B, Sigma, R, rng):
            lags = B[:, nd:]
            x0 = hist[::-1].reshape(-1) if p else np.zeros(0)
            Th = ma_coefficients(lags, n, p, steps)

            def path(E):
                out = np.zeros((steps, n))
                state = x0.copy()
                for t in range(steps):
                    y = B[:, :nd] @ det[t] + (lags @ state if p else 0) + R @ E[t]
                    out[t] = y
                    if p:
                        state = np.r_[y, state[:n * (p - 1)]]
                return out
            base = path(np.zeros((steps, n)))
            if conditions is None:
                if not shock_uncertainty:
                    return base
                return np.stack([path(rng.standard_normal((steps, n))) for _ in range(draws)])
            cond = conditions.reindex(columns=m.endog).to_numpy(float)
            cond = np.vstack([cond, np.full((max(steps - len(cond), 0), n), np.nan)])[:steps]
            sel = np.argwhere(np.isfinite(cond))
            # stacked y = base + M eps, with M built from the impulse responses
            M = np.zeros((len(sel), steps * n))
            for q, (t, i) in enumerate(sel):
                for s in range(t + 1):
                    M[q, s * n:(s + 1) * n] = (Th[t - s] @ R)[i]
            gap = cond[sel[:, 0], sel[:, 1]] - base[sel[:, 0], sel[:, 1]]
            Mp = np.linalg.pinv(M)
            mean = Mp @ gap
            if not shock_uncertainty:
                return path(mean.reshape(steps, n))
            proj = np.eye(steps * n) - Mp @ M
            return np.stack([path((mean + proj @ rng.standard_normal(steps * n)).reshape(steps, n)) for _ in range(draws)])

        rng = np.random.default_rng(seed)
        if params is not None:
            outs = [run(B, S, np.linalg.cholesky(S) if not isinstance(ident, Identification) else ident(B, S, rng), rng)
                    for B, S in params]
            return np.stack(outs)
        if self.h == 1 or regime is not None:
            r = regime or 1
            out = run(self.Bs[r - 1], self.Sigmas[r - 1], self.impact(ident, r, rng), rng)
        else:
            if conditions is not None or shock_uncertainty:
                raise ValueError("with regime switching, a conditional forecast or one with draws requires 'regime'")
            mu = self.updated_probs.to_numpy()[min(pos - p, len(self.updated_probs)) - 1]
            lagsr = [self.Bs[r][:, nd:] for r in range(self.h)]
            state = hist[::-1].reshape(-1) if p else np.zeros(0)
            out = np.zeros((steps, n))
            for t in range(steps):
                mu = mu @ self.Q
                y = sum(mu[r] * (self.Bs[r][:, :nd] @ det[t] + (lagsr[r] @ state if p else 0)) for r in range(self.h))
                out[t] = y
                if p:
                    state = np.r_[y, state[:n * (p - 1)]]
        if out.ndim == 2:
            return pd.DataFrame(out, index=index, columns=m.endog)
        return out

    # ---- parameter uncertainty
    def bootstrap(self, draws: int = 1000, seed: int = 0) -> np.ndarray:
        """Residual bootstrap (resampling with replacement, fixed initial conditions).

        Returns an array (parameters, draws) with the OLS re-estimates, in the order of ``names``.
        """
        if self.h > 1 or self.model.structural:
            raise ValueError("bootstrap is available for the reduced-form VAR without regime switching")
        m, rng = self.model, np.random.default_rng(seed)
        n, p, nd = m.n, m.nlags, m.nd
        U = self.residuals().to_numpy()
        T = U.shape[0]
        out = np.zeros((self.theta.size, draws))
        for b in range(draws):
            Ub = U[rng.integers(0, T, T)]
            Y = np.zeros((T, n))
            X = self.data.X.copy()
            state = X[0, nd:].copy()
            for t in range(T):
                X[t, nd:] = state
                Y[t] = self.B @ X[t] + Ub[t]
                if p:
                    state = np.r_[Y[t], state[:n * (p - 1)]]
            Bb = np.linalg.solve(X.T @ X, X.T @ Y).T
            E = Y - X @ Bb.T
            out[:, b] = np.r_[Bb.reshape(-1, order="F"), vech(E.T @ E / (T - m.K))]
        return out

    def sample_posterior(self, draws: int = 1000, seed: int = 0) -> list:
        """Draws (B, Sigma) from the posterior of the constant-parameter BVAR."""
        post = self.posterior
        if post is None:
            raise ValueError("the model was not estimated with a conjugate prior; use MCMC on the objective")
        rng = np.random.default_rng(seed)
        n, K = self.model.n, self.model.K
        out = []
        if post["kind"] == "draws":
            pick = rng.integers(0, len(post["B"]), draws)
            return [(post["B"][k], post["Sigma"][k]) for k in pick]
        if post["kind"] == "normal":
            L = np.linalg.cholesky(post["Omega"])
            for _ in range(draws):
                out.append(((post["b"] + L @ rng.standard_normal(n * K)).reshape(n, K, order="F"), post["Sigma"]))
            return out
        if post.get("restricted"):
            raise ValueError("conjugate sampler unavailable with linear restrictions")
        Lp = np.linalg.cholesky(post["Phi"])
        for _ in range(draws):
            Sig = _draw_inverse_wishart(post["S"], post["nu"], rng)
            Z = rng.standard_normal((n, K))
            out.append((post["B"] + np.linalg.cholesky(Sig) @ Z @ Lp.T, Sig))
        return out


# ================================================================ regime switching: filter
def hamilton_filter(logf: np.ndarray, Q: np.ndarray, pi0: np.ndarray | None = None, smooth: bool = True):
    """Hamilton filter and Kim smoother.

    ``logf[t, r]`` is the log density of ``y_t`` in regime r. Returns a dictionary with ``loglik``,
    ``loglik_t``, ``filtered`` (Pr(s_t | t-1)), ``updated`` (Pr(s_t | t)), ``smoothed`` and, if
    ``smooth``, ``joint`` (sum over t of Pr(s_t=i, s_{t+1}=j | T)). ``pi0`` is the distribution of
    s_1 before seeing y_1: ``'uniform'`` (default; it is what RISE uses in VARs), ``'ergodic'`` or
    a vector.
    """
    T, h = logf.shape
    if pi0 is None or (isinstance(pi0, str) and pi0 == "uniform"):
        pred = np.full(h, 1.0 / h)
    elif isinstance(pi0, str):
        pred = ergodic(Q)
    else:
        pred = np.asarray(pi0, float)
    filt, upd, ll = np.zeros((T, h)), np.zeros((T, h)), np.zeros(T)
    for t in range(T):
        filt[t] = pred
        mx = logf[t].max()
        w = pred * np.exp(logf[t] - mx)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return dict(loglik=-np.inf, loglik_t=ll, filtered=filt, updated=upd, smoothed=None, joint=None)
        ll[t] = np.log(s) + mx
        upd[t] = w / s
        pred = upd[t] @ Q
    out = dict(loglik=float(ll.sum()), loglik_t=ll, filtered=filt, updated=upd, smoothed=None, joint=None)
    if smooth:
        sm = np.zeros((T, h))
        sm[-1] = upd[-1]
        joint = np.zeros((h, h))
        for t in range(T - 2, -1, -1):
            ratio = np.divide(sm[t + 1], filt[t + 1], out=np.zeros(h), where=filt[t + 1] > 0)
            J = upd[t][:, None] * Q * ratio[None, :]
            joint += J
            sm[t] = J.sum(1)
        out["smoothed"], out["joint"] = sm, joint
    return out


def _regime_logdens(Y, X, A0, Bx, S, structural: bool) -> np.ndarray:
    T, n = Y.shape
    h = Bx.shape[0]
    out = np.zeros((T, h))
    for r in range(h):
        if structural:
            if np.any(S[r] <= 0):
                return np.full((T, h), -np.inf)
            sign, ld = np.linalg.slogdet(A0[r])
            if sign == 0:
                return np.full((T, h), -np.inf)
            E = (Y @ A0[r].T - X @ Bx[r].T) / S[r]
            out[:, r] = ld - np.log(S[r]).sum() - 0.5 * (E ** 2).sum(1) - 0.5 * n * np.log(2 * np.pi)
        else:
            try:
                L = np.linalg.cholesky(S[r])
            except np.linalg.LinAlgError:
                return np.full((T, h), -np.inf)
            E = linalg.solve_triangular(L, (Y - X @ Bx[r].T).T, lower=True).T
            out[:, r] = -np.log(np.diag(L)).sum() - 0.5 * (E ** 2).sum(1) - 0.5 * n * np.log(2 * np.pi)
    return out


def _reduced_form(A0, Bx, S, structural: bool):
    if not structural:
        return Bx.copy(), S.copy()
    Bs = np.stack([np.linalg.solve(A0[r], Bx[r]) for r in range(A0.shape[0])])
    imp = [np.linalg.solve(A0[r], np.diag(S[r])) for r in range(A0.shape[0])]
    return Bs, np.stack([M @ M.T for M in imp])


# ================================================================ priors of the transition probabilities
def _beta_logpdf(x, mean, std):
    k = mean * (1 - mean) / std ** 2 - 1
    if k <= 0:
        raise ValueError(f"infeasible beta prior with mean {mean} and standard deviation {std}")
    return stats.beta.logpdf(x, mean * k, (1 - mean) * k)


def dirichlet_hyperparameters(std_first: float, means) -> np.ndarray:
    """Dirichlet hyperparameters for one row of the transition matrix, in the RISE convention:
    ``{std, name_1, mean_1, name_2, mean_2, ...}``. The given means are those of the off-diagonal
    elements (the diagonal one is the complement) and ``std`` is the standard deviation of the
    first of them. Returns the alphas in the order (off-diagonal..., diagonal)."""
    m = np.r_[np.asarray(means, float), 1 - np.sum(means)]
    if np.any(m <= 0):
        raise ValueError("Dirichlet means must be positive and sum to less than one")
    a0 = m[0] * (1 - m[0]) / std_first ** 2 - 1
    if a0 <= 0:
        raise ValueError("standard deviation too large for the Dirichlet")
    return a0 * m


class TransitionPriors:
    """Priors of the transition probabilities, in the RISE format.

    ``spec`` is a dictionary: ``{'chain_tp_i_j': (mean, std)}`` for beta priors and
    ``{'dirichlet_k': (std, 'chain_tp_i_j', mean, 'chain_tp_i_l', mean, ...)}`` for a
    Dirichlet over one row. It also accepts the RISE cells ``(start, mean, std, 'beta')``.
    """

    def __init__(self, spec: dict | None, layout: _Layout):
        self.terms = []
        self.start = {}
        for key, val in (spec or {}).items():
            val = tuple(val)
            if key.lower().startswith("dirichlet"):
                names, means = list(val[1::2]), [float(v) for v in val[2::2]]
                alpha = dirichlet_hyperparameters(float(val[0]), means)
                self.terms.append(("dirichlet", [layout.names.index(nm) for nm in names], alpha))
                for nm, mu in zip(names, means):
                    self.start[layout.names.index(nm)] = mu
            else:
                if key not in layout.names:
                    raise ValueError(f"unknown transition probability: {key!r}")
                if len(val) >= 4 or (len(val) == 3 and not isinstance(val[2], str)):
                    start, mean, std = float(val[0]), float(val[1]), float(val[2])
                else:
                    mean, std = float(val[0]), float(val[1])
                    start = mean
                k = layout.names.index(key)
                self.terms.append(("beta", k, mean, std))
                self.start[k] = start

    def logpdf(self, theta) -> float:
        lp = 0.0
        for term in self.terms:
            if term[0] == "beta":
                x = theta[term[1]]
                if not 0 < x < 1:
                    return -np.inf
                lp += _beta_logpdf(x, term[2], term[3])
            else:
                x = theta[term[1]]
                x = np.r_[x, 1 - x.sum()]
                if np.any(x <= 0):
                    return -np.inf
                a = term[2]
                lp += special.gammaln(a.sum()) - special.gammaln(a).sum() + ((a - 1) * np.log(x)).sum()
        return float(lp)


# ================================================================ regime switching: objective and estimation
def _rref_parametrization(R: np.ndarray, r: np.ndarray, size: int):
    """``R theta = r``  ->  ``theta = d + K x`` where x are the free parameters themselves.

    Returns (K, d, free): ``free`` are the positions of theta that make up x.
    """
    if R.size == 0:
        return np.eye(size), np.zeros(size), np.arange(size)
    A = np.column_stack([R, r]).astype(float)
    rows, pivots = 0, []
    for col in range(size - 1, -1, -1):          # eliminate from back to front: the first names stay free
        if rows == A.shape[0]:
            break
        k = rows + int(np.argmax(np.abs(A[rows:, col])))
        if abs(A[k, col]) < 1e-12:
            continue
        A[[rows, k]] = A[[k, rows]]
        A[rows] /= A[rows, col]
        for i in range(A.shape[0]):
            if i != rows:
                A[i] -= A[i, col] * A[rows]
        pivots.append(col)
        rows += 1
    if np.any(np.abs(A[rows:, -1]) > 1e-9):
        raise ValueError("inconsistent linear restrictions")
    free = np.array([j for j in range(size) if j not in pivots], dtype=int)
    K, d = np.zeros((size, free.size)), np.zeros(size)
    K[free, np.arange(free.size)] = 1.0
    for i, col in enumerate(pivots):
        d[col] = A[i, -1]
        K[col] = -A[i, free]
    return K, d, free


class _SwitchingObjective:
    """Likelihood (Hamilton filter) and prior of a regime-switching or structural VAR."""

    def __init__(self, model: VAR, data: VARData, prior: VARPrior | None, transition_priors, regimes=None):
        self.model, self.data, self.lay = model, data, model.layout
        self.prior = prior
        self.tp = TransitionPriors(transition_priors, self.lay)
        self.regimes = None if regimes is None else np.asarray(regimes, int) - 1
        if self.regimes is not None and len(self.regimes) != data.Y.shape[0]:
            raise ValueError("'regimes' must have one entry per effective observation (sample minus the lags)")
        self.dummies = None
        if prior is not None:
            if prior.type != "sz":
                raise ValueError("in estimation by optimisation (regime switching or SVAR) only the 'sz' prior is available")
            self.dummies = sims_zha_dummies(data.levels, model.nd, model.nlags, prior)

    def matrices(self, theta):
        return self.lay.unpack(theta)

    def feasible(self, theta) -> bool:
        lay = self.lay
        for P in lay.chain_matrices(theta):
            if np.any(P < -1e-12) or np.any(P > 1 + 1e-12):
                return False
        return True

    def filter(self, theta, smooth: bool = False):
        A0, Bx, S, Q = self.matrices(theta)
        logf = _regime_logdens(self.data.Y, self.data.X, A0, Bx, S, self.model.structural)
        if not np.all(np.isfinite(logf)):
            return dict(loglik=-np.inf)
        if self.regimes is not None:
            T = len(self.regimes)
            ll = logf[np.arange(T), self.regimes]
            with np.errstate(divide="ignore"):
                trans = np.log(Q[self.regimes[:-1], self.regimes[1:]]) if Q.shape[0] > 1 else np.zeros(T - 1)
            probs = np.eye(Q.shape[0])[self.regimes]
            return dict(loglik=float(ll.sum() + trans.sum()), loglik_t=ll, filtered=probs, updated=probs, smoothed=probs, joint=None)
        return hamilton_filter(logf, np.clip(Q, 0, 1), smooth=smooth)

    def loglik(self, theta) -> float:
        if not self.feasible(theta):
            return -np.inf
        return self.filter(theta)["loglik"]

    def logpost(self, theta) -> float:
        """Log posterior. As RISE does (identified numerically, see docs/VALIDATION.md), the ``sz`` prior
        enters as dummy observations appended to the end of the sample and filtered together with
        it; without regime switching this is the Gaussian likelihood of the dummy observations."""
        if not self.feasible(theta):
            return -np.inf
        lp = self.tp.logpdf(theta)
        if not np.isfinite(lp):
            return -np.inf
        if self.dummies is None:
            ll = self.filter(theta)["loglik"]
        else:
            A0, Bx, S, Q = self.matrices(theta)
            st = self.model.structural
            lf = _regime_logdens(self.data.Y, self.data.X, A0, Bx, S, st)
            ld = _regime_logdens(self.dummies[0], self.dummies[1], A0, Bx, S, st)
            if not (np.all(np.isfinite(lf)) and np.all(np.isfinite(ld))):
                return -np.inf
            if self.regimes is not None:
                mx = ld.max(1)
                ll = self.filter(theta)["loglik"] + float((np.log(np.exp(ld - mx[:, None]).mean(1)) + mx).sum())
            else:
                ll = hamilton_filter(np.vstack([lf, ld]), np.clip(Q, 0, 1), smooth=False)["loglik"]
        return ll + lp if np.isfinite(ll) else -np.inf

    def logprior(self, theta) -> float:
        """Difference between the posterior and the likelihood of the data."""
        lp, ll = self.logpost(theta), self.loglik(theta)
        return lp - ll if np.isfinite(lp) and np.isfinite(ll) else -np.inf


def _default_start(model: VAR, d: VARData, tp: TransitionPriors, rng, jitter: float) -> np.ndarray:
    """Starting point: constant-parameter OLS, with the variances of each state scaled and the
    switching coefficients slightly perturbed (to leave the symmetric saddle point)."""
    lay, n = model.layout, model.n
    B = np.linalg.solve(d.X.T @ d.X, d.X.T @ d.Y).T
    U = d.Y - d.X @ B.T
    Sig = U.T @ U / len(U)
    h = lay.h
    A0 = np.tile(np.eye(n), (h, 1, 1))
    Bx = np.tile(B, (h, 1, 1))
    S = np.tile(np.sqrt(np.diag(Sig)), (h, 1)) if model.structural else np.tile(Sig, (h, 1, 1))
    theta = lay.pack(A0, Bx, S)
    for (blk, eq, col), ci in lay._chain_of.items():
        if ci is None:
            continue
        ns = lay.chains[ci].states
        for st in range(ns):
            k = lay._index[(blk, eq, col, ci, st)]
            if blk == "s":
                f = 2.0 ** ((st - (ns - 1) / 2) * 1.2) * np.exp(jitter * rng.standard_normal())
                theta[k] *= f if model.structural else f ** 2
            elif blk != "a0" or eq != col:
                theta[k] += (0.03 + jitter) * (abs(theta[k]) + 0.1) * rng.standard_normal()
    for (ci, i, j), k in lay.tp_index.items():
        theta[k] = tp.start.get(k, 0.1 / (lay.chains[ci].states - 1) * 2)
    return theta


def _em(model: VAR, d: VARData, obj: _SwitchingObjective, theta0, R, r, maxiter: int, tol: float):
    """EM (Hamilton, 1990) for the reduced-form VAR with regime switching.

    M step in two conditional stages: coefficients by GLS with the current covariances (which
    accommodates coefficients common to regimes and linear equality restrictions) and covariances
    from the weighted residuals; the transition probabilities of each chain come from the smoothed
    transitions.
    """
    lay, n, K, h = model.layout, model.n, model.K, model.h
    Y, X = d.Y, d.X
    coef_keys = [key for key in lay._chain_of if key[0] != "s"]
    coef_pos = sorted({k for key in coef_keys for k in lay.positions_all_states(*key)})
    where = {k: i for i, k in enumerate(coef_pos)}
    if R.size and np.any(np.abs(np.delete(R, coef_pos, axis=1)) > 0):
        raise ValueError("in EM the linear restrictions apply only to the coefficients; use method='optimize'")
    Kc, dc, _ = _rref_parametrization(R[:, coef_pos] if R.size else R.reshape(0, len(coef_pos)), r, len(coef_pos))
    M = np.zeros((h * n * K, len(coef_pos)))
    for rr in range(h):
        for (blk, eq, col) in coef_keys:
            c = col if blk == "c" else model.nd + (int(blk[1:]) - 1) * n + col
            M[rr * n * K + c * n + eq, where[lay.position(blk, eq, col, rr)]] = 1.0
    MK, Md = M @ Kc, M @ dc
    s_chain = lay._chain_of[("s", 0, 0)]
    theta = theta0.copy()
    best, best_ll, prev = theta.copy(), -np.inf, -np.inf
    it = 0
    for it in range(1, maxiter + 1):
        A0, Bx, S, Q = lay.unpack(theta)
        f = hamilton_filter(_regime_logdens(Y, X, A0, Bx, S, False), Q, smooth=True)
        ll = f["loglik"]
        if not np.isfinite(ll):
            break
        if ll > best_ll:
            best, best_ll = theta.copy(), ll
        if abs(ll - prev) < tol * (1 + abs(ll)):
            break
        prev = ll
        xi = f["smoothed"]
        blocks = [((X * xi[:, [rr]]).T @ X, (Y * xi[:, [rr]]).T @ X, np.linalg.inv(S[rr])) for rr in range(h)]
        vecB = _restricted_ls_stacked(blocks, MK, Md)
        Bn = vecB.reshape(h, K, n).transpose(0, 2, 1)
        Sn = np.zeros_like(S)
        groups = [list(range(h))] if s_chain is None else \
            [[rr for rr in range(h) if lay.regime_states[rr, s_chain] == g] for g in range(lay.chains[s_chain].states)]
        for g in groups:
            acc, w = np.zeros((n, n)), 0.0
            for rr in g:
                U = Y - X @ Bn[rr].T
                acc += (U * xi[:, [rr]]).T @ U
                w += xi[:, rr].sum()
            for rr in g:
                Sn[rr] = acc / max(w, 1e-300)
        Ps = []
        for ci, c in enumerate(lay.chains):
            N = np.zeros((c.states, c.states))
            st = lay.regime_states[:, ci]
            np.add.at(N, (st[:, None].repeat(h, 1), st[None, :].repeat(h, 0)), f["joint"])
            Ps.append(N / np.maximum(N.sum(1, keepdims=True), 1e-300))
        theta = lay.pack(A0, Bn, Sn, Ps)
    return best, dict(message=f"EM: {it} iterations", iterations=it)


def _restricted_ls_stacked(blocks, MK, Md):
    """Like ``_restricted_ls``, but with a joint map ``vec(B_1..B_h) = Md + MK x``."""
    n, K = blocks[0][1].shape
    size = n * K
    H = np.zeros((len(blocks) * size,) * 2)
    g = np.zeros(len(blocks) * size)
    for b, (Sxx, Syx, W) in enumerate(blocks):
        sl = slice(b * size, (b + 1) * size)
        H[sl, sl] = np.kron(Sxx, W)
        g[sl] = (W @ Syx).reshape(-1, order="F")
    x = np.linalg.lstsq(MK.T @ H @ MK, MK.T @ (g - H @ Md), rcond=None)[0]
    return Md + MK @ x


def _optimize(model: VAR, obj: _SwitchingObjective, theta0, R, r, ineq, maxiter: int, tol: float):
    lay = model.layout
    Kmat, dvec, free = _rref_parametrization(R, r, lay.size)
    x0 = np.linalg.lstsq(Kmat, theta0 - dvec, rcond=None)[0]
    lower, upper = np.full(lay.size, -np.inf), np.full(lay.size, np.inf)
    for (blk, eq, col), ci in lay._chain_of.items():
        if blk == "s" and (model.structural or eq == col):
            for k in lay.positions_all_states(blk, eq, col):
                lower[k] = 1e-8
    for k in lay.tp_index.values():
        lower[k], upper[k] = 1e-9, 1 - 1e-9
    G, g = [], []                     # G theta + g >= 0
    for coef, const in ineq:
        G.append(coef), g.append(const)
    for ci, c in enumerate(lay.chains):
        if c.states > 2:
            for i in range(c.states):
                row = np.zeros(lay.size)
                for j in range(c.states):
                    if j != i:
                        row[lay.tp_index[(ci, i, j)]] = -1.0
                G.append(row), g.append(1.0)
    pivots = np.setdiff1d(np.arange(lay.size), free)
    for k in pivots:                  # bounds of dependent parameters become linear constraints
        e = np.zeros(lay.size)
        e[k] = 1.0
        if np.isfinite(lower[k]):
            G.append(e), g.append(-lower[k])
        if np.isfinite(upper[k]):
            G.append(-e), g.append(upper[k])
    bounds = [(lower[k] if np.isfinite(lower[k]) else None, upper[k] if np.isfinite(upper[k]) else None) for k in free]
    cons = []
    if G:
        Gx, gx = np.array(G) @ Kmat, np.array(G) @ dvec + np.array(g)
        cons = [dict(type="ineq", fun=lambda x: Gx @ x + gx, jac=lambda x: Gx)]

    def fun(x):
        v = obj.logpost(dvec + Kmat @ x)
        return -v if np.isfinite(v) else 1e10
    x0 = np.clip(x0, [b[0] if b[0] is not None else -np.inf for b in bounds], [b[1] if b[1] is not None else np.inf for b in bounds])
    if fun(x0) >= 1e10:
        raise ValueError("infeasible starting point (likelihood or prior not finite); pass 'x0'")
    best = None
    x = x0
    for _ in range(4):                # restarts: SLSQP stops early on flat surfaces
        res = optimize.minimize(fun, x, method="SLSQP", bounds=bounds, constraints=cons,
                                options=dict(maxiter=maxiter, ftol=tol * 1e-2))
        if best is not None and res.fun > best.fun - 1e-7:
            if res.fun < best.fun:
                best = res
            break
        best, x = res, res.x
    return dvec + Kmat @ best.x, dict(message=str(best.message), iterations=int(best.nit), x=best.x, K=Kmat, d=dvec, fun=fun)


def _fit_switching(model: VAR, d: VARData, prior, restrictions, method, transition_priors=None, regimes=None,
                   x0=None, starts: int = 1, seed: int = 0, maxiter: int = 500, tol: float = 1e-8) -> VARResults:
    lay = model.layout
    R, r, ineq = model._split_restrictions(restrictions)
    Ra, ra = model._automatic_equalities()
    R, r = np.vstack([Ra, R]), np.r_[ra, r]
    obj = _SwitchingObjective(model, d, prior, transition_priors, regimes)
    if method is None:
        plain = not model.structural and model.is_switching and prior is None and not ineq and not transition_priors and regimes is None
        method = "em" if plain else "optimize"
    if method not in ("em", "optimize"):
        raise ValueError("method: 'em' or 'optimize'")
    if method == "em" and (model.structural or not model.is_switching or prior is not None or ineq or regimes is not None):
        raise ValueError("EM applies to the reduced-form VAR with regime switching, without prior and without inequalities")
    rng = np.random.default_rng(seed)
    best = None
    for k in range(max(starts, 1)):
        if x0 is not None and k == 0:
            th0 = np.array([x0[nm] for nm in lay.names], float) if isinstance(x0, (dict, pd.Series)) else np.asarray(x0, float)
        else:
            th0 = _default_start(model, d, obj.tp, rng, jitter=0.0 if k == 0 else 0.3)
        try:
            theta, info = (_em(model, d, obj, th0, R, r, maxiter, tol) if method == "em"
                           else _optimize(model, obj, th0, R, r, ineq, maxiter, tol))
        except ValueError:
            if starts <= 1 or (k == starts - 1 and best is None):
                raise
            continue
        val = obj.logpost(theta)
        if best is None or val > best[0]:
            best = (val, theta, info)
    _, theta, info = best
    return _switching_results(model, d, obj, theta, method, info)


def _switching_results(model: VAR, d: VARData, obj: _SwitchingObjective, theta, method: str = "", info: dict | None = None) -> VARResults:
    lay = model.layout
    info = info or {}
    A0, Bx, S, Q = lay.unpack(theta)
    f = obj.filter(theta, smooth=True)
    Bs, Sig = _reduced_form(A0, Bx, S, model.structural)
    cols = [f"regime_{k + 1}" for k in range(lay.h)]
    frame = lambda a: pd.DataFrame(a, index=d.index, columns=cols) if a is not None else None
    states = {}
    if f.get("smoothed") is not None:
        for ci, c in enumerate(lay.chains):
            st = lay.regime_states[:, ci]
            states[c.name] = pd.DataFrame({f"{c.name}_{g + 1}": f["smoothed"][:, st == g].sum(1) for g in range(c.states)}, index=d.index)
    return VARResults(model=model, data=d, theta=np.asarray(theta, float), A0=A0, Bs=Bs, Sigmas=Sig, Q=Q,
                      loglik=f["loglik"], prior=obj.prior, method=method, logprior=obj.logprior(theta),
                      S0=S if model.structural else None, filtered_probs=frame(f.get("filtered")),
                      updated_probs=frame(f.get("updated")), smoothed_probs=frame(f.get("smoothed")),
                      smoothed_state_probs=states, loglik_t=f.get("loglik_t"), message=info.get("message", ""),
                      iterations=info.get("iterations", 0), free=info.get("x"),
                      restriction_map=(info["K"], info["d"]) if "K" in info else None, objective=info.get("fun"))


def evaluate(model: VAR, data: pd.DataFrame, theta, sample=None, prior=None, transition_priors=None, regimes=None) -> VARResults:
    """Evaluate likelihood, prior and regime probabilities at a given parameter vector
    (``theta`` in the order of ``model.parameter_names``, or a dictionary/Series by name)."""
    d = model.make_data(data, sample)
    if isinstance(prior, str):
        prior = VARPrior(type=prior)
    if isinstance(theta, (dict, pd.Series)):
        theta = np.array([theta[nm] for nm in model.layout.names], float)
    obj = _SwitchingObjective(model, d, prior, transition_priors, regimes)
    return _switching_results(model, d, obj, np.asarray(theta, float), "evaluate")


class SVAR(VAR):
    """Structural VAR (RISE's ``svar``): ``A0 y_t = C d_t + sum_l A_l y_{t-l} + diag(s) e_t``.

    ``diag(A0) = 1`` is imposed automatically; the parameters are called ``c_i_k``, ``a0_i_j``,
    ``a1_i_j``, ... and ``s_i_i`` (i = equation, j = variable). Identification comes from the
    linear restrictions (``'a0(2,FFR)=0'``, ``'a0(3,pi)+a0(3,FFR)=0'``, ``'a1(1,pi,chain,2)=0'``)
    and the inequalities (``'a0(3,FFR)>=0'``) are respected in the optimisation. Estimation is
    always by direct optimisation: maximum likelihood or, with ``prior='sz'``, the posterior mode.
    """
    structural = True

    def _automatic_equalities(self):
        lay = self.layout
        rows = []
        for i in range(self.n):
            for k in lay.positions_all_states("a0", i, i):
                e = np.zeros(lay.size)
                e[k] = 1.0
                rows.append(e)
        return np.array(rows), np.ones(len(rows))
