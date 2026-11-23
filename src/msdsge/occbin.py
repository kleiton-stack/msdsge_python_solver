"""Occasionally binding constraints in 1st-order models, through RISE's regime mechanism.

Each constraint has a two-state chain: in the reference regime the constraint is slack
(``g >= 0`` holds) and in the alternative regime an indicator parameter switches the model equation
(for example ``R = bind*r_zlb + (1-bind)*R_TAYLOR``). The transition probabilities of the chains are
zero: agents do not expect the switch. This is the piecewise-linear solution of Guerrieri and
Iacoviello (2015), which RISE obtains with ``solve_occbin`` and simulates with
``simul_honor_constraints``; Binning and Maih (2017, section 8) use it as a benchmark for the
version with endogenous probabilities.

Syntax in the model file::

    markov_chain ocb(regimes=2);
    parameters(ocb) bind;
    bind(ocb,1) = 0;  bind(ocb,2) = 1;
    model;  R = bind*r_zlb + (1-bind)*R_TAYLOR;  ...  end;
    occbin_constraints;
      ocb: R >= 1;          // "chain:" is optional; without it, the k-th constraint uses the k-th chain
    end;

Usage::

    occ = solve_occbin(model)              # reference regime = 1 (like solve_occbin=1 in RISE)
    occ.simulate(periods=300, seed=1)      # honours the constraints
    occ.irf("ER", size=-5)                 # response with the constraint, in deviation from the path without the shock

Algorithm (per period t, given ``y_{t-1}`` and the shock ``e_t``; zero future shocks):

1. guess the sequence of regimes ``t, t+1, ...`` (all reference in the first pass);
2. given the sequence, compute the time-varying decision rules backwards, starting from the
   solution of the reference regime ``T``:
   ``P_s = -(A0_r + Ap_r P_{s+1})^{-1} Am_r``, ``c_s = -(A0_r + Ap_r P_{s+1})^{-1} (Ap_r c_{s+1} + k_r)``,
   where ``r`` is the regime of period ``s`` and ``k_r`` is the residual of the equations of regime
   ``r`` at the reference steady state;
3. simulate the path and check the constraints over the horizon; if the implied sequence of
   regimes differs from the guess, update it and go back to 2.

Two variants, chosen by ``OccbinSolution.lookahead``:

- ``False`` (default): RISE behaviour, identified by numerical comparison. If the reference step
  honours the constraints at ``t``, it is accepted without looking at later periods; otherwise,
  each constraint binds for a contiguous block of periods starting at ``t``.
- ``True``: original OccBin (Guerrieri and Iacoviello, 2015), which also reacts to violations
  expected in future periods and admits non-contiguous blocks. Matches Dynare.

The matrices are the derivatives at the steady state of the reference regime, imposed on all
regimes (RISE's ``steady_state_imposed`` and ``steady_state_unique``).

The module also provides :func:`simulate_through_switch`, for models with actual regime switching
(positive probabilities, as in ``globdynzlb.rs``), where the constraint is honoured through the
choice of the regime in each period (RISE's ``simul_honor_constraints_through_switch``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sympy as sp

from .model import FirstOrderSystem, Model
from .parser import _SS, timed
from .solve import SolveError, _mfi, _mnk, initial_guess


class OccbinError(RuntimeError):
    pass


@dataclass
class Constraint:
    """Constraint ``g >= 0`` associated with a two-state chain."""
    chain: str
    expr: sp.Expr
    text: str = ""
    relax: sp.Expr | None = None   # exit condition of the alternative regime (optional), ``relax >= 0``


def _compile(model: Model, expr: sp.Expr):
    """Function ``g(y_t, y_{t-1}, ss, p)`` of an expression in current and lagged variables."""
    y0 = [timed(v) for v in model.endogenous]
    ym = [timed(v, -1) for v in model.endogenous]
    p = [timed(v) for v in model.parameters]
    ss = [sp.Symbol(f"{v}__SS", real=True) for v in model.endogenous]
    expr = sp.sympify(expr).xreplace({_SS(timed(v)): s for v, s in zip(model.endogenous, ss)})
    extra = expr.free_symbols - set(y0) - set(ym) - set(p) - set(ss)
    if extra:
        raise OccbinError("a constraint may only use current or once-lagged endogenous variables and "
                          f"parameters; found: {sorted(map(str, extra))}")
    return sp.lambdify((y0, ym, ss, p), expr, modules="numpy")


def _constraints(model: Model, constraints, restr_map) -> list[Constraint]:
    spec = model.spec
    raw = list(spec.constraints) if constraints is None else list(constraints)
    if not raw:
        raise OccbinError("the model has no constraints (occbin_constraints block)")
    restr_map = dict(restr_map or {})          # chain -> order of the constraint (1-based), as in RISE
    by_order = {k: c for c, k in restr_map.items()}
    out = []
    for j, item in enumerate(raw):
        if isinstance(item, Constraint):
            out.append(item)
            continue
        chain, expr, text = item
        chain = chain or by_order.get(j + 1) or (model.chain_names[j] if j < len(model.chain_names) else None)
        if chain not in spec.chains:
            raise OccbinError(f"constraint {j + 1} ({text}) has no associated Markov chain")
        out.append(Constraint(chain, expr, text))
    for c in out:
        if spec.chains[c.chain].regimes != 2:
            raise OccbinError(f"the chain '{c.chain}' of a constraint must have two states")
    if len({c.chain for c in out}) != len(out):
        raise OccbinError("each constraint needs a chain of its own")
    return out


@dataclass
class OccbinSolution:
    """Piecewise-linear solution. Regimes are 1-based in the interface, in the order of ``model.regimes``."""
    model: Model
    params: np.ndarray            # (h, n_par)
    ss: np.ndarray                # (n,) steady state of the reference regime
    T: np.ndarray                 # (n, n) solution of the reference regime
    R: np.ndarray                 # (n, n_exo)
    A0: np.ndarray                # (h, n, n) derivatives of each regime at the reference steady state
    Am: np.ndarray
    Ap: np.ndarray
    B: np.ndarray
    k: np.ndarray                 # (h, n) residual of each regime at the reference steady state
    sigma: np.ndarray
    reference: int                # reference regime (1-based)
    constraints: list[Constraint]
    horizon: int = 40             # periods checked beyond the last constrained period
    max_iter: int = 200
    tol: float = 1e-10
    lookahead: bool = False       # False: like RISE; True: original OccBin (see the module docstring)
    _g: list = field(default_factory=list, repr=False)
    _relax: list = field(default_factory=list, repr=False)
    _regime_of: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        m = self.model
        self._g = [_compile(m, c.expr) for c in self.constraints]
        self._relax = [None if c.relax is None else _compile(m, c.relax) for c in self.constraints]
        ref = m.regimes[self.reference - 1]
        pos = [m.chain_names.index(c.chain) for c in self.constraints]
        for bits in np.ndindex(*(2,) * len(pos)):
            state = list(ref)
            for j, b in zip(pos, bits):
                if b:
                    state[j] = 3 - ref[j]
            self._regime_of[bits] = m.regimes.index(tuple(state))

    # ---------- core ----------
    def _names(self, variables):
        m = self.model
        names = [v for v in m.endogenous if not v.startswith("AUX_")] if variables is None else list(variables)
        return names, [m.endogenous.index(v) for v in names]

    def _rules(self, flags: np.ndarray):
        """Rules ``d_s = P_s d_{s-1} + c_s`` for the regime sequence ``flags`` (H, n_constraints)."""
        H, n = flags.shape[0], self.T.shape[0]
        P, c = self.T, np.zeros(n)
        Ps, cs = np.empty((H, n, n)), np.empty((H, n))
        ref_tail = True      # from here on, reference regime only
        D = self.R
        r0 = self.reference - 1
        for s in range(H - 1, -1, -1):
            r = self._regime_of[tuple(int(b) for b in flags[s])]
            if ref_tail and r == r0:
                Ps[s], cs[s] = self.T, 0.0
                continue
            ref_tail = False
            M = self.A0[r] + self.Ap[r] @ P
            try:
                X = np.linalg.solve(M, np.column_stack([self.Am[r], self.Ap[r] @ c + self.k[r], self.B[r]]))
            except np.linalg.LinAlgError as err:
                raise OccbinError(f"singular system with the regime sequence tried ({err})") from None
            P, c = -X[:, :n], -X[:, n]
            Ps[s], cs[s] = P, c
            if s == 0:
                D = -X[:, n + 1:]
        return Ps, cs, D

    def _path(self, d_prev, e, flags, length):
        """Expected path in deviations, ``length`` periods, with the sequence ``flags`` and then reference."""
        H = flags.shape[0]
        out = np.empty((length, d_prev.size))
        if H:
            Ps, cs, D = self._rules(flags)
        else:
            D = self.R
        d = d_prev
        for s in range(length):
            d = (Ps[s] @ d + cs[s]) if s < H else self.T @ d
            if s == 0:
                d = d + D @ e
            out[s] = d
        return out

    def _check(self, path, d_prev):
        """g of each constraint along the path (periods x constraints), with reference parameters."""
        p = self.params[self.reference - 1]
        y = self.ss + path
        ym = self.ss + np.vstack([d_prev, path[:-1]])
        return np.column_stack([np.broadcast_to(np.asarray(g(y.T, ym.T, self.ss, p), float), (len(y),))
                                for g in self._g])

    def step(self, d_prev: np.ndarray, e: np.ndarray, honor: bool = True):
        """One period: returns (deviation at t, flags of binding constraints at t, expected path).

        At each iteration, a reference period that violates the constraint moves to the alternative
        regime, and an alternative period goes back to reference when the exit condition holds.
        Without a declared ``relax``, the exit condition is the constraint itself evaluated in the
        "shadow" period: the period solved with the reference equations, expectations kept.
        """
        nc = len(self.constraints)
        if not honor:
            path = self._path(d_prev, e, np.zeros((0, nc), bool), 1)
            return path[0], np.zeros(nc, bool), path
        if not self.lookahead:
            # as in RISE: if the reference step honours the constraints at t, it is accepted
            path = self._path(d_prev, e, np.zeros((0, nc), bool), 1)
            if np.all(self._check(path, d_prev) >= -self.tol):
                return path[0], np.zeros(nc, bool), path
        flags = np.zeros((0, nc), bool)
        seen = []
        for _ in range(self.max_iter):
            H = flags.shape[0]
            L = H + self.horizon
            path = self._path(d_prev, e, flags, L)
            g = self._check(path, d_prev)
            full = np.zeros((L, nc), bool)
            full[:H] = flags
            new = full.copy()
            new[~full & (g < -self.tol)] = True
            if H:
                new[self._exit(path, d_prev, e, flags)] = False
            if not self.lookahead:
                # as in RISE: each constraint binds for a contiguous block starting at t
                for j in range(nc):
                    off = np.flatnonzero(~new[:, j])
                    new[(off[0] if off.size else L):, j] = False
            last = np.flatnonzero(new.any(1))
            new = new[: last[-1] + 1] if last.size else new[:0]
            if new.shape == flags.shape and np.array_equal(new, flags):
                return path[0], (flags[0] if H else np.zeros(nc, bool)), path
            key = new.tobytes() + bytes([new.shape[0] % 251])
            if key in seen:
                raise OccbinError("the regime sequence entered a cycle; no path honours the "
                                  "constraints with this algorithm")
            seen.append(key)
            flags = new
        raise OccbinError(f"the regime sequence did not converge in {self.max_iter} iterations")

    def _exit(self, path, d_prev, e, flags):
        """Constrained periods in which the constraint is no longer needed (boolean matrix like ``flags``)."""
        H, nc = flags.shape
        out = np.zeros((path.shape[0], nc), bool)
        p = self.params[self.reference - 1]
        prev = np.vstack([d_prev, path[:-1]])
        Ps, cs, _ = self._rules(flags)
        n = self.T.shape[0]
        for s in range(H):
            for j in np.flatnonzero(flags[s]):
                if self._relax[j] is not None:
                    val = float(self._relax[j](self.ss + path[s], self.ss + prev[s], self.ss, p))
                else:
                    # period s with constraint j switched off, expectations (P_{s+1}, c_{s+1}) kept
                    bits = [int(b) for b in flags[s]]
                    bits[j] = 0
                    r = self._regime_of[tuple(bits)]
                    Pn, cn = (Ps[s + 1], cs[s + 1]) if s + 1 < H else (self.T, np.zeros(n))
                    M = self.A0[r] + self.Ap[r] @ Pn
                    rhs = self.Am[r] @ prev[s] + self.Ap[r] @ cn + self.k[r] + (self.B[r] @ e if s == 0 else 0.0)
                    try:
                        shadow = -np.linalg.solve(M, rhs)
                    except np.linalg.LinAlgError:
                        continue
                    val = float(self._g[j](self.ss + shadow, self.ss + prev[s], self.ss, p))
                out[s, j] = val >= -self.tol
        return out

    # ---------- simulation ----------
    def simulate(self, periods: int = 100, seed: int = 0, shocks=None, burn: int = 0, y0=None,
                 honor_constraints: bool = True, variables=None) -> pd.DataFrame:
        """Simulates in levels starting from the steady state (or from ``y0``).

        ``shocks``: (periods, n_exo) matrix in standard-deviation units; without it, normals are
        drawn with ``seed``. Additional columns: ``regime`` (1-based) and the state of each chain.
        """
        m = self.model
        names, idx = self._names(variables)
        n_e = len(m.exogenous)
        if shocks is None:
            eta = np.random.default_rng(seed).standard_normal((periods + burn, n_e))
        else:
            eta = np.asarray(shocks, float).reshape(-1, n_e)
            burn = 0
        total = eta.shape[0]
        d = np.zeros(self.T.shape[0]) if y0 is None else np.asarray(y0, float) - self.ss
        out, reg = np.zeros((total, len(idx))), np.zeros(total, int)
        for t in range(total):
            d, active, _ = self.step(d, eta[t] * self.sigma, honor_constraints)
            out[t] = self.ss[idx] + d[idx]
            reg[t] = self._regime_of[tuple(int(b) for b in active)]
        df = pd.DataFrame(out[burn:], columns=names)
        df.insert(0, "regime", reg[burn:] + 1)
        for j, c in enumerate(m.chain_names):
            df.insert(1 + j, c, [m.regimes[r][j] for r in reg[burn:]])
        return df

    def irf(self, shock: str | None = None, periods: int = 40, size: float | None = None, y0=None,
            honor_constraints: bool = True, variables=None) -> pd.DataFrame:
        """Response to a shock in period 0, with the constraints, in deviation from the path without the shock.

        ``size``: size of the shock (default: one standard deviation); the sign matters, because the
        response is not symmetric. Index: (shock, period). Column ``regime`` holds the regime of the
        path with the shock.
        """
        m = self.model
        names, idx = self._names(variables)
        shocks = m.exogenous if shock is None else [shock]
        d0 = np.zeros(self.T.shape[0]) if y0 is None else np.asarray(y0, float) - self.ss
        zero = np.zeros(len(m.exogenous))
        base, d = np.zeros((periods, self.T.shape[0])), d0
        for t in range(periods):
            d, _, _ = self.step(d, zero, honor_constraints)
            base[t] = d
        rows, keys, regs = [], [], []
        for s in shocks:
            k = m.exogenous.index(s)
            e = zero.copy()
            e[k] = self.sigma[k] if size is None else size
            d = d0
            for t in range(periods):
                d, active, _ = self.step(d, e if t == 0 else zero, honor_constraints)
                rows.append(d[idx] - base[t][idx])
                regs.append(self._regime_of[tuple(int(b) for b in active)] + 1)
                keys.append((s, t))
        df = pd.DataFrame(rows, columns=names, index=pd.MultiIndex.from_tuples(keys, names=["shock", "period"]))
        df.insert(0, "regime", regs)
        return df


def solve_occbin(model: Model, params: dict | np.ndarray | None = None, reference: int = 1,
                 constraints=None, restr_map: dict | None = None, solver: str = "mfi",
                 shock_stderr: dict | None = None, tol: float = 1e-12, maxiter: int = 5000,
                 horizon: int = 40, lookahead: bool = False) -> OccbinSolution:
    """Solves the piecewise-linear model (RISE's ``solve_occbin``).

    ``reference``: reference regime (1-based), in which no constraint binds.
    ``restr_map``: ``{chain: order of the constraint}``, as in RISE; without it, the chain written in
    the constraint or the declaration order of the chains applies.
    ``lookahead``: ``False`` reproduces RISE's simulation; ``True`` uses the original OccBin.
    ``horizon``: periods checked beyond the last period with a binding constraint.
    """
    if not isinstance(params, np.ndarray):
        params = model.parameter_values(params)
    r0 = reference - 1
    if not 0 <= r0 < model.h:
        raise OccbinError(f"reference regime {reference} outside 1..{model.h}")
    cons = _constraints(model, constraints, restr_map)
    ss = model.steady_state(np.tile(params[r0], (model.h, 1)))[r0]
    sys = model.first_order_system(params, np.tile(ss, (model.h, 1)))
    if np.max(np.abs(sys.resid[r0])) > 1e-8:
        raise OccbinError("the steady state does not zero the equations of the reference regime")
    one = FirstOrderSystem(A0=sys.A0[[r0]], Am=sys.Am[[r0]], Ap=sys.Ap[[r0]], B=sys.B[[r0]], Q=np.ones((1, 1)))
    T0 = initial_guess(one, "backward")
    if solver == "mfi":
        T, _ = _mfi(one, T0, tol, maxiter)
    elif solver == "mnk":
        T, _ = _mnk(one, T0, tol, min(maxiter, 100))
    else:
        raise ValueError(f"unknown solver: {solver!r}")
    T = T[0]
    if np.max(np.abs(np.linalg.eigvals(T))) >= 1:
        raise SolveError("the solution of the reference regime is not stable")
    R = -np.linalg.solve(sys.A0[r0] + sys.Ap[r0] @ T, sys.B[r0])
    sd = {**model.spec.shock_stderr, **(shock_stderr or {})}
    sig = np.array([float(sd.get(s, 1.0)) for s in model.exogenous])
    return OccbinSolution(model=model, params=params, ss=ss, T=T, R=R, A0=sys.A0, Am=sys.Am, Ap=sys.Ap,
                          B=sys.B, k=sys.resid, sigma=sig, reference=reference, constraints=cons, horizon=horizon,
                          lookahead=lookahead)


# ---------------------------------------------------------------- constraint honoured through regime switching
def _plain_constraints(model: Model, constraints):
    raw = list(model.spec.constraints) if constraints is None else list(constraints)
    if not raw:
        raise OccbinError("the model has no constraints (occbin_constraints block)")
    out = []
    for item in raw:
        expr = item.expr if isinstance(item, Constraint) else (item[1] if isinstance(item, tuple) else item)
        out.append(_compile(model, expr))
    return out


def viable_regimes(sol, y_prev: np.ndarray, e: np.ndarray, constraints=None, tol: float = 1e-10, _g=None):
    """Candidate ``y_t`` of each regime and which regimes honour all the constraints.

    ``sol`` is a 1st-order :class:`msdsge.solve.Solution`. Returns ``(viable (h,), candidates (h, n))``.
    """
    g = _plain_constraints(sol.model, constraints) if _g is None else _g
    h = sol.T.shape[0]
    cand = np.stack([sol.ss[r] + sol.c[r] + sol.T[r] @ (y_prev - sol.ss[r]) + sol.R[r] @ e for r in range(h)])
    ok = np.array([all(float(f(cand[r], y_prev, sol.ss[r], sol.params[r])) >= -tol for f in g) for r in range(h)])
    return ok, cand


def simulate_through_switch(sol, periods: int = 100, seed: int = 0, shocks=None, regime0: int = 1, burn: int = 0,
                            y0=None, constraints=None, variables=None, tol: float = 1e-10) -> pd.DataFrame:
    """Simulation in which the constraints are honoured through regime switching.

    Equivalent to ``simul_honor_constraints=true`` with ``simul_honor_constraints_through_switch=true``
    in RISE: each period the candidate of each regime is computed, the regimes that violate some
    constraint are discarded and the regime is drawn among the viable ones, with the probabilities
    of the row of ``Q`` of the previous regime renormalised. ``regime0`` is the regime before the
    first period. If the viable regimes have zero probability, the draw is uniform among them (a
    choice of this package, not checked against RISE). With no viable regime, an error is raised.
    """
    m = sol.model
    if getattr(sol, "G2", None) is not None:
        raise NotImplementedError("constraints through regime switching only for 1st-order solutions")
    g = _plain_constraints(m, constraints)
    names, idx = sol._names(variables)
    rng = np.random.default_rng(seed)
    n_e, h = len(m.exogenous), sol.T.shape[0]
    if shocks is None:
        eta = rng.standard_normal((periods + burn, n_e))
    else:
        eta, burn = np.asarray(shocks, float).reshape(-1, n_e), 0
    total = eta.shape[0]
    r = regime0 - 1
    y_prev = sol.ss[r].copy() if y0 is None else np.asarray(y0, float)
    row = sol.Q[r]
    out, path = np.zeros((total, len(idx))), np.zeros(total, int)
    for t in range(total):
        ok, cand = viable_regimes(sol, y_prev, eta[t] * sol.sigma, tol=tol, _g=g)
        if not ok.any():
            raise OccbinError(f"period {t + 1}: no regime honours the constraints")
        p = np.where(ok, row, 0.0)
        p = p / p.sum() if p.sum() > 0 else ok / ok.sum()
        r = int(rng.choice(h, p=p))
        y = cand[r]
        out[t], path[t] = y[idx], r + 1
        row = sol.Q[r]
        if sol.endogenous_switching:
            row = np.clip(m.transition_matrix(y, y, y_prev, sol.params[r], sol.ss[r], regime=r), 0, 1)
            row = row / row.sum()
        y_prev = y
    df = pd.DataFrame(out[burn:], columns=names)
    df.insert(0, "regime", path[burn:])
    return df
