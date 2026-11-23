"""Numerical model: regimes, regime-specific calibration, steady state and first-order derivatives.

Conventions (see docs/LITERATURE.md, summary of Maih, 2015):
- ``Q[i, j] = Pr(s_{t+1}=j | s_t=i)``; rows sum to one.
- With several chains, the regimes are the Cartesian product, with the first declared chain
  varying most slowly.
- Approximation point: the regime-specific steady state (RISE default).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import optimize

from .parser import ModelSpec, _SS, load as _load_spec, parse as _parse_spec, timed


class SteadyStateError(RuntimeError):
    pass


@dataclass
class FirstOrderSystem:
    """Derivatives at the steady state of each regime: A0 y_t + Am y_{t-1} + Ap E y_{t+1} + B e_t = 0."""
    A0: np.ndarray  # (h, n, n)
    Am: np.ndarray  # (h, n, n)
    Ap: np.ndarray  # (h, n, n), not probability-weighted
    B: np.ndarray   # (h, n, n_exo)
    Q: np.ndarray   # (h, h), evaluated at the steady state of each origin regime
    resid: np.ndarray | None = None  # (h, n) residual of the equations at the approximation point


class Model:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        # function f(model, {parameter: value}) -> {name: value} with variables and, optionally, computed
        # parameters (equivalent to the RISE steady-state file); see solve(steady_state_file=...)
        self.steady_state_file = None
        self.steady_state_imposed = None    # None: follows the attribute of the steady_state_model block
        self._expand_auxiliaries()
        self.endogenous = list(self.spec.endogenous) + [a for a, _ in self._aux]
        self.exogenous = list(spec.exogenous)
        self.parameters = list(spec.parameters)
        self.chain_names = list(spec.chains)
        sizes = [spec.chains[c].regimes for c in self.chain_names]
        # each composite regime is a tuple with the (1-based) state of each chain
        self.regimes = list(itertools.product(*[range(1, k + 1) for k in sizes])) or [()]
        self.h = len(self.regimes)
        self.n = len(self.endogenous)
        self.endogenous_switching = any(c.is_endogenous(self.parameters) for c in spec.chains.values())
        self._compile()

    # ---------- construction ----------
    @classmethod
    def from_file(cls, path: str | Path) -> "Model":
        return cls(_load_spec(path))

    @classmethod
    def from_text(cls, text: str) -> "Model":
        return cls(_parse_spec(text))

    def _expand_auxiliaries(self) -> None:
        """Reduces leads and lags greater than 1 in endogenous variables using auxiliary variables."""
        spec = self.spec
        eqs = list(spec.equations)
        trans = {c: dict(ch.transitions) for c, ch in spec.chains.items()}
        aux: list[tuple[str, sp.Expr]] = []  # (name, definition at t)
        subs: dict[sp.Symbol, sp.Symbol] = {}
        exprs = eqs + [e for t in trans.values() for e in t.values()]
        used = set().union(*[e.free_symbols for e in exprs]) if exprs else set()
        # shock with a lead/lag: auxiliary variable a_t = e_t; e(k) becomes a(k)
        exo_aux: list[str] = []
        esubs: dict[sp.Symbol, sp.Symbol] = {}
        for name in spec.exogenous:
            ks = [k for k in range(-spec.max_lag, spec.max_lead + 1) if k and timed(name, k) in used]
            if ks:
                a = f"AUX_EXO_{name}"
                aux.append((a, timed(name)))
                exo_aux.append(a)
                esubs.update({timed(name, k): timed(a, k) for k in ks})
        if esubs:
            eqs = [e.xreplace(esubs) for e in eqs]
            trans = {c: {ij: e.xreplace(esubs) for ij, e in t.items()} for c, t in trans.items()}
            exprs = eqs + [e for t in trans.values() for e in t.values()]
            used = set().union(*[e.free_symbols for e in exprs])
        for name in list(spec.endogenous) + exo_aux:
            for sign, top in ((-1, spec.max_lag), (1, spec.max_lead)):
                deepest = max((k for k in range(2, top + 1) if timed(name, sign * k) in used), default=1)
                prev = name
                for k in range(2, deepest + 1):
                    a = f"AUX_{name}_{'LAG' if sign < 0 else 'LEAD'}_{k - 1}"
                    aux.append((a, timed(prev, sign)))  # a_t = prev_{t∓1}
                    subs[timed(name, sign * k)] = timed(a, sign)
                    prev = a
        self._aux = aux
        self._equations = [e.xreplace(subs) for e in eqs] + [timed(a) - d for a, d in aux]
        self._transitions = {c: {ij: e.xreplace(subs) for ij, e in t.items()} for c, t in trans.items()}

    def _compile(self) -> None:
        y0 = [timed(v) for v in self.endogenous]
        ym = [timed(v, -1) for v in self.endogenous]
        yp = [timed(v, 1) for v in self.endogenous]
        e = [timed(v) for v in self.exogenous]
        p = [timed(v) for v in self.parameters]
        ss_syms = [sp.Symbol(f"{v}__SS", real=True) for v in self.endogenous]
        ss_map = {_SS(timed(v)): s for v, s in zip(self.endogenous, ss_syms)}

        def clean(expr):
            expr = expr.xreplace(ss_map)
            if expr.atoms(sp.Function) & {a for a in expr.atoms(sp.Function) if a.func == _SS}:
                raise ValueError("steady_state() only accepts an endogenous variable without a lag")
            return expr

        F = sp.Matrix([clean(q) for q in self._equations])
        args = (yp, y0, ym, e, ss_syms, p)
        lam = lambda m: sp.lambdify(args, m, modules="numpy", cse=True)
        self._f = lam(F)
        self._jac = [lam(F.jacobian(v)) for v in (yp, y0, ym, e)]
        # static equations: everything at the same point, shocks at zero
        static = {**dict(zip(yp, y0)), **dict(zip(ym, y0)), **dict(zip(ss_syms, y0)), **{s: 0 for s in e}}
        Fs = F.xreplace(static)
        self._fs = sp.lambdify((y0, p), Fs, modules="numpy", cse=True)
        self._fs_jac = sp.lambdify((y0, p), Fs.jacobian(y0), modules="numpy", cse=True)
        self._trans_fn = {
            c: {ij: sp.lambdify(args, clean(ex), modules="numpy") for ij, ex in t.items()}
            for c, t in self._transitions.items()
        }
        # steady_state_model: sequential assignments to variables, parameters and auxiliary constants
        loc_names = [v for v, _ in self.spec.steady_state if v.endswith("__SSAUX")]
        loc = [sp.Symbol(v, real=True) for v in dict.fromkeys(loc_names)]
        self._ss_model = []
        for v, ex in self.spec.steady_state:
            kind = "local" if v.endswith("__SSAUX") else ("endo" if v in self.endogenous else "param")
            pos = {"local": lambda: [x.name for x in loc].index(v), "endo": lambda: self.endogenous.index(v),
                   "param": lambda: self.parameters.index(v)}[kind]()
            self._ss_model.append((kind, pos, sp.lambdify((y0, p, loc), ex.xreplace(static), modules="numpy")))
        self._ss_nloc = len(loc)
        # parameters that appear in some expression (the others may be left without a value, as in RISE)
        cal = [x for v in self.spec.calibration.values() for x in (v.values() if isinstance(v, dict) else [v])]
        every = (list(self._equations) + [x for t in self._transitions.values() for x in t.values()]
                 + [x for _, x in self.spec.steady_state] + [sp.sympify(x) for x in cal]
                 + list(self.spec.initval.values()) + [g for g, _ in self.spec.restrictions])
        self._used_params = {x.name for q in every for x in q.free_symbols}
        self._ss_params = [v for v, _ in self.spec.steady_state if v in self.parameters and v not in self.endogenous]
        self._ss_vars = [v for v, _ in self.spec.steady_state if v in self.endogenous]
        syms = {s.name: s for g, _ in self.spec.restrictions for s in g.free_symbols}
        self._restr_syms = list(syms.values())
        self._restr_fn = [(sp.lambdify(self._restr_syms, g, modules="numpy"), strict) for g, strict in self.spec.restrictions]

    def _fill_aux(self, y) -> None:
        """Auxiliaries copy their source variable (zero when the source is a shock)."""
        for a, d in self._aux:
            src = str(next(iter(d.free_symbols))).split("__")[0]
            y[self.endogenous.index(a)] = y[self.endogenous.index(src)] if src in self.endogenous else 0.0

    def steady_state_point(self, p, y=None):
        """Applies the steady-state file or the ``steady_state_model`` block to one row of parameters.

        Returns ``(y, p)``: the point (not yet verified) and the parameters, including the computed ones.
        """
        p = np.array(p, float)
        y = self._initval(np.nan_to_num(p)) if y is None else np.array(y, float)
        with np.errstate(all="ignore"):
            if self.steady_state_file is not None:
                res = self.steady_state_file(self, dict(zip(self.parameters, p)))
                for k, v in res.items():
                    if k in self.endogenous:
                        y[self.endogenous.index(k)] = float(v)
                    elif k in self.parameters:
                        p[self.parameters.index(k)] = float(v)
                    else:
                        raise SteadyStateError(f"the steady-state file returned an unknown name: '{k}'")
            else:
                loc = np.zeros(self._ss_nloc)
                for kind, pos, fn in self._ss_model:
                    {"endo": y, "param": p, "local": loc}[kind][pos] = fn(y, p, loc)
        self._fill_aux(y)
        return y, p

    def check_restrictions(self, params: np.ndarray | None = None) -> bool:
        """True if the inequalities in ``parameter_restrictions`` hold for the parameter matrix."""
        if not self._restr_fn:
            return True
        params = self.parameter_values() if params is None else params
        for r, state in enumerate(self.regimes):
            vals = []
            for s in self._restr_syms:
                if "__R__" in s.name:
                    name, chain, k = s.name.split("__R__")[0], *s.name.split("__R__")[1].rsplit("__", 1)
                    row = next(i for i, st in enumerate(self.regimes) if st[self.chain_names.index(chain)] == int(k))
                    vals.append(params[row, self.parameters.index(name)])
                else:
                    vals.append(params[r, self.parameters.index(s.name)])
            for fn, strict in self._restr_fn:
                g = float(fn(*vals))
                if not (g > 0 if strict else g >= 0):
                    return False
        return True

    # ---------- parameters ----------
    def parameter_values(self, overrides: dict | None = None) -> np.ndarray:
        """Matrix (h, n_par) with the value of each parameter in each composite regime.

        ``overrides``: ``{"beta": 0.98}`` or, for a switching parameter, ``{"phi": {1: 1.5, 2: 0.9}}``.
        """
        spec = self.spec
        cal = dict(spec.calibration)
        cal.update(overrides or {})
        out = np.full((self.h, len(self.parameters)), np.nan)
        syms = [timed(n) for n in self.parameters]
        for r, state in enumerate(self.regimes):
            exprs = {}
            for name in self.parameters:
                v = cal.get(name)
                if isinstance(v, dict):
                    v = v[state[self.chain_names.index(spec.switching[name])]]
                if v is None:
                    unused = name not in self._used_params and not (
                        name.startswith("stderr_") and name[7:] in spec.observables)
                    if name not in self._ss_params and self.steady_state_file is None and not unused:
                        raise ValueError(f"parameter without a value: '{name}'")
                    v = sp.nan   # computed in the steady state (block or file), just below
                exprs[timed(name)] = sp.sympify(v)
            # resolve dependencies between parameters (e.g. a = 1/beta)
            for _ in range(len(exprs) + 1):
                if all(not x.free_symbols for x in exprs.values()):
                    break
                known = {k: x for k, x in exprs.items() if not x.free_symbols}
                exprs = {k: x.xreplace(known) for k, x in exprs.items()}
            else:
                raise ValueError("circular definition among parameters")
            out[r] = [float(exprs[s]) for s in syms]
            if self._ss_params or self.steady_state_file is not None:
                out[r] = self.steady_state_point(out[r])[1]
                miss = [v for v, x in zip(self.parameters, out[r]) if np.isnan(x) and v in self._used_params]
                if miss:
                    raise ValueError(f"parameter without a value: '{miss[0]}'")
        return out

    # ---------- steady state ----------
    def steady_state(self, params: np.ndarray | None = None, guess=None, tol: float = 1e-10) -> np.ndarray:
        """Steady state of each regime, matrix (h, n)."""
        params = self.parameter_values() if params is None else params
        out = np.zeros((self.h, self.n))
        for r in range(self.h):
            p = params[r]
            if guess is None:
                y = self._initval(p)
            else:
                y = np.array(np.broadcast_to(guess, (self.h, self.n))[r], float)
            given = bool(self._ss_model) or self.steady_state_file is not None
            if given:
                y, p = self.steady_state_point(p, y)
                opts = self.spec.steady_state_options
                imposed = ("imposed" in opts) if self.steady_state_imposed is None else self.steady_state_imposed
                if self.steady_state_imposed is None and self.steady_state_file is not None:
                    imposed = bool(getattr(self.steady_state_file, "imposed", False))
                if imposed:   # use the supplied point without checking it (RISE steady_state_imposed)
                    if not np.all(np.isfinite(y)):
                        raise SteadyStateError(f"imposed steady state is not finite in regime {r + 1}")
                    out[r] = y
                    continue
                if "loop" in opts and self.steady_state_file is None:
                    y = self._solve_missing(y, p, r, tol)
            with np.errstate(all="ignore"):
                res = np.asarray(self._fs(y, p), float).ravel()
                if not given and guess is None and not self.spec.initval and not np.all(np.isfinite(res)):
                    # the zero starting point does not work (log, division): start from one, as a second attempt
                    y = np.ones(self.n)
                    self._fill_aux(y)
                    res = np.asarray(self._fs(y, p), float).ravel()
            if not np.all(np.isfinite(res)) or np.max(np.abs(res)) > tol:
                if not np.all(np.isfinite(y)):
                    raise SteadyStateError(
                        f"steady_state_model returns a non-finite value in regime {r + 1}; if the steady "
                        "state of the regime does not exist, use solve(steady_state='unique')")
                with np.errstate(all="ignore"):
                    sol = optimize.root(lambda x: np.asarray(self._fs(x, p), float).ravel(), y,
                                        jac=lambda x: np.asarray(self._fs_jac(x, p), float), tol=tol * 1e-2)
                    res = np.asarray(self._fs(sol.x, p), float).ravel()
                if not np.all(np.isfinite(res)) or np.max(np.abs(res)) > tol * 1e2:
                    raise SteadyStateError(
                        f"steady state not found in regime {r + 1} "
                        f"(maximum residual {np.max(np.abs(res)):.2e}); provide 'guess' or steady_state_model")
                y = sol.x
            out[r] = y
        return out

    def _solve_missing(self, y, p, r, tol):
        """``loop`` attribute: keeps the variables given in the block and solves only for the missing ones."""
        free = np.array([i for i, v in enumerate(self.endogenous)
                         if v not in self._ss_vars and not v.startswith("AUX_")], int)
        if free.size == 0:
            return y

        def full(x):
            z = y.copy()
            z[free] = x
            self._fill_aux(z)
            return z

        with np.errstate(all="ignore"):
            if not np.all(np.isfinite(np.asarray(self._fs(y, p), float))):
                y = y.copy()
                y[free] = 1.0      # zero does not work as a starting point (log, division)
            sol = optimize.least_squares(lambda x: np.asarray(self._fs(full(x), p), float).ravel(), y[free],
                                         jac=lambda x: np.asarray(self._fs_jac(full(x), p), float)[:, free],
                                         xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if not np.all(np.isfinite(sol.fun)) or np.max(np.abs(sol.fun)) > tol * 1e2:
            raise SteadyStateError(
                f"steady_state_model(loop): the missing variables do not close the system in regime {r + 1} "
                f"(maximum residual {np.max(np.abs(sol.fun)):.2e})")
        return full(sol.x)

    def _initval(self, p) -> np.ndarray:
        """Guess from the ``initval`` block (zero for whatever was not given)."""
        y = np.zeros(self.n)
        vals = dict(zip((timed(n) for n in self.parameters), p))
        for v, ex in self.spec.initval.items():
            y[self.endogenous.index(v)] = float(ex.xreplace(vals))
        self._fill_aux(y)
        return y

    # ---------- transition matrix ----------
    def transition_matrix(self, yp, y0, ym, params_row, ss_row, regime: int | None = None) -> np.ndarray:
        """Q evaluated at a point. With ``regime``, returns only the row of that origin regime."""
        e = np.zeros(len(self.exogenous))
        mats = []
        for c in self.chain_names:
            k = self.spec.chains[c].regimes
            P = np.zeros((k, k))
            for (i, j), fn in self._trans_fn[c].items():
                P[i - 1, j - 1] = float(fn(yp, y0, ym, e, ss_row, params_row))
            if np.any(P < -1e-12) or np.any(P.sum(1) > 1 + 1e-12):
                raise ValueError(f"invalid transition probabilities in chain '{c}'")
            P[np.diag_indices(k)] = 1 - P.sum(1)
            mats.append(P)
        Q = np.ones((1, 1))
        for P in mats:
            Q = np.kron(Q, P)
        return Q if regime is None else Q[regime]

    # ---------- derivatives ----------
    def exogenous_transition_matrix(self, params: np.ndarray) -> np.ndarray:
        """Q when the probabilities depend only on parameters (error if they are endogenous)."""
        if self.endogenous_switching:
            raise NotImplementedError("a single approximation point with endogenous probabilities is not supported yet")
        z = np.zeros(self.n)
        return np.stack([self.transition_matrix(z, z, z, params[r], z, regime=r) for r in range(self.h)])

    def first_order_system(self, params: np.ndarray | None = None, ss: np.ndarray | None = None) -> FirstOrderSystem:
        params = self.parameter_values() if params is None else params
        ss = self.steady_state(params) if ss is None else ss
        e = np.zeros(len(self.exogenous))
        mats = [[], [], [], []]
        Q = np.zeros((self.h, self.h))
        resid = np.zeros((self.h, self.n))
        for r in range(self.h):
            a = (ss[r], ss[r], ss[r], e, ss[r], params[r])
            resid[r] = np.asarray(self._f(*a), float).ravel()
            for k, fn in enumerate(self._jac):
                mats[k].append(np.array(fn(*a), float).reshape(self.n, -1))
            Q[r] = self.transition_matrix(ss[r], ss[r], ss[r], params[r], ss[r], regime=r)
        Ap, A0, Am, B = (np.stack(m) for m in mats)
        return FirstOrderSystem(A0=A0, Am=Am, Ap=Ap, B=B, Q=Q, resid=resid)

    def solve(self, params: dict | None = None, **options):
        """Solves the model. ``params`` overrides the calibration in the file."""
        from .solve import solve_first_order
        if "steady_state_file" in options:      # f(model, {parameter: value}) -> {name: value}
            self.steady_state_file = options.pop("steady_state_file")
        if "steady_state_imposed" in options:
            self.steady_state_imposed = options.pop("steady_state_imposed")
        return solve_first_order(self, self.parameter_values(params), **options)


    def solve_alternatives(self, params: dict | None = None, **options):
        """Distinct solutions found from random guesses (see solve.solve_alternatives)."""
        from .solve import solve_alternatives
        return solve_alternatives(self, self.parameter_values(params), **options)

    def estimate(self, data, **options):
        """Posterior mode or maximum likelihood (see estimation.estimate)."""
        from .estimation import estimate
        return estimate(self, data, **options)


def load(path: str | Path) -> Model:
    return Model.from_file(path)
