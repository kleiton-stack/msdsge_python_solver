"""Optimal policy in linear regime-switching models: commitment (Ramsey), discretion, loose
commitment and optimal simple rules (OSR).

Main reference: Debortoli, Maih and Nunes (2014), "Loose commitment in medium-scale
macroeconomic models", with the extension to Markov regimes used in RISE (additional chain
``loosecommit``). See docs/LITERATURE.md, entry on optimal policy.

Problem. With ``m`` private equations and ``n > m`` variables, in regime ``s``:

    Am_s y_{t-1} + A0_s y_t + Ap_s E_t y_{t+1} + B_s e_t = 0,

the planner maximises ``E sum_t beta^t u_s(y_t)``, with ``u`` quadratic (Hessian ``W_s``). Each
period the plan in force is kept with probability ``gamma_s`` and redone with ``1 - gamma_s``.
With multipliers ``lam_t`` (one per equation) the first-order conditions are

    W_s y_t + [A0_s' + (1-gamma_s) sum_s' Q[s,s'] (H_s')' Ap_s'] lam_t
      + (1/beta) Ap_{s(t-1)}' lam_{t-1} 1{plan kept at t}
      + beta E_t[Am_{s(t+1)}' lam_{t+1}] = 0,

where ``H_s'`` is the response of ``y_{t+1}`` to ``y_t`` when the plan is redone at t+1 and the
expectation of ``lam_{t+1}`` mixes the continuation of the plan (weight ``gamma_s``) and the new
plan (``1-gamma_s``). ``gamma = 1`` is the Ramsey plan (zero initial multipliers); ``gamma = 0`` is
the discretionary policy (Markov-perfect equilibrium). The fixed point in ``H`` is solved jointly
with the solution, by functional iteration.

Solution regimes. In the intermediate case the regimes double: the first ``h`` are "plan kept"
and the last ``h`` "plan redone" (chain ``loosecommit``, the slowest one). In the pure cases the
solution has the ``h`` regimes of the model.

Dating convention for the coefficients that switch regime (``timing``):
- ``"exact"`` (default): ``Ap`` of the regime at t-1 in the lagged term and ``Am`` of the regime at
  t+1 in the expected term, as the derivation requires. If ``Ap`` switches regime, the auxiliary
  variables ``MULTLAG_<var> = (Ap_s' lam_t)_var`` are added.
- ``"rise"``: everything with the coefficients of the current regime, which is what RISE (2019)
  does when it derives the conditions symbolically. Coincides with ``"exact"`` when ``Ap`` and
  ``Am`` do not switch.

Welfare scale: ``u = (1/2) y' W y`` with ``W`` the Hessian of the objective. RISE reports ``loss``
with ``W`` in place of ``W/2``, i.e. twice as much (see :func:`loss`, ``rise_scale``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy import optimize

from .parser import timed
from .solve import Solution, SolveError, mss_spectral_radius

__all__ = ["PolicyModel", "PlannerInfo", "OSRResult", "LossResult", "solve_optimal_policy", "loss",
           "optimal_simple_rule", "planner_weights"]


DEFAULT_DISCOUNT = 0.99   # discount used by RISE when the block does not give ``discount``


# ---------------------------------------------------------------------------------------------
# planner objective
# ---------------------------------------------------------------------------------------------
def _planner(model) -> dict:
    pl = getattr(model.spec, "planner", None)
    if pl is None:
        raise ValueError("the model has no planner_objective block")
    return pl


def _compile_planner(model):
    """Numeric functions (gradient, Hessian, discount, commitment) of the objective; cached."""
    cache = getattr(model, "_planner_fns", None)
    if cache is not None:
        return cache
    pl = _planner(model)
    y0 = [timed(v) for v in model.endogenous]
    p = [timed(v) for v in model.parameters]
    allowed = set(y0) | set(p)
    expr = sp.sympify(pl["expr"])
    extra = expr.free_symbols - allowed
    if extra:
        raise ValueError(f"planner_objective uses symbols that are not allowed (lags, leads or shocks): {sorted(map(str, extra))}")
    grad = sp.Matrix([sp.diff(expr, v) for v in y0])
    hess = grad.jacobian(y0)
    quadratic = not (hess.free_symbols & set(y0))
    lam = lambda e: sp.lambdify((y0, p), e, modules="numpy", cse=True)
    par = lambda e: None if e is None else sp.lambdify((p,), sp.sympify(e), modules="numpy")
    for key in ("discount", "commitment"):
        e = pl.get(key)
        if e is not None and sp.sympify(e).free_symbols - set(p):
            raise ValueError(f"planner_objective: '{key}' must depend on parameters only")
    cache = dict(grad=lam(grad), hess=lam(hess), quadratic=quadratic,
                 discount=par(pl.get("discount")), commitment=par(pl.get("commitment")))
    model._planner_fns = cache
    return cache


def planner_weights(model, params: np.ndarray, ss: np.ndarray | None = None):
    """Returns ``(W, grad, beta, gamma)`` per regime of the model.

    ``W`` (h, n, n) is the Hessian of the objective at the point ``ss`` (zero by default), ``grad``
    (h, n) the gradient, ``beta`` (h,) the discount (0.99 if not given, the RISE default) and
    ``gamma`` (h,) the probability of keeping the plan (``None`` if the block has no
    ``commitment``: the simple-rule case).
    """
    fns = _compile_planner(model)
    h, n = model.h, model.n
    ss = np.zeros((h, n)) if ss is None else ss
    W, g, beta, gamma = np.zeros((h, n, n)), np.zeros((h, n)), np.full(h, DEFAULT_DISCOUNT), None
    if fns["commitment"] is not None:
        gamma = np.zeros(h)
    for r in range(h):
        W[r] = np.array(fns["hess"](ss[r], params[r]), float).reshape(n, n)
        g[r] = np.array(fns["grad"](ss[r], params[r]), float).ravel()
        if fns["discount"] is not None:
            beta[r] = float(fns["discount"](params[r]))
        if gamma is not None:
            gamma[r] = float(fns["commitment"](params[r]))
    if np.any(beta <= 0) or np.any(beta > 1):
        raise ValueError(f"planner discount outside (0, 1]: {beta}")
    if gamma is not None and (np.any(gamma < -1e-12) or np.any(gamma > 1 + 1e-12)):
        raise ValueError(f"commitment probability outside [0, 1]: {gamma}")
    return W, g, beta, (None if gamma is None else np.clip(gamma, 0.0, 1.0))


# ---------------------------------------------------------------------------------------------
# augmented model (variables + multipliers), used by the Solution
# ---------------------------------------------------------------------------------------------
@dataclass
class PlannerInfo:
    """Data of the planner problem stored in the solution (``Solution.planner``)."""
    mode: str                 # "commitment", "discretion" or "loose"
    W: np.ndarray             # (H, N, N) Hessian of the objective in the variables of the solution
    beta: np.ndarray          # (H,)
    gamma: np.ndarray         # (h,) probability of keeping the plan in each regime of the model
    base_regime: np.ndarray   # (H,) regime of the model (0-based) of each regime of the solution
    replanning: np.ndarray    # (H,) bool: regime in which the plan is redone
    timing: str = "exact"


@dataclass
class PolicyModel:
    """Augmented model with the multipliers; plays the role of ``Solution.model``.

    Exposes what IRFs, simulation, filter and analyses use: names, regimes and transition matrix.
    """
    base: object
    endogenous: list[str]
    regimes: list[tuple]
    chain_names: list[str]
    Q: np.ndarray
    multipliers: list[str] = field(default_factory=list)
    endogenous_switching: bool = False

    @property
    def exogenous(self):
        return self.base.exogenous

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def spec(self):
        return self.base.spec

    @property
    def h(self) -> int:
        return len(self.regimes)

    @property
    def n(self) -> int:
        return len(self.endogenous)

    def parameter_values(self, overrides: dict | None = None) -> np.ndarray:
        p = self.base.parameter_values(overrides)
        return np.tile(p, (self.h // self.base.h, 1))

    def transition_matrix(self, yp=None, y0=None, ym=None, params_row=None, ss_row=None, regime: int | None = None):
        return self.Q if regime is None else self.Q[regime]

    def exogenous_transition_matrix(self, params=None) -> np.ndarray:
        return self.Q


# ---------------------------------------------------------------------------------------------
# linear system of the private sector
# ---------------------------------------------------------------------------------------------
def _private_system(model, params, tol=1e-8):
    """Matrices of the system (m equations, n variables) and approximation point of each regime.

    Requires a model that is linear in the variables. The point of each regime solves the static
    equations and sets the gradient of the objective to zero (zero multipliers in the steady state).
    """
    if model.endogenous_switching:
        raise NotImplementedError("optimal policy with endogenous transition probabilities is not supported")
    h, n, ne = model.h, model.n, len(model.exogenous)
    m = len(model._equations)
    fns = _compile_planner(model)
    if not fns["quadratic"]:
        raise NotImplementedError("planner_objective must be quadratic in the variables (linear-quadratic problem)")
    e0, z = np.zeros(ne), np.zeros(n)
    rng = np.random.default_rng(12345)
    probe = 1.0 + rng.random(n)
    A0, Am, Ap, B = (np.zeros((h, m, k)) for k in (n, n, n, ne))
    ss = np.zeros((h, n))
    for r in range(h):
        p = params[r]
        jac = lambda pt: [np.array(fn(pt, pt, pt, e0, pt, p), float).reshape(m, -1) for fn in model._jac]
        J = jac(z)
        with np.errstate(all="ignore"):
            J2 = jac(probe)
        if not all(np.all(np.isfinite(b)) and np.allclose(a, b, rtol=1e-10, atol=1e-12) for a, b in zip(J, J2)):
            raise NotImplementedError(
                "optimal policy requires a model that is linear in the endogenous variables (the derivatives change with the point)")
        Ap[r], A0[r], Am[r], B[r] = J
        const = np.asarray(model._f(z, z, z, e0, z, p), float).ravel()
        Wz = np.array(fns["hess"](z, p), float).reshape(n, n)
        gz = np.array(fns["grad"](z, p), float).ravel()
        if np.max(np.abs(const), initial=0.0) > 0 or np.max(np.abs(gz), initial=0.0) > 0:
            S = np.vstack([Ap[r] + A0[r] + Am[r], Wz])
            y = np.linalg.lstsq(S, -np.concatenate([const, gz]), rcond=None)[0]
            res = S @ y + np.concatenate([const, gz])
            if np.max(np.abs(res)) > tol * max(1.0, np.max(np.abs(y))):
                raise NotImplementedError(
                    f"regime {r + 1}: no point satisfies the equations and sets the gradient of the objective to zero "
                    "(a steady state with nonzero multipliers is not supported)")
            y[np.abs(y) < 1e-12 * max(1.0, np.max(np.abs(y)))] = 0.0
            ss[r] = y
    Q = np.stack([model.transition_matrix(z, z, z, params[r], z, regime=r) for r in range(h)])
    return A0, Am, Ap, B, Q, ss


def _lsolve(A, B):
    try:
        X = np.linalg.solve(A, B)
        if np.all(np.isfinite(X)):
            return X
    except np.linalg.LinAlgError:
        pass
    return np.linalg.lstsq(A, B, rcond=None)[0]


# ---------------------------------------------------------------------------------------------
# solution
# ---------------------------------------------------------------------------------------------
def solve_optimal_policy(model, params: np.ndarray | None = None, tol: float = 1e-12, maxiter: int = 5000,
                         timing: str = "exact", shock_stderr: dict | None = None, guess: np.ndarray | None = None,
                         commitment=None, discount=None, damping: float = 1.0, **options) -> Solution:
    """Solves the planner problem and returns a :class:`~msdsge.solve.Solution`.

    The variables of the solution are those of the model followed by the multipliers ``MULT_k``
    (one per equation, in equation order, auxiliaries included) and, if needed, ``MULTLAG_<var>``.

    ``commitment`` and ``discount`` (number or vector per regime) override those of the block.
    ``damping`` in (0, 1] damps the functional iteration (1 = no damping).
    """
    if timing not in ("exact", "rise"):
        raise ValueError("timing must be 'exact' or 'rise'")
    if options.get("order", 1) != 1:
        raise NotImplementedError("optimal policy at 1st order only (linear-quadratic problem)")
    if options.get("shock_horizon", 0):
        raise NotImplementedError("anticipated shocks with optimal policy are not supported")
    params = model.parameter_values() if params is None else params
    A0, Am, Ap, B, Q, ss = _private_system(model, params)
    h, m, n = A0.shape
    ne = B.shape[2]
    if m >= n:
        raise ValueError(f"optimal policy requires fewer equations ({m}) than variables ({n}): remove the policy rule")
    W, _, beta, gamma = planner_weights(model, params, ss)
    if discount is not None:
        beta = np.broadcast_to(np.asarray(discount, float), (h,)).copy()
    if commitment is not None:
        gamma = np.clip(np.broadcast_to(np.asarray(commitment, float), (h,)).copy(), 0.0, 1.0)
    if gamma is None:
        raise ValueError("planner_objective without 'commitment': use optimal_simple_rule (simple rule) "
                         "or pass commitment=...")
    mode = "commitment" if np.all(gamma == 1) else "discretion" if np.all(gamma == 0) else "loose"

    # auxiliary variables mu_t = Ap_s' lam_t when Ap switches regime (only the columns with a lead)
    lead_cols = np.flatnonzero(np.abs(Ap).sum((0, 1)) > 0)
    use_mu = timing == "exact" and h > 1 and not np.allclose(Ap, Ap[0])
    nmu = lead_cols.size if use_mu else 0
    N = n + m + nmu
    iy, il, imu = slice(0, n), slice(n, n + m), slice(n + m, N)

    # regimes of the solution: c = 0 (plan kept), c = 1 (plan redone)
    if mode == "loose":
        cs = [(c, s) for c in (0, 1) for s in range(h)]
    else:
        cs = [(0 if mode == "commitment" else 1, s) for s in range(h)]
    H = len(cs)
    Qb = np.zeros((H, H))
    for a, (_, s) in enumerate(cs):
        for b, (c2, s2) in enumerate(cs):
            Qb[a, b] = Q[s, s2] * (1.0 if mode != "loose" else (gamma[s] if c2 == 0 else 1 - gamma[s]))
    replan_of = {s: b for b, (c2, s) in enumerate(cs) if c2 == 1}   # "redone" regime of each s

    # fixed blocks; rows: [FOC (n); private equations (m); definition of mu (nmu)]
    G0 = np.zeros((h, N, N))
    Gm = np.zeros((H, N, N))
    Gp = np.zeros((h, h, N, N))          # [current s, future s']
    Bt = np.zeros((h, N, ne))
    rf, rc, rmu = slice(0, n), slice(n, n + m), slice(n + m, N)
    for s in range(h):
        G0[s][rf, iy] = W[s]
        G0[s][rf, il] = A0[s].T
        G0[s][rc, iy] = A0[s]
        Bt[s][rc] = B[s]
        if use_mu:
            G0[s][rmu, imu] = np.eye(nmu)
            G0[s][rmu, il] = -Ap[s][:, lead_cols].T
        for s2 in range(h):
            Gp[s, s2][rc, iy] = Ap[s]
            Gp[s, s2][rf, il] = beta[s] * (Am[s2].T if timing == "exact" else Am[s].T)
    for a, (c, s) in enumerate(cs):
        Gm[a][rc, iy] = Am[s]
        if c == 0:
            if use_mu:
                Gm[a][np.ix_(lead_cols, range(n + m, N))] = np.eye(nmu) / beta[s]
            else:
                Gm[a][rf, il] = Ap[s].T / beta[s]

    def U_of(T):
        U = G0.copy()
        for s in range(h):
            a = next(k for k, (_, s0) in enumerate(cs) if s0 == s)   # the row of Qb does not depend on c
            for b, (_, s2) in enumerate(cs):
                if Qb[a, b] != 0:
                    U[s] += Qb[a, b] * Gp[s, s2] @ T[b]
            if mode != "commitment" and gamma[s] < 1:
                # effect of y_t on expectations through the plan redone at t+1
                Hbar = sum(Q[s, s2] * T[replan_of[s2]][iy, iy] for s2 in range(h) if Q[s, s2] != 0)
                U[s][rf, il] += (1 - gamma[s]) * Hbar.T @ Ap[s].T
        return U

    T = np.zeros((H, N, N)) if guess is None else np.array(guess, float)
    if T.shape != (H, N, N):
        raise ValueError(f"guess must have shape {(H, N, N)}")
    its, diff = 0, np.inf
    for its in range(1, maxiter + 1):
        U = U_of(T)
        Tn = np.stack([-_lsolve(U[s], Gm[a]) for a, (_, s) in enumerate(cs)])
        if not np.all(np.isfinite(Tn)) or np.max(np.abs(Tn)) > 1e12:
            raise SolveError("optimal policy: the functional iteration diverged")
        diff = np.max(np.abs(Tn - T))
        T = Tn if damping == 1.0 else damping * Tn + (1 - damping) * T
        if diff < tol * max(1.0, np.max(np.abs(T))):
            break
    else:
        raise SolveError(f"optimal policy: the functional iteration did not converge in {maxiter} iterations "
                         f"(last change {diff:.2e})")
    U = U_of(T)
    R = np.stack([-_lsolve(U[s], Bt[s]) for (_, s) in cs])
    resid = float(max(np.max(np.abs(U[s] @ T[a] + Gm[a])) for a, (_, s) in enumerate(cs)))

    mult = [f"MULT_{k + 1}" for k in range(m)] + [f"MULTLAG_{model.endogenous[j]}" for j in (lead_cols if use_mu else [])]
    if mode == "loose":
        regimes = [(c + 1,) + tuple(model.regimes[s]) for c, s in cs]
        chains = ["loosecommit"] + list(model.chain_names)
    else:
        regimes, chains = list(model.regimes), list(model.chain_names)
    pm = PolicyModel(base=model, endogenous=list(model.endogenous) + mult, regimes=regimes, chain_names=chains,
                     Q=Qb, multipliers=mult)
    base_regime = np.array([s for _, s in cs])
    ssb = np.zeros((H, N))
    ssb[:, :n] = ss[base_regime]
    Wb = np.zeros((H, N, N))
    Wb[:, :n, :n] = W[base_regime]
    sd = {**model.spec.shock_stderr, **(shock_stderr or {})}
    sig = np.array([float(sd.get(x, 1.0)) for x in model.exogenous])
    sol = Solution(model=pm, params=params[base_regime], ss=ssb, c=np.zeros((H, N)), T=T, R=R,
                   R_ant=np.zeros((0,) + R.shape), Q=Qb, sigma=sig, solver=f"policy-{mode}", iterations=its,
                   residual=resid, mss_radius=mss_spectral_radius(T, Qb), endogenous_switching=False)
    sol.planner = PlannerInfo(mode=mode, W=Wb, beta=beta[base_regime], gamma=gamma, base_regime=base_regime,
                              replanning=np.array([c == 1 for c, _ in cs]), timing=timing)
    return sol


# ---------------------------------------------------------------------------------------------
# loss / welfare
# ---------------------------------------------------------------------------------------------
@dataclass
class LossResult:
    """Welfare ``v(x, r) = x' V_r x + d_r`` (``x`` in deviation from the point of the regime, variables at t)."""
    V: np.ndarray                    # (H, N, N)
    d: np.ndarray                    # (H,)
    period: np.ndarray               # (H,) E[u_t | regime at t], under the stationary distribution
    probabilities: np.ndarray        # (H,) ergodic distribution of the regimes
    unconditional: float             # E[v] = expected discounted sum of the objective
    conditional: float | None = None  # v(x0, regime), if requested

    @property
    def expected_period(self) -> float:
        """Unconditional E[u_t] (one-period objective)."""
        return float(self.probabilities @ self.period)


def _state_cols(T):
    return np.flatnonzero(np.abs(T).sum((0, 1)) > 0)


def _coupled_quadratic(T, Q, W, beta, tol=1e-13, maxiter=100000):
    """Solves ``P_i = W_i + beta_i sum_j Q[i,j] T_j' P_j T_j`` (coupled quadratic forms).

    Works with ``M_i = sum_j Q[i,j] T_j' P_j T_j`` restricted to the state columns. Small systems
    are solved by Kronecker (least squares, which admits unit roots in directions with no weight);
    large ones, by iteration.
    """
    H, N, _ = T.shape
    st = _state_cols(T)
    k = st.size
    if k == 0:
        return W.copy()
    Ts = T[:, :, st]                   # (H, N, k)
    Tss = T[:, st][:, :, st]           # (H, k, k)
    C = np.stack([sum(Q[i, j] * Ts[j].T @ W[j] @ Ts[j] for j in range(H)) for i in range(H)])
    if H * k * k <= 3600:
        big = np.eye(H * k * k)
        for i in range(H):
            for j in range(H):
                if Q[i, j] != 0:
                    big[i * k * k:(i + 1) * k * k, j * k * k:(j + 1) * k * k] -= Q[i, j] * beta[j] * np.kron(Tss[j].T, Tss[j].T)
        rhs = C.reshape(-1)
        try:
            x = np.linalg.solve(big, rhs)
            ok = np.all(np.isfinite(x)) and np.max(np.abs(big @ x - rhs)) < 1e-8 * max(1.0, np.max(np.abs(rhs)))
        except np.linalg.LinAlgError:
            ok = False
        if not ok:
            x = np.linalg.lstsq(big, rhs, rcond=None)[0]
            if np.max(np.abs(big @ x - rhs)) > 1e-8 * max(1.0, np.max(np.abs(rhs))):
                raise SolveError("non-finite loss: the quadratic form does not converge (unstable solution?)")
        M = x.reshape(H, k, k)
    else:
        M = C.copy()
        for _ in range(maxiter):
            Mn = C + np.stack([sum(Q[i, j] * beta[j] * Tss[j].T @ M[j] @ Tss[j] for j in range(H)) for i in range(H)])
            if not np.all(np.isfinite(Mn)) or np.max(np.abs(Mn)) > 1e14:
                raise SolveError("non-finite loss: the quadratic form diverges (unstable solution?)")
            done = np.max(np.abs(Mn - M)) < tol * max(1.0, np.max(np.abs(Mn)))
            M = Mn
            if done:
                break
        else:
            raise SolveError("loss: the iteration of the quadratic form did not converge")
    P = W.copy()
    for i in range(H):
        P[i][np.ix_(st, st)] += beta[i] * M[i]
    return (P + P.transpose(0, 2, 1)) / 2


def _solution_weights(sol):
    """(W, beta) in the variables and regimes of the solution, with ``u = (1/2) x' W x``."""
    info = getattr(sol, "planner", None)
    if info is not None:
        return info.W, info.beta
    W, g, beta, _ = planner_weights(sol.model, sol.params, sol.ss)
    if np.max(np.abs(g), initial=0.0) > 1e-8:
        raise NotImplementedError("nonzero gradient of the objective at the steady state: the quadratic loss "
                                  "around that point is not a valid approximation of welfare")
    return W, beta


def loss(sol: Solution, x0=None, regime: int = 1, rise_scale: bool = False) -> LossResult:
    """Planner welfare in a 1st-order solution (optimal or with a simple rule).

    ``v(x_t, r_t) = E_t sum_k beta^k u(x_{t+k})``, with ``u = (1/2) x' W x`` and ``W`` the Hessian of
    the ``planner_objective``. Returns the matrices ``V_r``, the constants ``d_r``, the
    unconditional value and, if ``x0`` is given (vector in levels, or dictionary name -> value in
    deviation, the rest zero), the value conditional on ``x0`` and ``regime`` (1-based).

    ``rise_scale=True`` uses ``W`` instead of ``W/2`` and reproduces RISE's ``loss`` (twice as much).
    The unconditional value requires the same discount in every regime.
    """
    from .filtering import ergodic_distribution
    W, beta = _solution_weights(sol)
    W = W if rise_scale else W / 2
    T, R, Q = sol.T, sol.R * sol.sigma, sol.Q
    H = T.shape[0]
    V = _coupled_quadratic(T, Q, W, beta)
    S = np.stack([R[j] @ R[j].T for j in range(H)])
    tr = np.array([np.sum(V[j] * S[j]) for j in range(H)])
    # d_i = beta_i sum_j Q[i,j] (d_j + tr(V_j R_j R_j'))
    if np.all(beta < 1):
        d = np.linalg.solve(np.eye(H) - beta[:, None] * Q, beta * (Q @ tr))
    else:
        d = np.full(H, np.nan if np.any(np.abs(Q @ tr) > 0) else 0.0)
    pi = ergodic_distribution(Q)
    # expected one-period objective: dual form without discounting, by age of the shock
    P1 = _coupled_quadratic(T, Q, W, np.ones(H))
    period_from = np.array([np.sum(P1[j] * S[j]) for j in range(H)])   # total effect of a shock that occurs in j
    e_period = float(pi @ period_from)
    # E[u_t | r_t = j]: tr(W_j Sigma_j) / pi_j, with Sigma_j = E[x x' 1{r=j}] (only if stationary)
    period = np.full(H, np.nan)
    try:
        Sig = _regime_second_moments(T, Q, S, pi)
        period = np.array([np.sum(W[j] * Sig[j]) / pi[j] if pi[j] > 0 else 0.0 for j in range(H)])
    except SolveError:
        pass
    if np.allclose(beta, beta[0]) and beta[0] < 1:
        uncond = e_period / (1 - beta[0])
    else:
        uncond = np.nan
    cond = None
    if x0 is not None:
        r = regime - 1
        if isinstance(x0, dict):
            x = np.zeros(T.shape[1])
            for kname, val in x0.items():
                x[sol.model.endogenous.index(kname)] = val
        else:
            x = np.asarray(x0, float) - sol.ss[r]
        cond = float(x @ V[r] @ x + d[r])
    return LossResult(V=V, d=d, period=period, probabilities=pi, unconditional=float(uncond), conditional=cond)


def _regime_second_moments(T, Q, S, pi, tol=1e-13, maxiter=20000):
    """``Sigma_j = E[x_t x_t' 1{r_t=j}] = sum_i Q[i,j] (T_j Sigma_i T_j' + pi_i S_j)``."""
    H, N, _ = T.shape
    if mss_spectral_radius(T, Q) >= 1 - 1e-10:
        raise SolveError("second moments do not exist (the solution is not mean-square stable)")
    Sig = np.zeros((H, N, N))
    for _ in range(maxiter):
        mix = np.einsum("ij,ikl->jkl", Q, Sig)
        new = np.stack([T[j] @ mix[j] @ T[j].T + (pi @ Q[:, j]) * S[j] for j in range(H)])
        done = np.max(np.abs(new - Sig)) < tol * max(1.0, np.max(np.abs(new)))
        Sig = new
        if done:
            return Sig
    raise SolveError("second moments: the iteration did not converge")


# ---------------------------------------------------------------------------------------------
# optimal simple rules
# ---------------------------------------------------------------------------------------------
@dataclass
class OSRResult:
    names: list[str]
    x: np.ndarray                # optimal coefficients
    x0: np.ndarray
    welfare: float               # unconditional welfare at the optimum (true scale, u = x'Wx/2)
    welfare0: float              # at the initial point
    params: dict                 # parameter overrides for Model.solve
    solution: Solution
    loss: LossResult
    success: bool
    message: str
    evaluations: int

    def summary(self):
        import pandas as pd
        return pd.DataFrame({"initial": self.x0, "optimal": self.x}, index=self.names)


def optimal_simple_rule(model, names=None, x0=None, bounds=None, method: str = "L-BFGS-B", starts: int = 1,
                        seed: int = 0, require_stable: bool = True, x0_state=None, regime: int = 1,
                        options: dict | None = None, **solve_options) -> OSRResult:
    """Chooses the coefficients of the policy rule that maximise the planner's welfare.

    The model has the policy rule among its equations and a ``planner_objective`` without
    ``commitment``. The optimised coefficients are, by default, the parameters of the
    ``estimated_params`` block (initial value and bounds), as in RISE, where ``estimate`` optimises
    the parameters with bounds in the ``parameterization``. The criterion is unconditional welfare;
    with ``x0_state`` (see :func:`loss`) it becomes welfare conditional on that state and on ``regime``.

    ``names`` may list parameters directly (``"phi"`` or ``("phi", chain_state)``).
    ``starts > 1`` adds random starts within the bounds (which must be finite).
    """
    from .solve import solve_first_order
    spec = model.spec
    _planner(model)
    items = []   # (label, name, chain state or None, initial, lower, upper)
    if names is None:
        for e in spec.estimated:
            if e.kind != "param":
                continue
            items.append((e.label, e.name, e.regime[1] if e.regime else None, e.init, e.lower, e.upper))
        if not items:
            raise ValueError("pass 'names' or declare the rule coefficients in estimated_params")
    else:
        for k, nm in enumerate(names):
            name, st = (nm, None) if isinstance(nm, str) else (nm[0], int(nm[1]))
            if name not in spec.parameters:
                raise ValueError(f"'{name}' is not a parameter of the model")
            cal = spec.calibration.get(name)
            if isinstance(cal, dict) and st is None:
                raise ValueError(f"'{name}' switches regime; use ('{name}', state)")
            init = cal[st] if isinstance(cal, dict) else cal
            lo, up = (-np.inf, np.inf) if bounds is None else bounds[k]
            items.append((name if st is None else f"{name}_{spec.switching[name]}_{st}", name, st,
                          None if init is None else float(sp.sympify(init)), lo, up))
    labels = [it[0] for it in items]
    start = np.array([it[3] for it in items], float) if x0 is None else np.asarray(x0, float)
    if not np.all(np.isfinite(start)):
        raise ValueError("coefficient without an initial value")
    lo = np.array([it[4] for it in items], float)
    up = np.array([it[5] for it in items], float)
    if bounds is not None and names is None:
        lo, up = np.array([b[0] for b in bounds], float), np.array([b[1] for b in bounds], float)

    def overrides(x):
        out: dict = {}
        for (_, name, st, *_), v in zip(items, x):
            if st is None:
                out[name] = float(v)
            else:
                cur = out.get(name)
                if cur is None:
                    cal = spec.calibration.get(name, {})
                    cur = out[name] = dict(cal) if isinstance(cal, dict) else {}
                cur[st] = float(v)
        return out

    count = [0]

    def evaluate(x):
        sol = solve_first_order(model, model.parameter_values(overrides(x)), **solve_options)
        res = loss(sol, x0=x0_state, regime=regime)
        return sol, res

    def objective(x):
        count[0] += 1
        if np.any(x < lo) or np.any(x > up):
            return 1e12
        try:
            sol, res = evaluate(x)
        except (SolveError, ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError):
            return 1e12
        if require_stable and not sol.is_stable:
            return 1e12
        val = res.unconditional if x0_state is None else res.conditional
        return -val if np.isfinite(val) else 1e12

    f0 = objective(start)
    if f0 >= 1e12:
        raise SolveError("simple rule: the model does not solve (or is unstable) at the initial point")
    bnds = [(None if not np.isfinite(a) else a, None if not np.isfinite(b) else b) for a, b in zip(lo, up)]
    trials = [start]
    if starts > 1:
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(up))):
            raise ValueError("random starts require finite bounds")
        rng = np.random.default_rng(seed)
        trials += [lo + rng.random(len(lo)) * (up - lo) for _ in range(starts - 1)]
    best = None
    for t0 in trials:
        if objective(t0) >= 1e12:
            continue
        kw = dict(method=method, options=options or {})
        if method in ("L-BFGS-B", "SLSQP", "TNC", "trust-constr", "Nelder-Mead", "Powell"):
            kw["bounds"] = bnds
        res = optimize.minimize(objective, t0, **kw)
        if best is None or res.fun < best.fun:
            best = res
    sol, lres = evaluate(best.x)
    return OSRResult(names=labels, x=np.array(best.x, float), x0=start, welfare=-float(best.fun), welfare0=-float(f0),
                     params=overrides(best.x), solution=sol, loss=lres, success=bool(best.success),
                     message=str(best.message), evaluations=count[0])
