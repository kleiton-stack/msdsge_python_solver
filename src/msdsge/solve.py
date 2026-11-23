"""First-order solution of regime-switching models (Maih, 2015, eqs. 11, 14, 15, 25 to 30).

"Full" version (static variables are not exploited): unknown ``T_r`` (n x n) per regime, with

    A0_r T_r + Am_r + (sum_r' Q[r,r'] Ap_r T_r') T_r = 0.

Solution: ``y_t - ss_r = T_r (y_{t-1} - ss_r) + R_r e_t`` when ``s_t = r``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SOLVERS = ("mfi", "mnk")


class SolveError(RuntimeError):
    pass


def _lsolve(A, B):
    """Solves A X = B regime by regime; if A is singular, uses the minimum-norm solution.

    The matrix U_r may be singular at the initial guess (for example, an interest-rate rule with a
    zero coefficient) without being singular at the solution. RISE carries on in these cases.
    """
    try:
        X = np.linalg.solve(A, B)
        if np.all(np.isfinite(X)):
            return X
    except np.linalg.LinAlgError:
        pass
    return np.stack([np.linalg.lstsq(A[r], B[r], rcond=None)[0] for r in range(A.shape[0])])


def _U(sys, T):
    """U_r = A0_r + sum_r' Q[r,r'] Ap_r T_r' (eq. 14)."""
    return sys.A0 + np.einsum("rs,rij,sjk->rik", sys.Q, sys.Ap, T)


def residual(sys, T) -> float:
    """Maximum norm of the residual of the quadratic matrix equation."""
    return float(np.max(np.abs(np.einsum("rij,rjk->rik", _U(sys, T), T) + sys.Am)))


def initial_guess(sys, kind: str = "backward", rng=None) -> np.ndarray:
    h, n, _ = sys.A0.shape
    if kind == "zeros":
        return np.zeros((h, n, n))
    if kind == "backward":  # T_r = -A0_r^{-1} Am_r, RISE default
        return -_lsolve(sys.A0, sys.Am)
    if kind == "random":
        rng = np.random.default_rng(rng)
        return rng.standard_normal((h, n, n)) * (np.abs(sys.Am).sum(1, keepdims=True) > 0)
    raise ValueError(f"unknown initial guess: {kind!r}")


def _states(sys) -> np.ndarray:
    """Indices of the state variables (they appear lagged in some regime)."""
    return np.flatnonzero(np.abs(sys.Am).sum((0, 1)) > 0)


def _mfi(sys, T, tol, maxiter):
    """Functional iteration (eq. 25)."""
    for it in range(1, maxiter + 1):
        U = _U(sys, T)
        Tn = -_lsolve(U, sys.Am)
        if not np.all(np.isfinite(Tn)) or np.max(np.abs(Tn)) > 1e12:
            raise SolveError("functional iteration diverged")
        # tolerance relative to the size of T: with large coefficients (EDO model, of order 1e3)
        # the absolute difference stalls at the rounding noise
        done = np.max(np.abs(Tn - T)) < tol * max(1.0, np.max(np.abs(Tn)))
        T = Tn
        if done:
            return T, it
    raise SolveError(f"functional iteration did not converge in {maxiter} iterations")


def _mnk(sys, T, tol, maxiter):
    """Newton with a direct (Kronecker) solution of the coupled Sylvester equation (eqs. 26 to 29).

    Works only with the state columns: X_r (n x k), with T_r = [0, X_r] and L selecting the
    state rows. Step: D_r + sum_r' Lp[r,r'] D_r' (L X_r) = -W_r, Lp = U_r^{-1} Q[r,r'] Ap_r.
    """
    h, n, _ = T.shape
    st = _states(sys)
    k = st.size
    if k == 0:
        return np.zeros_like(T), 0
    X = T[:, :, st]
    Amx = sys.Am[:, :, st]
    I = np.eye(n * k)
    for it in range(1, maxiter + 1):
        U = _U(sys, _embed(X, st, n))
        W = X + _lsolve(U, Amx)                                 # eq. 26
        err = np.max(np.abs(W))
        if err < tol:
            return _embed(X, st, n), it
        Lp0 = _lsolve(U, sys.Ap)
        big = np.zeros((h * n * k, h * n * k))
        for r in range(h):
            G = X[r][st, :]                                     # L X_r  (k x k)
            for q in range(h):
                blk = np.kron(G.T, sys.Q[r, q] * Lp0[r])
                if r == q:
                    blk = blk + I
                big[r * n * k:(r + 1) * n * k, q * n * k:(q + 1) * n * k] = blk
        rhs = np.concatenate([-W[r].ravel(order="F") for r in range(h)])
        try:
            d = np.linalg.solve(big, rhs)
        except np.linalg.LinAlgError:
            d = np.linalg.lstsq(big, rhs, rcond=None)[0]
        D = d.reshape(h, k, n).transpose(0, 2, 1)
        # simple line search (Maih, 2015, footnote 35): shorten the step if the residual gets worse
        step = 1.0
        for _ in range(30):
            Xn = X + step * D
            Wn = Xn + _lsolve(_U(sys, _embed(Xn, st, n)), Amx)
            if np.all(np.isfinite(Wn)) and np.max(np.abs(Wn)) < err * (1 - 1e-4 * step) or step < 1e-8:
                break
            step /= 2
        X = Xn
        if not np.all(np.isfinite(X)):
            raise SolveError("Newton diverged")
    raise SolveError(f"Newton did not converge in {maxiter} iterations")


def _embed(X, st, n):
    """Builds T (h, n, n) from the state columns X (h, n, k)."""
    T = np.zeros((X.shape[0], n, n))
    T[:, :, st] = X
    return T


def mss_spectral_radius(T, Q) -> float:
    """Spectral radius of the mean-square stability matrix (eq. 30; Costa et al., 2005).

    With ``x_{t+1} = T_{s_{t+1}} x_t``, the regime-specific second moments evolve with the block matrix
    ``(j, i) = Q[i, j] (T_j kron T_j)``. MSS if and only if the spectral radius is less than one.
    Uses only the state columns (variables that actually appear lagged).
    """
    h = T.shape[0]
    keep = np.flatnonzero(np.abs(T).sum((0, 1)) > 0)
    if keep.size == 0:
        return 0.0
    Tx = T[:, keep][:, :, keep]
    m = keep.size ** 2
    big = np.zeros((h * m, h * m))
    for j in range(h):
        K = np.kron(Tx[j], Tx[j])
        for i in range(h):
            big[j * m:(j + 1) * m, i * m:(i + 1) * m] = Q[i, j] * K
    return float(np.max(np.abs(np.linalg.eigvals(big))))


@dataclass
class Solution:
    model: object
    params: np.ndarray      # (h, n_par)
    ss: np.ndarray          # (h, n) approximation point of each regime
    c: np.ndarray           # (h, n) constant (uncertainty term); zero if the point is the regime ss
    T: np.ndarray           # (h, n, n)
    R: np.ndarray           # (h, n, n_exo), response to a unit shock
    R_ant: np.ndarray       # (K, h, n, n_exo): response at t to a known shock that will occur at t+k, k=1..K
    Q: np.ndarray           # (h, h) at the steady state
    sigma: np.ndarray       # standard deviation of the shocks
    solver: str
    iterations: int
    residual: float
    mss_radius: float
    endogenous_switching: bool
    G2: np.ndarray | None = None   # (h, n, nz, nz) 2nd order in z = [states, sigma, shocks]; see order2.py
    pert: object | None = None     # perturbation.Perturbation (orders 2 to 5); see perturbation.py

    @property
    def is_stable(self) -> bool:
        """Mean-square stable with Q evaluated at the steady state.

        With endogenous probabilities there is no established MSS criterion (Maih, 2015, §5.3);
        the value is informative only.
        """
        return self.mss_radius < 1.0

    def _names(self, variables):
        names = [v for v in self.model.endogenous if not v.startswith("AUX_")] if variables is None else list(variables)
        return names, [self.model.endogenous.index(v) for v in names]

    def irf(self, shock: str | None = None, periods: int = 40, regime: int | None = None,
            size: float | None = None, variables=None, anticipated: int = 0) -> pd.DataFrame:
        """IRF conditional on staying in the regime (1-based). Without ``regime``, all regimes.

        One-standard-deviation shock by default. Index: (shock, regime, period).
        ``anticipated=k``: the shock is announced in period 0 and occurs in period k (requires the
        model to have been solved with ``shock_horizon >= k``).
        """
        if anticipated > self.R_ant.shape[0]:
            raise ValueError(f"solve with shock_horizon >= {anticipated} to use anticipated={anticipated}")
        names, idx = self._names(variables)
        shocks = self.model.exogenous if shock is None else [shock]
        regs = range(self.T.shape[0]) if regime is None else [regime - 1]
        rows, keys = [], []
        for s in shocks:
            k = self.model.exogenous.index(s)
            amp = self.sigma[k] if size is None else size
            for r in regs:
                x = np.zeros(self.T.shape[1])
                for t in range(periods):
                    lead = anticipated - t          # periods left until the shock occurs
                    x = self.T[r] @ x
                    if lead == 0:
                        x = x + self.R[r][:, k] * amp
                    elif lead > 0:
                        x = x + self.R_ant[lead - 1][r][:, k] * amp
                    rows.append(x[idx])
                    keys.append((s, r + 1, t))
        return pd.DataFrame(rows, columns=names,
                            index=pd.MultiIndex.from_tuples(keys, names=["shock", "regime", "period"]))

    def simulate(self, periods: int = 200, seed: int = 0, regime0: int = 1, burn: int = 0,
                 regimes=None, variables=None, order: int | None = None, **options) -> pd.DataFrame:
        """Simulates the model in levels, drawing regimes (or following ``regimes``, 1-based).

        With endogenous probabilities, the transition from t to t+1 uses the simulated variables.
        If the model was solved to 2nd order, the simulation is 2nd order with pruning, unless
        ``order=1``.
        """
        if self.pert is not None and order != 1:
            # orders 2 to 5 (perturbation.py): pruning validated against RISE; extra options in **options
            from .perturbation import simulate as _simulate_high
            return _simulate_high(self, periods=periods, seed=seed, regime0=regime0, burn=burn, regimes=regimes,
                                  variables=variables, order=order, **options)
        m, rng = self.model, np.random.default_rng(seed)
        names, idx = self._names(variables)
        n_e, total = len(m.exogenous), periods + burn
        second = self.G2 is not None and order != 1
        if order == 2 and self.G2 is None:
            raise ValueError("solve with order=2 to simulate at 2nd order")
        st = np.flatnonzero(np.abs(self.T).sum((0, 1)) > 0)
        k = st.size
        r = regime0 - 1
        y_prev = self.ss[r].copy()
        d1 = np.zeros(self.T.shape[1])     # 1st-order component (pruning)
        d2 = np.zeros(self.T.shape[1])     # 2nd-order component
        out, path = np.zeros((total, len(idx))), np.zeros(total, int)
        for t in range(total):
            if regimes is not None:
                r = int(regimes[min(t, len(regimes) - 1)]) - 1
            eta = rng.standard_normal(n_e)
            e = eta * self.sigma
            if second:
                # pruning (Kim et al., 2008; Andreasen et al., 2018): the quadratic term uses only the 1st order
                z = np.concatenate([d1[st], [1.0], eta])
                d2 = self.T[r] @ d2 + 0.5 * self.G2[r] @ z @ z
                d1 = self.c[r] + self.T[r] @ d1 + self.R[r] @ e
                y = self.ss[r] + d1 + d2
            else:
                y = self.ss[r] + self.c[r] + self.T[r] @ (y_prev - self.ss[r]) + self.R[r] @ e
            out[t], path[t] = y[idx], r + 1
            if regimes is None and self.T.shape[0] > 1:
                row = self.Q[r]
                if self.endogenous_switching:
                    # leads inside the probability are evaluated at the current value
                    row = np.clip(m.transition_matrix(y, y, y_prev, self.params[r], self.ss[r], regime=r), 0, 1)
                    row = row / row.sum()
                r_new = int(rng.choice(len(row), p=row))
                if second and r_new != r:
                    # deviations are now measured relative to the steady state of the new regime
                    d1 = d1 + (self.ss[r] - self.ss[r_new])
                r = r_new
            y_prev = y
        df = pd.DataFrame(out[burn:], columns=names)
        df.insert(0, "regime", path[burn:])
        return df


    def variance_decomposition(self, **options):
        from .analysis import variance_decomposition
        return variance_decomposition(self, **options)

    def theoretical_autocovariances(self, **options):
        from .analysis import theoretical_autocovariances
        return theoretical_autocovariances(self, **options)

    def historical_decomposition(self, filtration, **options):
        from .analysis import historical_decomposition
        return historical_decomposition(self, filtration, **options)

    def forecast(self, filtration, **options):
        from .analysis import forecast
        return forecast(self, filtration, **options)

    def girf(self, shock, **options):
        from .analysis import girf
        return girf(self, shock, **options)

    def theoretical_autocorrelations(self, **options):
        from .analysis import theoretical_autocorrelations
        return theoretical_autocorrelations(self, **options)

    def density_forecast(self, origin, **options):
        """Simulation-based forecast, with bands and conditions (see forecasting.forecast)."""
        from .forecasting import forecast
        return forecast(self, origin, **options)

    def counterfactual(self, filtration, **options):
        from .forecasting import counterfactual
        return counterfactual(self, filtration, **options)

    def print_solution(self, **options):
        from .display import print_solution
        return print_solution(self, **options)

    def filter(self, data, **options):
        """Filters the data and returns the likelihood and regime probabilities (see filtering.run_filter)."""
        from .filtering import run_filter
        return run_filter(self, data, **options)


def solve_first_order(model, params: np.ndarray, solver: str = "mfi", guess: str | np.ndarray = "backward",
                      tol: float = 1e-12, maxiter: int = 5000, seed=None, shock_stderr: dict | None = None,
                      steady_state: str | None = None, shock_horizon: int = 0, order: int = 1,
                      log_vars=None) -> Solution:
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver: {solver!r}; options: {SOLVERS}")
    if (getattr(model.spec, "planner", None) or {}).get("commitment") is not None:
        # optimal policy (commitment, discretion or loose commitment): see optimal_policy.py
        from .optimal_policy import solve_optimal_policy
        return solve_optimal_policy(model, params, tol=tol, maxiter=maxiter, shock_stderr=shock_stderr,
                                    shock_horizon=shock_horizon, order=order)
    if steady_state is None:   # block attribute: steady_state_model(unique)
        steady_state = "unique" if ("unique" in model.spec.steady_state_options
                                    or getattr(model.steady_state_file, "unique", False)) else "regime"
    if steady_state == "regime":
        ss = model.steady_state(params)
    elif steady_state == "unique":
        # single point: steady state with the parameters at their ergodic mean (Maih, 2015, §3.1)
        from .filtering import ergodic_distribution
        pbar = ergodic_distribution(model.exogenous_transition_matrix(params)) @ params
        if model._ss_params or model.steady_state_file is not None:
            # parameters computed in the steady state take the single-point values in every regime
            pnew = model.steady_state_point(pbar)[1]
            cols = ([model.parameters.index(v) for v in model._ss_params] if model.steady_state_file is None
                    else np.flatnonzero(pnew != pbar))
            params = params.copy()
            params[:, cols] = pnew[cols]
            pbar = pnew
        ss = np.tile(model.steady_state(np.tile(pbar, (model.h, 1)))[0], (model.h, 1))
    else:
        raise ValueError("steady_state must be 'regime' or 'unique'")
    sys = model.first_order_system(params, ss)
    T0 = initial_guess(sys, guess, seed) if isinstance(guess, str) else np.array(guess, float)
    if solver == "mfi":
        T, its = _mfi(sys, T0, tol, maxiter)
    else:
        try:
            T, its = _mnk(sys, T0, tol, min(maxiter, 100))
        except SolveError:
            if not isinstance(guess, str) or guess == "random":
                raise
            # Newton may stall when the guess comes from a singular matrix; a few functional
            # iterations give a better starting point
            Tw = T0
            for _ in range(10):
                Tw = -_lsolve(_U(sys, Tw), sys.Am)
            if not np.all(np.isfinite(Tw)):
                raise
            T, its = _mnk(sys, Tw, tol, min(maxiter, 100))
            its += 10
    U = _U(sys, T)
    R = -np.linalg.solve(U, sys.B)                                   # eq. 15
    # anticipated shocks (eq. 16): R_k[r] = -U_r^{-1} sum_r' Q[r,r'] Ap_r R_{k-1}[r']
    R_ant = np.zeros((shock_horizon,) + R.shape)
    prev = R
    for k in range(shock_horizon):
        prev = -np.linalg.solve(U, np.einsum("rs,rij,sjk->rik", sys.Q, sys.Ap, prev))
        R_ant[k] = prev
    # constant (eq. 13): U_r c_r + sum_r' Q[r,r'] Ap_r c_r' = -resid_r; zero at the ss of the regime itself
    h, n = model.h, model.n
    c = np.zeros((h, n))
    if np.max(np.abs(sys.resid)) > 1e-10:
        big = np.zeros((h * n, h * n))
        for r in range(h):
            for q in range(h):
                big[r * n:(r + 1) * n, q * n:(q + 1) * n] = sys.Q[r, q] * sys.Ap[r] + (U[r] if r == q else 0)
        c = np.linalg.solve(big, -sys.resid.ravel()).reshape(h, n)
    sd = {**model.spec.shock_stderr, **(shock_stderr or {})}
    sig = np.array([float(sd.get(s, 1.0)) for s in model.exogenous])
    res_T, radius = residual(sys, T), mss_spectral_radius(T, sys.Q)
    logv = list(model.spec.log_vars if log_vars is None else log_vars)
    if logv:
        # log approximation (RISE solve_log_approx_vars): for x in log_vars the solution now
        # describes log(x); T and R are rescaled by the level of the approximation point and ss becomes log(ss)
        if order != 1:
            raise NotImplementedError("log_vars is only implemented at 1st order")
        idx = [model.endogenous.index(v) for v in logv]
        idx += [k for k, (a, d) in enumerate(model._aux, start=len(model.spec.endogenous))
                if str(next(iter(d.free_symbols))).split("__")[0] in [model.endogenous[i] for i in idx]]
        if np.any(ss[:, idx] <= 0):
            raise SolveError(f"log_vars requires a positive steady state: {[model.endogenous[i] for i in idx if np.any(ss[:, i] <= 0)]}")
        D = np.ones((h, n))
        D[:, idx] = ss[:, idx]
        T = T * D[:, None, :] / D[:, :, None]
        R = R / D[:, :, None]
        R_ant = R_ant / D[None, :, :, None]
        c = c / D
        ss = ss.copy()
        ss[:, idx] = np.log(ss[:, idx])
    sol = Solution(
        model=model, params=params, ss=ss, c=c, T=T, R=R, R_ant=R_ant, Q=sys.Q, sigma=sig, solver=solver,
        iterations=its, residual=res_T, mss_radius=radius,
        endogenous_switching=model.endogenous_switching,
    )
    if order == 2 and (np.any(c) or not model.endogenous_switching):
        from .order2 import solve_second_order
        sol.G2 = solve_second_order(sol)
    if 2 <= order <= 5 and not np.any(c):
        # general order p (includes endogenous probabilities); at exogenous 2nd order G2 comes from order2.py
        from .perturbation import solve_perturbation
        sol.pert = solve_perturbation(sol, order)
        if sol.G2 is None:
            sol.G2 = sol.pert.tensor(2)
    elif order not in (1, 2):
        raise NotImplementedError("orders 3 to 5 require a regime-specific steady state (no constant); the maximum order is 5")
    return sol


def solve_alternatives(model, params: np.ndarray, starts: int = 100, seed: int = 0, stable_only: bool = False,
                       atol: float = 1e-6, **options) -> list[Solution]:
    """Searches for distinct 1st-order solutions from random guesses (like RISE's ``solve_alternatives``).

    The algorithms return one solution per guess; there is no guarantee of finding them all. More than
    one mean-square-stable solution indicates indeterminacy among MSV solutions.
    """
    found: list[Solution] = []
    trials = [("backward", None), ("zeros", None)] + [("random", s) for s in np.random.SeedSequence(seed).generate_state(starts)]
    for guess, sd in trials:
        for solver in (("mfi", "mnk") if guess != "random" else ("mnk",)):
            try:
                sol = solve_first_order(model, params, solver=solver, guess=guess, seed=sd,
                                        **{"maxiter": 100 if solver == "mnk" else 5000, **options})
            except (SolveError, np.linalg.LinAlgError):
                continue
            if sol.residual > 1e-8 or (stable_only and not sol.is_stable):
                continue
            if not any(np.allclose(sol.T, f.T, atol=atol) for f in found):
                found.append(sol)
    return sorted(found, key=lambda x: x.mss_radius)
