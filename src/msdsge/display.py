"""Printing of the solution and model checks: ``print_solution``, ``resid`` and ``check``.

``print_solution`` follows the RISE format (one table per regime; rows: steady state, lagged
states, uncertainty term and shocks; columns: variables). The RISE text is that of a MATLAB
``table``; here the content and the order are the same and the spacing is our own (the test
compares after normalising whitespace). ``resid`` prints the residuals of the static equations
like RISE (one row per equation, one column per regime).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fmt(x: float, digits: int = 5) -> str:
    """Number in the style of MATLAB's ``format short g`` (5 significant digits)."""
    if not np.isfinite(x):
        return "NaN" if np.isnan(x) else ("Inf" if x > 0 else "-Inf")
    if x == 0:
        return "0"
    s = f"{x:.{digits}g}"
    if "e" in s:
        mant, exp = s.split("e")
        return f"{mant}e{int(exp):+03d}"
    return s


def regime_label(model, r: int) -> str:
    """Description of regime r (0-based) as in RISE: 'c = 1 & const = 1'."""
    parts = [f"{c} = {k}" for c, k in zip(model.chain_names, model.regimes[r])]
    return " & ".join(parts + ["const = 1"])


def solution_table(solution, regime: int = 1, variables=None, tol: float = 1e-12) -> pd.DataFrame:
    """Solution table of one regime (1-based): rows are the terms, columns the variables.

    Rows, in the RISE order: ``steady state`` (if any value is nonzero in some regime),
    lagged states ``name{-1}``, ``@sig`` (uncertainty constant, if nonzero) and shocks
    (``name{+k}`` for the anticipated ones).
    """
    m = solution.model
    r = regime - 1
    names = [v for v in m.endogenous if not v.startswith("AUX_")] if variables is None else \
        ([variables] if isinstance(variables, str) else list(variables))
    idx = [m.endogenous.index(v) for v in names]
    rows, labels = [], []
    if np.max(np.abs(solution.ss)) > tol:
        rows.append(solution.ss[r][idx])
        labels.append("steady state")
    for k in np.flatnonzero(np.abs(solution.T).sum((0, 1)) > 0):
        rows.append(solution.T[r][idx, k])
        labels.append(f"{m.endogenous[k]}{{-1}}")
    if np.max(np.abs(solution.c)) > tol:
        rows.append(solution.c[r][idx])
        labels.append("@sig")
    for k, e in enumerate(m.exogenous):
        rows.append(solution.R[r][idx, k])
        labels.append(e)
        for a in range(solution.R_ant.shape[0]):
            rows.append(solution.R_ant[a][r][idx, k])
            labels.append(f"{e}{{+{a + 1}}}")
    df = pd.DataFrame(np.array(rows).reshape(len(labels), len(names)), index=labels, columns=names)
    return df.where(df.abs() > tol, 0.0)


def print_solution(solution, variables=None, regimes=None, file=None) -> str:
    """Print the solution by regime in the RISE format and return the text.

    ``variables``: name or list (default: all, without the auxiliary ones). ``regimes``: 1-based list.
    The shock coefficients are the response to a unit shock (like ``Tz`` in RISE).
    """
    m = solution.model
    h = solution.T.shape[0]
    out = ["", "MODEL  SOLUTION #1/1", f"SOLVER :: {solution.solver}", ""]
    for r in (range(1, h + 1) if regimes is None else regimes):
        tab = solution_table(solution, r, variables)
        cells = [[_fmt(v) for v in row] for row in tab.to_numpy()]
        wl = max(len(s) for s in tab.index)
        widths = [max(len(c), 3, *(len(row[j]) for row in cells)) for j, c in enumerate(tab.columns)]
        pad = " " * (4 + wl)
        head = pad + "".join("    " + c.center(w) for c, w in zip(tab.columns, widths))
        rule = pad + "".join("    " + "_" * w for w in widths)
        out.append(f"Regime {r:4d} : {regime_label(m, r - 1)} ")
        out += [head, rule, ""]
        out += ["    " + lab.ljust(wl) + "".join("    " + c.rjust(w) for c, w in zip(row, widths))
                for lab, row in zip(tab.index, cells)]
        out += [rule, "", head, "", ""]
    text = "\n".join(out) + "\n"
    print(text, file=file, end="")
    return text


def resid(model, params: dict | None = None, at: str = "steady_state", show: bool = True, file=None) -> np.ndarray:
    """Residuals of the static equations in each regime. Returns an (n_equations, n_regimes) matrix.

    ``at="steady_state"``: at the computed steady state (should be zero; this is RISE's ``resid``).
    ``at="initval"``: at the values of ``steady_state_model``/``initval`` before any numerical
    solution, to locate the equation that does not hold. With ``show`` it prints like RISE:
    ``equation #k : r_regime1 r_regime2``. The equations of auxiliary variables come last.
    """
    p = model.parameter_values(params)
    if at == "steady_state":
        ss = model.steady_state(p)
    elif at == "initval":
        ss = np.zeros((model.h, model.n))
        for r in range(model.h):
            y = model._initval(p[r])
            for v, fn in model._ss_model:
                y[model.endogenous.index(v)] = fn(y, p[r])
            for a, d in model._aux:
                src = str(next(iter(d.free_symbols))).split("__")[0]
                y[model.endogenous.index(a)] = y[model.endogenous.index(src)]
            ss[r] = y
    else:
        raise ValueError("at must be 'steady_state' or 'initval'")
    with np.errstate(all="ignore"):
        res = np.column_stack([np.asarray(model._fs(ss[r], p[r]), float).ravel() for r in range(model.h)])
    if show:
        lines = [" "] + [f"equation #{k + 1} : " + " ".join(f"{v:.4g}" if v != 0 else "0" for v in row) + "  "
                         for k, row in enumerate(res)] + [""]
        print("\n".join(lines), file=file)
    return res


def check(model, solution=None, params: dict | None = None, show: bool = True, file=None) -> dict:
    """Model counts and classification of the variables; with ``solution``, a stability summary.

    Types (as in Dynare and RISE): ``static`` (neither lag nor lead), ``predetermined``
    (lag only), ``forward`` (lead only) and ``both``. The classification uses the first-order
    derivatives in all regimes. Returns a dictionary; with ``show`` it prints the summary.
    """
    if solution is not None:
        p, ss = solution.params, solution.ss
    else:
        p = model.parameter_values(params)
        ss = model.steady_state(p)
    sys = model.first_order_system(p, ss)
    lag = np.abs(sys.Am).sum((0, 1)) > 0
    lead = np.abs(sys.Ap).sum((0, 1)) > 0
    kinds = {}
    for k, v in enumerate(model.endogenous):
        kinds[v] = "both" if lag[k] and lead[k] else "predetermined" if lag[k] else "forward" if lead[k] else "static"
    counts = {t: sum(1 for x in kinds.values() if x == t) for t in ("static", "predetermined", "forward", "both")}
    info = {
        "endogenous": model.n, "auxiliary": len(model._aux), "exogenous": len(model.exogenous),
        "parameters": len(model.parameters), "equations": len(model._equations),
        "observables": len(model.spec.observables), "regimes": model.h,
        "chains": {c: model.spec.chains[c].regimes for c in model.chain_names},
        "endogenous_switching": model.endogenous_switching, "types": kinds, "counts": counts,
        "max_steady_state_residual": float(np.max(np.abs(sys.resid))),
    }
    if solution is not None:
        info["mss_radius"] = solution.mss_radius
        info["is_stable"] = solution.is_stable
        info["regime_spectral_radius"] = [float(np.max(np.abs(np.linalg.eigvals(T)))) for T in solution.T]
        info["residual"] = solution.residual
    if show:
        lines = [f"Equations: {info['equations']}   Endogenous: {info['endogenous']} (auxiliary: {info['auxiliary']})"
                 f"   Exogenous: {info['exogenous']}   Parameters: {info['parameters']}   Observables: {info['observables']}",
                 f"Regimes: {info['regimes']}" + (f"   Chains: {info['chains']}" if info["chains"] else "")
                 + ("   (endogenous probabilities)" if info["endogenous_switching"] else ""),
                 "Variables by type: " + ", ".join(f"{k} = {v}" for k, v in counts.items()),
                 f"Largest steady-state residual: {info['max_steady_state_residual']:.3g}"]
        if info["equations"] != info["endogenous"]:
            lines.append("WARNING: number of equations differs from the number of endogenous variables")
        if solution is not None:
            lines.append("Spectral radius of T by regime: " + ", ".join(f"{x:.4f}" for x in info["regime_spectral_radius"]))
            lines.append(f"Mean-square stability radius: {info['mss_radius']:.4f} "
                         f"({'stable' if info['is_stable'] else 'unstable'})")
        print("\n".join(lines), file=file)
    return info
