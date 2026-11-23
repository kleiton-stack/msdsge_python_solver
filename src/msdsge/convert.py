"""Converter from RISE model files (.rs) to the language of this package.

Covers: declarations (with quoted descriptions), ``parameters(chain,n)``, ``chain_tp_i_j``
probabilities (parameters) and endogenous ones (``!`` lines), definitions (``#`` lines),
``steady_state_model``, ``parameterization`` (calibration, bounds, quantile priors) and the
macro language (``@#include``, ``@#if``/``@#elseif``/``@#else``/``@#end``, ``@#for``, ``@{v}``).

Whatever is not converted raises :class:`ConversionError` with the source line, instead of being
silently dropped.
"""
from __future__ import annotations

import re
from pathlib import Path

_ID = r"[A-Za-z_][A-Za-z_0-9]*"
_BLOCKS = ("endogenous", "var", "exogenous", "varexo", "parameters", "observables", "varobs", "model",
           "steady_state_model", "parameterization", "planner_objective", "parameter_restrictions",
           "log_vars", "level_variables", "exogenous_definition", "end")
_UNSUPPORTED = ("level_variables", "exogenous_definition")
_DISTS = {"beta_pdf": "beta_pdf", "gamma_pdf": "gamma_pdf", "gam_pdf": "gamma_pdf", "normal_pdf": "normal_pdf",
          "inv_gamma_pdf": "inv_gamma_pdf", "uniform_pdf": "uniform_pdf", "lognormal_pdf": "lognormal_pdf",
          "weibull_pdf": "weibull_pdf",
          **{k: k for k in ("cauchy_pdf", "laplace_pdf", "logistic_pdf", "pareto_pdf", "left_triang_pdf",
                            "right_triang_pdf", "truncated_normal_pdf", "exponential_pdf")}}


class ConversionError(ValueError):
    pass


# ------------------------------------------------------------------ macros
def _cond(expr: str, flags: dict) -> bool:
    py = expr.replace("&&", " and ").replace("||", " or ").replace("~=", "!=")
    py = re.sub(r"~\s*", " not ", py)
    py = re.sub(r"\btrue\b", "True", py)
    py = re.sub(r"\bfalse\b", "False", py)
    env = {"strcmp": lambda a, b: a == b, "isempty": lambda a: not a, "__builtins__": {}}
    try:
        return bool(eval(py, env, dict(flags)))  # macro expressions from the model file itself
    except Exception as err:
        raise ConversionError(f"macro condition could not be evaluated: {expr!r} ({err}); provide 'flags'") from None


def _macros(lines: list[str], flags: dict, base: Path) -> list[str]:
    out: list[str] = []
    i = 0

    def block_end(start: int) -> int:
        depth = 0
        for j in range(start, len(lines)):
            m = re.match(r"\s*@#\s*(\w+)", lines[j])
            if not m:
                continue
            if m.group(1) in ("if", "for"):
                depth += 1
            elif m.group(1) in ("end", "endif", "endfor"):
                depth -= 1
                if depth == 0:
                    return j
        raise ConversionError("macro block without '@#end'")

    while i < len(lines):
        line = lines[i]
        m = re.match(r"\s*@#\s*(\w+)\s*(.*)", line)
        if not m:
            out.append(line)
            i += 1
            continue
        kw, arg = m.group(1), m.group(2).strip()
        if kw == "include":
            inc = base / arg.strip("\"' ")
            if not inc.exists():
                raise ConversionError(f"included file not found: {inc}")
            out += _macros(inc.read_text(encoding="utf-8", errors="replace").splitlines(), flags, inc.parent)
            i += 1
        elif kw == "if":
            end = block_end(i)
            # branches at the level of this if
            branches, depth, cur, cond = [], 0, [], arg
            for j in range(i + 1, end):
                mm = re.match(r"\s*@#\s*(\w+)\s*(.*)", lines[j])
                if mm and mm.group(1) in ("if", "for"):
                    depth += 1
                elif mm and mm.group(1) in ("end", "endif", "endfor"):
                    depth -= 1
                elif mm and depth == 0 and mm.group(1) in ("elseif", "else"):
                    branches.append((cond, cur))
                    cond, cur = (mm.group(2).strip() if mm.group(1) == "elseif" else None), []
                    continue
                cur.append(lines[j])
            branches.append((cond, cur))
            for c, body in branches:
                if c is None or _cond(c, flags):
                    out += _macros(body, flags, base)
                    break
            i = end + 1
        elif kw == "for":
            end = block_end(i)
            mm = re.fullmatch(rf"({_ID})\s+in\s+(.+)", arg)
            if not mm:
                raise ConversionError(f"unrecognised macro loop: {line.strip()!r}")
            var, src = mm.group(1), mm.group(2).strip()
            if r := re.fullmatch(r"(\d+)\s*:\s*(\d+)", src):
                values = list(range(int(r.group(1)), int(r.group(2)) + 1))
            elif src in flags:
                values = list(flags[src])
            else:
                raise ConversionError(f"loop list '{src}' is not defined; provide it in 'flags'")
            for v in values:
                body = [ln.replace("@{" + var + "}", str(v)) for ln in lines[i + 1:end]]
                out += _macros(body, {**flags, var: v}, base)
            i = end + 1
        else:
            raise ConversionError(f"unsupported macro directive: @#{kw}")
    return out


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/|%\{.*?%\}", lambda m: "\n" * m.group().count("\n"), text, flags=re.S)
    return "\n".join(re.sub(r"(//|%).*", "", ln) for ln in text.splitlines())


def _fix_expr(e: str) -> str:
    e = re.sub(rf"({_ID})\s*\{{\s*([+-]?\s*\d+)\s*\}}", lambda m: f"{m.group(1)}({int(m.group(2).replace(' ', '')):+d})", e)
    e = re.sub(rf"({_ID})\s*\{{\s*stst\s*\}}", r"steady_state(\1)", e)
    e = re.sub(rf"\$\(\s*({_ID})\s*\)", r"steady_state(\1)", e)
    return " ".join(e.split())


# ------------------------------------------------------------------ conversion
def rs_to_mod(path: str | Path, flags: dict | None = None, parameters: dict | None = None) -> str:
    """Converts a RISE .rs file and returns the text in the language of this package.

    ``flags``: macro-language variables (``rise_flags``). ``parameters``: values set in the
    MATLAB script with ``set(m,'parameters',p)``; keys ``name`` or ``name(chain,state)`` (also
    ``name_chain_state``, the form of RISE structs). They are appended after the calibration in the file.
    """
    path = Path(path)
    lines = _macros(path.read_text(encoding="utf-8", errors="replace").splitlines(), dict(flags or {}), path.parent)
    text = _strip_comments("\n".join(lines))

    # split into blocks by the keyword at the start of a statement
    # recent RISE versions also accept the "@" prefix in block names (@endogenous, @model)
    pat = re.compile(rf"(?m)^\s*@?({'|'.join(_BLOCKS)})\b(\s*\([^)\n]*\))?")
    marks = list(pat.finditer(text))
    if text[: marks[0].start() if marks else len(text)].strip():
        raise ConversionError("unrecognised text before the first block")
    blocks = []
    for k, m in enumerate(marks):
        body = text[m.end(): marks[k + 1].start() if k + 1 < len(marks) else len(text)]
        blocks.append((m.group(1), (m.group(2) or "").strip(), body))

    endo, exo, obs, const_pars = [], [], [], []
    chains: dict[str, int] = {}
    switching: dict[str, str] = {}
    eqs, defs, endo_tp, ssm, calib, estim = [], [], {}, [], [], []
    occ: list[str] = []   # "?" lines of the model block
    restr, logv, ss_opts, linear = [], [], [], False
    names = lambda body: [n for n in re.split(r"[\s,;]+", re.sub(r'"[^"]*"', " ", body)) if n]

    planner: list[str] = []
    for kw, arg, body in blocks:
        if kw in _UNSUPPORTED:
            raise ConversionError(f"block '{kw}' is not converted yet")
        if kw in ("endogenous", "var"):
            endo += names(body)
        elif kw in ("exogenous", "varexo"):
            exo += names(body)
        elif kw in ("observables", "varobs"):
            obs += names(body)
        elif kw == "log_vars":
            logv += names(body)
        elif kw == "parameter_restrictions":
            for stmt in re.split(r"[;\n]", body):
                if stmt.strip():
                    if not re.fullmatch(r"[^<>=]+(>=|<=|>|<)[^<>=]+", stmt.strip()):
                        raise ConversionError(f"unrecognised parameter restriction (inequalities only): {stmt.strip()!r}")
                    restr.append(" ".join(stmt.split()))
        elif kw == "parameters":
            if arg:
                m = re.fullmatch(rf"\(\s*({_ID})\s*,\s*(\d+)\s*\)", arg)
                if not m:
                    raise ConversionError(f"unrecognised declaration: parameters{arg}")
                chains[m.group(1)] = int(m.group(2))
                for n in names(body):
                    switching[n] = m.group(1)
            else:
                const_pars += names(body)
        elif kw == "model":
            if arg and arg.strip("() ") not in ("", "linear"):
                raise ConversionError(f"unsupported model attribute: {arg}")
            linear = linear or arg.strip("() ") == "linear"
            for stmt in body.split(";"):
                s = stmt.strip()
                if not s:
                    continue
                if s.startswith("?"):   # occasionally binding constraint: goes to the occbin_constraints block
                    if not re.fullmatch(r"\?[^<>]+(>=|<=)[^<>=]+", s, flags=re.S):
                        raise ConversionError(f"unrecognised constraint (use '? expression >= value'): {s!r}")
                    occ.append(_fix_expr(s[1:]))
                    continue
                if s.startswith("!"):
                    m = re.fullmatch(rf"!\s*({_ID})_tp_(\d+)_(\d+)\s*=\s*(.+)", s, flags=re.S)
                    if not m:
                        raise ConversionError(f"unrecognised endogenous probability: {s!r}")
                    endo_tp[(m.group(1), int(m.group(2)), int(m.group(3)))] = _fix_expr(m.group(4))
                elif s.startswith("#"):
                    defs.append(_fix_expr(s[1:]))
                else:
                    eqs.append(_fix_expr(s))
        elif kw == "planner_objective":
            # planner_objective{discount=..,commitment=..} expr;  ->  planner_objective(...); expr; end;
            m = re.fullmatch(r"\s*(?:\{([^}]*)\})?([^;]+);\s*", body, flags=re.S)
            if not m or planner:
                raise ConversionError(f"unrecognised planner_objective block: {body.strip()!r}")
            popts = ", ".join(" ".join(o.split()) for o in (m.group(1) or "").split(",") if o.strip())
            planner = [f"planner_objective({popts});" if popts else "planner_objective;", f"  {_fix_expr(m.group(2))};", "end;"]
        elif kw == "steady_state_model":
            for o in re.split(r"[\s,]+", arg.strip("() ")):
                if o and o not in ("imposed", "unique", "loop"):
                    raise ConversionError(f"unrecognised steady_state_model attribute: {o!r}")
                if o and o not in ss_opts:
                    ss_opts.append(o)
            ssm += [_fix_expr(s) for s in body.split(";") if s.strip()]
        elif kw == "parameterization":
            for stmt in body.split(";"):
                if stmt.strip():
                    c, e = _parse_param_line(stmt)
                    calib.append(c)
                    if e:
                        estim.append(e)

    # transition probabilities declared as parameters
    tp_pars = {}
    for n in const_pars:
        if m := re.fullmatch(rf"({_ID}?)_tp_(\d+)_(\d+)", n):
            chain = m.group(1)
            tp_pars[n] = (chain, int(m.group(2)), int(m.group(3)))
            chains[chain] = max(chains.get(chain, 0), int(m.group(2)), int(m.group(3)))
    for (chain, i, j) in endo_tp:
        chains[chain] = max(chains.get(chain, 0), i, j)
    for k, p in enumerate(ssm):
        lhs = p.split("=")[0].strip()
        if re.fullmatch(r"xx_ssmdef_\d+", lhs):      # RISE auxiliary constant
            ssm[k] = "# " + p
        elif lhs not in endo and lhs not in const_pars and lhs not in switching:
            raise ConversionError(f"steady_state_model assigns to '{lhs}', which is not an endogenous variable, "
                                  "a parameter or an auxiliary constant xx_ssmdef_N")
    for n, v in (parameters or {}).items():
        if n not in const_pars and n not in switching:
            # form of RISE structs: name_chain_state
            m = re.fullmatch(rf"({_ID})_({_ID})_(\d+)", n)
            cands = [(p_, c_) for p_, c_ in switching.items() if m and n == f"{p_}_{c_}_{m.group(3)}"]
            if not cands:
                raise ConversionError(f"'parameters': '{n}' is not a model parameter")
            n = f"{cands[0][0]}({cands[0][1]},{m.group(3)})"
        calib.append(f"{n} = {v!r}" if not isinstance(v, str) else f"{n} = {v}")

    out = [f"// Converted from {path.name} (RISE) by msdsge.convert.rs_to_mod", f"var {' '.join(endo)};"]
    if exo:
        out.append(f"varexo {' '.join(exo)};")
    if obs:
        out.append(f"varobs {' '.join(obs)};")
    if logv:
        out.append(f"log_vars {' '.join(logv)};")
    if const_pars:
        out.append(f"parameters {' '.join(const_pars)};")
    # RISE numbers the composite regimes with the chains in alphabetical order (the first one varies
    # most slowly); declaring them in that order preserves the regime numbering
    chains = dict(sorted(chains.items()))
    for c, n in chains.items():
        out.append(f"markov_chain {c}(regimes={n});")
        sw = [p for p, ch in switching.items() if ch == c]
        if sw:
            out.append(f"parameters({c}) {' '.join(sw)};")
    out.append("")
    out += [c + ";" for c in calib]
    for c in chains:
        out += ["", f"transition_probabilities({c});"]
        for n, (ch, i, j) in tp_pars.items():
            if ch == c:
                out.append(f"  p({i},{j}) = {n};")
        for (ch, i, j), ex in endo_tp.items():
            if ch == c:
                out.append(f"  p({i},{j}) = {ex};")
        out.append("end;")
    out += ["", "model(linear);" if linear else "model;"] + [f"  # {d};" for d in defs] + [f"  {e};" for e in eqs] + ["end;"]
    if occ:
        out += ["", "occbin_constraints;"] + [f"  {c};" for c in occ] + ["end;"]
    if ssm:
        out += ["", f"steady_state_model({', '.join(ss_opts)});" if ss_opts else "steady_state_model;"]
        out += [f"  {s};" for s in ssm] + ["end;"]
    if restr:
        out += ["", "parameter_restrictions;"] + [f"  {s};" for s in restr] + ["end;"]
    if planner:
        out += [""] + planner
    if estim:
        out += ["", "estimated_params;"] + [f"  {e};" for e in estim] + ["end;"]
    return "\n".join(out) + "\n"


def _parse_param_line(stmt: str) -> tuple[str, str | None]:
    """Returns (calibration line, estimated_params line or None)."""
    parts, depth, cur = [], 0, ""
    for ch in stmt.strip():
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    name, rest = parts[0], [p for p in parts[1:]]
    if not rest or not rest[0]:
        raise ConversionError(f"parameterization line without a value: {stmt.strip()!r}")
    calib = f"{name} = {rest[0]}"
    if len(rest) == 1:
        return calib, None
    if len(rest) == 3:
        return calib, f"{name}, {rest[0]}, {rest[1]}, {rest[2]}"
    if len(rest) in (4, 5, 6, 7):
        m = re.fullmatch(rf"({_ID})\s*(?:\(\s*([^)]+)\s*\))?", rest[3])
        if not m or m.group(1) not in _DISTS:
            raise ConversionError(f"unsupported distribution in: {stmt.strip()!r}")
        dist, prob = _DISTS[m.group(1)], m.group(2)
        extra = rest[4:]
        if prob is None and extra and re.fullmatch(r"[0-9.eE+-]+", extra[0]) and len(rest) in (5, 7):
            prob, extra = extra[0], extra[1:]            # old form: dist_pdf, prob
        bounds = f", {extra[0]}, {extra[1]}" if len(extra) == 2 else ""
        if prob is None:   # mean and std
            return calib, f"{name}, {rest[0]}{bounds or ', , '}, {dist}, {rest[1]}, {rest[2]}"
        return calib, f"{name}, {rest[0]}{bounds or ', , '}, {dist}({prob}), {rest[1]}, {rest[2]}"
    raise ConversionError(f"unrecognised parameterization line: {stmt.strip()!r}")
