"""Parser for the model language (Dynare style, with extensions for Markov chains).

Produces a :class:`ModelSpec` with the equations already in SymPy. Dated variables become symbols
indexed by the time shift, accessible through :func:`timed`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import sympy as sp


class ModelSyntaxError(ValueError):
    pass


def timed(name: str, shift: int = 0) -> sp.Symbol:
    """Symbol for ``name`` in period t+shift (same object for the same arguments)."""
    if shift == 0:
        return sp.Symbol(name, real=True)
    tag = f"F{shift}" if shift > 0 else f"L{-shift}"
    return sp.Symbol(f"{name}__{tag}", real=True)


_SS = sp.Function("steady_state")


@dataclass
class MarkovChain:
    name: str
    regimes: int
    # (i, j) -> expression for the probability of moving from i to j (i != j), 1-based
    transitions: dict[tuple[int, int], sp.Expr] = field(default_factory=dict)

    def is_endogenous(self, parameters) -> bool:
        """True if any probability depends on model variables (not only on parameters)."""
        pars = {timed(n) for n in parameters}
        return any(e.free_symbols - pars for e in self.transitions.values())


@dataclass
class EstimatedParam:
    """Line of the ``estimated_params`` block (same field order as Dynare)."""
    name: str                      # parameter or shock
    kind: str = "param"            # "param" or "stderr"
    regime: tuple[str, int] | None = None   # (chain, state) for a switching parameter
    init: float | None = None
    lower: float = -float("inf")
    upper: float = float("inf")
    prior: str | None = None       # beta, gamma, normal, inv_gamma, inv_gamma2, uniform
    mean: float | None = None      # mean; or lower quantile when ``prob`` is given
    std: float | None = None       # standard deviation; or upper quantile when ``prob`` is given
    prob: float | None = None      # mass between the two quantiles (RISE form: beta_pdf(0.90))

    @property
    def label(self) -> str:
        base = f"stderr_{self.name}" if self.kind == "stderr" else self.name
        return f"{base}_{self.regime[0]}_{self.regime[1]}" if self.regime else base


_PRIORS = {"beta_pdf": "beta", "gamma_pdf": "gamma", "normal_pdf": "normal", "inv_gamma_pdf": "inv_gamma",
           "inv_gamma1_pdf": "inv_gamma", "inv_gamma2_pdf": "inv_gamma2", "uniform_pdf": "uniform", "lognormal_pdf": "lognormal",
           "weibull_pdf": "weibull",
           # additional shapes from rise.def (priors_extra.py)
           "cauchy_pdf": "cauchy", "laplace_pdf": "laplace", "logistic_pdf": "logistic", "pareto_pdf": "pareto",
           "left_triang_pdf": "left_triang", "right_triang_pdf": "right_triang",
           "truncated_normal_pdf": "truncated_normal", "exponential_pdf": "exponential"}


@dataclass
class ModelSpec:
    endogenous: list[str] = field(default_factory=list)
    exogenous: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    observables: list[str] = field(default_factory=list)
    chains: dict[str, MarkovChain] = field(default_factory=dict)
    # parameter -> chain that governs it (absent = constant)
    switching: dict[str, str] = field(default_factory=dict)
    # constant parameter -> expression; switching parameter -> {regime: expression}
    calibration: dict[str, sp.Expr | dict[int, sp.Expr]] = field(default_factory=dict)
    equations: list[sp.Expr] = field(default_factory=list)  # lhs - rhs
    equation_text: list[str] = field(default_factory=list)
    steady_state: list[tuple[str, sp.Expr]] = field(default_factory=list)
    shock_stderr: dict[str, sp.Expr] = field(default_factory=dict)
    initval: dict[str, sp.Expr] = field(default_factory=dict)  # guess for the steady state
    estimated: list["EstimatedParam"] = field(default_factory=list)
    # occasionally binding constraints: (chain or None, g with g >= 0 in the reference regime, text); see occbin.py
    constraints: list[tuple[str | None, sp.Expr, str]] = field(default_factory=list)
    max_lead: int = 0
    max_lag: int = 0
    # inequality restrictions on parameters: (expression g, strict?), satisfied if g >= 0 (or g > 0)
    restrictions: list[tuple[sp.Expr, bool]] = field(default_factory=list)
    restriction_text: list[str] = field(default_factory=list)
    # attributes of steady_state_model(...): subset of {"imposed", "unique", "loop"}
    steady_state_options: set[str] = field(default_factory=set)
    log_vars: list[str] = field(default_factory=list)      # variables approximated in logs
    linear: bool = False                                    # model(linear)
    # planner_objective: {"expr", "discount", "commitment"} (expressions or None); see optimal_policy.py
    planner: dict | None = None


def regime_symbol(name: str, chain: str, state: int) -> sp.Symbol:
    """Symbol for ``name(chain,state)`` used in the parameter restrictions."""
    return sp.Symbol(f"{name}__R__{chain}__{state}", real=True)


_BLOCKS = ("model", "steady_state_model", "initval", "shocks", "estimated_params", "transition_probabilities",
           "parameter_restrictions")
_BLOCKS += ("occbin_constraints",)   # occasionally binding constraints (occbin.py)
_BLOCKS += ("planner_objective",)    # planner objective (optimal_policy.py)
_DECLS = ("var", "varexo", "parameters", "varobs", "markov_chain", "log_vars")
_IDENT = r"[A-Za-z_][A-Za-z_0-9]*"


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group().count("\n"), text, flags=re.S)
    return re.sub(r"(//|%).*", "", text)


def _statements(text: str):
    """Yields (line, statement), splitting on ';'."""
    line = 1
    for raw in text.split(";"):
        stmt = raw.strip()
        if stmt:
            yield line + raw[: len(raw) - len(raw.lstrip())].count("\n"), stmt
        line += raw.count("\n")


class _Parser:
    def __init__(self, text: str):
        self.spec = ModelSpec()
        self.local: dict[str, sp.Expr] = {}   # model-local variables (# name = expr)
        self.stmts = list(_statements(_strip_comments(text)))

    # ---------- expressions ----------
    def _expr(self, src: str, line: int, *, allow_timing: bool = True, regime_refs: bool = False) -> sp.Expr:
        spec = self.spec
        dated = set(spec.endogenous) | set(spec.exogenous)
        shifts: list[int] = []

        def repl(m: re.Match) -> str:
            name, k = m.group(1), int(m.group(2).replace(" ", ""))
            if name not in dated:
                return m.group(0)
            if not allow_timing:
                raise ModelSyntaxError(f"line {line}: lag/lead not allowed here: {m.group(0)}")
            shifts.append(k)
            return timed(name, k).name

        # lead/lag: x(+1), x(-1) and also x(1), as RISE and Dynare accept
        src = re.sub(rf"(?<![\w.])({_IDENT})\s*\(\s*([+-]?\s*\d+)\s*\)", repl, src)
        src = src.replace("^", "**")
        names = spec.endogenous + spec.exogenous + spec.parameters
        local = {n: timed(n) for n in names}
        if regime_refs:   # name(chain,state) in parameter_restrictions
            def ref(m: re.Match) -> str:
                name, chain, k = m.group(1), m.group(2), int(m.group(3))
                if spec.switching.get(name) != chain or not 1 <= k <= spec.chains[chain].regimes:
                    raise ModelSyntaxError(f"line {line}: invalid reference to a switching parameter: {m.group(0)}")
                sym = regime_symbol(name, chain, k)
                local[sym.name] = sym
                return sym.name
            src = re.sub(rf"(?<![\w.])({_IDENT})\s*\(\s*({_IDENT})\s*,\s*(\d+)\s*\)", ref, src)
        for n in dated:
            for k in set(shifts):
                local[timed(n, k).name] = timed(n, k)
        local.update(self.local)
        funcs = dict(
            exp=sp.exp, log=sp.log, ln=sp.log, sqrt=sp.sqrt, abs=sp.Abs, max=sp.Max, min=sp.Min,
            sin=sp.sin, cos=sp.cos, tan=sp.tan, erf=sp.erf, steady_state=_SS, log10=lambda x: sp.log(x, 10),
            sinh=sp.sinh, cosh=sp.cosh, tanh=sp.tanh, asin=sp.asin, acos=sp.acos, atan=sp.atan, sign=sp.sign,
            normcdf=lambda x, m=0, sd=1: (1 + sp.erf((x - m) / (sd * sp.sqrt(2)))) / 2,
            normpdf=lambda x, m=0, sd=1: sp.exp(-((x - m) / sd) ** 2 / 2) / (sd * sp.sqrt(2 * sp.pi)),
        )
        # replace each name by a neutral alias: model names may clash with Python reserved
        # words (del, lambda, in) or with SymPy objects (beta, gamma, pi, E, I)
        alias: dict[str, object] = {}

        def rename(m: re.Match) -> str:
            ident = m.group(0)
            call = src[m.end():].lstrip().startswith("(")
            if ident in local and not (call and ident in funcs and ident not in names):
                key = f"_s{len(alias)}_"
                alias[key] = local[ident]
                return key
            if ident in funcs:
                key = f"_f{len(alias)}_"
                alias[key] = funcs[ident]
                return key
            raise ModelSyntaxError(f"line {line}: undeclared symbol: '{ident}'")

        safe = re.sub(rf"(?<![\w.]){_IDENT}", rename, src)
        try:
            expr = sp.sympify(safe, locals=alias)
        except (sp.SympifyError, SyntaxError, TypeError) as err:
            raise ModelSyntaxError(f"line {line}: invalid expression: {src!r} ({err})") from None
        if shifts:
            spec.max_lead = max(spec.max_lead, max(shifts))
            spec.max_lag = max(spec.max_lag, -min(shifts))
        return expr

    def _equation(self, src: str, line: int) -> sp.Expr:
        parts = src.split("=")
        if len(parts) > 2:
            raise ModelSyntaxError(f"line {line}: more than one '=' in the equation")
        lhs = self._expr(parts[0], line)
        return lhs - self._expr(parts[1], line) if len(parts) == 2 else lhs

    # ---------- declarations ----------
    def _declare(self, kind: str, rest: str, line: int) -> None:
        spec = self.spec
        if kind == "markov_chain":
            m = re.fullmatch(rf"({_IDENT})\s*\(\s*regimes\s*=\s*(\d+)\s*\)", rest)
            if not m:
                raise ModelSyntaxError(f"line {line}: use 'markov_chain name(regimes=N);'")
            if int(m.group(2)) < 2:
                raise ModelSyntaxError(f"line {line}: a chain needs at least 2 regimes")
            spec.chains[m.group(1)] = MarkovChain(m.group(1), int(m.group(2)))
            return
        chain = None
        if kind == "parameters" and rest.startswith("("):
            m = re.match(rf"\(\s*({_IDENT})\s*\)", rest)
            if not m or m.group(1) not in spec.chains:
                raise ModelSyntaxError(f"line {line}: undeclared chain in 'parameters(...)'")
            chain, rest = m.group(1), rest[m.end():]
        names = [n for n in re.split(r"[\s,]+", rest) if n]
        target = {"var": spec.endogenous, "varexo": spec.exogenous, "log_vars": spec.log_vars,
                  "parameters": spec.parameters, "varobs": spec.observables}[kind]
        taken = set(spec.endogenous) | set(spec.exogenous) | set(spec.parameters) | set(spec.chains)
        for n in names:
            if not re.fullmatch(_IDENT, n):
                raise ModelSyntaxError(f"line {line}: invalid name: '{n}'")
            if kind == "varobs":
                # an observed exogenous variable (e.g. a deterministic trend) enters the filter as data
                if n not in spec.endogenous and n not in spec.exogenous:
                    raise ModelSyntaxError(f"line {line}: observable '{n}' is neither an endogenous nor an exogenous variable")
            elif kind == "log_vars":
                if n not in spec.endogenous:
                    raise ModelSyntaxError(f"line {line}: '{n}' in log_vars is not an endogenous variable")
            elif n in taken:
                raise ModelSyntaxError(f"line {line}: name declared twice: '{n}'")
            target.append(n)
            taken.add(n)
            if chain:
                spec.switching[n] = chain

    def _assign(self, stmt: str, line: int) -> None:
        spec = self.spec
        lhs, rhs = (s.strip() for s in stmt.split("=", 1))
        m = re.fullmatch(rf"({_IDENT})\s*\(\s*({_IDENT})\s*,\s*(\d+)\s*\)", lhs)
        if m:
            name, chain, reg = m.group(1), m.group(2), int(m.group(3))
            if spec.switching.get(name) != chain:
                raise ModelSyntaxError(f"line {line}: '{name}' was not declared in parameters({chain})")
            if not 1 <= reg <= spec.chains[chain].regimes:
                raise ModelSyntaxError(f"line {line}: regime {reg} outside chain '{chain}'")
            spec.calibration.setdefault(name, {})[reg] = self._expr(rhs, line, allow_timing=False)
            return
        if lhs not in spec.parameters:
            raise ModelSyntaxError(f"line {line}: assignment to something that is not a parameter: '{lhs}'")
        value = self._expr(rhs, line, allow_timing=False)
        if lhs in spec.switching:  # same value in every regime
            n = spec.chains[spec.switching[lhs]].regimes
            spec.calibration[lhs] = {r: value for r in range(1, n + 1)}
        else:
            spec.calibration[lhs] = value

    # ---------- blocks ----------
    def _block(self, head: str, body: list[tuple[int, str]], line: int) -> None:
        spec = self.spec
        if head == "model":
            for ln, s in body:
                if s.startswith("#"):   # model-local variable, as in Dynare: # name = expression;
                    lhs, _, rhs = s[1:].partition("=")
                    name = lhs.strip()
                    if not re.fullmatch(_IDENT, name) or not rhs:
                        raise ModelSyntaxError(f"line {ln}: use '# name = expression;'")
                    if name in spec.endogenous + spec.exogenous + spec.parameters or name in self.local:
                        raise ModelSyntaxError(f"line {ln}: name already in use: '{name}'")
                    self.local[name] = self._expr(rhs, ln)
                    continue
                spec.equations.append(self._equation(s, ln))
                spec.equation_text.append(" ".join(s.split()))
        elif head == "steady_state_model":
            saved = dict(self.local)   # block auxiliaries are kept apart from the model-local variables
            for ln, s in body:
                lhs, _, rhs = s.partition("=")
                name = lhs.strip()
                if name.startswith("#"):   # auxiliary constant of the block: # name = expression;
                    name = name[1:].strip()
                    if not re.fullmatch(_IDENT, name) or not rhs or name in spec.endogenous + spec.exogenous + spec.parameters:
                        raise ModelSyntaxError(f"line {ln}: use '# name = expression;' with a name not yet declared")
                    value = self._expr(rhs, ln, allow_timing=False)
                    self.local[name] = sp.Symbol(f"{name}__SSAUX", real=True)
                    spec.steady_state.append((self.local[name].name, value))
                    continue
                # endogenous variable or parameter computed in the steady state
                if name not in spec.endogenous + spec.parameters or not rhs:
                    raise ModelSyntaxError(
                        f"line {ln}: use 'variable = expression;', 'parameter = expression;' or "
                        "'# auxiliary = expression;' in steady_state_model")
                spec.steady_state.append((name, self._expr(rhs, ln, allow_timing=False)))
            self.local = saved
        elif head == "parameter_restrictions":
            for ln, s in body:
                m = re.fullmatch(r"(.+?)(>=|<=|>|<)(.+)", s, flags=re.S)
                if not m:
                    raise ModelSyntaxError(f"line {ln}: use 'expression >= expression;' in parameter_restrictions")
                a = self._expr(m.group(1), ln, allow_timing=False, regime_refs=True)
                b = self._expr(m.group(3), ln, allow_timing=False, regime_refs=True)
                g = a - b if m.group(2) in (">=", ">") else b - a
                bad = [str(x) for x in g.free_symbols if "__R__" not in x.name and x.name not in spec.parameters]
                if bad:
                    raise ModelSyntaxError(f"line {ln}: a restriction may only involve parameters (found {bad})")
                spec.restrictions.append((g, m.group(2) in (">", "<")))
                spec.restriction_text.append(" ".join(s.split()))
        elif head.startswith("planner_objective"):
            # planner_objective(discount=expr, commitment=expr); expression; end;  (see optimal_policy.py)
            if spec.planner is not None or len(body) != 1:
                raise ModelSyntaxError(f"line {line}: use a single 'planner_objective(...); expression; end;'")
            pl = {"expr": self._expr(body[0][1], body[0][0], allow_timing=False), "discount": None, "commitment": None}
            depth, cur, opts = 0, "", []
            for ch in head[len("planner_objective"):].strip()[1:-1]:
                depth += (ch == "(") - (ch == ")")
                if ch == "," and depth == 0:
                    opts.append(cur)
                    cur = ""
                else:
                    cur += ch
            for o in opts + [cur]:
                k, eq, v = o.partition("=")
                if not o.strip():
                    continue
                if k.strip() not in ("discount", "commitment") or not eq or not v.strip():
                    raise ModelSyntaxError(f"line {line}: invalid option in planner_objective: {o.strip()!r}")
                pl[k.strip()] = self._expr(v, line, allow_timing=False)
            spec.planner = pl
        elif head == "initval":
            for ln, s in body:
                lhs, _, rhs = s.partition("=")
                if lhs.strip() not in spec.endogenous or not rhs:
                    raise ModelSyntaxError(f"line {ln}: use 'variable = expression;' in initval")
                spec.initval[lhs.strip()] = self._expr(rhs, ln, allow_timing=False)
        elif head == "estimated_params":
            for ln, s in body:
                spec.estimated.append(self._estimated(s, ln))
        elif head == "occbin_constraints":
            # one constraint per line: "[chain:] expression >= expression" (or "<="); see occbin.py
            for ln, s in body:
                m = re.fullmatch(rf"(?:({_IDENT})\s*:)?\s*([^<>]+)(>=|<=)([^<>=]+)", s, flags=re.S)
                if not m or (m.group(1) and m.group(1) not in spec.chains):
                    raise ModelSyntaxError(f"line {ln}: use '[chain:] expression >= expression;' in occbin_constraints")
                g = self._expr(m.group(2), ln) - self._expr(m.group(4), ln)
                spec.constraints.append((m.group(1), g if m.group(3) == ">=" else -g, " ".join(s.split())))
        elif head == "shocks":
            current = None
            for ln, s in body:
                if m := re.fullmatch(rf"var\s+({_IDENT})", s):
                    current = m.group(1)
                    if current not in spec.exogenous:
                        raise ModelSyntaxError(f"line {ln}: '{current}' is not a declared shock")
                elif (m := re.fullmatch(r"stderr\s+(.+)", s, flags=re.S)) and current:
                    spec.shock_stderr[current] = self._expr(m.group(1), ln, allow_timing=False)
                else:
                    raise ModelSyntaxError(f"line {ln}: unrecognised statement in shocks: {s!r}")
        elif head.startswith("transition_probabilities"):
            m = re.fullmatch(rf"transition_probabilities\s*\(\s*({_IDENT})\s*\)", head)
            if not m or m.group(1) not in spec.chains:
                raise ModelSyntaxError(f"line {line}: use 'transition_probabilities(chain);'")
            chain = spec.chains[m.group(1)]
            for ln, s in body:
                mm = re.fullmatch(r"p\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(.+)", s, flags=re.S)
                if not mm:
                    raise ModelSyntaxError(f"line {ln}: use 'p(i,j) = expression;'")
                i, j = int(mm.group(1)), int(mm.group(2))
                if i == j:
                    raise ModelSyntaxError(f"line {ln}: p({i},{i}) is the complement of the row; do not declare it")
                if not (1 <= i <= chain.regimes and 1 <= j <= chain.regimes):
                    raise ModelSyntaxError(f"line {ln}: regime outside chain '{chain.name}'")
                chain.transitions[(i, j)] = self._expr(mm.group(3), ln)

    def _estimated(self, s: str, line: int) -> EstimatedParam:
        spec = self.spec
        # split on commas outside parentheses
        parts, depth, cur = [], 0, ""
        for ch in s:
            depth += ch == "("
            depth -= ch == ")"
            if ch == "," and depth == 0:
                parts.append(cur.strip())
                cur = ""
            else:
                cur += ch
        parts.append(cur.strip())
        head, rest = parts[0], parts[1:]
        ep = EstimatedParam(name=head)
        if m := re.fullmatch(rf"stderr\s+({_IDENT})", head):
            ep.name, ep.kind = m.group(1), "stderr"
            if ep.name not in spec.exogenous:
                raise ModelSyntaxError(f"line {line}: '{ep.name}' is not a declared shock")
        elif m := re.fullmatch(rf"({_IDENT})\s*\(\s*({_IDENT})\s*,\s*(\d+)\s*\)", head):
            ep.name, ep.regime = m.group(1), (m.group(2), int(m.group(3)))
            if spec.switching.get(ep.name) != ep.regime[0]:
                raise ModelSyntaxError(f"line {line}: '{ep.name}' does not switch with chain '{ep.regime[0]}'")
        elif head not in spec.parameters:
            raise ModelSyntaxError(f"line {line}: '{head}' is not a declared parameter")
        elif head in spec.switching:
            raise ModelSyntaxError(f"line {line}: '{head}' is regime-switching; use '{head}({spec.switching[head]},k)'")
        num = lambda x: float(self._expr(x, line, allow_timing=False)) if x else None
        shape = lambda x: re.fullmatch(r"(\w+)\s*(?:\(\s*([^)]+)\s*\))?", x)
        i = next((k for k, x in enumerate(rest) if shape(x) and shape(x).group(1).lower() in _PRIORS), None)
        pre = rest if i is None else rest[:i]
        if len(pre) > 3:
            raise ModelSyntaxError(f"line {line}: use 'name, initial, lower, upper, prior, mean, std'")
        if pre:
            ep.init = num(pre[0])
        if len(pre) == 3:
            ep.lower = -float("inf") if pre[1] == "" else num(pre[1])
            ep.upper = float("inf") if pre[2] == "" else num(pre[2])
        if i is not None:
            m = shape(rest[i])
            ep.prior = _PRIORS[m.group(1).lower()]
            if len(rest) < i + 3:
                raise ModelSyntaxError(f"line {line}: a prior needs two numbers (mean and std, or quantiles)")
            ep.mean, ep.std = num(rest[i + 1]), num(rest[i + 2])
            if m.group(2):
                ep.prob = num(m.group(2))
                if not 0 < ep.prob < 1 or not ep.mean < ep.std:
                    raise ModelSyntaxError(f"line {line}: a quantile prior requires lower < upper and 0 < prob < 1")
        return ep

    def parse(self) -> ModelSpec:
        it = iter(self.stmts)
        for line, stmt in it:
            word = re.match(_IDENT, stmt)
            key = word.group(0) if word else ""
            if key in _BLOCKS and "=" not in stmt.split("(")[0]:
                first = stmt[len(key):].strip()
                head = key
                if key == "transition_probabilities":
                    m = re.match(r"\s*\([^)]*\)", stmt[len(key):])
                    if not m:
                        raise ModelSyntaxError(f"line {line}: missing chain in transition_probabilities")
                    head, first = key + m.group(0).strip(), stmt[len(key) + m.end():].strip()
                elif key in ("steady_state_model", "model") and re.fullmatch(r"\([\w\s,]*\)", first):
                    opts = {o for o in re.split(r"[\s,]+", first.strip("()")) if o}
                    allowed = {"imposed", "unique", "loop"} if key == "steady_state_model" else {"linear"}
                    if opts - allowed:
                        raise ModelSyntaxError(f"line {line}: unknown attribute in {key}: {sorted(opts - allowed)}")
                    if key == "model":
                        self.spec.linear = "linear" in opts
                    else:
                        self.spec.steady_state_options |= opts
                elif key == "planner_objective" and (not first or (first.startswith("(") and first.endswith(")"))):
                    head = key + first
                elif first:
                    raise ModelSyntaxError(f"line {line}: missing ';' after '{key}'")
                body = []
                for ln, s in it:
                    if s == "end":
                        break
                    body.append((ln, s))
                else:
                    raise ModelSyntaxError(f"line {line}: block '{key}' without 'end;'")
                self._block(head, body, line)
            elif key in _DECLS and not re.match(rf"{key}\s*=", stmt):
                self._declare(key, stmt[len(key):].strip(), line)
            elif "=" in stmt:
                self._assign(stmt, line)
            else:
                raise ModelSyntaxError(f"line {line}: unrecognised statement: {stmt!r}")
        self._check()
        return self.spec

    def _check(self) -> None:
        spec = self.spec
        if spec.planner is not None and spec.planner["commitment"] is not None:
            # optimal policy: the policy rule is dropped from the model, so there are fewer equations than variables
            if not 0 < len(spec.equations) < len(spec.endogenous):
                raise ModelSyntaxError(
                    f"planner_objective with 'commitment' requires fewer equations ({len(spec.equations)}) than "
                    f"endogenous variables ({len(spec.endogenous)}): remove the policy rule")
        elif len(spec.equations) != len(spec.endogenous):
            raise ModelSyntaxError(
                f"{len(spec.equations)} equations for {len(spec.endogenous)} endogenous variables")
        for name, chain in spec.switching.items():
            cal = spec.calibration.get(name)
            if cal is not None and len(cal) != spec.chains[chain].regimes:
                raise ModelSyntaxError(f"parameter '{name}' has no value in every regime of '{chain}'")


def parse(text: str) -> ModelSpec:
    return _Parser(text).parse()


def load(path: str | Path) -> ModelSpec:
    return parse(Path(path).read_text(encoding="utf-8"))
