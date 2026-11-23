from pathlib import Path

import pytest
import sympy as sp

from msdsge import ModelSyntaxError, parse, timed
from msdsge.parser import load

EXAMPLE = Path(__file__).parents[1] / "examples" / "nk_ms.mod"


def test_example_file():
    m = load(EXAMPLE)
    assert m.endogenous == ["y", "pi", "i"]
    assert m.exogenous == ["e_m"]
    assert m.parameters == ["beta", "kappa", "rho", "phi_pi"]
    assert m.switching == {"phi_pi": "pol"}
    assert m.calibration["beta"] == sp.Rational(99, 100) or float(m.calibration["beta"]) == 0.99
    assert {k: float(v) for k, v in m.calibration["phi_pi"].items()} == {1: 1.5, 2: 0.9}
    assert (m.max_lead, m.max_lag) == (1, 1)
    assert len(m.equations) == 3
    # 'pi' and 'beta' are model symbols, not SymPy constants/functions
    assert m.equations[1].free_symbols == {timed("pi"), timed("pi", 1), timed("beta"), timed("kappa"), timed("y")}
    chain = m.chains["pol"]
    assert chain.is_endogenous(m.parameters)
    assert float(chain.transitions[(1, 2)]) == 0.05
    assert chain.transitions[(2, 1)].free_symbols == {timed("pi", -1)}
    assert float(m.shock_stderr["e_m"]) == 0.25


def test_comments_and_power():
    m = parse("""
        var x; varexo e; parameters a;  % comment
        a = 2^3; /* block
        comment */
        model; x = a*x(-1)^2 + e; end;
    """)
    assert float(m.calibration["a"]) == 8
    assert m.equations[0] == timed("x") - timed("a") * timed("x", -1) ** 2 - timed("e")


@pytest.mark.parametrize("src, msg", [
    ("var x; model; x = z; end;", "undeclared"),
    ("var x y; model; x = 1; end;", "equations"),
    ("var x; model; x = 1;", "without 'end;'"),
    ("var x; var x;", "twice"),
    ("var x; markov_chain c(regimes=2); transition_probabilities(c); p(1,1)=0.5; end; model; x=0; end;", "complement"),
    ("var x; parameters a; x = 1;", "not a parameter"),
])
def test_errors(src, msg):
    with pytest.raises(ModelSyntaxError, match=msg):
        parse(src)
