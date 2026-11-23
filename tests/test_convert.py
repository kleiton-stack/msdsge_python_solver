from pathlib import Path

import numpy as np
import pytest

from msdsge import Model, load
from msdsge.convert import ConversionError, rs_to_mod

ROOT = Path(__file__).parents[1]
RS = ROOT / "validation" / "matlab"


@pytest.mark.parametrize("case", ["A", "B"])
def test_converted_rs_gives_same_solution_as_handwritten_mod(case):
    conv = Model.from_text(rs_to_mod(RS / f"ref{case}.rs")).solve()
    hand = load(ROOT / "examples" / f"ref{case}.mod").solve()
    assert conv.T == pytest.approx(hand.T, abs=1e-12) and conv.R == pytest.approx(hand.R, abs=1e-12)
    assert conv.Q == pytest.approx(hand.Q) and conv.ss == pytest.approx(hand.ss)
    assert conv.endogenous_switching == hand.endogenous_switching


def test_macros_definitions_priors(tmp_path):
    (tmp_path / "inc.rs").write_text("parameters sig\nparameterization\n sig, 0.3;\n")
    (tmp_path / "m.rs").write_text('''
endogenous X, "Output", Z_A Z_B
exogenous E
parameters rho, a_tp_1_2 a_tp_2_1
parameters(a,2) k
@#include "inc.rs"
model
  # twice = 2*rho;
  X = twice/2*X{-1} + sig*E;   % comment
@#for c in Names
  Z_@{c} = k*X{+1};
@#end
@#if ~skip
parameterization
  rho, 0.5, 0.2, 0.8, beta_pdf(.90);
  k(a,1), 1; k(a,2), 2, 0, 5;
  a_tp_1_2, 0.1; a_tp_2_1, 1-.8;
@#end
''')
    text = rs_to_mod(tmp_path / "m.rs", flags={"Names": ["A", "B"], "skip": False})
    m = Model.from_text(text)
    assert m.endogenous == ["X", "Z_A", "Z_B"] and m.h == 2
    s = m.solve()
    assert s.T[0, 0, 0] == pytest.approx(0.5) and s.Q[1, 0] == pytest.approx(0.2)
    est = {e.label: e for e in m.spec.estimated}
    assert est["rho"].prior == "beta" and est["rho"].prob == pytest.approx(0.9)
    assert (est["k_a_2"].lower, est["k_a_2"].upper) == (0, 5)
    with pytest.raises(ConversionError, match="flags"):
        rs_to_mod(tmp_path / "m.rs")
