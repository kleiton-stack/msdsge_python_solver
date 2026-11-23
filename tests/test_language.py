"""Model language: parameter restrictions, steady_state_model with parameters and attributes,
Python steady-state file, lagged shocks, observed exogenous variables and measurement errors."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msdsge import Model
from msdsge.estimation import EstimationProblem
from msdsge.model import SteadyStateError
from msdsge.parser import ModelSyntaxError

sys.path.insert(0, str(Path(__file__).parents[1] / "validation"))

NK = """
var y pi i; varexo e_m; varobs i;
parameters beta kappa rho p12 p21;
markov_chain pol(regimes=2);
parameters(pol) phi;
beta = 0.99; kappa = 0.1; rho = 0.8; p12 = 0.05; p21 = 0.1;
phi(pol,1) = 1.5; phi(pol,2) = 0.9;
transition_probabilities(pol); p(1,2) = p12; p(2,1) = p21; end;
model;
  y = y(+1) - (i - pi(+1));
  pi = beta*pi(+1) + kappa*y;
  i = rho*i(-1) + (1-rho)*phi*pi + e_m;
end;
shocks; var e_m; stderr 0.25; end;
estimated_params;
  phi(pol,1), 1.5, 0, 5, gamma_pdf, 1.5, 0.5;
  phi(pol,2), 0.9, 0, 5, gamma_pdf, 1.0, 0.5;
  rho, 0.8, 0, 0.99, beta_pdf, 0.7, 0.1;
end;
parameter_restrictions;
  phi(pol,1) >= phi(pol,2);
  rho < 0.95;
end;
"""


def test_parameter_restrictions_enter_the_prior():
    m = Model.from_text(NK)
    assert m.spec.restriction_text == ["phi(pol,1) >= phi(pol,2)", "rho < 0.95"]
    assert m.check_restrictions()
    assert not m.check_restrictions(m.parameter_values({"phi": {1: 0.5, 2: 0.9}}))
    assert not m.check_restrictions(m.parameter_values({"rho": 0.96}))
    data = m.solve().simulate(periods=60, seed=1)[["i"]]
    prob = EstimationProblem(m, data)
    assert np.isfinite(prob.logprior(np.array([1.5, 0.9, 0.8])))
    assert prob.logprior(np.array([0.8, 0.9, 0.8])) == -np.inf      # regimes swapped
    assert prob.logpost(np.array([1.5, 0.9, 0.96])) == -np.inf      # rho >= 0.95
    assert np.isfinite(prob.logpost(np.array([1.5, 0.9, 0.8])))


def test_parameter_restrictions_reject_variables_and_bad_regimes():
    with pytest.raises(ModelSyntaxError):
        Model.from_text(NK.replace("rho < 0.95;", "y < 0.95;"))
    with pytest.raises(ModelSyntaxError):
        Model.from_text(NK.replace("phi(pol,1) >= phi(pol,2);", "phi(pol,1) >= phi(pol,3);"))


SS = """
var c k z kgap; varexo e; parameters alpha beta delta rho kss sig;
alpha = 0.33; beta = 0.99; delta = 0.025; rho = 0.9; sig = 0.01;
model;
  1/c = beta/c(+1)*(alpha*z(+1)*k^(alpha-1) + 1 - delta);
  c + k = z*k(-1)^alpha + (1-delta)*k(-1);
  log(z) = rho*log(z(-1)) + sig*e;
  kgap = k/kss;
end;
steady_state_model%s;
  z = 1;
  # rk = 1/beta - 1 + delta;
  kss = (alpha/rk)^(1/(1-alpha));
  k = kss;
  kgap = 1;%s
end;
"""
C_LINE = "\n  c = k^alpha - delta*k;"


def test_steady_state_model_computes_parameters_and_auxiliaries():
    m = Model.from_text(SS % ("", C_LINE))
    p = m.parameter_values()
    kss = (0.33 / (1 / 0.99 - 1 + 0.025)) ** (1 / 0.67)
    assert p[0, m.parameters.index("kss")] == pytest.approx(kss)
    ss = m.steady_state(p)[0]
    assert ss[m.endogenous.index("k")] == pytest.approx(kss)
    assert m.solve().is_stable
    # the computed parameter follows the parameters it depends on
    p2 = m.parameter_values({"beta": 0.98})
    assert p2[0, m.parameters.index("kss")] == pytest.approx((0.33 / (1 / 0.98 - 1 + 0.025)) ** (1 / 0.67))


def test_steady_state_attributes_imposed_and_loop():
    kss = (0.33 / (1 / 0.99 - 1 + 0.025)) ** (1 / 0.67)
    css = kss ** 0.33 - 0.025 * kss
    # loop: c is not given; only c is solved for, keeping the equations of the block
    m = Model.from_text(SS % ("(loop)", ""))
    assert m.spec.steady_state_options == {"loop"}
    ss = m.steady_state()[0]
    assert ss[m.endogenous.index("c")] == pytest.approx(css, abs=1e-9)
    assert ss[m.endogenous.index("k")] == pytest.approx(kss)
    # imposed: the wrong point is accepted unchecked; without the attribute it is corrected by the numerical solver
    wrong = SS % ("(imposed)", "\n  c = 1.01*(k^alpha - delta*k);")
    m = Model.from_text(wrong)
    assert m.steady_state()[0][m.endogenous.index("c")] == pytest.approx(1.01 * css)
    m = Model.from_text(wrong.replace("(imposed)", ""))
    assert m.steady_state()[0][m.endogenous.index("c")] == pytest.approx(css, abs=1e-8)
    # the solve option takes precedence over the block attribute
    m.steady_state_imposed = True
    assert m.steady_state()[0][m.endogenous.index("c")] == pytest.approx(1.01 * css)
    with pytest.raises(ModelSyntaxError):
        Model.from_text(SS % ("(whatever)", C_LINE))


def test_steady_state_file_as_python_function():
    base = (SS % ("", C_LINE)).split("steady_state_model")[0]

    def ssfile(model, p):
        rk = 1 / p["beta"] - 1 + p["delta"]
        k = (p["alpha"] / rk) ** (1 / (1 - p["alpha"]))
        return {"z": 1.0, "k": k, "c": k ** p["alpha"] - p["delta"] * k, "kgap": 1.0, "kss": k}   # kss is a new parameter

    m = Model.from_text(base)
    with pytest.raises(ValueError):
        m.parameter_values()                       # kss has no value
    s = m.solve(steady_state_file=ssfile)
    ref = Model.from_text(SS % ("", C_LINE)).solve()
    assert s.ss == pytest.approx(ref.ss)
    assert s.T == pytest.approx(ref.T)
    assert s.params == pytest.approx(ref.params)

    def bad(model, p):
        return {"does_not_exist": 1.0}
    with pytest.raises(SteadyStateError):
        Model.from_text(base).solve(steady_state_file=bad)


LAG = """
var y x%s; varexo e u; parameters rho a b;
rho = 0.5; a = 0.3; b = 0.2;
model;
  x = rho*x(-1) + e;
  y = a*y(+1) + x + %s;
end;
"""


def test_shocks_with_lags_and_leads():
    s = Model.from_text(LAG % ("", "b*e(-1) + 0.1*u(-2) + 0.3*e(+1)")).solve()
    # same thing with the auxiliaries written by hand; E_t e(+1) = 0 without anticipated shocks
    hand = LAG % (" el ul ul2", "b*el(-1) + 0.1*ul2(-1)") + ""
    hand = hand.replace("end;", "  el = e; ul = u; ul2 = ul(-1);\nend;")
    s2 = Model.from_text(hand).solve()
    a = s.irf(periods=8)[["y", "x"]].to_numpy()
    assert a == pytest.approx(s2.irf(periods=8)[["y", "x"]].to_numpy(), abs=1e-12)
    assert list(s.irf(periods=2).columns) == ["y", "x"]          # auxiliaries do not show up
    # with an anticipated shock, the lead of the shock now has an effect
    s3 = Model.from_text(LAG % ("", "0.3*e(+1)")).solve(shock_horizon=1)
    s4 = Model.from_text(LAG % ("", "0")).solve(shock_horizon=1)
    y3 = s3.irf("e", periods=3, anticipated=1)["y"].to_numpy()
    y4 = s4.irf("e", periods=3, anticipated=1)["y"].to_numpy()
    assert y3[0] - y4[0] == pytest.approx(0.3, abs=1e-12)


OBSX = """
var y u; varexo e trend; varobs y trend; parameters rho g sig;
rho = 0.7; g = 0.02; sig = 0.5;
model;
  u = rho*u(-1) + sig*e;
  y = g*trend + u;
end;
"""


def test_observed_exogenous_and_measurement_errors():
    rng = np.random.default_rng(0)
    T = 80
    u = np.zeros(T)
    for t in range(1, T):
        u[t] = 0.7 * u[t - 1] + 0.5 * rng.standard_normal()
    data = pd.DataFrame({"y": 0.02 * np.arange(1, T + 1) + u, "trend": np.arange(1.0, T + 1)})
    f = Model.from_text(OBSX).solve().filter(data)
    # equivalent: remove the trend from the data and filter the AR(1)
    ar = OBSX.replace("varexo e trend; varobs y trend;", "varexo e; varobs y;").replace("g*trend + ", "0*g + ")
    f0 = Model.from_text(ar).solve().filter(pd.DataFrame({"y": u}))
    assert f.loglik_t == pytest.approx(f0.loglik_t, abs=1e-10)
    for flt in ("imm", "kim"):
        assert Model.from_text(OBSX).solve().filter(data, filter=flt).loglik_t[1:] == pytest.approx(
            Model.from_text(ar).solve().filter(pd.DataFrame({"y": u}), filter=flt).loglik_t[1:], abs=1e-10)
    # measurement error declared as parameter stderr_<observable> (RISE convention)
    me = ar.replace("parameters rho g sig;", "parameters rho g sig stderr_y;").replace("sig = 0.5;", "sig = 0.5; stderr_y = 0.3;")
    f1 = Model.from_text(me).solve().filter(pd.DataFrame({"y": u}))
    f2 = Model.from_text(ar).solve().filter(pd.DataFrame({"y": u}), measurement_errors={"y": 0.3})
    assert f1.loglik == pytest.approx(f2.loglik, abs=1e-12)
    assert abs(f1.loglik - f0.loglik) > 1e-3
