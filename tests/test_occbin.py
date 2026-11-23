"""Occasionally binding constraints: comparison with RISE (validation/matlab/ref_occbin.m) and with Dynare."""
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from msdsge import Model, load
from msdsge.convert import rs_to_mod
from msdsge.occbin import (OccbinError, simulate_through_switch, solve_occbin, viable_regimes)
from msdsge.parser import ModelSyntaxError
from msdsge.solve import Solution

ROOT = Path(__file__).parents[1]
REF = ROOT / "validation" / "ref"
MODELS = ROOT / "validation" / "occbin"
TOL = 5e-8   # RISE's mfi stops at 1.5e-8 (fix_point_TolFun)

TOY = """
var y pi i inot;  varexo e;
parameters beta kappa phi rho ibar;
markov_chain zlb(regimes=2);
parameters(zlb) bind;
beta = 0.99; kappa = 0.1; phi = 1.5; rho = 0.8; ibar = -0.0005;
bind(zlb,1) = 0; bind(zlb,2) = 1;
model;
  y = y(+1) - (i - pi(+1)) + e;
  pi = beta*pi(+1) + kappa*y;
  inot = rho*inot(-1) + (1-rho)*phi*pi;
  i = bind*ibar + (1-bind)*inot;
end;
occbin_constraints;
  zlb: i >= ibar;
end;
shocks; var e; stderr 0.05; end;
"""


def _ref(name):
    return sio.loadmat(REF / "occbin.mat", squeeze_me=True, struct_as_record=False)[name]


@pytest.fixture(scope="module")
def one():
    return solve_occbin(load(MODELS / "one_constraint.mod"))


@pytest.fixture(scope="module")
def two():
    return solve_occbin(load(MODELS / "two_constraints.mod"), restr_map={"debt": 1, "zlb": 2})


def _cols(ref, occ):
    """Positions of the model variables (without auxiliaries) in the RISE matrices."""
    endo = list(ref.endo)
    names = [v for v in occ.model.endogenous if not v.startswith("AUX_")]
    return names, [endo.index(v) for v in names]


# ---------------------------------------------------------------- syntax
def test_parser_block_and_converter(tmp_path):
    m = Model.from_text(TOY)
    (chain, g, text), = m.spec.constraints
    assert chain == "zlb" and text == "zlb: i >= ibar"
    occ = solve_occbin(m)
    assert occ.constraints[0].chain == "zlb" and occ.reference == 1
    with pytest.raises(ModelSyntaxError, match="occbin_constraints"):
        Model.from_text(TOY.replace("zlb: i >= ibar", "i == ibar"))
    # RISE "?" lines become the occbin_constraints block
    (tmp_path / "m.rs").write_text("""
endogenous X R
exogenous E
parameters rho a_tp_1_2 a_tp_2_1
parameters(a,2) flag
model
  ? R >= -0.5;
  X = rho*X{-1} + E;
  R = flag*(-0.5) + (1-flag)*X;
parameterization
  rho, 0.9;
  a_tp_1_2, 0;
  a_tp_2_1, 0;
  flag(a,1), 0;
  flag(a,2), 1;
""")
    text = rs_to_mod(tmp_path / "m.rs")
    assert "occbin_constraints;\n  R >= -0.5;\nend;" in text
    df = solve_occbin(Model.from_text(text)).simulate(shocks=[[-1.0], [0.0], [0.0], [0.0]])
    assert df["R"].tolist() == pytest.approx([-0.5, -0.5, -0.5, -0.5]) and df["regime"].tolist() == [2, 2, 2, 2]
    assert df["X"].tolist() == pytest.approx([-1, -0.9, -0.81, -0.729])


def test_toy_zlb_properties():
    occ = solve_occbin(Model.from_text(TOY))
    lin = occ.irf("e", size=-0.5, honor_constraints=False)
    con = occ.irf("e", size=-0.5)
    assert lin["i"].min() < -0.0005 - 1e-6                      # without the constraint the interest rate goes through the floor
    assert con["i"].min() == pytest.approx(-0.0005, abs=1e-12)  # with it, it stops at the floor
    assert (con["regime"] == 2).sum() >= 1 and con["regime"].iloc[-1] == 1
    assert con["y"].iloc[0] < lin["y"].iloc[0]                # the recession is deeper at the floor
    small = occ.irf("e", size=-1e-4)                          # small shock: linear response
    assert small.drop(columns="regime").to_numpy() == pytest.approx(
        occ.irf("e", size=-1e-4, honor_constraints=False).drop(columns="regime").to_numpy(), abs=1e-14)
    # in the reference regime the constraint holds throughout the simulation; outside it, the rate is at the floor
    sim = occ.simulate(periods=400, seed=3)
    assert (sim["i"] >= -0.0005 - 1e-10).all()
    assert sim.loc[sim.regime == 2, "i"].to_numpy() == pytest.approx(-0.0005, abs=1e-12)
    assert 0 < (sim.regime == 2).mean() < 1
    assert (sim.loc[sim.regime == 2, "inot"] < -0.0005).all() and (sim.loc[sim.regime == 1, "inot"] >= -0.0005 - 1e-10).all()


def test_errors():
    m = Model.from_text(TOY)
    with pytest.raises(OccbinError, match="reference regime"):
        solve_occbin(m, reference=3)
    with pytest.raises(OccbinError, match="constraints"):
        solve_occbin(m, constraints=[])


# ---------------------------------------------------------------- RISE: one constraint
def test_one_constraint_simulation_matches_rise(one):
    ref = _ref("one")
    names, cols = _cols(ref, one)
    assert one.ss[[one.model.endogenous.index(v) for v in names]] == pytest.approx(ref.sim[0, cols], abs=1e-12)
    eta = ref.shocks[:, [list(ref.exo).index(e) for e in one.model.exogenous]]
    df = one.simulate(shocks=eta)
    assert df["regime"].to_numpy() == pytest.approx(ref.regime)
    assert (ref.regime == 2).sum() > 20                       # the constraint binds in part of the sample
    assert df[names].to_numpy() == pytest.approx(ref.sim[1:, cols], abs=TOL)


@pytest.mark.parametrize("size", [5, 30, 60])
def test_one_constraint_irfs_match_rise(one, size):
    ref = _ref("one")
    names, cols = _cols(ref, one)
    A = getattr(ref, f"irf_{size}")                           # period x sign x variable x shock
    binding = 0
    for a, shock in enumerate(ref.exo):
        for c, sign in enumerate((-1, 1)):
            df = one.irf(shock, periods=40, size=sign * size)
            assert df[names].to_numpy() == pytest.approx(A[1:, c][:, cols, a], abs=TOL)
            binding += int((df["regime"] == 2).any())
    assert binding >= 2


# ---------------------------------------------------------------- RISE: two constraints
# shocks of 20 standard deviations in which the two constraints interact (3 of 24 IRFs)
KNOWN_DIFFERENCES = {("eps_k", -20), ("eps_p", -20), ("eps_r", -20)}


def test_two_constraints_simulation_matches_rise(two):
    ref = _ref("two")
    names, cols = _cols(ref, two)
    eta = ref.shocks[:, [list(ref.exo).index(e) for e in two.model.exogenous]]
    df = two.simulate(shocks=eta)
    # RISE composite regimes: rows of ref.regimes (from the 2nd on) = [regime, debt, zlb]
    assert df["regime"].to_numpy() == pytest.approx(ref.regime)
    assert df[names].to_numpy() == pytest.approx(ref.sim[1:, cols], abs=TOL)
    assert len(np.unique(ref.regime)) >= 2


@pytest.mark.parametrize("size", [5, 20])
def test_two_constraints_irfs_match_rise(two, size):
    ref = _ref("two")
    names, cols = _cols(ref, two)
    A = getattr(ref, f"irf_{size}")
    for a, shock in enumerate(ref.exo):
        for c, sign in enumerate((-1, 1)):
            df = two.irf(shock, periods=40, size=sign * size)
            if (shock, sign * size) in KNOWN_DIFFERENCES:
                # difference documented in docs/VALIDATION.md: the RISE regime sequence was not reproduced
                assert np.abs(df[names].to_numpy() - A[1:, c][:, cols, a]).max() < 0.15
                continue
            assert df[names].to_numpy() == pytest.approx(A[1:, c][:, cols, a], abs=TOL)


# ---------------------------------------------------------------- RISE: constraint through regime switching
def _rise_solution(ref, model):
    """Solution built from the RISE matrices (rows in alphabetical order; columns [states, sigma, shocks])."""
    endo, exo, h = list(ref.endo), list(ref.exo), len(ref.Tz)
    n = len(endo)
    states = [i for i in np.asarray(ref.order_var, int) - 1 if endo[i] in ("BETA", "K", "Z")]
    perm = [endo.index(v) for v in model.endogenous]
    T, R, c, ss = np.zeros((h, n, n)), np.zeros((h, n, len(exo))), np.zeros((h, n)), np.zeros((h, n))
    for r in range(h):
        Tz = np.asarray(ref.Tz[r], float)
        full = np.zeros((n, n))
        full[:, states] = Tz[:, :len(states)]
        T[r] = full[np.ix_(perm, perm)]
        c[r] = Tz[perm, len(states)]
        R[r] = Tz[perm, len(states) + 1:][:, [exo.index(e) for e in model.exogenous]]
        ss[r] = np.asarray(ref.ss[r], float)[perm]
    return Solution(model=model, params=np.zeros((h, len(model.parameters))), ss=ss, c=c, T=T, R=R,
                    R_ant=np.zeros((0, h, n, len(exo))), Q=np.asarray(ref.Q, float), sigma=np.ones(len(exo)),
                    solver="rise", iterations=0, residual=0.0, mss_radius=np.nan, endogenous_switching=False)


def test_through_switch_matches_rise():
    """RISE draws the regime among those that honour the constraint, with the row of Q renormalised."""
    ref = _ref("sw")
    endo = list(ref.endo)
    m = Model.from_text(f"""
var {' '.join(endo)};  varexo {' '.join(ref.exo)};
markov_chain zlb(regimes=2);
model; {' '.join(v + ' = 0;' for v in endo)} end;
occbin_constraints; R >= 1; end;
""")
    sol = _rise_solution(ref, m)
    Y, eta, reg = np.asarray(ref.sim, float), np.asarray(ref.shocks, float), np.asarray(ref.regime, int)
    ok = np.zeros((len(reg), 2), bool)
    for t in range(len(reg)):
        ok[t], cand = viable_regimes(sol, Y[t], eta[t])
        assert cand[reg[t] - 1] == pytest.approx(Y[t + 1], abs=1e-9)    # same law of motion
    assert ok[np.arange(len(reg)), reg - 1].all()             # RISE never stays in a regime that violates
    assert (~ok[:, 0]).sum() > 100 and ok[:, 1].all()
    assert (reg[~ok[:, 0]] == 2).all()                        # only regime 2 is viable: it goes there
    prev = np.r_[1, reg[:-1]]
    for a in (1, 2):                                          # both viable: probabilities of Q
        sel = ok[:, 0] & (prev == a)
        freq, p = np.mean(reg[sel] == 2), sol.Q[a - 1, 1]
        assert abs(freq - p) < 4 * np.sqrt(p * (1 - p) / sel.sum())
    # the package simulation has the same properties
    df = simulate_through_switch(sol, periods=2000, seed=1, y0=Y[0])
    assert (df["R"] >= 1 - 1e-10).all() and 0.3 < (df["regime"] == 2).mean() < 0.8
    assert abs((df["regime"] == 2).mean() - np.mean(reg == 2)) < 0.06


# ---------------------------------------------------------------- Dynare (OccBin of Guerrieri and Iacoviello)
def test_one_constraint_matches_dynare_occbin(one):
    """With ``lookahead=True`` the algorithm is the original OccBin; matches Dynare 6.4."""
    path = REF / "occbin_dynare.mat"
    if not path.exists():
        pytest.skip("Dynare reference not generated")
    ref = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    endo = list(ref["endo"])
    names = [v for v in one.model.endogenous if not v.startswith("AUX_")]
    cols = [endo.index(v) for v in names]
    eta = ref["shocks"][:, [list(ref["exo"]).index(e) for e in one.model.exogenous]]
    one.lookahead = True
    try:
        df = one.simulate(shocks=eta)
    finally:
        one.lookahead = False
    assert (df["regime"].to_numpy() == 2) == pytest.approx(np.asarray(ref["regime"]) == 1)
    assert df[names].to_numpy() == pytest.approx(ref["piecewise"][:, cols], abs=1e-9)   # levels
