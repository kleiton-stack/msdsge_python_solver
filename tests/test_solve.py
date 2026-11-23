from pathlib import Path

import numpy as np
import pytest

from msdsge import Model, load
from msdsge.solve import SolveError, mss_spectral_radius, solve_first_order

EX = Path(__file__).parents[1] / "examples"

# FWZ (2011) parameterisations used in RISE's howto.m: 1, 2 and 4 stable solutions
# (Maih, 2015, Table 2)
FWZ = {
    1: dict(delta={1: 0, 2: 0}, rho={1: .9, 2: .9}, phi={1: .5, 2: .8}, p12=1 - .8, p21=1 - .9),
    2: dict(delta={1: -.7, 2: .4}, rho={1: 0, 2: 0}, phi={1: .5, 2: .8}, p12=0.0, p21=1 - .64),
    3: dict(delta={1: -.7, 2: -.2}, rho={1: 0, 2: 0}, phi={1: .2, 2: .4}, p12=1 - .9, p21=1 - .8),
}


def fwz(k):
    c = FWZ[k]
    text = (EX / "fwz2011.mod").read_text().replace("p(1,2) = 1-0.8", f"p(1,2) = {c['p12']}")
    text = text.replace("p(2,1) = 1-0.9", f"p(2,1) = {c['p21']}")
    m = Model.from_text(text)
    return m, m.parameter_values({k2: c[k2] for k2 in ("delta", "rho", "phi")})


def test_backward_looking_scalar():
    m = Model.from_text("var x; varexo e; parameters a; a = 0.7; model; x = a*x(-1) + e; end; shocks; var e; stderr 2; end;")
    s = m.solve()
    assert s.T[0, 0, 0] == pytest.approx(0.7) and s.R[0, 0, 0] == pytest.approx(1.0)
    assert s.mss_radius == pytest.approx(0.49)
    irf = s.irf(periods=3)
    assert irf["x"].tolist() == pytest.approx([2.0, 1.4, 0.98])


def test_forward_looking_analytic():
    # x = E x(+1)/phi + z, z AR(1): x = z / (1 - rho/phi)
    m = Model.from_text("""var x z; varexo e; parameters phi rho; phi = 1.5; rho = 0.8;
        model; x = x(+1)/phi + z; z = rho*z(-1) + e; end;""")
    for solver in ("mfi", "mnk"):
        s = m.solve(solver=solver)
        assert s.residual < 1e-10
        assert s.R[0, 0, 0] == pytest.approx(1 / (1 - 0.8 / 1.5))
        assert s.T[0, 0, 1] == pytest.approx(0.8 / (1 - 0.8 / 1.5))


def test_steady_state_nonlinear_and_auxiliaries():
    m = Model.from_text("""var k c; varexo e; parameters alpha beta delta;
        alpha = 0.33; beta = 0.99; delta = 0.025;
        model;
          1/c = beta/c(+1)*(alpha*k^(alpha-1) + 1 - delta);
          k = exp(e)*k(-1)^alpha + (1-delta)*k(-1) - c + 0.001*(k(-2) - k(-1));
        end;
        initval; k = 10; c = 1; end;""")
    assert "AUX_k_LAG_1" in m.endogenous
    ss = m.steady_state()
    k = ((1 / 0.99 - 1 + 0.025) / 0.33) ** (1 / (0.33 - 1))
    assert ss[0, 0] == pytest.approx(k) and ss[0, 2] == pytest.approx(k)
    assert ss[0, 1] == pytest.approx(k ** 0.33 - 0.025 * k)
    s = m.solve()
    assert s.residual < 1e-9 and s.is_stable


def test_mss_scalar_known_case():
    # x_{t+1} = a_{s_{t+1}} x_t, iid regimes with prob 1/2: MSS iff (a1^2 + a2^2)/2 < 1
    T = np.array([[[1.2]], [[0.5]]])
    Q = np.full((2, 2), 0.5)
    assert mss_spectral_radius(T, Q) == pytest.approx((1.44 + 0.25) / 2)
    # an explosive regime that is not absorbing can coexist with MSS; one that is too persistent cannot
    assert mss_spectral_radius(T, np.array([[.9, .1], [.1, .9]])) > 1


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("solver", ["mfi", "mnk"])
def test_fwz_backward_start(k, solver):
    m, p = fwz(k)
    try:
        s = solve_first_order(m, p, solver=solver)
    except SolveError:
        # Maih (2015, Table 2): functional iteration may fail to find a solution that exists
        assert solver == "mfi"
        return
    assert s.residual < 1e-9


@pytest.mark.parametrize("k, expected", [(1, 1), (2, 2), (3, 4)])
def test_fwz_number_of_stable_solutions(k, expected):
    m, p = fwz(k)
    found = []
    for seed in range(300):
        try:
            s = solve_first_order(m, p, solver="mnk", guess="random", seed=seed, maxiter=200)
        except SolveError:
            continue
        if s.residual < 1e-9 and s.is_stable and not any(np.allclose(s.T, t, atol=1e-6) for t in found):
            found.append(s.T)
    assert len(found) == expected


def test_endogenous_example_solves_and_simulates():
    s = load(EX / "nk_ms.mod").solve()
    assert s.endogenous_switching and s.residual < 1e-10
    assert s.Q[0] == pytest.approx([0.95, 0.05])
    assert s.Q[1, 0] == pytest.approx(1 / (1 + np.exp(2)))
    sim = s.simulate(periods=500, seed=1)
    assert set(sim["regime"]) == {1, 2} and np.isfinite(sim.to_numpy()).all()
