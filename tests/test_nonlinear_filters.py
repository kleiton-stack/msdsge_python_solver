"""Sigma-point filters (UKF, CKF, DDF), Riccati shortcut, initialisation options and imm/kim smoother.

RISE references (commit 1b2edfa): validation/matlab/ref_nlf.m, ref_riccati.m and ref_init.m.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from msdsge import Model
from msdsge.convert import rs_to_mod
from msdsge.filter_extras import diffuse_lyapunov
from msdsge.nonlinear_filters import rise_variable_order, sigma_filter_core, sigma_points, sigma_weights

ROOT = Path(__file__).parents[1]
REF = ROOT / "validation" / "ref"
MS = """var x z; varexo e u; varobs x;
markov_chain c(regimes=2);
parameters(c) a sig;
a(c,1) = 0.9; a(c,2) = 0.1; sig(c,1) = 0.2; sig(c,2) = 1.5;
transition_probabilities(c); p(1,2) = 0.05; p(2,1) = 0.10; end;
model; z = a*z(-1) + sig*e; x = z + 0.1*u; end;"""


def _load(name):
    return sio.loadmat(REF / name, squeeze_me=True, struct_as_record=False)


def _solve(name, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Model.from_text(rs_to_mod(ROOT / f"validation/matlab/{name}.rs")).solve(**kw)


def _rise_law(d):
    def f(r, X):   # X: [K, A, C, eta]; RISE's own solution matrices
        ss, T1, T2 = d.ss[r], d.T1[r], d.T2[r].reshape(4, 4, 4)
        z = np.column_stack([X[:, :2] - ss[:2], np.ones(len(X)), X[:, 3]])
        return ss + z @ T1.T + 0.5 * np.einsum("nij,pi,pj->pn", T2, z, z)
    return f


@pytest.mark.parametrize("name", ["o2c", "o2", "o2m"])
@pytest.mark.parametrize("rule", ["ukf", "ckf"])
def test_recursion_matches_rise_with_rise_solution(name, rule):
    """With RISE's solution matrices, the recursion reproduces RISE's increments."""
    d = _load("nlf.mat")[name]
    out = sigma_filter_core(_rise_law(d), np.atleast_2d(d.y).T, np.array([2]), np.zeros((1, 1)), np.array(d.Q, float),
                            d.PAI00, np.stack(d.a), np.stack(d.P), 1, rule)
    ref = np.ravel(getattr(d, rule + "_inc"))
    assert out[0] == pytest.approx(ref, abs=1e-8)
    assert out[1].T == pytest.approx(getattr(d, rule + "_PAItt"), abs=1e-8)


@pytest.mark.parametrize("name,tol", [("o2c", 1e-5), ("o2m", 1e-4)])
@pytest.mark.parametrize("rule", ["ukf", "ckf"])
def test_full_pipeline_matches_rise(name, tol, rule):
    """2nd-order solution of the package + filter: differs from RISE only by the tolerance of RISE's solver."""
    d = _load("nlf.mat")[name]
    s = _solve(name + "f", order=2)
    assert [s.model.endogenous[i] for i in rise_variable_order(s)] == list(d.order_var)
    r = s.filter(pd.read_csv(REF / f"nlf_{name}_data.csv"), filter=rule)
    assert r.loglik == pytest.approx(float(getattr(d, rule + "_ll")), abs=tol * 50)
    assert r.loglik_t == pytest.approx(np.ravel(getattr(d, rule + "_inc")), abs=tol)
    assert r.filtered_states["K"].to_numpy() == pytest.approx(getattr(d, rule + "_upd_K"), abs=1e-3)


def test_sigma_rules():
    for rule, npts in (("ukf", 11), ("ddf", 11), ("ckf", 10)):
        w = sigma_weights(5, rule)
        assert w.size == npts and w.sum() == pytest.approx(1)
    P = np.array([[2, 0.3, 0], [0.3, 1, 0], [0, 0, 0.5]])
    m = np.array([1.0, 2, 3])
    for rule in ("ukf", "ckf", "ddf"):
        X = sigma_points(m, P, rule)
        w = sigma_weights(3, rule)
        assert w @ X == pytest.approx(m)
        assert (X - m).T @ ((X - m) * w[:, None]) == pytest.approx(P)
    # values of RISE's utils.filtering.sigma_points(a, P, 'ukf')
    assert sigma_points(m, P, "ukf")[1] == pytest.approx([3.44948974278318, 2.36742346141748, 3])
    # singular covariance: sqrt(eps) floor on the eigenvalues, as in RISE
    X = sigma_points(np.zeros(3), np.diag([2.0, 1.0, 0.0]), "ukf")
    assert X[3, 2] == pytest.approx(0.000211431983345722, rel=1e-9)


def test_ddf_and_first_order_consistency():
    """DDF (no RISE reference in R2024a): close to the UKF; in a linear solution all three are the "rise" filter."""
    s2 = _solve("o2cf", order=2)
    data = pd.read_csv(REF / "nlf_o2c_data.csv")
    ll = {k: s2.filter(data, filter=k).loglik for k in ("ukf", "ckf", "ddf")}
    assert ll["ddf"] == pytest.approx(ll["ukf"], abs=0.05)
    s1 = _solve("o2cf")
    lin = s1.filter(data).loglik
    for k in ("ukf", "ckf", "ddf"):
        # not identical: the sqrt(eps) floor on the eigenvalues acts because C is observed without error
        assert s1.filter(data, filter=k).loglik == pytest.approx(lin, abs=0.01)
    assert abs(ll["ukf"] - lin) > 0.05


def test_estimation_problem_uses_sigma_point_filter():
    from msdsge.estimation import EstimationProblem
    text = rs_to_mod(ROOT / "validation/matlab/o2cf.rs") + "\nestimated_params;\n  rho, 0.9, 0.5, 0.99;\nend;\n"
    data = pd.read_csv(REF / "nlf_o2c_data.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr = EstimationProblem(Model.from_text(text), data, filter="ukf")
        ll = pr.loglik([0.9])
    assert ll == pytest.approx(float(_load("nlf.mat")["o2c"].ukf_ll), abs=1e-3)


def test_riccati_shortcut_matches_rise():
    """Increments equal to those of RISE over the whole grid of kf_riccati_tol, including the default 1e-6."""
    R = _load("riccati.mat")["ms"]
    s = Model.from_text(MS).solve()
    data = pd.read_csv(REF / "filter_ms_data.csv")[["x"]]
    for i, tol in enumerate(R.tols):
        r = s.filter(data, riccati_tol=float(tol), smooth=False)
        assert r.loglik_t == pytest.approx(R.Incr[:, i], abs=1e-11), tol
    ref = _load("filter.mat")["ms"]          # default RISE reference (kf_riccati_tol=1e-6)
    r = s.filter(data, riccati_tol=1e-6)
    assert r.loglik == pytest.approx(float(ref.LogLik), abs=1e-10)
    assert r.loglik_t == pytest.approx(np.ravel(ref.Incr), abs=1e-11)
    assert abs(s.filter(data).loglik - r.loglik) > 1e-3     # default 0: full recursion


def test_init_options_match_rise():
    d = _load("filter_init.mat")
    for name, dat in (("flt_ur", "ur"), ("flt_urms", "ur"), ("flt_ms", "ms")):
        s = _solve(name)
        data = pd.read_csv(REF / f"filter_{dat}_data.csv")[["x"]]
        cases = {"initvar5": dict(init_variance=5), "initvar5_pre3": dict(init_variance=5, presample=3)}
        if name == "flt_ms":
            cases.update(pre3=dict(presample=3), user_a0=dict(user_init=({"x": 0.3, "z": 0.2},)))
        for k, kw in cases.items():
            q = getattr(d[name], k)
            f = s.filter(data, smooth=False, **kw)
            assert f.loglik == pytest.approx(float(q.LogLik), abs=1e-9), (name, k)
            assert f.loglik_t == pytest.approx(np.ravel(q.Incr), abs=1e-10), (name, k)
        if name != "flt_ms":     # unit root: RISE fails (retcode 23); here, diffuse initialisation
            assert d[name].default.retcode == 23
            assert np.isfinite(s.filter(data, diffuse=10.0).loglik)


def test_diffuse_lyapunov_matches_rise():
    L = _load("filter_init.mat")["diffuse_lyapunov"]
    for i in range(len(L.T)):
        for j, dd in enumerate(L.d):
            if L.rc[i, j] == 0:     # case 3 (Jordan block at the unit root): RISE gives up
                P = diffuse_lyapunov(np.array(L.T[i], float), np.array(L.Q[i], float), float(dd))
                assert P == pytest.approx(L.P[i, j], abs=1e-10)


def test_imm_kim_smoothers_recover_latent_state():
    # larger measurement error (0.5): with 0.1 the state is almost observed and no smoother, not even
    # that of the "rise" filter, improves on the updated state (RMSE 0.0947 to 0.0962 against 0.0937)
    s = Model.from_text(MS.replace("0.1*u", "0.5*u")).solve()
    rm = {k: [] for k in ("rise", "imm", "kim", "imm_upd", "kim_upd")}
    for seed in range(8):
        sim = s.simulate(periods=400, seed=seed)
        for f in ("rise", "imm", "kim"):
            r = s.filter(sim[["x"]], filter=f)
            assert r.smoothed_states is not None and r.smoothed_shocks is not None
            rm[f].append(np.sqrt(np.mean((r.smoothed_states["z"].to_numpy() - sim["z"].to_numpy()) ** 2)))
            if f != "rise":
                rm[f + "_upd"].append(np.sqrt(np.mean((r.filtered_states["z"].to_numpy() - sim["z"].to_numpy()) ** 2)))
    m = {k: float(np.mean(v)) for k, v in rm.items()}
    assert m["imm"] < 0.97 * m["imm_upd"] and m["kim"] < 0.97 * m["kim_upd"]   # smoothing improves
    assert m["imm"] < 1.02 * m["rise"] and m["kim"] < 1.02 * m["rise"]          # at the level of "rise"


def test_smoothers_coincide_without_switching():
    ar = Model.from_text("""var x z; varexo e u; varobs x; parameters a; a = 0.7;
model; z = a*z(-1) + 0.5*e; x = z + 0.2*u; end;""").solve()
    sim = ar.simulate(periods=120, seed=4)
    res = {f: ar.filter(sim[["x"]], filter=f) for f in ("rise", "imm", "kim")}
    for f in ("imm", "kim"):
        assert res[f].smoothed_states.to_numpy() == pytest.approx(res["rise"].smoothed_states.to_numpy(), abs=1e-10)
        assert res[f].smoothed_shocks.to_numpy()[1:] == pytest.approx(res["rise"].smoothed_shocks.to_numpy()[1:], abs=1e-10)
