"""HDMR, Monte Carlo filtering and stability map: analytical values and RISE (validation/matlab/ref_sens.m)."""
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from msdsge import Model
from msdsge.sensitivity import (HDMR, MCF, ishigami, ishigami_indices, sample, sobol_g, sobol_g_indices,
                                sobol_indices, stability_map)

ROOT = Path(__file__).parents[1]
REF = ROOT / "validation" / "ref"
B3 = np.tile([-np.pi, np.pi], (3, 1))


@pytest.fixture(scope="module")
def ref():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sio.loadmat(REF / "sens.mat", squeeze_me=True, struct_as_record=False)


def test_sampling_procedures(ref):
    lb, ub = [0, -1, 2], [1, 1, 5]
    for proc in ("sobol", "halton", "latin_hypercube", "uniform"):
        x = sample(lb, ub, 64, proc, seed=1)
        assert x.shape == (64, 3) and np.all(x >= lb) and np.all(x <= ub)
        assert x == pytest.approx(sample(lb, ub, 64, proc, seed=1))
    # Latin hypercube: one point per stratum in each dimension
    u = sample([0], [1], 50, "latin_hypercube", seed=2)[:, 0]
    assert sorted(np.floor(u * 50).astype(int)) == list(range(50))
    # unscrambled Sobol: the first points coincide with RISE's; afterwards the sequences
    # diverge (different order and direction numbers), and RISE's Halton is also different
    mine = sample(lb, ub, 16, "sobol", scramble=False)
    assert mine[:2] == pytest.approx(np.asarray(ref["qmc_sobol"]).T[:2])
    assert np.unique(mine[:, 0]).size == 16
    with pytest.raises(ValueError):
        sample(lb, ub, 4, "nothing")


def test_hdmr_ishigami_analytic():
    an = ishigami_indices()
    h = HDMR(ishigami, B3, names=["a", "b", "c"], expansion_order=2, pol_order=8, n=2048).estimate()
    assert h.f0 == pytest.approx(an["mean"], abs=0.01)
    assert h.D == pytest.approx(an["variance"], rel=0.01)
    for term, val in an["S"].items():
        assert h.sensitivity(*term) == pytest.approx(val, abs=0.005)
    assert h.indices["sensitivity"].sum() == pytest.approx(1.0, abs=0.01)
    assert h.total_effects()["a"] == pytest.approx(an["S"][(0,)] + an["S"][(0, 2)], abs=0.01)
    assert h.r2 > 0.999
    # the metamodel predicts out of sample and the first-order effect of b is a*sin(b)^2 - a/2
    new = sample(B3[:, 0], B3[:, 1], 200, "uniform", seed=5)
    assert h.metamodel(new) == pytest.approx(ishigami(new), abs=0.25)
    fx = h.first_order_effect("b", np.array([-1.0, 0.5, 2.0]))
    assert fx.to_numpy() == pytest.approx(7 * np.sin([-1.0, 0.5, 2.0]) ** 2 - 3.5, abs=0.1)
    # Legendre polynomials and Monte Carlo projection give the same within sampling error
    h2 = HDMR(ishigami, B3, expansion_order=2, pol_order=8, optimal=False, n=2048).estimate("projection")
    assert h2.sensitivity(1) == pytest.approx(an["S"][(1,)], abs=0.01)
    assert h2.sensitivity(0, 2) == pytest.approx(an["S"][(0, 2)], abs=0.04)
    # check against the Saltelli estimator
    sb = sobol_indices(ishigami, B3, n=2048)
    assert sb["first_order"].to_numpy() == pytest.approx([an["S"][(0,)], an["S"][(1,)], 0.0], abs=0.03)


def test_hdmr_sobol_g_analytic():
    an = sobol_g_indices()
    b = np.tile([0.0, 1.0], (8, 1))
    h = HDMR(sobol_g, b, expansion_order=1, pol_order=8, n=4096).estimate()
    first = np.array([an["S"][(i,)] for i in range(8)])
    assert h.indices["sensitivity"].to_numpy() == pytest.approx(first, abs=0.005)
    assert h.D == pytest.approx(an["variance"], rel=0.02)


def test_hdmr_against_rise(ref):
    """Same sample and same options as RISE (order 4, 75% of the sample in the estimation).

    f0 and D are identical; the indices are within 0.012 (RISE's rule for estimating the
    coefficients is not known). With order 8 RISE returns indices larger than one; not compared.
    """
    I = ref["ishigami"]
    theta, f = np.asarray(I.theta).T, np.asarray(I.f)
    assert ishigami(theta) == pytest.approx(f, abs=1e-12)
    for tag, optimal in (("p4_o0", False), ("p4_o1", True)):
        r = getattr(I, tag)
        h = HDMR((f, theta), B3, expansion_order=2, pol_order=4, optimal=optimal,
                 sample_percentage=float(r.sample_percentage)).estimate("projection")
        assert h.N == int(r.Nobs)
        assert h.f0 == pytest.approx(float(r.f0), abs=1e-12)
        assert h.D == pytest.approx(float(r.D), abs=1e-10)
        theirs = np.array([float(a.sensitivity) for a in r.aggregate])
        assert h.indices["sensitivity"].to_numpy() == pytest.approx(theirs, abs=0.025)
        assert h.fit(outofsample=True)["r2"] > 0.7
    assert sum(float(a.sensitivity) for a in I.p8_o1.aggregate) > 5      # the RISE defect with order 8
    G = ref["satelli_sobol95"]
    r = G.p4_o0
    h = HDMR((np.asarray(G.f), np.asarray(G.theta).T), np.tile([0.0, 1.0], (8, 1)), expansion_order=2,
             pol_order=4, optimal=False, sample_percentage=0.75).estimate("projection")
    assert sobol_g(np.asarray(G.theta).T) == pytest.approx(np.asarray(G.f), abs=1e-12)
    assert (h.f0, h.D) == pytest.approx((float(r.f0), float(r.D)), abs=1e-10)
    theirs = np.array([float(a.sensitivity) for a in r.aggregate[:8]])
    assert h.indices["sensitivity"].to_numpy()[:8] == pytest.approx(theirs, abs=0.01)


def test_mcf_smirnov_and_correlations():
    # behaviour: x0 > 0.5 and x1 + x2 < 1; x3 is irrelevant
    mc = MCF(lambda x: (x[0] > 0.5 and x[1] + x[2] < 1, x.sum()), 2000, [0] * 4, [1] * 4,
             names=["a", "b", "c", "d"], procedure="sobol", seed=1)
    assert mc.share_behaved == pytest.approx(0.25, abs=0.02)
    ks = mc.smirnov()
    assert ks.loc["a", "pvalue"] < 1e-6 and ks.loc["b", "pvalue"] < 1e-6 and ks.loc["d", "pvalue"] > 0.01
    assert ks.loc["a", "statistic"] == pytest.approx(2 / 3, abs=0.05)
    cp = mc.correlation_patterns("behave", cutoff=0.01)
    assert set(zip(cp["param_1"], cp["param_2"])) == {("b", "c")} and cp["correlation"].iloc[0] < -0.3
    assert mc.cdf("a")["behave"].iloc[-1] == 1.0 and len(mc.user_outputs) == 2000
    assert mc.to_frame()["behaved"].sum() == mc.is_behaved.sum()


def test_mcf_solution_region_against_rise(ref):
    """RISE's MonteCarloFiltering example (trinity.rs), on the same 500 samples.

    RISE's 'retcode == 0' requires a stable solution; the region coincides with the stability region here.
    """
    from msdsge.convert import rs_to_mod
    rs = Path(os.environ.get("RISE_TOOLBOX", Path.home() / "RISE_toolbox")) / "examples/MonteCarloFiltering/trinity.rs"
    if not rs.exists():
        pytest.skip("RISE not installed")
    m = Model.from_text(rs_to_mod(rs))
    names = [str(n) for n in ref["mcf_names"]]
    S = np.asarray(ref["mcf_samples"]).T
    behaved = np.asarray(ref["mcf_is_behaved"]).astype(bool).ravel()

    def stable(x):
        try:
            return m.solve(dict(zip(names, x))).is_stable
        except Exception:
            return False
    mc = MCF(stable, S, names=names)
    assert (mc.is_behaved == behaved).mean() >= 0.99
    assert mc.share_behaved == pytest.approx(behaved.mean(), abs=0.01)
    ks = mc.smirnov()
    assert ks["pvalue"].idxmin() in names and ks["pvalue"].min() < 0.01


def test_stability_map():
    m = Model.from_text("""var y pi i; varexo e_m; parameters beta kappa rho phi;
        beta = 0.99; kappa = 0.1; rho = 0.5; phi = 1.5;
        model; y = y(+1) - (i - pi(+1)); pi = beta*pi(+1) + kappa*y; i = rho*i(-1) + (1-rho)*phi*pi + e_m; end;""")
    sm = stability_map(m, {"phi": (0.2, 3.0), "kappa": (0.05, 0.5)}, n=40, seed=1)
    d = sm.draws
    assert set(d["status"]) <= {"stable", "unstable", "fail"} and len(d) == 40
    # Taylor principle: with phi > 1 the iteration finds the stable solution
    assert (d.loc[d["phi"] > 1.05, "status"] == "stable").all()
    assert sm.shares().sum() == pytest.approx(1.0)
    if 0 < sm.mcf.is_behaved.sum() < 40:
        assert sm.smirnov().loc["phi", "pvalue"] < 0.05
    # with alternative guesses: phi < 1 yields more than one stable solution or none, uniqueness not guaranteed
    sm2 = stability_map(m, {"phi": (1.2, 2.5)}, n=6, alternatives=5, seed=2)
    assert set(sm2.draws["status"]) <= {"determinate", "indeterminate", "unstable", "fail"}
    assert (sm2.draws["n_stable"] >= 1).all()
