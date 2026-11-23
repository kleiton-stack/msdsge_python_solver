"""Marginal data density estimators: targets with a known constant and comparison with RISE."""
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from msdsge.mdd import ALGORITHMS, MDD, REFERENCES, mcmc_mdd

REF = Path(__file__).parents[1] / "validation" / "ref"
A = np.array([[1.0, 0.5, 0.2], [0.5, 2.0, -0.3], [0.2, -0.3, 0.5]])
MU = np.array([1.0, -2.0, 0.5])
LOGZ = 3.7
_P = np.linalg.inv(A)
_K = -0.5 * np.linalg.slogdet(2 * np.pi * A)[1]
D = np.array([2.0, 0.0, 0.0])


def gauss(x):
    d = x - MU
    return LOGZ + _K - 0.5 * d @ _P @ d


def bimodal(x):
    return LOGZ - 1.5 * np.log(2 * np.pi) + np.log(0.5) + np.logaddexp(-0.5 * np.sum((x - D) ** 2), -0.5 * np.sum((x + D) ** 2))


def _iid(target, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    if target == "gauss":
        x = rng.multivariate_normal(MU, A, n)
        f = gauss
    else:
        x = rng.standard_normal((n, 3)) + np.where(rng.uniform(size=(n, 1)) < 0.5, 1, -1) * D
        f = bimodal
    return x, np.array([f(v) for v in x]), f


def test_nine_rise_algorithms_plus_ce_with_references():
    assert set(ALGORITHMS) == {"mhm", "swz", "mueller", "bridge", "is", "ris", "cj", "laplace", "laplace_mcmc", "ce"}
    assert set(REFERENCES) == set(ALGORITHMS)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_gaussian_target(algorithm):
    x, lp, f = _iid("gauss")
    est = MDD(x, lp, f, mode=MU, hessian=_P, L=4000, seed=1).estimate(algorithm)
    assert est == pytest.approx(LOGZ, abs=0.01 if algorithm == "laplace" else 0.06)


@pytest.mark.parametrize("algorithm", [a for a in ALGORITHMS if not a.startswith("laplace")])
def test_bimodal_target(algorithm):
    x, lp, f = _iid("bimodal")
    est = MDD(x, lp, f, L=4000, seed=1).estimate(algorithm)
    assert est == pytest.approx(LOGZ, abs=0.08)


def test_laplace_sees_only_one_mode_of_the_mixture():
    """Known limit: Laplace at one mode of the mixture gives, up to the overlap, half the mass."""
    x, lp, f = _iid("bimodal")
    md = MDD(x, lp, f, mode=D, hessian=np.eye(3), L=100)
    assert md.laplace() == pytest.approx(LOGZ + np.log(0.5), abs=0.01)
    assert abs(md.laplace_mcmc() - LOGZ) > 0.05


def test_bounded_support():
    """Standard normal truncated to [0, 1]³: the constant is the mass of the box."""
    from scipy import stats
    rng = np.random.default_rng(0)
    x = stats.truncnorm(0, 1).rvs((20000, 3), random_state=rng)
    f = lambda v: -0.5 * v @ v - 1.5 * np.log(2 * np.pi)
    lp = np.array([f(v) for v in x])
    truth = 3 * np.log(stats.norm.cdf(1) - 0.5)
    md = MDD(x, lp, f, lower=np.zeros(3), upper=np.ones(3), L=4000, seed=2)
    for a in ("mhm", "swz", "mueller", "bridge", "is", "ris", "cj", "ce"):
        assert md.estimate(a) == pytest.approx(truth, abs=0.08), a


def test_mcmc_output_with_several_chains_and_interface():
    from msdsge.samplers import mh_sampler
    r = mh_sampler(gauss, MU, A, draws=6000, burn=1000, chains=2, seed=1)
    out = MDD(r.draws, r.logpost, gauss, L=3000).all(["mhm", "ris", "bridge", "is", "mueller", "swz", "cj", "ce", "laplace"])
    assert np.isnan(out.pop("laplace"))                 # no mode and Hessian
    assert all(v == pytest.approx(LOGZ, abs=0.1) for v in out.values()), out
    assert mcmc_mdd(r.draws, r.logpost, algorithm="mhm", tau=[0.5, 0.9]) == pytest.approx(LOGZ, abs=0.1)
    with pytest.raises(ValueError):
        MDD(r.draws, r.logpost).estimate("is")          # needs the kernel
    with pytest.raises(ValueError):
        MDD(r.draws, r.logpost).estimate("nothing")


# ---------------------------------------------------------------- against RISE
def _problem():
    import pandas as pd
    from msdsge import Model
    from msdsge.convert import rs_to_mod
    from msdsge.estimation import EstimationProblem
    m = Model.from_text(rs_to_mod(Path(__file__).parents[1] / "validation" / "matlab" / "est_ms2.rs"))
    return m, pd.read_csv(REF / "filter_ms_data.csv")


def test_posterior_at_rise_mode_and_laplace_formula():
    """At the RISE mode (estimate with kf_riccati_tol=0): same posterior, and the same Laplace MDD
    when the Hessian stored by RISE is used."""
    from msdsge.estimation import EstimationProblem
    ref = sio.loadmat(REF / "est_ms2.mat", squeeze_me=True)
    m, data = _problem()
    pr = EstimationProblem(m, data)
    mode = ref["pm_mode"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert pr.logpost(mode) == pytest.approx(ref["pm_log_post"], abs=1e-6)
        assert pr.loglik(mode) == pytest.approx(ref["pm_log_lik"], abs=1e-6)
        assert pr.logprior(mode) == pytest.approx(ref["pm_log_prior"], abs=1e-5)
        lap = MDD(mode[None], [ref["pm_log_post"]], pr.logpost, mode=mode, hessian=ref["pm_hessian"]).laplace()
    assert lap == pytest.approx(ref["pm_log_marginal_data_density_laplace"], abs=1e-6)


@pytest.mark.slow
def test_mode_and_hessian_match_rise():
    """Same mode as RISE; the Hessian matches central differences of RISE's own objective
    (the one stored by RISE's estimate is different, which is why its Laplace MDD differs by 0.55)."""
    ref = sio.loadmat(REF / "est_ms2.mat", squeeze_me=True)
    hess = sio.loadmat(REF / "est_ms2_hess.mat", squeeze_me=True)
    m, data = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.estimate(data)
    assert r.mode == pytest.approx(ref["pm_mode"], abs=2e-5)
    assert r.logpost_mode == pytest.approx(ref["pm_log_post"], abs=1e-6)
    lap_central = -hess["f0"] + 3.5 * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(hess["hessian_central"])[1]
    assert r.log_mdd_laplace == pytest.approx(lap_central, abs=5e-3)
    assert np.allclose(r.hessian, hess["hessian_central"], rtol=2e-2, atol=2.0)
    assert abs(r.log_mdd_laplace - ref["pm_log_marginal_data_density_laplace"]) == pytest.approx(0.547, abs=0.02)


@pytest.mark.slow
def test_mdd_from_rise_draws_matches_rise():
    """With the sample from RISE's mh_sampler, the estimators are close to those of RISE's mcmc_mdd."""
    from msdsge.estimation import EstimationProblem
    ref = sio.loadmat(REF / "est_ms2.mat", squeeze_me=True)
    if "mdd" not in ref:
        pytest.skip("RISE did not finish the reference sampling")
    m, data = _problem()
    pr = EstimationProblem(m, data)
    x, lp = ref["mh_x"], -ref["mh_f"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert pr.logpost(x[0, 5]) == pytest.approx(lp[0, 5], abs=1e-6)
        md = MDD(x, lp, pr.logpost, pr.lower, pr.upper, mode=ref["pm_mode"], hessian=ref["pm_hessian"], L=3000, seed=1)
        for name, val in zip(ref["mdd_algos"], ref["mdd"]):
            if np.isfinite(val):
                assert md.estimate(str(name).strip()) == pytest.approx(val, abs=0.3), name
