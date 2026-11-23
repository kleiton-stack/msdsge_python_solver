"""Samplers on targets with a known answer: correlated normal and bimodal mixture."""
import numpy as np
import pytest

from msdsge import samplers as S
from msdsge.diagnostics import ess

A = np.array([[1.0, 0.5, 0.2], [0.5, 2.0, -0.3], [0.2, -0.3, 0.5]])
MU = np.array([1.0, -2.0, 0.5])
LOGZ = 3.7                                           # log of the normalising constant of both targets
_P = np.linalg.inv(A)
_K = -0.5 * np.linalg.slogdet(2 * np.pi * A)[1]
D = np.array([2.0, 0.0, 0.0])


def gauss(x):
    d = x - MU
    return LOGZ + _K - 0.5 * d @ _P @ d


def bimodal(x):                                      # 0.5 N(-D, I) + 0.5 N(D, I)
    return LOGZ - 1.5 * np.log(2 * np.pi) + np.log(0.5) + np.logaddexp(-0.5 * np.sum((x - D) ** 2), -0.5 * np.sum((x + D) ** 2))


TARGETS = {"gauss": (gauss, MU, A, MU, np.diag(A)), "bimodal": (bimodal, D, np.eye(3), np.zeros(3), np.array([5.0, 1.0, 1.0]))}


def run(sampler, f, x0, cov, seed=1, **kw):
    if sampler == "mh":
        return S.mh_sampler(f, x0, cov, draws=6000, burn=1000, chains=2, seed=seed, adapt_covariance=True, **kw)
    if sampler == "apt":
        return S.apt_sampler(f, x0, cov, draws=4000, burn=1000, chains=2, levels=3, seed=seed, **kw)
    if sampler == "dsmh":
        return S.dsmh_sampler(f, x0, cov, draws=3000, stages=6, seed=seed, **kw)
    if sampler == "rrf":
        return S.rrf_sampler(f, x0, cov, draws=3000, stages=6, seed=seed, **kw)
    if sampler == "smc":
        return S.smc_from_kernel(f, S.InitialDensity(x0, 9 * cov), particles=800, stages=25, seed=seed, **kw)
    raise ValueError(sampler)


@pytest.mark.parametrize("target", ["gauss", "bimodal"])
@pytest.mark.parametrize("sampler", ["mh", "apt", "dsmh", "rrf", "smc"])
def test_sampler_recovers_moments(sampler, target):
    f, x0, cov, mean, var = TARGETS[target]
    r = run(sampler, f, x0, cov)
    x = r.flat
    assert r.draws.ndim == 3 and r.logpost.shape == r.draws.shape[:2]
    assert np.allclose(r.logpost.ravel()[:50], [f(v) for v in x[:50]])
    n_eff = np.maximum(ess(r.draws), 50)
    tol = 5 * np.sqrt(var / n_eff) + 0.05
    assert np.all(np.abs(x.mean(0) - mean) < tol), (x.mean(0) - mean, tol)
    assert np.all(np.abs(x.var(0) / var - 1) < 0.25), x.var(0) / var
    assert 0.1 < r.acceptance.mean() < 0.7
    if target == "bimodal":                          # both modes are visited in the right proportion
        assert 0.35 < np.mean(x[:, 0] > 0) < 0.65
    if r.log_mdd is not None:                        # dsmh, rrf and smc estimate the constant
        assert r.log_mdd == pytest.approx(LOGZ, abs=0.2)


def test_mh_scale_adapts_to_target_acceptance():
    r = S.mh_sampler(gauss, MU, A, draws=6000, burn=2000, seed=3, c=5.0, alpha=0.3)
    assert r.acceptance[0] == pytest.approx(0.3, abs=0.05) and r.extras["c"][0] < 5.0
    fixed = S.mh_sampler(gauss, MU, A, draws=3000, seed=3, c=5.0, fixed_scaling=True)
    assert fixed.extras["c"][0] == pytest.approx(5.0) and fixed.acceptance[0] < r.acceptance[0]


def test_mh_covariance_adaptation_learns_the_target():
    r = S.mh_sampler(gauss, MU, np.eye(3), draws=15000, burn=3000, seed=4, adapt_covariance=True)
    assert np.allclose(r.extras["cov"][0], A, atol=0.35)


def test_seeds_thinning_and_parallel_reproducibility():
    a = S.mh_sampler(gauss, MU, A, draws=300, chains=2, seed=7, thin=2)
    b = S.mh_sampler(gauss, MU, A, draws=300, chains=2, seed=7, thin=2, parallel=2)
    c = S.mh_sampler(gauss, MU, A, draws=300, chains=2, seed=8, thin=2)
    assert np.array_equal(a.draws, b.draws) and not np.array_equal(a.draws, c.draws)
    assert not np.array_equal(a.draws[0], a.draws[1])
    init = S.InitialDensity(MU, 9 * A)
    s1 = S.smc_from_kernel(gauss, init, particles=100, stages=5, seed=2)
    s2 = S.smc_from_kernel(gauss, init, particles=100, stages=5, seed=2, parallel=2)
    assert np.array_equal(s1.draws, s2.draws) and s1.log_mdd == s2.log_mdd
    d1 = S.dsmh_sampler(gauss, MU, A, draws=200, stages=3, groups=4, seed=2)
    d2 = S.dsmh_sampler(gauss, MU, A, draws=200, stages=3, groups=4, seed=2, parallel=2)
    assert np.array_equal(d1.draws, d2.draws)


@pytest.mark.parametrize("sampler", ["mh", "apt", "dsmh", "smc"])
def test_bounds_are_respected_and_truncated_constant_is_right(sampler):
    """Standard normal truncated to [0, 1]³: no draw leaves the box and the constant is the mass of the box."""
    from scipy import stats
    f = lambda x: -0.5 * x @ x - 1.5 * np.log(2 * np.pi)
    lo, up = np.zeros(3), np.ones(3)
    x0, cov = np.full(3, 0.5), np.eye(3) * 0.08
    if sampler == "smc":
        r = S.smc_from_kernel(f, S.InitialDensity(x0, 9 * cov, lo, up), particles=600, stages=15, seed=1)
    elif sampler == "dsmh":
        r = S.dsmh_sampler(f, x0, cov, lo, up, draws=2000, stages=5, seed=1)
    else:
        r = getattr(S, sampler + "_sampler")(f, x0, cov, lo, up, draws=3000, burn=500, seed=1)
    assert np.all(r.flat >= 0) and np.all(r.flat <= 1)
    truth = stats.truncnorm(0, 1).mean()
    assert np.all(np.abs(r.flat.mean(0) - truth) < 0.04)
    if r.log_mdd is not None:
        assert r.log_mdd == pytest.approx(3 * np.log(stats.norm.cdf(1) - 0.5), abs=0.1)


def test_dsmh_raises_lambda_1_when_first_stage_is_degenerate():
    r = S.dsmh_sampler(gauss, MU, A, draws=1000, stages=4, lambda_1=1e-5, seed=1)
    assert r.extras["lambda"][0] > 1e-5 and r.extras["lambda"][-1] == 1.0
    assert np.all(np.diff(r.extras["lambda"]) > 0)


def test_smc_likelihood_tempering_from_prior():
    """Normal mean with normal prior: posterior and marginal data density have closed forms."""
    from scipy import stats
    y = np.array([0.8, 1.4, 0.3, 1.9, 1.1])
    loglik = lambda th: float(np.sum(stats.norm(th[0], 1.0).logpdf(y)))
    logprior = lambda th: float(stats.norm(0, 2.0).logpdf(th[0]))
    r = S.smc_sampler(loglik, logprior, lambda rng, n: rng.normal(0, 2.0, (n, 1)), particles=1500, stages=20, seed=5)
    v = 1 / (len(y) + 0.25)
    truth = stats.multivariate_normal(np.zeros(5), np.eye(5) + 4.0).logpdf(y)
    assert r.flat.mean() == pytest.approx(v * y.sum(), abs=0.05) and r.flat.var() == pytest.approx(v, rel=0.2)
    assert r.log_mdd == pytest.approx(truth, abs=0.1)
    ad = S.smc_sampler(loglik, logprior, lambda rng, n: rng.normal(0, 2.0, (n, 1)), particles=1500, adaptive=True, seed=5)
    assert ad.log_mdd == pytest.approx(truth, abs=0.1) and ad.extras["phi"][-1] == 1.0


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["mh", "apt", "dsmh", "smc"])
def test_posterior_sample_on_switching_model(sampler):
    """The samplers on the regime-switching model agree with each other and with the mode."""
    import warnings
    from msdsge import Model
    from test_estimation import MS
    m = Model.from_text(MS)
    data = m.solve().simulate(periods=200, seed=11)[["x"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.estimate(data)
        opts = {"mh": dict(draws=1500, burn=500, chains=2), "apt": dict(draws=1000, burn=300, levels=3),
                "dsmh": dict(draws=1000, stages=6), "smc": dict(particles=400, stages=30)}[sampler]
        r = r.sample(sampler, seed=1, **opts)
    sd = np.sqrt(np.diag(r.cov))
    x = r.draws.reshape(-1, r.mode.size)
    assert np.all(np.abs(x.mean(0) - r.mode) < 1.5 * sd + 0.02)
    assert np.all(x >= r.problem.lower) and np.all(x <= r.problem.upper)
    if r.sampler_result.log_mdd is not None:
        assert abs(r.sampler_result.log_mdd - r.log_mdd_laplace) < 1.5
