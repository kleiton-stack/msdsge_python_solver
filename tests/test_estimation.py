import numpy as np
import pytest
from scipy import integrate

from msdsge import Model
from msdsge.estimation import Prior, mcmc

MS = """var x z; varexo e u; varobs x;
markov_chain c(regimes=2);
parameters p12 p21;
parameters(c) a sig;
p12 = 0.05; p21 = 0.10;
a(c,1) = 0.9; a(c,2) = 0.1; sig(c,1) = 0.2; sig(c,2) = 1.5;
transition_probabilities(c); p(1,2) = p12; p(2,1) = p21; end;
model; z = a*z(-1) + sig*e; x = z + 0.1*u; end;
estimated_params;
  a(c,1), 0.7, 0, 0.999, beta_pdf, 0.7, 0.15;
  sig(c,1), inv_gamma_pdf, 0.3, 0.2;
  sig(c,2), 1.0, 0.01, 10, inv_gamma_pdf, 1.0, 0.5;
  p12, beta_pdf, 0.1, 0.05;
end;"""


@pytest.mark.parametrize("shape, m, s", [("normal", 1, .5), ("beta", .7, .1), ("gamma", 2, 1), ("uniform", 1, .3),
                                         ("inv_gamma", .5, .3), ("inv_gamma2", .5, .3)])
def test_priors_integrate_to_one_with_requested_moments(shape, m, s):
    p = Prior(shape, m, s)
    lo, up = p.support()
    lo, up = max(lo, m - 60 * s), min(up, m + 400 * s)
    f = lambda x: np.exp(p.logpdf(x))
    opts = dict(limit=400, points=[m])
    mass = integrate.quad(f, lo, up, **opts)[0]
    mean = integrate.quad(lambda x: x * f(x), lo, up, **opts)[0]
    var = integrate.quad(lambda x: (x - mean) ** 2 * f(x), lo, up, **opts)[0]
    assert mass == pytest.approx(1, abs=1e-4)
    assert mean == pytest.approx(m, rel=2e-3) and np.sqrt(var) == pytest.approx(s, rel=3e-2)


def test_ml_ar1_matches_ols():
    m = Model.from_text("""var x; varexo e; parameters a; a = 0.5; varobs x;
        model; x = a*x(-1) + e; end; shocks; var e; stderr 1; end;
        estimated_params; a, 0.5, -0.99, 0.99; stderr e, 1, 0.01, 10; end;""")
    x = m.solve({"a": 0.8}, shock_stderr={"e": 0.5}).simulate(periods=2000, seed=5)[["x"]]
    r = m.estimate(x)
    v = x["x"].to_numpy()
    ols = (v[1:] @ v[:-1]) / (v[:-1] @ v[:-1])
    assert not r.problem.bayesian and r.hessian_pd
    assert r.mode[0] == pytest.approx(ols, abs=5e-3)
    assert r.mode[1] == pytest.approx(np.std(v[1:] - ols * v[:-1]), abs=5e-3)


def test_bayesian_switching_model_mode_and_mcmc():
    m = Model.from_text(MS)
    data = m.solve().simulate(periods=400, seed=11)[["x"]]
    r = m.estimate(data)
    assert r.problem.names == ["a_c_1", "sig_c_1", "sig_c_2", "p12"]
    assert r.hessian_pd and np.isfinite(r.log_mdd_laplace)
    assert r.logpost_mode >= r.problem.logpost(r.problem.initial())
    true = np.array([0.9, 0.2, 1.5, 0.05])
    assert np.all(np.abs(r.mode - true) < 4 * np.sqrt(np.diag(r.cov)) + 0.05)
    r = mcmc(r, draws=600, chains=2, burn=200, scale=0.8, seed=1)
    assert r.draws.shape == (2, 600, 4) and np.all((r.acceptance > 0.05) & (r.acceptance < 0.8))
    s = r.summary()
    assert {"mode", "post_mean", "ci_low", "ci_high"} <= set(s.columns)
    assert np.all(r.rhat() < 1.3)
    assert abs(r.log_mdd_mhm() - r.log_mdd_laplace) < 3


@pytest.mark.parametrize("shape, lo, up", [("beta", .3235, .3891), ("gamma", .5, 5.0), ("normal", -1, 2),
                                           ("inv_gamma", .1, 2.0), ("uniform", 0, 1), ("lognormal", .2, 3), ("weibull", .5, 4)])
def test_quantile_priors(shape, lo, up):
    p = Prior(shape, lo, up, prob=0.90)
    assert p._dist.cdf(up) - p._dist.cdf(lo) == pytest.approx(0.90, abs=1e-6)
    assert p._dist.cdf(lo) == pytest.approx(0.05, abs=1e-6)
    assert lo < p.mean < up
