"""Additional priors, Dirichlet, endogenous priors, optimisers and diagnostics."""
import warnings

import numpy as np
import pytest
from scipy import integrate, stats

from msdsge import Model
from msdsge import diagnostics as dg
from msdsge import optimizers as opt
from msdsge.estimation import EstimationProblem, Prior
from msdsge.priors_extra import DirichletPrior, EndogenousPrior, stdev_moments

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
  sig(c,1), 0.3, 0.01, 5, truncated_normal_pdf, 0.3, 0.2;
  p12, 0.1, 0.001, 0.9;
end;"""


@pytest.mark.parametrize("shape, m, s", [("laplace", 1, .5), ("logistic", 1, .5), ("pareto", 2, .5), ("left_triang", 1, .3),
                                         ("right_triang", 1, .3), ("exponential", 2, 2)])
def test_extra_priors_integrate_to_one_with_requested_moments(shape, m, s):
    p = Prior(shape, m, s)
    lo, up = p.support()
    lo, up = max(lo, m - 60 * s), min(up, m + 3000 * s)
    f = lambda x: np.exp(p.logpdf(x))
    opts = dict(limit=500, points=[m])
    mass = integrate.quad(f, lo, up, **opts)[0]
    mean = integrate.quad(lambda x: x * f(x), lo, up, **opts)[0]
    var = integrate.quad(lambda x: (x - mean) ** 2 * f(x), lo, up, **opts)[0]
    assert mass == pytest.approx(1, abs=1e-3) and mean == pytest.approx(m, rel=5e-3)
    assert np.sqrt(var) == pytest.approx(s, rel=5e-2)


def test_cauchy_and_truncated_normal():
    c = Prior("cauchy", 1.0, 0.5)
    assert c.logpdf(1.7) == pytest.approx(stats.cauchy(1.0, 0.5).logpdf(1.7))
    t = Prior("truncated_normal", 0.3, 0.2, bounds=(0.0, 1.0))
    assert t.support() == (0.0, 1.0) and t.logpdf(-0.1) == -np.inf
    assert integrate.quad(lambda x: np.exp(t.logpdf(x)), 0, 1)[0] == pytest.approx(1, abs=1e-8)
    assert t.logpdf(0.3) > stats.norm(0.3, 0.2).logpdf(0.3)


@pytest.mark.parametrize("shape, lo, up", [("cauchy", -1, 2), ("laplace", -1, 2), ("logistic", -1, 2), ("pareto", 1, 4),
                                           ("left_triang", 0.2, 1.5), ("right_triang", 0.2, 1.5), ("truncated_normal", 0.1, 0.8)])
def test_extra_priors_by_quantiles(shape, lo, up):
    p = Prior(shape, lo, up, prob=0.90, bounds=(0.0, 1.0) if shape == "truncated_normal" else None)
    assert p._dist.cdf(lo) == pytest.approx(0.05, abs=1e-6) and p._dist.cdf(up) == pytest.approx(0.95, abs=1e-6)
    assert np.isfinite(p.mean) and np.isfinite(p.std)


def test_dirichlet_prior_in_rise_format():
    names = ["a", "p12", "p13"]
    d = DirichletPrior.from_rise(names, [0.05, "p12", 0.1, "p13", 0.2])
    assert d.mean == pytest.approx([0.1, 0.2])
    a0 = d.alpha.sum()
    assert np.sqrt(0.7 * 0.3 / (a0 + 1)) == pytest.approx(0.05)            # standard deviation of the diagonal
    th = np.array([9.0, 0.15, 0.25])
    assert d.logpdf(th) == pytest.approx(stats.dirichlet(d.alpha).logpdf([0.15, 0.25, 0.60]))
    assert d.logpdf(np.array([9.0, 0.6, 0.5])) == -np.inf
    x = np.array([d.rvs(np.random.default_rng(i)) for i in range(4000)])
    assert x.mean(0) == pytest.approx([0.1, 0.2], abs=0.01) and x.std(0) == pytest.approx(d.std, rel=0.1)
    same = DirichletPrior.from_rise(names, {"names": ["p12", "p13"], "alpha": d.alpha})
    assert same.logpdf(th) == pytest.approx(d.logpdf(th))


def _data(m):
    return m.solve().simulate(periods=150, seed=11)[["x"]]


def test_dirichlet_and_endogenous_priors_enter_the_posterior():
    m = Model.from_text(MS)
    data = _data(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = EstimationProblem(m, data)
        x = base.initial()
        dpr = EstimationProblem(m, data, dirichlet=[0.05, "p12", 0.1])
        d = dpr.joint_priors[0]
        assert d.logpdf(x) == pytest.approx(stats.beta(d.alpha[0], d.alpha[1]).logpdf(x[2]))   # Dirichlet of 2 = beta
        assert dpr.logpost(x) - base.logpost(x) == pytest.approx(d.logpdf(x))
        endo = EndogenousPrior(stdev_moments(["x"], regime=1), [(0.5, 0.1, "normal")])
        epr = EstimationProblem(m, data, endogenous_priors=endo)
        sd_x = stdev_moments(["x"])(epr.solve(x))[0]
        z = base.solve(x)
        assert sd_x == pytest.approx(np.sqrt(x[1] ** 2 / (1 - x[0] ** 2) + 0.01), rel=1e-8)
        assert epr.logpost(x) - base.logpost(x) == pytest.approx(stats.norm(0.5, 0.1).logpdf(sd_x))
        assert epr.bayesian and dpr.bayesian and z is not None


# ---------------------------------------------------------------- optimisers
def _rosen(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


@pytest.mark.parametrize("method, tol", [("bee_gate", 5e-2), ("fmincon", 1e-3), ("fminunc", 1e-3),
                                         ("csminwel", 1e-3), ("fminsearch", 1e-3), ("differential_evolution", 5e-2),
                                         ("dual_annealing", 0.5)])
def test_optimizers_on_rosenbrock(method, tol):
    r = opt.minimize(_rosen, np.array([-1.0, 1.5]), [-2, -2], [2, 2], method=method, seed=1)
    assert np.all(np.abs(r.x - 1) < tol * 4) and r.fun < tol and r.nfev > 0


def test_blockwise_on_correlated_quadratic():
    """Block descent converges slowly in curved valleys (Rosenbrock); the test uses a quadratic."""
    H = np.array([[2.0, 0.8, 0.1], [0.8, 1.0, 0.3], [0.1, 0.3, 1.5]])
    f = lambda x: float(0.5 * (x - 1) @ H @ (x - 1))
    r = opt.minimize(f, np.zeros(3), [-5] * 3, [5] * 3, method="blockwise", blocks=[[0, 1], [2]], sweeps=100)
    assert r.x == pytest.approx(np.ones(3), abs=1e-3)


def test_bee_colony_finds_global_minimum_of_multimodal_function_and_respects_bounds():
    rastrigin = lambda x: float(10 * x.size + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))
    r = opt.abc(rastrigin, np.array([3.2, -2.7]), [-5.12, -5.12], [5.12, 5.12], colony=30, maxiter=400, seed=2)
    assert r.fun < 1e-3 and np.all(np.abs(r.x) < 1e-2)
    half = opt.abc(lambda x: float(np.sum((x + 1) ** 2)), np.array([0.5, 0.5]), [0, 0], [2, 2], seed=1)
    assert np.all(half.x >= 0) and half.x == pytest.approx([0, 0], abs=1e-3)


def test_multistart_escapes_local_minimum():
    f = lambda x: float((x[0] ** 2 - 1) ** 2 + 0.3 * x[0])        # global minimum near -1, local near +1
    one = opt.minimize(f, np.array([1.0]), [-3], [3], method="fmincon")
    many = opt.multistart(f, np.array([1.0]), [-3], [3], starts=8, method="fmincon", seed=1, spread=1.0)
    assert one.x[0] > 0 and many.x[0] < 0 and many.fun < one.fun and len(many.extras["runs"]) == 8


def test_estimate_with_bee_gate_and_restarts():
    m = Model.from_text("""var x; varexo e; parameters a; a = 0.5; varobs x;
        model; x = a*x(-1) + e; end; shocks; var e; stderr 1; end;
        estimated_params; a, 0.5, -0.99, 0.99; stderr e, 1, 0.01, 10; end;""")
    x = m.solve({"a": 0.8}, shock_stderr={"e": 0.5}).simulate(periods=300, seed=5)[["x"]]
    base = m.estimate(x)
    bee = m.estimate(x, optimizer="bee_gate", optimizer_options={"cycles": 60}, seed=1)
    multi = m.estimate(x, optimizer="fmincon", restarts=2, seed=1)
    assert bee.mode == pytest.approx(base.mode, abs=1e-4) and multi.mode == pytest.approx(base.mode, abs=1e-4)
    prof = dg.check_optimum(base, points=7)
    assert set(prof) == {"a", "e"} or len(prof) == 2
    assert all(df.attrs["is_max"] and {"value", "log_post", "log_lik", "log_prior"} <= set(df.columns) for df in prof.values())
    ident = dg.identification(base, method="hessian")
    assert ident["identified"] and ident["rank"] == 2
    assert dg.identification(base, method="moments")["identified"]


# ---------------------------------------------------------------- diagnostics
def _ar1(rho, n, chains, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    x = np.zeros((chains, n, 2))
    e = rng.standard_normal((chains, n, 2))
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    x[0] += shift
    return x


def test_gelman_rubin_and_brooks_gelman():
    ok, bad = _ar1(0.5, 4000, 3, 0), _ar1(0.5, 4000, 3, 0, shift=3.0)
    g, b = dg.gelman_rubin(ok, ["a", "b"]), dg.gelman_rubin(bad, ["a", "b"])
    assert np.all(g["R"] < 1.01) and g["Rp"] < 1.02 and list(g["R"].index) == ["a", "b"]
    assert np.all(b["R"] > 1.2) and b["Rp"] > 1.2
    assert dg.gelman_rubin(ok[:1], split=True)["R"].max() < 1.02
    with pytest.raises(ValueError):
        dg.gelman_rubin(ok[:1])
    path = dg.psrf_path(ok, ["a", "b"], points=5)
    assert path.shape[1] == 2 and path.iloc[-1].max() < 1.01


def test_effective_sample_size_and_geweke():
    rho = 0.8
    x = _ar1(rho, 20000, 2, 1)
    assert dg.ess(x) == pytest.approx(40000 * (1 - rho) / (1 + rho), rel=0.15)
    assert dg.ess(_ar1(0.0, 5000, 2, 2)) == pytest.approx(10000, rel=0.1)
    gw = dg.geweke(x, ["a", "b"])
    assert len(gw) == 4 and np.all(np.abs(gw["z"]) < 3.5)
    drift = x[0, :, 0] + np.linspace(0, 5, 20000)
    assert abs(dg.geweke(drift)["z"].iloc[0]) > 4
    ac = dg.autocorrelations(x, lags=3)
    assert ac.shape == (2, 4, 2) and ac[0, 1, 0] == pytest.approx(rho, abs=0.03)
    assert dg.cumulative_means(x)[:, -1] == pytest.approx(x.mean(1))
    s = dg.summary(x, ["a", "b"])
    assert {"mean", "std", "ess", "rhat"} <= set(s.columns)


def test_trace_and_prior_posterior_data():
    from msdsge.estimation import mcmc
    m = Model.from_text(MS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.estimate(_data(m))
        r = r.sample("mh", draws=150, chains=2, seed=1) if r.hessian_pd else mcmc(r, draws=150, cov=np.eye(3) * 1e-4)
    tr = dg.trace_data(r.draws, r.problem.names)
    assert list(tr) == r.problem.names and tr[r.problem.names[0]].shape == (2, 150)
    pp = dg.prior_posterior_data(r, points=50)
    first = pp[r.problem.names[0]]
    assert {"grid", "prior", "posterior", "mode"} <= set(first) and "prior" not in pp["p12"]
    assert np.trapezoid(first["prior"], first["grid"]) == pytest.approx(1, abs=0.02)
    out = dg.trace_plot(r.draws, r.problem.names)       # figure with matplotlib, data without it
    assert out is not None
