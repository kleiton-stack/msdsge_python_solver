"""DSGE-VAR (Del Negro and Schorfheide, 2004). The DSGE-VAR of the local RISE does not run (solve and
estimate end in an error), so the tests check the formula against independent calculations and limits."""
import numpy as np
import pytest
from scipy import stats

from msdsge import Model
from msdsge.dsge_var import (DSGEVARProblem, dsge_moments, dsge_var, dsge_var_loglik, dsge_var_restrictions,
                             estimate_dsge_var)
from msdsge.var import VAR, gaussian_loglik

TXT = """var x z; varexo e u; varobs x z;
parameters a sig su {extra};
a = 0.8; sig = 0.5; su = 0.3; {cal}
model; z = a*z(-1) + sig*e; x = 0.5*z + 0.3*x(-1) + su*u; end;
shocks; var e; stderr 1; var u; stderr 1; end;
estimated_params;
  a, 0.7, 0.01, 0.99, beta_pdf, 0.7, 0.15;
  sig, 0.4, 0.01, 5, inv_gamma_pdf, 0.5, 0.5;
  su, 0.4, 0.01, 5, inv_gamma_pdf, 0.5, 0.5;
  {est}
end;"""


def model(weight=None, estimate_weight=False):
    extra = "dsge_prior_weight" if weight is not None else ""
    cal = f"dsge_prior_weight = {weight};" if weight is not None else ""
    est = "dsge_prior_weight, 1.0, 0.1, 20, gamma_pdf, 1.5, 1.0;" if estimate_weight else ""
    return Model.from_text(TXT.format(extra=extra, cal=cal, est=est))


@pytest.fixture(scope="module")
def setup():
    m = model()
    s = m.solve()
    data = s.simulate(periods=300, seed=4)[["x", "z"]]
    return m, s, data


def test_population_moments_and_restrictions(setup):
    m, s, data = setup
    GYY, GXY, GXX = dsge_moments(s, nlags=2)
    Phi, Sig = dsge_var_restrictions(GYY, GXY, GXX)
    # the DSGE is a VAR(1) in the observables: x = 0.3 x(-1) + 0.4 z(-1) + ..., z = 0.8 z(-1) + ...
    assert Phi == pytest.approx(np.array([[0, 0.3, 0.4, 0, 0], [0, 0, 0.8, 0, 0]]), abs=1e-9)
    imp = np.array([[0.3, 0.25], [0.0, 0.5]])           # columns (u, e) in the order (su, 0.5 sig; 0, sig)
    assert Sig == pytest.approx(imp @ imp.T, abs=1e-9)
    sim = s.simulate(periods=200000, seed=1)[["x", "z"]].to_numpy()
    assert GYY == pytest.approx(sim.T @ sim / len(sim), abs=0.02)
    assert GXY[1:3] == pytest.approx(sim[:-1].T @ sim[1:] / (len(sim) - 1), abs=0.02)


def test_likelihood_matches_conjugate_identity(setup):
    """p(Y) = p(Y | Phi, Sigma) p(Phi, Sigma) / p(Phi, Sigma | Y), with the scipy densities."""
    m, s, data = setup
    lam, p = 0.7, 2
    dv = dsge_var(s, data, nlags=p, lam=lam)
    Y, X = dv.data.Y, dv.data.X
    T, n = Y.shape
    K = X.shape[1]
    GYY, GXY, GXX = dv.moments
    Phi0, Sig0 = np.diag([1.0, 0.9]) @ dv.Phi + 0.01, np.array([[0.2, 0.05], [0.05, 0.3]])
    prior = (stats.invwishart.logpdf(Sig0, df=lam * T - K, scale=lam * T * dv.Sigma_dsge)
             + stats.matrix_normal.logpdf(Phi0.T, mean=dv.Phi_dsge.T, rowcov=np.linalg.inv(lam * T * GXX), colcov=Sig0))
    post = (stats.invwishart.logpdf(Sig0, df=(1 + lam) * T - K, scale=(1 + lam) * T * dv.Sigma)
            + stats.matrix_normal.logpdf(Phi0.T, mean=dv.Phi.T, rowcov=np.linalg.inv(lam * T * GXX + X.T @ X), colcov=Sig0))
    assert dv.loglik == pytest.approx(gaussian_loglik(Y - X @ Phi0.T, Sig0) + prior - post, abs=1e-7)


def test_limits_in_lambda(setup):
    m, s, data = setup
    inf = dsge_var(s, data, nlags=2, lam=np.inf)
    assert inf.loglik == pytest.approx(gaussian_loglik(inf.data.Y - inf.data.X @ inf.Phi_dsge.T, inf.Sigma_dsge))
    big = dsge_var(s, data, nlags=2, lam=1e7)
    assert big.loglik == pytest.approx(inf.loglik, abs=1e-2) and big.Phi == pytest.approx(inf.Phi_dsge, abs=1e-5)
    small = dsge_var(s, data, nlags=2, lam=0.03)
    ols = VAR(["x", "z"], nlags=2).fit(data, method="ml")
    assert np.abs(small.Phi - ols.B).max() < np.abs(inf.Phi - ols.B).max()
    assert small.Phi == pytest.approx(ols.B, abs=0.05)
    with pytest.raises(ValueError):
        dsge_var(s, data, nlags=2, lam=0.001)            # improper prior
    # with the true DSGE, imposing the restrictions is better than almost ignoring them
    assert inf.loglik > small.loglik
    # IRFs: in the limit, the impact in the DSGE-VAR is that of the DSGE
    idx = [m.endogenous.index(v) for v in ("x", "z")]
    assert inf.impact().to_numpy() == pytest.approx((s.R[0] * s.sigma)[idx], abs=1e-8)
    assert dsge_var(s, data, nlags=2, lam=1.0).irf(8).shape == (16, 2)
    vr = dsge_var(s, data, nlags=2, lam=1.0).to_var()
    assert vr.forecast(4).shape == (4, 2) and len(dsge_var(s, data, nlags=2, lam=1.0).sample_posterior(5)) == 5


def test_weight_parameter_in_model(setup):
    _, s, data = setup
    mw = model(weight=1.5)
    a = dsge_var(mw.solve(), data, nlags=2)
    assert a.lam == 1.5 and a.loglik == pytest.approx(dsge_var(s, data, nlags=2, lam=1.5).loglik)
    with pytest.raises(ValueError):
        dsge_var(s, data, nlags=2)


def test_joint_estimation(setup):
    m, s, data = setup
    r = estimate_dsge_var(m, data, nlags=2, lam=2.0)
    assert isinstance(r.problem, DSGEVARProblem) and r.hessian_pd
    assert r.mode == pytest.approx([0.8, 0.5, 0.3], abs=0.08)
    assert r.problem.loglik(r.mode) == pytest.approx(r.problem.dsge_var(r.mode).loglik)
    assert r.logpost_mode >= r.problem.logpost(r.problem.initial())
    mw = model(weight=1.0, estimate_weight=True)
    rw = estimate_dsge_var(mw, data, nlags=2)
    assert rw.problem.names[-1] == "dsge_prior_weight" and rw.mode[-1] > 1.0   # data generated by the DSGE itself
