"""Reduced-form and structural VAR: comparison with RISE (validation/matlab/ref_var.m and
ref_var_ms.m) and property tests where RISE draws at random or lacks the feature."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from msdsge.var import SVAR, VAR, MarkovChain, VARPrior, evaluate, hamilton_filter, vech

REF = Path(__file__).parents[1] / "validation" / "ref"
E3 = ["robs", "dy", "pinfobs"]
SMP = ("1960Q1", "1999Q4")
NRM = [("robs{0}@s1", "+"), ("dy{0}@s2", "+"), ("pinfobs{0}@s3", "+")]
CASES = [NRM + [("robs{0}@s2", 0), ("robs{0}@s3", 0), ("dy{0}@s3", 0)],
         NRM + [("robs{0}@s2", 0), ("robs{0}@s3", 0), ("dy{inf}@s3", 0)],
         NRM + [("robs{inf}@s2", 0), ("robs{inf}@s3", 0), ("dy{inf}@s3", 0)],
         NRM + [("robs{inf}@s2", 0), ("robs{inf}@s3", 0), ("dy{0}@s3", 0)],
         [("robs{0}@s1", "-"), ("dy{0}@s2", "-"), ("pinfobs{0}@s3", "-"), ("robs{0}@s2", 0), ("robs{0}@s3", 0), ("dy{1}@s3", 0)],
         NRM + [("dy{0}@s1", 0), ("pinfobs{0}@s1", 0), ("dy{inf}@s2", 0)]]
LIN0 = ["a1(1,pi)=0", "a2(1,pi)=0", "a1(1,ygap)=0", "a2(1,ygap)=0", "a2(1,FFR)=0", "a0(2,FFR)=0", "a1(2,FFR)=0",
        "a2(2,FFR)=0", "a1(2,ygap)=0", "a2(2,ygap)=0", "a1(3,FFR)=0", "a2(3,FFR)=0", "a1(3,pi)=0", "a2(3,pi)=0",
        "a0(3,pi)+a0(3,FFR)=0"]
NL0 = ["a0(3,FFR)>=0", "a1(1,FFR)>=0", "a1(1,FFR)<=1"]


@pytest.fixture(scope="module")
def rf():
    ref = sio.loadmat(REF / "var_rf.mat", squeeze_me=True, struct_as_record=False)
    names = [str(x) for x in ref["sw_names"]]
    df = pd.DataFrame(ref["sw_data"], columns=names, index=pd.period_range("1947Q3", periods=len(ref["sw_data"]), freq="Q"))
    df["tt"] = np.arange(1, len(df) + 1) / 100
    return ref, df


@pytest.fixture(scope="module")
def ms():
    ref = sio.loadmat(REF / "var_ms.mat", squeeze_me=True, struct_as_record=False)
    vl = [str(x) for x in ref["names"]]
    df = pd.DataFrame(ref["data"], columns=vl, index=pd.period_range("1954Q3", periods=len(ref["data"]), freq="Q"))
    return ref, df, vl


def _stack(d, names):
    return np.stack([d[v].to_numpy() for v in names], axis=2)


def test_ols_and_analyses_match_rise_howto(rf):
    """7-variable VAR(3) of RFVAR_identification/howto.m: estimation and all the analyses."""
    ref, df = rf
    o = ref["sw7"]
    endog = [str(x) for x in o.endog]
    r = VAR(endog, nlags=3).fit(df)
    assert r.names == [str(x) for x in o.pnames]
    assert r.theta == pytest.approx(o.p, abs=1e-10)
    assert r.B == pytest.approx(o.B, abs=1e-10) and r.Sigma == pytest.approx(o.S, abs=1e-12)
    assert r.loglik == pytest.approx(float(o.ll), abs=1e-8)
    irf = r.irf(12).to_numpy().reshape(7, 12, 7).transpose(1, 2, 0)
    assert irf == pytest.approx(o.irf, abs=1e-10)
    vd = r.variance_decomposition(40)
    V = np.stack([vd.loc[v].to_numpy() for v in endog], axis=2)
    assert V[:40] == pytest.approx(o.vd, abs=1e-10) and V[40].T == pytest.approx(o.vd_inf, abs=1e-10)
    hd = r.historical_decomposition()
    assert _stack(hd, endog) == pytest.approx(o.hd, abs=1e-9)
    assert hd["robs"].sum(axis=1).to_numpy() == pytest.approx(r.data.Y[:, 0], abs=1e-9)
    assert r.structural_shocks().to_numpy() == pytest.approx(o.shocks, abs=1e-9)
    C, R = r.autocov(5)
    assert C == pytest.approx(o.autocov, abs=1e-9) and R == pytest.approx(o.autocorr, abs=1e-10)
    assert r.forecast(8).to_numpy() == pytest.approx(o.fk[3:], abs=1e-10)
    start = pd.Period("2000Q1")
    assert r.forecast(8, start=start).to_numpy() == pytest.approx(o.fk_2000[3:], abs=1e-10)
    cond = df.loc["2000Q1":"2000Q4", ["robs"]].copy()
    cond["dy"] = df.loc["2000Q1":"2000Q2", "dy"]
    fc = r.forecast(8, start=start, conditions=cond)
    assert fc.to_numpy() == pytest.approx(o.fk_cond[3:], abs=1e-9)
    assert fc["robs"].to_numpy()[:4] == pytest.approx(df.loc["2000Q1":"2000Q4", "robs"].to_numpy())


@pytest.mark.parametrize("k", range(1, 7))
def test_exact_identification_matches_rise(rf, k):
    """Short-run, long-run, mixed and horizon-1 zeros, with sign normalisation."""
    ref, df = rf
    o = ref["sw3"]
    r = VAR(E3, nlags=2).fit(df)
    idn = r.identify(CASES[k - 1], ["s1", "s2", "s3"])
    R = r.impact(idn)
    assert idn.is_exact and list(getattr(o, f"ident{k}")) == [1, 0, 0]
    assert R == pytest.approx(getattr(o, f"R{k}"), abs=1e-10)
    assert R @ R.T == pytest.approx(r.Sigma, abs=1e-12) and idn.check(r.B, R)


def test_identified_irf_vd_hd_match_rise(rf):
    ref, df = rf
    o = ref["sw3"]
    r = VAR(E3, nlags=2).fit(df)
    idn = r.identify(CASES[3], ["s1", "s2", "s3"])
    assert r.irf(12, idn).to_numpy().reshape(3, 12, 3).transpose(1, 2, 0) == pytest.approx(o.irf4, abs=1e-10)
    vd = r.variance_decomposition(40, idn)
    V = np.stack([vd.loc[v].to_numpy() for v in E3], axis=2)
    assert V[:40] == pytest.approx(o.vd4, abs=1e-10) and V[40].T == pytest.approx(o.vd4_inf, abs=1e-10)
    assert _stack(r.historical_decomposition(idn), E3) == pytest.approx(o.hd4, abs=1e-9)
    assert r.structural_shocks(idn).to_numpy() == pytest.approx(o.shocks4, abs=1e-9)


def test_sign_restrictions_howto_properties(rf):
    """Sign restrictions of howto.m: RISE draws rotations, so properties are compared."""
    ref, df = rf
    o = ref["sw7"]
    endog = [str(x) for x in o.endog]
    r = VAR(endog, nlags=3).fit(df)
    restr = [("robs{0}@mp", "+"), ("dy@mp", "-"), ("pinfobs@mp", "-"), ("robs{0}@ad", "+"), ("pinfobs@ad", "+"),
             ("dy@ad", "+"), ("dy@as", "+"), ("pinfobs@as", "-"), ("robs@ls", "-"), ("dy@ls", "+"),
             ("labobs@ls", "+"), ("dw@ls", "-")]
    idn = r.identify(restr, ["mp", "ad", "as", "ls"], max_trials=6000, seed=1)
    assert idn.is_under and list(o.sign_ident) == [0, 0, 1]
    R = r.impact(idn)
    assert R @ R.T == pytest.approx(r.Sigma, abs=1e-10) and idn.check(r.B, R)
    # the matrix returned by RISE's Rfunc is just a draw with R R' = Sigma (it is neither checked nor
    # ordered by the named shocks; RISE filters the draws afterwards, inside irf)
    assert np.asarray(o.sign_R) @ np.asarray(o.sign_R).T == pytest.approx(r.Sigma, abs=1e-10)
    assert not np.allclose(R, r.impact(r.identify(restr, ["mp", "ad", "as", "ls"], seed=2)))
    irf = r.irf(8, idn, shocks=["mp"])
    assert irf.loc[("mp", 0), "robs"] > 0 and irf.loc[("mp", 0), "dy"] < 0
    # zeros and signs together, with slack: long-run zero and signs at several horizons
    mix = [("robs{0}@s1", "+"), ("dy{0:2}@s1", "-"), ("pinfobs{0:2}@s1", "-"), ("dy{inf}@s1", 0), ("robs{0}@s2", 0)]
    idm = r.identify(mix, ["s1", "s2"], seed=3)
    assert idm.check(r.B, r.impact(idm))


def test_exogenous_no_constant_and_linear_restrictions_match_rise(rf):
    ref, df = rf
    o = ref["misc"]
    r = VAR(E3, exog=["tt"], nlags=2).fit(df, SMP)
    assert r.names == [str(x) for x in o.names_exo]
    assert r.B == pytest.approx(o.B_exo, abs=1e-9) and r.loglik == pytest.approx(float(o.ll_exo), abs=1e-8)
    r = VAR(E3, nlags=2, constant=False).fit(df, SMP)
    assert r.B == pytest.approx(o.B_noconst, abs=1e-10) and r.loglik == pytest.approx(float(o.ll_noconst), abs=1e-8)
    lr = ["b1(1,dy)=0", "b2(1,dy)=0", "b1(2,robs)+b1(2,pinfobs)=0"]
    r = VAR(E3, nlags=2).fit(df, SMP, restrictions=lr)
    assert r.B == pytest.approx(o.B_linres, abs=1e-10) and r.Sigma == pytest.approx(o.S_linres, abs=1e-12)
    assert r.loglik == pytest.approx(float(o.ll_linres), abs=1e-8)
    ml = VAR(E3, nlags=2).fit(df, SMP, restrictions=lr, method="ml")      # iterated GLS: higher likelihood
    U = r.residuals().to_numpy()
    from msdsge.var import gaussian_loglik
    assert ml.loglik >= gaussian_loglik(U, U.T @ U / len(U)) - 1e-9 and ml.B[0, 2] == 0


def test_sims_zha_prior_matches_rise(rf):
    """sz prior (dummy observations) at the default hyperparameters, with and without linear restrictions."""
    ref, df = rf
    o = ref["misc"]
    r = VAR(E3, nlags=2).fit(df, SMP, prior="sz")
    assert r.B == pytest.approx(o.B_sz, abs=1e-10) and r.Sigma == pytest.approx(o.S_sz, abs=1e-12)
    assert r.loglik == pytest.approx(float(o.ll_sz), abs=1e-8)
    lr = ["b1(1,dy)=0", "b2(1,dy)=0", "b1(2,robs)+b1(2,pinfobs)=0"]
    rl = VAR(E3, nlags=2).fit(df, SMP, prior="sz", restrictions=lr)
    assert rl.B == pytest.approx(o.B_sz_linres, abs=1e-10) and rl.Sigma == pytest.approx(o.S_sz_linres, abs=1e-12)
    # sampler: coefficient means equal to RISE's within Monte Carlo error; the degrees of freedom of
    # RISE's Wishart are larger (see docs/VALIDATION.md), hence the tolerance on the covariances
    dr = r.sample_posterior(4000, seed=1)
    th = np.array([np.r_[b.reshape(-1, order="F"), vech(s)] for b, s in dr])
    assert th.mean(0)[:21] == pytest.approx(np.asarray(o.sz_draw_mean)[:21], abs=0.02)
    assert th.mean(0)[21:] == pytest.approx(np.asarray(o.sz_draw_mean)[21:], rel=0.2, abs=0.02)


@pytest.mark.parametrize("ptype", ["minnesota", "jeffrey", "nw", "normal-wishart", "inw"])
def test_other_priors_shrink_towards_prior_mean(rf, ptype):
    """Priors that do not run in the local RISE: checks analytical limits."""
    ref, df = rf
    assert ptype in [str(x[0]) for x in ref["misc"].broken]
    ols = VAR(E3, nlags=2).fit(df, SMP)
    r = VAR(E3, nlags=2).fit(df, SMP, prior=ptype, draws=300, burn=100)
    assert len(r.sample_posterior(20)) == 20
    if ptype == "jeffrey":
        assert r.B == pytest.approx(ols.B, abs=1e-10)
        return
    rw = np.zeros_like(ols.B)
    rw[:, 1:4] = np.eye(3)
    tight = VAR(E3, nlags=2).fit(df, SMP, prior=VARPrior(ptype, L1=1e-4, L4=1e6), draws=200, burn=50)
    loose = VAR(E3, nlags=2).fit(df, SMP, prior=VARPrior(ptype, L1=1e3, L2=1.0, L4=1e3), draws=200, burn=50)
    assert np.abs(tight.B[:, 1:] - rw[:, 1:]).max() < 0.02
    assert np.abs(loose.B - ols.B).max() < (0.1 if ptype == "inw" else 0.05)
    assert np.linalg.norm(r.B[:, 1:] - rw[:, 1:]) < np.linalg.norm(ols.B[:, 1:] - rw[:, 1:])


def test_bootstrap_and_simulated_forecast(rf):
    ref, df = rf
    r = VAR(E3, nlags=2).fit(df, SMP)
    bs = r.bootstrap(200, seed=0)
    assert bs.shape == (r.theta.size, 200)
    assert np.abs(bs.mean(1) - r.theta).max() < 4 * bs.std(1).max()
    sims = r.forecast(6, shock_uncertainty=True, draws=2000, seed=0)
    assert sims.mean(0) == pytest.approx(r.forecast(6).to_numpy(), abs=0.1)


@pytest.mark.parametrize("case", ["const", "vol", "vol3"])
def test_svar_likelihood_posterior_probabilities_irf_match_rise(ms, case):
    """Zha's SVAR (driver0, driver6): at the RISE parameter vector, likelihood, posterior,
    filtered/updated/smoothed probabilities and IRFs per regime."""
    ref, df, vl = ms
    chains = {"const": [], "vol": [MarkovChain("vol", 2, ("s",))], "vol3": [MarkovChain("syncvol", 3, ("s",))]}[case]
    tp = {"vol_tp_1_2": (0.1, 0.05), "vol_tp_2_1": (0.2, 0.1)} if case == "vol" else None
    m = SVAR(vl, nlags=2, markov_chains=chains)
    o = ref[case]
    assert sorted(m.parameter_names) == sorted(str(x) for x in o.pnames)
    r = evaluate(m, df, dict(zip([str(x) for x in o.pnames], o.p)), ("1960Q1", "2015Q2"), prior="sz", transition_priors=tp)
    assert r.loglik == pytest.approx(float(o.ll), abs=1e-7)
    # vol3: the Dirichlet prior does not enter RISE's objective (difference of 3e-4 without it)
    assert -r.logpost == pytest.approx(float(o.f0), abs=1e-3 if case == "vol3" else 1e-4)
    assert r.Bs == pytest.approx(np.moveaxis(np.atleast_3d(o.B), 2, 0), abs=1e-10)
    assert r.Sigmas == pytest.approx(np.moveaxis(np.atleast_3d(o.S), 2, 0), abs=1e-10)
    if m.h > 1:
        assert r.Q == pytest.approx(o.Q, abs=1e-12)
        assert r.smoothed_probs.to_numpy() == pytest.approx(o.smoothed_probs, abs=1e-9)
        assert r.updated_probs.to_numpy() == pytest.approx(o.updated_probs, abs=1e-9)
        assert r.filtered_probs.to_numpy() == pytest.approx(o.filtered_probs[:-1], abs=1e-9)
    A = r.irf(12).to_numpy().reshape(3, m.h, 12, 3).transpose(2, 3, 0, 1)
    assert A == pytest.approx(np.asarray(o.irf).reshape(12, 3, 3, -1), abs=1e-10)


def test_svar_posterior_mode_matches_rise_driver0(ms):
    ref, df, vl = ms
    o = ref["const"]
    r = SVAR(vl, nlags=2).fit(df, ("1960Q1", "2015Q2"), prior="sz", restrictions=LIN0 + NL0)
    assert -r.logpost == pytest.approx(float(o.f0), abs=1e-5)
    th = dict(zip([str(x) for x in o.pnames], o.p))
    assert r.theta == pytest.approx([th[k] for k in r.names], abs=1e-4)
    p = r.params
    assert p["a0_1_1"] == 1 and p["a1_1_2"] == 0 and p["a0_3_2"] == pytest.approx(-p["a0_3_1"]) and p["a0_3_1"] >= -1e-9


@pytest.mark.slow
def test_switching_svar_mode_not_worse_than_rise(ms):
    ref, df, vl = ms
    m = SVAR(vl, nlags=2, markov_chains=[MarkovChain("vol", 2, ("s",))])
    r = m.fit(df, ("1960Q1", "2015Q2"), prior="sz", restrictions=["a0(1,pi)=0", "a0(1,ygap)=0", "a0(2,ygap)=0"],
              transition_priors={"vol_tp_1_2": (0.1, 0.05), "vol_tp_2_1": (0.2, 0.1)})
    assert -r.logpost <= float(ref["vol"].f0) + 1e-4


def _simulate_ms(T=600, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.array([[0.95, 0.05], [0.10, 0.90]])
    c = [np.array([0.0, 0.0]), np.array([1.0, -1.0])]
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    L = [np.eye(2) * 0.3, np.array([[1.0, 0.0], [0.4, 0.8]])]
    y, s, out, st = np.zeros(2), 0, [], []
    for _ in range(T):
        s = rng.choice(2, p=Q[s])
        y = c[s] + A @ y + L[s] @ rng.standard_normal(2)
        out.append(y), st.append(s)
    return pd.DataFrame(out, columns=["x", "z"]), np.array(st), Q, A


def test_ms_var_em_recovers_regimes_and_agrees_with_direct_optimization():
    df, st, Q, A = _simulate_ms()
    m = VAR(["x", "z"], nlags=1, markov_chains=[MarkovChain("reg", 2, ("c", "s"))])
    r = m.fit(df)                                   # EM
    assert r.method == "em" and "b1_1_1" in r.names and "c_1_1_reg_2" in r.names
    sm = r.smoothed_probs.to_numpy()
    k = int(np.argmax([np.mean((sm[:, j] > 0.5) == (st[1:] == 1)) for j in range(2)]))
    assert np.mean((sm[:, k] > 0.5) == (st[1:] == 1)) > 0.93
    assert r.B[:, 1:] == pytest.approx(A, abs=0.08)
    assert sm.sum(1) == pytest.approx(1) and r.Q.sum(1) == pytest.approx(1)
    # the EM point is stationary for the direct optimisation of the same likelihood
    d = m.fit(df, method="optimize", x0=r.params)
    assert d.loglik == pytest.approx(r.loglik, abs=1e-3) and d.loglik >= r.loglik - 1e-6
    const = VAR(["x", "z"], nlags=1).fit(df, method="ml")
    assert r.loglik > const.loglik + 50
    # known regimes and forecast with a mixture of regimes
    known = evaluate(m, df, r.theta, regimes=st[1:] + 1)
    assert np.isfinite(known.loglik)
    f = r.forecast(4)
    assert f.shape == (4, 2) and np.all(np.isfinite(f.to_numpy()))
    assert r.girf(6, draws=50).shape == (12, 2)


def test_hamilton_filter_against_brute_force():
    rng = np.random.default_rng(0)
    T, h = 6, 2
    logf = rng.standard_normal((T, h))
    Q = np.array([[0.8, 0.2], [0.3, 0.7]])
    f = hamilton_filter(logf, Q, pi0="ergodic")
    import itertools
    from msdsge.var import ergodic
    pi, tot, marg = ergodic(Q), 0.0, np.zeros((T, h))
    for path in itertools.product(range(h), repeat=T):
        p = pi[path[0]] * np.exp(logf[0, path[0]])
        for t in range(1, T):
            p *= Q[path[t - 1], path[t]] * np.exp(logf[t, path[t]])
        tot += p
        for t in range(T):
            marg[t, path[t]] += p
    assert f["loglik"] == pytest.approx(np.log(tot)) and f["smoothed"] == pytest.approx(marg / tot)


def test_restriction_parser_errors():
    m = SVAR(["a", "b"], nlags=1, markov_chains=[MarkovChain("ch", 2, ("a0",))])
    coef, const, op = m.layout.parse_restriction("a0(2,a,ch,1)+2*a1(1,b)>=0.5")
    assert op == ">=" and const == -0.5 and coef[m.parameter_names.index("a1_1_2")] == 2
    with pytest.raises(ValueError):
        m.layout.parse_restriction("a0(2,a)=0")          # chain and state missing
    with pytest.raises(ValueError):
        VAR(["a", "b"], nlags=1).fit(pd.DataFrame({"a": [1.0, np.nan, 2, 3, 4, 5, 6], "b": np.arange(7.0)}))
