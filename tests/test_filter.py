import numpy as np
import pandas as pd
import pytest

from msdsge import Model

AR = """var x; varexo e; parameters a; a = 0.7; varobs x;
model; x = a*x(-1) + e; end; shocks; var e; stderr 0.5; end;"""

MS = """var x z; varexo e u; varobs x;
markov_chain c(regimes=2);
parameters(c) a sig;
a(c,1) = 0.9; a(c,2) = 0.1; sig(c,1) = 0.2; sig(c,2) = 1.5;
transition_probabilities(c); p(1,2) = 0.05; p(2,1) = 0.10; end;
model; z = a*z(-1) + sig*e; x = z + 0.1*u; end;"""


def test_kalman_matches_exact_ar1_likelihood():
    s = Model.from_text(AR).solve()
    x = s.simulate(periods=200, seed=3)[["x"]]
    v = x["x"].to_numpy()
    # exact likelihood of a stationary Gaussian AR(1)
    ll = -0.5 * (np.log(2 * np.pi * 0.25 / (1 - 0.49)) + v[0] ** 2 * (1 - 0.49) / 0.25)
    e = v[1:] - 0.7 * v[:-1]
    ll += np.sum(-0.5 * (np.log(2 * np.pi * 0.25) + e ** 2 / 0.25))
    for f in ("rise", "imm", "kim"):
        assert s.filter(x, filter=f).loglik == pytest.approx(ll, rel=1e-10)


def test_missing_data_is_skipped():
    s = Model.from_text(AR).solve()
    x = s.simulate(periods=50, seed=3)[["x"]]
    x.iloc[10] = np.nan
    r = s.filter(x)
    assert np.isfinite(r.loglik) and r.loglik_t[10] == 0


def test_switching_filters_recover_regimes_and_agree():
    s = Model.from_text(MS).solve()
    sim = s.simulate(periods=600, seed=11)
    res = {f: s.filter(sim[["x"]], filter=f) for f in ("rise", "imm", "kim")}
    for r in res.values():
        assert np.allclose(r.filtered_probs.sum(axis=1), 1) and np.allclose(r.smoothed_probs.sum(axis=1), 1)
        hit = (r.smoothed_probs["regime_2"] > 0.5).to_numpy() == (sim["regime"] == 2).to_numpy()
        assert hit.mean() > 0.85
    # different approximate filters: close, not identical
    assert res["imm"].loglik == pytest.approx(res["kim"].loglik, rel=0.02)
    # the likelihood tells the true model apart from one with the volatility regimes swapped
    wrong = Model.from_text(MS).solve({"sig": {1: 1.5, 2: 0.2}})
    assert wrong.filter(sim[["x"]]).loglik < res["rise"].loglik
