import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from msdsge import Model
from test_filter import AR, MS

ROOT = Path(__file__).parents[1]
REF = ROOT / "validation" / "ref"




def test_autocovariances_ar1():
    s = Model.from_text(AR).solve()
    ac = s.theoretical_autocovariances(lags=2)
    assert ac[0, 0] == pytest.approx([0.25 / 0.51, 0.7 * 0.25 / 0.51, 0.49 * 0.25 / 0.51])


def test_historical_decomposition_adds_up_and_is_exact_without_switching():
    m = Model.from_text("""var x z; varexo e u; parameters a; a = 0.8; varobs x z;
        model; z = a*z(-1) + e; x = 0.5*x(-1) + z + u; end;""")
    s = m.solve()
    sim = s.simulate(periods=80, seed=2)
    f = s.filter(sim[["x", "z"]])
    hd = s.historical_decomposition(f)
    assert hd["x"].sum(axis=1).to_numpy() == pytest.approx(f.smoothed_states["x"].to_numpy(), abs=1e-10)
    assert hd["z"]["u"].abs().max() < 1e-12          # u does not affect z
    # with all variables observed, the initial contribution decays at the model's rate
    init = hd["z"]["init"].to_numpy()
    assert init[5] == pytest.approx(init[4] * 0.8, abs=1e-9)
    # switching model: the columns still sum to the smoothed variable
    s2 = Model.from_text(MS).solve()
    f2 = s2.filter(pd.read_csv(REF / "filter_ms_data.csv"))
    hd2 = s2.historical_decomposition(f2)
    assert hd2["z"].sum(axis=1).to_numpy() == pytest.approx(f2.smoothed_states["z"].to_numpy(), abs=1e-10)


def test_forecast_and_girf():
    s = Model.from_text(AR).solve()
    x = s.simulate(periods=50, seed=3)[["x"]]
    f = s.filter(x)
    fc = s.forecast(f, steps=3)
    assert fc["x"].to_numpy() == pytest.approx(x["x"].iloc[-1] * 0.7 ** np.arange(1, 4))
    # without regime switching, the generalised IRF is the usual IRF
    assert s.girf("e", periods=5, draws=3)["x"].to_numpy() == pytest.approx(s.irf(periods=5)["x"].to_numpy())
    # with exogenous switching, it starts at the response of the initial regime and stays between the regimes
    s2 = Model.from_text(MS).solve()
    g = s2.girf("e", periods=6, draws=300, regime0=1, seed=4)["z"].to_numpy()
    irf = s2.irf("e", periods=6)
    assert g[0] == pytest.approx(irf.loc[("e", 1), "z"].iloc[0])
    assert irf.loc[("e", 2), "z"].iloc[1] - 1e-12 <= g[1] <= irf.loc[("e", 1), "z"].iloc[1] + 1e-12
