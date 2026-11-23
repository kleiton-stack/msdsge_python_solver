"""2nd order against RISE: growth model with regime-switching depreciation."""
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from msdsge import Model
from msdsge.convert import rs_to_mod

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("name", ["o2c", "o2"])   # o2c: same parameters in both regimes
def test_second_order_matches_rise(name):
    ref = sio.loadmat(ROOT / "validation/ref/o2.mat", squeeze_me=True, struct_as_record=False)[name]
    m = Model.from_text(rs_to_mod(ROOT / f"validation/matlab/{name}.rs"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = m.solve(order=2)
    rows = [m.endogenous.index(v) for v in ref.endo]          # z = [K, A, sigma, E] in both
    for r in range(2):
        Tzz = np.array(ref.Tzz[r]).reshape(3, 4, 4)
        assert s.G2[r][rows] == pytest.approx(Tzz, abs=1e-6)
    if name == "o2":
        assert np.abs(s.G2[:, :, 2, 2]).max() > 0.5        # large sigma-sigma term: ss difference between regimes


def test_second_order_simulation_pruned():
    m = Model.from_text(rs_to_mod(ROOT / "validation/matlab/o2c.rs"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = m.solve(order=2)
    a = s.simulate(periods=3000, seed=1, regimes=[1])
    b = s.simulate(periods=3000, seed=1, regimes=[1], order=1)
    assert np.isfinite(a.to_numpy()).all()
    # same shock sequence: the 2nd order differs little and shifts the mean (precautionary saving)
    assert 0 < np.abs(a["K"] - b["K"]).max() < 2.0
    assert a["K"].mean() != pytest.approx(b["K"].mean(), abs=1e-6)
    # productivity is log-linear: mean level above 1 because of the Jensen correction
    assert a["A"].mean() > b["A"].mean() - 1e-3
