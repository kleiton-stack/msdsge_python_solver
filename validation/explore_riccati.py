"""Exploration: stopping rule of the RISE Riccati shortcut (kf_riccati_tol).

Reference: validation/matlab/ref_riccati.m (increments for a grid of tolerances). For each
tolerance, shows the period at which the RISE increments start to differ from the full recursion,
the period at which the package freezes and the maximum difference. The rule that reproduces every
case is in msdsge.filter_extras.RiccatiShortcut: the gain being compared includes the shock rows
(a factor sig/0.01 = 20 relative to the gain of z in this model, which was the clue).
"""
import pathlib

import numpy as np
import pandas as pd
import scipy.io as sio

from msdsge import Model

exec(pathlib.Path("tests/test_filter.py").read_text().split("def test_")[0])
R = sio.loadmat("validation/ref/riccati.mat", squeeze_me=True, struct_as_record=False)["ms"]
s = Model.from_text(MS).solve()
data = pd.read_csv("validation/ref/filter_ms_data.csv")[["x"]]
for i, tol in enumerate(R.tols):
    r = s.filter(data, riccati_tol=float(tol), smooth=False)
    nz = np.flatnonzero(np.abs(R.Incr[:, i] - R.Incr[:, 0]) > 1e-14)
    print(f"tol {tol:9.2e} | RISE differs from t={nz[0] if nz.size else None} | "
          f"max diff incr {np.max(np.abs(r.loglik_t - R.Incr[:, i])):.2e} | loglik diff {r.loglik - R.LogLik[i]:.2e}")
