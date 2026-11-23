"""Exploration: order 1 to 3 simulation, with and without pruning, against RISE with the same shocks and regimes."""
import sys
import warnings

import numpy as np
import scipy.io as sio

from explore_perturbation import FILES, ROOT
from msdsge import Model
from msdsge.convert import rs_to_mod
from msdsge.perturbation import simulate

warnings.filterwarnings("ignore")


def compare(name, S, verbose=True, **kw):
    model = Model.from_text(rs_to_mod(FILES[name]))
    s = model.solve(order=3)
    out = {}
    for tag in S._fieldnames:
        x = getattr(S, tag)
        if hasattr(x, "error"):
            continue
        order, pruned = int(tag[1]), tag[4] == "1"
        names = list(x.names)
        endo = [v for v in names if v in model.endogenous]
        data = np.asarray(x.data, float)
        y0 = np.array([data[0, names.index(v)] for v in model.endogenous])
        shocks = np.column_stack([data[1:, names.index(e)] for e in model.exogenous])
        regimes = np.asarray(x.regime, float)[1:].astype(int) if model.h > 1 else None
        df = simulate(s, periods=shocks.shape[0], shocks=shocks, regimes=regimes, order=order, pruned=pruned,
                      initial=y0, variables=endo, **kw)
        ref = np.column_stack([data[1:, names.index(v)] for v in endo])
        dif = np.abs(df[endo].to_numpy() - ref)
        out[tag] = dif.max()
        if verbose:
            first = int(np.argmax(dif.max(1) > 1e-6)) if dif.max() > 1e-6 else -1
            print(f"  {name} {tag}: max diff {dif.max():.3e}  (first period with diff > 1e-6: {first}; "
                  f"regime switches: {0 if regimes is None else int((np.diff(regimes) != 0).sum())})")
    return out


if __name__ == "__main__":
    d = sio.loadmat(ROOT / "validation/ref/pert.mat", squeeze_me=True, struct_as_record=False)
    for name in (sys.argv[1:] or ["o2c", "o2", "o2e", "small_nk"]):
        if f"sim_{name}" in d:
            compare(name, d[f"sim_{name}"])
