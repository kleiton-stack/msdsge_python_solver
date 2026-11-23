"""Exploration: the package's order 2 to 5 solution against RISE (validation/ref/pert.mat)."""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio

from msdsge import Model
from msdsge.convert import rs_to_mod

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parents[1]
EX = Path(os.environ.get("RISE_TOOLBOX", Path.home() / "RISE_toolbox")) / "examples"
FILES = {
    "o2c": ROOT / "validation/matlab/o2c.rs", "o2": ROOT / "validation/matlab/o2.rs",
    "o2e": ROOT / "validation/matlab/o2e.rs", "o2r": ROOT / "validation/matlab/o2r.rs",
    "small_nk": EX / "Nonlinear3rdOrder/Binning/small_nk.rs",
    "frwz_nk": EX / "MarkovSwitching/FoersterRubioRamirezWaggonerZha/frwz_nk.rs",
}
FIELDS = ["Tz", "Tzz", "Tzzz", "Tzzzz", "Tzzzzz"]


def rise_layout(model, pert, ref):
    """Package rows and columns in the RISE ordering (rows: ref.endo; columns: ref.state_list)."""
    rows = [model.endogenous.index(v) for v in ref.endo]
    st = [model.endogenous[i] for i in pert.states]
    cols = []
    for name in np.atleast_1d(ref.state_list):
        if name == "@sig":
            cols.append(len(st))
        elif name.endswith("{-1}"):
            cols.append(st.index(name[:-4]))
        else:
            cols.append(len(st) + 1 + model.exogenous.index(name))
    return rows, cols


def compare(name, ref, order=None):
    model = Model.from_text(rs_to_mod(FILES[name]))
    order = order or len(np.atleast_1d(ref.time))
    t0 = time.perf_counter()
    s = model.solve(order=order)
    total = time.perf_counter() - t0
    rows, cols = rise_layout(model, s.pert, ref)
    out = {}
    for j in range(1, order + 1):
        mine = s.pert.tensor(j)[:, rows]
        for ax in range(j):
            mine = np.take(mine, cols, axis=2 + ax)
        nz = len(cols)
        refj = np.stack([np.asarray(np.atleast_1d(getattr(ref, FIELDS[j - 1]))[r], float).reshape(len(rows), -1)
                         for r in range(model.h)]) if model.h > 1 else \
            np.asarray(getattr(ref, FIELDS[j - 1]), float).reshape(1, len(rows), -1)
        dif = np.abs(mine.reshape(model.h, len(rows), -1) - refj)
        out[j] = (dif.max(), np.abs(refj).max())
    return s, total, out


if __name__ == "__main__":
    d = sio.loadmat(ROOT / "validation/ref/pert.mat", squeeze_me=True, struct_as_record=False)
    for name in (sys.argv[1:] or FILES):
        if name not in d or hasattr(d[name], "error"):
            print(name, "no reference:", getattr(d.get(name), "error", "")[:200])
            continue
        ref = d[name]
        s, total, out = compare(name, ref)
        print(f"{name}: package {total:.2f}s {dict((k, round(v, 3)) for k, v in s.pert.seconds.items())} | "
              f"RISE by order {np.round(np.atleast_1d(ref.time), 3)} retcode {ref.retcode}")
        for j, (dif, scale) in out.items():
            print(f"   order {j}: max diff {dif:.3e} (largest RISE coefficient {scale:.3e})")
