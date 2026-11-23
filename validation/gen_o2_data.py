"""Generates simulated data (2nd order, pruned) for the validation of the nonlinear filters."""
import warnings
from pathlib import Path

from msdsge import Model
from msdsge.convert import rs_to_mod

ROOT = Path(__file__).parents[1]
for name, seed in (("o2c", 5), ("o2", 7), ("o2m", 9)):
    m = Model.from_text(rs_to_mod(ROOT / f"validation/matlab/{name}f.rs"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = m.solve(order=2)
    sim = s.simulate(periods=200, seed=seed, burn=100)
    sim[["C"]].to_csv(ROOT / f"validation/ref/nlf_{name}_data.csv", index=False)
    print(name, sim[["C", "K"]].describe().loc[["mean", "std"]].to_dict())
