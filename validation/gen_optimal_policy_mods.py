"""Converts the RISE optimal-policy examples to the package's model language.

Output in ``validation/optimal_policy/*.mod`` (read by ``tests/test_optimal_policy.py``).
Usage: ``.venv/bin/python validation/gen_optimal_policy_mods.py``.
"""
import os
from pathlib import Path

from msdsge.convert import rs_to_mod

here = Path(__file__).parent
ex = Path(os.environ.get("RISE_TOOLBOX", Path.home() / "RISE_toolbox")) / "examples"
out = here / "optimal_policy"
out.mkdir(exist_ok=True)
# Taylor-rule parameters declared without a value in usmodel_sr_switch.rs (they do not enter the model)
unused = {k: 0.0 for k in ("crpi", "crr", "cry", "crdy")}
cases = {
    "canonical_osr": (ex / "OptimalSimpleRules/Canonical_osr.rs", {}),
    "canonical": (ex / "StochasticReplanning/Canonical.rs", {}),
    "canonical_plain": (ex / "StochasticReplanning/Estimation_examples/Canonical_plain.rs", {}),
    "fm95": (ex / "StochasticReplanning/FurherMoore95/fm95_ms_lc.rs", {}),
    "usmodel_ms_lc": (ex / "StochasticReplanning/usmodel_ms_lc.rs", {}),
    "usmodel_sr_switch": (ex / "StochasticReplanning/Estimation_examples/usmodel_sr_switch.rs", unused),
    "op_canL": (here / "matlab/op_canL.rs", {}),
    "op_canG": (here / "matlab/op_canG.rs", {}),
}
for name, (path, pars) in cases.items():
    text = rs_to_mod(path, parameters=pars) if pars else rs_to_mod(path)
    (out / f"{name}.mod").write_text(text, encoding="utf-8")
    print(name, "ok")
