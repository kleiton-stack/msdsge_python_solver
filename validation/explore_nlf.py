"""Exploration: RISE sigma-point filters (UKF, CKF) against msdsge.nonlinear_filters.

Reference: validation/matlab/ref_nlf.m (writes validation/ref/nlf.mat). The MATLAB script captures
the structure that dsge.filter hands to kf_user_algo (initial state and covariance, solution
matrices) and logs the calls to the law of motion, which reveals the sigma points. Here: (1) the
package's law of motion against the logged calls; (2) the recursion with the RISE matrices;
(3) the full pipeline (the package's 2nd-order solution).
"""
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio

from msdsge import Model
from msdsge.convert import rs_to_mod
from msdsge.nonlinear_filters import _transition, rise_variable_order, sigma_filter_core

warnings.filterwarnings("ignore")
D = sio.loadmat("validation/ref/nlf.mat", squeeze_me=True, struct_as_record=False)


def rise_law(d):
    def f(r, X):
        ss, T1, T2 = d.ss[r], d.T1[r], d.T2[r].reshape(4, 4, 4)
        z = np.column_stack([X[:, :2] - ss[:2], np.ones(len(X)), X[:, 3]])
        return ss + z @ T1.T + 0.5 * np.einsum("nij,pi,pj->pn", T2, z, z)
    return f


for name in ("o2c", "o2", "o2m"):
    d = D[name]
    s = Model.from_text(rs_to_mod(f"validation/matlab/{name}f.rs")).solve(order=2)
    f = _transition(s, rise_variable_order(s))
    data = pd.read_csv(f"validation/ref/nlf_{name}_data.csv")
    print(name, "| RISE default filter (DDF):", getattr(d, "default_error", "ran"))
    for k in ("ukf", "ckf"):
        X, O, R, E = (getattr(d, f"{k}_{q}") for q in "XORE")
        err = max(np.max(np.abs(f(int(R[i]) - 1, np.r_[X[:3, i], E[i]][None])[0] - O[:, i])) for i in range(X.shape[1]))
        ref = np.ravel(getattr(d, k + "_inc"))
        core = sigma_filter_core(rise_law(d), np.atleast_2d(d.y).T, np.array([2]), np.zeros((1, 1)), np.array(d.Q, float),
                                 d.PAI00, np.stack(d.a), np.stack(d.P), 1, k)
        r = s.filter(data, filter=k)
        print(f"  {k}: law of motion {err:.1e} | recursion with the RISE solution: max diff incr {np.max(np.abs(core[0] - ref)):.1e}"
              f" | pipeline: loglik {r.loglik:.6f} RISE {float(getattr(d, k + '_ll')):.6f} max diff incr {np.max(np.abs(r.loglik_t - ref)):.1e}")
    print(f"  ddf (package, no reference): loglik {s.filter(data, filter='ddf').loglik:.6f}")
