"""Exploration: where do the sigma terms of the RISE 2nd-order solution come from when the steady state differs across regimes?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.io as sio
from msdsge import Model
from msdsge.convert import rs_to_mod
from msdsge.order2 import _hessian_entries
from msdsge.solve import _U, _states
np.set_printoptions(precision=6, suppress=True, linewidth=200)
d = sio.loadmat('validation/ref/o2.mat', squeeze_me=True, struct_as_record=False)["o2"]
m = Model.from_text(rs_to_mod("validation/matlab/o2.rs")); s = m.solve()
params, ss = s.params, s.ss; sy = m.first_order_system(params, ss); h, n, ne = m.h, m.n, 1
st = _states(sy); k = st.size; nz = k + 1 + ne
order = [m.endogenous.index(v) for v in d.endo]; zmap = list(range(nz))
ref = np.stack([np.array(d.Tzz[r]).reshape(n, nz, nz) for r in range(h)])

def solve(gap_state, gap_level, lead_point):
    G = np.zeros((h, n, nz)); G[:, :, :k] = s.T[:, :, st]; G[:, :, k] = s.c; G[:, :, k + 1:] = s.R * s.sigma
    E = np.zeros((nz, nz)); E[k + 1, k] = 1.0
    idx, fn = _hessian_entries(m); nv = 3 * n + ne
    A = np.zeros((h, n, nz * nz)); Cq = np.zeros((h, h, nz * nz, nz * nz)); U = _U(sy, s.T)
    Apq = np.zeros((h, h, n, n))
    for r in range(h):
        for q in range(h):
            Hm = np.zeros((nz, nz)); Hm[:k] = G[r][st]; Hm[k, k] = 1
            if gap_state: Hm[:k, k] += (ss[r] - ss[q])[st]
            Cq[r, q] = np.kron(Hm, Hm) + np.kron(E, E)
            yp = ss[q] if lead_point else ss[r]
            a = (yp, ss[r], ss[r], np.zeros(ne), ss[r], params[r])
            Apq[r, q] = np.array(m._jac[0](*a), float).reshape(n, n)
            vals = np.asarray(fn(*a), float)
            V0 = np.zeros((nv, nz)); V0[:n] = G[q] @ Hm; V0[n:2 * n] = G[r]; V0[2 * n + st, np.arange(k)] = 1; V0[3 * n, k + 1] = s.sigma[0]
            if gap_level: V0[:n, k] += ss[q] - ss[r]
            V1 = np.zeros((nv, nz)); V1[:n] = G[q] @ E
            for (eq, i, j), val in zip(idx, vals):
                term = np.outer(V0[i], V0[j]) + np.outer(V1[i], V1[j])
                if i != j: term = term + term.T
                A[r, eq] += sy.Q[r, q] * val * term.ravel()
    m2 = nz * nz; size = h * n * m2
    def apply(x):
        X = x.reshape(h, n, m2); out = np.zeros_like(X)
        for r in range(h):
            out[r] = U[r] @ X[r]
            for q in range(h): out[r] += sy.Q[r, q] * Apq[r, q] @ X[q] @ Cq[r, q]
        return out.ravel()
    big = np.column_stack([apply(e) for e in np.eye(size)])
    G2 = np.linalg.solve(big, -A.ravel()).reshape(h, n, nz, nz)
    return G2[:, order][:, :, zmap][:, :, :, zmap]

for name, args in {"baseline (package)": (0, 0, 0), "state gap": (1, 0, 0), "state gap + level": (1, 1, 0), "level only": (0, 1, 0),
                   "lead at the ss of the future regime": (0, 0, 1), "state gap + future point": (1, 0, 1), "all": (1, 1, 1)}.items():
    G2 = solve(*args); dif = np.abs(G2 - ref)
    sig = zmap.index(k) if False else 2
    print(f"{name:32s} max diff total {dif.max():.3e} | without sigma {np.delete(np.delete(dif, 2, 2), 2, 3).max():.2e} | x-sigma {dif[:, :, :2, 2].max():.2e} | sigma-sigma {dif[:, :, 2, 2].max():.2e}")
print("RISE sigma-sigma", ref[:, :, 2, 2], "\nRISE x-sigma", ref[:, :, 0, 2], ref[:, :, 1, 2])
