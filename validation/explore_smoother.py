"""Exploration: which backward recursion reproduces the RISE smoothed states on the MS model with a latent state?"""
import pathlib, numpy as np, pandas as pd, scipy.io as sio
from scipy import linalg
from msdsge import Model
from msdsge.filtering import ergodic_distribution
exec(pathlib.Path("tests/test_filter.py").read_text().split("def test_")[0])
r = sio.loadmat('validation/ref/smooth_ms.mat', squeeze_me=True, struct_as_record=False)
s = Model.from_text(MS).solve(); y = pd.read_csv("validation/ref/filter_ms_data.csv")["x"].to_numpy()
T, RS, Q = s.T, s.R * s.sigma, s.Q; h, n = 2, 2; ix = s.model.endogenous.index("x"); iz = s.model.endogenous.index("z")
RR = np.einsum("rik,rjk->rij", RS, RS); pi = ergodic_distribution(Q)
P0 = linalg.solve_discrete_lyapunov(np.einsum("r,rij->ij", pi, T), (lambda Rb: Rb @ Rb.T)(np.einsum("r,rij->ij", pi, RS)))
nT = len(y)
a = np.zeros((nT, h, n)); P = np.zeros((nT, h, n, n)); au = np.zeros((nT, h, n)); Pu = np.zeros((nT, h, n, n))
v = np.zeros((nT, h)); F = np.zeros((nT, h)); K = np.zeros((nT, h, n)); post = np.zeros((nT, h)); pred = np.zeros((nT, h)); W = np.zeros((nT, h, h))
mu = pi.copy(); ac = np.zeros((h, n)); Pc = np.stack([P0] * h)
for t in range(nT):
    a[t], P[t], pred[t] = ac, Pc, mu
    lam = np.zeros(h)
    for j in range(h):
        F[t, j] = Pc[j][ix, ix]; v[t, j] = y[t] - ac[j][ix]; K[t, j] = Pc[j][:, ix] / F[t, j]
        au[t, j] = ac[j] + K[t, j] * v[t, j]; Pu[t, j] = Pc[j] - np.outer(K[t, j], Pc[j][ix, :])
        lam[j] = -0.5 * (np.log(2 * np.pi * F[t, j]) + v[t, j] ** 2 / F[t, j])
    w = mu * np.exp(lam); post[t] = w / w.sum()
    joint = Q * post[t][:, None]; mu = joint.sum(0); W[t] = joint / mu   # W[t][i,k] = Pr(s_t=i | s_{t+1}=k, Y_t)
    ac = np.stack([T[k] @ (W[t][:, k] @ au[t]) for k in range(h)])
    Pc = np.stack([T[k] @ np.einsum("i,ijl->jl", W[t][:, k], Pu[t]) @ T[k].T + RR[k] for k in range(h)])
sm = np.zeros((nT, h)); sm[-1] = post[-1]
for t in range(nT - 2, -1, -1):
    sm[t] = post[t] * (Q @ (sm[t + 1] / pred[t + 1])); sm[t] /= sm[t].sum()
print("smoothed prob diff", np.max(np.abs(sm[:, 0] - r["smo1"])))
Zv = np.zeros(n); Zv[ix] = 1

def smooth(variant):
    xs = np.zeros((nT, h, n)); es = np.zeros((nT, h, RS.shape[2])); rn = np.zeros((h, n))
    for t in range(nT - 1, -1, -1):
        rc = np.zeros((h, n))
        for j in range(h):
            if t + 1 < nT:
                if variant == "smoothed prob":      w = Q[j] * sm[t + 1] / pred[t + 1]; w = w / w.sum()
                elif variant == "own regime":    w = np.eye(h)[j]
                elif variant == "Q":                 w = Q[j]
                elif variant == "filter adjoint": w = W[t][j]
                elif variant == "weighted adjoint": w = W[t][j] * sm[t + 1] / np.maximum(sm[t, j], 1e-300) * 0 + W[t][j] * pred[t + 1] / post[t, j] * sm[t + 1] / pred[t + 1]
                mj = sum(w[k] * (T[k].T @ rn[k]) for k in range(h))
            else: mj = np.zeros(n)
            rc[j] = Zv * v[t, j] / F[t, j] + mj - Zv * (K[t, j] @ mj)
            xs[t, j] = a[t, j] + P[t, j] @ rc[j]; es[t, j] = RS[j].T @ rc[j]
        rn = rc
    return xs, es

for variant in ("smoothed prob", "own regime", "Q", "filter adjoint", "weighted adjoint"):
    xs, es = smooth(variant)
    ez = np.einsum("tj,tj->t", sm, xs[:, :, iz])
    print(f"{variant:20s} expected z: {np.max(np.abs(ez - r['smo_z'])):.2e} | z by regime: {np.max(np.abs(xs[:, :, iz] - r['smo_z_r'])):.2e} | e by regime (t>0): {np.max(np.abs(es[1:, :, 0] - r['smo_e_r'][1:])):.2e}")
print("RISE z by regime, first rows\n", r["smo_z_r"][:4], "\nRISE expected", r["smo_z"][:4], "\nsm", sm[:4])
