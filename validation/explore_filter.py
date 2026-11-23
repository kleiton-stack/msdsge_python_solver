"""Exploration: which recursion reproduces the RISE filter (2019 commit) on the test MS model?"""
import pathlib, numpy as np, pandas as pd, scipy.io as sio
from scipy import linalg
from msdsge import Model
from msdsge.filtering import ergodic_distribution, _collapse
exec(pathlib.Path("tests/test_filter.py").read_text().split("def test_")[0])
r = sio.loadmat('validation/ref/filter.mat', squeeze_me=True, struct_as_record=False)["ms"]
s = Model.from_text(MS).solve(); y = pd.read_csv("validation/ref/filter_ms_data.csv")["x"].to_numpy()
T, RS, Q = s.T, s.R * s.sigma, s.Q; h, n = 2, 2; iz = s.model.endogenous.index("x")
RR = np.einsum("rik,rjk->rij", RS, RS); pi = ergodic_distribution(Q)
inc = np.ravel(r.Incr)
F0 = None
# variance implied by the first observation
print("incr[0] RISE:", inc[0], "y0", y[0])
cands = {
 "lyap(T ergodic, RR ergodic)": linalg.solve_discrete_lyapunov(np.einsum("r,rij->ij", pi, T), np.einsum("r,rij->ij", pi, RR)),
 "mean of the per-regime lyap": np.einsum("r,rij->ij", pi, np.stack([linalg.solve_discrete_lyapunov(T[j], RR[j]) for j in range(h)])),
}
Tb = np.einsum("r,rij->ij", pi, T); Rb = np.einsum("r,rij->ij", pi, RS)
cands["lyap(T ergodic, R ergodic)"] = linalg.solve_discrete_lyapunov(Tb, Rb@Rb.T)
# exact second moment of the switching system: V_j = T_j (sum_i q_ij V_i) T_j' + pi_j RR_j
m2 = n*n; big = np.zeros((h*m2, h*m2)); rhs = np.zeros(h*m2)
for j in range(h):
    rhs[j*m2:(j+1)*m2] = pi[j]*RR[j].ravel()
    for i in range(h):
        big[j*m2:(j+1)*m2, i*m2:(i+1)*m2] = Q[i, j]*np.kron(T[j], T[j])
V = np.linalg.solve(np.eye(h*m2)-big, rhs).reshape(h, n, n)
cands["exact second moment (sum)"] = V.sum(0)
for k, P in cands.items(): print(f"  {k}: P_xx = {P[iz, iz]:.10f}  ll0 = {-0.5*(np.log(2*np.pi*P[iz,iz]) + y[0]**2/P[iz,iz]):.10f}")

def upd(a, P, yt):
    F = P[iz, iz]; v = yt - a[iz]; K = P[:, iz]/F
    return a + K*v, P - np.outer(K, P[iz, :]), -0.5*(np.log(2*np.pi*F) + v*v/F)

def run(P0, variant):
    mu = pi.copy(); a = np.zeros((h, n)); P = np.stack([P0]*h); out = []; probs = []
    # a, P: predicted by regime at t
    for t in range(len(y)):
        au, Pu, lam = np.zeros_like(a), np.zeros_like(P), np.zeros(h)
        for j in range(h): au[j], Pu[j], lam[j] = upd(a[j], P[j], y[t])
        w = mu*np.exp(lam); out.append(np.log(w.sum())); post = w/w.sum(); probs.append(post)
        joint = Q*post[:, None]; mu = joint.sum(0)
        for j in range(h):
            wj = joint[:, j]/mu[j]
            if variant == "mix-then-predict":
                a0, P0_ = _collapse(wj, au, Pu); a[j] = T[j]@a0; P[j] = T[j]@P0_@T[j].T + RR[j]
            elif variant == "mix without dispersion":
                a0 = wj@au; P0_ = np.einsum("k,kij->ij", wj, Pu); a[j] = T[j]@a0; P[j] = T[j]@P0_@T[j].T + RR[j]
            elif variant == "post weights, without dispersion":
                a0 = post@au; P0_ = np.einsum("k,kij->ij", post, Pu); a[j] = T[j]@a0; P[j] = T[j]@P0_@T[j].T + RR[j]
            elif variant == "post weights, with dispersion":
                a0, P0_ = _collapse(post, au, Pu); a[j] = T[j]@a0; P[j] = T[j]@P0_@T[j].T + RR[j]
            elif variant == "no mixing (own regime)":
                a[j] = T[j]@au[j]; P[j] = T[j]@Pu[j]@T[j].T + RR[j]
            elif variant == "predict with origin regime i":
                aa = np.stack([T[i]@au[i] for i in range(h)]); PP = np.stack([T[i]@Pu[i]@T[i].T + RR[i] for i in range(h)])
                a[j], P[j] = _collapse(wj, aa, PP)
    return np.array(out), np.array(probs)

upd_ref = np.column_stack([r.updated_regime_probabilities__regime_1, r.updated_regime_probabilities__regime_2])
for k, P0 in cands.items():
    if "R ergodic)" not in k: continue
    for v in ("mix-then-predict", "mix without dispersion", "post weights, without dispersion", "post weights, with dispersion", "no mixing (own regime)"):
        o, pr = run(P0, v)
        print(f"{k:32s} | {v:32s} | loglik {o.sum():.8f} | max diff incr {np.max(np.abs(o-inc)):.2e} | max diff prob {np.max(np.abs(pr-upd_ref)):.2e}")
print("RISE loglik", r.LogLik)
P0 = cands["lyap(T ergodic, R ergodic)"]
for v in ("mix-then-predict", "mix without dispersion"):
    o, pr = run(P0, v)
    print(v, "\n incr diff", (o-inc)[:6], "\n prob diff", (pr-upd_ref)[:6, 0])
print("P0", P0, "\nfilt RISE", r.filtered_regime_probabilities__regime_1[:4])
o, pr = run(P0, "mix without dispersion"); d = np.abs(o-inc)
big = np.flatnonzero(d > 1e-9); print("n diff", big.size, "first", big[:10], d[big[:10]])
t0 = big[0]; print("RISE probs around", upd_ref[t0-2:t0+2], "mine", pr[t0-2:t0+2], "y", y[t0-2:t0+2])
r0 = sio.loadmat('validation/ref/filter_ms_tol0.mat', squeeze_me=True, struct_as_record=False)["tol0"]
print("RISE riccati_tol=0: loglik", r0.LogLik, "| mine (mix without dispersion)", o.sum(), "| max diff incr", np.max(np.abs(o-np.ravel(r0.Incr))))

def run_ric(P0, tol, rule):
    mu = pi.copy(); a = np.zeros((h, n)); P = np.stack([P0]*h); out = []; frozen = False; K = np.zeros((h, n)); F = np.zeros(h); Pu = np.zeros_like(P)
    for t in range(len(y)):
        au, lam = np.zeros_like(a), np.zeros(h)
        for j in range(h):
            if not frozen:
                F[j] = P[j][iz, iz]; K[j] = P[j][:, iz]/F[j]; Pu[j] = P[j] - np.outer(K[j], P[j][iz, :])
            v = y[t] - a[j][iz]; au[j] = a[j] + K[j]*v; lam[j] = -0.5*(np.log(2*np.pi*F[j]) + v*v/F[j])
        w = mu*np.exp(lam); out.append(np.log(w.sum())); post = w/w.sum()
        joint = Q*post[:, None]; mu = joint.sum(0)
        Pn = P.copy()
        for j in range(h):
            wj = joint[:, j]/mu[j]
            a[j] = T[j]@(wj@au)
            if not frozen:
                Pn[j] = T[j]@np.einsum("k,kij->ij", wj, Pu)@T[j].T + RR[j]
        if not frozen:
            if rule == "abs": frozen = np.max(np.abs(Pn-P)) < tol
            elif rule == "rel": frozen = np.max(np.abs(Pn-P))/max(np.max(np.abs(P)),1e-300) < tol
            P = Pn
            if frozen: print("  froze at t =", t, rule)
    return np.array(out)
for rule in ("abs", "rel"):
    oo = run_ric(P0, 1e-6, rule); print(rule, "loglik", oo.sum(), "RISE default", r.LogLik, "max diff", np.max(np.abs(oo-inc)))
