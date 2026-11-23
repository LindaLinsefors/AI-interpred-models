"""E31. Are 'level' and 'runner' the natural directions inside the plane spanned by the two slow real modes
(eigenvalues 1.0099 and 0.9735), or would another pair in that plane read max and runner-up more cleanly?
Run:  python -B hide/e31_level_runner_plane.py > hide/outputs/e31_level_runner_plane.txt"""
from common import *
from modes import LEFT, RIGHT, LAM, coords

w4 = w[4]
cosang = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
rl, rr, ll, lr = (np.real(v) for v in (RIGHT["level"], RIGHT["runner"], LEFT["level"], LEFT["runner"]))
print(f"eigenvalues {LAM['level'].real:.4f}, {LAM['runner'].real:.4f}; ratio after 9 steps {(LAM['level'].real / LAM['runner'].real) ** 9:.2f}")
print(f"cos(r_level, r_runner) = {cosang(rl, rr):+.3f};  cos(l_level, l_runner) = {cosang(ll, lr):+.3f}")

x = sample(1_000_000, 3131)
H = run(x); st = running_stats(x); c = coords(H)
lev, run_ = c["level"][:, 1:] / w4, c["runner"][:, 1:] / w4            # coordinates of h_t, t = 1..9
tgt = {"M(t-1)+": np.maximum(st["M1"][:, :-1], 0), "S(t-1)+": np.maximum(st["M2"][:, :-1], 0),
       "M(t)+": np.maximum(st["M1"][:, 1:], 0), "S(t)+": np.maximum(st["M2"][:, 1:], 0)}
Ho = H[:, 1:][..., ORD].reshape(-1, 13)

def fit(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b, 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()

P = np.stack([lev.ravel(), run_.ravel()], 1)
print("\nreadouts a*level + b*runner (no intercept), all states t = 1..9; R2 centred")
print("target      a        b       R2 plane | R2 level only  R2 runner only | R2 all 13 units | cos of best 13-unit readout with the plane")
Q = np.linalg.qr(np.stack([ll, lr], 1))[0]
for k, y in tgt.items():
    y = y.ravel()
    b, r2 = fit(P, y); _, r2l = fit(P[:, :1], y); _, r2r = fit(P[:, 1:], y); b13, r213 = fit(Ho, y)
    print(f"{k:8s} {b[0]:+8.3f} {b[1]:+8.3f}   {r2:.3f}   |    {r2l:.3f}         {r2r:.3f}      |     {r213:.3f}      |  {np.linalg.norm(Q.T @ b13) / np.linalg.norm(b13):.3f}")

print("\nthe same plane fit per step t (does the best direction drift?)   target S(t-1)+ : a, b, b/a, R2")
for t in range(1, 9):
    b, r2 = fit(np.stack([lev[:, t], run_[:, t]], 1), tgt["S(t-1)+"][:, t])
    print(f"  t = {t + 1}: {b[0]:+.3f} {b[1]:+.3f}  ratio {b[0] / b[1]:+.2f}  R2 {r2:.3f}")

print("\nwhat one step writes into a general plane coordinate q = a*level + b*runner (units of w4):")
g = -w
terms = {"unit 4": W[ORD, 4], "unit 5": W[ORD, 5], "unit 9": W[ORD, 9]}
terms.update({f"comp {k} off": np.eye(13)[ORD.index(k)] * g[k] / w4 for k in COMP})
for k, v in terms.items():
    print(f"  {k:10s}: level {ll @ v:+.3f}   runner {lr @ v:+.3f}")
