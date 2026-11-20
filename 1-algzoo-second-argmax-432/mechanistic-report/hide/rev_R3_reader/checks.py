"""Reviewer R3 (cold reader): numerical checks of points raised while reading report.md.
Writes checks_out.txt beside this script. Run with:  python -B checks.py
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from pathlib import Path
import numpy as np
from common import *            # w, W, U, COMP, PULSE, WAVE, ORD, run, sample, running_stats, second_argmax
import modes

OUT = Path(__file__).resolve().parent / "checks_out.txt"
lines = []
def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    lines.append(s)

np.set_printoptions(precision=3, suppress=True, linewidth=220)
g = -w
K = w[4]

# ---------------------------------------------------------------- M1 zero padding
P("=== M1. Zero-padding identity: h_t(x_0..x_t) == h_9(0,..,0,x_0..x_t) ===")
x = sample(400_000, seed=4242)
H = run(x)
st = running_stats(x)
maxerr = 0.0
for t in range(10):
    xp = np.zeros_like(x)
    xp[:, 9 - t:] = x[:, : t + 1]
    maxerr = max(maxerr, np.abs(run(xp)[:, -1] - H[:, t]).max())
P("max |h_9(padded) - h_t| over t = 0..9 and 400k sequences:", maxerr)
P("t | P(restricted argmax = runner-up | S_t>0) | same, all states | P(unrestricted argmax falls on a padding index | S_t>0) | P(unrestricted argmax = runner-up | S_t>0)")
for t in range(1, 10):
    s = H[:, t] @ U.T
    valid = np.arange(10) >= 9 - t
    pos_r = np.where(valid, s, -np.inf).argmax(1) + t - 9
    pos_u = s.argmax(1) + t - 9
    ok_r = pos_r == st["p2"][:, t]
    pos = st["M2"][:, t] > 0
    P(f"{t} | {ok_r[pos].mean():.3f} | {ok_r.mean():.3f} | {(pos_u[pos] < 0).mean():.4f} | {(pos_u[pos] == st['p2'][pos, t]).mean():.3f}")

# ---------------------------------------------------------------- M2 Sigma
P("\n=== M2. Sigma = Uo A pinv(Uo) ===")
A = W[np.ix_(ORD, ORD)]
Uo = U[:, ORD]
Sig = Uo @ A @ np.linalg.pinv(Uo)
P("relative residual (Frobenius):", np.linalg.norm(Uo @ A - Sig @ Uo) / np.linalg.norm(Uo @ A))
P("singular values of Uo:", np.linalg.svd(Uo, compute_uv=False))
P("column sums over positions, 1^T Uo:", Uo.sum(0))
P("|1^T Uo| / (sqrt(10) * |Uo|_2):", np.linalg.norm(Uo.sum(0)) / (np.sqrt(10) * np.linalg.norm(Uo, 2)))
P("Sigma:")
for i in range(10):
    P("  " + " ".join(f"{v:6.2f}" for v in Sig[i]))
off = Sig.copy()
for i in range(9):
    off[i, i + 1] = 0
P("max |entry| off the shift diagonal, rows 0-6:", np.abs(off[:7]).max(), " rows 0-8:", np.abs(off[:9]).max())
P("row 9:", Sig[9], " mean", Sig[9].mean())
eS = np.linalg.eigvals(Sig)
eA = np.linalg.eigvals(A)
def show(e):
    e = sorted(e, key=lambda v: -abs(v))
    return "  ".join(f"{abs(v):.3f}@{np.degrees(np.angle(v)):.1f}" for v in e if v.imag >= -1e-12)
P("eig(A)     (one of each pair):", show(eA))
P("eig(Sigma) (one of each pair):", show(eS))
# pure 10-slot shift with the same feedback row: is Sigma's spectrum that of a feedback shift register?
Sh = np.zeros((10, 10))
for i in range(9):
    Sh[i, i + 1] = 1
Sh[9] = Sig[9]
P("eig(pure shift + Sigma's row 9):", show(np.linalg.eigvals(Sh)))
# how much of each eigenmode of A does Uo see (norm of Uo r / norm r)
lam, Vv = np.linalg.eig(A)
for k in np.argsort(-abs(lam)):
    if lam[k].imag < -1e-12:
        continue
    r = Vv[:, k]
    P(f"  mode {abs(lam[k]):.3f}@{np.degrees(np.angle(lam[k])):7.2f}: |Uo r|/|r| = {np.linalg.norm(Uo @ r)/np.linalg.norm(r):.3f}")

# ---------------------------------------------------------------- M3 slopes
P("\n=== M3. Direct slope of each logit in x_t through the comparators and through all on-units ===")
P("U[i,COMP] . w[COMP], i = 0..9:", U[:, COMP] @ w[COMP])
P("U[i,ORD] . w[ORD],  i = 0..9:", U[:, ORD] @ w[ORD])
P("index 9 minus mean of indices 0-8 (comparators only):", (U[9, COMP] @ w[COMP]) - (U[:9, COMP] @ w[COMP]).mean())
P("index 9 minus each of 0-8 (all 13 ordinary units on):", (U[9, ORD] @ w[ORD]) - (U[:9, ORD] @ w[ORD]))
P("U[9,4]*w4:", U[9, 4] * w[4], "  U[8,5]:", U[8, 5])

# ---------------------------------------------------------------- M4 threshold law, age convention
P("\n=== M4. Threshold law: which age does phi(a) refer to? ===")
R = modes.RIGHT["clock"]
P("clock eigenvalue:", abs(modes.LAM["clock"]), np.degrees(np.angle(modes.LAM["clock"])))
P("from the weights: theta_j(t+1) contains  Re( z(t) * c_j ),  c_j = 2 W[j,ORD].R_clock / g_j  (unit 4: -2 W[4,ORD].R/w4)")
for j in COMP + [4]:
    c = 2 * (W[j, ORD] @ R) / (g[j] if j != 4 else -w[4])
    P(f"  unit {j}: K*|c_j| = {K*abs(c):.3f}   arg c_j = {np.degrees(np.angle(c)):7.1f} deg")
P("report: C = 0.715, 0.397, 0.234, 0.726, 0.160 ; delta = -154, -79, -4, +125, -7 for units 1, 6, 7, 8, 4")

# data: medians of u_j = (theta_j(t) - S)/(M - S), M,S = order statistics of x_0..x_{t-1}
phi = np.radians([0, 37, 80, 123, 166, 209, 251, 294, 338])          # report, a = 1..9
rho = lambda a: 1.02 * 0.951 ** (a - 2)
C = {1: 0.715, 6: 0.397, 7: 0.234, 8: 0.726, 4: 0.160}
dl = {1: -154, 6: -79, 7: -4, 8: 125, 4: -7}
th = {}
for j in COMP:
    th[j] = np.einsum("btk,k->bt", H[:, :-1], W[j]) / g[j]          # theta_j(t), t = 1..9 -> column t-1
th[4] = -np.einsum("btk,k->bt", H[:, :-1], W[4]) / w[4]
M_ = st["M1"][:, :-1]; S_ = st["M2"][:, :-1]; aM = st["a1"][:, :-1]  # prefix x_0..x_{t-1}; age of max in state h_{t-1}
age_applied = aM + 1                                                 # steps since the max arrived, when x_t is compared
sel = (S_ > 0) & ((M_ - S_) > 0.2 * M_)
P("medians of (theta_j - S)/(M - S) by 'steps since the max arrived when the threshold is applied' (S>0, gap>0.2M):")
med = {}
for j in COMP + [4]:
    u = (th[j] - S_) / (M_ - S_)
    med[j] = np.array([np.median(u[sel & (age_applied == a)]) for a in range(1, 10)])
    P(f"  unit {j}: " + " ".join(f"{v:6.2f}" for v in med[j]))
P("fit  u_j(a) = const_j + C_j rho(a') cos(phi(a') + delta_j)  with report's C, delta, phi, rho; rms residual over applied ages 2..9")
P("   convention X: a' = a (age when applied)      convention Y: a' = a - 1 (age in the state the threshold is read from)")
for j in COMP + [4]:
    out = []
    for shift in (0, 1):
        a_app = np.arange(2, 10)
        a_use = a_app - shift
        pred = C[j] * rho(a_use) * np.cos(phi[a_use - 1] + np.radians(dl[j]))
        resid = med[j][a_app - 1] - pred
        out.append((resid.mean(), resid.std()))
    P(f"  unit {j}: X: const {out[0][0]:5.2f}, rms resid {out[0][1]:.3f}    Y: const {out[1][0]:5.2f}, rms resid {out[1][1]:.3f}")

# ---------------------------------------------------------------- M5 impulse plane
P("\n=== M5. Impulse x = (1,0,..,0): plane offset vs level mode ===")
xi = np.zeros((1, 10)); xi[0, 0] = 1
Hi = run(xi)[0]
X = Hi[2:10][:, ORD]
m = X.mean(0); Xc = X - m
_, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
V2 = Vt[:2]
n = m - V2.T @ (V2 @ m)
P("|n| =", np.linalg.norm(n), "  in-plane radius (rms) =", np.sqrt(((Xc @ V2.T) ** 2).sum(1).mean()),
  "  variance in 2 PCs =", (sv[:2] ** 2).sum() / (sv ** 2).sum())
cm = {k: modes.LEFT[k] @ m for k in modes.NAMES}
P("mode coordinates of the mean state m (per K):", {k: (np.round(abs(v) / K, 3)) for k, v in cm.items()})
comp = {k: (2 * np.real(cm[k] * modes.RIGHT[k]) if modes.IS_COMPLEX[k] else np.real(cm[k] * modes.RIGHT[k])) for k in modes.NAMES}
P("norm of each mode's part of m:", {k: round(float(np.linalg.norm(v)), 2) for k, v in comp.items()}, " |m| =", np.linalg.norm(m))
perp = lambda v: v - V2.T @ (V2 @ v)
P("norm of each mode's part of n (component perpendicular to the plane):",
  {k: round(float(np.linalg.norm(perp(v))), 2) for k, v in comp.items()})
Rl = np.real(modes.RIGHT["level"])
cosang = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
P("cos(n, level vector) =", cosang(n, Rl), "  cos(m, level vector) =", cosang(m, Rl),
  "  cos(n, level part of m) =", cosang(n, comp["level"]))
ci = modes.coords(Hi)
P("impulse, t = 1..9: level/K:", np.round(ci["level"][1:] / K, 4))
P("   ratios level(t+1)/level(t):", np.round(ci["level"][2:] / ci["level"][1:-1], 4))
P("   runner/K:", np.round(ci["runner"][1:] / K, 3))
P("   |clock|/K:", np.round(np.abs(ci["clock"][1:]) / K, 4), " phase:", np.round(np.degrees(np.angle(ci["clock"][1:])) % 360, 1))
P("   report rho(a) = 1.02*0.951^(a-2), a = 1..9:", np.round(rho(np.arange(1, 10)), 4))

# ---------------------------------------------------------------- M6 Theta_5 arithmetic
P("\n=== M6. Theta_5 = 0.01265 runner - 0.02872 level with runner = K(2M+5S), level = 0.93 K M ===")
P("coefficient of M:", 0.01265 * K * 2 - 0.02872 * 0.93 * K, "  coefficient of S:", 0.01265 * K * 5)

# ---------------------------------------------------------------- M7 unit 2's recurrence
P("\n=== M7. What the dots hide in h_2(t+1) = 0.969 h_2 + 1.731 h_4 + ... ===")
P("W[2, :] =", W[2])
contrib = H[:, 2:9, :] * W[2]                    # contribution of each unit at t to pre-activation of unit 2 at t+1
tot = contrib.sum(-1)
grp = {"self (unit 2)": [2], "unit 4": [4], "comparators": COMP, "units 5, 9": [5, 9], "other wave units": [u for u in WAVE if u != 2]}
for k, idx in grp.items():
    P(f"  mean share of unit 2's recurrent input from {k}: {contrib[..., idx].sum(-1).mean() / tot.mean():.3f}")
P("  median h_2 / M over states with max age >= 2 and M > 0:",
  np.median((H[:, :, 2] / st["M1"])[(st["a1"] >= 2) & (st["M1"] > 0)]))

# ---------------------------------------------------------------- M8 new maximum: how often all four comparators off
P("\n=== M8. New positive maximum above a positive old maximum: all four comparators off ===")
newmax = np.zeros_like(x, dtype=bool)
newmax[:, 1:] = (x[:, 1:] > st["M1"][:, :-1]) & (st["M1"][:, :-1] > 0)
alloff = (H[:, :, COMP] == 0).all(-1)
P("P(all four off | new max over positive old max) =", alloff[newmax].mean())
exc = np.zeros_like(x); exc[:, 1:] = (x[:, 1:] - st["M1"][:, :-1]) / np.maximum(st["M1"][:, :-1] - np.maximum(st["M2"][:, :-1], 0), 1e-12)
ok = newmax.copy(); ok[:, 1] = False            # need a runner-up for the old gap
for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1e9)):
    s_ = ok & (exc >= lo) & (exc < hi)
    P(f"   excess over old max / old gap(S+) in [{lo}, {hi}): n = {s_.sum()}, P(all off) = {alloff[s_].mean():.3f}")

OUT.write_text("\n".join(lines), encoding="utf-8")
