"""B9. More checks.
(1) F2 exactness for unit 4 and the comparators (gate == threshold rule) on data.
(2) Thresholds of unit 4 and of the comparators one step after a record, network vs oracle-size run: is the sag carried over?
(3) Unit 2 cut off from everything except itself and unit 4 (leaky integrator made real): thresholds and accuracy.
(4) What negative inputs leave in the level and runner coordinates (exact coefficients and data).
(5) Accuracy with <= 1 positive input on Unif(-1,1).
Seeds 1090..1094.
"""
from blib import *

# ------------------------------------------------------------------ (1)
x = sample(400_000, 1090)
H, Z = run(x, return_pre=True)
Hprev = np.concatenate([np.zeros_like(H[:, :1]), H[:, :-1]], 1)
ok4 = ((Z[:, :, 4] > 0) == (x > -(Hprev @ W[4]) / w[4])).mean()
okc = {j: ((Z[:, :, j] > 0) == (x < -(Hprev @ W[j]) / w[j])).mean() for j in COMP}
print("(1) [MEASURED n = 400k x 10 steps, seed 1090] gate == threshold rule: unit 4:", ok4, " comparators:", okc)
Hh, J = traces(x[:2000])
print("    F1: max |h_t - sum_p J[t,p] x_p| on 2000 sequences:", np.abs(np.einsum("btpu,bp->btu", J, x[:2000]) - Hh).max())

# ------------------------------------------------------------------ (2)
print("\n(2) thresholds one step after a record at s (for x_{s+1}), network vs oracle-size run  [INTERVENTION n = 400k, seed 1091]")
x = sample(400_000, 1091)
st = prefix_stats(x)
Hn = run(x)


def run_oracle(x):
    B = len(x); h = np.zeros((B, N)); Mp = np.zeros(B); Hh = np.empty((B, 10, N))
    for t in range(10):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        h[:, 4] = W4 * pos(x[:, t] - Mp)
        Hh[:, t] = h
        Mp = np.maximum(Mp, x[:, t])
    return Hh


Ho = run_oracle(x)
M, a = st["M1"], st["a1"]
Mold = pos(np.where(np.isfinite(st["M1prev"]), st["M1prev"], 0))
tt = np.arange(10)[None, :]
rec = (a == 0) & (tt >= 1) & (tt <= 8) & (Mold > 0.2)
ii, ss = np.where(rec)
thr = lambda Hx, j, i_, s_: -(Hx[i_, s_] @ W[j]) / w[j]
print("    direct kick of unit 4 on each threshold at s+1, per unit of (x_s - theta_4): c_j = -W[j,4]*w_4/w_j:",
      {j: round(-W[j, 4] * W4 / w[j], 4) for j in [4] + COMP})
print("    threshold | run | coefficient on sag = M_old - theta_4(s) | on M_new | on M_old | R2c")
for j in [4] + COMP:
    for name, Hx in (("network", Hn), ("oracle", Ho)):
        sag = Mold[ii, ss] - thr(Hx, 4, ii, ss - 1)
        y = thr(Hx, j, ii, ss)
        A_ = np.column_stack([sag, M[ii, ss], Mold[ii, ss]])
        b, *_ = np.linalg.lstsq(A_, y, rcond=None)
        r2 = 1 - ((y - A_ @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"    theta_{j}(s+1) | {name} | {b[0]:+.4f} | {b[1]:+.4f} | {b[2]:+.4f} | {r2:.4f}")

# ------------------------------------------------------------------ (3)
print("\n(3) unit 2 cut off from all inputs except itself and unit 4 (and the direct input)  [INTERVENTION n = 400k, seed 1092]")
x = sample(400_000, 1092)
sa = second_argmax(x)
st = prefix_stats(x)
Hn = run(x)
variants = {}
W2 = W.copy(); keep = W2[2, [2, 4]].copy(); W2[2, :] = 0; W2[2, [2, 4]] = keep
variants["row 2 = (0.969 self, 1.731 from unit 4) only"] = W2
W3 = W2.copy(); W3[2, 2] = 1.0
variants["same, with the self-weight set to 1 (no leak)"] = W3
W4_ = W.copy(); W4_[2, 9] = 0
variants["only the weight from unit 9 removed"] = W4_
W5_ = W.copy(); W5_[2, COMP] = 0
variants["only the weights from the comparators removed"] = W5_
W6_ = W.copy(); W6_[2, [u for u in WAVE if u != 2]] = 0
variants["only the weights from the other wave units removed"] = W6_
print(f"    network accuracy {(logits(Hn).argmax(1) == sa).mean():.4f}")
Mx, ax = st["M1"][:, :-1], st["a1"][:, :-1]
print("    variant | accuracy | agreement | theta_4/M slope by age of the max 1, 3, 5, 8 (network: "
      + ", ".join(f"{(theta4(Hn[:, :-1])[(ax == g) & (Mx > 0)] * Mx[(ax == g) & (Mx > 0)]).sum() / (Mx[(ax == g) & (Mx > 0)] ** 2).sum():.3f}" for g in (1, 3, 5, 8))
      + ") | h_2/(10.157 M) slope at ages 1, 3, 5, 8 | P(unit 4 on | not a positive-part record)")
Mprev_p = pos(np.where(np.isfinite(st["M1prev"]), st["M1prev"], 0))
for name, Wv in variants.items():
    Hv, Zv = run(x, W=Wv, return_pre=True)
    ans = logits(Hv).argmax(1)
    thv = -(Hv[:, :-1] @ Wv[4]) / w[4]
    sl = lambda y, g: (y[(ax == g) & (Mx > 0)] * Mx[(ax == g) & (Mx > 0)]).sum() / (Mx[(ax == g) & (Mx > 0)] ** 2).sum()
    fa = (Zv[:, :, 4] > 0)[~(x > Mprev_p)].mean()
    print(f"    {name} | {(ans == sa).mean():.4f} | {(ans == logits(Hn).argmax(1)).mean():.4f} | " + ", ".join(f"{sl(thv, g):.3f}" for g in (1, 3, 5, 8))
          + " | " + ", ".join(f"{sl(Hv[:, :-1, 2] / W4, g):.3f}" for g in (1, 3, 5, 8)) + f" | {fa:.4f}")

# ------------------------------------------------------------------ (4)
print("\n(4) what a sub-threshold input leaves in the slow modes")
print("    [EXACT] per unit of x_t when all 13 ordinary units stay on: level += L_level.w[ORD] =", round(float((LEFT['level'] @ w[ORD]).real), 4),
      "; runner += L_runner.w[ORD] =", round(float((LEFT['runner'] @ w[ORD]).real), 4))
print("    their effect on theta_4 per unit x: level", round(float((C4['level'] * (LEFT['level'] @ w[ORD])).real), 5),
      "; runner", round(float((C4['runner'] * (LEFT['runner'] @ w[ORD])).real), 5), " (opposite signs, nearly cancel)")
rng = np.random.default_rng(1093)
xn = -np.abs(rng.standard_normal((200_000, 10)))
Hn_ = run(xn)
m = coords(Hn_)
print("    [MEASURED n = 200k all-negative sequences, seed 1093] regression of the coordinates at t = 9 (divided by 10.157) on the ten inputs:")
for nme in ("level", "runner"):
    y = np.real(m[nme][:, 9]) / W4
    A_ = np.column_stack([xn, np.ones(len(xn))])
    b, *_ = np.linalg.lstsq(A_, y, rcond=None)
    r2 = 1 - ((y - A_ @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    lam_ = LAM[nme].real
    pred = (LEFT[nme] @ w[ORD]).real / W4 * lam_ ** (9 - np.arange(10))
    print(f"      {nme}: coefficients on x_0..x_9: {np.round(b[:10], 4)}  intercept {b[10]:+.4f}  R2 {r2:.3f}")
    print(f"      {'':{len(nme)}s}  free-flow prediction L.w*lam^(9-p)/10.157: {np.round(pred, 4)}")

# ------------------------------------------------------------------ (5)
print("\n(5) accuracy with k positive inputs, Unif(-1,1) conditioned on the signs  [MEASURED 200k per k, seed 1094+k]")
for k in (0, 1, 2, 3):
    rng = np.random.default_rng(1094 + k)
    mag = rng.uniform(0, 1, (200_000, 10))
    order = np.argsort(rng.random((200_000, 10)), axis=1)
    xu = mag * np.where(order < k, 1.0, -1.0)
    print(f"    k = {k}: accuracy {(logits(run(xu)).argmax(1) == second_argmax(xu)).mean():.4f}")
