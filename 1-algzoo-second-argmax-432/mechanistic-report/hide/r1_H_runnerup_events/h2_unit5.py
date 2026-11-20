"""H2. Pre-activation of unit 5 (and 9) at t+1 after an N event at t.

Exact identity (any step):
  z_u(t+1) = w_u x_{t+1} + sum_{j in COMP} kappa_uj [theta_j(t) - x_t]_+ + Omega_u(t) + sum_{v in PULSE} W[u,v] h_t[v]
  kappa_uj = W[u,j] |w_j|,   Omega_u(t) = sum_{i in WAVE} W[u,i] h_t[i].
"""
from hcommon import *

kap = {u: {j: W[u, j] * abs(w[j]) for j in COMP} for u in (5, 9)}
print("[EXACT] kappa_5j = W[5,j]|w_j| :", {j: round(v, 3) for j, v in kap[5].items()})
print("[EXACT] kappa_9j = W[9,j]|w_j| :", {j: round(v, 3) for j, v in kap[9].items()})
print("[EXACT] sum_wave W[5,i] w_i =", round(W[5, WAVE] @ w[WAVE], 4), "  sum_wave W[9,i] w_i =", round(W[9, WAVE] @ w[WAVE], 4))
print("[EXACT] W[5,PULSE] =", np.round(W[5, PULSE], 3), " W[9,PULSE] =", np.round(W[9, PULSE], 3), " w_5, w_9 =", round(w[5], 3), round(w[9], 3))
print("[EXACT] c_x with all 13 ordinary units on: unit 5:", round(W[5, ORD] @ w[ORD], 3), " unit 9:", round(W[9, ORD] @ w[ORD], 3))

cols = {}
for seed in (1004, 1005, 1006):
    x = sample(400_000, seed)
    e = Ev(x)
    # identity check on all steps t = 0..8
    relu = lambda v: np.maximum(v, 0)
    for u in (5, 9):
        rhs = w[u] * x[:, 1:] + e.H[:, :-1][:, :, WAVE] @ W[u, WAVE] + e.H[:, :-1][:, :, PULSE] @ W[u, PULSE]
        for j in COMP:
            rhs = rhs + kap[u][j] * relu(e.th[j][:, :-1] - x[:, :-1])
        print(f"seed {seed}: identity for unit {u}: max |z - rhs| = {np.abs(e.Z[:, 1:, u] - rhs).max():.2e}")
    ok = (e.tt >= 2) & (e.tt <= 8) & (e.So > 0)
    isN = ok & (x > e.So) & (x < e.Mo) & ~e.G[:, :, 4]
    b, t = np.nonzero(isN)
    d = dict(a=e.a1[b, t], a2=e.a2[b, t], t=t, M=e.Mo[b, t], S=e.So[b, t], x=x[b, t], xn=x[b, t + 1],
             z5=e.Z[b, t + 1, 5], z9=e.Z[b, t + 1, 9], h5t=e.H[b, t, 5], h9t=e.H[b, t, 9],
             Om5=e.H[b, t][:, WAVE] @ W[5, WAVE], Om9=e.H[b, t][:, WAVE] @ W[9, WAVE],
             Om5lin=(e.Hprev[b, t] @ W.T)[:, WAVE] @ W[5, WAVE],      # wave memory term if no wave unit were clipped, without the x_t part
             cx5=(e.G[b, t] * (W[5] * w)).sum(1), cx9=(e.G[b, t] * (W[9] * w)).sum(1))
    for j in COMP:
        d[f"th{j}"] = e.th[j][b, t]; d[f"on{j}"] = e.G[b, t, j]
    for k, v in d.items():
        cols.setdefault(k, []).append(v)
    del e
d = {k: np.concatenate(v) for k, v in cols.items()}
n = len(d["a"]); print("N events (unit 4 silent, 2<=t<=8, S_old>0): n =", n)
r = (d["x"] - d["S"]) / (d["M"] - d["S"])
y5 = d["z5"] - w[5] * d["xn"]            # what the state at t sends to unit 5
y9 = d["z9"] - w[9] * d["xn"]
pat = d["on7"].astype(int) * 2 + d["on8"].astype(int)
PN = {3: "7on 8on", 2: "7on 8off", 1: "7off 8on", 0: "7off 8off"}

print("\n=== unit 5: regression  z5(t+1) - w5 x_{t+1} ~ cx x_t + cS S_old + cM M   per (age, pattern of comparators 7, 8 at t) ===")
print("  a  pattern    share   cx(exact mean)  cx(fit)   cS(fit)   cM(fit)    R2     | h5(t+1) on  | median z5/(K M)")
store = {}
for a in range(1, 9):
    for p in (3, 2, 1, 0):
        s = (d["a"] == a) & (pat == p)
        if s.sum() < 1000: continue
        bb, R2 = lsq(y5[s], [d["x"][s], d["S"][s], d["M"][s]])
        store[(a, p)] = (bb, R2)
        print(f"  {a}  {PN[p]:9s}  {s.sum() / (d['a'] == a).sum():.3f}   {d['cx5'][s].mean():8.3f}      {bb[0]:8.3f}  {bb[1]:8.3f}  {bb[2]:8.3f}   {R2:.4f}  |  {(d['z5'][s] > 0).mean():.4f}     | {np.median(d['z5'][s] / (K * d['M'][s])):.3f}")

print("\n=== building blocks: per age, regress each state term on (S_old, M) over the same N events (no intercept) ===")
print("  a  |  theta_7: bS, bM, R2      |  theta_8: bS, bM, R2      | Omega5 memory (no x part): bS, bM, R2 | W[5,5]h5(t)+W[5,9]h9(t): bx, bS, bM, R2")
blocks = {}
for a in range(1, 9):
    s = d["a"] == a
    b7, R7 = lsq(d["th7"][s], [d["S"][s], d["M"][s]])
    b8, R8 = lsq(d["th8"][s], [d["S"][s], d["M"][s]])
    bo, Ro = lsq(d["Om5"][s] - 0 * d["x"][s], [d["x"][s], d["S"][s], d["M"][s]])
    pu = W[5, 5] * d["h5t"][s] + W[5, 9] * d["h9t"][s]
    bp, Rp = lsq(pu, [d["x"][s], d["S"][s], d["M"][s]])
    blocks[a] = (b7, b8, bo, bp)
    print(f"  {a}  | {b7[0]:6.3f} {b7[1]:6.3f} {R7:.4f}  | {b8[0]:6.3f} {b8[1]:6.3f} {R8:.4f}  | Omega5 ~ x,S,M: {bo[0]:6.3f} {bo[1]:6.3f} {bo[2]:6.3f} {Ro:.4f} | {bp[0]:6.3f} {bp[1]:6.3f} {bp[2]:6.3f} {Rp:.4f}")

print("\n=== composed prediction (threshold laws + wave term + pulses) vs direct fit, per (age, pattern) ===")
print("  a  pattern   |  cx: composed / fit  |  cS: composed / fit  |  cM: composed / fit")
for (a, p), (bb, R2) in store.items():
    b7, b8, bo, bp = blocks[a]
    on7, on8 = p >> 1, p & 1
    # comparators 1 and 6 are ignored in the composition (kappa = -0.011, +0.165)
    cx = -kap[5][7] * on7 - kap[5][8] * on8 + bo[0] + bp[0]
    cS = kap[5][7] * on7 * b7[0] + kap[5][8] * on8 * b8[0] + bo[1] + bp[1]
    cM = kap[5][7] * on7 * b7[1] + kap[5][8] * on8 * b8[1] + bo[2] + bp[2]
    print(f"  {a}  {PN[p]:9s} |  {cx:7.3f} / {bb[0]:7.3f}   |  {cS:7.3f} / {bb[1]:7.3f}   |  {cM:7.3f} / {bb[2]:7.3f}")

print("\n=== size of h_{t+1}[5] after an N event, in units of K*M and K*(x_t - S) ===")
print("  a  | median h5/(K M) | median h5/(K (x-S)) | median h5/(K (M-x)) | corr(h5, x-S) corr(h5, M-x) corr(h5, M) corr(h5,S) corr(h5,x)")
for a in range(1, 9):
    s = d["a"] == a
    h5 = np.maximum(d["z5"][s], 0)
    c = lambda v: np.corrcoef(h5, v)[0, 1]
    print(f"  {a}  |  {np.median(h5 / (K * d['M'][s])):6.3f}        |  {np.median(h5 / (K * (d['x'][s] - d['S'][s]))):7.3f}           |  {np.median(h5 / (K * (d['M'][s] - d['x'][s]))):7.3f}        |"
          f"  {c(d['x'][s] - d['S'][s]):6.3f}  {c(d['M'][s] - d['x'][s]):6.3f}  {c(d['M'][s]):6.3f}  {c(d['S'][s]):6.3f}  {c(d['x'][s]):6.3f}")

print("\n=== unit 9 at t+1 after an N event ===")
print(f"  P(unit 9 on at t+1 | N) = {(d['z9'] > 0).mean():.4f};  by pattern of (7, 8, 1) at t:")
p3 = d["on7"].astype(int) * 4 + d["on8"].astype(int) * 2 + d["on1"].astype(int)
for p in range(8):
    s = p3 == p
    if s.sum() < 50: continue
    print(f"    7 {'on ' if p & 4 else 'off'} 8 {'on ' if p & 2 else 'off'} 1 {'on ' if p & 1 else 'off'}: share {s.mean():.4f}  P(9 on) = {(d['z9'][s] > 0).mean():.4f}   median z9/(K M) = {np.median(d['z9'][s] / (K * d['M'][s])):7.3f}")
print("  terms of z9(t+1) (means over N events with unit 9 ON at t+1 / OFF), in units of K*M:")
for lab, s in (("9 on ", d["z9"] > 0), ("9 off", d["z9"] <= 0)):
    KM = K * d["M"][s]
    terms = {f"kappa9_{j}[th-x]+": kap[9][j] * np.maximum(d[f"th{j}"][s] - d["x"][s], 0) / KM for j in COMP}
    terms["Omega9"] = d["Om9"][s] / KM
    terms["W[9,5]h5+W[9,9]h9"] = (W[9, 5] * d["h5t"][s] + W[9, 9] * d["h9t"][s]) / KM
    terms["w9 x_{t+1}"] = w[9] * d["xn"][s] / KM
    print(f"   {lab} (n={s.sum()}): " + "  ".join(f"{k}={np.mean(v):.4f}" for k, v in terms.items()) + f"  total={np.mean(d['z9'][s] / KM):.4f}")
print("  P(unit 9 on at t+1 | N) by age and r:")
RB = np.linspace(0, 1, 6); rb = np.clip(np.digitize(r, RB) - 1, 0, 4)
for a in range(1, 9):
    print(f"   a={a}: " + "  ".join((f"{(d['z9'][(d['a'] == a) & (rb == i)] > 0).mean():.3f}" if ((d['a'] == a) & (rb == i)).sum() > 200 else "  .  ") for i in range(5)))
