"""Reviewer R1 (numbers audit): data checks of statements in report.md whose conditions looked narrower
than the text.  N(0,1) inputs, fresh seeds.  Run:  python -B rev02_data.py > rev02_data_out.txt
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT, LAM, coords

np.set_printoptions(precision=3, suppress=True, linewidth=220)
K = w[4]

# ------------------------------------------------------------------ (A) e17 split by k AND by step of the wipe
print("== (A) section 3.5 item 1 / summary: wipe the comparators once at t0 = pM + k (same sample and selection as e17)")
x = sample(300_000, 1717)
st = running_stats(x)
truth = second_argmax(x)
pM = x.argmax(1)


def run_wipe(x, t0, units):
    B = len(x); h = np.zeros((B, N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        sel = np.where(t0 == t)[0]
        if len(sel):
            h[np.ix_(sel, units)] = 0.0
    return (h @ U.T).argmax(1)


sel0 = (st["M2"][:, 9] > 0.2) & (st["M1"][:, 9] - st["M2"][:, 9] > 0.2)
print("   P(answer = max) [P(answer = runner-up)] by k (rows) and step of the wipe t0 (columns); '.' = fewer than 500 sequences")
print("   k |" + "".join(f"   t0={t0}      " for t0 in range(1, 10)) + "|  all t0<=8        t0=9")
tot = {}
for k in range(0, 9):
    t0 = pM + k
    ok = sel0 & (t0 <= 9)
    a = run_wipe(x[ok], t0[ok], COMP)
    tm, tr, tt = pM[ok], truth[ok], t0[ok]
    row = f"   {k} |"
    for s in range(1, 10):
        q = tt == s
        row += f" {np.mean(a[q] == tm[q]):.3f}[{np.mean(a[q] == tr[q]):.3f}]" if q.sum() >= 500 else "      .       "
    q8 = tt <= 8; q9 = tt == 9
    row += f"|  {np.mean(a[q8] == tm[q8]):.3f}[{np.mean(a[q8] == tr[q8]):.3f}] n={q8.sum():6d}"
    row += f"  {np.mean(a[q9] == tm[q9]):.3f}[{np.mean(a[q9] == tr[q9]):.3f}] n={q9.sum()}" if q9.sum() >= 500 else ""
    print(row)
    tot[k] = (a, tm, tr, tt)
# pooled over k >= 1 (a step with no new maximum), t0 <= 8
A_ = np.concatenate([tot[k][0][tot[k][3] <= 8] for k in range(1, 9)])
M_ = np.concatenate([tot[k][1][tot[k][3] <= 8] for k in range(1, 9)])
R_ = np.concatenate([tot[k][2][tot[k][3] <= 8] for k in range(1, 9)])
print(f"   pooled over k = 1..8, wipe at step <= 8: P(answer = max) = {np.mean(A_ == M_):.3f}, P(answer = runner-up) = {np.mean(A_ == R_):.3f}, other = {np.mean((A_ != M_) & (A_ != R_)):.3f}, n = {len(A_)}")

# ------------------------------------------------------------------ common sample for the rest
x = sample(600_000, 9191)
H, Z = run(x, return_pre=True)
st = running_stats(x)
B = len(x); ar = np.arange(B)
M, S, a1, a2 = st["M1"], st["M2"], st["a1"], st["a2"]
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)
L = H @ U.T
Uo = U[:, ORD]; A = W[np.ix_(ORD, ORD)]
Sig = Uo @ A @ np.linalg.pinv(Uo)

# ------------------------------------------------------------------ (B) Sigma on REAL steps
print("\n== (B) section 3.2: does Sigma predict the network's ACTUAL next scores?  s_(t+1) vs Sigma s_t, positions 0-8, mean over 0-8 removed")
print("   (e10 instead compares U_o A h_t with Sigma U_o h_t: a hypothetical resting step, pulse units left out of s)")


def r2(pred, act, idx):
    p = pred[:, idx] - pred[:, idx].mean(1, keepdims=True)
    a = act[:, idx] - act[:, idx].mean(1, keepdims=True)
    return 1 - ((a - p) ** 2).sum() / (a ** 2).sum()


idx8 = list(range(9)); idx7 = list(range(8))
print("    t -> t+1 | e10's quantity | real step, x_(t+1) < S_t/2, S_t > 0.1 | ... and resting gate pattern at t and t+1 | real step, any x_(t+1) below S_t |   (second number in each cell: positions 0-7 only)")
rest = (Z[:, :, ORD] > 0).all(2) & (Z[:, :, PULSE] <= 0).all(2)
for t in range(2, 9):
    h = H[:, t][:, ORD]
    e10 = r2((h @ Uo.T) @ Sig.T, (h @ A.T) @ Uo.T, idx8)
    pred = L[:, t] @ Sig.T
    q1 = (S[:, t] > 0.1) & (x[:, t + 1] < S[:, t] / 2)
    q2 = q1 & rest[:, t] & rest[:, t + 1]
    q3 = (S[:, t] > 0.1) & (x[:, t + 1] < S[:, t])
    cells = []
    for q in (q1, q2, q3):
        cells.append(f"{r2(pred[q], L[q, t + 1], idx8):.3f} / {r2(pred[q], L[q, t + 1], idx7):.3f} (n={q.sum()})")
    print(f"    {t} -> {t+1} |    {e10:.4f}     | " + " | ".join(cells))

# ------------------------------------------------------------------ (C) m at a new maximum: effect of the dropped conditions
print("\n== (C) section 3.4 'at a new maximum s becomes m': R2 of s_t (older inputs) against the shifted m_(t-1), three variants")
Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
Mv = L - H @ Uc.T


def r2s(a, b, centre=True):
    if centre:
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    sl = (a * b).sum() / (b * b).sum()
    return 1 - ((a - sl * b) ** 2).sum() / (a ** 2).sum(), sl, 1 - ((a - b) ** 2).sum() / (a ** 2).sum()


print("    t | e15b: inputs 0..t-2, old max >= 1 step old, mean removed, free slope | same but slope fixed at 1 ('equal') | inputs 0..t-1 (previous input included) | all four comparators cut at t / not all cut (e15b variant)")
for t in range(3, 10):
    rec = (x[:, t] > M[:, t - 1]) & (M[:, t - 1] > 0.3)
    s1 = rec & (st["p1"][:, t - 1] < t - 1)
    new, old = L[s1, t][:, 9 - t: 8], Mv[s1, t - 1][:, 10 - t: 9]
    a, sl, a_eq = r2s(new, old)
    new2, old2 = L[rec, t][:, 9 - t: 9], Mv[rec, t - 1][:, 10 - t: 10]
    b, slb, _ = r2s(new2, old2)
    allcut = (Z[:, t][:, COMP] <= 0).all(1)
    c1 = r2s(L[s1 & allcut, t][:, 9 - t: 8], Mv[s1 & allcut, t - 1][:, 10 - t: 9])[0]
    c2 = r2s(L[s1 & ~allcut, t][:, 9 - t: 8], Mv[s1 & ~allcut, t - 1][:, 10 - t: 9])[0] if (s1 & ~allcut).sum() > 200 else np.nan
    print(f"    {t} |  R2 {a:.3f} (slope {sl:.2f})  |  R2 {a_eq:.3f}  |  R2 {b:.3f} (slope {slb:.2f})  |  {c1:.3f} / {c2:.3f}   share all cut {np.mean(allcut[s1]):.3f}")

# ------------------------------------------------------------------ (D) comparators cut at a new maximum, by age of the old maximum
print("\n== (D) section 5.4 / summary: 'all four comparators switch off' at a new maximum above a positive old maximum, by age g of the old maximum in h_(t-1)")
rows = []
for g in range(0, 9):
    q = np.zeros((B, 10), bool)
    for t in range(1, 10):
        q[:, t] = (x[:, t] > M[:, t - 1]) & (M[:, t - 1] > 0) & (a1[:, t - 1] == g)
    allcut = (Z[:, :, COMP] <= 0).all(2)
    u4 = Z[:, :, 4] > 0
    rows.append((g, q.sum(), allcut[q].mean(), u4[q].mean()))
print("    g (state age) : n : P(all four cut) : P(unit 4 on)")
for r in rows:
    print(f"    {r[0]} : {r[1]:7d} : {r[2]:.3f} : {r[3]:.5f}")
qall = np.zeros((B, 10), bool)
for t in range(1, 10):
    qall[:, t] = (x[:, t] > M[:, t - 1]) & (M[:, t - 1] > 0)
print(f"    all ages: P(all four cut) = {(Z[:, :, COMP] <= 0).all(2)[qall].mean():.3f}, P(unit 4 on) = {(Z[:, :, 4] > 0)[qall].mean():.5f}, n = {qall.sum()}")

# ------------------------------------------------------------------ (E) unit 2 as 'M'
print("\n== (E) summary / 4.1: h_2 / M (median) by age of the max (rows 1..9) and relative gap (M - S+)/M; states with M > 0.3")
gap = (M - Sp) / np.where(M > 0, M, np.nan)
bins = [(0, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 0.85), (0.85, 1.0001)]
print("    age |" + "".join(f"  ({lo:.2f},{hi:.2f}]" for lo, hi in bins) + " |   all")
for a in range(1, 10):
    row = f"     {a}  |"
    for lo, hi in bins:
        q = (M > 0.3) & (a1 == a) & (gap > lo) & (gap <= hi)
        row += f"     {np.median(H[q][:, 2] / M[q]):6.2f}  "
    q = (M > 0.3) & (a1 == a)
    row += f" | {np.median(H[q][:, 2] / M[q]):6.2f}"
    print(row)
q = (M > 0.3) & (a1 >= 1)
r = H[q][:, 2] / M[q]
print(f"    all ages >= 1: median {np.median(r):.2f}, 5%..95% {np.percentile(r, 5):.2f}..{np.percentile(r, 95):.2f};  R2 of h_2 = 17.3 M (no refit): {1 - ((H[q][:, 2] - 17.3 * M[q]) ** 2).sum() / ((H[q][:, 2] - H[q][:, 2].mean()) ** 2).sum():.3f}")
print(f"    supply b2*h2/M to the thresholds at 0.05*h2: median {np.median(0.05 * r):.2f}, 5%..95% {np.percentile(0.05 * r, 5):.2f}..{np.percentile(0.05 * r, 95):.2f}")

# ------------------------------------------------------------------ (F) threshold law of section 4.1
print("\n== (F) section 4.1 threshold law: theta_j = A_j M + B_j S+ + (M - S+) C_j rho(a) cos(phi(a) + delta_j), a = age of the max in the state that sets the threshold (1..8)")
phi = np.radians([0.3, 36.5, 79.5, 123.3, 166.0, 208.6, 251.0, 294.0, 337.7])       # arm C, a = 1..9 (report rounds these)
phi1 = np.radians([0.7, 33.5, 75.8, 120.0, 163.6, 206.3, 248.8, 291.0])               # arm C one-coefficient law, state ages 1..8
rho1 = np.array([1.058, 1.094, 1.060, 1.031, 1.035, 1.027, 1.026, 1.017])
units = [1, 6, 7, 8, 4]
Amid = np.array([(0.681 + 0.696) / 2, (0.776 + 0.790) / 2, (0.803 + 0.822) / 2, (0.764 + 0.784) / 2, (0.810 + 0.833) / 2])
Bmid = np.array([(0.218 + 0.266) / 2, (0.159 + 0.200) / 2, (0.150 + 0.192) / 2, (0.133 + 0.171) / 2, (0.135 + 0.175) / 2])
Cj = np.array([0.715, 0.397, 0.234, 0.726, 0.160])
dj = np.radians([-154.4, -79.3, -3.9, 125.1, -6.6])
Hs, Ms, Ss, as_ = H[:, :9], M[:, :9], Sp[:, :9], a1[:, :9]
ok = (Ms > 0.3) & (as_ >= 1) & (as_ <= 8)
th = np.stack([-(Hs @ W[j]) / w[j] for j in units], -1)                                 # (B, 9, 5) thresholds for the next input
aa = as_[ok]
pred_rep = (Amid * Ms[ok][:, None] + Bmid * Ss[ok][:, None]
            + (Ms[ok] - Ss[ok])[:, None] * Cj * (1.02 * 0.951 ** (aa - 2))[:, None] * np.cos(phi[aa - 1][:, None] + dj))
pred_c = (Amid * Ms[ok][:, None] + Bmid * Ss[ok][:, None]
          + (Ms[ok] - Ss[ok])[:, None] * Cj * rho1[aa - 1][:, None] * np.cos(phi1[aa - 1][:, None] + dj))
tt = th[ok]
print("    centred R2 of the law against the measured thresholds (no refitting), units 1, 6, 7, 8, 4:")
print("      as printed in the report (rho = 1.02*0.951^(a-2), phi = 0, 37, 80, ...):", np.round(1 - ((tt - pred_rep) ** 2).sum(0) / ((tt - tt.mean(0)) ** 2).sum(0), 3))
print("      arm C's version (rho = 1.06, 1.09, 1.06, 1.03, 1.04, 1.03, 1.03, 1.02; phi = 1, 34, 76, ...):", np.round(1 - ((tt - pred_c) ** 2).sum(0) / ((tt - tt.mean(0)) ** 2).sum(0), 3))
print("    median threshold position (theta - S+)/(M - S+) by state age 1..8, states with S+ > 0 and gap/M > 0.3:  measured | report's law | arm C's law")
sel = (Ss[ok] > 0) & ((Ms[ok] - Ss[ok]) / Ms[ok] > 0.3)
pos = lambda t_: (t_ - Ss[ok][:, None]) / (Ms[ok] - Ss[ok])[:, None]
for k, j in enumerate(units):
    line = f"      unit {j}: "
    for a in range(1, 9):
        q = sel & (aa == a)
        line += f" {np.median(pos(tt)[q, k]):5.2f}|{np.median(pos(pred_rep)[q, k]):5.2f}|{np.median(pos(pred_c)[q, k]):5.2f} "
    print(line)

# ------------------------------------------------------------------ (G) figure panel f and table 3.1
print("\n== (G) figure panel (f): fraction of states whose top score is the runner-up so far, with and without the condition S_t > 0 (the caption says 'fraction of states')")
for t in range(1, 10):
    valid = np.arange(10) >= 9 - t
    ps = np.where(valid, L[:, t], -np.inf).argmax(1) + t - 9
    sp = S[:, t] > 0
    print(f"    t={t}: given S_t > 0: {np.mean(ps[sp] == st['p2'][sp, t]):.3f}   all states: {np.mean(ps == st['p2'][:, t]):.3f}   share of states with S_t > 0: {sp.mean():.3f}")

# ------------------------------------------------------------------ (H) unit 5 / unit 4 event rates quoted in section 5
print("\n== (H) section 5 parenthesis: firing rates after events (steps t = 1..8 for the delayed ones)")
rec = np.zeros((B, 10), bool); nru = np.zeros((B, 10), bool); low = np.zeros((B, 10), bool)
for t in range(1, 10):
    rec[:, t] = x[:, t] > M[:, t - 1]
    nru[:, t] = (x[:, t] < M[:, t - 1]) & (x[:, t] > np.where(np.isfinite(S[:, t - 1]), S[:, t - 1], -np.inf))
    low[:, t] = np.isfinite(S[:, t - 1]) & (x[:, t] < S[:, t - 1])
pos_ = x > 0
u4 = Z[:, :, 4] > 0
u5n = np.zeros((B, 10), bool); u5n[:, :9] = Z[:, 1:, 5] > 0
t8 = np.zeros((B, 10), bool); t8[:, 1:9] = True
print(f"    unit 4 on | positive new max: {u4[rec & pos_].mean():.5f};  | negative new max: {u4[rec & ~pos_].mean():.5f};  | new max above a positive old max: {u4[rec & (np.roll(M, 1, 1) > 0)][:].mean():.5f}")
print(f"    unit 5 on at t+1 | positive new max: {u5n[rec & pos_ & t8].mean():.4f};  positive new runner-up: {u5n[nru & pos_ & t8].mean():.4f};  input below the runner-up: {u5n[low & t8].mean():.4f};  below a positive runner-up: {u5n[low & t8 & (np.roll(S, 1, 1) > 0)].mean():.4f}")
print(f"    shares of steps 1..9: new max {rec[:, 1:].mean():.4f}, new runner-up {nru[:, 1:].mean():.4f}, below {low[:, 1:].mean():.4f}")
