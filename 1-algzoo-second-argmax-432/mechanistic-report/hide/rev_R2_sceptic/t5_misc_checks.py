"""Reviewer R2, test 5: three small checks.
(A) Section 3.2: is the R2 = 0.985-0.992 of the shift matrix Sigma a statement about the network's own steps?
    Compare the hypothetical linear step (lead's e10) with the ACTUAL next state on quiet steps.
(B) Section 3.4: how well does V[p] = -60 (M - 0.87 S) cos(42.75 deg (p - pM)) describe single sequences, and how
    large is its error next to the decision margin?
(C) Conditioning of the eigenbasis: are 'clock' (0.951 @ 42.75) and 'short' (0.735 @ 42.99) well separated directions?
(D) B_c 1 ~ 0 restated in activation units (to compare with DraggonZ's 'push of size 23.6 leaves 0.19').
python -B t5_misc_checks.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT, NAMES, IS_COMPLEX, LAM

oi = np.array(ORD)
Uo = U[:, oi]; A = W[np.ix_(oi, oi)]
Sig = Uo @ A @ np.linalg.pinv(Uo)

# ------------------------------------------------------------------ (A)
print("(A) Sigma: rows 0..6, size of the shift entry and of everything else in the row")
for i in range(7):
    row = Sig[i].copy(); sh = row[i + 1]; row[i + 1] = 0
    print(f"   row {i}: Sig[i,i+1] = {sh:.2f};  largest other |entry| = {np.abs(row).max():.2f} (column {np.abs(row).argmax()});  "
          f"norm of the other entries = {np.linalg.norm(row):.2f};  sum of the other entries = {row.sum():+.2f}")
x = sample(400_000, 9701)
H, Z = run(x, return_pre=True)
st = running_stats(x)
S_all = H @ U.T
def r2c(a, b, idx):
    a = a[:, idx] - a[:, idx].mean(1, keepdims=True); b = b[:, idx] - b[:, idx].mean(1, keepdims=True)
    return 1 - ((a - b) ** 2).sum() / (a ** 2).sum()
print("   R2 over logit indices 0..7 (mean removed), target = scores one step later:")
print("    t -> t+1 | hypothetical linear step, all states (lead's e10) | ACTUAL next step, quiet (x_{t+1} < S_t/2, S_t > 0.1): Sigma s_t  /  pure shift | "
      "share of those quiet steps that are exactly resting at t+1 | ACTUAL, resting only | ACTUAL, quiet but not resting")
idx = np.arange(0, 8)
for t in range(2, 9):
    h = H[:, t][:, oi]
    hyp = r2c((h @ A.T) @ Uo.T, S_all[:, t] @ Sig.T, idx)
    quiet = (x[:, t + 1] < 0.5 * st["M2"][:, t]) & (st["M2"][:, t] > 0.1)
    rest = quiet & (Z[:, t + 1][:, oi] > 0).all(1) & (Z[:, t + 1][:, PULSE] <= 0).all(1)
    pred = S_all[:, t] @ Sig.T
    shift = np.zeros_like(pred); shift[:, :9] = S_all[:, t][:, 1:]
    act = S_all[:, t + 1]
    print(f"    {t} -> {t+1} |   {hyp:.3f}   |   {r2c(act[quiet], pred[quiet], idx):.3f}  /  {r2c(act[quiet], shift[quiet], idx):.3f}   |   {rest.sum() / quiet.sum():.3f}   |"
          f"   {r2c(act[rest], pred[rest], idx):.3f}   |   {r2c(act[quiet & ~rest], pred[quiet & ~rest], idx):.3f}")

# ------------------------------------------------------------------ (B)
print("\n(B) the cosine law for V at t = 9 (all four comparators on, S > 0.2, max and runner-up not at index 9)")
t = 9; B = len(x); ar = np.arange(B)
Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
V = H[:, t] @ Uc.T; s_all = S_all[:, t]
M, S = st["M1"][:, t], st["M2"][:, t]; pM, pS = st["p1"][:, t], st["p2"][:, t]
ok = (Z[:, t][:, COMP] > 0).all(1) & (S > 0.2) & (pM < 9) & (pS < 9)
om = np.radians(42.75)
P = np.arange(10)[None, :]
Vp = -60 * (M - 0.87 * S)[:, None] * np.cos(om * (P - pM[:, None]))
for label, idx in (("indices 0..7", np.arange(0, 8)), ("indices 0..8", np.arange(0, 9))):
    a = V[ok][:, idx] - V[ok][:, idx].mean(1, keepdims=True); b = Vp[ok][:, idx] - Vp[ok][:, idx].mean(1, keepdims=True)
    r2 = 1 - ((a - b) ** 2).sum() / (a ** 2).sum()
    per = 1 - ((a - b) ** 2).sum(1) / (a ** 2).sum(1)
    print(f"   pooled R2 of the closed form over {label} (mean over positions removed): {r2:.3f};  per-sequence R2: median {np.median(per):.3f}, "
          f"10th percentile {np.percentile(per, 10):.3f}, share below 0.5: {np.mean(per < 0.5):.3f}")
print("   by age of the max a1 (pooled R2 over indices 0..7):  " + "  ".join(
    f"a1={a}: {1 - (((V - Vp)[ok & (st['a1'][:, t] == a)][:, :8] - (V - Vp)[ok & (st['a1'][:, t] == a)][:, :8].mean(1, keepdims=True)) ** 2).sum() / ((V[ok & (st['a1'][:, t] == a)][:, :8] - V[ok & (st['a1'][:, t] == a)][:, :8].mean(1, keepdims=True)) ** 2).sum():.2f}"
    for a in range(2, 10)))
print("   error of the closed form in the quantity that decides the answer, V[pS] - V[pM], next to the margin s[pS] - s[pM] (units of M):")
print("   gap/M bin    | median margin | error of closed form: median, IQR, sd | share of cases with |error| > margin")
g = (M - S) / M
for lo, hi in ((0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8)):
    q = ok & (g > lo) & (g <= hi) & (pS <= 7) & (pM <= 7)
    err = ((V[ar, pS] - V[ar, pM]) - (Vp[ar, pS] - Vp[ar, pM]))[q] / M[q]
    mar = (s_all[ar, pS] - s_all[ar, pM])[q] / M[q]
    print(f"   ({lo:.2f},{hi:.2f}]  |  {np.median(mar):6.1f}  |  {np.median(err):+6.1f}, [{np.percentile(err, 25):+6.1f}, {np.percentile(err, 75):+6.1f}], {err.std():5.1f}  |  {np.mean(np.abs(err) > mar):.3f}   n={q.sum()}")
print("   sign of the separation matters (report's own table: A(d>0) vs A(d<0)): V[pS] - V[pM] per unit (M - 0.87 S), median, by signed d = pS - pM")
for d in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6):
    q = ok & (pS - pM == d) & (g > 0.15) & (pS <= 7) & (pM <= 7)
    if q.sum() > 300:
        v = ((V[ar, pS] - V[ar, pM]) / (M - 0.87 * S))[q]
        print(f"      d = {d:+d}: {np.median(v):6.1f}   (closed form {60 * (1 - np.cos(om * d)):6.1f})   n={q.sum()}")

# ------------------------------------------------------------------ (C)
print("\n(C) eigenbasis of the resting map")
lam, Vr = np.linalg.eig(A)
print(f"   condition number of the eigenvector matrix (unit-norm columns): {np.linalg.cond(Vr / np.linalg.norm(Vr, axis=0)):.1f}")
def plane(n):
    v = RIGHT[n]; Q, _ = np.linalg.qr(np.column_stack([v.real, v.imag])); return Q
for a_, b_ in (("clock", "short"), ("clock", "h2"), ("h2", "h3")):
    sv = np.linalg.svd(plane(a_).T @ plane(b_), compute_uv=False)
    print(f"   principal angles between the {a_} plane and the {b_} plane: {np.degrees(np.arccos(np.clip(sv, -1, 1))).round(1)} deg")
for n in ("level", "runner"):
    v = RIGHT[n].real / np.linalg.norm(RIGHT[n].real)
    for m_ in ("level", "runner"):
        if m_ > n:
            u = RIGHT[m_].real / np.linalg.norm(RIGHT[m_].real); print(f"   cos(RIGHT_{n}, RIGHT_{m_}) = {v @ u:.3f}")
print("   norms |LEFT_k| |RIGHT_k| (1 for a normal matrix): " + ", ".join(f"{n} {np.linalg.norm(LEFT[n]) * np.linalg.norm(RIGHT[n]):.1f}" for n in NAMES))

# ------------------------------------------------------------------ (D)
g_ = -w[COMP]
Bc = W[np.ix_(COMP, COMP)] * g_[None, :] / g_[:, None]
print("\n(D) B_c 1 =", np.round(Bc @ np.ones(4), 4), ";  in activation units: |w_COMP| =", round(np.linalg.norm(w[COMP]), 2),
      ", |W[COMP,COMP] w_COMP| =", round(np.linalg.norm(W[np.ix_(COMP, COMP)] @ w[COMP]), 3),
      "; store: |W[COMP,2]| =", round(np.linalg.norm(W[COMP, 2]), 3), ", after one step of the block:", round(np.linalg.norm(W[np.ix_(COMP, COMP)] @ W[COMP, 2]), 3),
      "; detector: |W[COMP,4]| =", round(np.linalg.norm(W[COMP, 4]), 3), ", after one step:", round(np.linalg.norm(W[np.ix_(COMP, COMP)] @ W[COMP, 4]), 3))
print("    W[COMP,2] / w[COMP] =", np.round(W[COMP, 2] / w[COMP], 4))
