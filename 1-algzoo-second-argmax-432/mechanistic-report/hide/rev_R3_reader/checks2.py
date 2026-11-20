"""Reviewer R3 (cold reader): second batch of checks. Writes checks2_out.txt beside this script.
Run with:  python -B checks2.py
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from pathlib import Path
import numpy as np
from scipy.linalg import null_space, subspace_angles
from common import *
import modes

OUT = Path(__file__).resolve().parent / "checks2_out.txt"
lines = []
def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    lines.append(s)

np.set_printoptions(precision=3, suppress=True, linewidth=220)
K = w[4]
A = W[np.ix_(ORD, ORD)]
Uo = U[:, ORD]
ci = [ORD.index(j) for j in COMP]                  # comparator rows inside ORD
wi = [i for i in range(13) if i not in ci]         # wave-unit rows inside ORD

# ---------------------------------------------------------------- N1 null space of the readout
P("=== N1. Null space of Uo (10 x 13, rank 10) against the level and clock eigenvectors ===")
Nn = null_space(Uo)
P("dim ker Uo =", Nn.shape[1])
Rl = np.real(modes.RIGHT["level"]); Rr = np.real(modes.RIGHT["runner"])
Rc = modes.RIGHT["clock"]
B_lc = np.stack([Rl, Rc.real, Rc.imag], 1)
P("principal angles (deg) between ker Uo and span{level, Re clock, Im clock}:", np.degrees(subspace_angles(Nn, B_lc)))
B_rc = np.stack([Rr, Rc.real, Rc.imag], 1)
P("principal angles (deg) between ker Uo and span{runner, Re clock, Im clock}:", np.degrees(subspace_angles(Nn, B_rc)))
for name in modes.NAMES:
    r = modes.RIGHT[name]
    for part, v in (("re", r.real), ("im", r.imag)):
        if np.linalg.norm(v) < 1e-9:
            continue
        frac = np.linalg.norm(Nn.T @ v) / np.linalg.norm(v)
        P(f"  {name:7s} {part}: fraction of the eigenvector's norm inside ker Uo = {frac:.3f}")
# same with the mean over indices 0-8 removed and index 9 dropped (what decides the argmax among older inputs)
C9 = np.eye(9) - np.ones((9, 9)) / 9
Uc = C9 @ Uo[:9]
N2 = null_space(Uc, rcond=1e-10)
P("readout of indices 0-8 with their mean removed: rank", np.linalg.matrix_rank(Uc), " singular values", np.linalg.svd(Uc, compute_uv=False))
P("  principal angles (deg) between its null space (dim %d) and span{level, runner, Re clock, Im clock}:" % N2.shape[1],
  np.degrees(subspace_angles(N2, np.stack([Rl, Rr, Rc.real, Rc.imag], 1))))

# ---------------------------------------------------------------- N2 pure clock state with the comparators zeroed
P("\n=== N2. Pure clock state, comparators set to zero: how much of the clock coordinate is left? ===")
L = modes.LEFT["clock"]
s_same = (L[wi] * Rc[wi]).sum()          # coefficient of z
s_conj = (L[wi] * np.conj(Rc[wi])).sum() # coefficient of conj(z)
P(f"z' = a z + b conj(z):  a = {abs(s_same):.3f} @ {np.degrees(np.angle(s_same)):.1f} deg,  b = {abs(s_conj):.3f} @ {np.degrees(np.angle(s_conj)):.1f} deg")
P("share of |LEFT_clock|^2 on the comparators:", (abs(L[ci]) ** 2).sum() / (abs(L) ** 2).sum())
P("share of |RIGHT_clock|^2 on the comparators:", (abs(Rc[ci]) ** 2).sum() / (abs(Rc) ** 2).sum())
P("readout of the two halves of the clock right vector, indices 0-7: |U_comp R_comp|, |U_wave R_wave|, |sum|:")
a_ = Uo[:8][:, ci] @ Rc[ci]; b_ = Uo[:8][:, wi] @ Rc[wi]
P("  ", np.round(2 * abs(a_), 2), np.round(2 * abs(b_), 2), np.round(2 * abs(a_ + b_), 2))
P("   sum/(|a|+|b|) overall:", abs(a_ + b_).sum() / (abs(a_) + abs(b_)).sum())

# ---------------------------------------------------------------- N3 sum of scores and row 9 of Sigma
P("\n=== N3. What 'row 9 = 1.4 x the sum of all scores' means on data ===")
x = sample(400_000, seed=777)
H = run(x); st = running_stats(x)
for t in (3, 5, 7):
    h = H[:, t][:, ORD]
    s = h @ Uo.T
    nxt9 = (h @ A.T) @ Uo[9]                     # score at index 9 after a free step
    tot = s.sum(1)
    sel = st["M1"][:, t] > 0
    cc = np.corrcoef(tot[sel], st["M1"][sel, t])[0, 1]
    P(f"t={t}: corr(sum of scores, M) = {cc:.3f}; slope of sum on M = {np.polyfit(st['M1'][sel, t], tot[sel], 1)[0]:.2f};"
      f" rms(sum)/rms(score entries) = {tot.std() / s.std():.3f};"
      f" corr(next index-9 score, 1.43*sum) = {np.corrcoef(nxt9, 1.43 * tot)[0, 1]:.3f};"
      f" slope of next index-9 score on M = {np.polyfit(st['M1'][sel, t], nxt9[sel], 1)[0]:.2f}")

# ---------------------------------------------------------------- N4 why m ranks the max first less often at t = 9
P("\n=== N4. m_t ranks the maximum first: t = 9 against t = 8, by index of the maximum ===")
for t in (8, 9):
    s = H[:, t] @ U.T
    V = H[:, t][:, COMP] @ U[:, COMP].T
    m = s - V
    older = (np.arange(10) >= 9 - t) & (np.arange(10) <= 8)
    pos = np.where(older, m, -np.inf).argmax(1) + t - 9
    cond = (st["M2"][:, t] > 0) & (st["p1"][:, t] != t)
    ok = pos == st["p1"][:, t]
    P(f"t={t}: overall {ok[cond].mean():.3f}")
    idx_max = st["p1"][:, t] + 9 - t
    P("   by index of the maximum (0..8): " + " ".join(f"{i}:{ok[cond & (idx_max == i)].mean():.3f}" for i in range(9 - t, 9)))
    wrong = cond & ~ok
    P("   when wrong, index chosen: " + " ".join(f"{i}:{((pos + 9 - t)[wrong] == i).mean():.2f}" for i in range(9 - t, 9)))
    if t == 9:
        older1 = (np.arange(10) >= 1) & (np.arange(10) <= 8)
        pos1 = np.where(older1, m, -np.inf).argmax(1)
        c1 = cond & (st["p1"][:, t] >= 1)
        P(f"   t=9 with index 0 excluded from the candidates and from the cases: {(pos1 == st['p1'][:, t])[c1].mean():.3f}")

# ---------------------------------------------------------------- N5 panel a: is the radius the gap?
P("\n=== N5. Panel a: median |z|/(K M) by age of the maximum and relative gap, against the gap itself ===")
z = modes.coords(H)["clock"]
M_ = st["M1"]; S_ = np.maximum(st["M2"], 0); a1 = st["a1"]
rel = (M_ - S_) / M_
ok = (M_ > 0) & np.isfinite(st["M2"])
for lo, hi in ((0.0, 0.05), (0.15, 0.30), (0.45, 0.60), (0.85, 1.00)):
    b = ok & (rel >= lo) & (rel < hi)
    mg = np.median(rel[b])
    row = [np.median(np.abs(z[b & (a1 == a)]) / (K * M_[b & (a1 == a)])) for a in range(1, 10)]
    P(f"gap/M in [{lo},{hi}) (median {mg:.2f}): radius/M at ages 1..9: " + " ".join(f"{v:.2f}" for v in row)
      + f"   radius/gap at ages 2, 5, 9: {row[1]/mg:.2f} {row[4]/mg:.2f} {row[8]/mg:.2f}")

OUT.write_text("\n".join(lines), encoding="utf-8")
