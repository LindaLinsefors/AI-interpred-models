"""S1: among inputs older than the current one (indices 9-t..8), given S_t > 0 and p1 < t,
P(argmax m_t = position of the max of x_0..x_t), t = 2..9.  Plus breakdowns (age of the max, gap, what
is chosen instead) and the same for the full score s_t.

Usage: python -B s1.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 10
LEAD = [0.972, 0.942, 0.948, 0.953, 0.958, 0.964, 0.966, 0.891]
GAPS = [0, 0.05, 0.15, 0.3, 0.5, 0.8, 1.0]

ts = range(2, 10)
n = {t: 0 for t in ts}; hit_m = dict(n); hit_s = dict(n); m_is_p2 = dict(n); s_is_p2 = dict(n)
n_age = np.zeros((10, 10)); h_age = np.zeros((10, 10))                 # [t, a1]
n_gap = np.zeros((10, len(GAPS) - 1)); h_gap = np.zeros_like(n_gap)    # [t, gap bin]
n_ag = np.zeros((10, len(GAPS) - 1)); h_ag = np.zeros_like(n_ag)       # [a1, gap bin], pooled over t
# variants of the conditioning
n_neg = dict(n); h_neg = dict(n)          # S_t <= 0, p1 < t
n_all9 = dict(n); h_all9 = dict(n)        # S_t > 0, any p1, argmax over indices 9-t..9 (current input included)
n_cur = dict(n); h_cur = dict(n)          # S_t > 0, p1 = t, argmax over 9-t..9
n_on = dict(n); h_on = dict(n)            # S_t > 0, p1 < t, all four comparators on at t
n_off = dict(n); h_off = dict(n)          # S_t > 0, p1 < t, not all four on

for x, H, M, S, p1, p2 in batches(nb, seed0):
    for t in ts:
        s, V, m = scores(H[:, t])
        am = m[:, 9 - t:9].argmax(1)                  # position among 0..t-1
        as_ = s[:, 9 - t:9].argmax(1)
        P1, P2, Mt, St = p1[:, t], p2[:, t], M[:, t], S[:, t]
        c = (St > 0) & (P1 < t)
        n[t] += c.sum(); hit_m[t] += (am == P1)[c].sum(); hit_s[t] += (as_ == P1)[c].sum()
        m_is_p2[t] += (am == P2)[c].sum(); s_is_p2[t] += (as_ == P2)[c].sum()
        a1 = t - P1
        np.add.at(n_age[t], a1[c], 1); np.add.at(h_age[t], a1[c], (am == P1)[c])
        g = np.digitize(((Mt - St) / Mt)[c], GAPS, right=True) - 1
        np.add.at(n_gap[t], g, 1); np.add.at(h_gap[t], g, (am == P1)[c])
        np.add.at(n_ag, (a1[c], g), 1); np.add.at(h_ag, (a1[c], g), (am == P1)[c])
        c2 = (St <= 0) & (P1 < t)
        n_neg[t] += c2.sum(); h_neg[t] += (am == P1)[c2].sum()
        am9 = m[:, 9 - t:].argmax(1)
        c3 = St > 0
        n_all9[t] += c3.sum(); h_all9[t] += (am9 == P1)[c3].sum()
        c4 = (St > 0) & (P1 == t)
        n_cur[t] += c4.sum(); h_cur[t] += (am9 == P1)[c4].sum()
        on = (H[:, t][:, COMP] > 0).all(1)
        n_on[t] += (c & on).sum(); h_on[t] += (am == P1)[c & on].sum()
        n_off[t] += (c & ~on).sum(); h_off[t] += (am == P1)[c & ~on].sum()

print(f"S1  seeds {seed0}..{seed0 + nb - 1}, {nb * B_DEFAULT} sequences")
print("t      n      P(argmax m = p1)  lead   se     | P(argmax m = p2) | P(argmax s = p1)  P(argmax s = p2)")
for i, t in enumerate(ts):
    p = hit_m[t] / n[t]
    print(f"{t}  {n[t]:8d}   {p:.4f}          {LEAD[i]:.3f}  {np.sqrt(p * (1 - p) / n[t]):.4f} |  {m_is_p2[t] / n[t]:.4f}"
          f"          |  {hit_s[t] / n[t]:.4f}           {s_is_p2[t] / n[t]:.4f}")

print("\nP(argmax m = p1) by age of the max a1 (columns 1..9), rows t = 2..9   [n in thousands below]")
for t in ts:
    print(t, " ".join(f"{h_age[t, a] / n_age[t, a]:.3f}" if n_age[t, a] else "  -  " for a in range(1, 10)))
print("pooled over t by a1:", " ".join(f"{h_age[2:, a].sum() / n_age[2:, a].sum():.3f}" for a in range(1, 10)))

print("\nP(argmax m = p1) by gap/M = (M - S)/M bins", GAPS, " rows t = 2..9")
for t in ts:
    print(t, " ".join(f"{h_gap[t, k] / n_gap[t, k]:.3f}" for k in range(len(GAPS) - 1)),
          "  n:", " ".join(f"{int(n_gap[t, k]):7d}" for k in range(len(GAPS) - 1)))
print("\nP(argmax m = p1) by a1 (rows 1..9) and gap bin (columns), pooled over t = 2..9")
for a in range(1, 10):
    print(a, " ".join(f"{h_ag[a, k] / n_ag[a, k]:.3f}" for k in range(len(GAPS) - 1)),
          "  n:", " ".join(f"{int(n_ag[a, k]):7d}" for k in range(len(GAPS) - 1)))

print("\nvariants of the conditioning (P that argmax m is the position of the max)")
print("t   S<=0,p1<t        S>0, all p1, idx 9-t..9   S>0, p1=t, idx 9-t..9   S>0,p1<t, 4 comps on   S>0,p1<t, not all on")
for t in ts:
    print(f"{t}   {h_neg[t] / n_neg[t]:.4f} (n={n_neg[t]})   {h_all9[t] / n_all9[t]:.4f}"
          f"                 {h_cur[t] / n_cur[t]:.4f} (n={n_cur[t]})     {h_on[t] / n_on[t]:.4f} (n={n_on[t]})"
          f"    {h_off[t] / n_off[t]:.4f} (n={n_off[t]})")
