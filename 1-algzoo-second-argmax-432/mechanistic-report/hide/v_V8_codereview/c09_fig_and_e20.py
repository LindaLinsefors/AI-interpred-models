"""Code review of fig_mechanism.py and e20_unit_orbits.py (claim O1).

 (1) fig_mechanism.npz is older than fig_mechanism.py (16:37 vs 16:38).  Recompute with the current
     compute() (np.savez disabled, nothing is written) and compare with the cache.
 (2) Panel (a) / e20: the gap is (M - max(S, 0)) / M, the legend says (M - S)/M.  The bin (0.85, 1.0]
     then contains every state with S <= 0 (gap exactly 1).  How much of that bin is S <= 0, and does
     the picture change with the literal (M - S)/M ?
 (3) Panel (b): thresholds; no matching experiment script among e10-e25, so recompute independently and
     show the spread that the medians hide.
 (4) Panels (d), (f): compare with e13 / e12 / e15b on the figure's own sample.
 (5) e20: the docstring says 'the runner-up has not changed for >= 2 steps'; the code's a2 >= 2 does
     not ensure that at a1 = 1.  Effect on the O1 numbers.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy
numpy.savez = lambda *a, **k: None                      # compute() must not overwrite the lead's cache
import fig_mechanism as fm
from common import *
from modes import coords

cache = dict(np.load(HIDE / "fig_mechanism.npz"))
out = fm.compute()
print("(1) cache vs recomputation with the current compute() (same seed 4242):")
for k in cache:
    a, b = cache[k], out[k]
    same = a.shape == b.shape and np.allclose(a, b, equal_nan=True, rtol=1e-10, atol=1e-10)
    print(f"    {k:14s} shape {a.shape}  identical: {same}" + ("" if same else f"   max abs diff {np.nanmax(np.abs(a - b))}"))

x = sample(1_000_000, 4242)
H, Z = run(x, return_pre=True)
st = running_stats(x)
M, S, a1, a2 = st["M1"], st["M2"], st["a1"], st["a2"]
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)
Mpos = np.where(M > 0, M, np.nan)
with np.errstate(invalid="ignore"):
    g_code = (M - Sp) / Mpos
    g_lit = (M - np.where(np.isfinite(S), S, np.nan)) / Mpos
z = coords(H)["clock"] / (10.157 * Mpos)

print("\n(2) panel (a): |z|/(10.157 M) and phase by age of the max; code's gap (M - S+)/M versus the literal (M - S)/M of the legend")
print("    share of each bin's states that have S <= 0 (gap exactly 1 in the code):")
for lo, hi in fm.GAP_BINS:
    row = []
    for a in range(1, 10):
        s = (M > 0.3) & (a1 == a) & (g_code > lo) & (g_code <= hi)
        row.append(f"{np.mean(S[s] <= 0):.2f}")
    print(f"      bin ({lo}, {hi}]: ages 1..9: " + " ".join(row))
print("    third bin (0.85, 1.0]: radius@phase of the median, code | literal gap (S > 0 only) | only S <= 0")
for a in range(1, 10):
    row = []
    for gg, extra in ((g_code, np.ones_like(M, bool)), (g_lit, S > 0), (g_code, S <= 0)):
        s = (M > 0.3) & (a1 == a) & (gg > 0.85) & (gg <= 1.0) & extra
        c = np.median(z[s].real) + 1j * np.median(z[s].imag)
        row.append(f"{abs(c):.3f}@{np.degrees(np.angle(c)) % 360:5.1f} (n={s.sum():6d})")
    print(f"      age {a}: " + " | ".join(row))
print("    all three bins, code's definition: radius of the median / bin centre of the gap (is 'radius = gap'?)")
for i, (lo, hi) in enumerate(fm.GAP_BINS):
    zz = out["clock_by_age"][i, 1:]
    print(f"      bin ({lo}, {hi}]: |z| = " + " ".join(f"{abs(c):.3f}" for c in zz) + f"   bin centre {(lo + hi) / 2:.3f}")

print("\n(3) panel (b): (theta - S)/(M - S) by age of the max at the comparison; median [q10, q90]; states with S > 0, M - S > 0.2")
Mo, So, ao = M[:, :-1], S[:, :-1], a1[:, :-1] + 1
ok = (Mo - So > 0.2) & (So > 0)
for j in COMP + [4]:
    theta = -(H[:, :-1] @ W[j]) / w[j]
    with np.errstate(invalid="ignore"):
        pos = (theta - So) / (Mo - So)
    row = []
    for a in range(1, 10):
        v = pos[ok & (ao == a)]
        row.append(f"{np.median(v):5.2f} [{np.percentile(v, 10):5.2f},{np.percentile(v, 90):5.2f}]")
    print(f"      theta_{j}: " + "  ".join(row))
    k = (COMP + [4]).index(j)
    print(f"               cache medians: " + " ".join(f"{v:5.2f}" for v in cache["thresholds"][k, 1:]))
print("    fraction of those states in which the threshold is BELOW the runner-up (position < 0) / ABOVE the max (position > 1), by age 1..9:")
for j in COMP + [4]:
    theta = -(H[:, :-1] @ W[j]) / w[j]
    with np.errstate(invalid="ignore"):
        pos = (theta - So) / (Mo - So)
    print(f"      theta_{j}: below S: " + " ".join(f"{np.mean(pos[ok & (ao == a)] < 0):.2f}" for a in range(1, 10))
          + "   above M: " + " ".join(f"{np.mean(pos[ok & (ao == a)] > 1):.2f}" for a in range(1, 10)))

print("\n(4) panel (f) on the figure's sample:")
print("    cache acc_s (all units -> runner-up | S_t > 0), t = 1..9:", np.round(cache["acc_s"][1:], 4))
print("    cache acc_m (no comparators -> max | S_t > 0, max not current, index 9 excluded), t = 2..9:", np.round(cache["acc_m"][2:], 4))
L = H @ U.T
Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
Mall = L - H @ Uc.T
un_s, un_m, un_m_all = [], [], []
for tt in range(1, 10):
    valid = np.arange(10) >= 9 - tt
    ps = np.where(valid, L[:, tt], -np.inf).argmax(1) + tt - 9
    un_s.append(np.mean(ps == st["p2"][:, tt]))
    pm = np.where(valid & (np.arange(10) <= 8), Mall[:, tt], -np.inf).argmax(1) + tt - 9
    pm_all = np.where(valid, Mall[:, tt], -np.inf).argmax(1) + tt - 9
    un_m.append(np.mean(pm == st["p1"][:, tt])); un_m_all.append(np.mean(pm_all == st["p1"][:, tt]))
print("    WITHOUT the conditioning (all states): all units -> runner-up, t = 1..9:", np.round(un_s, 4))
print("    WITHOUT the conditioning: no comparators -> max, index 9 excluded, t = 1..9:", np.round(un_m, 4))
print("    WITHOUT the conditioning and with index 9 allowed (the literal 'readout without the comparators'):", np.round(un_m_all, 4))

print("\n(5) e20 / O1 on e20's sample (seed 2020): 'quiet' as coded (a1 >= 1, a2 >= 2) versus as described (also no record at t-1, i.e. a1 >= 2)")
del H, Z, L, Mall
x = sample(1_500_000, 2020)
H = run(x)
st = running_stats(x)
M, S, a1, a2 = st["M1"], st["M2"], st["a1"], st["a2"]
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)
g = (M - Sp) / np.where(M > 0, M, np.nan)
quiet = (M > 0.3) & (a1 >= 1) & (a2 >= 2)
print("    share of 'quiet' states at a1 = 1 (the runner-up changed one step ago because the old max was demoted):", round(float(np.mean(a1[quiet] == 1)), 3))
print("    share of S <= 0 states in the bin (0.85, 1.0], by age 1..9:",
      " ".join(f"{np.mean(S[quiet & (a1 == a) & (g > 0.85) & (g <= 1.0)] <= 0):.2f}" for a in range(1, 10)))
for lo, hi in ((0.15, 0.30), (0.85, 1.0)):
    med = np.array([np.median(H[quiet & (a1 == a) & (g > lo) & (g <= hi)] / M[quiet & (a1 == a) & (g > lo) & (g <= hi)][:, None], axis=0) for a in range(1, 10)])
    print(f"    bin ({lo}, {hi}]: half-range of median h_j/M over ages 1..9 | ages 2..9 only:  " +
          "  ".join(f"unit {j}: {(med[:, j].max() - med[:, j].min()) / 2:.1f} | {(med[1:, j].max() - med[1:, j].min()) / 2:.1f}" for j in (1, 8, 6, 7)))
    print(f"                       largest median h_12/M over ages 1..9: {med[:, 12].max():.1f}")
lo, hi = 0.85, 1.0
for lab, extra in (("S > 0 only (literal gap in (0.85, 1])", S > 0), ("S <= 0 only", S <= 0)):
    rows = []
    for a in range(1, 10):
        s = quiet & (a1 == a) & (g > lo) & (g <= hi) & extra
        rows.append(np.median(H[s] / M[s][:, None], axis=0) if s.sum() > 200 else np.full(16, np.nan))
    med = np.array(rows)
    print(f"    bin (0.85, 1.0], {lab}: half-range over ages 1..9: " +
          "  ".join(f"unit {j}: {(np.nanmax(med[:, j]) - np.nanmin(med[:, j])) / 2:.1f}" for j in (1, 8, 6, 7)) + f";  max median h_12/M = {np.nanmax(med[:, 12]):.1f}")
