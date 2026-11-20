"""Reviewer N1: print the cached data behind mechanism_summary.png (read-only) to check the caption and the
table of section 3.3.  Run: python -B n1_figcache.py"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=220)
d = np.load(Path(__file__).resolve().parent.parent / "fig_mechanism.npz")
print("keys:", list(d.keys()))
V_EDGES = np.array([-1.0, -0.8, -0.6, -0.45, -0.35, -0.25, -0.18, -0.12, -0.07, -0.03, 0.0,
                    0.03, 0.07, 0.12, 0.18, 0.25, 0.35, 0.45, 0.6, 0.8, 1.0])
vc = 0.5 * (V_EDGES[:-1] + V_EDGES[1:])
print("\n(d) bin centres:", vc)
for c, lab in enumerate(["age 0", "age 1", "age>=2"]):
    print(lab, np.round(d["score_med"][c], 2))
    print("   ratio med/centre:", np.round(d["score_med"][c] / vc, 1))
print("\n(a) |z|/(10.157 M) and phase by age 1..9, three gap bins")
for i in range(3):
    z = d["clock_by_age"][i, 1:]
    print("  radius", np.round(np.abs(z), 3), " phase", np.round(np.degrees(np.angle(z)) % 360, 1))
print("\n(b) thresholds (rows: comparators 1, 6, 7, 8, unit 4), ages 1..9")
print(np.round(d["thresholds"][:, 1:], 2))
print("\n(f) acc_s t=1..9:", np.round(d["acc_s"][1:], 4))
print("    acc_m t=2..9:", np.round(d["acc_m"][2:], 4))
print("\n(e) Sigma rows 0-8 superdiagonal:", np.round([d["Sigma"][i, i + 1] for i in range(9)], 3))
dd, A, Am = d["vote_d"], d["vote_A"], d["vote_Am"]
print("\n(c) mean over ages by d=-7..7:  V:", np.round([A[dd == k].mean() for k in range(-7, 8)], 1))
print("                              m:", np.round([Am[dd == k].mean() for k in range(-7, 8)], 1))
print("                            sum:", np.round([(A + Am)[dd == k].mean() for k in range(-7, 8)], 1))
print("    number of dots:", len(dd), " max |V+m| per dot:", np.abs(A + Am).max())
