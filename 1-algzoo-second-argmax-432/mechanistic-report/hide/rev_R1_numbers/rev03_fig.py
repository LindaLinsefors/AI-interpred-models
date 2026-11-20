"""Reviewer R1: print what the figure's cache holds, to compare with the caption and with numbers in the text.
Run:  python -B rev03_fig.py > rev03_fig_out.txt   (reads ../fig_mechanism.npz, writes nothing)"""
import sys
sys.dont_write_bytecode = True
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
d = np.load(r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/fig_mechanism.npz")
print("keys:", list(d.keys()))
za = d["clock_by_age"]
for i, lab in enumerate(["gap 0.15-0.30", "gap 0.45-0.60", "gap 0.85-1.00"]):
    print(f"(a) {lab}: |z|/(10.157 M) by age 1..9:", np.round(np.abs(za[i, 1:]), 3), " phase:", np.round(np.degrees(np.angle(za[i, 1:])) % 360, 1))
print("    ratio of radii, small-gap bin / large-gap bin, by age:", np.round(np.abs(za[0, 1:]) / np.abs(za[2, 1:]), 3), " (gap ratio 0.225/0.925 = 0.243; (M-0.87S) ratio = 0.35)")
th = d["thresholds"]
for k, j in enumerate([1, 6, 7, 8, 4]):
    print(f"(b) unit {j}: median position by age-at-comparison 1..9:", np.round(th[k, 1:], 2))
vd, vA = d["vote_d"], d["vote_A"]
print("(c) vote per unit gap by d (mean, min, max, n) vs -60cos:")
for dd in range(-7, 8):
    v = vA[vd == dd]
    print(f"    d={dd:+d}: {v.mean():6.1f} [{v.min():6.1f}, {v.max():6.1f}] n={len(v)}   -60cos = {-60*np.cos(np.radians(42.748*dd)):6.1f}")
V_EDGES = np.array([-1.0, -0.8, -0.6, -0.45, -0.35, -0.25, -0.18, -0.12, -0.07, -0.03, 0.0, 0.03, 0.07, 0.12, 0.18, 0.25, 0.35, 0.45, 0.6, 0.8, 1.0])
vc = 0.5 * (V_EDGES[:-1] + V_EDGES[1:])
print("(d) bin centres:", np.round(vc, 3))
for c, lab in enumerate(["arriving now", "one step old", "two or more"]):
    print(f"    {lab:13s}:", np.round(d["score_med"][c], 1))
print("    slope (score / v) for 'arriving now':", np.round(d["score_med"][0] / vc, 1))
print("(e) Sigma max |off-superdiagonal| rows 0-6:", round(max(abs(d['Sigma'][i, j]) for i in range(7) for j in range(10) if j != i + 1), 3))
print("(f) acc_s (given S_t > 0) t=1..9:", np.round(d["acc_s"][1:], 4))
print("    acc_m (given S_t > 0, max not current, older inputs only) t=2..9:", np.round(d["acc_m"][2:], 4))
