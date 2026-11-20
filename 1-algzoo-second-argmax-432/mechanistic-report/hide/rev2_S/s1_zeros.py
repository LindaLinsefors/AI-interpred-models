"""Referee S, check 1: are the ZEROS of units 0, 11, 10 needed at a new maximum, or only their low values?
Selection and forcing as in v_V7_units/v7_record_dyn.py (largest input at t_r; the runner-up of the whole
sequence is the largest of x_0..x_{t_r-1} and > 0.3; old maximum a steps old at t_r; t_r <= 8).
Variants applied to one unit k at step t_r, then the run continues with the real inputs:
  forced   h[k] <- h_{t_r-1}[k]            (V7's test: the unit keeps its previous, large value)
  unclip   h[k] <- pre-activation z[k]     (the ReLU of that unit is removed at that step: negative instead of 0)
  mirror   h[k] <- |z[k]|                  (same distance from zero, on the positive side)
  c*prev   h[k] <- c * h_{t_r-1}[k]        (dose-response between 0 and the forced value)
Also: unit k linear (no ReLU) at every step of the sequence.
"""
import sys
from collections import defaultdict
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *

SEEDS = [9101, 9102, 9103]
NB = 400_000
PAIRS = [(0, 2), (11, 5), (10, 6), (12, 4)]
DOSES = [0.05, 0.1, 0.25, 0.5]


def run_lin(x, k):
    h = np.zeros((len(x), N))
    for t in range(x.shape[1]):
        z = x[:, t, None] * w + h @ W.T
        h = np.maximum(z, 0.0)
        h[:, k] = z[:, k]
    return h


cnt = defaultdict(float)
size = defaultdict(list)
for seed in SEEDS:
    x = sample(NB, seed)
    H, Z = run(x, return_pre=True)
    order = np.argsort(-x, axis=1, kind="stable")
    p1, p2 = order[:, 0], order[:, 1]
    Sx = x[np.arange(NB), p2]
    ok = (p2 < p1) & (Sx > 0.3)
    age = p1 - p2
    lin_ans = {k: (run_lin(x, k) @ U.T).argmax(1) for k, _ in PAIRS}
    base_ans = logits(H).argmax(1)
    for k, a in PAIRS:
        for tr in range(max(a, 2), 9):
            mk = ok & (p1 == tr) & (age == a)
            n = int(mk.sum())
            h_tr, z_tr, h_pv, M0 = H[mk, tr], Z[mk, tr], H[mk, tr - 1], Sx[mk]
            var = {"base": h_tr[:, k], "forced": h_pv[:, k], "unclip": z_tr[:, k], "mirror": np.abs(z_tr[:, k])}
            for c in DOSES:
                var[f"{c}*prev"] = c * h_pv[:, k]
            for name, val in var.items():
                h = h_tr.copy(); h[:, k] = val
                hf = run(x[mk, tr + 1:], h0=h)[:, -1]
                cnt[(k, a, name)] += int(((hf @ U.T).argmax(1) == p2[mk]).sum())
            cnt[(k, a, "linear always")] += int((lin_ans[k][mk] == p2[mk]).sum())
            cnt[(k, a, "n")] += n
            size[(k, a, "z/M")].append(z_tr[:, k] / M0)
            size[(k, a, "prev/M")].append(h_pv[:, k] / M0)
            size[(k, a, "off")].append(h_tr[:, k] == 0)

print(f"seeds {SEEDS}, {NB} each; records at t_r = max(a,2)..8 pooled; P(answer = old maximum)")
names = ["base", "forced", "unclip", "mirror"] + [f"{c}*prev" for c in DOSES] + ["linear always"]
print("unit age       n   " + "".join(f"{s:>15s}" for s in names))
for k, a in PAIRS:
    n = cnt[(k, a, "n")]
    print(f"{k:4d} {a:3d} {int(n):7d}   " + "".join(f"{cnt[(k, a, s)] / n:15.3f}" for s in names))
print("\nsizes at the record step (medians, in units of the old maximum M0):")
for k, a in PAIRS:
    z = np.concatenate(size[(k, a, "z/M")]); pv = np.concatenate(size[(k, a, "prev/M")]); off = np.concatenate(size[(k, a, "off")])
    print(f"  unit {k:2d}, age {a}: exactly zero in {off.mean():.3f}; pre-activation median {np.median(z):+.2f} "
          f"(5%..95%: {np.quantile(z, .05):+.2f}..{np.quantile(z, .95):+.2f}); value one step earlier median {np.median(pv):.2f}")
