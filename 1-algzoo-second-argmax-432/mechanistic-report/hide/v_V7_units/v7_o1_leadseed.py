"""V7: after my own numbers were done, rerun O1 on the lead's own sample (1.5M sequences, seed 2020, generated in
one call so the random stream matches; the network is run in chunks) to see whether the small differences in
the third bin (unit 11: 0.86 vs my 0.851; unit 10: 0.98 vs my 0.973) are sampling noise of that seed."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *

x = sample(1_500_000, 2020)
BINS = [(0.15, 0.30), (0.45, 0.60), (0.85, 1.0)]
parts = []
for i in range(0, len(x), 300_000):
    xc = x[i:i + 300_000]
    H = run(xc)
    rs = running_stats(xc)
    M, S, a1, a2 = rs["M1"], rs["M2"], rs["a1"], rs["a2"]
    b, t = np.nonzero((M > 0.3) & (a1 >= 1) & (a2 >= 2))
    parts.append(dict(r=(H[b, t] / M[b, t, None]).astype(np.float32), zero=H[b, t] == 0,
                      g=(M[b, t] - np.maximum(S[b, t], 0)) / M[b, t], a1=a1[b, t]))
D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
print(f"lead's sample: seed 2020, 1.5M sequences; selected states {D['g'].size}")
for u, a in [(0, 2), (11, 5), (10, 6), (12, 4)]:
    out = []
    for lo, hi in BINS:
        mk = (D["g"] > lo) & (D["g"] <= hi) & (D["a1"] == a)
        f = D["zero"][mk, u].mean()
        out.append(f"{f:.4f} (n={mk.sum()}, se={np.sqrt(f * (1 - f) / mk.sum()):.4f})")
    print(f"  unit {u:2d} at a1={a}: " + "   ".join(out))
med = np.array([[np.median(D["r"][(D["g"] > lo) & (D["g"] <= hi) & (D["a1"] == a)], axis=0) for a in range(1, 10)] for lo, hi in BINS])
print("  largest median of h_12/M: " + "  ".join(f"bin{k + 1}: {med[k, :, 12].max():.2f}" for k in range(3)))
for u in [1, 8, 6, 7]:
    print(f"  half-range unit {u}: " + "  ".join(f"bin{k + 1}: {np.ptp(med[k, :, u]) / 2:.2f}" for k in range(3)))
