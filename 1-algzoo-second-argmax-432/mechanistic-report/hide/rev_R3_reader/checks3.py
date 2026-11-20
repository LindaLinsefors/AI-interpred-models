"""Reviewer R3: third batch (row 9 of Sigma on data; how common small gaps are; real quiet steps). Writes checks3_out.txt."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from pathlib import Path
import numpy as np
from common import *
import modes
OUT = Path(__file__).resolve().parent / "checks3_out.txt"
lines = []
def P(*a):
    s = " ".join(str(v) for v in a); print(s); lines.append(s)
K = w[4]
A = W[np.ix_(ORD, ORD)]; Uo = U[:, ORD]
Sig = Uo @ A @ np.linalg.pinv(Uo)
x = sample(400_000, seed=777); H = run(x); st = running_stats(x)

P("=== R1. Row-wise fit of Sigma on real states (free step = A applied to the real state, no input, no gates) ===")
for t in (3, 5, 7):
    h = H[:, t][:, ORD]; s = h @ Uo.T; nxt = (h @ A.T) @ Uo.T; pred = s @ Sig.T
    r2 = 1 - ((nxt - pred) ** 2).mean(0) / nxt.var(0)
    P(f"t={t}: R2 per row 0..9 (raw, no mean removal): " + " ".join(f"{v:.3f}" for v in r2))

P("\n=== R2. Real next step against Sigma s_t, indices 0-8 with mean removed, by kind of step ===")
M1 = st["M1"]; M2 = st["M2"]
for t in (3, 5, 7):
    s0 = H[:, t] @ U.T; s1 = H[:, t + 1] @ U.T          # real scores, all 16 units
    pred = s0 @ Sig.T                                   # Sigma applied to the full real scores
    shift = np.zeros_like(s0); shift[:, :9] = s0[:, 1:] # pure shift by one index
    xn = x[:, t + 1]; S = np.maximum(M2[:, t], 0)
    kinds = {"all steps": np.ones(len(x), bool),
             "x_{t+1} < 0.5 S (S>0)": (M2[:, t] > 0) & (xn < 0.5 * S),
             "0.75 S < x_{t+1} < S": (M2[:, t] > 0) & (xn > 0.75 * S) & (xn < S),
             "new runner-up": (xn > M2[:, t]) & (xn < M1[:, t]),
             "new maximum": xn > M1[:, t]}
    for name, sel in kinds.items():
        def r2(p, lo=0, hi=8):                          # indices lo..hi-1 = inputs older than x_t after the step
            a = s1[sel][:, lo:hi]; b = p[sel][:, lo:hi]
            a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
            return 1 - ((a - b) ** 2).sum() / (a ** 2).sum()
        P(f"t={t} -> {t+1}, {name:24s} n={sel.sum():7d}: R2 Sigma (idx 0-7) = {r2(pred):6.3f}   pure shift (idx 0-7) = {r2(shift):6.3f}")

P("\n=== R3. How common are small relative gaps (states with max age >= 1, S > 0) ===")
a1 = st["a1"]; ok = (a1 >= 1) & (M2 > 0)
rel = ((M1 - M2) / M1)[ok]
for q in (0.05, 0.1, 0.2, 0.3, 0.5):
    P(f"P(gap/M < {q}) = {(rel < q).mean():.3f}")
z = np.abs(modes.coords(H)["clock"])[ok] / K
gap = (M1 - M2)[ok]
P("median |z|/(K gap):", np.median(z / gap), "  quartiles:", np.percentile(z / gap, [25, 75]))
P("least squares |z|/K = a (M - S) + b S over those states: ", np.linalg.lstsq(np.stack([gap, M2[ok]], 1), z, rcond=None)[0])
OUT.write_text("\n".join(lines), encoding="utf-8")
