"""Verifier V1, attack E6: is "Sigma is a shift" / "the early readout works" specific to this network, or is it
what ANY network trained on the task shows (i.e. forced by the function, not by this mechanism)?

Population: the 672 bias-free networks of the user's replica sweep (read-only):
  1-algzoo-second-argmax-432/replica/hide/runs_modal/r1/none_bs*_s*.pt (10 files x 64 seeds) and final/none_bs64_s200.pt (32).
Same architecture and task as M_16,10, trained from the notebook recipe on Unif(-1,1).
For every network:
  ORD' = units that are on in >= 50 % of the states at t >= 3 (N(0,1) inputs, seed 6201, 20k sequences).
         (For the original this rule returns exactly the 13 non-pulse units.)
  shift = mean over i = 0..6 of cos(u_i A', u_{i+1}),  u_i = U[i, ORD'], A' = W[ORD', ORD']
  Sigma' = U' A' pinv(U'): mean and min of Sigma'[i,i+1], i = 0..6 (only if |ORD'| >= 10)
  early readout (T3) at t = 2, 4, 6 given S_t > 0, and final accuracy on the same sample.
Run: python -B v1_attack_population.py -> v1_attack_population.txt
"""
from vlib import *
import torch

P = Tee("v1_attack_population.txt")
RUNS = HERE.parents[2] / "replica" / "hide" / "runs_modal"
files = sorted((RUNS / "r1").glob("none_bs*_s*.pt")) + [RUNS / "final" / "none_bs64_s200.pt"]
x = sample(20_000, 6201)
tgt = second_argmax(x)
tops = {t: top3(x, t) for t in (2, 4, 6)}


def analyse(w_, W_, U_):
    H, Z = run(x, w=w_, W=W_, return_pre=True)
    acc = np.mean((H[:, 9] @ U_.T).argmax(1) == tgt)
    pon = (Z[:, 3:] > 0).mean((0, 1))
    ordp = np.where(pon >= 0.5)[0]
    Up, Ap = U_[:, ordp], W_[np.ix_(ordp, ordp)]
    UA = Up @ Ap
    cs = np.mean([UA[i] @ Up[i + 1] / (np.linalg.norm(UA[i]) * np.linalg.norm(Up[i + 1]) + 1e-12) for i in range(7)])
    UA16 = U_ @ W_
    cs16 = np.mean([UA16[i] @ U_[i + 1] / (np.linalg.norm(UA16[i]) * np.linalg.norm(U_[i + 1]) + 1e-12) for i in range(7)])
    if len(ordp) >= 10:
        Sg = UA @ np.linalg.pinv(Up)
        sup = np.array([Sg[i, i + 1] for i in range(7)])
        smean, smin = sup.mean(), sup.min()
    else:
        smean = smin = np.nan
    early = []
    for t in (2, 4, 6):
        p, v = tops[t]
        pos = v[:, 1] > 0
        early.append(np.mean((H[:, t] @ U_.T)[:, 9 - t:].argmax(1)[pos] == p[pos, 1]))
    return [acc, len(ordp), cs, cs16, smean, smin, *early]


ref = analyse(w, W, U)
P("ORIGINAL M_16,10: acc %.4f  |ORD'| %d  mean cos(u_i A', u_i+1) %.4f  all-16-unit version %.4f  Sigma' superdiag mean %.3f min %.3f  early readout t=2,4,6: %.4f %.4f %.4f" % tuple(ref))
pon = (run(x, return_pre=True)[1][:, 3:] > 0).mean((0, 1))
P("   units selected as ORD' for the original:", list(np.where(pon >= 0.5)[0]), " (ORD =", ORD, ")")

rows, tags = [], []
for f in files:
    o = torch.load(f, weights_only=False, map_location="cpu")
    pr = {k: v.double().numpy() for k, v in o["params"].items()}
    for k in range(pr["W_hh"].shape[0]):
        rows.append(analyse(pr["W_ih"][k][:, 0], pr["W_hh"][k], pr["W_out"][k]))
        tags.append((f.name, k))
R = np.array(rows)
P(f"\npopulation: {len(R)} bias-free networks from {len(files)} files")
names = ["acc (N(0,1), 20k)", "|ORD'|", "mean cos(u_i A',u_i+1)", "all-16 cos", "Sigma' superdiag mean", "Sigma' superdiag min", "early t=2", "early t=4", "early t=6"]
bands = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.94), (0.94, 1.0)]
P("by final accuracy band: n | median [10th, 90th pct] of each quantity")
for lo, hi in bands:
    m = (R[:, 0] >= lo) & (R[:, 0] < hi)
    if m.sum() == 0:
        continue
    P(f"--- acc in [{lo}, {hi}): n = {m.sum()}")
    for j, nm in enumerate(names):
        col = R[m, j]; col = col[~np.isnan(col)]
        if len(col):
            P(f"     {nm:24s} median {np.median(col):7.3f}  [{np.percentile(col,10):7.3f}, {np.percentile(col,90):7.3f}]  n={len(col)}")
good = R[:, 0] >= 0.9
P(f"\nnetworks with acc >= 0.90: {good.sum()};  of these, mean cos >= 0.9: {(R[good, 2] >= 0.9).sum()},  >= 0.8: {(R[good, 2] >= 0.8).sum()},  >= 0.5: {(R[good, 2] >= 0.5).sum()},  >= original ({ref[2]:.3f}): {(R[good, 2] >= ref[2]).sum()}")
P(f"   all-16-unit cos >= 0.8: {(R[good, 3] >= 0.8).sum()};  original's all-16 value {ref[3]:.3f}")
P(f"   early readout t=4 within 0.03 of (or above) the final accuracy: {(R[good, 7] >= R[good, 0] - 0.03).sum()} of {good.sum()};  t=2: {(R[good, 6] >= R[good, 0] - 0.03).sum()};  t=6: {(R[good, 8] >= R[good, 0] - 0.03).sum()}")
P("   Spearman over all networks: acc vs mean cos: %.3f ; acc vs early t=4: %.3f" % (
    np.corrcoef(np.argsort(np.argsort(R[:, 0])), np.argsort(np.argsort(R[:, 2])))[0, 1],
    np.corrcoef(np.argsort(np.argsort(R[:, 0])), np.argsort(np.argsort(R[:, 7])))[0, 1]))
top = np.argsort(-R[:, 0])[:15]
P("\nthe 15 most accurate networks: file, index, acc, |ORD'|, mean cos, all-16 cos, Sigma' superdiag mean/min, early t=2/4/6")
for i in top:
    P(f"   {tags[i][0]:22s} {tags[i][1]:3d}  {R[i,0]:.4f}  {int(R[i,1]):2d}  {R[i,2]:6.3f}  {R[i,3]:6.3f}  {R[i,4]:6.3f} {R[i,5]:6.3f}   {R[i,6]:.3f} {R[i,7]:.3f} {R[i,8]:.3f}")
