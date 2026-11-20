"""Verifier V1: extra checks on T2 conventions and on how many independent numbers the tape holds.
(1) T2 under other conventions: all 10 positions; per-position R^2 (Sigma model and pure shift); seed 6001.
(2) Effective dimensionality of the tape U h_t / M and of h_t / M (PCA across states, S_t > 0.2); seed 6002.
(3) How much of the actual tape U h_t is the ordinary-unit part U_o h_t[ORD] (what T1/T2 speak about)?
Run: python -B v1_extra_checks.py -> v1_extra_checks.txt
"""
from vlib import *

P = Tee("v1_extra_checks.txt")
Uo = U[:, ORD]; A = W[np.ix_(ORD, ORD)]
Sig = Uo @ A @ np.linalg.pinv(Uo)

P("=== (1) T2 conventions (seed 6001, 200k) ===")
x = sample(200_000, 6001)
H = run(x)
for t in (3, 5, 7):
    ho = H[:, t][:, ORD]
    l0 = ho @ Uo.T; l1 = ho @ (Uo @ A).T; pr = l0 @ Sig.T

    def r2(pos):
        y = l1[:, pos] - l1[:, pos].mean(1, keepdims=True); p_ = pr[:, pos] - pr[:, pos].mean(1, keepdims=True)
        return 1 - ((y - p_) ** 2).sum() / (y ** 2).sum()
    y9 = l1[:, :9] - l1[:, :9].mean(1, keepdims=True); p9 = pr[:, :9] - pr[:, :9].mean(1, keepdims=True)
    per_pos = 1 - ((y9 - p9) ** 2).sum(0) / (y9 ** 2).sum(0)
    # pure shift per position: l1[i] vs l0[i+1], centred over positions 0-8 / 1-9 respectively
    s9 = l0[:, 1:10] - l0[:, 1:10].mean(1, keepdims=True)
    per_pos_shift = 1 - ((y9 - s9) ** 2).sum(0) / (y9 ** 2).sum(0)
    raw9 = 1 - ((l1[:, 9] - pr[:, 9]) ** 2).sum() / ((l1[:, 9] - l1[:, 9].mean()) ** 2).sum()
    P(f" t={t}: claim convention (pos 0-8) {r2(list(range(9))):.4f} | all 10 positions, mean over 10 removed {r2(list(range(10))):.4f} | index 9 alone (centred across states) {raw9:.4f}")
    P("       per-position R2, Sigma model (pos 0-8 centring): " + " ".join(f"{v:.3f}" for v in per_pos))
    P("       per-position R2, pure shift l1[i] ~ l0[i+1]:     " + " ".join(f"{v:.3f}" for v in per_pos_shift))
    P("       share of the centred variance of l1 at each position: " + " ".join(f"{v:.3f}" for v in (y9 ** 2).sum(0) / (y9 ** 2).sum()))

P("\n=== (2) effective dimensionality (seed 6002, 200k; states with S_t > 0.2; everything divided by M_t) ===")
x = sample(200_000, 6002)
H = run(x)
for t in (5, 9):
    p, val = top3(x, t)
    k = val[:, 1] > 0.2
    Mm = val[k, 0][:, None]
    tape = (H[k, t] @ U.T)[:, 9 - t:] / Mm
    tape_c = tape - tape.mean(1, keepdims=True)                 # per-state mean over valid positions removed (argmax-irrelevant)
    hh = H[k, t] / Mm
    for name, X in (("tape U h_t (valid positions)", tape), ("tape, per-state mean removed", tape_c), ("hidden state h_t (16 units)", hh), ("ordinary part h_t[ORD]", hh[:, ORD])):
        Xc = X - X.mean(0)
        ev = np.linalg.svd(Xc, compute_uv=False) ** 2
        cum = np.cumsum(ev) / ev.sum()
        pr_ = ev.sum() ** 2 / (ev ** 2).sum()
        P(f" t={t} {name:30s}: cumulative variance of PCs {np.round(cum[:10], 3)}  participation ratio {pr_:.2f}  #PCs for 95%: {int(np.searchsorted(cum, 0.95) + 1)}, 99%: {int(np.searchsorted(cum, 0.99) + 1)}")

P("\n=== (3) the ordinary-unit part of the actual tape (seed 6002) ===")
P(" t | centred R2 of U_o h_t[ORD] for the actual U h_t over valid positions (per-state mean removed) | same, states with all pulse units off | share of states with a pulse unit on")
for t in range(1, 10):
    full = (H[:, t] @ U.T)[:, 9 - t:]
    part = (H[:, t][:, ORD] @ Uo.T)[:, 9 - t:]
    y = full - full.mean(1, keepdims=True); p_ = part - part.mean(1, keepdims=True)
    off = (H[:, t][:, PULSE] == 0).all(1)
    P(f" {t} | {1 - ((y - p_) ** 2).sum() / (y ** 2).sum():.4f} | {1 - ((y[off] - p_[off]) ** 2).sum() / (y[off] ** 2).sum():.4f} | {1 - off.mean():.4f}")
