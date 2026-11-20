"""E1. Verify the exact source identity numerically (max abs error), on N(0,1) and Unif(-1,1) inputs,
and the closed form of the comparator / unit-4 amplitudes in terms of thresholds."""
from src import *

for dist, seed in (("normal", 1000), ("uniform", 1001)):
    x = sample(200_000, seed, dist)
    H, Z, p, c = amplitudes(x)
    L = logits(H)
    # (a) state identity
    inj = x[:, :, None] * w[ORD] + c                                    # (B, 10, 13)
    inj[:, 1:] += p[:, :-1] @ W[np.ix_(ORD, PULSE)].T
    h9 = np.einsum("aij,baj->bi", APOW[AGE], inj)
    err_state = np.abs(h9 - H[:, 9, ORD]).max()
    # (b) logit identity through the kernels
    CX, CP, CC = contributions(x, p, c)
    Lsum = CX.sum(1) + CP.sum((1, 2)) + CC.sum((1, 2))
    err_logit = np.abs(Lsum - L).max()
    # (c) same through the position-selecting path used later
    pos = np.stack([second_argmax(x), x.argmax(1)], 1)
    CXs, CPs, CCs = contributions(x, p, c, pos)
    Lsel = CXs.sum(1) + CPs.sum((1, 2)) + CCs.sum((1, 2))
    err_sel = np.abs(Lsel - np.take_along_axis(L, pos, 1)).max()
    print(f"{dist:8s} seed {seed} n={len(x)}: max|h9 err| = {err_state:.2e}   max|logit err| = {err_logit:.2e}   "
          f"max|selected-position err| = {err_sel:.2e}   (max|logit| = {np.abs(L).max():.1f}, max|h9| = {np.abs(H[:, 9]).max():.1f})")
    print(f"   min clip amount = {c.min():.2e} (must be >= 0), min pulse = {p.min():.2e}")
    # (d) closed forms: comparator clip = |w_j| max(x_s - theta_j, 0); unit 4 = w_4 max(x_s - theta_4, 0)
    Hprev = np.concatenate([np.zeros((len(x), 1, 16)), H[:, :-1]], 1)
    for j in COMP:
        theta = -(Hprev @ W[j]) / w[j]
        e = np.abs(c[:, :, ORD.index(j)] - abs(w[j]) * np.maximum(x - theta, 0)).max()
        print(f"   comparator {j}: max|c - |w_j| (x - theta_j)_+| = {e:.2e}")
    theta4 = -(Hprev @ W[4]) / w[4]
    print(f"   unit 4: max|h4 - w_4 (x - theta_4)_+| = {np.abs(H[:, :, 4] - w[4] * np.maximum(x - theta4, 0)).max():.2e}")
    # negative control: drop the clip term -> identity must fail
    h9_noclip = np.einsum("aij,baj->bi", APOW[AGE], inj - c)
    print(f"   control without the clip term: max|h9 err| = {np.abs(h9_noclip - H[:, 9, ORD]).max():.1f}")
    # sizes of the three groups of terms in the logits (rms over sequences and positions)
    print(f"   rms logits {L.std():.2f};  rms of class sums: raw {CX.sum(1).std():.2f}, pulses {CP.sum((1,2)).std():.2f}, clips {CC.sum((1,2)).std():.2f}")
