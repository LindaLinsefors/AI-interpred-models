"""Functional test of the description of V: replace the comparators' vote by a MODEL of it and see whether
m + V_model still ranks the runner-up first.

t = 9, all four comparators on, S > 0.1, both the max and the runner-up at positions <= 7; scores compared on
indices 0..7 only (where the readout columns are sinusoidal).  Models of V (fitted on half the data, scored on
the other half):
  L0  no vote at all (m alone)
  L1  literal:  -60 (M - 0.87 S) cos(omega (i - p1))
  L2  literal with fitted constants K0, c
  L3  the lead's regression per a1:        A_i (M - S) + B_i S + C_i x9
  L4  the same regression per (a1, a2) cell
  L5  the omega-sinusoid + offset part of the TRUE V of that sequence (what the sinusoidal readout passes)
  true V

Usage: python -B x10_model_V_function.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 10

Vs, ms, cols = [], [], []
for x, H, M, S, p1, p2 in batches(nb, seed0):
    h = H[:, 9]
    sel = (h[:, COMP] > 0).all(1) & (S[:, 9] > 0.1) & (p1[:, 9] <= 7) & (p2[:, 9] <= 7)
    s, V, m = scores(h[sel])
    Vs.append(V[:, :8]); ms.append(m[:, :8])
    cols.append(np.stack([M[sel, 9], S[sel, 9], x[sel, 9], p1[sel, 9], p2[sel, 9]], 1))
V = np.concatenate(Vs); m = np.concatenate(ms)
Mt, St, x9, P1, P2 = np.concatenate(cols).T
P1 = P1.astype(int); P2 = P2.astype(int); a1 = 9 - P1; a2 = 9 - P2
n = len(V); r = np.arange(n)
train = r % 2 == 0; test = ~train
X = np.stack([Mt - St, St, x9], 1)
pos = np.arange(8.0)
cw = np.cos(OMEGA * (pos[None, :] - P1[:, None]))
print(f"seeds {seed0}..{seed0 + nb - 1}; n = {n} (train {train.sum()}, test {test.sum()})")

models = {"L0 no vote (m alone)": np.zeros_like(V), "L1 literal -60 (M - 0.87 S) cos": -60 * (Mt - 0.87 * St)[:, None] * cw}
# L2: fitted K0, c (centred over positions, since an offset common to all positions does not change the ranking)
Vc = V - V.mean(1, keepdims=True); cwc = cw - cw.mean(1, keepdims=True)
D1 = -(Mt[:, None] * cwc); D2 = St[:, None] * cwc
cf = np.linalg.lstsq(np.stack([D1[train].ravel(), D2[train].ravel()], 1), Vc[train].ravel(), rcond=None)[0]
models[f"L2 fitted -{cf[0]:.1f} (M - {cf[1] / cf[0]:.3f} S) cos"] = -(cf[0] * Mt - cf[1] * St)[:, None] * cw
L3 = np.zeros_like(V); L4 = np.zeros_like(V)
for a in range(2, 10):
    k = a1 == a
    L3[k] = X[k] @ np.linalg.lstsq(X[k & train], V[k & train], rcond=None)[0]
    for b in range(2, 10):
        kk = k & (a2 == b)
        if (kk & train).sum() > 200:
            L4[kk] = X[kk] @ np.linalg.lstsq(X[kk & train], V[kk & train], rcond=None)[0]
models["L3 lead's regression per a1"] = L3
models["L4 same regression per (a1, a2) cell"] = L4
G = np.stack([np.cos(OMEGA * pos), np.sin(OMEGA * pos), np.ones(8)], 1)
models["L5 sinusoid part of the true V"] = (V @ np.linalg.pinv(G).T) @ G.T
models["true V"] = V

print("\nscored on the test half, indices 0..7:")
print(f"{'model of V':40s}  R^2 of V (centred per state)   P(argmax(m + V_model) = runner-up)   P(= max)   P(score[p2] > score[p1])   P(runner-up first among non-max)")
Vct = Vc[test]
for nm, Vm in models.items():
    sc = (m + Vm)[test]
    am = sc.argmax(1)
    Vmc = Vm[test] - Vm[test].mean(1, keepdims=True)
    r2 = 1 - ((Vct - Vmc) ** 2).sum() / (Vct ** 2).sum()
    rt = np.arange(test.sum())
    z = sc.copy(); z[rt, P1[test]] = -np.inf
    print(f"{nm:40s}  {r2:6.3f}                        {np.mean(am == P2[test]):.4f}                             {np.mean(am == P1[test]):.4f}"
          f"     {np.mean(sc[rt, P2[test]] > sc[rt, P1[test]]):.4f}                     {np.mean(z.argmax(1) == P2[test]):.4f}")

print("\nby gap/M of the top two (test half): P(argmax = runner-up) for L1, L3, L4, L5, true")
gap = (Mt - St) / Mt
for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.0)):
    c = test & (gap > lo) & (gap <= hi)
    out = []
    for nm in list(models)[1:]:
        if nm.startswith("L2"):
            continue
        out.append(f"{np.mean((m + models[nm])[c].argmax(1) == P2[c]):.3f}")
    print(f"  gap ({lo}, {hi}]: " + "  ".join(out) + f"   n = {c.sum()}")
