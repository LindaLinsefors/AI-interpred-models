"""Code review of e10_tape_shift.py and fig_mechanism panel (e) (claims T1, T2).

 (a) T1 numbers recomputed (largest 'other' entry in rows 0..6; row 9 range) and a null: how small is the
     residual |Uo A - Sig Uo| / |Uo A| for a readout that has nothing to do with A?  (Uo has rank 10 in a
     13-dim space, so some closure is automatic.)
 (b) T2 with other R2 conventions (the claim: per-state mean over positions 0..8 removed, pooled SS).
 (c) T2 is about a HYPOTHETICAL step (all 13 ordinary units on, pulses ignored, no input).  How well does
     Sig, or a plain shift, predict the scores after a REAL step, s_{t+1} = U h_{t+1} from s_t = U h_t ?
     Also: how often is the 'resting' gate pattern (13 on, pulses off) the actual pattern?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *

Uo = U[:, ORD]
A = W[np.ix_(ORD, ORD)]
Sig = Uo @ A @ np.linalg.pinv(Uo)
res = np.linalg.norm(Uo @ A - Sig @ Uo) / np.linalg.norm(Uo @ A)
print("(a) relative residual:", round(res, 4), "  rank(Uo) =", np.linalg.matrix_rank(Uo))
off = Sig.copy()
for i in range(9):
    off[i, i + 1] = 0
print("    largest |entry| other than Sig[i, i+1], per row 0..8:", np.round(np.abs(off[:9]).max(1), 3))
print("    rows 0..6: max =", round(float(np.abs(off[:7]).max()), 3), "  (the claim says <= 0.42; 0.42 is Sig[7,1] in row 7)")
print("    superdiagonal Sig[i, i+1], i = 0..8:", np.round([Sig[i, i + 1] for i in range(9)], 3))
print("    row 9: min", round(float(Sig[9].min()), 3), "max", round(float(Sig[9].max()), 3))
# the same residual with the mean over positions 0..8 projected out of the rows 0..8 (what the argmax over old positions sees)
P = np.eye(10); P[:9, :9] -= 1 / 9
print("    residual after removing the mean over positions 0..8 (rows 0..8 only):",
      round(float(np.linalg.norm((P @ (Uo @ A - Sig @ Uo))[:9]) / np.linalg.norm((P @ Uo @ A)[:9])), 4))

rng = np.random.default_rng(0)
null1, null2, null3 = [], [], []
for _ in range(2000):
    G = rng.standard_normal((10, 13))                                   # random readout
    S_ = G @ A @ np.linalg.pinv(G)
    null1.append(np.linalg.norm(G @ A - S_ @ G) / np.linalg.norm(G @ A))
    Q, _ = np.linalg.qr(rng.standard_normal((13, 13)))                   # the real readout, rotated at random in state space
    G = Uo @ Q
    S_ = G @ A @ np.linalg.pinv(G)
    null2.append(np.linalg.norm(G @ A - S_ @ G) / np.linalg.norm(G @ A))
    G = Uo[:, rng.permutation(13)]                                       # the real readout with its 13 columns permuted
    S_ = G @ A @ np.linalg.pinv(G)
    null3.append(np.linalg.norm(G @ A - S_ @ G) / np.linalg.norm(G @ A))
for nm, v in (("random Gaussian readout", null1), ("real readout rotated by a random orthogonal matrix", null2), ("real readout with permuted columns", null3)):
    v = np.array(v)
    print(f"    null ({nm}): mean {v.mean():.3f}, 1% quantile {np.quantile(v, 0.01):.3f}, min {v.min():.3f}   (real: {res:.3f})")

x = sample(100_000, 1010)
H, Z = run(x, return_pre=True)
st = running_stats(x)
print("\n(b) T2 on the same sample as e10: R2 of Sig-prediction of Uo A h from Uo h, positions 0..8")
print("     t | e10 (per-state mean removed, pooled) | also centred per position across states | no mean removal, uncentred | no mean removal, centred per position | worst single position (centred per position)")
for t in (3, 5, 7):
    h = H[:, t][:, ORD]
    l0 = h @ Uo.T; l1 = (h @ A.T) @ Uo.T; pred = l0 @ Sig.T
    c0 = (l1 - l1[:, :9].mean(1, keepdims=True))[:, :9]; cp = (pred - pred[:, :9].mean(1, keepdims=True))[:, :9]
    r_e10 = 1 - ((c0 - cp) ** 2).sum() / (c0 ** 2).sum()
    r_cc = 1 - ((c0 - cp) ** 2).sum() / ((c0 - c0.mean(0)) ** 2).sum()
    a, b = l1[:, :9], pred[:, :9]
    r_raw = 1 - ((a - b) ** 2).sum() / (a ** 2).sum()
    r_rawc = 1 - ((a - b) ** 2).sum() / ((a - a.mean(0)) ** 2).sum()
    per = 1 - ((c0 - cp) ** 2).sum(0) / ((c0 - c0.mean(0)) ** 2).sum(0)
    print(f"     {t} | {r_e10:.4f} | {r_cc:.4f} | {r_raw:.4f} | {r_rawc:.4f} | {per.min():.4f} at position {per.argmin()}   (all: {np.round(per, 3)})")

print("\n(c) REAL steps.  Fraction of states with the resting gate pattern (13 ordinary units on, pulses off):")
G = Z > 0
rest = G[:, :, ORD].all(2) & ~G[:, :, PULSE].any(2)
print("     by t:", np.round(rest.mean(0), 3), " overall:", round(float(rest.mean()), 3))
print("     among states with M_t > 0.3, a1 >= 2, x_t < S_t:", round(float(rest[(st['M1'] > 0.3) & (st['a1'] >= 2) & (st['a2'] >= 1)].mean()), 3))


def r2pair(actual, pred):
    """e10's convention: positions 0..8, per-state mean removed, pooled; plus best common slope."""
    a = (actual - actual[:, :9].mean(1, keepdims=True))[:, :9]; p = (pred - pred[:, :9].mean(1, keepdims=True))[:, :9]
    sl = (a * p).sum() / (p * p).sum()
    return 1 - ((a - p) ** 2).sum() / (a ** 2).sum(), 1 - ((a - sl * p) ** 2).sum() / (a ** 2).sum(), sl


S_all = H @ U.T
shift = np.zeros((10, 10))
for i in range(9):
    shift[i, i + 1] = 1.0
print("\n     actual s_{t+1} = U h_{t+1} (positions 0..8, per-state mean removed) predicted from s_t = U h_t:")
print("     selection                                 t->t+1 |  Sig s_t: R2 (slope 1), R2 (best slope), slope | pure shift: R2 (slope 1), R2 (best slope), slope |  n")
for lab, selfun in (
        ("all states", lambda t: np.ones(len(x), bool)),
        ("no event: x_{t+1} < S_t, S_t > 0.1, all four comparators on at t+1 (e15b iv)", lambda t: (x[:, t + 1] < st["M2"][:, t]) & (st["M2"][:, t] > 0.1) & G[:, t + 1][:, COMP].all(1)),
        ("same and resting gate pattern at t+1", lambda t: (x[:, t + 1] < st["M2"][:, t]) & (st["M2"][:, t] > 0.1) & rest[:, t + 1]),
        ("new max at t+1 (x_{t+1} > M_t > 0.3)", lambda t: (x[:, t + 1] > st["M1"][:, t]) & (st["M1"][:, t] > 0.3)),
):
    print(f"     -- {lab}")
    for t in (2, 4, 6, 8):
        s = selfun(t)
        a = S_all[s, t + 1]; l0 = S_all[s, t]
        r1 = r2pair(a, l0 @ Sig.T); r2_ = r2pair(a, l0 @ shift.T)
        # only indices that hold real inputs after the step: positions of inputs 0..t  -> indices 9-(t+1) .. 8
        print(f"        {t}->{t+1} |  {r1[0]:7.3f}  {r1[1]:6.3f}  {r1[2]:5.2f} | {r2_[0]:7.3f}  {r2_[1]:6.3f}  {r2_[2]:5.2f} |  {s.sum()}")
print("\n     the same restricted to the indices that hold real inputs after the step (inputs 0..t-1 -> indices 8-t..7), per-state mean over those removed:")
for lab, selfun in (
        ("all states", lambda t: np.ones(len(x), bool)),
        ("no event (as above)", lambda t: (x[:, t + 1] < st["M2"][:, t]) & (st["M2"][:, t] > 0.1) & G[:, t + 1][:, COMP].all(1)),
):
    print(f"     -- {lab}")
    for t in (2, 4, 6, 8):
        s = selfun(t)
        idx = np.arange(8 - t, 8)
        a = S_all[s, t + 1][:, idx]
        row = []
        for Mx in (Sig, shift):
            p = (S_all[s, t] @ Mx.T)[:, idx]
            aa = a - a.mean(1, keepdims=True); pp = p - p.mean(1, keepdims=True)
            sl = (aa * pp).sum() / (pp * pp).sum()
            row.append(f"{1 - ((aa - pp) ** 2).sum() / (aa ** 2).sum():7.3f}  {1 - ((aa - sl * pp) ** 2).sum() / (aa ** 2).sum():6.3f}  {sl:5.2f}")
        print(f"        {t}->{t+1} |  Sig: " + row[0] + "  | pure shift: " + row[1] + f" |  {s.sum()}")
