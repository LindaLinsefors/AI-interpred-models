"""Follow-up to c06: T2 on REAL steps, with the choices that favour the claim.

Prediction: Sig applied to the ordinary-unit scores Uo h_t[ORD] (exactly e10's l0), or to the full
scores U h_t.  Target: the real next scores U h_{t+1}, or their ordinary-unit part Uo h_{t+1}[ORD].
Positions 0..8 as in e10, and positions 0..7 (index 8 is the slot that unit 5 writes directly).
Convention of e10: per-state mean over the positions removed, pooled SS, slope fixed at 1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *

Uo = U[:, ORD]
A = W[np.ix_(ORD, ORD)]
Sig = Uo @ A @ np.linalg.pinv(Uo)
x = sample(400_000, 1011)
H, Z = run(x, return_pre=True)
st = running_stats(x)
G = Z > 0
rest = G[:, :, ORD].all(2) & ~G[:, :, PULSE].any(2)
allon = G[:, :, COMP].all(2)


def r2(actual, pred, npos):
    a = (actual - actual[:, :npos].mean(1, keepdims=True))[:, :npos]; p = (pred - pred[:, :npos].mean(1, keepdims=True))[:, :npos]
    return 1 - ((a - p) ** 2).sum() / (a ** 2).sum()


sels = (
    ("hypothetical free step (e10): target Uo A h_t[ORD]", None),
    ("real step, all states", lambda t: np.ones(len(x), bool)),
    ("real step, no event: x_(t+1) < S_t, S_t > 0.1, four comparators on at t+1", lambda t: (x[:, t + 1] < st["M2"][:, t]) & (st["M2"][:, t] > 0.1) & allon[:, t + 1]),
    ("real step, no event and x_(t+1) < 0.5 S_t (unit 5 stays silent)", lambda t: (x[:, t + 1] < 0.5 * st["M2"][:, t]) & (st["M2"][:, t] > 0.1) & allon[:, t + 1]),
    ("real step, no event and the resting gate pattern at t+1", lambda t: (x[:, t + 1] < st["M2"][:, t]) & (st["M2"][:, t] > 0.1) & rest[:, t + 1]),
)
print("R2 (e10's convention) of the Sig-prediction; columns: prediction from Uo h_t[ORD] -> target U h_(t+1) | -> target Uo h_(t+1)[ORD] | prediction from U h_t -> target U h_(t+1);"
      "  each for positions 0..8 and for positions 0..7")
for lab, f in sels:
    print(f"-- {lab}")
    for t in (3, 5, 7, 8):
        h = H[:, t]
        if f is None:
            s = np.ones(len(x), bool)
            l0 = h[:, ORD] @ Uo.T
            tgt = (h[:, ORD] @ A.T) @ Uo.T
            print(f"   t={t}: positions 0..8: {r2(tgt, l0 @ Sig.T, 9):.4f}   positions 0..7: {r2(tgt, l0 @ Sig.T, 8):.4f}")
            continue
        s = f(t)
        l0o = h[s][:, ORD] @ Uo.T; l0f = h[s] @ U.T
        tf = H[s, t + 1] @ U.T; to = H[s, t + 1][:, ORD] @ Uo.T
        print(f"   t={t}->{t+1}: 0..8: {r2(tf, l0o @ Sig.T, 9):6.3f} | {r2(to, l0o @ Sig.T, 9):6.3f} | {r2(tf, l0f @ Sig.T, 9):6.3f}"
              f"     0..7: {r2(tf, l0o @ Sig.T, 8):6.3f} | {r2(to, l0o @ Sig.T, 8):6.3f} | {r2(tf, l0f @ Sig.T, 8):6.3f}     n = {s.sum()}")

print("\nargmax test: does the argmax over the positions of inputs 0..t survive a real no-event step as a shift by one index?")
for t in (3, 5, 7, 8):
    s = (x[:, t + 1] < st["M2"][:, t]) & (st["M2"][:, t] > 0.1) & allon[:, t + 1]
    l0 = H[s, t] @ U.T; l1 = H[s, t + 1] @ U.T
    valid0 = np.arange(10) >= 9 - t
    a0 = np.where(valid0, l0, -np.inf).argmax(1) + t - 9
    valid1 = (np.arange(10) >= 8 - t) & (np.arange(10) <= 8)
    a1_ = np.where(valid1, l1, -np.inf).argmax(1) + t - 8
    pred = np.where(valid1, l0 @ Sig.T, -np.inf).argmax(1) + t - 8
    print(f"   t={t}->{t+1}: P(same input on top among inputs 0..t) = {np.mean(a0 == a1_):.3f};  P(argmax of Sig s_t = real argmax at t+1) = {np.mean(pred == a1_):.3f}   n = {s.sum()}")
