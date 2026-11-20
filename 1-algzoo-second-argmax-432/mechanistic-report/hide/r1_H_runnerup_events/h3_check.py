"""H3 check: for N events with no comparator clipped, the counterfactual mode change at t+2 is the unit-5 pulse times
its write vector LEFT_k . W[ORD,5] (prediction from the weights)."""
from hcommon import *
from h3_event import load_events

d = load_events()
s = (d["ncomp"] == 0) & (d["t"] <= 7) & (d["q2"] < 0)
print("check: N events with NO comparator clipped (t<=7, two negative followers), n =", s.sum(), "; ages:", {a: int((s & (d['a'] == a)).sum()) for a in range(1, 9)})
print("  regress the counterfactual mode change at t+2 on the unit-5 pulse alone: dm_k = c_k * h5.  Prediction from the weights: c_k = LEFT_k . W[ORD,5]")
print("  mode    | fitted c_k           | LEFT_k.W[ORD,5]      | R2")
for n in NAMES:
    y = d[f"dm_{n}"][s]; h = d["h5"][s] - d["h5cf"][s]
    c = (h @ y) / (h @ h)
    R2 = 1 - np.sum(np.abs(y - c * h) ** 2) / np.sum(np.abs(y) ** 2)
    pred = LEFT[n] @ W[ORD, 5]
    f = (lambda v: f"{abs(v):.3f}@{np.degrees(np.angle(v)):7.1f}") if IS_COMPLEX[n] else (lambda v: f"{np.real(v):8.3f}       ")
    print(f"  {n:7s} | {f(c)}      | {f(pred)}      | {R2:.4f}")
