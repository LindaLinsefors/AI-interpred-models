"""I1: why is the wipe result a knife-edge in the scale c for k <= 4?  Seeds 6010, 6011.

Hypothesis: with the comparators at exactly 0, pulse units 5 and 9 fire at t0 + 1 as after a record (unit 5 writes a
high score for the input of step t0, unit 9 - gain W[9,ORD].w[ORD] = 84.3 per unit of input - fires only if the
comparators are within a few % of zero and takes that score away again).  A residual of 2 % leaves unit 5 on and
switches unit 9 off, so the network answers t0.
"""
from v3lib import *

out = Tee("a10_knife_edge.txt")
x, pM, pS, M, S, nb = gather_I1(500_000, 6010)
out(f"I1 pool: seeds 6010..{6010 + nb - 1}, n = {len(x)}")
out(f"W[5,ORD].w[ORD] = {W[5, ORD] @ w[ORD]:.3f}, W[9,ORD].w[ORD] = {W[9, ORD] @ w[ORD]:.3f}, W[5,COMP].w[COMP] = {W[5, COMP] @ w[COMP]:.3f}, "
    f"W[9,COMP].w[COMP] = {W[9, COMP] @ w[COMP]:.3f}")


def run_two(xv, t0, fn0, fn1):
    B = len(xv)
    h = np.zeros((B, N)); h1 = np.zeros((B, N))
    for t in range(T):
        h = np.maximum(xv[:, t, None] * w + h @ W.T, 0.0)
        rows = np.flatnonzero(t0 == t)
        if rows.size and fn0 is not None:
            h[rows] = fn0(h[rows], rows)
        rows = np.flatnonzero(t0 + 1 == t)
        if rows.size:
            if fn1 is not None:
                h[rows] = fn1(h[rows], rows)
            h1[rows] = h[rows]
    return h, h1


def set_units(units, values):
    def fn(h, rows):
        h = h.copy(); h[:, units] = values[rows][:, units] if values is not None else 0.0
        return h
    return fn


out("\nrows: k = 1..6 (t0 <= 7).  entries: P(ans = pM) | P(ans = t0);   then median h5/M and h9/M at t0 + 1")
for label, c in (("c = 0 (wipe)", 0.0), ("c = 0.005", 0.005), ("c = 0.01", 0.01), ("c = 0.02", 0.02), ("c = 0.05", 0.05), ("c = 0.1", 0.1), ("c = 1 (nothing)", 1.0)):
    row = []
    for k in range(1, 7):
        v = pM + k <= 7
        t0 = (pM + k)[v]
        h9, h1 = run_two(x[v], t0, scale_units(COMP, c), None)
        a = answer(h9)
        row.append(f"{np.mean(a == pM[v]):.3f}|{np.mean(a == t0):.3f} h5 {np.median(h1[:, 5] / M[v]):5.2f} h9 {np.median(h1[:, 9] / M[v]):4.2f} P(h9>0) {np.mean(h1[:, 9] > 0):.2f}")
    out(f"  {label:16s} " + "   ".join(row))

out("\npatching at t0 + 1 (P(ans = pM) | P(ans = t0), k = 1..6):")
tests = [("wipe, unit 9 at t0+1 := 0", 0.0, [9], "zero"), ("wipe, unit 5 at t0+1 := 0", 0.0, [5], "zero"), ("wipe, units 5 and 9 := 0", 0.0, [5, 9], "zero"),
         ("c = 0.02, unit 9 := its value in the wiped run", 0.02, [9], "wiped"), ("c = 0.02, unit 5 := 0", 0.02, [5], "zero"),
         ("c = 0.02, units 5 and 9 := 0", 0.02, [5, 9], "zero"),
         ("nothing, unit 5 := its value in the wiped run", 1.0, [5], "wiped"), ("nothing, unit 9 := its value in the wiped run", 1.0, [9], "wiped")]
for label, c, units, src in tests:
    row = []
    for k in range(1, 7):
        v = pM + k <= 7
        t0 = (pM + k)[v]
        _, h1w = run_two(x[v], t0, scale_units(COMP, 0.0), None)
        h9, _ = run_two(x[v], t0, scale_units(COMP, c), set_units(units, h1w if src == "wiped" else None))
        a = answer(h9)
        row.append(f"{np.mean(a == pM[v]):.3f}|{np.mean(a == t0):.3f}")
    out(f"  {label:48s} " + "  ".join(row))
