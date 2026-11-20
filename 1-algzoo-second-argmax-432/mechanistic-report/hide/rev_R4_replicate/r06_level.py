"""R4 / r06: section 8, the level-removal experiment and its controls.
At step t0 (3, 5, 7) the level (or clock) coordinate of h_t0 is scaled by f; the run continues either
with 'quiet' inputs (x_t -> -|x_t| for t > t0, so the top two cannot change) or with the real inputs.
Seeds 7071-7072.  Output: r06_level_out.txt
"""
from rlib import *
import modes

out = open(__file__.replace(".py", "_out.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s); out.write(s + "\n"); out.flush()


Rl = modes.RIGHT["level"].real
Rc = modes.RIGHT["clock"]
P("level eigenvector: imaginary part max %.1e ; all entries positive: %s" % (np.abs(modes.RIGHT["level"].imag).max(), (Rl > 0).all() or (Rl < 0).all()))


def cont(h, xs, t0):
    """Continue from state h (after step t0) with inputs xs[:, t0+1:]. Returns final h and Z at t0+1."""
    z1 = None
    for t in range(t0 + 1, 10):
        z = xs[:, t, None] * w + h @ W.T
        if z1 is None:
            z1 = z
        h = np.maximum(z, 0.0)
    return h, z1


def edit(h, mode, f):
    h = h.copy()
    m = modes.coords(h)
    if mode == "level":
        h[:, ORD] += (f - 1) * np.outer(m["level"], Rl)
    else:
        h[:, ORD] += (f - 1) * 2 * np.real(np.multiply.outer(m["clock"], Rc))
    return h


N = 200_000
edits = [("level", 0.0), ("level", 0.25), ("level", 0.5), ("level", 0.75), ("level", 1.0), ("level", 1.25), ("level", 1.5),
         ("clock", 0.0), ("clock", 0.5), ("clock", 1.25), ("clock", 1.5)]
res = {}
for sd in (7071, 7072):
    x = gauss(N, sd)
    H, Z = fwd(x)
    st = prefix_stats(x)
    for t0 in (3, 5, 7):
        a1 = t0 - st["pM"][:, t0]
        pop = (a1 >= 2) & (st["S"][:, t0] > 0)
        xq = x.copy(); xq[:, t0 + 1:] = -np.abs(x[:, t0 + 1:])        # quiet continuation
        h0 = H[pop, t0]
        tgt = st["pS"][pop, t0]
        ref_h, ref_z1 = cont(h0, xq[pop], t0)
        for mode, f in edits:
            he = edit(h0, mode, f)
            hf, z1 = cont(he, xq[pop], t0)
            ans = (hf @ U.T).argmax(1)
            newly_off = ((z1[:, ORD] <= 0) & (ref_z1[:, ORD] > 0)).sum(1)
            r = res.setdefault((mode, f, t0), np.zeros(8))
            r += [pop.sum(), (ans == tgt).sum(), newly_off.sum(), (he[:, ORD] < 0).sum(), 0, 0, 0, 0]
        # real later inputs: all sequences with a1 >= 2 and M > 0, scored against the true answer
        pop2 = (a1 >= 2) & (st["M"][:, t0] > 0)
        true = st["pS"][pop2, -1]
        kept = st["pM"][pop2, -1] == st["pM"][pop2, t0]              # the max at t0 is still the final max
        ref_h2, ref_z2 = cont(H[pop2, t0], x[pop2], t0)
        for mode, f in edits:
            he = edit(H[pop2, t0], mode, f)
            hf, z1 = cont(he, x[pop2], t0)
            ok = (hf @ U.T).argmax(1) == true
            r = res[(mode, f, t0)]
            r[4:] += [pop2.sum(), ok.sum(), (ok & kept).sum(), kept.sum()]
            rr = res.setdefault(("real_off", mode, f, t0), np.zeros(2))
            rr += [((z1[:, ORD] <= 0) & (ref_z2[:, ORD] > 0)).sum(), pop2.sum()]

P("=== SECTION 8: quiet continuation, states with max >= 2 steps old and S(t0) > 0 (2 x 200k sequences, seeds 7071-7072)")
P("edit            t0 = 3 / 5 / 7:  accuracy | ordinary units newly OFF at t0+1 | negative entries in the edited state (of 13)")
for mode, f in edits:
    r = [res[(mode, f, t0)] for t0 in (3, 5, 7)]
    P("%-5s x %.2f   n %s   acc %s | newly off %s | negative %s" % (
        mode, f, "/".join("%d" % a[0] for a in r), " / ".join("%.3f" % (a[1] / a[0]) for a in r),
        " / ".join("%.2f" % (a[2] / a[0]) for a in r), " / ".join("%.2f" % (a[3] / a[0]) for a in r)))
P("\nreal later inputs, states with max >= 2 steps old and M(t0) > 0, scored against the true answer:")
P("edit            t0 = 3 / 5 / 7:  accuracy | acc if the max at t0 stays the max | acc if it is later overtaken | units newly OFF at t0+1")
for mode, f in edits:
    r = [res[(mode, f, t0)] for t0 in (3, 5, 7)]
    ro = [res[("real_off", mode, f, t0)] for t0 in (3, 5, 7)]
    P("%-5s x %.2f   acc %s | kept %s | overtaken %s | newly off %s" % (
        mode, f, " / ".join("%.3f" % (a[5] / a[4]) for a in r), " / ".join("%.3f" % (a[6] / a[7]) for a in r),
        " / ".join("%.3f" % ((a[5] - a[6]) / (a[4] - a[7])) for a in r), " / ".join("%.2f" % (a[0] / a[1]) for a in ro)))
out.close()
