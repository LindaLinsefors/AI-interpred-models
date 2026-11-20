"""R4 / r10: section 5.4, 'the removal of the comparators' share of the clock accounts for +28.0 of +28.3'.
Sequences whose final max (at s >= 3) overtook the true runner-up (old max at p0, g >= 2), old runner-up > 0.
The comparators that are cut at step s hold 0 instead of their (negative) pre-activation.  The part of that
difference that is the propagated clock mode, 2 Re(lambda z_(s-1) R_clock[j]), is pushed to the logits with
the gates of the actual run held fixed, and read at p0 relative to the bystander positions.
Seeds 7111-7112.  Output: r10_clock_share_out.txt
"""
from rlib import *
import modes

out = open(__file__.replace(".py", "_out.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s); out.write(s + "\n"); out.flush()


Rc, lam = modes.RIGHT["clock"], modes.LAM["clock"]
ci = [ORD.index(j) for j in COMP]
acc = {k: [] for k in ("created", "clock", "comp_all", "u4", "s", "g", "clock_q0", "peak")}
for sd in (7111, 7112):
    x = gauss(400_000, sd)
    st = prefix_stats(x)
    pM, pS = st["pM"][:, -1], st["pS"][:, -1]
    rows = np.arange(len(x))
    S0 = prev(st["S"], -np.inf)[rows, pM]
    sel = (pS < pM) & (pM - pS >= 2) & (pM >= 3) & (S0 > 0)
    xs, s, p0 = x[sel], pM[sel], pS[sel]
    B = len(xs); r = np.arange(B)
    q0 = prev(prefix_stats(xs)["pS"], -1)[r, s]
    S0 = S0[sel]
    H, Z = fwd(xs)
    lg = H[:, -1] @ U.T
    xc = xs.copy(); xc[r, s] = np.minimum(S0, 0) - 1.0
    lgc = fwd(xc)[0][:, -1] @ U.T
    pos = np.arange(10)[None, :]
    bys = (pos < s[:, None]) & (pos != p0[:, None]) & (pos != q0[:, None])
    rel = lambda A, p: A[r, p] - (A * bys).sum(1) / bys.sum(1)
    # perturbations of h_s
    zc = modes.coords(H[r, s - 1])["clock"]
    prop = 2 * np.real(np.multiply.outer(lam * zc, Rc))            # propagated clock mode on ORD at step s
    cut = Z[r, s][:, COMP] <= 0
    d_clock = np.zeros((B, 16)); d_clock[:, COMP] = np.where(cut, -prop[:, ci], 0.0)
    d_comp = np.zeros((B, 16)); d_comp[:, COMP] = np.where(cut, -Z[r, s][:, COMP], 0.0)
    d_u4 = np.zeros((B, 16)); d_u4[:, 4] = H[r, s, 4]
    # readout with the gates of the actual run: logits = Rm h_s + (terms of later inputs)
    dl = {k: np.zeros((B, 10)) for k in ("clock", "comp_all", "u4")}
    for sv in range(3, 10):
        m = s == sv
        Rm = np.broadcast_to(U, (m.sum(), 10, 16)).copy()
        for t in range(9, sv, -1):
            Rm = (Rm * (Z[m, t] > 0)[:, None, :]) @ W
        for k, d in (("clock", d_clock), ("comp_all", d_comp), ("u4", d_u4)):
            dl[k][m] = np.einsum("bpk,bk->bp", Rm, d[m])
    acc["created"].append(rel(lg, p0) - rel(lgc, p0))
    for k in ("clock", "comp_all", "u4"):
        acc[k].append(rel(dl[k], p0))
    acc["clock_q0"].append(rel(dl["clock"], q0))
    earlier = np.where(pos < s[:, None], dl["clock"], -np.inf)
    acc["peak"].append(earlier.argmax(1) == p0)
    acc["s"].append(s); acc["g"].append(s - p0)
A = {k: np.concatenate(v) for k, v in acc.items()}
P("n = %d sequences (final max at s>=3 overtook the runner-up, g>=2, old runner-up > 0), seeds 7111-7112" % len(A["s"]))
P("at the old max p0, relative to bystander positions:")
P("  created by the new max (actual - no-new-max counterfactual): %.2f" % A["created"].mean())
P("  clock part of the comparators' cut (their share of the propagated clock removed): %.2f" % A["clock"].mean())
P("  whole comparator cut: %.2f  (so the non-clock part is %.2f) ; unit-4 pulse: %.2f ; remainder (gate changes etc.): %.2f" % (
    A["comp_all"].mean(), (A["comp_all"] - A["clock"]).mean(), A["u4"].mean(), (A["created"] - A["comp_all"] - A["u4"]).mean()))
P("  clock part at the old runner-up q0: %.2f ; clock part peaks at p0 among earlier positions in %.3f" % (A["clock_q0"].mean(), A["peak"].mean()))
P("  by s = 3..9: created %s" % np.round([A["created"][A["s"] == v].mean() for v in range(3, 10)], 1))
P("               clock   %s" % np.round([A["clock"][A["s"] == v].mean() for v in range(3, 10)], 1))
P("  per-sequence correlation created vs clock part: %.2f" % np.corrcoef(A["created"], A["clock"])[0, 1])
out.close()
