"""a3: use of a transplanted tape in the future.  Transplant at step T0 (quiet states), then continue with CHOSEN
inputs x_(T0+1)..x_9 and read the final answer.

truth_A / truth_B = the correct answer (position of the second largest) of A's / B's prefix followed by that future.
Hybrid "fw" = A's rest + B's tape, "rv" = B's rest + A's tape.  Reported for each pair kind and future:
   intact accuracies of A and B;
   where the truths differ:  P(answer = truth of the REST's owner), P(answer = truth of the TAPE's owner), P(other);
   where they agree:         P(answer = truth).
Clipped-at-zero versions in square brackets.

Futures (K = number of remaining steps; "low" = -M_A * U(0.05, 0.6), a negative input):
  low       low ... low
  nat       A's own remaining inputs
  newM@1    1.3 max(M_A, M_B) at the first remaining step, then low      (a new maximum; truth = the old maximum)
  newM@K    low ..., then 1.3 max(M_A, M_B) at step 9
  newS@1    z halfway between max(S_A, S_B) and min(M_A, M_B), then low  (a new runner-up; truth = that step)
  newS@K    the same z at step 9
  subS@1    y halfway between max(T_A, T_B, 0) and min(S_A, S_B), then low  (a new third; truth = the old runner-up)
  subS@K    the same y at step 9
  midS@1/K  z halfway between S_A and S_B (only S_val, |S_A - S_B| > 0.1 M): a new runner-up for one, a third for the other
  midM@1/K  z halfway between M_A and M_B (only M_val, |M_A - M_B| > 0.1 M): a new maximum for one, a runner-up for the other
"""
import sys
from tp import *

T0 = int(sys.argv[1]) if len(sys.argv) > 1 else 6
N = 600_000
K = 9 - T0
x, st = quiet_sample(N, T0, seed=31)
hA = state_at(x, T0)
rng = np.random.default_rng(300 + T0)
print(f"transplant at step {T0}, {K} further inputs; {len(x)} quiet states of {N} sequences")


def futures(kind, a, b, xa, rng):
    n = len(xa)
    low = -a["M"][:, None] * rng.uniform(0.05, 0.6, (n, K))
    put = lambda v, k: np.concatenate([low[:, :k], v[:, None], low[:, k + 1:]], 1)
    out = {"low": (low, np.ones(n, bool)), "nat": (xa[:, T0 + 1:], np.ones(n, bool))}
    nm = 1.3 * np.maximum(a["M"], b["M"])
    z = (np.maximum(a["S"], b["S"]) + np.minimum(a["M"], b["M"])) / 2
    zok = np.minimum(a["M"], b["M"]) - np.maximum(a["S"], b["S"]) > 0.1 * a["M"]
    th = np.maximum(np.maximum(a["T3"], b["T3"]), 0)
    y = (th + np.minimum(a["S"], b["S"])) / 2
    yok = np.minimum(a["S"], b["S"]) - th > 0.1 * a["M"]
    for lab, v, ok in (("newM", nm, np.ones(n, bool)), ("newS", z, zok), ("subS", y, yok)):
        out[lab + "@1"], out[lab + f"@{K}"] = (put(v, 0), ok), (put(v, K - 1), ok)
    for key, lab in (("S", "midS"), ("M", "midM")):
        if kind == key + "_val":
            v, ok = (a[key] + b[key]) / 2, np.abs(a[key] - b[key]) > 0.1 * a["M"]
            out[lab + "@1"], out[lab + f"@{K}"] = (put(v, 0), ok), (put(v, K - 1), ok)
    return out


def rates(ans, tR, tT):
    d = tR != tT
    r = lambda m, tgt: np.mean(ans[m] == tgt[m]) if m.any() else np.nan
    return d.mean(), r(d, tR), r(d, tT), r(~d, tR)


for kind in KINDS:
    xB, ok = make_pair(kind, x, st, T0, rng)
    sB = prefix_stats(xB, T0)
    ok &= sB["quiet"]
    xa, xb = x[ok], xB[ok]
    a, b = {k: v[ok] for k, v in st.items()}, {k: v[ok] for k, v in sB.items()}
    hA_, hB_ = hA[ok], state_at(xb, T0)
    fw, rv = hybrid(hA_, hB_), hybrid(hB_, hA_)
    print(f"\n--- {kind}: n = {ok.sum()};  hybrids with a negative entry: fw {neg_share(fw):.3f}, rv {neg_share(rv):.3f}")
    print("   future     n      acc A  acc B | truths differ: share | fw: rest's  tape's  other [clipped rest's tape's] |"
          " rv: rest's  tape's [clipped] | truths agree: fw correct [clipped]  rv correct")
    for lab, (fut, fok) in futures(kind, a, b, xa, rng).items():
        f = fut[fok]
        tA = second_argmax(np.concatenate([xa[fok, : T0 + 1], f], 1))
        tB = second_argmax(np.concatenate([xb[fok, : T0 + 1], f], 1))
        accA, accB = np.mean(finish(hA_[fok], f) == tA), np.mean(finish(hB_[fok], f) == tB)
        sh, fR, fT, fS = rates(finish(fw[fok], f), tA, tB)
        _, cR, cT, cS = rates(finish(fw[fok], f, clip=True), tA, tB)
        _, rR, rT, rS = rates(finish(rv[fok], f), tB, tA)
        _, dR, dT, dS = rates(finish(rv[fok], f, clip=True), tB, tA)
        print(f"   {lab:8s} {fok.sum():7d}  {accA:.3f}  {accB:.3f} | {sh:20.3f} |     {fR:.3f}   {fT:.3f}  {1 - fR - fT:.3f} "
              f"[{cR:.3f} {cT:.3f}]          |     {rR:.3f}   {rT:.3f} [{dR:.3f} {dT:.3f}] |"
              f"       {fS:.3f} [{cS:.3f}]   {rS:.3f}")
