"""B6 probe: level perturbation on clean two-pulse inputs (deterministic). M = 1.5 at p, S at q, background b."""
from blib import *

L, R = LEFT["level"].real, RIGHT["level"].real


def perturb(h, f):
    h2 = h.copy()
    h2[:, ORD] += ((f - 1) * (h[:, ORD] @ L))[:, None] * R[None, :]
    return h2


def show(p, q, Sv, b, t0, f, verbose=True):
    x = np.full((1, 10), b); x[0, p] = 1.5; x[0, q] = Sv
    H, Z = run(x, return_pre=True)
    Hs, Zs = run(x[:, t0 + 1:], return_pre=True, h0=perturb(H[:, t0], f))
    lb, lp = logits(H)[0], Hs[0, -1] @ U.T
    dg = [(t0 + 1 + k, [u for u in range(N) if (Zs[0, k, u] > 0) != (Z[0, t0 + 1 + k, u] > 0)]) for k in range(Hs.shape[1])]
    dg = [d for d in dg if d[1]]
    if verbose:
        print(f"p={p} q={q} S={Sv} b={b} t0={t0} f={f}: answer {lb.argmax()} -> {lp.argmax()};  gate changes (step, units): {dg}")
        print("   base logits     :", np.round(lb, 2))
        print("   perturbed logits:", np.round(lp, 2))
        print("   difference      :", np.round(lp - lb, 2))
    return lb.argmax(), lp.argmax(), dg


for b in (-1.0, -0.3):
    for (p, q) in ((1, 2), (2, 1), (0, 3)):
        for f in (1.2, 0.8):
            show(p, q, 1.0, b, 3, f)

print("\nsummary over all position pairs with max(p, q) <= t0 = 4, S in {0.3, 1.0, 1.4}, b in {-1, -0.3}: answers")
for f in (0.8, 0.9, 1.1, 1.2):
    tot = flip_p = flip_other = nogate_flip = 0
    for b in (-1.0, -0.3):
        for Sv in (0.3, 1.0, 1.4):
            for p in range(5):
                for q in range(5):
                    if p == q:
                        continue
                    a0, a1, dg = show(p, q, Sv, b, 4, f, verbose=False)
                    tot += 1
                    flip_p += (a1 == p) and (a0 == q)
                    flip_other += (a1 != a0) and (a1 != p)
                    nogate_flip += (a1 != a0) and not dg
    print(f"  f = {f}: {tot} cases; baseline answer q kept in {tot - flip_p - flip_other}; flipped to the max position p: {flip_p}; to another position: {flip_other}; flips without any gate change: {nogate_flip}")
