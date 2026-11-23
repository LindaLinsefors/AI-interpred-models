"""v4: ATTACK 7 and 8 on the summary "the network reads only (a_M, a_S, S/M) from the tape".

ATTACK 7. Sufficiency in the brief's sense (i) needs REAL donors with the same V, not an artificial one.
  donor R1: A with every input except the top two redrawn from N(0,1) below S          (same a_M, a_S, S/M exactly)
  donor R2: another real quiet sequence with the same (a_M, a_S) and the nearest S/M, rescaled to A's M
  fw = rest of A + tape of the donor;  rv = rest of the donor + tape of A (truth = the donor's ... same position).
  Split by whether a third lies within 0.1 M of the runner-up in A or in the donor.
ATTACK 8. S_pos transplants followed by POSITIVE low inputs (the investigator used negative ones only).
"""
from scipy.stats import norm
from v_common import *

N = 500_000
ar_ = lambda n: np.arange(n)

for t in (9, 6):
    x, st = quiet(N, t, 6100 + t)
    rng = np.random.default_rng(6200 + t)
    n, pre = len(x), x[:, : t + 1]
    hA = state(x, t)
    # R1
    top2 = np.zeros(pre.shape, bool); top2[ar_(n), st["pM"]] = True; top2[ar_(n), st["pS"]] = True
    x1 = x.copy(); x1[:, : t + 1] = np.where(top2, pre, norm.ppf(rng.random(pre.shape) * norm.cdf(st["S"])[:, None]))
    # R2: neighbours in S/M within each cell (a_M, a_S)
    ratio = st["S"] / st["M"]
    order = np.lexsort((ratio, st["aS"], st["aM"]))
    partner = np.empty(n, int)
    partner[order[0::2][: len(order[1::2])]] = order[1::2]
    partner[order[1::2]] = order[0::2][: len(order[1::2])]
    if n % 2:
        partner[order[-1]] = order[-2]
    same = (st["aM"][partner] == st["aM"]) & (st["aS"][partner] == st["aS"]) & (np.abs(ratio[partner] - ratio) < 0.02)
    x2 = x[partner] * (st["M"] / st["M"][partner])[:, None]
    low = lows(st["M"], 9 - t, rng)
    put = lambda v, k: np.concatenate([low[:, :k], v[:, None], low[:, k + 1:]], 1)
    futs = {"read out at once": low} if t == 9 else {"3 low": low, "new max @7": put(1.3 * st["M"], 0), "new max @9": put(1.3 * st["M"], 2),
                                                    "new S @7": put((st["S"] + st["M"]) / 2, 0)}
    print(f"\nATTACK 7, t = {t}: {n} quiet states.  Accuracy against A's truth (fw) / the donor's truth (rv)")
    for lab, xb, ok in (("R1: all but the top two redrawn below S", x1, np.ones(n, bool)), ("R2: other real sequence, same (a_M, a_S), S/M within 0.02", x2, same)):
        sb = stats(xb, t)
        ok = ok & sb["quiet"] & (sb["pS"] == st["pS"]) & (sb["pM"] == st["pM"])
        hB = state(xb, t)
        fw, rv = hyb(hA, hB), hyb(hB, hA)
        nearA, nearB = st["T"] > st["S"] - 0.1 * st["M"], sb["T"] > sb["S"] - 0.1 * sb["M"]
        dT = np.linalg.norm(part(hB) - part(hA), axis=1) / np.linalg.norm(part(hA), axis=1)
        print(f"   {lab}: n = {ok.sum()};  |tape_B - tape_A| / |tape_A| median {np.median(dT[ok]):.2f} (quartiles {np.quantile(dT[ok], .25):.2f}, {np.quantile(dT[ok], .75):.2f});"
              f" negative entry fw {np.mean((fw[ok] < -1e-9).any(1)):.2f}")
        for fl, f in futs.items():
            if t == 9:
                tA = tB = st["pS"]
                f = f[:, :0]
            else:
                tA = np.argsort(np.concatenate([pre, f], 1), 1)[:, -2]
                tB = np.argsort(np.concatenate([xb[:, : t + 1], f], 1), 1)[:, -2]
            vf = ok & (tA == tB) if "new S" not in fl else ok & (tA == tB) & (sb["S"] < f[:, 0]) & (f[:, 0] < sb["M"])
            cA, cB, cf, cr, cc = final(hA, f) == tA, final(hB, f) == tB, final(fw, f) == tA, final(rv, f) == tB, final(fw, f, True) == tA
            print(f"      {fl:17s} n = {vf.sum():6d}: A intact {cA[vf].mean():.3f}, donor intact {cB[vf].mean():.3f}, fw {cf[vf].mean():.3f} [clipped {cc[vf].mean():.3f}], rv {cr[vf].mean():.3f}")
            if fl in ("read out at once", "3 low"):
                for l2, m in (("no third within 0.1 M of S in A or donor", vf & ~nearA & ~nearB), ("near third in the donor only", vf & ~nearA & nearB),
                              ("near third in A only", vf & nearA & ~nearB), ("near third in both", vf & nearA & nearB)):
                    print(f"         {l2:42s} n = {m.sum():6d}: A intact {cA[m].mean():.3f}, fw {cf[m].mean():.3f}, rv {cr[m].mean():.3f}")
                farA = st["T"] < st["S"] - 0.4 * st["M"]
                farB = sb["T"] < sb["S"] - 0.4 * sb["M"]
                m = vf & farA & farB
                print(f"         {'third more than 0.4 M below S in both':42s} n = {m.sum():6d}: A intact {cA[m].mean():.3f}, fw {cf[m].mean():.3f}, rv {cr[m].mean():.3f}")
                hit = cf
                print("         fw by a_S: " + "  ".join(f"{s}: {hit[vf & (st['aS'] == s)].mean():.3f}" for s in range(2, t + 1)))

# ---------------------------------------------------------------- ATTACK 8
print("\nATTACK 8. S_pos transplant at T0, then low inputs of either sign (truth: the tape owner's / the rest owner's runner-up)")
for T0 in (6, 4, 8):
    x, st = quiet(N, T0, 6300 + T0)
    rng = np.random.default_rng(6400 + T0)
    j, ok = pick_low_slot(x, st, T0, rng)
    xB = swapped(x, st["pS"], j)
    ok &= stats(xB, T0)["quiet"]
    a, b = sub(st, ok), sub(stats(xB, T0), ok)
    hA_, hB_ = state(x[ok], T0), state(xB[ok], T0)
    fw = hyb(hA_, hB_)
    K = 9 - T0
    for lab, fut in (("negative, -M U(0.05, 0.6)", lows(a["M"], K, rng)), ("positive, S U(0.05, 0.5)", a["S"][:, None] * rng.uniform(0.05, 0.5, (len(hA_), K))),
                     ("positive, S U(0.5, 0.9)", a["S"][:, None] * rng.uniform(0.5, 0.9, (len(hA_), K))), ("zeros", np.zeros((len(hA_), K)))):
        f = final(fw, fut)
        print(f"   T0 = {T0}, {K} inputs {lab:27s} n = {ok.sum()}: tape's S {np.mean(f == b['pS']):.3f}, rest's S {np.mean(f == a['pS']):.3f},"
              f" a later input {np.mean(f > T0):.3f};  intact A {np.mean(final(hA_, fut) == a['pS']):.3f}, intact B {np.mean(final(hB_, fut) == b['pS']):.3f}")
