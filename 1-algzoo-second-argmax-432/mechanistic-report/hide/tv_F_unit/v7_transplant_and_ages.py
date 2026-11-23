"""V7. Two extra checks.
A. Does the runner-up's identity among old inputs travel with the tape or with the rest?  Quiet final states,
   random quiet donor (rescaled by the ratio of the maxima), score-level:  rest(recipient) + tape(donor)  and
   tape(recipient) + rest(donor).  P(top old entry = recipient's runner-up) and P(= donor's runner-up).
   Also restricted to donors with a different runner-up position, and to pairs with the same age of the maximum.
B. Claim 10 (weak): coefficient of S in each tape coordinate per cell (a_M, a_S), ratio between successive a_S,
   against the mode's eigenvalue (the ratio does not depend on the gauge of the eigenvector).
Run:  python -B v7_transplant_and_ages.py > v7_transplant_and_ages.txt"""
from vcommon import *

rng = np.random.default_rng(9700)
d = population(sample(500_000, 9701)); n = len(d["h"]); ar = np.arange(n)
pc = pieces(d["h"]); full = d["h"] @ U.T; tape = sum(pc[m] for m in TAPE); rest = full - tape
M = d["M"]; dn = rng.permutation(n); sc = (M / M[dn])[:, None]
print(f"A. {n} quiet final states (seed 9701), random quiet donors")
def report(nm, Sc, q):
    tp = top_old(Sc, d["valid"])
    print(f"   {nm:44s} n {q.sum():6d}  top old entry = recipient's runner-up {np.mean((tp == d['iS'])[q]):.3f}   = donor's runner-up {np.mean((tp == d['iS'][dn])[q]):.3f}"
          f"   = recipient's maximum {np.mean((tp == d['iM'])[q]):.3f}   = donor's maximum {np.mean((tp == d['iM'][dn])[q]):.3f}")
diff = d["iS"] != d["iS"][dn]
for lab, q in (("all pairs", np.ones(n, bool)), ("donor's runner-up elsewhere", diff), ("... and the same age of the maximum", diff & (d["aM"] == d["aM"][dn])),
               ("... and the donor's runner-up not at the recipient's maximum", diff & (d["iS"][dn] != d["iM"]) & (d["iS"] != d["iM"][dn]))):
    print(f"  {lab}")
    report("recipient's rest + DONOR's tape piece", rest + tape[dn] * sc, q)
    report("recipient's tape piece + DONOR's rest", tape + rest[dn] * sc, q)

print("\nB. coefficient of S/M in each tape coordinate / M, fitted per cell (a_M, a_S) on [1, S/M]; ratio coefficient(a_S + 1) / coefficient(a_S) as modulus@angle")
ho = d["h"][:, ORD]; sm = d["S"] / M
for m in TAPE:
    z = (ho @ LVEC[m]) / M; lam = EIG[m]
    def coef(aM, aS):
        q = (d["aM"] == aM) & (d["aS"] == aS); F = np.stack([np.ones(q.sum()), sm[q]], 1)
        b = np.linalg.lstsq(F, np.stack([z[q].real, z[q].imag], 1), rcond=None)[0]; return b[1, 0] + 1j * b[1, 1], q.sum()
    print(f"   {m}: eigenvalue {abs(lam):.3f}@{np.degrees(np.angle(lam)):.0f}")
    for aM, rng_ in ((9, range(2, 9)), (2, range(3, 10)), (6, [2, 3, 4, 5, 7, 8, 9])):
        cs = [coef(aM, a)[0] for a in rng_]; ns = [coef(aM, a)[1] for a in rng_]
        rat = [cs[j + 1] / cs[j] for j in range(len(cs) - 1) if rng_[j + 1] == rng_[j] + 1]
        print(f"      a_M = {aM}, a_S = {list(rng_)} (n {min(ns)}-{max(ns)}): ratios " + "  ".join(f"{abs(r):.2f}@{np.degrees(np.angle(r)):4.0f}" for r in rat))
