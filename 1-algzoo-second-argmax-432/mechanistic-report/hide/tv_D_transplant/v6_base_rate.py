"""v6: ATTACK 10.  The investigator reads "with a_S = 2 the rest piece alone has the runner-up on top in 34-99%" as
"a young runner-up is still held by the rest".  Alternative: the rest piece favours the age-2 entry whatever input
sits there (a slot bias), so the hit rate at a_S = 2 is a base rate and not stored information.

Quiet final states.  For each piece, among entries of age >= 2:
  P(top entry has age k)  for all states, for states with a_S = k, and for states with a_S != k and a_M != k.
Then the failing configuration (donor a_S = 2, a_M = 9) of the S_pos transplant: the three contrasts that decide it.
"""
from v_common import *

x, st = quiet(500_000, 9, 4101)
h = state(x, 9)
n = len(x)
ar = np.arange(n)
s_all = h @ U.T
s_tape = h[:, O] @ P_T.T @ U[:, O].T
s_rest = s_all - s_tape
print(f"{n} quiet final states.  P(top entry among ages >= 2 has age k)")
for lab, s in (("rest piece", s_rest), ("tape piece", s_tape), ("full scores", s_all)):
    age = 9 - s[:, :8].argmax(1)
    print(f"   {lab}\n      age k:                              " + "".join(f"{k:7d}" for k in range(2, 10)))
    print("      all states                          " + "".join(f"{np.mean(age == k):7.3f}" for k in range(2, 10)))
    print("      states with a_S = k                 " + "".join(f"{np.mean(age[st['aS'] == k] == k):7.3f}" for k in range(2, 10)))
    print("      states with a_S != k and a_M != k   " + "".join(f"{np.mean(age[(st['aS'] != k) & (st['aM'] != k)] == k):7.3f}" for k in range(2, 10)))
    print("      states with a_T = k (third there)   " + "".join(f"{np.mean(age[st['aT'] == k] == k):7.3f}" for k in range(2, 10)))

# the failing donor configuration: runner-up of age 2, maximum of age 9
rng = np.random.default_rng(4102)
j, ok = pick_low_slot(x, st, 9, rng)
xB = swapped(x, st["pS"], j)
sB = stats(xB, 9)
ok &= sB["quiet"]
for aM in (9, 8, 5):
    m = ok & (sB["aS"] == 2) & (sB["aM"] == aM)
    hA, hB = h[m], state(xB[m], 9)
    M = st["M"][m]
    k = m.sum()
    i = np.arange(k)
    con = lambda hh, P: (((hh[:, O] @ P.T @ U[:, O].T) if P is not None else hh @ U.T)[i, sB["pS"][m]]
                         - ((hh[:, O] @ P.T @ U[:, O].T) if P is not None else hh @ U.T)[i, st["pM"][m]]) / M
    tB, fB, fA = con(hB, P_T), con(hB, None), con(hA, None)
    tA = con(hA, P_T)
    rB, rA = fB - tB, fA - tA
    med = lambda v: f"{np.median(v):6.1f}"
    print(f"\ndonor B with a_S = 2, a_M = {aM} (n = {k}): entry of B's runner-up slot (age 2) minus entry of the maximum, in M (medians)")
    print(f"   B intact: full {med(fB)} = tape piece {med(tB)} + rest piece {med(rB)}       A's rest piece (far-below input in that slot): {med(rA)}"
          f"       hybrid rest A + tape B: {med(rA + tB)}   (positive in {np.mean(rA + tB > 0):.3f}; B intact positive in {np.mean(fB > 0):.3f})")
    print(f"   so the rest piece's contrast changes by {med(rB - rA)} M when the runner-up enters the age-2 slot, the tape piece's by {med(tB - tA)} M")
