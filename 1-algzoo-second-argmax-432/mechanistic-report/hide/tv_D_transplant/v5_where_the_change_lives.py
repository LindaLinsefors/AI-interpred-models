"""v5: ATTACK 9.  Is "the answer follows the tape" trivial because the rest hardly changes when the runner-up moves?

S_pos pairs at t = 9 (B = A with the runner-up swapped with a far-below old input at slot j).  The score change
ds = s(B) - s(A) splits exactly into a tape piece and a rest piece.  Reported, in units of M:
  the change of the contrast between the two swapped entries, c = ds[pS_B] - ds[pS_A], for each piece,
  the share of |ds|^2 over entries of age >= 2 that is in each piece, and the same for single modes of the rest.
"""
from v_common import *

x, st = quiet(400_000, 9, 5101)
rng = np.random.default_rng(5102)
j, ok = pick_low_slot(x, st, 9, rng)
xB = swapped(x, st["pS"], j)
ok &= stats(xB, 9)["quiet"]
a, b = sub(st, ok), sub(stats(xB, 9), ok)
dh = state(xB[ok], 9) - state(x[ok], 9)
n = ok.sum()
ar = np.arange(n)
Uo = U[:, O]
pieces = {"full": dh @ U.T, "tape": dh[:, O] @ P_T.T @ Uo.T}
pieces["rest"] = pieces["full"] - pieces["tape"]
for m in ("clock", "level", "runner", "trans", "echo"):
    pieces["  rest: " + m] = dh[:, O] @ proj([m]).T @ Uo.T
pieces["  rest: pulse units"] = dh[:, [4, 5, 9]] @ U[:, [4, 5, 9]].T
tot = (pieces["full"][:, :8] ** 2).sum()
print(f"S_pos pairs at t = 9, n = {n}.  c = change of (entry of B's runner-up slot - entry of A's runner-up slot), in M")
print("   piece                 c: 10% / median / 90%          sum of squared change over ages >= 2, relative to the full change")
for lab, ds in pieces.items():
    c = (ds[ar, b["pS"]] - ds[ar, a["pS"]]) / a["M"]
    print(f"   {lab:20s}  {np.quantile(c, .1):6.1f} / {np.median(c):6.1f} / {np.quantile(c, .9):6.1f}              {(ds[:, :8] ** 2).sum() / tot:.3f}")
print("   by A's a_S (the slot the runner-up leaves) and B's a_S (the slot it enters): median c of the rest piece / of the tape piece")
for s in range(2, 10):
    cr = (pieces["rest"][ar, b["pS"]] - pieces["rest"][ar, a["pS"]]) / a["M"]
    ct = (pieces["tape"][ar, b["pS"]] - pieces["tape"][ar, a["pS"]]) / a["M"]
    print(f"      a_S = {s}:  leaves: {np.median(cr[a['aS'] == s]):6.1f} / {np.median(ct[a['aS'] == s]):6.1f}     enters: {np.median(cr[b['aS'] == s]):6.1f} / {np.median(ct[b['aS'] == s]):6.1f}")
