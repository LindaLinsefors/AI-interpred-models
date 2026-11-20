"""S5: at t = 9, sequences with all four comparators on, S > 0.2, p1 < 9, p2 < 9.
D = |p1 - p2|, g = (M - S)/M. Medians (in units of M) of V[p2]-V[p1], m[p2]-m[p1], s[p2]-s[p1], and of the
formula 60 (M - 0.87 S)(1 - cos(omega D))/M; P(s[p2] > s[p1]) per cell.

Usage: python -B s5.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 20
GAPS = [0, 0.05, 0.15, 0.3, 0.5, 0.8, 1.0]
LEAD = {(3, 1): (20.3, -12.8, 7.8, 21.0), (3, 2): (32.8, -24.1, 8.8, 31.5), (3, 3): (49.0, -39.3, 10.2, 45.7),
        (3, 4): (68.7, -58.7, 10.5, 63.4), (4, 3): (61.7, -47.7, 13.8, 56.2), (1, 3): (11.4, -6.2, 4.8, 7.5)}

cols = []
n_tot = 0
for x, H, M, S, p1, p2 in batches(nb, seed0):
    h = H[:, 9]
    sel = (h[:, COMP] > 0).all(1) & (S[:, 9] > 0.2) & (p1[:, 9] < 9) & (p2[:, 9] < 9)
    n_tot += len(x)
    s, V, m = scores(h[sel])
    r = np.arange(sel.sum()); P1, P2, Mt, St = p1[sel, 9], p2[sel, 9], M[sel, 9], S[sel, 9]
    cols.append(np.stack([P1, P2, Mt, St, V[r, P2] - V[r, P1], m[r, P2] - m[r, P1], s[r, P2] - s[r, P1]], 1))
d = np.concatenate(cols)
P1, P2, Mt, St, dV, dm, ds = d.T
P1 = P1.astype(int); P2 = P2.astype(int)
D = np.abs(P1 - P2); g = (Mt - St) / Mt
gb = np.digitize(g, GAPS, right=True) - 1
form = 60 * (Mt - 0.87 * St) * (1 - np.cos(OMEGA * D)) / Mt
print(f"S5  seeds {seed0}..{seed0 + nb - 1}; {n_tot} sequences, selected {len(d)} ({len(d) / n_tot:.4f})")
print("\ncells quoted by the lead: (D, gap bin): median dV/M, dm/M, ds/M, formula   [lead]   n")
for (Dv, k), lead in LEAD.items():
    c = (D == Dv) & (gb == k)
    med = [np.median((v / Mt)[c]) for v in (dV, dm, ds)] + [np.median(form[c])]
    print(f"D={Dv} gap ({GAPS[k]}, {GAPS[k + 1]}]:  " + "  ".join(f"{v:7.2f}" for v in med) + "   lead " + str(lead) + f"   n = {c.sum()}")

print("\nall cells: median dV/M | formula median | ratio dV/formula    rows D = 1..8, columns gap bins", GAPS)
for Dv in range(1, 9):
    row = []
    for k in range(len(GAPS) - 1):
        c = (D == Dv) & (gb == k)
        row.append(f"{np.median((dV / Mt)[c]):6.1f}|{np.median(form[c]):5.1f}|{np.median((dV / Mt)[c]) / np.median(form[c]):5.2f}" if c.sum() > 50 else "      -      ")
    print(Dv, "   ".join(row))
print("\nall cells: median dm/M , median ds/M")
for Dv in range(1, 9):
    print(Dv, "   ".join(f"{np.median((dm / Mt)[(D == Dv) & (gb == k)]):7.1f},{np.median((ds / Mt)[(D == Dv) & (gb == k)]):6.1f}"
                         for k in range(len(GAPS) - 1)))
print("\nP(s[p2] > s[p1]) per cell  (n)")
for Dv in range(1, 9):
    print(Dv, "   ".join(f"{(ds > 0)[(D == Dv) & (gb == k)].mean():.4f} ({((D == Dv) & (gb == k)).sum():6d})" for k in range(len(GAPS) - 1)))
print("\nP(m[p2] < m[p1]) per cell (m ranks the max above the runner-up)")
for Dv in range(1, 9):
    print(Dv, "   ".join(f"{(dm < 0)[(D == Dv) & (gb == k)].mean():.4f}" for k in range(len(GAPS) - 1)))
print("\nP(V[p2] > V[p1]) per cell (the comparators' vote favours the runner-up over the max)")
for Dv in range(1, 9):
    print(Dv, "   ".join(f"{(dV > 0)[(D == Dv) & (gb == k)].mean():.4f}" for k in range(len(GAPS) - 1)))

mn = min((ds > 0)[(D == Dv) & (gb == k)].mean() for Dv in range(1, 7) for k in range(1, len(GAPS) - 1))
print(f"\nmin over D = 1..6 and gap bins > 0.05 of P(s[p2] > s[p1]): {mn:.4f}")
small = [(ds > 0)[(D == Dv) & (gb == 0)].mean() for Dv in range(1, 7)]
print("gap <= 0.05, D = 1..6:", " ".join(f"{v:.4f}" for v in small))

# signed separation: does V[p2]-V[p1] depend on whether the runner-up is older or younger than the max?
print("\nmedian dV/M by signed separation p2 - p1 (columns gap bins): negative = runner-up older than the max")
for sd in list(range(-8, 0)) + list(range(1, 9)):
    print(f"{sd:+d}", "   ".join(f"{np.median((dV / Mt)[((P2 - P1) == sd) & (gb == k)]):6.1f}" if (((P2 - P1) == sd) & (gb == k)).sum() > 50 else "   -  "
                             for k in range(len(GAPS) - 1)),
          "   | dm/M:", "  ".join(f"{np.median((dm / Mt)[((P2 - P1) == sd) & (gb == k)]):6.1f}" if (((P2 - P1) == sd) & (gb == k)).sum() > 50 else "   -  "
                                  for k in range(len(GAPS) - 1)))

# Is dV/M linear in g with the claimed intercept?  dV/M = K (1 - cos wD) (1 - c + c g)  ->  fit K_D and c per D
print("\nper D: least squares dV/M ~ k0 + k1 g ; implied  K = (k0 + k1)/(1 - cos wD),  c = k1/(k0 + k1)   (lead: K = 60, c = 0.87)")
for Dv in range(1, 9):
    c = D == Dv
    k1, k0 = np.polyfit(g[c], (dV / Mt)[c], 1)
    print(f"D={Dv}: k0 = {k0:6.2f}  k1 = {k1:6.2f}   K = {(k0 + k1) / (1 - np.cos(OMEGA * Dv)):6.1f}   c = {k1 / (k0 + k1):.3f}   n = {c.sum()}")
