"""s04 (task F3 i). H_dev from the weights only.

State  v(a) = L * RIGHT[level] + 2 Re(r e^{i phi(a)} RIGHT[clock])   (a = age of the max at the step of zeroing)
-> zero the four comparators -> propagate j = 0..6 steps with A -> read with U.
Old max's position: p* = 9 - j - a.
Everything is linear, so the clock part (per unit r) and the level part (per unit L) are reported separately.
phi(a), rho(a) are my own re-measurement (s03): they agree with the lead's F6 within 1 degree.
"""
from fcommon import *

d = np.load(HIDE / "r1_F_record_event" / "s03_clock.npz")
phi, rho = d["phi"], d["rho"]
E = np.eye(13); E[CI, CI] = 0                       # zero the comparators
Rl, Rc = RIGHT["level"].real, RIGHT["clock"]
Apow = [np.linalg.matrix_power(A, j) for j in range(9)]


def clock_state(a):
    return 2 * np.real(np.exp(1j * np.radians(phi[a])) * Rc)


print("=== A. clock part, per unit r:  logits = U A^j E 2Re(e^{i phi(a)} R_clock) ===")
print("columns: a, j, p* = 9-j-a | logit at p* | argmax position (value) | offset argmax - p* | 2nd best (value) |"
      " bump = logit[p*] - mean of the other 9 | same WITHOUT zeroing (hidden clock): logit at p*, max |logit|")
tab = {}
for a in range(1, 10):
    for j in range(0, 7):
        ps = 9 - j - a
        if ps < 0:
            continue
        lg = Uo @ Apow[j] @ E @ clock_state(a)
        lg0 = Uo @ Apow[j] @ clock_state(a)
        o = np.argsort(-lg)
        tab[(a, j)] = (lg[ps], o[0], lg[o[0]], lg[ps] - np.delete(lg, ps).mean(), lg)
        print(f"  a={a} j={j} p*={ps} | {lg[ps]:6.2f} | {o[0]} ({lg[o[0]]:5.2f}) | {o[0] - ps:+d} | {o[1]} ({lg[o[1]]:5.2f}) |"
              f" {lg[ps] - np.delete(lg, ps).mean():6.2f} | {lg0[ps]:6.2f}, {np.abs(lg0).max():5.2f}")

print("\nsummary over the (a, j) cells with a >= 2:")
cells = [(a, j) for (a, j) in tab if a >= 2]
off = np.array([tab[c][1] - (9 - c[1] - c[0]) for c in cells])
print(f"  cells: {len(cells)};  peak exactly at p*: {(off == 0).sum()};  at p*+1: {(off == 1).sum()};  at p*-1: {(off == -1).sum()};  elsewhere: {(np.abs(off) > 1).sum()}")
print("  logit at p* per unit r:  min %.2f  median %.2f  max %.2f" % tuple(np.percentile([tab[c][0] for c in cells], [0, 50, 100])))
print("  bump (p* minus mean of others) per unit r:  min %.2f  median %.2f  max %.2f" % tuple(np.percentile([tab[c][3] for c in cells], [0, 50, 100])))
print("  by j (a >= 2): mean logit at p*, mean bump, fraction of cells with the peak at p*")
for j in range(7):
    cj = [c for c in cells if c[1] == j]
    if cj:
        print(f"     j={j}: n={len(cj)}  logit[p*] = {np.mean([tab[c][0] for c in cj]):5.2f}   bump = {np.mean([tab[c][3] for c in cj]):5.2f}"
              f"   peak at p*: {np.mean([tab[c][1] == 9 - c[1] - c[0] for c in cj]):.2f}")
print("  full logit vectors for a = 4 (per unit r), rows j = 0..5 (p* = 5 - j):")
for j in range(0, 6):
    print(f"     j={j} p*={5-j}: " + " ".join(f"{v:6.2f}" for v in tab[(4, j)][4]))

print("\n=== B. level part, per unit level coordinate:  U A^j E R_level  (without zeroing: U A^j R_level = lam^j U R_level) ===")
for j in range(0, 7):
    lg = Uo @ Apow[j] @ E @ Rl
    print(f"  j={j}: " + " ".join(f"{v:6.2f}" for v in lg) + f"   | no zeroing: " + " ".join(f"{v:5.2f}" for v in Uo @ Apow[j] @ Rl))

print("\n=== C. both parts with realistic sizes: M0 = 1, S0 = 0.5 or 0.9: level = 0.93*10.157*M0, r = 10.157*rho(a)*(M0-S0) ===")
for S0 in (0.5, 0.9, 0.0):
    hit = []
    for a in range(2, 10):
        for j in range(0, 7):
            ps = 9 - j - a
            if ps < 0:
                continue
            lg = Uo @ Apow[j] @ E @ (0.93 * w[4] * Rl + w[4] * rho[a] * (1 - S0) * clock_state(a))
            hit.append((a, j, ps, int(lg.argmax()), lg[ps] - np.delete(lg, ps).max()))
    hit = np.array(hit)
    print(f"  S0 = {S0}: argmax = p* in {np.mean(hit[:, 2] == hit[:, 3]):.2f} of {len(hit)} cells;"
          f" margin of p* over best other: median {np.median(hit[:, 4]):.1f}, min {hit[:, 4].min():.1f};"
          f" misses (a, j, p*, argmax): {[tuple(int(v) for v in h[:4]) for h in hit if h[2] != h[3]][:12]}")

print("\n=== D. the same question for the complete trace of a record: unit-4 kick K4 = W[ORD,4] (coordinate 1 in every mode) ===")
print("free flow to age a (state A^(a-1) K4), then zero comparators, j more steps, read. Compared with no zeroing.")
print("columns: a, j, p* | logit[p*] no zeroing | logit[p*] zeroed | change at p* | argmax of the CHANGE (value) | offset")
offs = []
for a in range(1, 10):
    for j in range(0, 7):
        ps = 9 - j - a
        if ps < 0:
            continue
        v = Apow[a - 1] @ W[ORD, 4]
        l0, l1 = Uo @ Apow[j] @ v, Uo @ Apow[j] @ E @ v
        ch = l1 - l0
        offs.append((a, j, int(ch.argmax()) - ps, ch[ps]))
        if j in (0, 2, 5):
            print(f"  a={a} j={j} p*={ps} | {l0[ps]:6.2f} | {l1[ps]:6.2f} | {ch[ps]:6.2f} | {int(ch.argmax())} ({ch.max():5.2f}) | {int(ch.argmax()) - ps:+d}")
offs = np.array(offs)
m2 = offs[:, 0] >= 2
print(f"  a>=2: argmax of the change is at p* in {np.mean(offs[m2, 2] == 0):.2f} of {m2.sum()} cells; change at p*: median {np.median(offs[m2, 3]):.2f},"
      f" range [{offs[m2, 3].min():.2f}, {offs[m2, 3].max():.2f}]")

print("\n=== E. which modes does the comparator-zeroing make visible?  per mode k (unit coordinate, phase chosen = clock-like "
      "'age a' phase a*angle_k): norm over positions of U A^j (E - I) R_k for j = 0, 2 ===")
for n in NAMES:
    Rk = RIGHT[n]
    for j in (0, 2):
        if IS_COMPLEX[n]:
            c = Uo @ Apow[j] @ (E - np.eye(13)) @ Rk           # complex readout vector; logit change = 2 Re(z c)
            print(f"  {n:6s} j={j}: 2|c| by position: " + " ".join(f"{2*abs(v):5.2f}" for v in c) + f"   (before zeroing 2|U R|: " +
                  " ".join(f"{2*abs(v):4.1f}" for v in Uo @ Apow[j] @ Rk) + ")")
        else:
            c = Uo @ Apow[j] @ (E - np.eye(13)) @ Rk.real
            print(f"  {n:6s} j={j}:    c by position: " + " ".join(f"{v:5.2f}" for v in c) + f"   (before zeroing U R: " +
                  " ".join(f"{v:4.1f}" for v in Uo @ Apow[j] @ Rk.real) + ")")
