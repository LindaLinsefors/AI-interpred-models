"""C5d. The wave units are not always on: their gates follow the clock phase (age of the max).
(1) natural on-share of each ordinary unit by max-age;  (2) after a phase shift by k steps the gates of the next step follow age a + k.
200k N(0,1) sequences, seed 1503."""
from cl import *
NS, SEED = 200_000, 1503
x, H, Z, st = gen(NS, SEED)
M, S, a1 = st["M1"], pos(st["M2"]), st["a1"]
G = Z > 0
ok = (M > 0)
ok[:, 0] = False
print("=== (1) [MEASURED] share of states in which each ordinary unit is ON, by age a of the max (states with M > 0, t >= 1) ===")
print("   a  |      n  | " + "  ".join(f"u{u:<3d}" for u in WAVE) + " | " + "  ".join(f"c{u:<3d}" for u in COMP) + " | all 13 on")
nat = np.zeros((10, 9))
for a in range(10):
    s = ok & (a1 == a)
    g = G[s]
    nat[a] = g[:, WAVE].mean(0)
    print(f"   {a}  | {int(s.sum()):7d} | " + "  ".join(f"{v:.2f}" for v in g[:, WAVE].mean(0)) + " | " + "  ".join(f"{v:.2f}" for v in g[:, COMP].mean(0)) + f" | {g[:, ORD].all(1).mean():.3f}")
s = ok & (a1 >= 1)
print(f"   all a >= 1: all 13 ordinary units on in {G[s][:, ORD].all(1).mean():.3f} of states; all 9 wave units on in {G[s][:, WAVE].all(1).mean():.3f}; all 4 comparators on in {G[s][:, COMP].all(1).mean():.3f}")
# restricted to quiet steps (input below S+ of the previous step, so no event at this step)
q = ok & (a1 >= 1)
q[:, 1:] &= x[:, 1:] < S[:, :-1]
print(f"   quiet steps only (x_t < S+_(t-1), a >= 1): all 13 on in {G[q][:, ORD].all(1).mean():.3f}; all wave on {G[q][:, WAVE].all(1).mean():.3f}; all comparators on {G[q][:, COMP].all(1).mean():.3f}")
print("   quiet steps, on-share of the wave units by age:")
for a in range(1, 10):
    s = q & (a1 == a)
    print(f"   {a}  | {int(s.sum()):7d} | " + "  ".join(f"{v:.2f}" for v in G[s][:, WAVE].mean(0)))

print("\n=== (2) [INTERVENTION] phase shift by k steps at step t (max-age a): wave-unit gates at step t+1 for quiet inputs x_{t+1} < S+_t ===")
print("   D = mean over the 9 wave units of |on-share after the edit - natural on-share at age a+1+k| (target) and vs natural at age a+1 (unshifted)")
def edit(h, F):
    h2 = h.copy(); zc = h[:, ORD] @ LC
    h2[:, ORD] += 2 * np.real(np.multiply.outer((F - 1) * zc, RC)); return h2
natq = np.array([G[q & (a1 == a)][:, WAVE].mean(0) if (q & (a1 == a)).any() else np.zeros(9) for a in range(10)])
Dt, Du = [], []
for k in (-2, -1, 1, 2):
    for a in range(1, 8):
        if not (1 <= a + 1 + k <= 9):
            continue
        chunks = []
        for t in range(1, 9):
            idx = np.nonzero((M[:, t] > 0) & (a1[:, t] == a) & (x[:, t + 1] < S[:, t]))[0]
            h2 = edit(H[idx, t], np.exp(1j * k * OMEGA))
            pre = x[idx, t + 1, None] * w + h2 @ W.T
            chunks.append(pre[:, WAVE] > 0)
        g = np.concatenate(chunks).mean(0)
        Dt.append(np.mean(abs(g - natq[a + 1 + k]))); Du.append(np.mean(abs(g - natq[a + 1])))
print(f"   over {len(Dt)} (a, k) cells: mean D(target) = {np.mean(Dt):.3f}, mean D(unshifted) = {np.mean(Du):.3f}; target closer in {np.mean(np.array(Dt) < np.array(Du)):.2f} of cells")

# ---------------------------------------------------------------- per-unit view and mode composition
print("   per unit: on-share at t+1 after the edit, k = +1 and k = -1, by natural age a+1 of the next step; 'nat' rows are the natural quiet-step shares")
show = [0, 3, 10, 11, 12, 15]
si = [WAVE.index(u) for u in show]
print("   natural (quiet) by age 1..9:")
for u, j in zip(show, si):
    print(f"     u{u:<2d}: " + "  ".join(f"{natq[a, j]:.2f}" for a in range(1, 10)))
rot_all = [n for n in NAMES if IS_COMPLEX[n]] + ["alt"]
def shift(h, k, which):
    h2 = h.copy(); d = 0
    for n in which:
        c = h[:, ORD] @ LEFT[n]
        if IS_COMPLEX[n]:
            d = d + 2 * np.real(np.multiply.outer((np.exp(1j * k * np.angle(LAM[n])) - 1) * c, RIGHT[n]))
        else:
            d = d + np.real(np.multiply.outer(((-1.0) ** k - 1) * np.real(c), RIGHT[n]))
    h2[:, ORD] += d; return h2
for label, which in (("clock only", ["clock"]), ("all rotating modes + alt", rot_all)):
    for k in (1, -1):
        print(f"   {label}, k = {k:+d}: on-share at t+1 by the natural age a+1 = 2..8 of that step (compare with the natural row shifted by k)")
        tab = np.full((len(show), 9), np.nan)
        for a in range(1, 8):
            chunks = []
            for t in range(1, 9):
                idx = np.nonzero((M[:, t] > 0) & (a1[:, t] == a) & (x[:, t + 1] < S[:, t]))[0]
                h2 = shift(H[idx, t], k, which)
                chunks.append((x[idx, t + 1, None] * w + h2 @ W.T)[:, show] > 0)
            tab[:, a + 1] = np.concatenate(chunks).mean(0)
        for r, u in enumerate(show):
            print(f"     u{u:<2d}: " + "  ".join(f"{tab[r, a]:.2f}" for a in range(2, 9)))

print("\n=== [EXACT] free response of the ordinary units to one unit of unit-4 kick: A^(a-1) W[ORD,4] by age a (no inputs, no clipping, pulses 5, 9 ignored) ===")
resp = np.zeros((10, 13)); v = W[ORD, 4].copy()
for a in range(1, 10):
    resp[a] = v; v = A @ v
print("   age   " + "  ".join(f"u{u:<5d}" for u in ORD))
for a in range(1, 10):
    print(f"   {a}     " + "  ".join(f"{resp[a, i]:+6.2f}" for i in range(13)))
print("   entries of the free response that are negative (the unit would be clipped): " + ", ".join(f"u{ORD[i]}@a={a}" for a in range(1, 10) for i in range(13) if resp[a, i] < 0))
print("   mode composition of the response of units 0, 10, 11 (level | runner | clock | h2 | alt | h3 | short | trans | echo):")
for u in (0, 10, 11):
    i = ORD.index(u)
    for a in range(1, 10):
        parts = [(2 if IS_COMPLEX[n] else 1) * np.real(LAM[n] ** (a - 1) * RIGHT[n][i]) for n in NAMES]
        print(f"     u{u} a={a}: " + "  ".join(f"{p:+.2f}" for p in parts) + f"   sum {sum(parts):+.2f}")
