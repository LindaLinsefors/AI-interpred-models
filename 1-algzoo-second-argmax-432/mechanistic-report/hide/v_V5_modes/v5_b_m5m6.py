"""Verifier V5, part B: claims M5 and M6 (per-cell fits of the clock and level coordinates).

Per cell (a1, a2), states with S_t > 0.05, pooled over all t:
    z_clock / w4  ~  alpha (M - S) + beta S      (complex least squares, no intercept)
    level  / w4  ~  alpha (M - S) + beta S      (real least squares, no intercept)
Seeds 5010-5013, 4 x 200k sequences.   Run:  python -B v5_b_m5m6.py > v5_b_m5m6.txt
"""
from v5_lib import *

MINN = 2000
acc = {}
def add(key, arr):
    acc.setdefault(key, []).append(arr)

for seed in SEEDS:
    x, H, Z, st, m = batch(seed)
    tt = np.broadcast_to(np.arange(T)[None, :], st["M"].shape)
    ok = st["S"] > 0.05
    for k, a in (("M", st["M"]), ("S", st["S"]), ("a1", st["a1"]), ("a2", st["a2"]), ("t", tt),
                 ("z", m["clock"] / W4), ("lev", m["level"] / W4)):
        add(k, a[ok])
D = {k: np.concatenate(v) for k, v in acc.items()}
M, S, a1, a2, tt, z, lev = (D[k] for k in ("M", "S", "a1", "a2", "t", "z", "lev"))
g = M - S
X = np.stack([g, S], 1)
print(f"states with S > 0.05: {len(M)} of {len(SEEDS) * NB * T}  (seeds {SEEDS}, {NB} sequences each)")

print("=" * 110)
print("M5  clock: per cell (a1, a2): alpha (|.|@deg), beta, uncentred R2, centred R2, n      [cells with n >=", MINN, "]")
rows = {}
for A1 in range(0, 10):
    for A2 in range(0, 10):
        sel = (a1 == A1) & (a2 == A2)
        n = int(sel.sum())
        if A1 == A2 or n < MINN:
            continue
        c, r2u, r2c = lsq(X[sel], z[sel])
        rows[(A1, A2)] = (c[0], c[1], r2u, r2c, n)
        print(f"   a1={A1} a2={A2}  alpha={pol(c[0])}  beta={pol(c[1])}  R2u={r2u:.4f}  R2c={r2c:.4f}  n={n}")
print("\n   summary per a1 (over a2 cells):  |alpha| range, phase range, |beta| range, min R2u, min R2c;  prediction (a1-1)*42.75 and |lam|^(a1-1)")
claim_mod = {1: "1.0", 2: "1.0", 3: "1.0", 4: "0.95", 5: "0.90", 6: "0.87", 7: "0.82", 8: "0.78", 9: "0.7-0.9"}
claim_ph = {1: 0, 2: 37, 3: 80, 4: 123, 5: 166, 6: 209, 7: 252, 8: 294, 9: 338}
for A1 in range(0, 10):
    cells = [v for (p, q), v in rows.items() if p == A1]
    if not cells:
        continue
    al = np.array([v[0] for v in cells]); be = np.array([v[1] for v in cells])
    ph = np.degrees(np.angle(al * np.exp(-1j * np.angle(al.mean())))) + np.degrees(np.angle(al.mean()))
    ph_mean = np.degrees(np.angle(al.mean())) % 360
    print(f"   a1={A1}: |alpha| {np.abs(al).min():.3f}..{np.abs(al).max():.3f} (mean {np.abs(al).mean():.3f}; claim {claim_mod.get(A1)})"
          f"  phase {ph.min() % 360:6.1f}..{ph.max() % 360:6.1f} (circ mean {ph_mean:6.1f}; claim {claim_ph.get(A1)}; (a1-1)*omega = {((A1 - 1) * np.degrees(OMEGA)) % 360:6.1f})"
          f"  |beta| {np.abs(be).min():.3f}..{np.abs(be).max():.3f}  minR2u {min(v[2] for v in cells):.4f}  minR2c {min(v[3] for v in cells):.4f}"
          f"  |lam|^(a1-1) = {abs(LAM['clock']) ** (A1 - 1):.3f}")
all_b = np.array([abs(v[1]) for (p, q), v in rows.items() if p >= 1])
all_r = np.array([v[2] for (p, q), v in rows.items() if p >= 1])
print(f"\n   over all cells with a1 >= 1: |beta| in [{all_b.min():.3f}, {all_b.max():.3f}]   min uncentred R2 = {all_r.min():.4f}"
      f"  (cell {min(((v[2], k) for k, v in rows.items() if k[0] >= 1))[1]})   number of cells = {len(all_b)}")

print("\n   pooled over a2, per a1:  (i) alpha(M-S)+beta S   (ii) alpha(M-S) only   (iii) alpha M only    [uncentred R2 | centred R2]")
for A1 in range(0, 10):
    sel = a1 == A1
    if sel.sum() < MINN:
        continue
    c, r2u, r2c = lsq(X[sel], z[sel])
    c1, r1u, r1c = lsq(g[sel, None], z[sel])
    c2, r2u_, r2c_ = lsq(M[sel, None], z[sel])
    ratio = c[1] / c[0]
    print(f"   a1={A1}: alpha={pol(c[0])} beta={pol(c[1])} beta/alpha={pol(ratio)} -> z ~ alpha (M - c S), c = 1 - beta/alpha = {1 - ratio.real:.3f}{-ratio.imag:+.3f}i"
          f"  R2u={r2u:.4f} R2c={r2c:.4f} | gap only: alpha={pol(c1[0])} R2u={r1u:.4f} R2c={r1c:.4f} | M only: alpha={pol(c2[0])} R2u={r2u_:.4f} R2c={r2c_:.4f}  n={int(sel.sum())}")

print("\n   dependence on t inside a cell (alpha per t), for a few cells:")
for cell in ((1, 3), (2, 4), (3, 1), (4, 6), (5, 2)):
    out = []
    for t0 in range(max(cell), 10):
        sel = (a1 == cell[0]) & (a2 == cell[1]) & (tt == t0)
        if sel.sum() >= MINN:
            c, r2u, _ = lsq(X[sel], z[sel])
            out.append(f"t={t0}: {pol(c[0])} R2u={r2u:.3f}")
    print(f"   cell {cell}: " + " | ".join(out))

print("=" * 110)
print("M6  level: per cell (a1, a2): alpha, beta (real), uncentred R2, centred R2")
rowsL = {}
for A1 in range(0, 10):
    for A2 in range(0, 10):
        sel = (a1 == A1) & (a2 == A2)
        n = int(sel.sum())
        if A1 == A2 or n < MINN:
            continue
        c, r2u, r2c = lsq(X[sel], lev[sel])
        rowsL[(A1, A2)] = (c[0], c[1], r2u, r2c, n)
        print(f"   a1={A1} a2={A2}  alpha={c[0]:.3f}  beta={c[1]:.3f}  R2u={r2u:.4f}  R2c={r2c:.4f}  n={n}")
for lo in (0, 1):
    al = np.array([v[0] for (p, q), v in rowsL.items() if p >= lo]); be = np.array([v[1] for (p, q), v in rowsL.items() if p >= lo])
    r = np.array([v[2] for (p, q), v in rowsL.items() if p >= lo]); rc = np.array([v[3] for (p, q), v in rowsL.items() if p >= lo])
    print(f"   cells with a1 >= {lo}: alpha in [{al.min():.3f}, {al.max():.3f}]  beta in [{be.min():.3f}, {be.max():.3f}]  min R2u = {r.min():.4f}  min centred R2 = {rc.min():.4f}")
print("\n   pooled over a2 per a1:  (i) alpha(M-S)+beta S  (ii) alpha M only   [uncentred | centred R2];  and mean of level/(w4 M)")
for A1 in range(0, 10):
    sel = a1 == A1
    c, r2u, r2c = lsq(X[sel], lev[sel])
    c2, r2u_, r2c_ = lsq(M[sel, None], lev[sel])
    print(f"   a1={A1}: alpha={c[0]:.3f} beta={c[1]:.3f} R2u={r2u:.4f} R2c={r2c:.4f} | M only: {c2[0]:.3f} R2u={r2u_:.4f} R2c={r2c_:.4f} | median level/(w4 M) = {np.median(lev[sel] / M[sel]):.3f}")
