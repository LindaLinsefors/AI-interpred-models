"""S2 / S3: at t = 9, sequences with all four comparators on and S_9 > 0.1; per age a1 = 2..9 of the max,
least squares  V[i] ~ A_i (M - S) + B_i S + C_i x_9  (no intercept), i = 0..9.

Usage: python -B s2_s3.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 10
LEAD_A = {0: -62, 1: -48, -1: -42, 2: -6, -2: 7, 3: 40, -3: 59, 4: 60, -4: 82, 5: 47, -5: 61, 6: 14, -6: 14,
          7: -25, -7: -33}


def collect(seed0, nb):
    Vs, Xs, a1s, a2s, hCs, n_tot, n_on, n_s = [], [], [], [], [], 0, 0, 0
    for x, H, M, S, p1, p2 in batches(nb, seed0):
        h = H[:, 9]
        on = (h[:, COMP] > 0).all(1)
        sel = on & (S[:, 9] > 0.1)
        n_tot += len(x); n_on += on.sum(); n_s += (S[:, 9] > 0.1).sum()
        _, V, _ = scores(h[sel])
        Vs.append(V)
        Xs.append(np.stack([(M[:, 9] - S[:, 9])[sel], S[sel, 9], x[sel, 9]], 1))
        a1s.append(9 - p1[sel, 9]); a2s.append(9 - p2[sel, 9]); hCs.append(h[sel][:, COMP])
    print(f"sequences {n_tot}; all four comparators on at t=9: {n_on / n_tot:.4f}; S_9 > 0.1: {n_s / n_tot:.4f}; "
          f"both: {sum(map(len, Vs)) / n_tot:.4f}")
    return np.concatenate(Vs), np.concatenate(Xs), np.concatenate(a1s), np.concatenate(a2s), np.concatenate(hCs)


def fit(X, Y):
    """No-intercept least squares. Returns coef (k, 10), SSres (10,), sum y^2 (10,), centred SStot (10,)."""
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    res = Y - X @ beta
    return beta, (res ** 2).sum(0), (Y ** 2).sum(0), ((Y - Y.mean(0)) ** 2).sum(0)


def by_d(tab, i_max=7):
    """tab[a1][i] -> mean over a1 of tab at i = d + 9 - a1 (i <= i_max), with the number of terms."""
    out = {}
    for d in range(-7, 8):
        v = [tab[a][d + 9 - a] for a in tab if 0 <= d + 9 - a <= i_max]
        out[d] = (np.mean(v), len(v), np.min(v), np.max(v))
    return out


def cosfit(y, pos, om=OMEGA):
    """y ~ alpha cos(om pos) + beta sin(om pos) + c. Returns amplitude, phase (rad), c, centred R^2."""
    G = np.stack([np.cos(om * pos), np.sin(om * pos), np.ones(len(pos))], 1)
    co = np.linalg.lstsq(G, y, rcond=None)[0]
    r = y - G @ co
    return np.hypot(co[0], co[1]), np.arctan2(co[1], co[0]), co[2], 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()


if __name__ == "__main__":
    V, X, a1, a2, hC = collect(seed0, nb)
    print(f"S2/S3  seeds {seed0}..{seed0 + nb - 1}; selected rows {len(V)}")
    A, Bc, C = {}, {}, {}
    print("\nper a1: n; R^2 variants [uncentred pooled i=0..9 | uncentred pooled i=0..7 | centred pooled 0..9 | centred pooled 0..7]"
          "; per-index R^2 range (uncentred; centred)")
    for a in range(2, 10):
        k = a1 == a
        beta, ssr, ss0, ssc = fit(X[k], V[k])
        A[a], Bc[a], C[a] = beta
        r2u, r2c = 1 - ssr / ss0, 1 - ssr / ssc
        print(f"a1={a} n={k.sum():7d}  {1 - ssr.sum() / ss0.sum():.3f} | {1 - ssr[:8].sum() / ss0[:8].sum():.3f} | "
              f"{1 - ssr.sum() / ssc.sum():.3f} | {1 - ssr[:8].sum() / ssc[:8].sum():.3f} ;  "
              f"uncentred per i: {r2u.min():.3f}..{r2u.max():.3f} (i<=7: {r2u[:8].min():.3f}..{r2u[:8].max():.3f});  "
              f"centred per i: {r2c.min():.3f}..{r2c.max():.3f} (i<=7: {r2c[:8].min():.3f}..{r2c[:8].max():.3f})")
        print("      per-index uncentred R^2:", " ".join(f"{v:.3f}" for v in r2u))
        print("      per-index centred   R^2:", " ".join(f"{v:.3f}" for v in r2c))
    for name, tab in (("A_i", A), ("B_i", Bc), ("C_i", C)):
        print(f"\n{name}: rows a1 = 2..9 (max at position 9 - a1), columns index i = 0..9")
        for a in range(2, 10):
            print(f"a1={a} p1={9 - a} ", " ".join(f"{v:7.1f}" for v in tab[a]))
    print("\nA as a function of d = i - (9 - a1), i <= 7, averaged over a1  [mine | lead | n terms | min..max over a1 | -60 cos(42.75 d)]")
    Ad, Bd, Cd = by_d(A), by_d(Bc), by_d(C)
    for d in sorted(Ad, key=lambda d: (abs(d), -d)):
        print(f"d={d:+d}  A = {Ad[d][0]:7.1f} | {LEAD_A[d]:5d} | {Ad[d][1]} | {Ad[d][2]:7.1f}..{Ad[d][3]:7.1f} | "
              f"{-60 * np.cos(OMEGA * d):7.1f}    B = {Bd[d][0]:6.1f} ({Bd[d][2]:6.1f}..{Bd[d][3]:6.1f})   "
              f"C = {Cd[d][0]:6.2f} ({Cd[d][2]:6.2f}..{Cd[d][3]:6.2f})")
    ds = np.array(sorted(Ad)); Am = np.array([Ad[d][0] for d in ds]); Bm = np.array([Bd[d][0] for d in ds])
    for name, y in (("A(d)", Am), ("B(d)", Bm)):
        amp, ph, c, r2 = cosfit(y, ds.astype(float))
        print(f"cosine fit of the averaged {name} over d = -7..7 at 42.75 deg: amplitude {amp:.1f}, minimum at d = "
              f"{(np.degrees(ph) + 180) / np.degrees(OMEGA):+.2f} (mod {360 / np.degrees(OMEGA):.2f}), offset {c:+.1f}, R^2 {r2:.3f}")
    print("\nS3: index 9 per a1:   C_9          A_9         B_9")
    for a in range(2, 10):
        print(f"a1={a}   {C[a][9]:8.2f}   {A[a][9]:8.1f}   {Bc[a][9]:8.1f}")
    print("C_9 range:", min(C[a][9] for a in A), max(C[a][9] for a in A), " A_9 range:", min(A[a][9] for a in A),
          max(A[a][9] for a in A), " B_9 range:", min(Bc[a][9] for a in A), max(Bc[a][9] for a in A))
    print("index 8 per a1 (A_8, B_8, C_8):", " ".join(f"({A[a][8]:.1f}, {Bc[a][8]:.1f}, {C[a][8]:.1f})" for a in A))
    print("\nper a1: cosine fit of A_i and of B_i over i = 0..7 (42.75 deg): amplitude, position of the minimum, offset, R^2")
    for a in range(2, 10):
        pos = np.arange(8.0)
        aa, ph, c, r2 = cosfit(A[a][:8], pos)
        ba, phb, cb, r2b = cosfit(Bc[a][:8], pos)
        per = 360 / np.degrees(OMEGA)
        i0 = ((np.degrees(ph) + 180) / np.degrees(OMEGA)) % per
        i0b = ((np.degrees(phb) + 180) / np.degrees(OMEGA)) % per
        print(f"a1={a} p1={9 - a}:  A: amp {aa:5.1f}  min at i = {i0:5.2f} (p1 = {9 - a})  offset {c:+6.1f}  R^2 {r2:.3f}   |   "
              f"B: amp {ba:5.1f}  min at i = {i0b:5.2f}  offset {cb:+6.1f}  R^2 {r2b:.3f}   | argmin_i A = {A[a][:8].argmin()}, argmin_i B = {Bc[a][:8].argmin()}")
