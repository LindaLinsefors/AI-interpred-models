"""Verifier V5, part F: attacks on "the level mode is the centre of the state's orbit and scales with the running maximum M" (M6).

 F1  level/w4 against M_t, M_(t-1) and their positive parts, on ALL states (no S filter): which one does it follow? negative maxima?
 F2  is the centre of the orbit the level mode?  (a) impulse x = (1, 0, ..., 0), (b) data: mean state per age of the max, averaged over ages
 F3  how much of the level law is unit 2 alone?
Seeds 5040, 5041 (2 x 200k sequences).   Run:  python -B v5_f_level_centre.py > v5_f_level_centre.txt
"""
from v5_lib import *

acc = {}
def add(k, a):
    acc.setdefault(k, []).append(a)
SEEDS_F = [5040, 5041]
for seed in SEEDS_F:
    x, H, Z, st, m = batch(seed)
    Mprev = np.full_like(st["M"], -np.inf); Mprev[:, 1:] = st["M"][:, :-1]
    tt = np.broadcast_to(np.arange(T)[None, :], st["M"].shape)
    for k, a in (("lev", m["level"] / W4), ("M", st["M"]), ("S", st["S"]), ("Mprev", Mprev), ("a1", st["a1"]), ("a2", st["a2"]), ("t", tt),
                 ("h2", H[:, :, 2]), ("run", m["runner"] / W4), ("z", m["clock"] / W4)):
        add(k, a.reshape(-1))
    add("Ho", H[:, :, ORD].reshape(-1, 13))
D = {k: np.concatenate(v) for k, v in acc.items()}
lev, M, S, Mprev, a1, a2, tt, h2, Ho = (D[k] for k in ("lev", "M", "S", "Mprev", "a1", "a2", "t", "h2", "Ho"))
print(f"seeds {SEEDS_F}, {NB} sequences each")

print("=" * 120)
print("F1  level/w4 on ALL states with t >= 1 (no filter on S or M): one-regressor least squares without intercept")
s = tt >= 1
for nm, reg in (("M_t (window incl. x_t)", M), ("M_(t-1) (window excl. x_t)", Mprev), ("max(M_t, 0)", np.maximum(M, 0)), ("max(M_(t-1), 0)", np.maximum(Mprev, 0))):
    c, ru, rc = lsq(reg[s, None], lev[s])
    print(f"   level/w4 ~ {c[0]:.3f} * {nm:28s}  uncentred R2 = {ru:.4f}   centred R2 = {rc:.4f}")
print("   by a1 (t >= 1), regressor max(M_(t-1), 0): slope, centred R2 | regressor max(M_t, 0): slope, centred R2")
for A1 in range(0, 10):
    s = (tt >= 1) & (a1 == A1)
    c, ru, rc = lsq(np.maximum(Mprev[s], 0)[:, None], lev[s])
    c2, ru2, rc2 = lsq(np.maximum(M[s], 0)[:, None], lev[s])
    print(f"      a1={A1}: {c[0]:.3f}, {rc:.4f} | {c2[0]:.3f}, {rc2:.4f}")
print("   binned by M_(t-1) (t >= 1, all a1): median level/w4, and median of level/(w4 M_(t-1))")
for lo, hi in ((-np.inf, -1), (-1, -0.5), (-0.5, -0.2), (-0.2, 0), (0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1), (1, 1.5), (1.5, 2), (2, np.inf)):
    s = (tt >= 1) & (Mprev > lo) & (Mprev <= hi)
    print(f"      M_(t-1) in ({lo:5.1f},{hi:5.1f}]: median level/w4 = {np.median(lev[s]):7.4f}   median ratio = {np.median(lev[s] / Mprev[s]):7.3f}   10%-90% of level/w4: {np.percentile(lev[s], 10):7.3f} .. {np.percentile(lev[s], 90):7.3f}   n={int(s.sum())}")
print("   M6 form on states the claim left out (a1 >= 1):  level/w4 ~ alpha (M - S) + beta S")
for nm, s in (("S > 0.05 (claim)", (a1 >= 1) & (S > 0.05)), ("-inf < S <= 0.05, M > 0.05", (a1 >= 1) & (S <= 0.05) & np.isfinite(S) & (M > 0.05)), ("S < -0.3, M > 0.3", (a1 >= 1) & (S < -0.3) & np.isfinite(S) & (M > 0.3)), ("M <= 0", (a1 >= 1) & (M <= 0) & np.isfinite(S))):
    c, ru, rc = lsq(np.stack([M[s] - S[s], S[s]], 1), lev[s])
    c1, ru1, rc1 = lsq(np.maximum(M[s], 0)[:, None], lev[s]) if (M[s] > 0).any() else ((np.nan,), np.nan, np.nan)
    print(f"      {nm:28s}: alpha = {c[0]:.3f}, beta = {c[1]:.3f}, R2u = {ru:.4f}, R2c = {rc:.4f}   |  level/w4 ~ {c1[0]:.3f} max(M,0): R2u = {ru1:.4f}, R2c = {rc1:.4f}   rms level/w4 = {np.sqrt(np.mean(lev[s] ** 2)):.4f}  n={int(s.sum())}")
print("   growth of the level between records (no new max): ratio level_(t+1)/level_t by age; eigenvalue is 1.0099")
levs = lev.reshape(-1, T); a1s = a1.reshape(-1, T); Ms = M.reshape(-1, T)
for A1 in range(1, 9):
    s = (a1s[:, :-1] == A1) & (a1s[:, 1:] == A1 + 1) & (Ms[:, :-1] > 0.3)
    rr = levs[:, 1:][s] / levs[:, :-1][s]
    print(f"      a1 {A1}->{A1 + 1}: median ratio = {np.median(rr):.4f}  [q10 {np.percentile(rr, 10):.4f}, q90 {np.percentile(rr, 90):.4f}]")

print("=" * 120)
print("F2  CENTRE OF THE ORBIT")
names_real = [n for n in NAMES if not IS_COMPLEX[n]]
def decompose(vec):
    """mode pieces (13-vectors) of an ordinary-unit state vector"""
    out = {}
    for n in NAMES:
        c = vec @ LEFT[n]
        out[n] = (2 * np.real(c * RIGHT[n]) if IS_COMPLEX[n] else np.real(c * RIGHT[n]), c)
    return out
nn = np.linalg.norm
cosv = lambda a, b: float(a @ b / nn(a) / nn(b))
print("   (a) impulse x = (1, 0, ..., 0):")
Hi = run(np.eye(1, T))[0]
print("       units off (h = 0) per step t:", {t: [u for u in ORD if Hi[t, u] == 0] for t in range(T)})
print("       pulse units 4, 5, 9 per step:", np.round(Hi[:, PULSE], 3).tolist())
mi = coords(Hi)
print("       level/w4 per step:", np.round(mi["level"] / W4, 3), "  clock |z|/w4:", np.round(np.abs(mi["clock"]) / W4, 3), "  clock phase:", np.round(np.degrees(np.angle(mi["clock"])), 0))
for lo in (2, 1):
    cen = Hi[lo:, ORD].mean(0)
    pieces = decompose(cen)
    print(f"       centre := mean of h_t[ORD] over t = {lo}..9: |centre| = {nn(cen):.2f};  cos(centre, level eigenvector) = {cosv(cen, RIGHT['level'].real):.4f};"
          f"  |centre - level piece| / |centre| = {nn(cen - pieces['level'][0]) / nn(cen):.3f}")
    print("          mode pieces of the centre, |piece| (raw units): " + ", ".join(f"{n} {nn(p[0]):.2f}" for n, p in pieces.items()))
# best-fit plane through h_2..h_9 (PCA on ORD), perpendicular from the origin
Xc = Hi[2:, ORD]
mu = Xc.mean(0)
_, sv, Vt = np.linalg.svd(Xc - mu, full_matrices=False)
V2 = Vt[:2]
nperp = mu - V2.T @ (V2 @ mu)
print(f"       plane of PC0, PC1 of h_2..h_9 (ORD): variance share {sv[:2] @ sv[:2] / (sv @ sv):.4f}; perpendicular from origin |n| = {nn(nperp):.2f};"
      f" cos(n, level eigenvector) = {cosv(nperp, RIGHT['level'].real):.4f}; cos(mean, level eigenvector) = {cosv(mu, RIGHT['level'].real):.4f}")
print(f"       cos between the clock plane (Re, Im of RIGHT clock) and the PC plane: singular values of the 2x2 overlap = "
      f"{np.round(np.linalg.svd(np.linalg.qr(np.stack([RIGHT['clock'].real, RIGHT['clock'].imag], 1))[0].T @ V2.T, compute_uv=False), 4)}")

print("\n   (b) data: mean of h[ORD] / M per age a1 = 1..8 (states with M > 0.3 and relative gap r = (M - max(S,0))/M in a bin), then the unweighted mean over the 8 ages = 'centre'")
sig = np.sqrt((Ho ** 2).mean(0))            # rms activation per ordinary unit (for the second normalisation)
r = (M - np.maximum(S, 0)) / np.where(M > 0, M, np.nan)
for lo, hi in ((0.6, 1.0), (0.3, 0.6), (0.0, 0.3)):
    mus = []
    for A1 in range(1, 9):
        s = (a1 == A1) & (M > 0.3) & (r > lo) & (r <= hi)
        mus.append((Ho[s] / M[s, None]).mean(0))
    mus = np.array(mus)
    cen = mus.mean(0)
    pieces = decompose(cen)
    radius = np.sqrt(((mus - cen) ** 2).sum(1).mean())
    print(f"       r in ({lo},{hi}]: |centre|/M = {nn(cen):.2f}; rms distance of the 8 age-means from the centre = {radius:.2f};  cos(centre, level eigenvector) = {cosv(cen, RIGHT['level'].real):.4f}"
          f" (in rms units: {cosv(cen / sig, RIGHT['level'].real / sig):.4f});  level coordinate of the centre / w4 = {pieces['level'][1].real / W4:.3f}")
    print(f"          |centre - level piece| / |centre| = {nn(cen - pieces['level'][0]) / nn(cen):.3f} (rms units: {nn((cen - pieces['level'][0]) / sig) / nn(cen / sig):.3f});"
          f"  |centre - level - runner pieces| / |centre| = {nn(cen - pieces['level'][0] - pieces['runner'][0]) / nn(cen):.3f}")
    print("          |mode piece| of the centre (raw units): " + ", ".join(f"{n} {nn(p[0]):.2f}" for n, p in pieces.items()))
    print("          centre per unit / (w4 * level eigenvector entry): " + " ".join(f"u{u}:{cen[i] / (W4 * RIGHT['level'].real[i]):.2f}" for i, u in enumerate(ORD)))

print("=" * 120)
print("F3  the level coordinate against unit 2 alone (states with a1 >= 1, S > 0.05)")
s = (a1 >= 1) & (S > 0.05)
c, ru, rc = lsq(h2[s, None], lev[s])
print(f"   level/w4 ~ {c[0]:.5f} * h_2: R2u = {ru:.4f}, centred R2 = {rc:.4f};   LEFT[level] entry of unit 2 / w4 = {LEFT['level'].real[ORD.index(2)] / W4:.5f}")
c, ru, rc = lsq(M[s, None], h2[s])
print(f"   h_2 ~ {c[0]:.3f} * M_t: R2u = {ru:.4f}, centred R2 = {rc:.4f};   level/w4 ~ M alone: centred R2 = {lsq(M[s, None], lev[s])[2]:.4f}")
