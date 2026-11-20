"""V4 verifier: claim K6 (clock coordinate of the comparator thresholds). Seeds 5006 + 5007 (2 x 400k)."""
from vsetup import *

out = Tee("v4_k6.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=220)

# left eigenvector of Bc: ell @ Bc = lam * ell, for the eigenvalue with positive angle
ev, vecs = np.linalg.eig(Bc.T)
k = int(np.argmax(np.angle(ev) * (np.abs(ev) > 0.5)))
lam = ev[k]
ell = vecs[:, k]
ell = ell / (ell @ (w[4] * b4))                  # plain (non-conjugating) dot product
out(f"lambda = {lam:.5f}: |lambda| = {abs(lam):.4f}, angle = {np.degrees(np.angle(lam)):.3f} deg")
out("check ell @ Bc - lam * ell:", np.abs(ell @ Bc - lam * ell).max(), "  ell.(w4 b4) =", ell @ (w[4] * b4))
out("ell =", ell)
out("|ell| per component:", np.abs(ell), " phases (deg):", np.degrees(np.angle(ell)))
out(f"|ell . 1| = {abs(ell.sum()):.4f}   (claimed 0.013);  ell.1 = {ell.sum():.4f}")
out(f"ell . b2 = {ell @ b2:.5f}, |ell.b2| * 17.3 = {abs(ell @ b2) * 17.3:.4f}  (what unit 2 = 17.3 M leaks into z_c, per unit M)")
out(f"|ell|_2 = {np.linalg.norm(ell):.4f},  |ell.1| / (|ell|_2 * 2) = {abs(ell.sum()) / (np.linalg.norm(ell) * 2):.4f}")

Z, Ms, Ss, A1, A2, TT = [], [], [], [], [], []
for seed in (5006, 5007):
    x = sample(400_000, seed)
    H = run(x)
    th, _ = thresholds(H)
    rs = running_stats(x)
    Z.append(th[:, 1:] @ ell)                    # z_c for t = 0..9 : ell . theta(t+1)
    Ms.append(rs["M1"]); Ss.append(rs["M2"]); A1.append(rs["a1"]); A2.append(rs["a2"])
    TT.append(np.broadcast_to(np.arange(10), x.shape))
Z, Ms, Ss, A1, A2, TT = map(np.concatenate, (Z, Ms, Ss, A1, A2, TT))
Sp = np.maximum(Ss, 0.0)


def cmedian(z):
    return np.median(z.real) + 1j * np.median(z.imag)


def fit(sel, phase=None):
    zc = Z[sel]
    med = cmedian(zc / Ms[sel])
    ph = np.angle(med) if phase is None else phase
    y = np.real(zc * np.exp(-1j * ph))
    X = np.stack([Ms[sel], Sp[sel]], 1)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2u = 1 - ((y - X @ b) ** 2).sum() / (y ** 2).sum()
    return med, ph, b, r2u, int(sel.sum())


claimed_phase = [33.5, 72.0, 120.5, 167.3, 218.6, 262.8, 308.8, 346.2]
claimed_bM = [0.93, 0.83, 0.75, 0.68, 0.62, 0.62, 0.66, 0.73]
claimed_ratio = [-0.91, -0.94, -0.96, -0.91, -0.84, -0.80, -0.77, -0.78]
for label, tmax in (("t <= 9 (theta(10) included)", 9), ("t <= 8", 8)):
    out(f"\n=== K6, reading: {label}; states with a1(t) = a, M_t > 0.3; n = 800000 sequences (seeds 5006, 5007) ===")
    out("  a :      n   phase(deg) [claimed]  |median z/M|   b_M [claimed]   b_S   b_S/b_M [claimed]   uncentred R2")
    for a in range(1, 9):
        sel = (A1 == a) & (Ms > 0.3) & (TT <= tmax)
        med, ph, b, r2u, n = fit(sel)
        out(f"  {a} : {n:7d}  {np.degrees(ph) % 360:7.1f}  [{claimed_phase[a-1]:5.1f}]    {abs(med):.3f}      "
            f"{b[0]:.3f} [{claimed_bM[a-1]:.2f}]   {b[1]:+.3f}   {b[1] / b[0]:+.3f} [{claimed_ratio[a-1]:+.2f}]      {r2u:.4f}")

out("\n=== sampling noise: the two seeds separately (t <= 9) ===")
half = Z.shape[0] // 2
for part, sl in (("seed 5006", slice(0, half)), ("seed 5007", slice(half, None))):
    rows = []
    for a in range(1, 9):
        sel = np.zeros_like(A1, dtype=bool)
        sel[sl] = (A1[sl] == a) & (Ms[sl] > 0.3)
        med, ph, b, r2u, n = fit(sel)
        rows.append((np.degrees(ph) % 360, b[0], b[1] / b[0], r2u))
    rows = np.array(rows)
    out(f"  {part}: phase {rows[:, 0]}")
    out(f"  {part}: b_M   {rows[:, 1]}")
    out(f"  {part}: ratio {rows[:, 2]}")
    out(f"  {part}: R2u   {rows[:, 3]}")

out("\n=== alternative definition of the 'median' phase: angle of the mean of z/M, and median of the angles ===")
for a in range(1, 9):
    sel = (A1 == a) & (Ms > 0.3)
    q = Z[sel] / Ms[sel]
    ref = np.angle(cmedian(q))
    dang = np.angle(q * np.exp(-1j * ref))
    out(f"  a = {a}: angle(mean) = {np.degrees(np.angle(q.mean())) % 360:6.1f}, angle(componentwise median) = {np.degrees(ref) % 360:6.1f}, "
        f"median of angles = {(np.degrees(ref + np.median(dang))) % 360:6.1f}; IQR of angle about it: {np.degrees(np.percentile(dang, [25, 75]))}")

out("\n=== age 0 (new max at t) for reference: z_c(t+1) should be ~ the increment, phase ~ 0 ===")
sel = (A1 == 0) & (Ms > 0.3) & (TT >= 1)
med, ph, b, r2u, n = fit(sel)
out(f"  a = 0: n = {n}, phase = {np.degrees(ph):.1f}, |median z/M| = {abs(med):.3f}, b_M = {b[0]:.3f}, b_S = {b[1]:+.3f}, ratio = {b[1]/b[0]:+.3f}, R2u = {r2u:.4f}")
