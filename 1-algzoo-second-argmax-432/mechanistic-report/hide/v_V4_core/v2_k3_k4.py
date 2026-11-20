"""V4 verifier: claims K3 (threshold budget) and K4 (unit 2 vs running max). Seeds 5002, 5003."""
from vsetup import *

out = Tee("v2_k3_k4.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=200)
PULSE_NO4 = [5, 9]

# rows: the four comparators (gauge g_j) and unit 4 (gauge -w_4, because z_4 = w_4 (x - theta_4))
ROWS = COMP + [4]
GAUGE = np.concatenate([g, [-w[4]]])


def parts(H):
    """Contribution of each source group to theta_row(t+1), for t = 0..9. Returns dict name -> (B, T, 5)."""
    res = {}
    for name, src in (("unit2", [2]), ("wave_no2", WAVE_NO2), ("comp", COMP), ("unit4", [4]), ("pulse59", PULSE_NO4)):
        res[name] = (H[:, :, src] @ W[np.ix_(ROWS, src)].T) / GAUGE
    return res


for seed in (5002, 5003):
    x = sample(400_000, seed)
    H = run(x)
    rs = running_stats(x)
    M, a1 = rs["M1"], rs["a1"]
    P = parts(H)
    tt = np.broadcast_to(np.arange(10), M.shape)

    out(f"\n================ seed {seed}, n = 400000 ================")
    out("=== K3: parts of theta_j(t+1)/M_t;  selection t <= 8, a1(t) >= 2, M_t > 0.3 ===")
    sel = (tt <= 8) & (a1 >= 2) & (M > 0.3)
    out("number of states:", int(sel.sum()))
    out("columns: j = 1, 6, 7, 8, unit-4 threshold")
    tot = sum(P.values())
    for name in ("unit2", "wave_no2", "comp", "unit4", "pulse59"):
        q = P[name][sel] / M[sel][:, None]
        out(f"  {name:9s} mean {q.mean(0)}   std {q.std(0)}   median {np.median(q, 0)}")
    q = tot[sel] / M[sel][:, None]
    out(f"  {'TOTAL':9s} mean {q.mean(0)}   std {q.std(0)}   median {np.median(q, 0)}")

    out("--- same, but selection also allowing t = 9 (theta(10) is hypothetical) ---")
    sel9 = (a1 >= 2) & (M > 0.3)
    for name in ("unit2", "wave_no2", "comp"):
        q = P[name][sel9] / M[sel9][:, None]
        out(f"  {name:9s} mean {q.mean(0)}   std {q.std(0)}")

    out("--- K3 by age of the max (t <= 8, M_t > 0.3): mean of each part / M_t ---")
    for a in range(0, 9):
        s = (tt <= 8) & (a1 == a) & (M > 0.3)
        out(f"  a1 = {a}: n = {int(s.sum()):7d}  unit2 {(P['unit2'][s] / M[s][:, None]).mean(0)}  "
            f"wave_no2 {(P['wave_no2'][s] / M[s][:, None]).mean(0)}  comp {(P['comp'][s] / M[s][:, None]).mean(0)}"
            f"  total {(tot[s] / M[s][:, None]).mean(0)}")

    out("--- K3 with a ratio-of-means estimator (sum part / sum M) instead of mean of ratios ---")
    for name in ("unit2", "wave_no2", "comp"):
        out(f"  {name:9s} {P[name][sel].sum(0) / M[sel].sum()}")

    # ------------------------------------------------------------ K4
    out("\n=== K4: median of h_2(t) / (17.3 M_{t-1}) by age of the max at t (M_{t-1} > 0.3) ===")
    Mprev = np.full_like(M, -np.inf)
    Mprev[:, 1:] = M[:, :-1]
    a1prev = np.full_like(a1, -1)
    a1prev[:, 1:] = a1[:, :-1]
    r = H[:, :, 2] / (17.3 * np.where(Mprev > 0.3, Mprev, np.nan))
    med, n_, iqr = [], [], []
    for a in range(1, 9):
        s = (a1 == a) & (Mprev > 0.3)
        med.append(np.median(r[s]))
        n_.append(int(s.sum()))
        iqr.append(np.percentile(r[s], [25, 75]))
    out("  reading A: age = a1(t) (window x_0..x_t), a = 1..8:")
    out("    median:", np.array(med))
    out("    n     :", n_)
    out("    IQR   :", np.array(iqr).T)
    med = []
    for a in range(1, 9):
        s = (a1prev == a) & (Mprev > 0.3)
        med.append(np.median(r[s]))
    out("  reading B: age = a1(t-1) (age at t-1 of the max of x_0..x_{t-1}), a = 1..8:")
    out("    median:", np.array(med))
    med = []
    for a in range(1, 9):
        s = (a1prev == a - 1) & (Mprev > 0.3)
        med.append(np.median(r[s]))
    out("  reading C: age measured at t of the max of x_0..x_{t-1} (= a1(t-1) + 1), incl. states with a new max at t, a = 1..8:")
    out("    median:", np.array(med))
    out("  claimed: 1.02, 1.02, 1.07, 1.05, 0.98, 0.90, 0.85, 0.86")

    # how good a 'running max' is unit 2? centred R^2 of h_2(t) on M_{t-1}^+ , slope through origin
    ok = np.isfinite(Mprev)
    Mp = np.maximum(Mprev[ok], 0)
    h2 = H[:, :, 2][ok]
    slope = (h2 * Mp).sum() / (Mp * Mp).sum()
    r2c = 1 - ((h2 - slope * Mp) ** 2).sum() / ((h2 - h2.mean()) ** 2).sum()
    out(f"  h_2(t) ~ k * max(M_(t-1), 0), all t >= 1: k = {slope:.3f}, centred R2 = {r2c:.4f}")
    sl2 = (h2 * Mp).sum() / (Mp * Mp).sum()
    ok2 = ok & (Mprev > 0.3)
    q = H[:, :, 2][ok2] / Mprev[ok2]
    out(f"  h_2(t) / M_(t-1) for M_(t-1) > 0.3: mean {q.mean():.3f}, median {np.median(q):.3f}, 5%..95% = {np.percentile(q, [5, 95])}")
