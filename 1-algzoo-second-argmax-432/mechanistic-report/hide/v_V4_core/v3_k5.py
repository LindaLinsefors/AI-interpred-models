"""V4 verifier: claim K5 (six-unit simulation), plus how good the simulation is on events and thresholds.
Seeds 5004 (main), 5005 (repeat)."""
from vsetup import *

out = Tee("v3_k5.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=220)

Wcc = W[np.ix_(CORE, CORE)]
wc = w[CORE]
r = np.real(RIGHT["level"])                      # level eigenvector of A on ORD
out("level eigenvalue:", LAM["level"], " max |imag part of RIGHT[level]|:", np.abs(np.imag(RIGHT["level"])).max())
r_of = {u: r[ORD.index(u)] for u in ORD}
add = np.array([sum(W[j, u] * r_of[u] for u in WAVE_NO2) / r_of[2] for j in CORE])
out("CORE order:", CORE)
out("compensation added to W[j, 2]:", add, " (claimed +0.031, -0.128, +0.085, +0.100, +0.109, +0.191)")
out("compensation in threshold gauge (divide by g_j; unit 4: by -w_4):", add[2:] / g, add[1] / -w[4])
out("W[2,2] + add = ", W[2, 2] + add[0])
Wcc_comp = Wcc.copy()
Wcc_comp[:, 0] += add                            # column 0 of the core = unit 2


def sim(x, Wm):
    B = x.shape[0]
    h = np.zeros((B, 6))
    Hs = np.empty((B, 10, 6))
    for t in range(10):
        h = np.maximum(h @ Wm.T + x[:, t, None] * wc, 0.0)
        Hs[:, t] = h
    return Hs


def r2c(pred, real, axis=None):
    """centred R^2 = 1 - SS_res / SS_tot."""
    return 1 - ((pred - real) ** 2).sum(axis) / ((real - real.mean(axis, keepdims=True)) ** 2).sum(axis)


for seed in (5004, 5005):
    x = sample(400_000, seed)
    H = run(x)
    Hr = H[:, :, CORE]
    rs = running_stats(x)
    M, S, a1 = rs["M1"], rs["M2"], rs["a1"]
    out(f"\n================ seed {seed}, n = 400000 ================")
    sims = {"plain": sim(x, Wcc), "compensated": sim(x, Wcc_comp)}
    for name, Hs in sims.items():
        pooled = np.array([r2c(Hs[:, :, i].ravel(), Hr[:, :, i].ravel()) for i in range(6)])
        out(f"K5 {name:12s}: pooled centred R2 for units {CORE}: {pooled}")
    out("   claimed plain 0.965, 0.981, 0.932, 0.964, 0.979, 0.877; compensated 0.991, 0.995, 0.989, 0.995, 0.993, 0.991")
    if seed != 5004:
        continue

    # ---------------------------------------------------------------- per-step R2
    for name, Hs in sims.items():
        out(f"\n--- {name}: centred R2 per step t = 0..9 (rows) for units {CORE} (columns) ---")
        for t in range(10):
            with np.errstate(divide="ignore", invalid="ignore"):
                row = np.array([r2c(Hs[:, t, i], Hr[:, t, i]) for i in range(6)])
            out(f"   t = {t}: {row}")

    # ---------------------------------------------------------------- trivial baselines for the activations
    out("\n--- baselines that use NO core dynamics at all (pooled centred R2, all steps) ---")
    Mprev = np.zeros_like(M)
    Mprev[:, 1:] = np.maximum(M[:, :-1], 0.0)                   # running max of x_0..x_{t-1}, floored at 0
    base = np.empty_like(Hr)
    base[:, :, 0] = 17.3 * Mprev                               # unit 2 = 17.3 * previous running max
    base[:, :, 1] = w[4] * np.maximum(x - Mprev, 0.0)          # unit 4 = w4 [x - prev max]_+
    for i, j in enumerate(COMP):
        base[:, :, 2 + i] = g[i] * np.maximum(Mprev - x, 0.0)   # comparator = g_j [prev max - x]_+  (no rotation)
    out("   'all thresholds = running max of x_0..x_(t-1), floored at 0':",
        np.array([r2c(base[:, :, i].ravel(), Hr[:, :, i].ravel()) for i in range(6)]))
    base0 = np.empty_like(Hr)
    base0[:, :, 0] = 0
    base0[:, :, 1] = w[4] * np.maximum(x, 0.0)
    for i, j in enumerate(COMP):
        base0[:, :, 2 + i] = g[i] * np.maximum(-x, 0.0)
    out("   'all thresholds = 0' (h_j = ReLU(w_j x_t)):                ",
        np.array([r2c(base0[:, :, i].ravel(), Hr[:, :, i].ravel()) for i in range(6)]))

    # ---------------------------------------------------------------- thresholds: sim vs real
    th_real, th4_real = thresholds(H)                           # (B, 11, 4), (B, 11)
    out("\n--- thresholds theta_j(t), j = 1, 6, 7, 8 and unit 4: simulation's own thresholds vs the real ones ---")
    out("   (theta^sim(t) uses only the six simulated units; real theta(t) uses all 16 real units)")
    for name, Wm in (("plain", Wcc), ("compensated", Wcc_comp)):
        Hs = sims[name]
        Hs_prev = np.concatenate([np.zeros((x.shape[0], 1, 6)), Hs], axis=1)
        th_sim = (Hs_prev @ Wm[2:6].T) / g
        th4_sim = -(Hs_prev @ Wm[1]) / w[4]
        allth_sim = np.concatenate([th_sim, th4_sim[:, :, None]], axis=2)
        allth_real = np.concatenate([th_real, th4_real[:, :, None]], axis=2)
        out(f"   {name}: centred R2 per step (rows t = 1..9; columns theta_1, theta_6, theta_7, theta_8, theta_4), "
            f"then mean and rms of (sim - real)/M_(t-1) on states with M_(t-1) > 0.3")
        for t in range(1, 10):
            row = np.array([r2c(allth_sim[:, t, i], allth_real[:, t, i]) for i in range(5)])
            s = M[:, t - 1] > 0.3
            d = (allth_sim[s, t] - allth_real[s, t]) / M[s, t - 1][:, None]
            out(f"     t = {t}: R2 {row}   bias {d.mean(0)}   rms {np.sqrt((d ** 2).mean(0))}")
        pooled = np.array([r2c(allth_sim[:, 1:10, i].ravel(), allth_real[:, 1:10, i].ravel()) for i in range(5)])
        out(f"     pooled t = 1..9: R2 {pooled}")
        # R2 of the DIFFERENCES between thresholds (common level removed) and of the common level
        dif_sim = th_sim[:, 1:10] - th_sim[:, 1:10].mean(2, keepdims=True)
        dif_real = th_real[:, 1:10] - th_real[:, 1:10].mean(2, keepdims=True)
        out(f"     differences theta_j - mean_j theta (pooled t = 1..9): R2 per j "
            f"{np.array([r2c(dif_sim[:, :, i].ravel(), dif_real[:, :, i].ravel()) for i in range(4)])}; "
            f"per step, all j pooled: {np.array([r2c(dif_sim[:, t].ravel(), dif_real[:, t].ravel()) for t in range(9)])}")
        lev_sim, lev_real = th_sim[:, 1:10].mean(2), th_real[:, 1:10].mean(2)
        out(f"     common level mean_j theta (pooled): R2 {r2c(lev_sim.ravel(), lev_real.ravel()):.4f}; per step "
            f"{np.array([r2c(lev_sim[:, t], lev_real[:, t]) for t in range(9)])}")

    # ---------------------------------------------------------------- unit 4 firing events
    out("\n--- unit 4 firing events (h_4 > 0): real network vs simulation, per step ---")
    newmax = np.zeros_like(x, dtype=bool)
    newmax[:, 0] = x[:, 0] > 0
    newmax[:, 1:] = x[:, 1:] > np.maximum(M[:, :-1], 0.0)        # x_t beats max(0, x_0..x_{t-1})
    fr = Hr[:, :, 1] > 0
    out(f"   real unit 4 vs ground truth 'x_t > max(0, x_0..x_(t-1))': P(fire | new max) = {(fr & newmax).sum() / newmax.sum():.4f}, "
        f"P(new max | fire) = {(fr & newmax).sum() / fr.sum():.4f}, rate fire = {fr.mean():.4f}, rate new max = {newmax.mean():.4f}")
    for name, Hs in sims.items():
        fs = Hs[:, :, 1] > 0
        out(f"   {name}: overall recall P(sim fires | real fires) = {(fs & fr).sum() / fr.sum():.4f}, "
            f"precision P(real fires | sim fires) = {(fs & fr).sum() / fs.sum():.4f}, "
            f"disagreement rate per step-state = {(fs != fr).mean():.4f}; "
            f"sequences with identical unit-4 firing pattern at all 10 steps: {(fs == fr).all(1).mean():.4f}")
        out("     t :  P(real fires)  recall  precision  false-fire rate P(sim fires | real silent)  "
            "R2 of h4 on real-firing states  median sim/real ratio there")
        for t in range(10):
            a, b = fr[:, t], fs[:, t]
            rec = (a & b).sum() / max(a.sum(), 1)
            prec = (a & b).sum() / max(b.sum(), 1)
            ff = (b & ~a).sum() / max((~a).sum(), 1)
            rr = r2c(Hs[a, t, 1], Hr[a, t, 1]) if a.sum() > 10 else np.nan
            ratio = np.median(Hs[a, t, 1] / Hr[a, t, 1])
            out(f"     {t} :  {a.mean():.4f}        {rec:.4f}  {prec:.4f}     {ff:.4f}                              {rr:.4f}      {ratio:.4f}")
        # size of the false fires relative to M
        ffm = fs & ~fr & (M > 0.3)
        if ffm.sum() > 0:
            out(f"     false fires (M > 0.3): median h4_sim/(w4 M) = {np.median(Hs[:, :, 1][ffm] / (w[4] * M[ffm])):.4f}, "
                f"90th pct = {np.percentile(Hs[:, :, 1][ffm] / (w[4] * M[ffm]), 90):.4f}")

    # ---------------------------------------------------------------- comparator gate agreement
    out("\n--- comparator gates (on/off) agreement, per step: P(all four gates equal) plain / compensated ---")
    gr = Hr[:, :, 2:] > 0
    for t in range(10):
        row = [((sims[n][:, t, 2:] > 0) == gr[:, t]).all(1).mean() for n in ("plain", "compensated")]
        out(f"   t = {t}: {row[0]:.4f} / {row[1]:.4f}")
    out("   real network: P(all four comparators on) per step:", gr.all(2).mean(0))
    out("   real network: P(number of comparators on = 0,1,2,3,4), all steps:",
        np.bincount(gr.sum(2).ravel(), minlength=5) / gr[:, :, 0].size)
