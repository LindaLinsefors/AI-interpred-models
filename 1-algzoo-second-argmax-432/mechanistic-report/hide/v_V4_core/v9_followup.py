"""V4 verifier, follow-ups.
(1) POSITIVE new runner-ups: how many comparators do they switch off; is the runner-up 'registered' by the block?
(2) registered vs unregistered younger runner-ups: |z_c| / (M - S+) and final accuracy.
(3) unit 2 by S+/M (model-free): is the 'running max' in unit 2 contaminated by the runner-up?
(4) who changes the rotation rate?  exact split of z_c(t+1) - lambda z_c(t) into unit 2 / other wave units / pulses 5, 9.
Seeds 5018, 5019 (2 x 400k)."""
from vsetup import *

out = Tee("v9_followup.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=220)

ev, vecs = np.linalg.eig(Bc.T)
k = int(np.argmax(np.angle(ev) * (np.abs(ev) > 0.5)))
lam = ev[k]
ell = vecs[:, k] / (vecs[:, k] @ (w[4] * b4))


def pol(c):
    return f"{abs(c):.3f}@{np.degrees(np.angle(c)) % 360:5.1f}"


D = {n_: [] for n_ in ("x", "H", "non", "u4", "Zc", "M", "S", "a1", "a2", "p1", "p2", "ok", "z2", "zw", "zp")}
for seed in (5018, 5019):
    x = sample(400_000, seed)
    H, Zp = run(x, return_pre=True)
    th, _ = thresholds(H)
    rs = running_stats(x)
    D["x"].append(x); D["H"].append(H[:, :, [2]]); D["non"].append((Zp[:, :, COMP] > 0).sum(2)); D["u4"].append(Zp[:, :, 4] > 0)
    D["Zc"].append(th[:, 1:] @ ell)
    for n_ in ("M", "S", "a1", "a2", "p1", "p2"):
        D[n_].append(rs[{"M": "M1", "S": "M2"}.get(n_, n_)])
    D["ok"].append(logits(H).argmax(1) == second_argmax(x))
    D["z2"].append(H[:, :, 2] * (ell @ b2))
    D["zw"].append(((H[:, :, WAVE_NO2] @ W[np.ix_(COMP, WAVE_NO2)].T) / g) @ ell)
    D["zp"].append(((H[:, :, [5, 9]] @ W[np.ix_(COMP, [5, 9])].T) / g) @ ell)
x, non, u4, Zc, M, S, a1, a2, p1, p2, ok, z2, zw, zp = (np.concatenate(D[n_]) for n_ in
                                                      ("x", "non", "u4", "Zc", "M", "S", "a1", "a2", "p1", "p2", "ok", "z2", "zw", "zp"))
h2 = np.concatenate(D["H"])[:, :, 0]
Sp = np.maximum(S, 0.0)
tt = np.broadcast_to(np.arange(10), x.shape)
Mprev = np.concatenate([np.full((x.shape[0], 1), -np.inf), M[:, :-1]], axis=1)
Sprev = np.concatenate([np.full((x.shape[0], 1), -np.inf), S[:, :-1]], axis=1)

# ------------------------------------------------------------------ (1)
out("=== (1) new runner-up at step t (S_(t-1) < x_t < M_(t-1)), t = 2..9, M_(t-1) > 0.3, by sign/size of the newcomer ===")
newsec = (x < Mprev) & (x > Sprev) & (tt >= 2) & (Mprev > 0.3)
rise = (x - np.maximum(Sprev, 0.0)) / Mprev                   # how far above the old runner-up (floored at 0), in units of M
for label, cond in (("x_t <= 0 (S+ does not change)", x <= 0), ("x_t > 0, rise <= 0.05 M", (x > 0) & (rise <= 0.05)),
                    ("x_t > 0, rise in (0.05, 0.2] M", (x > 0) & (rise > 0.05) & (rise <= 0.2)),
                    ("x_t > 0, rise in (0.2, 0.5] M", (x > 0) & (rise > 0.2) & (rise <= 0.5)), ("x_t > 0, rise > 0.5 M", (x > 0) & (rise > 0.5)),
                    ("all positive newcomers", x > 0)):
    sel = newsec & cond
    out(f"  {label:34s}: share of events {sel.sum() / newsec.sum():.3f};  P(#comparators on = 0..4) = {np.bincount(non[sel], minlength=5) / sel.sum()}")
out("  positive newcomers, P(all four comparators stay on) by age of the max at t (a1 = 1..8):")
row = []
for a in range(1, 9):
    sel = newsec & (x > 0) & (a1 == a)
    row.append(f"{(non[sel] == 4).mean():.3f}")
out("    " + "  ".join(row))
out("  positive newcomers with rise > 0.05 M, same:")
row = []
for a in range(1, 9):
    sel = newsec & (x > 0) & (rise > 0.05) & (a1 == a)
    row.append(f"{(non[sel] == 4).mean():.3f} (n={int(sel.sum())})")
out("    " + "  ".join(row))

# ------------------------------------------------------------------ (2)
out("\n=== (2) final states (t = 9) whose runner-up is younger than the max (p2 > p1), S > 0, M > 0.3 ===")
out("  'registered' = at the step the runner-up arrived (p2) at least one comparator was switched off; 'unregistered' = all four stayed on")
fin = (p2[:, 9] > p1[:, 9]) & (S[:, 9] > 0) & (M[:, 9] > 0.3)
non_at_p2 = np.take_along_axis(non, np.maximum(p2[:, [9]], 0), axis=1)[:, 0]
reg = fin & (non_at_p2 < 4)
unreg = fin & (non_at_p2 == 4)
out(f"  share unregistered: {unreg.sum() / fin.sum():.4f}  (n = {int(fin.sum())})")
gap = (M[:, 9] - S[:, 9])
xs = np.sort(x, 1)
relST = (xs[:, -2] - np.maximum(xs[:, -3], 0)) / xs[:, -1]     # how far the runner-up is above the third (floored at 0), in units of M
for label, s_ in (("registered", reg), ("unregistered", unreg)):
    ratio = np.abs(Zc[s_, 8]) / gap[s_]                        # z_c(8) = ell . theta(9): the thresholds used at the last step
    out(f"  {label:13s}: n = {int(s_.sum()):7d}; accuracy {ok[s_].mean():.4f}; median |ell.theta(9)| / (M - S) = {np.median(ratio):.3f} "
        f"(IQR {np.percentile(ratio, [25, 75])}); median (S - T+)/M = {np.median(relST[s_]):.3f}; median (M - S)/M = {np.median(gap[s_] / M[s_, 9]):.3f}")
out("  matched on the lead of the runner-up over the third, relST = (S - T+)/M, runner-up arrived at p2 <= 8:")
for lo, hi in ((0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4)):
    rr = []
    for label, s_ in (("registered", reg), ("unregistered", unreg)):
        c = s_ & (relST > lo) & (relST <= hi) & (p2[:, 9] <= 8)
        rr.append(f"{label}: acc {ok[c].mean():.4f}, med |z|/(M-S) {np.median(np.abs(Zc[c, 8]) / gap[c]):.3f} (n = {int(c.sum())})")
    out(f"    relST in ({lo}, {hi}]:  " + ";  ".join(rr))

# ------------------------------------------------------------------ (3)
out("\n=== (3) unit 2 against the running max, by S+/M (states with a1(t) = age, M_(t-1) > 0.3): median h_2(t) / (17.3 M) ===")
bins = [(-0.01, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0)]
out("  age :  " + "   ".join(f"S+/M in ({lo:.2f},{hi:.2f}]" for lo, hi in bins) + "   all (claim K4)")
sr = np.maximum(Sprev, 0.0) / Mprev
for a in range(1, 9):
    row = []
    for lo, hi in bins:
        sel = (a1 == a) & (Mprev > 0.3) & (sr > lo) & (sr <= hi)
        row.append(f"{np.median(h2[sel] / (17.3 * Mprev[sel])):.3f}")
    sel = (a1 == a) & (Mprev > 0.3)
    out(f"   {a}  :  " + "              ".join(row) + f"          {np.median(h2[sel] / (17.3 * Mprev[sel])):.3f}")

# ------------------------------------------------------------------ (4)
out("\n=== (4) exact one-step identity when all four comparators are on and unit 4 is off at step t+1: ===")
out("    z_c(t+1) = lambda z_c(t) - lambda (ell.1) x_(t+1) + [unit 2] + [other wave units] + [pulses 5, 9]")
z0, z1 = Zc[:, :-1], Zc[:, 1:]
sel = (a1[:, 1:] >= 1) & (M[:, :-1] > 0.3) & (tt[:, 1:] <= 8) & (non[:, 1:] == 4) & ~u4[:, 1:]
resid = z1 - lam * z0 + lam * ell.sum() * x[:, 1:] - z2[:, 1:] - zw[:, 1:] - zp[:, 1:]
out(f"  check: max |residual| = {np.abs(resid[sel]).max():.2e}  (n = {int(sel.sum())})")
out(f"  lambda = {pol(lam)}.  Regression of each extra term on (z_c(t), M_t): term ~ dL * z_c(t) + K * M")
X = np.stack([z0[sel], M[:, :-1][sel].astype(complex)], 1)
tot = 0
for name, term in (("unit 2", z2[:, 1:]), ("other wave units", zw[:, 1:]), ("pulses 5, 9", zp[:, 1:]), ("-lambda (ell.1) x", -lam * ell.sum() * x[:, 1:])):
    c, *_ = np.linalg.lstsq(X, term[sel], rcond=None)
    tot = tot + c
    out(f"    {name:18s}: dL = {c[0].real:+.4f}{c[0].imag:+.4f}i ({pol(c[0])}),  K = {pol(c[1])};  mean |term|/M = {np.abs(term[sel] / M[:, :-1][sel]).mean():.4f}")
out(f"    sum: L = lambda + dL = {pol(lam + tot[0])}, K = {pol(tot[1])}   (direct fit in v7: L = 0.943@41.6, K = 0.039@283.2)")
out("  the same by age of the max at t (dL from unit 2 | from the other wave units | L total):")
aprev = a1[:, :-1]
for a in range(0, 8):
    sa = sel & (aprev == a)
    Xa = np.stack([z0[sa], M[:, :-1][sa].astype(complex)], 1)
    c2, *_ = np.linalg.lstsq(Xa, z2[:, 1:][sa], rcond=None)
    cw, *_ = np.linalg.lstsq(Xa, zw[:, 1:][sa], rcond=None)
    cp, *_ = np.linalg.lstsq(Xa, zp[:, 1:][sa], rcond=None)
    ct, *_ = np.linalg.lstsq(Xa, z1[sa], rcond=None)
    out(f"    age {a}: unit 2 dL = {c2[0].real:+.3f}{c2[0].imag:+.3f}i | wave dL = {cw[0].real:+.3f}{cw[0].imag:+.3f}i | pulses dL = {cp[0].real:+.3f}{cp[0].imag:+.3f}i "
        f"| L = {pol(ct[0])}  (lambda = {pol(lam)})")
