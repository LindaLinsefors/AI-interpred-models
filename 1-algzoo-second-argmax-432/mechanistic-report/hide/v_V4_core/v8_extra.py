"""V4 verifier, extra checks.
(a) K6 with the scale taken out: centred R2 of Re(z_c e^{-i phase})/M against S+/M (uncentred R2 is inflated by |M|).
(b) does the comparator block see a NEW runner-up?  events by number of comparators that it switches off.
(c) one-step perturbation of the four thresholds in the real network: a common shift vs a difference pattern.
(d) closed-core hybrids: accuracy as a function of the gaps (fragile near ties only, or broken everywhere?).
(e) does unit 2 hold only the running max?  h_2(t) ~ a M + b S+ by age.
Seeds 5015 (a, b, e), 5016 (c), 5017 (d); n = 400000 each."""
from vsetup import *

out = Tee("v8_extra.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=220)

ev, vecs = np.linalg.eig(Bc.T)
k = int(np.argmax(np.angle(ev) * (np.abs(ev) > 0.5)))
lam = ev[k]
ell = vecs[:, k] / (vecs[:, k] @ (w[4] * b4))
evr, V = np.linalg.eig(Bc)
kr = int(np.argmax(np.angle(evr) * (np.abs(evr) > 0.5)))
vr = V[:, kr] / (ell @ V[:, kr])


def cmed(z):
    return np.median(z.real) + 1j * np.median(z.imag)


# =============================================================================================== (a), (b), (e)
x = sample(400_000, 5015)
H, Zp = run(x, return_pre=True)
th, th4 = thresholds(H)
rs = running_stats(x)
M, S, a1, a2 = rs["M1"], rs["M2"], rs["a1"], rs["a2"]
Sp = np.maximum(S, 0.0)
tt = np.broadcast_to(np.arange(10), x.shape)
Zc = th[:, 1:] @ ell
base = (tt <= 8) & (M > 0.3) & (tt >= 1)

out("=== (a) K6 regression with the scale removed (seed 5015): y = Re(z_c e^{-i phase_a}), fit y ~ b_M M + b_S S+ (as the lead) ===")
out("  a1 :  uncentred R2 (lead)   centred R2 of y   centred R2 of y/M with the SAME fit   centred R2 of y/M refitted as y/M ~ c0 + c1 S+/M   corr(y/M, S+/M)")
for a in range(1, 9):
    sel = base & (a1 == a)
    ph = np.angle(cmed(Zc[sel] / M[sel]))
    y_ = np.real(Zc[sel] * np.exp(-1j * ph))
    X = np.stack([M[sel], Sp[sel]], 1)
    b, *_ = np.linalg.lstsq(X, y_, rcond=None)
    res = y_ - X @ b
    r2u = 1 - (res ** 2).sum() / (y_ ** 2).sum()
    r2c = 1 - (res ** 2).sum() / ((y_ - y_.mean()) ** 2).sum()
    yn, sn = y_ / M[sel], Sp[sel] / M[sel]
    r2n = 1 - ((res / M[sel]) ** 2).sum() / ((yn - yn.mean()) ** 2).sum()
    Xn = np.stack([np.ones_like(sn), sn], 1)
    bn, *_ = np.linalg.lstsq(Xn, yn, rcond=None)
    r2nn = 1 - ((yn - Xn @ bn) ** 2).sum() / ((yn - yn.mean()) ** 2).sum()
    out(f"  {a}  :  {r2u:.4f}               {r2c:.4f}            {r2n:.4f}                               {r2nn:.4f}  (c0 {bn[0]:.3f}, c1 {bn[1]:+.3f}, -c1/c0 {-bn[1] / bn[0]:.3f})          {np.corrcoef(yn, sn)[0, 1]:+.3f}")

out("\n=== (b) new runner-up at step t (S_(t-1) < x_t < M_(t-1)), t >= 2, M > 0.3: how many comparators does it switch off? ===")
newsec = np.zeros_like(x, dtype=bool)
newsec[:, 1:] = (x[:, 1:] < M[:, :-1]) & (x[:, 1:] > S[:, :-1])
non = (Zp[:, :, COMP] > 0).sum(2)
Mshift = np.concatenate([np.full((x.shape[0], 1), -np.inf), M[:, :-1]], axis=1)
R = np.take_along_axis(Mshift, rs["p1"], axis=1)
Rp = np.maximum(R, 0.0)
Sprev = np.concatenate([np.full((x.shape[0], 1), -np.inf), S[:, :-1]], axis=1)   # runner-up before step t
evsel = newsec & (tt >= 2) & (tt <= 8) & (M > 0.3)
thmin = th[:, :10].min(2)                                                       # min_j theta_j(t)
out(f"  events: {int(evsel.sum())};  P(#comparators on at t = 0..4): {np.bincount(non[evsel], minlength=5) / evsel.sum()}")
out("  n_on : share   median (x_t - old runner-up+)/M   median (M - x_t)/M   median min_j theta_j(t)/M    then z_c ~ alpha (M - S+) + beta S+ + gamma (S+ - Sold+):  Re(gamma/alpha)")
Soldp = np.maximum(Sprev, 0.0)
for n_on in (4, 3, 2, 1):
    sel = evsel & (non == n_on)
    Xc = np.stack([(M - Sp)[sel], Sp[sel], (Sp - Soldp)[sel]], 1).astype(complex)
    coef, *_ = np.linalg.lstsq(Xc, Zc[sel], rcond=None)
    out(f"   {n_on}   : {sel.sum() / evsel.sum():.3f}   {np.median((x[sel] - Soldp[sel]) / M[sel]):.3f}                        "
        f"{np.median((M[sel] - x[sel]) / M[sel]):.3f}                 {np.median(thmin[sel] / M[sel]):.3f}                      "
        f"alpha {abs(coef[0]):.3f}, Re(gamma/alpha) = {(coef[2] / coef[0]).real:+.3f}")
out("  (gamma = alpha means: the clock coordinate still carries the OLD gap M - Sold, i.e. the block did not register the new runner-up)")
out("  lowest threshold against the current runner-up, all states t = 2..8 with M > 0.3 and a1 >= 1:  quantiles of (min_j theta_j(t+1) - S_t+)/M")
sel = base & (tt >= 2) & (a1 >= 1)
q = (th[:, 1:].min(2)[sel] - Sp[sel]) / M[sel]
out("    5%, 25%, 50%, 75%, 95%:", np.percentile(q, [5, 25, 50, 75, 95]), "  P(min theta > S+):", (q > 0).mean())
for a in range(1, 9):
    sa = sel & (a1 == a)
    qa = (th[:, 1:].min(2)[sa] - Sp[sa]) / M[sa]
    out(f"    a1 = {a}: median {np.median(qa):+.3f}, IQR {np.percentile(qa, [25, 75])}, P(min theta > S+) = {(qa > 0).mean():.3f}")
out("  => a later input x with S < x < min_j theta_j would be a new runner-up that switches no comparator off.")
# probability that the NEXT input, if it becomes the new runner-up, is invisible to the block
nxt_sec = np.zeros_like(x, dtype=bool); nxt_sec[:, :-1] = newsec[:, 1:]
nxt_all_on = np.zeros_like(x, dtype=bool); nxt_all_on[:, :-1] = non[:, 1:] == 4
for a in range(0, 8):
    sa = (tt <= 7) & (tt >= 1) & (M > 0.3) & (a1 == a) & nxt_sec
    out(f"    age of max at t = {a}: P(all four stay on | x_(t+1) is a new runner-up) = {nxt_all_on[sa].mean():.3f}  (n = {int(sa.sum())})")

out("\n=== (e) does unit 2 hold only the running max?  h_2(t) ~ a M_(t-1) + b S_(t-1)+, states with a1(t) = age >= 1, M > 0.3 (seed 5015) ===")
h2 = H[:, :, 2]
for a in range(1, 9):
    sel = (a1 == a) & (M > 0.3)
    Sold = np.maximum(Sprev[sel], 0.0)                         # runner-up of x_0..x_{t-1}
    X = np.stack([M[sel], Sold], 1)
    b, *_ = np.linalg.lstsq(X, h2[sel], rcond=None)
    res = h2[sel] - X @ b
    r2n = 1 - ((res / M[sel]) ** 2).sum() / (((h2[sel] / M[sel]) - (h2[sel] / M[sel]).mean()) ** 2).sum()
    b1 = (h2[sel] * M[sel]).sum() / (M[sel] ** 2).sum()
    r1 = 1 - ((h2[sel] - b1 * M[sel]) ** 2).sum() / (h2[sel] ** 2).sum()
    out(f"  age {a}: h2 ~ {b[0]:.2f} M {b[1]:+.2f} S+  (b/a = {b[1] / b[0]:+.3f});  M alone: {b1:.2f} M, uncentred R2 {r1:.4f};  "
        f"std of h2/M = {(h2[sel] / M[sel]).std():.2f}; share of the variance of h2/M explained by S+/M: {r2n:.3f}")

# =============================================================================================== (c)
out("\n=== (c) one-step perturbation of the comparator thresholds in the real network (seed 5016) ===")
out("  at step t0 + 1 the four thresholds are shifted: theta -> theta + delta * pattern, delta = 0.05 * M_t0 (states with M_t0 > 0.3);")
out("  pattern 'common' = (1,1,1,1); 'difference' = a unit-4-like kick, w4*b4 minus its mean, rescaled to norm 2. Reported: |change of theta(t0+2)| etc.")
x = sample(400_000, 5016)
rs = running_stats(x)
M = rs["M1"]
H, Zp = run(x, return_pre=True)
th0, _ = thresholds(H)
non = (Zp[:, :, COMP] > 0).sum(2)
u4on = Zp[:, :, 4] > 0
kick = w[4] * b4 - (w[4] * b4).mean()
kick = kick / np.linalg.norm(kick) * 2
ans0 = logits(H).argmax(1)


def run_pert(t1, pattern, delta):
    """shift the comparator thresholds at step t1 by delta[:, None] * pattern (threshold units)."""
    B = x.shape[0]
    h = np.zeros((B, 16)); Hn = np.empty((B, 10, 16))
    for t in range(10):
        z = x[:, t, None] * w + h @ W.T
        if t == t1:
            z[:, COMP] += g * (delta[:, None] * pattern)
        h = np.maximum(z, 0.0); Hn[:, t] = h
    return Hn


for t0 in (3, 5):
    t1 = t0 + 1
    delta = 0.05 * M[:, t0]
    valid = M[:, t0] > 0.3
    for pname, pat in (("common", np.ones(4)), ("difference", kick)):
        for sign in (+1, -1):
            Hn = run_pert(t1, pat, sign * delta)
            thn, _ = thresholds(Hn)
            dth = (thn - th0) / (sign * delta[:, None, None])              # response per unit perturbation, (B, 11, 4)
            ansn = logits(Hn).argmax(1)
            for label, cond in (("all four on at t1, unit 4 off", (non[:, t1] == 4) & ~u4on[:, t1]),
                                ("1-3 on at t1", (non[:, t1] > 0) & (non[:, t1] < 4)), ("none on at t1", non[:, t1] == 0)):
                s_ = valid & cond
                norms = [np.sqrt((dth[s_, t1 + k_] ** 2).sum(1)).mean() / 2 for k_ in (1, 2, 3)]
                zc = [np.abs((dth[s_, t1 + k_] @ ell)).mean() for k_ in (1, 2, 3)]
                out(f"  t0 = {t0}, {pname:10s} {'+' if sign > 0 else '-'}0.05 M, {label:30s} (n = {int(s_.sum()):6d}): "
                    f"mean |d theta|/|pert| after 1, 2, 3 steps: {norms[0]:.3f}, {norms[1]:.3f}, {norms[2]:.3f};  "
                    f"mean |d z_c| per unit: {zc[0]:.3f}, {zc[1]:.3f}, {zc[2]:.3f};  answer changed: {np.mean(ansn[s_] != ans0[s_]):.4f}")
    out(f"  (Bc alone predicts for 'common': |Bc 1|/2 = {np.linalg.norm(Bc @ np.ones(4)) / 2:.4f}; for 'difference': |Bc kick|/2 = {np.linalg.norm(Bc @ kick) / 2:.3f}, "
        f"|Bc^2 kick|/2 = {np.linalg.norm(Bc @ Bc @ kick) / 2:.3f}, |Bc^3 kick|/2 = {np.linalg.norm(Bc @ Bc @ Bc @ kick) / 2:.3f})")

# =============================================================================================== (d)
out("\n=== (d) closed-core hybrids: accuracy by gaps (seed 5017) ===")
x = sample(400_000, 5017)
y = second_argmax(x)
xs = np.sort(x, 1)
gapMS = (xs[:, -1] - xs[:, -2]) / np.abs(xs[:, -1])
gapST = (xs[:, -2] - xs[:, -3]) / np.abs(xs[:, -2])
r = np.real(RIGHT["level"]); r_of = {u: r[ORD.index(u)] for u in ORD}
add = {j: sum(W[j, u] * r_of[u] for u in WAVE_NO2) / r_of[2] for j in CORE}
WF = W.copy(); WF[np.ix_(CORE, OTHER10)] = 0
WG = WF.copy()
for j in CORE:
    WG[j, 2] += add[j]
Bw = W[np.ix_(COMP, WAVE_NO2)] / g[:, None]
WP1 = W.copy(); WP1[np.ix_(COMP, WAVE_NO2)] = Bw.mean(0, keepdims=True) * g[:, None]
WP3 = W.copy(); WP3[COMP, 2] = b2.mean() * g
pos = (xs[:, -2] > 0.3)
out("  rows: network; columns: all | S > 0.3 & both relative gaps > 0.2 ('easy') | easy & gaps > 0.4 | (M-S)/M <= 0.1 | (S-T)/S <= 0.1 (S > 0.3)")
for name, Wm in (("intact", W), ("F closed core (plain)", WF), ("G closed core (compensated)", WG),
                 ("P1 wave->comp level-only", WP1), ("P3 uniform b2", WP3)):
    ans = logits(run(x, W=Wm)).argmax(1)
    ok = ans == y
    easy = pos & (gapMS > 0.2) & (gapST > 0.2)
    easy2 = pos & (gapMS > 0.4) & (gapST > 0.4)
    out(f"  {name:28s}: {ok.mean():.4f} | {ok[easy].mean():.4f} (n = {int(easy.sum())}) | {ok[easy2].mean():.4f} (n = {int(easy2.sum())}) | "
        f"{ok[pos & (gapMS <= 0.1)].mean():.4f} | {ok[pos & (gapST <= 0.1)].mean():.4f}")
