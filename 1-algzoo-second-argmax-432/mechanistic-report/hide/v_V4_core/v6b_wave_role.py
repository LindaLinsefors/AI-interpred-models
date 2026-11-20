"""V4 verifier, attack 2b: WHAT do the other wave units feed into the comparators - a level, or differences?
(1) how well is the wave(no 2) part of each threshold predicted by unit 2 (a level proxy)?
(2) ablations that keep only the common-level part / only the difference part of the wave -> comparator weights;
(3) the sources of the clock coordinate z_c = ell . theta(t+1) by age of the max;
(4) what the closed-core hybrid answers.     Seed 5009, n = 400000."""
from vsetup import *

out = Tee("v6b_wave_role.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=220)

x = sample(400_000, 5009)
y = second_argmax(x)
order = np.argsort(-x, axis=1)
pmax, pthird = order[:, 0], order[:, 2]
rs = running_stats(x)
M, S, a1, a2 = rs["M1"], rs["M2"], rs["a1"], rs["a2"]
tt = np.broadcast_to(np.arange(10), M.shape)
H0 = run(x)
ans0 = logits(H0).argmax(1)
THR = COMP + [4]
GAUGE = np.concatenate([g, [-w[4]]])

# ------------------------------------------------------------------ (1) is the wave part a level?
out("=== (1) wave(no 2) part of theta_row(t+1) [rows 1, 6, 7, 8, unit 4], t = 1..8, M_t > 0.3 ===")
wave_part = (H0[:, :, WAVE_NO2] @ W[np.ix_(THR, WAVE_NO2)].T) / GAUGE       # (B, T, 5)
h2 = H0[:, :, 2]
s = (tt >= 1) & (tt <= 8) & (M > 0.3)
cfit = []
for i, row in enumerate(THR):
    p, q = wave_part[s][:, i], h2[s]
    c = (p * q).sum() / (q * q).sum()
    r2c = 1 - ((p - c * q) ** 2).sum() / ((p - p.mean()) ** 2).sum()
    r2u = 1 - ((p - c * q) ** 2).sum() / (p ** 2).sum()
    # normalised by M: how much of the variation of part/M is explained by h2/M
    pn, qn = p / M[s], q / M[s]
    cc = np.corrcoef(pn, qn)[0, 1]
    cfit.append(c * GAUGE[i])                                             # back to raw weight units
    out(f"  row {row}: best c with part ~ c*h2: {c:.5f} (x17.3 = {c * 17.3:.4f} M); centred R2 = {r2c:.4f}, uncentred R2 = {r2u:.4f}; "
        f"corr(part/M, h2/M) = {cc:+.3f}; mean part/M = {pn.mean():.4f}, std part/M = {pn.std():.4f}")
cfit = np.array(cfit)
out("  best-fit extra weight on W[row, 2] (raw units):", cfit, " (K5 compensation: 0.0853, 0.0999, 0.1088, 0.191, -0.128)")
out("  mean of wave(no 2) part / M by age of the max a1 = 0..8 (rows), columns rows 1, 6, 7, 8, unit 4:")
for a in range(9):
    sa = (tt <= 8) & (a1 == a) & (M > 0.3)
    out(f"    a1 = {a}: {(wave_part[sa] / M[sa][:, None]).mean(0)}   std {(wave_part[sa] / M[sa][:, None]).std(0)}")

# ------------------------------------------------------------------ (2) level-only / difference-only ablations


def evaluate(name, Wm):
    H = run(x, W=Wm)
    ans = logits(H).argmax(1)
    out(f"  {name:78s} acc {np.mean(ans == y):.4f}   same as intact {np.mean(ans == ans0):.4f}")
    return ans


out("\n=== (2) ablations of the wave(no 2) -> comparator weights in the real network (intact accuracy first) ===")
evaluate("intact", W)
Wm = W.copy(); Wm[np.ix_(THR, WAVE_NO2)] = 0
evaluate("A   zero wave(no 2) -> comparators + unit 4", Wm)
Wm = W.copy(); Wm[np.ix_(THR, WAVE_NO2)] = 0; Wm[THR, 2] += cfit
evaluate("A-fit  A + best-fit level replacement on W[row,2] (least squares on real data)", Wm)
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = 0; Wm[COMP, 2] += cfit[:4]
evaluate("A'-fit same for the four comparators only (unit 4 untouched)", Wm)
# gauge picture: column of thresholds each wave unit writes: W[COMP,u]/g
Bw = W[np.ix_(COMP, WAVE_NO2)] / g[:, None]                               # (4, 8)
lvl = Bw.mean(0, keepdims=True)                                           # common level each wave unit writes
out("  wave -> comparator weights in threshold gauge, W[j,u]/g_j (rows j = 1,6,7,8; columns u =", WAVE_NO2, "):\n", Bw)
out("  common-level part (mean over j):", lvl[0])
out("  difference part norms per unit:", np.linalg.norm(Bw - lvl, axis=0), "  level part norms (|mean|*2):", np.abs(lvl[0]) * 2)
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = (np.ones((4, 1)) * lvl) * g[:, None]
evaluate("P1  wave(no 2) -> comparators: keep ONLY the common-level part", Wm)
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = (Bw - lvl) * g[:, None]
evaluate("P2  wave(no 2) -> comparators: keep ONLY the difference part", Wm)
Wm = W.copy(); Wm[COMP, 2] = b2.mean() * g
evaluate("P3  unit 2 -> comparators made exactly uniform (b2 -> mean b2)", Wm)
Wm = W.copy(); Wm[COMP, 2] = b2.mean() * g; Wm[4, 2] = -b2.mean() * w[4]
evaluate("P3' unit 2 -> comparators and unit 4 all with the same threshold weight mean(b2)", Wm)
# difference part split along the clock plane of Bc and the rest
ev, vecs = np.linalg.eig(Bc.T)
k = int(np.argmax(np.angle(ev) * (np.abs(ev) > 0.5)))
ell = vecs[:, k] / (vecs[:, k] @ (w[4] * b4))
evr, V = np.linalg.eig(Bc)
kr = int(np.argmax(np.angle(evr) * (np.abs(evr) > 0.5)))
vr = V[:, kr] / (ell @ V[:, kr])                                          # right eigenvector with ell . vr = 1
proj_clock = lambda th: 2 * np.real(np.multiply.outer(th @ ell, vr))      # component of thresholds in the clock plane (th: (..., 4))
Bw_clock = proj_clock(Bw.T).T                                              # (4, 8)
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = (Bw - Bw_clock) * g[:, None]
evaluate("P4  wave(no 2) -> comparators: REMOVE the component in the clock plane of Bc", Wm)
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = Bw_clock * g[:, None]
evaluate("P5  wave(no 2) -> comparators: keep ONLY the component in the clock plane of Bc", Wm)
out("  |ell . (W[COMP,u]/g)| per wave unit u (clock-coordinate kick per unit activation):", np.abs(ell @ Bw))

# ------------------------------------------------------------------ (3) sources of z_c by age
out("\n=== (3) sources of z_c = ell . theta(t+1): mean of z_c^source * exp(-i phase_a) / M_t by age (t <= 8, M > 0.3) ===")
out("   phase_a = angle of the componentwise median of total z_c / M at that age; each entry: real part (along the total), imaginary part")
src = {"comp": COMP, "unit2": [2], "unit4": [4], "wave_no2": WAVE_NO2, "pulse59": [5, 9]}
zs = {n_: ((H0[:, :, u] @ W[np.ix_(COMP, u)].T) / g) @ ell for n_, u in src.items()}
ztot = sum(zs.values())
out("    a :   total |median|   " + "   ".join(f"{n_:>16s}" for n_ in src))
for a in range(0, 9):
    sa = (tt <= 8) & (a1 == a) & (M > 0.3) & (tt >= 1)
    q = ztot[sa] / M[sa]
    med = np.median(q.real) + 1j * np.median(q.imag)
    rot = np.exp(-1j * np.angle(med))
    row = "   ".join(f"{(zs[n_][sa] / M[sa] * rot).mean().real:+7.3f} {(zs[n_][sa] / M[sa] * rot).mean().imag:+7.3f}i" for n_ in src)
    out(f"    {a} :   {abs(med):.3f} @ {np.degrees(np.angle(med)) % 360:5.1f}   {row}")
out("   same, mean ABSOLUTE size |z_c^source| / M:")
for a in range(0, 9):
    sa = (tt <= 8) & (a1 == a) & (M > 0.3) & (tt >= 1)
    out(f"    {a} :   total {np.abs(ztot[sa] / M[sa]).mean():.3f}   " + "   ".join(f"{n_} {np.abs(zs[n_][sa] / M[sa]).mean():.3f}" for n_ in src))

# ------------------------------------------------------------------ (4) what does the closed-core hybrid answer?
out("\n=== (4) answers of the hybrids (true answer = runner-up position) ===")
r = np.real(RIGHT["level"]); r_of = {u: r[ORD.index(u)] for u in ORD}
add = {j: sum(W[j, u] * r_of[u] for u in WAVE_NO2) / r_of[2] for j in CORE}
WF = W.copy(); WF[np.ix_(CORE, OTHER10)] = 0
WG = WF.copy()
for j in CORE:
    WG[j, 2] += add[j]
gapMS = (np.sort(x, 1)[:, -1] - np.sort(x, 1)[:, -2])
for name, Wm in (("intact", W), ("F closed core (plain)", WF), ("G closed core (compensated)", WG)):
    H = run(x, W=Wm)
    ans = logits(H).argmax(1)
    out(f"  {name:28s}: P(runner-up) {np.mean(ans == y):.4f}  P(max) {np.mean(ans == pmax):.4f}  P(third) {np.mean(ans == pthird):.4f}  "
        f"P(other) {np.mean((ans != y) & (ans != pmax) & (ans != pthird)):.4f};  accuracy by position of the true answer 0..9: "
        f"{np.array([np.mean(ans[y == p] == p) for p in range(10)])}")
    if Wm is not W:
        # check that the core of the hybrid equals the stand-alone six-unit simulation
        Wcc = Wm[np.ix_(CORE, CORE)]
        h = np.zeros((x.shape[0], 6)); err = 0.0
        for t in range(10):
            h = np.maximum(h @ Wcc.T + x[:, t, None] * w[CORE], 0.0)
            err = max(err, np.abs(h - H[:, t][:, CORE]).max())
        out(f"      max |core of hybrid - stand-alone six-unit simulation| = {err:.2e}")
