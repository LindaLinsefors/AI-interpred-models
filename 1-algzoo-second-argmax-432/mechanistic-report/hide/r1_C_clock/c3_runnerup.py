"""C3. Radius update of the clock when a new runner-up arrives (S_old < x_t < M, unit 4 silent).

A. exact gains of the four comparators.   B. N(0,1) data (400k sequences, seed 1300).   C. controlled inputs."""
from cl import *

# =============================================================== A
print("=== A. [EXACT] a comparator j clipped by an input v adds  LC[j] |w_j| (v - theta_j)  to the clock coordinate ===")
cj = {j: LC[ORD.index(j)] * abs(w[j]) / K4 for j in COMP}
for j in COMP:
    print(f"   unit {j}: LC[j] = {fmt(LC[ORD.index(j)])}   |w_j| = {abs(w[j]):.3f}   gain c_j = LC[j]|w_j|/K4 = {fmt(cj[j])}   RC[j] = {fmt(RC[ORD.index(j)])}   LC[j]RC[j] = {fmt(LC[ORD.index(j)] * RC[ORD.index(j)])}")
print(f"   sum_j LC[j] RC[j] over comparators = {fmt(sum(LC[ORD.index(j)] * RC[ORD.index(j)] for j in COMP))};   all four clipped: sum_j c_j = {fmt(sum(cj.values()))}  (= -(comparator part of ECHO)/K4; ECHO/K4 = {fmt(ECHO / K4, 4)})")
laws = np.load(HERE / "c1_laws.npz")
phi3 = np.degrees(laws["phi3"]) % 360
print("   direction needed to shrink the radius at max-age a (phi(a) + 180) and the angle of each comparator's gain from it:")
for a in range(1, 10):
    need = (phi3[a] + 180) % 360
    print(f"     a={a}: phi = {phi3[a]:6.1f}, need {wrap(need):+7.1f}:  " + "  ".join(f"u{j}: {wrap(deg(cj[j]) - need):+7.1f}" for j in COMP))

# =============================================================== B
x, H, Z, st = gen(400_000, 1300)
tm = clock_terms(x, H, Z)
z = tm["z"] / K4
M, S, Tq, a1, a2 = st["M1"], pos(st["M2"]), pos(st["M3"]), st["a1"], st["a2"]
G = Z > 0
th = thresholds(H)
alpha3 = laws["alpha3"]
print("\n=== B. [MEASURED n=400k sequences, seed 1300] events: S+_{t-1} < x_t < M_{t-1}, M_{t-1} > 0, t >= 1 ===")
bb, tt = np.nonzero((x[:, 1:] > S[:, :-1]) & (x[:, 1:] < M[:, :-1]) & (M[:, :-1] > 0)); tt = tt + 1
v, Mo, So, age = x[bb, tt], M[bb, tt - 1], S[bb, tt - 1], a1[bb, tt]
u = (v - So) / (Mo - So)
fire4 = G[bb, tt, 4]
print(f"   events: {bb.size};  unit 4 fires (false record): {fire4.mean():.4f};  no comparator clipped: {G[bb, tt][:, COMP].all(1).mean():.4f};  all four clipped: {(~G[bb, tt][:, COMP]).all(1).mean():.4f}")
print("   share of events with unit 4 firing, by age of max (rows) and u = (v - S)/(M - S) bin (columns)")
ub = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
print("        " + "".join(f"[{lo:.1f},{hi:.1f})  " for lo, hi in zip(ub[:-1], ub[1:])))
for a in range(1, 10):
    print(f"     a={a}: " + "   ".join(f"{fire4[(age == a) & (u >= lo) & (u < hi)].mean():8.4f}" for lo, hi in zip(ub[:-1], ub[1:])))
print("   clipped share of comparators (1,6,7,8) by age of the max at the event and u-bin  (entries: four shares)")
for a in range(1, 10):
    row = []
    for lo, hi in zip(ub[:-1], ub[1:]):
        q = (age == a) & (u >= lo) & (u < hi)
        row.append("/".join(f"{(1 - G[bb, tt][q][:, j].mean()):.2f}" for j in COMP))
    print(f"     a={a}: " + "   ".join(row))
print("   immediate update [FIT]: (LC.clip_t + ECHO x_t)/K4 ~ kappa (v - S_old) per age (events without unit 4); ideal would be kappa = -alpha(a)")
upd = sum(tm[k][bb, tt - 1] for k in ("clip1", "clip6", "clip7", "clip8", "clipwave", "echo")) / K4
for a in range(1, 10):
    q = (age == a) & ~fire4
    c, e = cfit(upd[q], [(v - So)[q]])
    rat = c[0] / (-alpha3[a])
    print(f"     a={a}: kappa = {fmt(c[0])}  EV {e:.3f} | -alpha(a) = {fmt(-alpha3[a])} | kappa/(-alpha) = {abs(rat):.3f}@{deg(rat):+6.1f} (radial part {rat.real:+.3f}, tangential {rat.imag:+.3f})   n={q.sum()}")

# delays on data: fit z(t+d) ~ alpha' (M - S_old) + k (v - S_old) + c S_old  on events with nothing above S_old arriving in t+1..t+d
print("   after d steps [FIT]: z(t+d) ~ A (M - S_old) + k_d (v - S_old) + c S_old, events without unit 4 and with no input above v in t+1..t+d;")
print("   'update fraction' = -k_d / A (1 = radius fully updated to M - v; complex: real part radial, imaginary part tangential)")
print("     a\\d " + "".join(f"{d:>26d}" for d in range(4)))
for a in range(1, 9):
    row = []
    for d in range(4):
        q = (age == a) & ~fire4 & (tt + d <= 9)
        bq, tq = bb[q], tt[q]
        good = np.ones(bq.size, bool)
        for k in range(1, d + 1):
            good &= x[bq, tq + k] < v[q]
        bq, tq = bq[good], tq[good]
        if bq.size < 2000:
            row.append(f"{'-':>26s}"); continue
        Mq, Sq, vq = Mo[q][good], So[q][good], v[q][good]
        c, e = cfit(z[bq, tq + d], [Mq - Sq, vq - Sq, Sq])
        f = -c[1] / c[0]
        row.append(f"   {f.real:+.2f}{f.imag:+.2f}i (EV {e:.3f})")
    print(f"     a={a} " + "".join(row))

# =============================================================== C
print("\n=== C. controlled inputs: M = 1.5 at p1, optional earlier S_old, input v at p2 = p1 + a, constant background b ===")
print("   R(d) = z_with_v(p2 + d) / z_without_v(p2 + d);  ideal: R = (M - v)/(M - S_old+) real.  u = (v - S_old+)/(M - S_old+),  u_eff(d) = 1 - |R(d)|.")
Mv = 1.5
US = np.array([-0.2, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
rows = []          # (kind, b, p1, a, u, d, R, fired4, clipped pattern)
seqs_base, seqs_ev, meta = [], [], []
for b in (-1.0, -0.3):
    for kind, Sold, off in (("none", 0.0, None), ("old max 0.5 one step before M", 0.5, -1), ("old max 1.0 one step before M", 1.0, -1),
                            ("old max 0.5 three steps before M", 0.5, -3), ("S_old 0.5 one step after M", 0.5, +1)):
        for p1 in range(0, 8):
            p0 = None if off is None else p1 + off
            if p0 is not None and (p0 < 0 or p0 > 9):
                continue
            base = np.full(10, b); base[p1] = Mv
            if p0 is not None:
                base[p0] = Sold
            for a in range(1, 10 - p1):
                p2 = p1 + a
                if p0 is not None and p2 <= p0:
                    continue
                if p0 is not None and p2 == p0:
                    continue
                for uu in US:
                    ev = base.copy(); ev[p2] = Sold + uu * (Mv - Sold)
                    seqs_base.append(base); seqs_ev.append(ev); meta.append((kind, b, p1, a, uu, p2))
Xb, Xe = np.array(seqs_base), np.array(seqs_ev)
Hb, He, Ze = run(Xb), *run(Xe, return_pre=True)
zb, ze = Hb[..., ORD] @ LC / K4, He[..., ORD] @ LC / K4
kinds = np.array([m[0] for m in meta]); bs = np.array([m[1] for m in meta]); p1s = np.array([m[2] for m in meta])
ages = np.array([m[3] for m in meta]); us = np.array([m[4] for m in meta]); p2s = np.array([m[5] for m in meta])
idx = np.arange(len(meta))
fired4 = Ze[idx, p2s, 4] > 0
clipped = Ze[idx, p2s][:, COMP] <= 0
print(f"   sequences: {len(meta)};  unit 4 fired at the event in {fired4.sum()} of them (all at u >= {us[fired4].min() if fired4.any() else np.nan});  by u: "
      + ", ".join(f"u={uu}: {fired4[us == uu].mean():.2f}" for uu in US))
# exact identity check: z_ev(p2) - z_base(p2) = sum_clipped c_j (v - theta_j) + ECHO (v - b)/K4 + wave clip difference
thb = thresholds(Hb)                                       # thresholds for the next step
vv = Xe[idx, p2s]
pred = np.zeros(len(meta), complex)
for k, j in enumerate(COMP):
    thj = thb[idx, p2s - 1, k]
    pred += np.where(vv > thj, cj[j] * (vv - thj), 0) - np.where(bs > thj, cj[j] * (bs - thj), 0)
pred += ECHO * (vv - bs) / K4
clipw_e = np.maximum(-Ze[idx, p2s][:, WAVE], 0) @ LC[wi] / K4
Zb = run(Xb, return_pre=True)[1]
clipw_b = np.maximum(-Zb[idx, p2s][:, WAVE], 0) @ LC[wi] / K4
err = abs(ze[idx, p2s] - zb[idx, p2s] - pred - (clipw_e - clipw_b))
print(f"   [EXACT] identity z_ev(p2) - z_base(p2) = sum_clipped c_j (v - theta_j) + ECHO (v - b) + wave-clip difference: max abs error {err.max():.2e}")
print(f"   size of the wave-clip difference relative to the comparator part (rms ratio): {np.sqrt(np.mean(abs(clipw_e - clipw_b) ** 2) / np.mean(abs(pred) ** 2)):.3f}")

R = {}
for d in range(4):
    okd = p2s + d <= 9
    r = np.full(len(meta), np.nan, complex)
    r[okd] = ze[idx[okd], p2s[okd] + d] / zb[idx[okd], p2s[okd] + d]
    R[d] = r
sel0 = ~fired4
print("\n   clipped comparators at the event (units 1,6,7,8; 1 = clipped), most common pattern per (age a, u); '*' = pattern varies across configurations")
print("     a\\u " + "".join(f"{uu:>7.2f}" for uu in US))
for a in range(1, 10):
    row = []
    for uu in US:
        q = (ages == a) & (us == uu) & sel0
        if q.sum() == 0:
            row.append(f"{'-':>7s}"); continue
        pats, cnt = np.unique(clipped[q].astype(int) @ np.array([1000, 100, 10, 1]), return_counts=True)
        row.append(f"{pats[cnt.argmax()]:04d}{'*' if pats.size > 1 else ' '}".rjust(7))
    print(f"     a={a} " + "".join(row))

for d in range(4):
    print(f"\n   delay d = {d}:  median over configurations of  u_eff - u  /  phase shift arg R (deg)   [rows: age a of the max at the event; unit-4 firings excluded]")
    print("     a\\u " + "".join(f"{uu:>14.2f}" for uu in US))
    for a in range(1, 10):
        row = []
        for uu in US:
            q = (ages == a) & (us == uu) & sel0 & np.isfinite(R[d])
            if q.sum() == 0:
                row.append(f"{'-':>14s}"); continue
            ue = 1 - abs(R[d][q]) - max(uu, 0)
            row.append(f"{np.median(ue):+.3f}/{np.median(deg(R[d][q])):+5.1f}".rjust(14))
        print(f"     a={a} " + "".join(row))
print("\n   summary over u in [0.1, 0.9], all configurations without unit-4 firing:")
print("   d |  mean |u_eff - u|   90th pct |u_eff - u|   LS slope of u_eff on u |  mean |phase shift| deg   90th pct   max |  n")
for d in range(4):
    q = sel0 & np.isfinite(R[d]) & (us >= 0.1) & (us <= 0.9)
    ue = 1 - abs(R[d][q]); uu = us[q]
    slope = np.sum(ue * uu) / np.sum(uu * uu)
    ph = abs(deg(R[d][q]))
    print(f"   {d} |  {np.mean(abs(ue - uu)):.3f}   {np.percentile(abs(ue - uu), 90):.3f}   {slope:.3f} |  {ph.mean():.1f}   {np.percentile(ph, 90):.1f}   {ph.max():.1f} |  {q.sum()}")
print("   same, split by configuration kind (d = 0 and d = 2): mean |u_eff - u|, mean |phase shift|")
for kind in np.unique(kinds):
    out = []
    for d in (0, 2):
        q = sel0 & np.isfinite(R[d]) & (us >= 0.1) & (us <= 0.9) & (kinds == kind)
        out.append(f"d={d}: {np.mean(abs(1 - abs(R[d][q]) - us[q])):.3f}, {np.mean(abs(deg(R[d][q]))):.1f} deg (n={q.sum()})")
    print(f"     {kind:34s} " + "   ".join(out))
print("   same, split by background b (d = 2):", {b: round(float(np.mean(abs(1 - abs(R[2][q]) - us[q]))), 3) for b in (-1.0, -0.3) for q in [sel0 & np.isfinite(R[2]) & (us >= 0.1) & (us <= 0.9) & (bs == b)]})
print("   inputs below S_old (u = -0.2): mean | |R| - 1 | and mean |phase shift| at d = 0, 1, 2:",
      [(round(float(np.mean(abs(abs(R[d][q]) - 1))), 4), round(float(np.mean(abs(deg(R[d][q])))), 2)) for d in range(3) for q in [np.isfinite(R[d]) & (us == -0.2)]])

# does the answer follow? final answer of the network on the event sequences
ans = logits(He).argmax(1)
true = second_argmax(Xe)
print(f"\n   final answers on the controlled event sequences: correct {np.mean(ans == true):.4f} (n = {len(meta)});  by u: " + ", ".join(f"{uu}: {np.mean((ans == true)[us == uu]):.2f}" for uu in US))
