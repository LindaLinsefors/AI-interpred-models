"""C2b. How much of the old clock survives a record: exact Jacobians through the actual gates, an intervention
(zero the clock at s-1 and rerun), and the alpha-budget of the first two steps.  400k N(0,1) sequences, seed 1201."""
from cl import *

x, H, Z, st = gen(400_000, 1201)
tm = clock_terms(x, H, Z)
z = tm["z"] / K4
M, S, Tq, a1, a2 = st["M1"], pos(st["M2"]), pos(st["M3"]), st["a1"], st["a2"]
G = (Z > 0).astype(float)

B_, s_ = np.nonzero((a1[:, 1:9] == 0) & (M[:, 0:8] > 0)); s_ = s_ + 1
quiet = (a1[B_, s_ + 1] == 1) & (a2[B_, s_ + 1] >= 2)
aold = a1[B_, s_ - 1]
print(f"records s=1..8 over a positive old max: {B_.size}; with a quiet next step: {quiet.sum()}")

print("\n(a) comparators still ON at the record step, by age of the old max (share of records)")
for a in range(0, 8):
    q = aold == a
    print(f"   a_old={a}: units 1,6,7,8 on: {np.round(G[B_, s_][q][:, COMP].mean(0), 4)}   any on: {G[B_, s_][q][:, COMP].any(1).mean():.4f}   n={q.sum()}")

# ---------------------------------------------------------------- (b) exact Jacobian d z(s+1) / d z(s-1) through the actual gates
v1 = W[:, ORD] @ RC                    # effect of a unit clock coordinate (h[ORD] += 2 Re(dz RC)) on the pre-activations at s
v1c = W[:, ORD] @ np.conj(RC)
Gs, Gs1 = G[B_, s_], G[B_, s_ + 1]
def jac(v, mask=None):
    g = Gs if mask is None else Gs * mask
    return (((g * v) @ W.T) * Gs1)[:, ORD] @ LC
only4 = np.zeros(16); only4[4] = 1
not4 = 1 - only4
a_all, b_all = jac(v1), jac(v1c)
a_4, b_4 = jac(v1, only4), jac(v1c, only4)
a_r, b_r = jac(v1, not4), jac(v1c, not4)
print("\n(b) [EXACT per sample, averaged] dz(s+1) = a dz(s-1) + b conj(dz(s-1)) with the gates of steps s, s+1 held fixed")
print(f"   W[4,ORD].RC = {fmt(v1[4], 4)}  (the unit-4 path: a_4 = W[4,ORD].RC, b_4 = conj(a_4) when no clipping at s+1)")
print(f"   free-flow value for 'comparators off at s, everything on at s+1, unit 4 excluded': lam^2 s_clock = {fmt(LAMC ** 2 * (LC[wi] @ RC[wi]), 4)}")
for label, q in (("all records", np.ones(B_.size, bool)), ("quiet next step", quiet)):
    print(f"   {label}: mean a = {fmt(a_all[q].mean(), 4)}  mean b = {fmt(b_all[q].mean(), 4)} | via unit 4: a = {fmt(a_4[q].mean(), 4)} b = {fmt(b_4[q].mean(), 4)}"
          f" | all other paths: a = {fmt(a_r[q].mean(), 4)} b = {fmt(b_r[q].mean(), 4)}   (sd of |a_rest| {abs(a_r[q]).std():.3f})")
print("   by age of old max (quiet next step): mean a_rest, mean b_rest, and the net factor on the old clock assuming z(s-1) ~ r e^{i phi}: |a + b e^{-2 i phi}|")
for a in range(0, 8):
    q = quiet & (aold == a)
    ph = np.angle(z[B_, s_ - 1][q])
    net_r = a_r[q] + b_r[q] * np.exp(-2j * ph)
    net = a_all[q] + b_all[q] * np.exp(-2j * ph)
    print(f"     a_old={a}: a_rest {fmt(a_r[q].mean())}  b_rest {fmt(b_r[q].mean())}  mean net (rest) {fmt(net_r.mean())}   mean net (incl. unit 4) {fmt(net.mean())}   n={q.sum()}")

# ---------------------------------------------------------------- (c) intervention: remove the clock at s-1, rerun steps s and s+1
h0 = H[B_, s_ - 1].copy()
zc = h0[:, ORD] @ LC
h0[:, ORD] -= 2 * np.real(np.multiply.outer(zc, RC))
neg = (h0[:, ORD] < -1e-9).any(1).mean()
Hi = run(np.stack([x[B_, s_], x[B_, s_ + 1]], 1), h0=h0)
zi = Hi[..., ORD] @ LC / K4
dz1 = z[B_, s_ + 1] - zi[:, 1]
zp = z[B_, s_ - 1]
print(f"\n(c) [INTERVENTION] clock coordinate set to 0 at s-1 (other modes and pulses unchanged), steps s, s+1 rerun with the real inputs. n = {B_.size}")
print(f"   check: clock coordinate of the edited state max |.| = {abs(h0[:, ORD] @ LC).max():.2e};  edited states with a negative ordinary unit: {neg:.3f}")
dh4 = (H[B_, s_, 4] - Hi[:, 0, 4]) / K4
for label, q in (("all records", np.ones(B_.size, bool)), ("quiet next step", quiet)):
    c, e = cfit(dz1[q], [zp[q], np.conj(zp[q])])
    c4, e4 = cfit(dh4[q].astype(complex), [zp[q], np.conj(zp[q])])
    cr, er = cfit((dz1 - dh4)[q], [zp[q], np.conj(zp[q])])
    print(f"   {label}: z(s+1) - z_edited(s+1) ~ a z(s-1) + b conj z(s-1): a = {fmt(c[0], 4)}  b = {fmt(c[1], 4)}  EV {e:.4f}")
    print(f"        change of the kick h4(s)/K4 ~ a z + b conj z: a = {fmt(c4[0], 4)} b = {fmt(c4[1], 4)} EV {e4:.4f};   everything else: a = {fmt(cr[0], 4)}  b = {fmt(cr[1], 4)}  EV {er:.4f}")
    print(f"        rms of the change in z(s+1): {np.sqrt(np.mean(abs(dz1[q]) ** 2)):.4f}; rms z(s-1): {np.sqrt(np.mean(abs(zp[q]) ** 2)):.4f}; rms z(s+1): {np.sqrt(np.mean(abs(z[B_, s_ + 1][q]) ** 2)):.4f}"
          f"   ratio rms(change)/rms(z(s-1)) = {np.sqrt(np.mean(abs(dz1[q]) ** 2) / np.mean(abs(zp[q]) ** 2)):.3f}")
    rec_kept = (Hi[:, 0, 4] > 0)[q].mean()
    print(f"        unit 4 still fires at s in the edited run: {rec_kept:.5f}")
print("   by age of the old max (quiet): |change in z(s+1)| / |z(s-1)| median, and phase of change relative to z(s-1) (median)")
for a in range(1, 8):
    q = quiet & (aold == a) & (abs(zp) > 0.05)
    rat = dz1[q] / zp[q]
    print(f"     a_old={a}: median |ratio| {np.median(abs(rat)):.3f}  median phase {np.degrees(np.angle(np.median(rat.real) + 1j * np.median(rat.imag))):+.1f}"
          f"   of which kick change/|z(s-1)| median {np.median(dh4[q] / abs(zp[q])):+.3f}   n={q.sum()}")

# ---------------------------------------------------------------- (d) alpha-budget for a1 = 1 and a1 = 2
print("\n(d) [FIT] budget of the (M - S+) coefficient: each exact term of z(t) regressed on (M - S+, S+, T+) of step t (complex LS, no intercept)")
names = ["carried", "p4", "p5", "p9", "echo", "clip1", "clip6", "clip7", "clip8", "clipwave"]
t_idx = np.broadcast_to(np.arange(T), x.shape)
ok = M > 0
for A, cond_label, cond in ((1, "a1 = 1, a2 >= 2", (a1 == 1) & (a2 >= 2) & ok), (2, "a1 = 2, a2 >= 3", (a1 == 2) & (a2 >= 3) & ok), (3, "a1 = 3, a2 >= 4", (a1 == 3) & (a2 >= 4) & ok)):
    bb, tt = np.nonzero(cond)
    cols = [(M - S)[bb, tt], S[bb, tt], Tq[bb, tt]]
    ctot, etot = cfit(z[bb, tt], cols)
    print(f"   --- {cond_label}: n = {bb.size};  z ~ alpha G + beta S + gamma T: alpha {fmt(ctot[0])} beta {fmt(ctot[1])} gamma {fmt(ctot[2])} EV {etot:.4f}")
    acc = 0
    for k in names:
        v = tm[k][bb, tt - 1] / K4
        c, e = cfit(v, cols)
        acc = acc + c
        if abs(c).max() > 2e-3:
            print(f"       {k:9s} alpha-part {fmt(c[0], 4)}   beta-part {fmt(c[1], 4)}   gamma-part {fmt(c[2], 4)}")
    print(f"       sum       alpha-part {fmt(acc[0], 4)}   beta-part {fmt(acc[1], 4)}   gamma-part {fmt(acc[2], 4)}")
    for p in (5, 9):
        v = H[bb, tt - 1, p] / K4
        c, e = cfit(v.astype(complex), cols)
        print(f"       activation h{p}(t-1)/K4 ~ {c[0].real:+.3f} (M-S) {c[1].real:+.3f} S {c[2].real:+.3f} T   (EV {e:.3f}; share on {np.mean(v > 0):.3f})")
