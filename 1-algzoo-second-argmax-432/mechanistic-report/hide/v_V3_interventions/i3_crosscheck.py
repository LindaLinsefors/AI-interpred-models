"""Cross-check of run_iv against an independent path through common.run(x, h0=...), and of my I2 selection
against the same selection written with common.running_stats.  Seed 5950."""
from v3lib import *

out = Tee("i3_crosscheck.txt")
x = sample(400_000, 5950)

# I1, k = 2, t0 = 5 (pM = 3): run to step 5, zero the comparators, continue from that state with common.run
ok, pM, pS, M, S = select_I1(x)
m = ok & (pM == 3)
xs = x[m]
h5 = run(xs[:, :6])[:, -1].copy()
h5[:, COMP] = 0.0
a_ref = (run(xs[:, 6:], h0=h5)[:, -1] @ U.T).argmax(1)
a_mine = answer(run_iv(xs, np.full(len(xs), 5), scale_units(COMP, 0.0)))
out(f"I1 k=2, t0=5: n = {len(xs)}; answers identical: {np.array_equal(a_ref, a_mine)}; P(ans = pM) = {np.mean(a_ref == 3):.4f}")

# I2, t_r = 6: clock x 0 and rotations k = +-1 at step 5
ok, tr, p0, q0, M0, S0 = select_I2(x)
m = ok & (tr == 6)
xs, p0s = x[m], p0[m]
h5 = run(xs[:, :6])[:, -1].copy()
for f, k in ((0.0, 0.0), (1.0, 1.0), (1.0, -1.0)):
    h = h5.copy()
    z = h[:, ORD] @ LEFT["clock"]
    z2 = f * z * np.exp(1j * k * np.angle(LAM["clock"]))
    h[:, ORD] += 2 * np.real(np.multiply.outer(z2 - z, RIGHT["clock"]))
    a_ref = (run(xs[:, 6:], h0=h)[:, -1] @ U.T).argmax(1)
    a_mine = answer(run_iv(xs, np.full(len(xs), 5), mode_fn("clock", f=f, k=k)))
    out(f"I2 t_r=6, f={f}, k={k:+}: n = {len(xs)}; answers identical: {np.array_equal(a_ref, a_mine)}; P(p0) = {np.mean(a_ref == p0s):.4f}; "
        f"P(p0-1) = {np.mean(a_ref == p0s - 1):.4f}; P(p0+1) = {np.mean(a_ref == p0s + 1):.4f}; P(t_r) = {np.mean(a_ref == 6):.4f}")

# the I2 selection written with common.running_stats
st = running_stats(x)
ar = np.arange(len(x))
trr = x.argmax(1)
okk = trr >= 3
p0r = np.where(okk, st["p1"][ar, np.maximum(trr - 1, 0)], -1)
q0r = np.where(okk, st["p2"][ar, np.maximum(trr - 1, 0)], -1)
M0r = st["M1"][ar, np.maximum(trr - 1, 0)]; S0r = st["M2"][ar, np.maximum(trr - 1, 0)]
okk &= (trr - 1 - p0r >= 2) & (S0r > 0.2) & (M0r - S0r > 0.1) & (second_argmax(x) == p0r) & (st["p3"][:, 9] == q0r) & (trr <= 8)
out(f"I2 selection: mine {ok.sum()}, via running_stats {okk.sum()}, identical masks: {np.array_equal(ok, okk)}; "
    f"p0 identical on the selection: {np.array_equal(p0[ok], p0r[ok])}; q0 identical: {np.array_equal(q0[ok], q0r[ok])}")
