"""V1. Checks of claims 1, 2 and 9 (identity, kinds of write, leave-out by kind, pulse-only fits, wave clips).
Own projector, other seeds.   Run:  python -B v1_kinds.py > v1_kinds.txt"""
from vcommon import *
import tape_common as tc                     # only to compare definitions

print("(0) DEFINITIONS: my projector against tape_common.py")
print(f"    max |P - P_TAPE| = {np.abs(P - tc.P_TAPE).max():.1e};  rank {np.linalg.matrix_rank(P)};  |P P - P| = {np.abs(P @ P - P).max():.1e};  |P A - A P| = {np.abs(P @ A - A @ P).max():.1e}")

B = 300_000
x = sample(B, 7101)
H, Z, Amp = amplitudes(x)
print(f"\n(a) EXACTNESS, {B} N(0,1) sequences (seed 7101)")
for t in (3, 6, 9):
    mine = rebuild(Amp, t)
    print(f"    t = {t}: max |sum of writes - P h_t[o]| = {np.abs(mine - H[:, t][:, ORD] @ P.T).max():.1e};  against tape_common.tape_part: {np.abs(mine - tc.tape_part(H[:, t])[:, ORD]).max():.1e}")
st = top3(x, 9); q = st["quiet"]; M = st["M"]; h9 = H[:, 9]; ans0 = answer(h9); corr = second_argmax(x)
print(f"    quiet final states {q.sum()} ({q.mean():.3f}); accuracy there {np.mean(ans0[q] == corr[q]):.4f}; tc.prefix_stats agrees: {(tc.prefix_stats(x, 9)['quiet'] == q).all()}")


def kinds_table(x, H, Amp, t, label):
    """Std ratio, covariance share and answers changed per kind of write; state at step t read out directly with U
    (zero-prefix identity), tape scores of ages >= 2 = indices 9-t .. 7."""
    st = top3(x, t); q = st["quiet"]; M = st["M"][q, None]; h = H[:, t]; a0 = answer(h)[q]
    idx = np.arange(9 - t, 8)
    whole = rebuild(Amp[q], t) / M @ Uo[idx].T
    dele = np.mean(answer(with_tape(h[q], 0 * h[q][:, ORD])) != a0)
    row = f"    {label:22s} n = {q.sum():7d}  deleted {dele:.3f} |"
    for name, chs in KINDS.items():
        m = np.zeros(NCH); m[chs] = 1
        part = rebuild(Amp[q], t, m)
        sh, sr = cov_share(part / M @ Uo[idx].T, whole)
        ch = np.mean(answer(with_tape(h[q], rebuild(Amp[q], t, 1 - m))) != a0)
        row += f"  {name}: {sr:.2f} / {sh:+.2f} / {ch:.3f} |"
    print(row)


print("\n(b, c) PER KIND OF WRITE: std ratio / covariance share in the tape scores of ages >= 2 / share of answers changed when dropped")
kinds_table(x, H, Amp, 9, "N(0,1), t = 9")
for t in (4, 6, 8):
    kinds_table(x, H, Amp, t, f"N(0,1), t = {t} (read by U)")
xu = sample(B, 7102, "uniform"); Hu, Zu, Ampu = amplitudes(xu)
kinds_table(xu, Hu, Ampu, 9, "Unif(-1,1), t = 9")

print("\n    more leave-outs at t = 9 (N(0,1)): share of answers changed on quiet final states")
u0 = 4 + ORD.index(0)
for name, chs in {"clip of unit 0": [u0], "wave clips except unit 0": [c for c in WCH if c != u0], "pulses 4, 5, 9": [1, 2, 3]}.items():
    m = np.ones(NCH); m[chs] = 0
    print(f"      drop {name:26s} {np.mean(answer(with_tape(h9[q], rebuild(Amp[q], 9, m))) != ans0[q]):.4f}")

print("\n(c2) ATTACK: is 'necessary' more than 'a large cancelling term'?  Self-consistent removal: the network is re-run and at every step")
print("     the tape-mode part of one kind of write is taken out of the state, so later amplitudes respond. Quiet final states of the ORIGINAL run.")
print("     neg = share of final states with an activation below -1e-9 on the way; 'clip' = the same with the state clipped at 0 after each removal")
Dp = D @ P.T                                                       # tape part of a unit write per channel
def rerun_without(chs, clip):
    h = np.zeros((B, 16)); neg = np.zeros(B, bool)
    for s in range(10):
        z = x[:, s, None] * w + h @ W.T
        amp = np.zeros((B, NCH)); amp[:, 0] = x[:, s]; amp[:, 1:4] = h[:, PULSE]; amp[:, 4:] = np.maximum(-z[:, ORD], 0)
        h = np.maximum(z, 0)
        h[:, ORD] -= amp[:, chs] @ Dp[chs]
        neg |= (h < -1e-9).any(1)
        if clip: h = np.maximum(h, 0)
    return answer(h), neg
for name, chs in KINDS.items():
    a1, neg = rerun_without(chs, False); a2, _ = rerun_without(chs, True)
    print(f"      {name:16s} changed {np.mean(a1[q] != ans0[q]):.4f}   neg {neg[q].mean():.3f}   clipped: changed {np.mean(a2[q] != ans0[q]):.4f}")

print("\n(e) PULSES ONLY: R2 of the tape score entry of age 2..9 (quiet final states, / M); second half of the sample is the test set for the fitted kernel")
sq = rebuild(Amp[q], 9) / M[q, None] @ Uo.T
m = np.zeros(NCH); m[[1, 2, 3]] = 1
sp = rebuild(Amp[q], 9, m) / M[q, None] @ Uo.T
print("    exact kernel:   " + " ".join(f"{r2(sq[:, 9 - a], sp[:, 9 - a]):6.3f}" for a in range(2, 10)))
Xp = Amp[q][:, :, 1:4].reshape(q.sum(), -1) / M[q, None]; n2 = q.sum() // 2
b = np.linalg.lstsq(Xp[:n2], sq[:n2], rcond=None)[0]
print("    fitted kernel:  " + " ".join(f"{r2(sq[n2:, 9 - a], (Xp[n2:] @ b)[:, 9 - a]):6.3f}" for a in range(2, 10)) + "   (held out)")
m = np.zeros(NCH); m[[1, 2, 3] + CCH] = 1
sp = rebuild(Amp[q], 9, m) / M[q, None] @ Uo.T
print("    pulses + comparators, exact kernel: " + " ".join(f"{r2(sq[:, 9 - a], sp[:, 9 - a]):6.3f}" for a in range(2, 10)))

print("\n(f) WAVE CLIPS by steps since the latest positive new maximum (tau), steps 1..9: share of steps with the unit clipped")
vals, ages = prefix_before(x); kind = arrival_kind(x, vals)
tau = np.full((B, 10), -1); last = np.full(B, -1)
for s in range(10):
    last = np.where(kind[:, s] == 1, s, last); tau[:, s] = np.where(last >= 0, s - last, -1)
tau[:, 0] = -1
print("    unit  tau = 0     1     2     3     4     5     6")
for u in WAVE:
    a = Amp[:, :, 4 + ORD.index(u)] > 0
    print(f"    {u:4d}     " + " ".join(f"{a[tau == tv].mean():5.2f}" for tv in range(7)))
a0_ = Amp[:, :, u0]; sel = tau == 2
print(f"    clip of unit 0 at tau = 2, in units of the running maximum: median {np.median((a0_ / np.maximum.accumulate(x, 1))[sel]):.2f}, quartiles "
      f"{np.percentile((a0_ / np.maximum.accumulate(x, 1))[sel], 25):.2f} - {np.percentile((a0_ / np.maximum.accumulate(x, 1))[sel], 75):.2f}")
