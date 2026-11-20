"""E17 (intervention). If the comparators hold s - m (the vote that hides the max), then wiping them once
at a step where no record occurs should make the network treat the standing max as the answer.
At step t0 we set h_t0[1,6,7,8] = 0 (everything else untouched) and continue with the real inputs.
Control: set four wave units to 0 instead.
The wipe step is t0 = pM + k, with pM the position of the final max and k = 1..9 its age at the wipe, so
t0 runs up to 9.  A wipe at the last step is a direct ablation of the readout rather than a test of what
the comparators carry forward, so the results are given separately for t0 <= 7, t0 = 8 and t0 = 9."""
from common import *

x = sample(300_000, 1717)
st = running_stats(x)
truth = second_argmax(x)
pM_final = x.argmax(1)

def run_wipe(x, t0, units):
    B = len(x); h = np.zeros((B, N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        if units is not None:
            sel = (t0 == t)
            h[np.ix_(np.where(sel)[0], units)] = 0.0
    return (h @ U.T).argmax(1)

base = run_wipe(x, np.full(len(x), -1), None)
print(f"baseline: answer = runner-up {np.mean(base == truth):.4f}, answer = max {np.mean(base == pM_final):.4f}")
print("wipe at step t0 = pM + k (the final max is at pM, k = 1..9 is its age at the wipe; all sequences with pM <= 9 - k, i.e. t0 <= 9, and S_final > 0.2, max - runner-up > 0.2)")
keep = (st["M2"][:, 9] > 0.2) & (st["M1"][:, 9] - st["M2"][:, 9] > 0.2)
rows = []       # one row per (sequence, k): k, t0, answer after the wipe = max / runner-up / the wipe step, control answer = max / runner-up
for k in range(1, 10):
    t0 = pM_final + k
    ok = (t0 <= 9) & keep
    a = run_wipe(x[ok], t0[ok], COMP)
    c = run_wipe(x[ok], t0[ok], [10, 11, 12, 15])
    tm, tr, tt = pM_final[ok], truth[ok], t0[ok]
    rows.append(np.column_stack([np.full(ok.sum(), k), tt, a == tm, a == tr, a == tt, c == tm, c == tr]))
K, T0, amax, arun, at0, cmax, crun = np.concatenate(rows).T
awipe = at0 * (1 - arun)        # answer = the wipe step and the runner-up is elsewhere, so that max / runner-up / wipe step / other are exclusive

print("\nP(answer = max) after wiping the comparators, by lag k (rows) and wipe step t0 (columns):")
print("     k | " + "  ".join(f"t0={t} " for t in range(1, 10)))
for k in range(1, 10):
    print(f"     {k} | " + "  ".join(f"{amax[(K == k) & (T0 == t)].mean():.3f}" if t >= k else "  .  " for t in range(1, 10)))

print("\nBelow, 'the wipe step t0' leaves out the sequences whose runner-up is itself the input of step t0 (they count as runner-up), so the")
print("first four columns add up to 1; the column in brackets is the plain event answer = t0, with those sequences included.")
for label, q in (("t0 <= 7", T0 <= 7), ("t0 = 8", T0 == 8), ("t0 = 9 (the last step)", T0 == 9), ("t0 <= 8 (every step but the last)", T0 <= 8),
                 ("t0 <= 9 (mixes the regimes above; the pooling of the first version of this script)", T0 <= 9)):
    print(f"\npooled over wipe steps {label}:")
    print("     k | P(answer = max)  P(runner-up)  P(the wipe step t0)  P(other)  [P(answer = t0)] |  control (wave units 10,11,12,15 wiped): P(max)  P(runner-up) |       n")
    for k in [*range(1, 10), "all"]:
        s = q & (K == k) if k != "all" else q
        if s.any():
            print(f"   {k:>3} |     {amax[s].mean():.3f}           {arun[s].mean():.3f}           {awipe[s].mean():.3f}            {1 - (amax + arun + awipe)[s].mean():.3f}"
                  f"        [{at0[s].mean():.3f}]      |                                           {cmax[s].mean():.3f}   {crun[s].mean():.3f}        | {s.sum():7d}")
