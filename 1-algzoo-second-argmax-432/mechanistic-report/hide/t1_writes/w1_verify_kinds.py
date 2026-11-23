"""W1. (a) Verify the update rule and its unrolled form for the four tape modes.
(b) On quiet final states: how much of each tape coordinate comes from each KIND of write and from which lag.
(c) Leave-out test: drop one kind of write, rebuild the tape, put it into h_9, count changed answers.
(d) Where the wave-unit clips happen.
Run:  python -B w1_verify_kinds.py > w1_verify_kinds.txt"""
from wcommon import *

B = 300_000
x = sample(B, 101)
H, Z, Amp = amplitudes(x)
st = prefix_stats(x, 9); q = st["quiet"]; M = st["M"]
h9 = H[:, 9]; c9 = tape_coords(h9)

print("(a) EXACTNESS")
# one-step rule in complex mode coordinates, all steps
zc = H[..., ORD] @ LT.T                                             # (B, 10, 4)
wr = np.einsum("bsc,mc->bsm", Amp, LT @ D.T)                        # the write of each step in mode coordinates
prev = np.concatenate([np.zeros((B, 1, 4)), zc[:, :-1]], 1)
print(f"   one-step rule  zeta(t) = lam zeta(t-1) + writes(t):  max |error| = {np.abs(zc - LAMT * prev - wr).max():.2e}  (coordinates are O(10))")
for t in (3, 6, 9):
    print(f"   unrolled sum at t = {t}: max |c - sum_s K[t-s] Amp[s]| = {np.abs(rebuild(Amp, t) - tape_coords(H[:, t])).max():.2e}")
print(f"   B = {B} sequences, quiet final states: {q.sum()} ({q.mean():.3f})")

cq = c9[q] / M[q, None]
sq = cq @ R_SCORE[:8].T                                             # tape scores of ages >= 2 (indices 0..7), units of M
def share(part):
    """Covariance share of a part (n, 7) in each coordinate, and in the tape scores of ages >= 2."""
    pc = part - part.mean(0); cc = cq - cq.mean(0)
    sc = pc @ R_SCORE[:8].T; ss = sq - sq.mean(0)
    return (pc * cc).sum(0) / (cc ** 2).sum(0), (sc * ss).sum() / (ss ** 2).sum()

print("\n(b) SHARE OF THE VARIANCE (covariance share, sums to 1 over the rows) of each tape coordinate / M_t, quiet final states")
print("    last column: the same for the tape scores of ages >= 2 (indices 0..7) taken together")
print(f"    {'kind of write':18s} " + " ".join(f"{n:>9s}" for n in CNAMES) + "    scores")
parts = {}
for name, chs in GROUPS.items():
    m = np.zeros(NCH); m[chs] = 1
    parts[name] = rebuild(Amp[q], 9, m) / M[q, None]
    sh, shs = share(parts[name])
    print(f"    {name:18s} " + " ".join(f"{v:9.3f}" for v in sh) + f"   {shs:7.3f}")
print("    std of the part itself / std of the coordinate (parts cancel if these add to more than 1):")
for name in GROUPS:
    print(f"    {name:18s} " + " ".join(f"{v:9.2f}" for v in parts[name].std(0) / cq.std(0)) +
          f"   {np.sqrt(((parts[name] - parts[name].mean(0)) @ R_SCORE[:8].T).var(0).sum() / sq.var(0).sum()):7.2f}")
print("    mean of the coordinate / M_t:  " + " ".join(f"{v:9.2f}" for v in cq.mean(0)) + "   std: " + " ".join(f"{v:6.2f}" for v in cq.std(0)))
for name in GROUPS:
    print(f"    mean of part {name:15s} " + " ".join(f"{v:9.2f}" for v in parts[name].mean(0)))

print("\n    by single channel (covariance share in the tape scores of ages >= 2; std ratio):")
for ch in range(NCH):
    m = np.zeros(NCH); m[ch] = 1
    p = rebuild(Amp[q], 9, m) / M[q, None]
    print(f"      {CH[ch]:4s} share {share(p)[1]:7.3f}   std ratio {np.sqrt(((p - p.mean(0)) @ R_SCORE[:8].T).var(0).sum() / sq.var(0).sum()):6.2f}   fires in {np.mean(Amp[q][:, :, ch] > 0):.3f} of steps")

print("\n    by age of the write (9 - s), all kinds together: covariance share in the scores, and per kind")
print("    age   all    " + "  ".join(f"{n:>15s}" for n in GROUPS))
for age in range(10):
    row = []
    for chs in [list(range(NCH))] + list(GROUPS.values()):
        m = np.zeros((10, NCH)); m[9 - age, chs] = 1
        row.append(share(rebuild(Amp[q], 9, m) / M[q, None])[1])
    print(f"    {age:3d} {row[0]:6.3f}   " + "  ".join(f"{v:15.3f}" for v in row[1:]))

print("\n(c) LEAVE-OUT TEST on quiet final states: drop a kind of write, rebuild the tape, set it in h_9; share of changed answers")
ans0 = answer(h9); corr = second_argmax(x)
print(f"    baseline accuracy on quiet final states {np.mean(ans0[q] == corr[q]):.4f};  tape deleted entirely: changed {np.mean(answer(set_tape(h9, 0 * c9))[q] != ans0[q]):.3f}")
tests = dict(GROUPS)
tests["wave clips except unit 0"] = [c for c in WAVE_CH if c != 4 + ORD.index(0)]
tests["clip of unit 0"] = [4 + ORD.index(0)]
tests["pulses 4, 5, 9"] = [1, 2, 3]
tests["everything except pulses"] = [0] + COMP_CH + WAVE_CH
tests["input + wave clips except unit 0"] = [0] + tests["wave clips except unit 0"]
for name, chs in tests.items():
    m = np.ones(NCH); m[chs] = 0
    a = answer(set_tape(h9, rebuild(Amp, 9, m)))
    print(f"    drop {name:34s} changed {np.mean(a[q] != ans0[q]):.4f}   accuracy {np.mean(a[q] == corr[q]):.4f}")

print("\n(d) WAVE-UNIT CLIPS: share of steps 1..9 with the unit clipped, mean clip / max(M_(s-1), 0) when clipped, by steps since the latest positive new maximum (tau)")
kind, Mp, Sp, _ = roles(x)
tau = np.full((B, 10), -1)
last = np.full(B, -1)
for s in range(10):
    last = np.where(kind[:, s] == 1, s, last)
    tau[:, s] = np.where(last >= 0, s - last, -1)
Mnow = np.maximum(np.maximum.accumulate(x, 1), 1e-9)
print("    unit   all steps | tau = 0     1      2      3      4      5      6      7+   (share clipped / mean size in units of the running max)")
for u in WAVE:
    a = Amp[:, :, 4 + ORD.index(u)]
    row = f"    {u:4d}   {np.mean(a[:, 1:] > 0):.3f}     |"
    for tv in range(8):
        sel = (tau == tv) if tv < 7 else (tau >= 7)
        sel[:, 0] = False
        row += f" {np.mean(a[sel] > 0):.2f}/{np.mean((a / Mnow)[sel & (a > 0)]) if (sel & (a > 0)).any() else 0:5.2f}"
    print(row)
