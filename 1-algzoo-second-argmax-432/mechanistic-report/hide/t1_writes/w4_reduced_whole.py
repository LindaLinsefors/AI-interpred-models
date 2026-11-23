"""W4. The reduced description as a whole, and the functional test with the run continued.
(1) Tape score ENTRIES of ages 2..9 rebuilt with the exact kernel from subsets of the write channels (real amplitudes):
    why the pulses alone (e36) cannot give the tape.
(2) Named writes only: amplitudes from the L2 / L3 models of w3 (thresholds from M+, S+, T+, x_(s-1) and their ages),
    with and without the writes of the classes far, neg, spont.  Quiet final states.
(3) The same at t = 4, 6, 7 with the run continued to step 9 (share of states with a negative activation; repeated clipped).
Test set = the 200k sequences of w3 (seed 304); modelled drives are read from w3_cache.npz.
Run:  python -B w4_reduced_whole.py > w4_reduced_whole.txt"""
from wcommon import *

x = sample(200_000, 304); B = len(x)
H, Z, Amp = amplitudes(x)
cache = np.load("w3_cache.npz")
corr = second_argmax(x)

def amp_from_drive(Yhat):
    Zh = Z.copy(); Zh[:, 1:] = x[:, 1:, None] * w + Yhat.reshape(B, 9, 16).astype(np.float64)
    A = np.zeros_like(Amp); A[:, :, 0] = x
    A[:, 1:, 1:4] = np.maximum(Zh[:, :-1][:, :, PULSE], 0)
    A[:, :, 4:] = np.maximum(-Zh[:, :, ORD], 0)
    return A
AL = {"L2": amp_from_drive(cache["L2"]), "L3": amp_from_drive(cache["L3"])}

st = prefix_stats(x, 9); q = st["quiet"]; M = st["M"]
h9 = H[:, 9]; c9 = tape_coords(h9); ans0 = answer(h9)
cq = c9[q] / M[q, None]; sq = cq @ R_SCORE.T                                     # all ten entries, units of M
print(f"test sequences {B}; quiet final states n = {q.sum()}; baseline accuracy {np.mean(ans0[q] == corr[q]):.4f}")

print("\n(1) R2 of the tape score entry of age a (quiet final states, entries / M_t), exact kernel, REAL amplitudes of a subset of channels")
print("    channels kept                         a = 2      3      4      5      6      7      8      9   | answers changed")
u0 = 4 + ORD.index(0)
subsets = {"pulses 4, 5, 9 (as e36, no fit)": [1, 2, 3], "pulses + comparators off": [1, 2, 3] + COMP_CH,
           "pulses + comparators + clip of unit 0": [1, 2, 3] + COMP_CH + [u0], "all but the input term": list(range(1, NCH)), "all 17": list(range(NCH))}
for nm, chs in subsets.items():
    m = np.zeros(NCH); m[chs] = 1
    c = rebuild(Amp, 9, m); s = (c[q] / M[q, None]) @ R_SCORE.T
    a = answer(set_tape(h9, c))
    print(f"    {nm:38s} " + " ".join(f"{r2(sq[:, 9 - a_], s[:, 9 - a_]):6.3f}" for a_ in range(2, 10)) + f"   | {np.mean(a[q] != ans0[q]):.4f}")
# the e36 way: free least-squares kernel on the pulses, but with every lag allowed
P = np.concatenate([Amp[q][:, :, 1:4].reshape(q.sum(), -1) / M[q, None]], 1)
bfit = np.linalg.lstsq(P, sq, rcond=None)[0]
print(f"    {'pulses, kernel FITTED by least squares':38s} " + " ".join(f"{r2(sq[:, 9 - a_], (P @ bfit)[:, 9 - a_]):6.3f}" for a_ in range(2, 10)))

print("\n(2) NAMED WRITES ONLY, quiet final states: R2 per coordinate, R2 of the scores of ages >= 2, answers changed, accuracy")
wc = write_classes(x, 9)
def line(label, c):
    ch = c[q] / M[q, None]; a = answer(set_tape(h9, c))
    print(f"    {label:58s} " + " ".join(f"{v:6.3f}" for v in r2(cq, ch)) +
          f"  scores {1 - (((ch - cq) @ R_SCORE[:8].T) ** 2).sum() / ((sq[:, :8] - sq[:, :8].mean(0)) ** 2).sum():6.3f} | changed {np.mean(a[q] != ans0[q]):.4f}  acc {np.mean(a[q] == corr[q]):.4f}")
keepA = np.isin(wc, [0, 1, 2, 3, 4, 5]); keepB = np.isin(wc, [0, 1, 2, 3, 4, 5, 8]); keepC = np.isin(wc, [0, 1, 2])
line("real amplitudes, all classes (check: exact)", rebuild(Amp, 9))
line("real amplitudes, classes M, S, exmax, exrun, near", rebuild(Amp, 9, keepA))
for lev, A in AL.items():
    line(f"{lev} amplitudes, all classes", rebuild(A, 9))
    line(f"{lev} amplitudes, without far, neg", rebuild(A, 9, keepB))
    line(f"{lev} amplitudes, classes M, S, exmax, exrun, near", rebuild(A, 9, keepA))
    line(f"{lev} amplitudes, classes M and S only", rebuild(A, 9, keepC))
print("    reference: a transplant of the tape of a RANDOM other quiet final state, rescaled to the recipient's M_t")
perm = np.random.default_rng(5).permutation(np.where(q)[0]); don = np.arange(B); don[np.where(q)[0]] = perm
line("tape of a random quiet state (rescaled by M)", c9[don] / M[don, None] * M[:, None])
print("    split of the L3 result by configuration (answers changed):")
a3 = answer(set_tape(h9, rebuild(AL["L3"], 9))); chg = a3 != ans0
for nm, sel in (("runner-up after the maximum", q & (st["pS"] > st["pM"])), ("runner-up before the maximum", q & (st["pS"] < st["pM"]))):
    print(f"      {nm:30s} n = {sel.sum():6d}  changed {chg[sel].mean():.4f}   by age of the runner-up 2..9: " +
          " ".join(f"{chg[sel & (st['aS'] == a)].mean():.3f}" for a in range(2, 10)))
gap = (st["S"] - st["T3"]) / M
print("      by (S - T)/M, the runner-up's lead over the third: " + "  ".join(
    f"[{lo:.2f},{hi:.2f}) {chg[q & (gap >= lo) & (gap < hi)].mean():.3f}" for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 9))))

print("\n(3) RUN CONTINUED: quiet states at step t, tape replaced at step t, inputs x_(t+1)..x_9 as they were; answer at step 9")
print("    neg = share of modified states with an activation below -1e-9;  'clipped' = same test with the modified state clipped at 0")
for t in (4, 6, 7):
    stt = prefix_stats(x, t); qt = stt["quiet"]; ht = H[:, t]; wct = write_classes(x, t)
    ref = answer(continue_from(ht[qt], x[qt, t + 1:])[:, -1])
    assert (ref == ans0[qt]).all()
    print(f"  t = {t}: quiet states n = {qt.sum()}")
    cands = {"L3 amplitudes, all classes": rebuild(AL["L3"], t), "L2 amplitudes, all classes": rebuild(AL["L2"], t),
             "L3 amplitudes, classes M, S, exmax, exrun, near": rebuild(AL["L3"], t, np.isin(wct, [0, 1, 2, 3, 4, 5])),
             "real amplitudes, classes M and S only": rebuild(Amp, t, np.isin(wct, [0, 1, 2])),
             "real amplitudes, without far, neg": rebuild(Amp, t, np.isin(wct, [0, 1, 2, 3, 4, 5, 8])),
             "tape deleted": 0 * rebuild(Amp, t)}
    for nm, c in cands.items():
        hm = set_tape(ht, c)[qt]
        a1 = answer(continue_from(hm, x[qt, t + 1:])[:, -1]); a2 = answer(continue_from(np.maximum(hm, 0), x[qt, t + 1:])[:, -1])
        print(f"    {nm:50s} changed {np.mean(a1 != ref):.4f}   neg {np.mean((hm < -1e-9).any(1)):.3f}   clipped: changed {np.mean(a2 != ref):.4f}")
