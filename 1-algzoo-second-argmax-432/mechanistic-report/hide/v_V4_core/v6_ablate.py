"""V4 verifier, attack 2: does unit 2 alone supply the level? is the core functionally closed?
Weight ablations in the REAL 16-unit network (all other weights untouched, readout untouched). Seed 5008, n = 400000."""
from vsetup import *

out = Tee("v6_ablate.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=220)

x = sample(400_000, 5008)
y = second_argmax(x)
rs = running_stats(x)
M, a1 = rs["M1"], rs["a1"]
tt = np.broadcast_to(np.arange(10), M.shape)
sel = (tt <= 8) & (a1 >= 2) & (M > 0.3)                       # the K3 selection
newmax = np.zeros_like(x, dtype=bool)
newmax[:, 0] = x[:, 0] > 0
newmax[:, 1:] = x[:, 1:] > np.maximum(M[:, :-1], 0.0)

r = np.real(RIGHT["level"])
r_of = {u: r[ORD.index(u)] for u in ORD}
add = {j: sum(W[j, u] * r_of[u] for u in WAVE_NO2) / r_of[2] for j in CORE}
THR = COMP + [4]                                              # units that have a threshold
ROWS = COMP + [4]
GAUGE = np.concatenate([g, [-w[4]]])

H0 = run(x)
ans0 = logits(H0).argmax(1)


def evaluate(name, Wm):
    H = run(x, W=Wm)
    ans = logits(H).argmax(1)
    th = (H @ Wm[ROWS].T) / GAUGE                              # theta(t+1), t = 0..9, own thresholds of the modified net
    lvl = (th[sel] / M[sel][:, None]).mean(0)
    fire = H[:, :, 4] > 0
    prec = (fire & newmax).sum() / max(fire.sum(), 1)
    rec = (fire & newmax).sum() / newmax.sum()
    h2 = np.median(H[:, 1:, 2][(a1[:, 1:] >= 1) & (M[:, :-1] > 0.3)] / M[:, :-1][(a1[:, 1:] >= 1) & (M[:, :-1] > 0.3)])
    out(f"{name:66s} acc {np.mean(ans == y):.4f}  same answer as intact {np.mean(ans == ans0):.4f}  "
        f"mean theta(t+1)/M [1,6,7,8,u4] {lvl}  unit4: P(new max|fire) {prec:.3f} P(fire|new max) {rec:.3f}  med h2/M {h2:.2f}")
    return np.mean(ans == y)


def variant(rows=(), cols=(), comp_rows=(), scale2=None, scale_rows=THR):
    Wm = W.copy()
    if len(rows):
        Wm[np.ix_(list(rows), list(cols))] = 0.0
    for j in comp_rows:
        Wm[j, 2] += add[j]
    if scale2 is not None:
        Wm[list(scale_rows), 2] *= scale2
    return Wm


out("K3 selection for the threshold column: t <= 8, a1 >= 2, M > 0.3.  n = 400000, seed 5008")
out("compensation (K5) add to W[j,2]:", {j: round(float(v), 4) for j, v in add.items()})
evaluate("intact network", W)
out("\n--- the 0.04-0.11 M that the other wave units add to the thresholds ---")
evaluate("A  zero wave(no 2) -> comparators + unit 4", variant(THR, WAVE_NO2))
evaluate("A' zero wave(no 2) -> comparators only", variant(COMP, WAVE_NO2))
evaluate("A''zero wave(no 2) -> unit 4 only", variant([4], WAVE_NO2))
evaluate("B  A + K5 compensation on W[j,2], j in comparators + unit 4", variant(THR, WAVE_NO2, comp_rows=THR))
evaluate("C  zero wave(no 2) -> unit 2", variant([2], WAVE_NO2))
evaluate("C' C + K5 compensation on W[2,2]", variant([2], WAVE_NO2, comp_rows=[2]))
evaluate("D  zero wave(no 2) -> all six core units", variant(CORE, WAVE_NO2))
evaluate("E  D + K5 compensation on all six", variant(CORE, WAVE_NO2, comp_rows=CORE))
out("\n--- functional closure of the core: the six core units receive NOTHING from the other ten ---")
evaluate("F  zero other ten -> core (core == plain six-unit simulation)", variant(CORE, OTHER10))
evaluate("G  F + K5 compensation (core == compensated simulation)", variant(CORE, OTHER10, comp_rows=CORE))
evaluate("H  zero pulses 5, 9 -> core only", variant(CORE, [5, 9]))
evaluate("H' zero units 5, 9 -> comparators + unit 4 only", variant(THR, [5, 9]))
out("\n--- unit 2 as the supplier of the level ---")
evaluate("I  zero unit 2 -> comparators + unit 4", variant(THR, [2]))
evaluate("I' zero unit 2 -> comparators only", variant(COMP, [2]))
evaluate("I''zero unit 2 -> unit 4 only", variant([4], [2]))
for s in (0.8, 0.9, 0.95, 1.05, 1.1, 1.2):
    evaluate(f"J  unit 2 -> comparators + unit 4 scaled by {s}", variant(scale2=s))
for s in (0.9, 1.1):
    evaluate(f"J' unit 2 -> comparators only scaled by {s}", variant(scale2=s, scale_rows=COMP))
    evaluate(f"J''unit 2 -> unit 4 only scaled by {s}", variant(scale2=s, scale_rows=[4]))
out("\n--- controls: the same wave units' weights into OTHER receivers, and other senders into the thresholds ---")
evaluate("K  zero comparators -> comparators (the Bc block itself)", variant(COMP, COMP))
evaluate("L  zero unit 4 -> comparators (b4)", variant(COMP, [4]))
evaluate("L' zero unit 4 -> unit 2", variant([2], [4]))
evaluate("N  zero comparators -> unit 2 and unit 4", variant([2, 4], COMP))
for u in WAVE_NO2:
    evaluate(f"O  zero unit {u} -> comparators + unit 4", variant(THR, [u]))
