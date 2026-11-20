"""V4 verifier, attack 2c: the lead's idealisation taken literally in the real network:
'the comparators get ONE common level, supplied by unit 2 alone'.  Do the difference parts fed by unit 2 (non-uniform b2)
and by the other wave units merely cancel each other, or are they needed?   Seed 5014, n = 400000."""
from vsetup import *

out = Tee("v6c_idealised_level.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=220)

x = sample(400_000, 5014)
y = second_argmax(x)
H0 = run(x)
ans0 = logits(H0).argmax(1)
rs = running_stats(x)
M, a1 = rs["M1"], rs["a1"]
tt = np.broadcast_to(np.arange(10), M.shape)


def evaluate(name, Wm):
    ans = logits(run(x, W=Wm)).argmax(1)
    out(f"  {name:100s} acc {np.mean(ans == y):.4f}   same as intact {np.mean(ans == ans0):.4f}")


Bw = W[np.ix_(COMP, WAVE_NO2)] / g[:, None]                 # wave -> comparator weights, threshold gauge (4, 8)
lvl = Bw.mean(0, keepdims=True)

# best static (least-squares) stand-in of the wave(no 2) part of each comparator threshold by unit 2
s = (tt >= 1) & (tt <= 8) & (M > 0.3)
wave_part = (H0[:, :, WAVE_NO2] @ W[np.ix_(COMP, WAVE_NO2)].T) / g
h2 = H0[:, :, 2]
cfit = np.array([(wave_part[s][:, i] * h2[s]).sum() / (h2[s] ** 2).sum() for i in range(4)])   # threshold gauge
out("b2 =", b2, " mean", b2.mean())
out("best-fit stand-in c_j (threshold gauge) for the wave(no 2) part:", cfit, " mean", cfit.mean())
out("b2 + c =", b2 + cfit, " mean", (b2 + cfit).mean())

evaluate("intact", W)
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = lvl * g[:, None]
evaluate("P1  wave(no 2) -> comparators: only the common-level part kept", Wm)
Wm = W.copy(); Wm[COMP, 2] = b2.mean() * g
evaluate("P3  unit 2 -> comparators made uniform (mean b2)", Wm)
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = lvl * g[:, None]; Wm[COMP, 2] = b2.mean() * g
evaluate("Q1  P1 + P3: every level supplier (unit 2 and the other wave units) feeds the four comparators uniformly", Wm)
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = 0; Wm[COMP, 2] = (b2 + cfit) * g
evaluate("Q2  wave(no 2) -> comparators removed, unit 2 weight b2 + c_j (non-uniform static stand-in; = A'-fit)", Wm)
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = 0; Wm[COMP, 2] = (b2 + cfit).mean() * g
evaluate("Q4  THE IDEALISATION: wave(no 2) -> comparators removed, unit 2 feeds one uniform level mean(b2 + c)", Wm)
for scale in (0.9, 0.95, 1.05, 1.1):
    Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = 0; Wm[COMP, 2] = (b2 + cfit).mean() * scale * g
    evaluate(f"Q4x idealisation with the uniform level scaled by {scale}", Wm)
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = 0; Wm[COMP, 2] = b2 * g * ((b2 + cfit).mean() / b2.mean())
evaluate("Q5  wave(no 2) -> comparators removed, unit 2 keeps its own pattern b2, rescaled to the same mean level", Wm)
# pulses 5, 9 -> comparators also removed, on top of Q4
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2 + [5, 9])] = 0; Wm[COMP, 2] = (b2 + cfit).mean() * g
evaluate("Q6  Q4 + pulses 5, 9 -> comparators removed (comparators hear only comparators, unit 2 (uniform), unit 4)", Wm)
# keep the real net but remove from the wave -> comparator weights only what a level cannot do, unit by unit
for u in WAVE_NO2:
    i = WAVE_NO2.index(u)
    Wm = W.copy(); Wm[COMP, u] = lvl[0, i] * g
    evaluate(f"R   unit {u:2d} -> comparators: only its common-level part kept", Wm)
