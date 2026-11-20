"""V4 verifier: dose-response of closing the core in the real network, on Gaussian and on uniform inputs.
The weights from the other ten units into the six core units are multiplied by (1 - f).  Seeds 5022 (normal), 5023 (uniform)."""
from vsetup import *

out = Tee("v11_dose.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=220)
THR = COMP + [4]
r = np.real(RIGHT["level"]); r_of = {u: r[ORD.index(u)] for u in ORD}
add = {j: sum(W[j, u] * r_of[u] for u in WAVE_NO2) / r_of[2] for j in CORE}

for dist, seed in (("normal", 5022), ("uniform", 5023)):
    x = sample(300_000, seed, dist)
    y = second_argmax(x)
    out(f"\n=== inputs iid {dist}, n = 300000, seed {seed} ===")
    out("   f :  other ten -> core x (1-f)   same + K5 compensation x f   wave(no 2) -> comparators + unit 4 x (1-f)   "
        "wave(no 2) -> comparators: difference part x (1-f)   pulses 5, 9 -> core x (1-f)")
    Bw = W[np.ix_(COMP, WAVE_NO2)] / g[:, None]
    lvl = Bw.mean(0, keepdims=True)
    for f in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        accs = []
        Wm = W.copy(); Wm[np.ix_(CORE, OTHER10)] *= (1 - f)
        accs.append(np.mean(logits(run(x, W=Wm)).argmax(1) == y))
        for j in CORE:
            Wm[j, 2] += f * add[j]
        accs.append(np.mean(logits(run(x, W=Wm)).argmax(1) == y))
        Wm = W.copy(); Wm[np.ix_(THR, WAVE_NO2)] *= (1 - f)
        accs.append(np.mean(logits(run(x, W=Wm)).argmax(1) == y))
        Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = (lvl + (1 - f) * (Bw - lvl)) * g[:, None]
        accs.append(np.mean(logits(run(x, W=Wm)).argmax(1) == y))
        Wm = W.copy(); Wm[np.ix_(CORE, [5, 9])] *= (1 - f)
        accs.append(np.mean(logits(run(x, W=Wm)).argmax(1) == y))
        out(f"  {f:4.2f}:  {accs[0]:.4f}                     {accs[1]:.4f}                        {accs[2]:.4f}                                      "
            f"{accs[3]:.4f}                                             {accs[4]:.4f}")
