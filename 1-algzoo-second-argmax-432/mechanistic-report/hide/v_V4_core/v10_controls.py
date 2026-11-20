"""V4 verifier, controls for the ablations: how sensitive is the real network to perturbations of the SAME size that are
(i) a pure common level, (ii) a random difference pattern?  Seed 5020 (data), 5021 (random patterns)."""
from vsetup import *

out = Tee("v10_controls.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=220)

x = sample(200_000, 5020)
y = second_argmax(x)
rng = np.random.default_rng(5021)


def acc(Wm):
    return np.mean(logits(run(x, W=Wm)).argmax(1) == y)


out(f"intact accuracy (n = 200000, seed 5020): {acc(W):.4f}")

# ---------------------------------------------------------------- unit 2 -> comparators (b2)
d2 = b2 - b2.mean()
n2 = np.linalg.norm(d2)
out(f"\n--- unit 2 -> comparators: b2 = {b2}; its non-uniform part {d2}, norm {n2:.5f} (x 17.3 M = {n2 * 17.3:.4f} M) ---")
Wm = W.copy(); Wm[COMP, 2] = b2.mean() * g
out(f"  P3  non-uniform part removed:                                acc {acc(Wm):.4f}")
Wm = W.copy(); Wm[COMP, 2] = (b2 + d2) * g
out(f"  P3b non-uniform part DOUBLED:                                acc {acc(Wm):.4f}")
Wm = W.copy(); Wm[COMP, 2] = (b2 - 0.5 * d2) * g
out(f"  P3c non-uniform part halved:                                 acc {acc(Wm):.4f}")
for sgn in (+1, -1):
    Wm = W.copy(); Wm[COMP, 2] = (b2 + sgn * n2 / 2) * g
    out(f"  common shift of the same norm ({'+' if sgn > 0 else '-'}{n2 / 2:.5f} on all four):          acc {acc(Wm):.4f}")
res = []
for i in range(8):
    r_ = rng.standard_normal(4); r_ -= r_.mean(); r_ *= n2 / np.linalg.norm(r_)
    Wm = W.copy(); Wm[COMP, 2] = (b2 + r_) * g
    res.append(acc(Wm))
out(f"  random zero-mean pattern of the same norm ADDED (8 draws):   acc {np.array(res)}  mean {np.mean(res):.4f}")

# ---------------------------------------------------------------- wave(no 2) -> comparators
Bw = W[np.ix_(COMP, WAVE_NO2)] / g[:, None]
lvl = Bw.mean(0, keepdims=True)
Dw = Bw - lvl
nD, nL = np.linalg.norm(Dw), np.linalg.norm(np.ones((4, 1)) * lvl)
out(f"\n--- wave(no 2) -> comparators (threshold gauge, 4 x 8): |difference part|_F = {nD:.5f}, |common-level part|_F = {nL:.5f} ---")
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = (np.ones((4, 1)) * lvl) * g[:, None]
out(f"  P1  difference part removed:                                 acc {acc(Wm):.4f}")
Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = Dw * g[:, None]
out(f"  P2  common-level part removed:                               acc {acc(Wm):.4f}")
res = []
for i in range(8):
    r_ = rng.standard_normal((4, 8)); r_ -= r_.mean(0, keepdims=True); r_ *= nD / np.linalg.norm(r_)
    Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = (Bw + r_) * g[:, None]
    res.append(acc(Wm))
out(f"  random difference pattern of norm |difference part| ADDED (8 draws): acc {np.array(res)}  mean {np.mean(res):.4f}")
res = []
for i in range(8):
    r_ = np.ones((4, 1)) * rng.standard_normal((1, 8)); r_ *= nD / np.linalg.norm(r_)
    Wm = W.copy(); Wm[np.ix_(COMP, WAVE_NO2)] = (Bw + r_) * g[:, None]
    res.append(acc(Wm))
out(f"  random common-level pattern of the same norm ADDED (8 draws):        acc {np.array(res)}  mean {np.mean(res):.4f}")

# ---------------------------------------------------------------- the comparator block itself, for scale
out("\n--- for scale: the comparator block Bc ---")
P = np.eye(4) - np.ones((4, 4)) / 4
for name, Bn in (("Bc -> Bc P (input common level removed exactly; changes Bc by |.|_F = %.4f)" % np.linalg.norm(Bc - Bc @ P), Bc @ P),
                 ("Bc -> P Bc (output common level removed; changes Bc by %.3f)" % np.linalg.norm(Bc - P @ Bc), P @ Bc)):
    Wm = W.copy(); Wm[np.ix_(COMP, COMP)] = Bn * g[:, None] / g[None, :]
    out(f"  {name}: acc {acc(Wm):.4f}")
res = []
for i in range(8):
    r_ = rng.standard_normal((4, 4)); r_ *= np.linalg.norm(Bc - Bc @ P) / np.linalg.norm(r_)
    Wm = W.copy(); Wm[np.ix_(COMP, COMP)] = (Bc + r_) * g[:, None] / g[None, :]
    res.append(acc(Wm))
out(f"  random perturbation of Bc with the same norm as Bc - Bc P (8 draws): acc {np.array(res)}  mean {np.mean(res):.4f}")
