"""B8. Why does the oracle size hurt? Path-specific removal of the 'excess' of a record kick.

On a true positive-part record the network's unit 4 gives h4 = 10.157*(x_t - theta_4) = oracle + delta,
delta = 10.157*(Mp_{t-1} - theta_4(t)) (the sag of the threshold at that moment).
Unit 4 acts on the next step through W[:, 4]:  W[ORD, 4] = sum over modes of the mode vectors (kick = 1 per mode),
plus W[4,4] (self-inhibition, holds the threshold for one step), W[5,4], W[9,4]; and through U[:, 4] at t = 9.
Here delta is removed from chosen paths only. All paths together = oracle size on true records (false firings untouched).
Data: N(0,1), seeds 1080, 1081 (400k each).
"""
from blib import *

VEC = {}
for n in NAMES:
    v = np.zeros(N)
    v[ORD] = 2 * RIGHT[n].real if IS_COMPLEX[n] else RIGHT[n].real
    VEC[n] = v
for q in PULSE:
    v = np.zeros(N); v[q] = W[q, 4]; VEC[f"u{q}"] = v
assert np.allclose(sum(VEC.values()), W[:, 4], atol=1e-10), "paths must add up to W[:, 4]"
ALL = list(VEC) + ["readout"]


def run_paths(x, paths, records_only=True):
    V = sum((VEC[p] for p in paths if p != "readout"), np.zeros(N))
    B = len(x); h = np.zeros((B, N)); Mp = np.zeros(B); d = np.zeros(B)
    for t in range(10):
        z = x[:, t, None] * w + h @ W.T - d[:, None] * V
        h = np.maximum(z, 0)
        orc = W4 * pos(x[:, t] - Mp)
        if records_only:
            d = np.where((x[:, t] > Mp) & (h[:, 4] > 0), h[:, 4] - orc, 0.0)
        else:
            d = h[:, 4] - orc
        Mp = np.maximum(Mp, x[:, t])
    lg = h @ U.T
    if "readout" in paths:
        lg = lg - d[:, None] * U[:, 4][None, :]
    return lg


def run_oracle(x):
    B = len(x); h = np.zeros((B, N)); Mp = np.zeros(B)
    for t in range(10):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        h[:, 4] = W4 * pos(x[:, t] - Mp)
        Mp = np.maximum(Mp, x[:, t])
    return h @ U.T


sets = [("none (network)", []), ("all paths (oracle size on true records)", ALL),
        ("level only", ["level"]), ("runner only", ["runner"]), ("level + runner", ["level", "runner"]), ("clock only", ["clock"]),
        ("short only", ["short"]), ("h2 only", ["h2"]), ("h3 only", ["h3"]), ("alt only", ["alt"]), ("trans only", ["trans"]), ("echo only", ["echo"]),
        ("all 9 ordinary modes", list(NAMES)), ("unit 4 self-inhibition only", ["u4"]), ("unit 5 only", ["u5"]), ("unit 9 only", ["u9"]), ("readout only", ["readout"]),
        ("all except level", [p for p in ALL if p != "level"]), ("all except level + runner", [p for p in ALL if p not in ("level", "runner")]),
        ("all except clock", [p for p in ALL if p != "clock"]), ("all except u4", [p for p in ALL if p != "u4"]),
        ("level + runner + u4", ["level", "runner", "u4"]), ("level + runner + clock + u4", ["level", "runner", "clock", "u4"])]
res = {name: [0, 0] for name, _ in sets}
res_full = [0, 0]
NS = 0
for seed in (1080, 1081):
    x = sample(400_000, seed)
    sa = second_argmax(x)
    base = logits(run(x)).argmax(1)
    NS += len(x)
    for name, paths in sets:
        a = run_paths(x, paths).argmax(1)
        res[name][0] += (a == base).sum(); res[name][1] += (a == sa).sum()
        if name.startswith("none"):
            assert (a == base).all()
    # consistency: all paths with records_only=False must equal the plain oracle patch
    a1 = run_paths(x, ALL, records_only=False).argmax(1)
    a2 = run_oracle(x).argmax(1)
    res_full[0] += (a1 == a2).sum(); res_full[1] += (a2 == sa).sum()
print(f"[INTERVENTION n = {NS}, N(0,1), seeds 1080, 1081]")
print(f"check: 'all paths, all firings' reproduces the oracle patch of B6(ii): identical answers in {res_full[0]/NS:.6f} of sequences; oracle accuracy {res_full[1]/NS:.4f}")
print("excess removed from | agreement with the network | task accuracy | change of accuracy (pp)")
b = res["none (network)"][1] / NS
for name, _ in sets:
    print(f"{name} | {res[name][0]/NS:.4f} | {res[name][1]/NS:.4f} | {100*(res[name][1]/NS - b):+.2f}")
