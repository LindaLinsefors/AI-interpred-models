"""V5c. Light version of the remaining rows of v5b (which was too slow): boosted-tree decoders of the runner-up's
position among old entries from (a) the 6 other coordinates (level, runner, clock, trans, echo), (b) all 13, (c) the 7
tape coordinates again as a reference with the same settings.  Inputs are scale-free (divided by their norm).
Quiet final states, N(0,1); fit on 80,000 states, scored on 60,000 others.
Run:  python -B v5c_decoders_rest.py > v5c_decoders_rest.txt"""
from sklearn.ensemble import HistGradientBoostingClassifier

from vcommon import *

d = population(sample(250_000, 9521)); n = len(d["h"]); ho = d["h"][:, ORD]; y = d["iS"]
tr = np.arange(n) < 80_000; te = (np.arange(n) >= 80_000) & (np.arange(n) < 140_000)
def cols(names):
    out = []
    for m in names:
        z = ho @ LVEC[m]
        out += [z.real, z.imag] if abs(EIG[m].imag) > 1e-9 else [z.real]
    X = np.stack(out, 1); return X / np.linalg.norm(X, axis=1, keepdims=True)
REST5 = ["level", "runner", "clock", "trans", "echo"]
print(f"{n} quiet final states (seed 9521); majority class {np.bincount(y[te]).max() / te.sum():.3f}")
for nm, names in (("7 tape coordinates", TAPE), ("6 other coordinates", REST5), ("all 13 coordinates", TAPE + REST5)):
    X = cols(names)
    gb = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.2, max_leaf_nodes=63, random_state=0).fit(X[tr], y[tr])
    print(f"   {nm:22s} boosted trees: accuracy {gb.score(X[te], y[te]):.3f}", flush=True)
