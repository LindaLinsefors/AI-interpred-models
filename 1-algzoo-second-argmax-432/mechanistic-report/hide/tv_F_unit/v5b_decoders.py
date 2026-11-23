"""V5b. Claim 9, decoders: is the runner-up's position (among old entries) contained in the 7 tape coordinates?
The investigator fitted a linear decoder only (0.904, not converged).  Here: a converged linear decoder and two
NONLINEAR decoders (gradient-boosted trees, a small MLP) on scale-free inputs c / |c| (the tape alone does not know M).
Quiet final states, N(0,1); fit on 150,000 states, scored on 80,000 others.
Run:  python -B v5b_decoders.py > v5b_decoders.txt"""
import warnings

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from vcommon import *

warnings.filterwarnings("ignore")
d = population(sample(400_000, 9511)); n = len(d["h"]); ho = d["h"][:, ORD]
def cols(names):
    out = []
    for m in names:
        z = ho @ LVEC[m]
        out += [z.real, z.imag] if abs(EIG[m].imag) > 1e-9 else [z.real]
    X = np.stack(out, 1); return X / np.linalg.norm(X, axis=1, keepdims=True)
y = d["iS"]; tr = np.arange(n) < 150_000; te = (np.arange(n) >= 150_000) & (np.arange(n) < 230_000)
pc = pieces(d["h"]); tape = sum(pc[m] for m in TAPE); full = d["h"] @ U.T
print(f"{n} quiet final states (seed 9511).  The network's own readout: tape piece alone {np.mean((top_old(tape, d['valid']) == y)[te]):.3f}, full score {np.mean((top_old(full, d['valid']) == y)[te]):.3f}")
REST5 = ["level", "runner", "clock", "trans", "echo"]
for nm, names in (("7 tape coordinates", TAPE), ("6 other coordinates", REST5), ("all 13 coordinates", TAPE + REST5), ("tape + level + runner + clock", TAPE + REST5[:3])):
    X = cols(names); sd = X[tr].std(0); Xs = X / sd
    lin = LogisticRegression(C=100.0, fit_intercept=False, max_iter=5000, tol=1e-6).fit(Xs[tr], y[tr])
    row = f"   {nm:30s} linear (no intercept, {int(lin.n_iter_[0])} iterations) {lin.score(Xs[te], y[te]):.3f}"
    if nm in ("7 tape coordinates", "6 other coordinates"):
        gb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.15, max_leaf_nodes=63, random_state=0).fit(Xs[tr], y[tr])
        row += f"   boosted trees {gb.score(Xs[te], y[te]):.3f}"
        mlp = MLPClassifier((128, 128), max_iter=40, random_state=0, batch_size=512).fit(Xs[tr], y[tr])
        row += f"   MLP 128-128 {mlp.score(Xs[te], y[te]):.3f} (train {mlp.score(Xs[tr][:50_000], y[tr][:50_000]):.3f})"
        if nm == "7 tape coordinates":
            p = mlp.predict(Xs[te]); t_ok = (top_old(tape, d["valid"]) == y)[te]; f_ok = (top_old(full, d["valid"]) == y)[te]
            row += f"\n        MLP on the tape coordinates: right where the network's full readout is right {np.mean((p == y[te])[f_ok]):.3f}, where it is wrong {np.mean((p == y[te])[~f_ok]):.3f};"
            row += f" right where the tape piece's own top entry is wrong {np.mean((p == y[te])[~t_ok]):.3f} (n {(~t_ok).sum()})"
            for a in (2, 3):
                qa = (d["aS"] == a)[te]
                row += f"\n        runner-up of age {a}: MLP {np.mean((p == y[te])[qa]):.3f}, tape piece's top entry {np.mean(t_ok[qa]):.3f}, full {np.mean(f_ok[qa]):.3f}"
    print(row, flush=True)
