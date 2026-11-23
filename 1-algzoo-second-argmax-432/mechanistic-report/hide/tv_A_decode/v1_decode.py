"""v1: recompute the decoding headline numbers with my own code and another seed.
Quiet final states (t = 9, a_M >= 2, a_S >= 2, S > 0) of 400,000 N(0,1) sequences, seed 101; first 75% train, rest test.
Sources (each divided by its own norm):  T = 7 tape coordinates,  R = 9 rest coordinates,  F = 16 units.
Decoders: lin = logistic regression (C = 10) / least squares on standardised features;
          flex = HistGradientBoosting on the same features + modulus and phase of the complex coordinates (100,000 train).
Extra (attack on 'the rest lacks a_S'): stronger decoders of a_S from the rest (kNN, MLP, boosting on all training states).
Output: v1_decode.txt"""
import time
from vcommon import *
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

out = Tee(__file__.replace(".py", ".txt"))
x, h, st = quiet_final(400_000, 101)
B = len(x); ntr = int(0.75 * B); NF = 100_000
M, S, T3, aM, aS, aT, pS = (st[k] for k in ("M", "S", "T", "aM", "aS", "aT", "pS"))
Tp = np.maximum(T3, 0); ans = answer(h); bef = aS > aM
c, r = tape_coords(h), rest_coords(h)
nc, nr, nh = (np.linalg.norm(a, axis=1, keepdims=True) for a in (c, r, h))
pol = lambda re, im, n: np.stack([np.hypot(re, im) / n[:, 0], np.arctan2(im, re)], 1)
lin = {"T": c / nc, "R": r / nr, "F": h / nh}
flex = {"T": np.hstack([c / nc] + [pol(c[:, 1 + 2 * j], c[:, 2 + 2 * j], nc) for j in range(3)]),
        "R": np.hstack([r / nr, pol(r[:, 2], r[:, 3], nr)])}
flex["F"] = np.hstack([h / nh, flex["T"], flex["R"], nc / nh, nr / nh])
rng = np.random.default_rng(7)
out(f"quiet final states {B} of 400000 (seed 101); train {ntr}, flexible decoder {NF}, test {B - ntr}; S before M {bef.mean():.3f}")


def fit_score(src, y, kind, dec, mask=None):
    X = (lin if dec == "lin" else flex)[src]
    idx = np.arange(B) if mask is None else np.flatnonzero(mask)
    tr, te = idx[idx < ntr], idx[idx >= ntr]
    if dec != "lin": tr = tr[:NF]
    if kind == "reg":
        m = make_pipeline(StandardScaler(), LinearRegression()) if dec == "lin" else HGR(max_iter=300, early_stopping=True, random_state=1)
        return 1 - ((y[te] - m.fit(X[tr], y[tr]).predict(X[te])) ** 2).mean() / y[te].var()
    m = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=400)) if dec == "lin" else HGC(max_iter=200, early_stopping=True, random_state=1)
    m.fit(X[tr], y[tr])
    return roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]) if kind == "auc" else (m.predict(X[te]) == y[te]).mean()


other = np.ones(x.shape, bool); other[np.arange(B), pS] = False
n10 = ((np.abs(x - S[:, None]) < 0.10 * M[:, None]) & other).sum(1)
nrec = (x >= np.maximum.accumulate(x, 1)).sum(1)
tasks = [("a_S", "cls", aS, None), ("a_M", "cls", aM, None), ("a_T", "cls", aT, None), ("S before M", "cls", bef.astype(int), None),
         ("network's answer", "cls", ans, None), ("answer correct", "cls", (ans == pS).astype(int), None),
         ("n within 0.10M of S (0/1/2+)", "cls", np.minimum(n10, 2), None), ("number of records (1..4+)", "cls", np.minimum(nrec, 4), None),
         ("shuffled a_S", "cls", rng.permutation(aS), None),
         ("S/M", "reg", S / M, None), ("T+/S", "reg", Tp / S, None), ("shuffled S/M", "reg", rng.permutation(S / M), None)]
for a in range(2, 10):
    tasks.append((f"age {a}: within 0.1M of S", "auc", ((S - x[:, 9 - a]) / M < 0.1).astype(int), (aM != a) & (aS != a)))
jobs = [(nm, s, d, delayed(fit_score)(s, y, kind, d, mask)) for nm, kind, y, mask in tasks for s in "TRF"
        for d in (("lin",) if kind == "auc" else ("lin", "flex"))]
t0 = time.time()
res = dict(zip([j[:3] for j in jobs], Parallel(n_jobs=18)(j[3] for j in jobs)))
out(f"({len(jobs)} fits, {time.time() - t0:.0f} s)   accuracy (cls), R2 (reg), ROC AUC (auc)")
out(f"{'target':30s} {'majority':>8s} | " + " | ".join(f"{s + ' lin':>7s} {s + ' flex':>7s}" for s in "TRF"))
for nm, kind, y, mask in tasks:
    te = y[ntr:] if mask is None else y[ntr:][mask[ntr:]]
    base = np.bincount(te).max() / len(te) if kind == "cls" else (0.5 if kind == "auc" else 0.0)
    out(f"{nm:30s} {base:8.3f} | " + " | ".join(f"{res[nm, s, 'lin']:7.3f} {res.get((nm, s, 'flex'), np.nan):7.3f}" for s in "TRF"))

out("\nA. linear decoder of a_S from the tape, split by configuration (test states)")
m = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=400)).fit(lin["T"][:ntr], aS[:ntr])
pred = m.predict(lin["T"][ntr:]); y = aS[ntr:]; ok = pred == y; b = bef[ntr:]
gT, gM = ((S - T3) / M)[ntr:], ((M - S) / M)[ntr:]
out(f"   all {ok.mean():.3f} | S before M {ok[b].mean():.3f} | S after M {ok[~b].mean():.3f} | third within 0.02 M {ok[gT < .02].mean():.3f} | "
    f"maximum within 0.05 M {ok[gM < .05].mean():.3f}")
out("   by a_S = 2..9: " + " ".join(f"{ok[y == a].mean():.3f}" for a in range(2, 10)))
out(f"   wrong answers ({(~ok).sum()}) name the third's age {(pred == aT[ntr:])[~ok].mean():.3f}, the maximum's age {(pred == aM[ntr:])[~ok].mean():.3f}")
out("   readout matrix as a fixed 'decoder': argmax of the tape piece U[:, o] h_tape = runner-up position: "
    f"{((tape13(c) @ U[:, ORD].T).argmax(1) == pS)[ntr:].mean():.3f}")

out("\nB. ATTACK: does the tape hold the TRUE runner-up's age or the network's ANSWER?  (linear decoders from T, test states)")
age_ans = 9 - ans
m2 = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=400)).fit(lin["T"][:ntr], age_ans[:ntr])
p2 = m2.predict(lin["T"][ntr:]); net_ok = (ans == pS)[ntr:]; ya = age_ans[ntr:]
out(f"   network right on {net_ok.mean():.3f} of the test states ({(~net_ok).sum()} wrong)")
out(f"   decoder trained on a_S:        names a_S {ok.mean():.3f} overall; where the network is wrong: names a_S {ok[~net_ok].mean():.3f}, names the network's answer {(pred == ya)[~net_ok].mean():.3f}")
out(f"   decoder trained on the answer: names the answer {(p2 == ya).mean():.3f} overall; where the network is wrong: names a_S {(p2 == y)[~net_ok].mean():.3f}, names the network's answer {(p2 == ya)[~net_ok].mean():.3f}")

out("\nC. ATTACK on 'the rest lacks a_S': stronger decoders of a_S from the rest (9 numbers / own norm + clock modulus and phase)")
Xr = StandardScaler().fit(flex["R"][:ntr]).transform(flex["R"]); Xt = StandardScaler().fit(flex["T"][:ntr]).transform(flex["T"])


def strong(kind, X, y_):
    mdl = {"kNN k=25 (100k train)": lambda: KNeighborsClassifier(25), "MLP 128-128 (all train)": lambda: MLPClassifier((128, 128), max_iter=60, early_stopping=True, random_state=0),
           "boosting 600 iter, all train": lambda: HGC(max_iter=600, learning_rate=0.1, early_stopping=False, random_state=2)}[kind]()
    n = NF if kind.startswith("kNN") else ntr
    return (mdl.fit(X[:n], y_[:n]).predict(X[ntr:]) == y_[ntr:]).mean()


kinds = ("kNN k=25 (100k train)", "MLP 128-128 (all train)", "boosting 600 iter, all train")
t0 = time.time()
rs = Parallel(n_jobs=6)(delayed(strong)(k, X, aS) for X in (Xr, Xt) for k in kinds)
for i, k in enumerate(kinds):
    out(f"   {k:30s} a_S from the rest {rs[i]:.3f}   from the tape {rs[3 + i]:.3f}")
out(f"   ({time.time() - t0:.0f} s)")

out("\nD. values from raw coordinates: homogeneous flexible decoder |source| * g(source/|source|), and per-class (a_M, a_S) least squares (no intercept)")


def homog(src, y_):
    n = (nc if src == "T" else nr)[:, 0]
    g = HGR(max_iter=300, early_stopping=True, random_state=1).fit(flex[src][:NF], (y_ / n)[:NF])
    return 1 - ((y_[ntr:] - n[ntr:] * g.predict(flex[src][ntr:])) ** 2).mean() / y_[ntr:].var()


def per_class(X, y_):
    p = np.full(B, np.nan); g = aM * 10 + aS
    for k in np.unique(g):
        i = np.flatnonzero(g == k); t = i[i < ntr]
        p[i] = X[i] @ np.linalg.lstsq(X[t], y_[t], rcond=None)[0]
    return 1 - ((y_ - p)[ntr:] ** 2).mean() / y_[ntr:].var()


hv = Parallel(n_jobs=4)(delayed(homog)(s, y_) for y_ in (M, S) for s in "TR")
out(f"   M: tape {hv[0]:.3f}  rest {hv[1]:.3f} (flexible) | per class linear: tape {per_class(c, M):.3f}  rest {per_class(r, M):.3f}")
out(f"   S: tape {hv[2]:.3f}  rest {hv[3]:.3f} (flexible) | per class linear: tape {per_class(c, S):.3f}  rest {per_class(r, S):.3f}")
Xc = np.hstack([np.ones((B, 1)), c / nc])
out(f"   S/M per class affine on the tape direction c/|c|: R2 {per_class(Xc, S / M):.3f}")
