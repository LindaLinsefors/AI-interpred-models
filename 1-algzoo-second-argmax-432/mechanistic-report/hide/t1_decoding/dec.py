"""Decoding helpers (angle DECODING of the question "what does the tape store?", 2026-11-23).

Population: QUIET FINAL STATES, t = 9, a_M >= 2, a_S >= 2, S > 0, iid N(0,1) inputs (data()).
Sources, all made scale-free by dividing by their OWN Euclidean norm (so the T decoder sees nothing but the tape):
    T   the 7 tape coordinates c / |c|
    Tn  the 7 tape coordinates divided by the norm of the full state, c / |h|   (tape + overall scale)
    R   the rest: (level, runner, Re clock, Im clock, trans, echo, h4, h5, h9) / |r|      9 numbers
    F   the full state h / |h|                                                             16 numbers
Decoders:
    lin   affine least squares (regression) or multinomial logistic regression, C = 10, on standardised features
    flex  HistGradientBoosting on the same features plus modulus and phase of each complex coordinate; the F feature
          set contains the T and R feature sets, so F >= T, R also for the trees.
Scores on held-out sequences: accuracy (baseline = majority class), ROC AUC for rare yes/no targets (baseline 0.5)
or R2 (baseline 0 = predicting the mean).
"""
import sys
import time

ROOT = "c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report"
sys.path.insert(0, ROOT + "/hide")
sys.dont_write_bytecode = True
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tape_common import *

SRC = ("T", "Tn", "R", "F")
N_SEQ, N_FLEX = 400_000, 100_000          # sequences sampled; training states used by the flexible decoder


def rest_coords(h):
    m = coords(h)
    return np.stack([m["level"], m["runner"], m["clock"].real, m["clock"].imag, m["trans"], m["echo"],
                     h[:, 4], h[:, 5], h[:, 9]], 1)


def feats(h):
    """States (B, 16) -> (lin, flex, norms): dicts source -> feature matrix / own norm."""
    c, r = tape_coords(h), rest_coords(h)
    nc, nr, nh = (np.linalg.norm(a, axis=1, keepdims=True) for a in (c, r, h))
    pol = lambda re, im, n: np.stack([np.hypot(re, im) / n[:, 0], np.arctan2(im, re)], 1)
    Tp = np.hstack([pol(c[:, 1 + 2 * j], c[:, 2 + 2 * j], nc) for j in range(3)])
    lin = {"T": c / nc, "Tn": c / nh, "R": r / nr, "F": h / nh}
    flex = {"T": np.hstack([c / nc, Tp]), "Tn": np.hstack([c / nh, Tp, nc / nh]),
            "R": np.hstack([r / nr, pol(r[:, 2], r[:, 3], nr)])}
    flex["F"] = np.hstack([h / nh, flex["T"], flex["R"], nc / nh, nr / nh])
    return lin, flex, {"T": nc[:, 0], "Tn": nh[:, 0], "R": nr[:, 0], "F": nh[:, 0]}


def data(n=N_SEQ, seed=11):
    """Quiet final states: inputs x (B, 10), states h (B, 16), prefix_stats dict, and derived targets D."""
    x = sample(n, seed)
    h = run(x)[:, 9]
    st = prefix_stats(x, 9)
    q = st["quiet"]
    x, h, st = x[q], h[q], {k: v[q] for k, v in st.items()}
    M, S, T3 = st["M"][:, None], st["S"][:, None], st["T3"][:, None]
    ar, pos = np.arange(len(x)), np.arange(10)
    D = dict(rel=(x - S) / M,                                   # by position p; age a is column 9 - a
             relc=(np.maximum(x, 0) - S) / M,                   # the same with negative inputs set to 0
             rank=np.argsort(np.argsort(-x, 1), 1))             # 0 = maximum
    other = np.ones(x.shape, bool); other[ar, st["pS"]] = False
    D["n10"] = ((np.abs(D["rel"]) < 0.10) & other).sum(1)       # inputs other than the runner-up within 0.10 M of S
    D["n25"] = ((np.abs(D["rel"]) < 0.25) & other).sum(1)
    rec = x >= np.maximum.accumulate(x, 1)                      # x_p was the maximum when it arrived
    before = rec & (pos < st["pM"][:, None])
    D["rec"], D["nrec"] = rec, rec.sum(1)
    D["a_prev"] = np.where(before.any(1), np.argmax(before[:, ::-1], 1), -1)   # age of the last overtaken maximum (-1: none)
    D["a_min"], D["min"] = 9 - x.argmin(1), x.min(1) / M[:, 0]
    D["S_before_M"] = st["aS"] > st["aM"]
    D["ans"] = answer(h)
    return x, h, st, D


def _score(Xtr, ytr, Xte, yte, kind, flex):
    if kind in ("cls", "auc"):
        m = (HistGradientBoostingClassifier(max_iter=200, early_stopping=True, random_state=0) if flex else
             make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=400)))
        m.fit(Xtr, ytr)
        return (m.predict(Xte) == yte).mean() if kind == "cls" else roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    m = (HistGradientBoostingRegressor(max_iter=300, early_stopping=True, random_state=0) if flex else
         make_pipeline(StandardScaler(), LinearRegression()))
    res = yte - m.fit(Xtr, ytr).predict(Xte)
    return 1 - (res ** 2).mean() / yte.var()


def table(tasks, lin, flex, n_train, srcs=SRC, n_jobs=10, out=print, n_flex=N_FLEX):
    """tasks: list of (name, kind, y, mask or None). Rows 0..n_train-1 train, the rest test (one state per sequence).
    Prints one line per task: baseline, then lin/flex score for every source. Returns {(name, src, dec): score}."""
    idx = np.arange(len(next(iter(lin.values()))))
    jobs, base = [], {}
    for name, kind, y, mask in tasks:
        ok = np.ones(len(idx), bool) if mask is None else mask
        tr, te = idx[ok & (idx < n_train)], idx[ok & (idx >= n_train)]
        if kind == "cls":
            vals, cnt = np.unique(y[tr], return_counts=True)
            base[name] = ((y[te] == vals[cnt.argmax()]).mean(), len(tr), len(te))
        else:
            base[name] = (0.5 if kind == "auc" else 0.0, len(tr), len(te))
        for s in srcs:
            jobs.append((name, s, "lin", delayed(_score)(lin[s][tr], y[tr], lin[s][te], y[te], kind, False)))
            trf = tr[:n_flex]
            jobs.append((name, s, "flex", delayed(_score)(flex[s][trf], y[trf], flex[s][te], y[te], kind, True)))
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs)(j[3] for j in jobs)
    sc = {j[:3]: r for j, r in zip(jobs, res)}
    out(f"{'target':34s} {'n_test':>7s} {'base':>6s} | " + " | ".join(f"{s + ' lin':>7s} {s + ' flex':>7s}" for s in srcs))
    for name, kind, y, mask in tasks:
        b, ntr, nte = base[name]
        out(f"{name:34s} {nte:7d} {b:6.3f} | " + " | ".join(f"{sc[name, s, 'lin']:7.3f} {sc[name, s, 'flex']:7.3f}" for s in srcs))
    out(f"   ({len(jobs)} fits, {time.time() - t0:.0f} s; cls: accuracy, base = majority class; auc: ROC AUC, base 0.5; reg: R2, base 0)")
    return sc, base


class Tee:
    """print to the screen and to a .txt file next to the script."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(v) for v in a)
        print(s, flush=True); self.f.write(s + "\n"); self.f.flush()
