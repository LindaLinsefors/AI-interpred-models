"""d5: WHERE in the tape are the ages, and where does the decoder of the runner-up's age fail?  Quiet final states (t = 9).
A. k-nearest-neighbour decoders (k = 25, 100,000 training states) on subsets of the tape coordinates, each subset divided
   by its own norm (so one complex coordinate alone = its phase only, alt alone = its sign only).
B. The linear decoder of a_S from the tape (multinomial logistic regression on c/|c|): accuracy split by configuration,
   and what it answers when it is wrong.
C. All steps: one linear decoder for a_S and a_M trained on the quiet states of every step t = 3..9 pooled, scored per step.
Output: d5_where.txt"""
from dec import *
from sklearn.neighbors import NearestNeighbors

out = Tee(__file__.replace(".py", ".txt"))
x, h, st, D = data()
B = len(x); n_train = int(0.75 * B); tr, te = slice(0, 100_000), slice(n_train, B)
c = tape_coords(h); M, S, T3 = st["M"], st["S"], st["T3"]
cols = {"alt": [0], "short": [1, 2], "h2": [3, 4], "h3": [5, 6]}
subsets = [("short",), ("h2",), ("h3",), ("alt",), ("short", "h3"), ("short", "h2"), ("h2", "h3"), ("short", "alt"),
           ("h2", "h3", "alt"), ("short", "h3", "alt"), ("short", "h2", "alt"), ("short", "h2", "h3"), ("alt", "short", "h2", "h3")]
targets = {"a_S": st["aS"], "a_M": st["aM"], "a_new=min": np.minimum(st["aS"], st["aM"]), "S before M": D["S_before_M"].astype(int),
           "(a_M, a_S)": st["aM"] * 10 + st["aS"], "a_T": st["aT"]}


def knn(sub):
    """one neighbour search per subset, then a majority vote of the 25 neighbours for every target."""
    X = c[:, sum((cols[m] for m in sub), [])]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    if X.shape[1] == 1:                                    # alt alone: only its sign is left (tiny jitter breaks the ties)
        X = np.hstack([X, np.zeros_like(X)]) + 1e-6 * np.random.default_rng(0).standard_normal((len(X), 2))
    idx = NearestNeighbors(n_neighbors=25).fit(X[tr]).kneighbors(X[te], return_distance=False)
    acc = []
    for y in targets.values():
        cls = np.unique(y); votes = np.stack([(y[tr][idx] == k).sum(1) for k in cls], 1)
        acc.append((cls[votes.argmax(1)] == y[te]).mean())
    return acc


out(f"quiet final states {B}; kNN: train 100000, test {B - n_train}")
out("")
out("A. accuracy of a kNN decoder from subsets of the tape coordinates (each subset divided by its own norm)")
maj = [max(np.mean(y[te] == v) for v in np.unique(y)) for y in targets.values()]
out(f"   {'coordinates':24s}" + "".join(f"{k:>12s}" for k in targets))
out(f"   {'majority class':24s}" + "".join(f"{v:12.3f}" for v in maj))
res = Parallel(n_jobs=6)(delayed(knn)(s) for s in subsets)
for s, r in zip(subsets, res):
    out(f"   {' + '.join(s):24s}" + "".join(f"{v:12.3f}" for v in r))

out("")
out("B. linear decoder of a_S from the tape (c/|c|), test states")
X = c / np.linalg.norm(c, axis=1, keepdims=True)
m = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=400)).fit(X[:n_train], st["aS"][:n_train])
pred = m.predict(X[te]); aS, aM, aT = st["aS"][te], st["aM"][te], st["aT"][te]
ok = pred == aS
gapT = ((S - T3) / M)[te]; gapM = ((M - S) / M)[te]; bef = D["S_before_M"][te]
out(f"   all {ok.mean():.3f}   S before M {ok[bef].mean():.3f}   S after M {ok[~bef].mean():.3f}")
out("   by the distance of the third below the runner-up, (S - T)/M:")
for lo, hi in ((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 99)):
    k = (gapT >= lo) & (gapT < hi)
    out(f"      {lo:4.2f}-{hi:5.2f}  n {k.sum():6d}  accuracy {ok[k].mean():.3f}   wrong answers that name the third's age {(pred == aT)[k & ~ok].mean():.3f}")
out("   by the gap between maximum and runner-up, (M - S)/M:")
for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 1)):
    k = (gapM >= lo) & (gapM < hi)
    out(f"      {lo:4.2f}-{hi:5.2f}  n {k.sum():6d}  accuracy {ok[k].mean():.3f}   wrong answers that name the maximum's age {(pred == aM)[k & ~ok].mean():.3f}")
out("   by a_S:  " + "  ".join(f"{a}: {ok[aS == a].mean():.3f}" for a in range(2, 10)))
w = ~ok
out(f"   the {w.sum()} wrong answers name: the third's age {(pred == aT)[w].mean():.3f}, the maximum's age {(pred == aM)[w].mean():.3f}, "
    f"a_S + 1 {(pred == aS + 1)[w].mean():.3f}, a_S - 1 {(pred == aS - 1)[w].mean():.3f}, a_S +- 2 {(np.abs(pred - aS) == 2)[w].mean():.3f}")
net_ok = (D["ans"] == st["pS"])[te]
out(f"   network's own answer correct on these states: {net_ok.mean():.3f};  decoder right where the network is right {ok[net_ok].mean():.3f}, where it is wrong {ok[~net_ok].mean():.3f};"
    f"  decoder names the network's (wrong) answer there: {(9 - pred == D['ans'][te])[~net_ok].mean():.3f}")

out("")
out("C. all steps t = 3..9: quiet states pooled (150,000 sequences; 75% of sequences train), one linear decoder, accuracy per step")
xs = sample(150_000, 12); Hs = run(xs); ntr = 112_500
feat, lab, step, seq = {k: [] for k in ("T", "R", "F")}, {"a_S": [], "a_M": []}, [], []
for t in range(3, 10):
    p = prefix_stats(xs, t); q = p["quiet"]; ht = Hs[q, t]
    l, _, _ = feats(ht)
    for k in feat: feat[k].append(l[k])
    lab["a_S"].append(p["aS"][q]); lab["a_M"].append(p["aM"][q]); step.append(np.full(q.sum(), t)); seq.append(np.flatnonzero(q))
feat = {k: np.vstack(v) for k, v in feat.items()}; lab = {k: np.concatenate(v) for k, v in lab.items()}
step, seq = np.concatenate(step), np.concatenate(seq); trn = seq < ntr
out(f"   quiet states: {len(step)} (per step: " + ", ".join(f"t={t}: {(step == t).sum()}" for t in range(3, 10)) + ")")


def pooled(k, nm):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=400)).fit(feat[k][trn], lab[nm][trn])
    hit = m.predict(feat[k][~trn]) == lab[nm][~trn]
    return [hit.mean()] + [hit[step[~trn] == t].mean() for t in range(3, 10)]


jobs = [(k, nm) for nm in lab for k in feat]
for (k, nm), r in zip(jobs, Parallel(n_jobs=6)(delayed(pooled)(k, nm) for k, nm in jobs)):
    out(f"   {nm} from {k}: all {r[0]:.3f} | per step t = 3..9: " + " ".join(f"{v:.3f}" for v in r[1:]))
