"""d1: which global quantities can be decoded from the tape (T), the rest (R) and the full state (F)?
Quiet final states (t = 9). See dec.py for sources, decoders and scores. Output: d1_headline.txt"""
from dec import *

out = Tee(__file__.replace(".py", ".txt"))
x, h, st, D = data()
lin, flex, norms = feats(h)
B = len(x); n_train = int(0.75 * B)
rng = np.random.default_rng(0)
M, S, T3 = st["M"], st["S"], st["T3"]
Tp = np.maximum(T3, 0)
out(f"quiet final states: {B} of {N_SEQ} sequences; train {n_train} (flexible decoder: first {N_FLEX}), test {B - n_train}")
out(f"S before M: {D['S_before_M'].mean():.3f};  third within 0.1 M of S: {((S - T3) / M < 0.1).mean():.3f};  T3 > 0: {(T3 > 0).mean():.3f}")

after = ~D["S_before_M"]
tasks = [
    ("age of runner-up a_S (8 cl)", "cls", st["aS"], None),
    ("age of maximum a_M (8 cl)", "cls", st["aM"], None),
    ("age of third a_T (10 cl)", "cls", st["aT"], None),
    ("S arrived before M (2 cl)", "cls", D["S_before_M"].astype(int), None),
    ("network's answer (10 cl)", "cls", D["ans"], None),
    ("answer is correct (2 cl)", "cls", (D["ans"] == st["pS"]).astype(int), None),
    ("n within 0.10M of S: 0/1/2+", "cls", np.minimum(D["n10"], 2), None),
    ("n within 0.25M of S: 0/1/2/3+", "cls", np.minimum(D["n25"], 3), None),
    ("number of records 1/2/3/4+", "cls", np.minimum(D["nrec"], 4), None),
    ("age last overtaken max | S after M", "cls", D["a_prev"], after & (D["a_prev"] >= 0)),
    ("CONTROL age of smallest (10 cl)", "cls", D["a_min"], None),
    ("CONTROL shuffled a_S", "cls", rng.permutation(st["aS"]), None),
    ("S/M", "reg", S / M, None),
    ("T+/M   (T+ = max(T,0))", "reg", Tp / M, None),
    ("T+/S", "reg", Tp / S, None),
    ("(S - T+)/M", "reg", (S - Tp) / M, None),
    ("CONTROL min/M (clipped at -5)", "reg", np.maximum(D["min"], -5), None),
    ("CONTROL shuffled S/M", "reg", rng.permutation(S / M), None),
]
sc, base = table(tasks, lin, flex, n_train, out=out)

# ---- values M, S, T+ from RAW coordinates
out("\nVALUES from raw coordinates (not scale-free). R2 on test. 'norm' = c * |source| alone; 'lin' = affine least squares")
out("on the raw coordinates; 'flex' = |source| * g(source / |source|), g = gradient boosting (a homogeneous decoder).")
raw = {"T": tape_coords(h), "R": rest_coords(h), "F": h}


def value_scores(y, s):
    tr, te = slice(0, n_train), slice(n_train, B)
    n = np.linalg.norm(raw[s], axis=1)
    r2 = lambda p: 1 - ((y[te] - p) ** 2).mean() / y[te].var()
    c = (n[tr] @ y[tr]) / (n[tr] @ n[tr])
    l = LinearRegression().fit(raw[s][tr], y[tr]).predict(raw[s][te])
    g = HistGradientBoostingRegressor(max_iter=300, early_stopping=True, random_state=0).fit(flex[s][:N_FLEX], (y / n)[:N_FLEX])
    return r2(c * n[te]), r2(l), r2(n[te] * g.predict(flex[s][te]))


vals = [("M", M), ("S", S), ("T+ = max(T,0)", Tp), ("M - S", M - S), ("S - T+", S - Tp)]
res = Parallel(n_jobs=8)(delayed(value_scores)(y, s) for _, y in vals for s in ("T", "R", "F"))
out(f"{'value':16s} | " + " | ".join(f"{s + ' norm':>7s} {s + ' lin':>7s} {s + ' flex':>7s}" for s in ("T", "R", "F")))
for i, (nm, _) in enumerate(vals):
    out(f"{nm:16s} | " + " | ".join("%7.3f %7.3f %7.3f" % res[3 * i + j] for j in range(3)))
