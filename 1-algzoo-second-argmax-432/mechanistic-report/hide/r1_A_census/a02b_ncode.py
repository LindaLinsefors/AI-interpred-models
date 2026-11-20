"""A2/A3 extra: (i) which comparators switch off at an N* event, by the age of the max; (ii) an age-dependent threshold
rule for the comparators and unit 4:   on/off by comparing x_t with  S+ + u_j(a) (M - S+),
a = age of the max in h_(t-1), u_j(a) = median threshold position at that age (fitted on seeds 1000, 1001 with the A5
selection S > 0, M - S > 0.2; scored on all steps t >= 1 of seeds 1002, 1003; when no positive input has occurred yet
the rule is 'comparators on, unit 4 off iff x_t < 0')."""
from cc import *
from a02_events import eff_class

UN = [8, 6, 1, 7, 4]
fitdata = {j: [[] for _ in range(9)] for j in UN}
test = []
ncode = {}
for seed, x, H, Z in batches():
    ev = events(x); e = eff_class(x, ev); G = Z > 0
    B = len(x)
    Hprev = np.concatenate([np.zeros((B, 1, N)), H[:, :-1]], 1)
    Mp, Sp = ev["Mp"], ev["Sp"]
    a1p = np.concatenate([np.full((B, 1), -1), ev["a1"][:, :-1]], 1)
    # (i) comparator code of N* events
    comp = sum(G[:, :, j].astype(int) << k for k, j in enumerate(COMP))
    isN = (e == 1); isN[:, 0] = False
    for a in range(9):
        m = isN & (a1p == a)
        c = ncode.setdefault(a, np.zeros((16, 2)))
        c[:, 0] += np.bincount(comp[m], minlength=16)
        c[:, 1] += np.bincount(comp[m], weights=G[:, :, 4][m], minlength=16)
    if seed <= 1001:
        with np.errstate(invalid="ignore"):
            sel = (Sp > 0) & (Mp - Sp > 0.2)
        sel[:, 0] = False
        for j in UN:
            th = -(Hprev @ W[j]) / w[j]
            with np.errstate(invalid="ignore", divide="ignore"):
                u = (th - Sp) / (Mp - Sp)
            for a in range(9):
                fitdata[j][a].append(u[sel & (a1p == a)])
    else:
        test.append((x, G, Mp, Sp, a1p))
ufit = {j: np.array([np.median(np.concatenate(fitdata[j][a])) for a in range(9)]) for j in UN}

print("## (i) comparator pattern at an N* event, by the age of the max in h_(t-1) (2^20 sequences)\n")
cset = lambda v: "{" + ",".join(str(j) for k, j in enumerate(COMP) if not (v >> k) & 1) + "}"
rows = []
for a in range(9):
    c = ncode[a]; n = c[:, 0].sum(); o = np.argsort(-c[:, 0])[:4]
    rows.append([a, int(n), f"{c[:, 1].sum() / n:.4f}"] + [f"off{cset(v)} {c[v, 0] / n:.3f}" for v in o])
print(md_table(["age of max in h_(t-1)", "n (N* steps)", "P(4 on)", "most common comparator OFF set", "2nd", "3rd", "4th"], rows))

print("\n## (ii) age-dependent threshold rule\n")
print("fitted median positions u_j(a), a = 0..8 (seeds 1000, 1001):\n")
print(md_table(["unit"] + [f"a={a}" for a in range(9)], [[j] + [f"{v:+.3f}" for v in ufit[j]] for j in UN]))
rows = []
for j in UN:
    tp = tn = fp = fn = 0
    for x, G, Mp, Sp, a1p in test:
        Mp0, Sp0 = np.maximum(Mp, 0), np.maximum(Sp, 0)
        thr = Sp0 + ufit[j][np.clip(a1p, 0, 8)] * (Mp0 - Sp0)
        thr = np.where(Mp > 0, thr, 0.0)                       # no positive input yet: threshold 0
        below = (x < thr)[:, 1:]
        pred = below if j in COMP else ~below
        truth = G[:, 1:, j]
        tp += (pred & truth).sum(); tn += (~pred & ~truth).sum(); fp += (pred & ~truth).sum(); fn += (~pred & truth).sum()
    n = tp + tn + fp + fn
    name = f"{j} on(t) <=> x_t < S+ + u_{j}(a)(M - S+)" if j in COMP else "4 on(t) <=> x_t > S+ + u_4(a)(M - S+)"
    rows.append([name, n, f"{(tp + tn) / n:.5f}", f"{max(tp + fn, tn + fp) / n:.4f}", tp, tn, fp, fn])
print()
print(md_table(["rule (scored on seeds 1002, 1003, steps t >= 1)", "n steps", "accuracy", "base rate", "TP", "TN", "FP", "FN"], rows))
