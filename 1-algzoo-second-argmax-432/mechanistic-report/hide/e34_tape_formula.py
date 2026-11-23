"""E34. A formula for the tape piece of the scores, eq. (4a) of tape.md, as a function of the inputs.
D[i] = s_tape[i] - s_tape[i_S]  (entry minus the runner-up's entry), final states, indices 0..7,
maximum and runner-up at least two steps old, M > 0.3, S > 0.1.
Run:  python -B hide/e34_tape_formula.py > hide/outputs/e34_tape_formula.txt"""
from scipy.optimize import least_squares

from common import *
from modes import RIGHT, coords

B = 1_000_000
x = sample(B, 3434)
h = run(x)[:, 9]
c = coords(h)
Uo = U[:, ORD]
G = np.real(np.multiply.outer(c["alt"], Uo @ RIGHT["alt"]))
for n in ("short", "h2", "h3"):
    G = G + 2 * np.real(np.multiply.outer(c[n], Uo @ RIGHT[n]))
order = np.argsort(-x, 1)
pM, pS = order[:, 0], order[:, 1]
ar = np.arange(B)
M, S = x[ar, pM], x[ar, pS]
ok = (pM <= 7) & (pS <= 7) & (M > 0.3) & (S > 0.1)
D = G[:, :8] - G[ar, pS][:, None]
idx = np.broadcast_to(np.arange(8), (B, 8))
Mb, Sb, aMb, aSb = (np.broadcast_to(v[:, None], (B, 8)) for v in (M, S, 9 - pM, 9 - pS))
below = ok[:, None] & (x[:, :8] < S[:, None])
med = lambda v: float(np.median(v)) if len(v) else float("nan")
r2 = lambda y, yh: 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()

print("A. Does the saturation scale with M or with S?  median of D by depth below the runner-up, three bins of S/M")
sm = Sb / Mb
for label, depth, scale in (("depth (S - x)/M, D in units of M", (Sb - x[:, :8]) / Mb, Mb), ("depth (S - x)/S, D in units of S", (Sb - x[:, :8]) / Sb, Sb)):
    print("  " + label)
    edges = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0, 2.0, 99]
    print("    depth bin   " + "".join(f"{f'S/M {lo}-{hi}':>14s}" for lo, hi in ((0.1, 0.4), (0.4, 0.7), (0.7, 1.0))))
    for lo, hi in zip(edges[:-1], edges[1:]):
        row = [med((D / scale)[below & (depth > lo) & (depth <= hi) & (sm > a) & (sm <= b)]) for a, b in ((0.1, 0.4), (0.4, 0.7), (0.7, 1.0))]
        print(f"    {lo:4.2f}-{hi:5.2f}  " + "".join(f"{v:14.1f}" for v in row))

# ---- B. one formula for the inputs below the runner-up:  D = -k min(d, L) - h max(d - L, 0),  d = S - x,  L = lM M + lS S
rng = np.random.default_rng(0)
sel = np.flatnonzero(below.ravel()); sub = rng.choice(sel, 400_000, replace=False)
d, y, Ms, Ss, ai, aM, aS = ((Sb - x[:, :8]).ravel()[sub], D.ravel()[sub], Mb.ravel()[sub], Sb.ravel()[sub],
                             (9 - idx).ravel()[sub], aMb.ravel()[sub], aSb.ravel()[sub])
def model(p, d, Ms, Ss):
    L = np.maximum(p[1] * Ms + p[2] * Ss, 0)
    return -p[0] * np.minimum(d, L) - p[3] * np.maximum(d - L, 0)
def fit(q, free=(1, 1)):
    f = lambda p: model([p[0], p[1] * free[0], p[2] * free[1], p[3]], d[q], Ms[q], Ss[q]) - y[q]
    p = least_squares(f, [70, 0.1, 0.1, 3]).x
    p = [p[0], p[1] * free[0], p[2] * free[1], p[3]]
    return p, r2(y[q], model(p, d[q], Ms[q], Ss[q]))
allq = np.ones(len(y), bool)
print("\nB. D = -k min(S - x, L) - h max(S - x - L, 0),  L = lM M + lS S     (400k entries below the runner-up)")
for name, free in (("L = lM M + lS S", (1, 1)), ("L = lM M", (1, 0)), ("L = lS S", (0, 1))):
    p, q2 = fit(allq, free)
    print(f"   {name:16s}: k = {p[0]:6.1f}, lM = {p[1]:6.3f}, lS = {p[2]:6.3f}, h = {p[3]:5.2f};  R2 = {q2:.3f}")
p0, _ = fit(allq)
res = y - model(p0, d, Ms, Ss)
print("   residual of the first fit (median, in units of M) by order of arrival relative to the top two:")
i_ = 9 - ai; lo_, hi_ = np.minimum(9 - aM, 9 - aS), np.maximum(9 - aM, 9 - aS)
for name, q in (("after both", i_ > hi_), ("between", (i_ > lo_) & (i_ < hi_)), ("before both", i_ < lo_)):
    print(f"     {name:12s} {med(res[q] / Ms[q]):6.2f}")
print("   the same formula fitted separately for each age of the entry:")
print("     age      k      lM      lS      h     R2")
yh = np.zeros_like(y)
for a in range(2, 10):
    q = ai == a
    p, q2 = fit(q); yh[q] = model(p, d[q], Ms[q], Ss[q])
    print(f"     {a}   {p[0]:6.1f}  {p[1]:6.3f}  {p[2]:6.3f}  {p[3]:5.2f}  {q2:.3f}")
print(f"   R2 over all ages with per-age parameters: {r2(y, yh):.3f}")
print("   per age of the entry AND position relative to the top two (after both / between / before both):")
yh2 = np.zeros_like(y)
for a in range(2, 10):
    for name, oq in (("after", i_ > hi_), ("between", (i_ > lo_) & (i_ < hi_)), ("before", i_ < lo_)):
        q = (ai == a) & oq
        if q.sum() > 2000:
            p, q2 = fit(q); yh2[q] = model(p, d[q], Ms[q], Ss[q])
            print(f"     age {a} {name:8s} n = {q.sum():6d}  k = {p[0]:6.1f}  lM = {p[1]:6.3f}  lS = {p[2]:6.3f}  h = {p[3]:5.2f}  R2 = {q2:.3f}")
        else:
            yh2[q] = yh[q]
print(f"   R2 over all with per-(age, order) parameters: {r2(y, yh2):.3f}")

# ---- C. the maximum's entry:  D[i_M] = alpha M + beta S, in cells
q = ok
yM = D[ar[q], pM[q]]; Mq, Sq, am, as_ = M[q], S[q], 9 - pM[q], 9 - pS[q]
def cellfit(key):
    yh = np.zeros_like(yM); tab = {}
    for kv in np.unique(key):
        s = key == kv
        X = np.stack([Mq[s], Sq[s]], 1); b = np.linalg.lstsq(X, yM[s], rcond=None)[0]
        yh[s] = X @ b; tab[int(kv)] = b
    return r2(yM, yh), tab
print("\nC. The maximum's entry, D[i_M] = alpha M + beta S fitted per cell;  R2 over all cells together")
for name, key in (("one cell", np.zeros_like(am)), ("age of the maximum (8 cells)", am), ("signed distance i_S - i_M (14 cells)", (am - as_) + 20),
                  ("age of maximum x age of runner-up (56 cells)", am * 10 + as_)):
    q2, tab = cellfit(key)
    print(f"   {name:46s} R2 = {q2:.3f}")
    if "signed" in name:
        print("     i_S - i_M:  " + " ".join(f"{k - 20:+6d}" for k in tab)); print("     alpha:      " + " ".join(f"{b[0]:6.1f}" for b in tab.values()))
        print("     beta:       " + " ".join(f"{b[1]:6.1f}" for b in tab.values()))
    if "56" in name:
        for lab, j in (("alpha", 0), ("beta", 1)):
            print(f"     {lab}, rows = age of the maximum 2..9, columns = age of the runner-up 2..9")
            for a in range(2, 10):
                print("       " + " ".join(f"{tab[a * 10 + b][j]:6.1f}" if a != b else "     ." for b in range(2, 10)))
