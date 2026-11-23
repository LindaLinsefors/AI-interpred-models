"""E38. (A) Eq. (11) of tape.md again, with entry-level filters and every step t.
        (B) Can s_tape_t[9 - a] be ANY function of (x_(t-a), S_t, M_t)?  Hold the triple fixed, resample the rest.
Filters (per entry, at step t): the entry has age a >= 2; the maximum and the runner-up of x_0..x_t are both at least
two steps old; S_t > 0; (A only) the entry's input is below S_t.  No filter on sequences.
A state at step t is computed as the final state of (0, ..., 0, x_0, ..., x_t), which is the same state exactly.
Run:  python -B hide/e38_tape_fixed_triple.py > hide/outputs/e38_tape_fixed_triple.txt"""
from scipy.stats import norm

from common import *
from modes import RIGHT, coords

Uo = U[:, ORD]
def tape(xr):                                           # xr: (B, 10) right-aligned inputs -> tape piece (B, 10)
    c = coords(run(xr)[:, 9])
    G = np.real(np.multiply.outer(c["alt"], Uo @ RIGHT["alt"]))
    for n in ("short", "h2", "h3"):
        G = G + 2 * np.real(np.multiply.outer(c[n], Uo @ RIGHT[n]))
    return G
r2 = lambda y, yh: 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()

# ---------------------------------------------------------------- A. natural data, all steps
B = 300_000
x = sample(B, 3838)
rows = []                                               # per entry: t, a, aM, aS, x_entry, S, M, tape entry, tape at the runner-up
for t in range(4, 10):                                  # t >= 4 is needed for three different inputs of age >= 2
    xr = np.zeros((B, 10)); xr[:, 9 - t:] = x[:, :t + 1]
    G = tape(xr)
    pre = x[:, :t + 1]; order = np.argsort(-pre, 1); pM, pS = order[:, 0], order[:, 1]; ar = np.arange(B)
    M, S = pre[ar, pM], pre[ar, pS]; aM, aS = t - pM, t - pS
    ok = (aM >= 2) & (aS >= 2) & (S > 0)
    for a in range(2, t + 1):
        q = ok & (aM != a) & (aS != a)
        rows.append(np.stack([np.full(q.sum(), t), np.full(q.sum(), a), aM[q], aS[q], pre[q, t - a], S[q], M[q],
                              G[q, 9 - a], G[ar[q], 9 - aS[q]]], 1))
E = np.concatenate(rows)
t_, a_, aM_, aS_, xe, S_, M_, g, gS = E.T
below = xe < S_
print(f"A. {below.sum()} entries below the runner-up pass the filters ({B} sequences, steps t = 4..9)")
u = ((S_ - xe) / M_)[below]; lhs = ((g - gS) / M_)[below]
edges = np.concatenate([[0], np.geomspace(0.005, 6, 60)]); mid = 0.5 * (edges[:-1] + edges[1:]); k = np.digitize(u, edges) - 1
phi = np.array([-np.median(lhs[k == j]) if (k == j).sum() else np.nan for j in range(len(mid))])
rhs = -np.interp(u, mid, phi)
print(f"   eq. (11), Phi = median curve:  R2 = {r2(lhs, rhs):.3f},  correlation = {np.corrcoef(lhs, rhs)[0, 1]:.3f}")
print("   Phi at u = 0.01, 0.05, 0.1, 0.2, 0.4, 1, 2:", np.round(np.interp([0.01, 0.05, 0.1, 0.2, 0.4, 1, 2], mid, phi), 1))
print("   u bin       n      median   quartiles of (entry - runner-up's entry)/M     P(< 0)")
for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1), (1, 99)):
    q = (u > lo) & (u <= hi)
    print(f"   {lo:4.2f}-{hi:5.2f} {q.sum():8d}   {np.median(lhs[q]):6.1f}   {np.percentile(lhs[q], 25):6.1f} .. {np.percentile(lhs[q], 75):6.1f}     {np.mean(lhs[q] < 0):.3f}")

# ---------------------------------------------------------------- B. triple fixed, the rest resampled
print("\nB. Triple (x, S, M) held fixed. Configuration (step, ages of entry, maximum, runner-up) drawn from the entries of part A;")
print("   every other input drawn from N(0,1) truncated below S.  200,000 states per triple.  Values in units of M.")
rng = np.random.default_rng(38)
N = 200_000
cfg = E[rng.integers(0, len(E), N), :4].astype(int)     # t, a, aM, aS
key = cfg[:, 0] * 1000 + cfg[:, 1] * 100 + cfg[:, 2] * 10 + cfg[:, 3]
def experiment(xv, Sv, Mv):
    xr = np.zeros((N, 10))
    oth = norm.ppf(rng.uniform(0, 1, (N, 10)) * norm.cdf(Sv))          # N(0,1) given < S
    for t in range(4, 10):
        q = cfg[:, 0] == t
        xr[q, 9 - t:] = oth[q, 9 - t:]
    ar = np.arange(N)
    xr[ar, 9 - cfg[:, 1]] = xv; xr[ar, 9 - cfg[:, 2]] = Mv; xr[ar, 9 - cfg[:, 3]] = Sv
    G = tape(xr)
    return G[ar, 9 - cfg[:, 1]] / Mv, (G[ar, 9 - cfg[:, 1]] - G[ar, 9 - cfg[:, 3]]) / Mv
def spread(y):
    ks, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv); mean = np.bincount(inv, y) / cnt
    within = np.sqrt((np.bincount(inv, y ** 2) / cnt - mean ** 2).clip(0))
    big = cnt >= 30
    return y.mean(), y.std(), np.sqrt(np.average((mean[big] - y.mean()) ** 2, weights=cnt[big])), np.average(within[big], weights=cnt[big])
print("   M = 1.5 in all.  Columns: mean, total sd, sd between configurations, mean sd within a configuration (= from the other inputs' values only)")
print("     S     x     u=(S-x)/M |  the entry itself s_tape[9-a]/M        |  entry minus runner-up's entry, /M")
res = {}
for Sv in (0.5, 1.0):
    for uv in (0.02, 0.1, 0.3, 1.0):
        xv = Sv - uv * 1.5
        e, d = experiment(xv, Sv, 1.5)
        res[(Sv, uv)] = (e.mean(), d.mean())
        print(f"   {Sv:4.1f} {xv:6.2f}   {uv:5.2f}     | " + "  ".join(f"{v:7.2f}" for v in spread(e)) + "   | " + "  ".join(f"{v:7.2f}" for v in spread(d)))
print("   maximum's own entry and runner-up's own entry for comparison (x = M or x = S is not a separate input there): skipped")
print("\n   Signal for comparison: change of the mean when x moves from u = 0.02 to u = 1.0 at fixed S, M:")
for Sv in (0.5, 1.0):
    print(f"     S = {Sv}: entry itself {res[(Sv, 1.0)][0] - res[(Sv, 0.02)][0]:7.2f},  entry minus runner-up's {res[(Sv, 1.0)][1] - res[(Sv, 0.02)][1]:7.2f}")

# ---------------------------------------------------------------- C. eq. (11) with a separate Phi for each configuration
print("\nC. Natural data of part A again: Phi fitted as a median curve separately per configuration (24 bins of u each)")
ub = np.concatenate([[0], np.geomspace(0.01, 6, 24)]); kb = np.digitize(u, ub) - 1
tt, aa, am, as_ = (v[below].astype(int) for v in (t_, a_, aM_, aS_))
sm = (S_ / M_)[below]
def cell_r2(cell):
    code = np.unique(cell * 100 + kb, return_inverse=True)[1]
    order = np.argsort(code, kind="stable"); cs = code[order]; ys = lhs[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]); meds = np.array([np.median(s_) for s_ in np.split(ys, starts[1:])])
    pred = np.empty_like(lhs); pred[order] = np.repeat(meds, np.diff(np.r_[starts, len(cs)]))
    res = lhs - pred
    return r2(lhs, pred), res.std(), np.percentile(res, 25), np.percentile(res, 75), len(meds)
for name, cell in (("one Phi for all", np.zeros_like(aa)), ("per age of the entry a", aa), ("per (a, a_S)", aa * 10 + as_), ("per (a, a_M)", aa * 10 + am),
                   ("per (a, a_M, a_S)", aa * 100 + am * 10 + as_), ("per (t, a, a_M, a_S)", tt * 1000 + aa * 100 + am * 10 + as_),
                   ("per (a, a_M, a_S) and 3 bins of S/M", (aa * 100 + am * 10 + as_) * 10 + np.digitize(sm, [0.4, 0.7]))):
    q2, sd, q25, q75, nb = cell_r2(cell)
    print(f"   {name:38s} R2 = {q2:.3f}   residual sd {sd:5.2f} M, quartiles {q25:+.2f} / {q75:+.2f} M   ({nb} medians fitted)")
near = u < 0.1
print("   near the runner-up (u < 0.1), residual of the per-(a, a_M, a_S) curves as an error in the input (slope 71.5):")
code = np.unique((aa * 100 + am * 10 + as_) * 100 + kb, return_inverse=True)[1]
order = np.argsort(code, kind="stable"); cs = code[order]; starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
meds = np.array([np.median(s_) for s_ in np.split(lhs[order], starts[1:])]); pred = np.empty_like(lhs); pred[order] = np.repeat(meds, np.diff(np.r_[starts, len(cs)]))
res = (lhs - pred)[near]
print(f"     sd {res.std() / 71.5:.4f} M, quartiles {np.percentile(res, 25) / 71.5:+.4f} / {np.percentile(res, 75) / 71.5:+.4f} M   (one Phi for all: sd {(lhs - rhs)[near].std() / 71.5:.4f} M)")
