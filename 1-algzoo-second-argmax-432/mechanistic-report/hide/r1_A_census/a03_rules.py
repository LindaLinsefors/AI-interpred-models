"""A3: Boolean rules for every unit's gate, with confusion counts on 2^20 sequences.

Notation: gK(t) = gate of unit K at step t;  R*(t): x_t > max(M_{t-1}, 0);  N*(t): max(S_{t-1}, 0) < x_t < M_{t-1};
L*(t): x_t < max(S_{t-1}, 0);  M, S = largest and second largest of x_0..x_{t-1}.  At t = 0: R* iff x_0 > 0.
A rule "target <=> predictor" is scored on all steps where every lagged quantity exists.
"""
from cc import *
from a02_events import eff_class

rules = {}          # name -> [TP, TN, FP, FN]
order = []


def score(name, pred, truth, valid=None):
    if valid is not None:
        pred, truth = pred[valid], truth[valid]
    c = np.array([(pred & truth).sum(), (~pred & ~truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()])
    if name not in rules:
        rules[name] = np.zeros(4, dtype=np.int64); order.append(name)
    rules[name] += c


def lag(a, k):
    """a[:, t-k] aligned at t, for t >= k only (returns the (B, T-k) block)."""
    return a[:, : a.shape[1] - k] if k else a


first = True
for seed, x, H, Z in batches():
    ev = events(x); G = Z > 0
    g = lambda i: G[:, :, i]
    e = eff_class(x, ev)
    Rs, Ns, Ls = e == 0, e == 1, e == 2
    Mp0, Sp0 = np.maximum(ev["Mp"], 0), np.maximum(ev["Sp"], 0)
    allc_off = ~g(1) & ~g(6) & ~g(7) & ~g(8)
    anyc_off = ~(g(1) & g(6) & g(7) & g(8))

    # ---- unit 4 (truth: on at t), all t
    score("4 on(t) <=> R*(t)", Rs, g(4))
    score("4 on(t) <=> literal R(t) [t>=1]", (ev["cls"] == R_)[:, 1:], g(4)[:, 1:])
    score("4 on(t) <=> 7 off(t)", ~g(7), g(4))
    score("4 on(t) <=> 1,6,7,8 all off(t)", allc_off, g(4))
    # ---- comparators (truth: on at t)
    for j in COMP:
        score(f"{j} on(t) <=> not R*(t)", ~Rs, g(j))
        score(f"{j} on(t) <=> L*(t)", Ls, g(j))
        score(f"{j} on(t) <=> 4 off(t)", ~g(4), g(j))
    # constant threshold position u: comparator j on iff x_t < S+ + u (M+ - S+)
    if first:
        best_u = {}
        grid = np.round(np.arange(-0.2, 1.2001, 0.01), 2)
        for j in COMP + [4]:
            accs = [(((x < Sp0 + u * (Mp0 - Sp0)) == (g(j) if j != 4 else ~g(4)))[:, 1:]).mean() for u in grid]
            best_u[j] = grid[int(np.argmax(accs))]
        print("best constant threshold positions u_j (fitted on seed %d only): %s" % (seed, best_u))
    for j in COMP:
        score(f"{j} on(t) <=> x_t < S+ + {best_u[j]:.2f} (M+ - S+)  [u fitted on seed 1000; t>=1]",
              (x < Sp0 + best_u[j] * (Mp0 - Sp0))[:, 1:], g(j)[:, 1:])
    score(f"4 on(t) <=> x_t > S+ + {best_u[4]:.2f} (M+ - S+)  [u fitted on seed 1000; t>=1]",
          (x > Sp0 + best_u[4] * (Mp0 - Sp0))[:, 1:], g(4)[:, 1:])

    # ---- unit 9 (truth: on at t), t >= 1
    y = g(9)[:, 1:]
    score("9 on(t) <=> 4 on(t-1)", lag(g(4), 1), y)
    score("9 on(t) <=> R*(t-1)", lag(Rs, 1), y)
    score("9 on(t) <=> 7 off(t-1)", lag(~g(7), 1), y)
    score("9 on(t) <=> 7,8 off(t-1)", lag(~g(7) & ~g(8), 1), y)
    score("9 on(t) <=> 1,6,7,8 all off(t-1)", lag(allc_off, 1), y)
    score("9 on(t) <=> 4 on(t-1) and 8 off(t-1)", lag(g(4) & ~g(8), 1), y)
    score("9 on(t) <=> 4 on(t-1) and not R*(t)", lag(g(4), 1) & ~Rs[:, 1:], y)
    score("9 on(t) <=> R*(t-1) and not R*(t)", lag(Rs, 1) & ~Rs[:, 1:], y)
    # ---- unit 5 (truth: on at t), t >= 1
    y = g(5)[:, 1:]
    score("5 on(t) <=> R*(t-1) or N*(t-1)   [x_(t-1) > S+]", lag(~Ls, 1), y)
    score("5 on(t) <=> literal R or N at t-1 [t>=2]", lag(ev["cls"] != L_, 1)[:, 1:], y[:, 1:])
    score("5 on(t) <=> 8 off(t-1)", lag(~g(8), 1), y)
    score("5 on(t) <=> 6 off(t-1)", lag(~g(6), 1), y)
    score("5 on(t) <=> any comparator off(t-1)", lag(anyc_off, 1), y)
    score("5 on(t) <=> x_(t-1) > 0", lag(x > 0, 1), y)
    for c in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        score(f"5 on(t) <=> x_(t-1) > {c} S+ (and > 0)", lag((x > c * Sp0) & (x > 0), 1), y)
    # ---- unit 0 (truth: OFF at t)
    y = ~g(0)[:, 1:]
    score("0 off(t) <=> 7 on(t-1) and 9 on(t-1)", lag(g(7) & g(9), 1), y)
    score("0 off(t) <=> 9 on(t-1)", lag(g(9), 1), y)
    score("0 off(t) <=> 9 on(t-1) and 4 off(t-1)", lag(g(9) & ~g(4), 1), y)
    score("0 off(t) <=> 5 on(t-1) and 9 on(t-1) and 4 off(t-1)", lag(g(5) & g(9) & ~g(4), 1), y)
    y2 = ~g(0)[:, 2:]
    score("0 off(t) <=> 7 on(t-1) and 9 on(t-1)  [t>=2]", lag(g(7) & g(9), 1)[:, 1:], y2)
    score("0 off(t) <=> R*(t-2) and not R*(t-1)  [t>=2]", lag(Rs, 2) & ~lag(Rs, 1)[:, 1:], y2)
    score("0 off(t) <=> R*(t-2)  [t>=2]", lag(Rs, 2), y2)
    score("0 off(t) <=> 4 on(t-2) and 4 off(t-1)  [t>=2]", lag(g(4), 2) & ~lag(g(4), 1)[:, 1:], y2)
    score("0 off(t) <=> 4 on(t-2) and 7 on(t-1)  [t>=2]", lag(g(4), 2) & lag(g(7), 1)[:, 1:], y2)
    score("0 off(t) <=> a1(t) == 2  [t>=2]", (ev["a1"] == 2)[:, 2:], y2)
    # ---- wave units: 'always on from step t0'
    for i, t0 in ((2, 2), (13, 2), (13, 3), (12, 2), (14, 2), (14, 3), (10, 1), (11, 1), (3, 1), (15, 1), (0, 1)):
        yy = g(i)[:, t0:]
        score(f"{i} on(t) for all t >= {t0}", np.ones_like(yy), yy)
    # ---- wave units: single-age rules in the age of the max a1 (= steps since the last R event, literal)
    a1 = ev["a1"]
    for i, ages in ((11, (5,)), (10, (6,)), (10, (6, 7)), (12, (4,)), (14, (2,)), (15, (3, 4)), (15, (2, 3, 4, 5)), (3, (4, 5)), (3, (1, 2, 4, 5, 6))):
        score(f"{i} off(t) <=> a1(t) in {ages}  [t>=1]", np.isin(a1, ages)[:, 1:], ~g(i)[:, 1:])
    # delay-line rules found by search
    score("10 off(t) <=> 11 off(t-1)  [t>=2]", lag(~g(11), 1)[:, 1:], ~g(10)[:, 2:])
    score("11 off(t) <=> 12 off(t-1)  [t>=2]", lag(~g(12), 1)[:, 1:], ~g(11)[:, 2:])
    for k, i in ((2, 0), (5, 11), (6, 10), (4, 12)):
        score(f"{i} off(t) <=> 4 on(t-{k})  [t>={k}]", lag(g(4), k), ~g(i)[:, k:])
    first = False

rows = []
for name in order:
    tp, tn, fp, fn = rules[name]
    n = tp + tn + fp + fn
    rows.append([name, n, f"{(tp + tn) / n:.5f}", f"{max(tp + fn, tn + fp) / n:.4f}", tp, tn, fp, fn])
print(md_table(["rule (target <=> predictor)", "n steps", "accuracy", "base rate", "TP", "TN", "FP", "FN"], rows))
