"""A3 (wave units): when is each wave unit off?

sR = steps since the last R* event (new positive maximum; 0 at the event), sN = steps since the last N* event
(new positive runner-up; 'none' if there was none).  Regular steps: t >= 1 and M_t > 0 (so sR is defined).
Rule ladder for every wave unit (lookup tables fitted on seeds 1000-1001, scored on seeds 1002-1003):
  (i) always on;  (ii) lookup on sR;  (iii) lookup on (sR, sN);  (iv) lookup on (sR, sN, decile bin of S+/M).
A lookup predicts 'off' in a cell iff the unit is off in more than half of the training steps of that cell.
"""
from cc import *
from a02_events import eff_class

WV = WAVE
data = []
for seed, x, H, Z in batches():
    ev = events(x); e = eff_class(x, ev)
    B = len(x); tt = np.broadcast_to(np.arange(T), (B, T))
    lastR = np.maximum.accumulate(np.where(e == 0, tt, -1), 1)
    lastN = np.maximum.accumulate(np.where(e == 1, tt, -1), 1)
    sR = np.where(lastR >= 0, tt - lastR, -1); sN = np.where(lastN >= 0, tt - lastN, -1)
    M, S = ev["M"], ev["S"]
    sm = np.clip(np.where(M > 0, np.maximum(S, 0) / np.where(M > 0, M, 1), 0), 0, 0.999999)
    reg = (tt >= 1) & (M > 0)
    data.append(dict(off=~(Z[:, :, WV] > 0)[reg], sR=sR[reg], sN=sN[reg], bin=(sm[reg] * 10).astype(int),
                     seed=np.full(reg.sum(), seed)))
D = {k: np.concatenate([d[k] for d in data]) for k in data[0]}
n = len(D["sR"])
train = D["seed"] <= 1001; test = ~train
print(f"regular steps (t >= 1, M_t > 0): {n};  train {train.sum()}, test {test.sum()}\n")

print("## A3.w1 P(unit off | sR) and P(unit off | sN), regular steps, all four seeds\n")
rows = [[f"sR={a}", int((D['sR'] == a).sum())] + [f"{D['off'][D['sR'] == a, k].mean():.3f}" for k in range(len(WV))] for a in range(10)]
print(md_table(["", "n"] + [f"unit {i}" for i in WV], rows))
print()
rows = [[("no N* yet" if a < 0 else f"sN={a}"), int((D['sN'] == a).sum())] + [f"{D['off'][D['sN'] == a, k].mean():.3f}" for k in range(len(WV))] for a in range(-1, 9)]
print(md_table(["", "n"] + [f"unit {i}" for i in WV], rows))

print("\n## A3.w2 P(unit off | sR, sN) (rows sR = 0..9; columns sN = none, 0..8; '.' if n < 1000)\n")
for i in (3, 10, 11, 15, 12, 14, 13, 0, 2):
    k = WV.index(i)
    print(f"\nunit {i}\n")
    rows = []
    for a in range(10):
        r = [f"sR={a}"]
        for b in range(-1, 9):
            m = (D["sR"] == a) & (D["sN"] == b)
            r.append(f"{D['off'][m, k].mean():.2f}" if m.sum() >= 1000 else ".")
        rows.append(r)
    print(md_table([""] + ["none"] + [str(b) for b in range(9)], rows))

print("\n## A3.w3 rule ladder: accuracy on the test seeds (1002, 1003)\n")


def lookup_acc(keys_train, keys_test, y_train, y_test, size):
    cnt1 = np.bincount(keys_train[y_train], minlength=size); cnt0 = np.bincount(keys_train[~y_train], minlength=size)
    pred = (cnt1 > cnt0)[keys_test]
    return (pred == y_test).mean(), pred


k1 = D["sR"]; k2 = D["sR"] * 10 + (D["sN"] + 1); k3 = k2 * 10 + D["bin"]
rows = []
for k, i in enumerate(WV):
    y = D["off"][:, k]
    base = max(y[test].mean(), 1 - y[test].mean())
    a1, _ = lookup_acc(k1[train], k1[test], y[train], y[test], 10)
    a2, _ = lookup_acc(k2[train], k2[test], y[train], y[test], 100)
    a3, _ = lookup_acc(k3[train], k3[test], y[train], y[test], 1000)
    cnt1 = np.bincount(k1[train][y[train]], minlength=10); cnt0 = np.bincount(k1[train][~y[train]], minlength=10)
    offset = [a for a in range(10) if cnt1[a] > cnt0[a]]
    rows.append([i, f"{y[test].mean():.4f}", f"{base:.4f}", f"{a1:.4f}", str(offset) if offset else "never", f"{a2:.4f}", f"{a3:.4f}"])
print(md_table(["unit", "P(off)", "(i) always on", "(ii) lookup on sR", "sR values predicted off", "(iii) lookup on (sR, sN)",
                "(iv) lookup on (sR, sN, S+/M decile)"], rows))

print("\n## A3.w4 single-age rules with confusion counts (regular steps, all four seeds)\n")
rows = []
for i, ages in ((0, (2,)), (11, (5,)), (10, (6,)), (10, (6, 7)), (12, (4,)), (14, (0, 2)), (15, (2, 3, 4)), (3, (2, 4, 5))):
    y = D["off"][:, WV.index(i)]; pred = np.isin(D["sR"], ages)
    tp = (pred & y).sum(); tn = (~pred & ~y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum()
    rows.append([f"{i} off(t) <=> sR(t) in {ages}", n, f"{(tp + tn) / n:.5f}", f"{max(y.mean(), 1 - y.mean()):.4f}", tp, tn, fp, fn])
print(md_table(["rule", "n steps", "accuracy", "base rate", "TP", "TN", "FP", "FN"], rows))
