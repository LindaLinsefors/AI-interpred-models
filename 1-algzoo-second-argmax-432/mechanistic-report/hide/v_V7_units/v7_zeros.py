"""V7 attack 1: what are the exact zeros of units 0, 12, 11, 10 tied to?

Alternatives tested:
  (a) the age a_1 of the maximum (lead's reading);
  (b) the time since the last record-like event of any kind: new runner-up (a_2 < a_1), or a firing of unit 4
      that is not the maximum ("misfire");
  (c) only present when there was something before the maximum (max = first positive input);
  (d) only for the lead's selection (M > 0.3, a_2 >= 2), not for small M or a fresh runner-up;
  (e) an artefact of pooling over t.
Seeds 5011..5015, 400k sequences each, iid N(0,1).
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *

SEEDS = [5011, 5012, 5013, 5014, 5015]
NB = 400_000
PAIRS = [(0, 2), (12, 4), (11, 5), (10, 6)]
UNITS = [0, 12, 11, 10]


def collect(seed):
    x = sample(NB, seed)
    H = run(x)
    rs = running_stats(x)
    tt = np.arange(T)[None, :]
    fired = H[:, :, 4] > 0
    last4 = np.maximum.accumulate(np.where(fired, tt, -1), axis=1)
    e4 = np.where(last4 >= 0, tt - last4, -1)                      # steps since unit 4 last fired
    npos_cum = np.cumsum(x > 0, axis=1)                             # positives in x_0..x_t
    p1 = rs["p1"]
    npos_before = np.where(p1 > 0, np.take_along_axis(npos_cum, np.maximum(p1 - 1, 0), 1), 0)   # positives before p1
    npos_after = npos_cum - np.take_along_axis(npos_cum, p1, 1)     # positives after p1 up to t
    keep = rs["a1"] >= 1
    b, t = np.nonzero(keep)
    S = rs["M2"][b, t]
    M = rs["M1"][b, t]
    return dict(zero=(H[b, t][:, UNITS] == 0), M=M, S=S, a1=rs["a1"][b, t], a2=rs["a2"][b, t], a3=rs["a3"][b, t],
                t=t, e4=e4[b, t], nb=npos_before[b, t], na=npos_after[b, t],
                fire_now=fired[b, t])


parts = [collect(s) for s in SEEDS]
D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
n_all = D["M"].size
print(f"seeds {SEEDS}, {NB} sequences each; states with a1 >= 1: {n_all}")
big = D["M"] > 0.3
g = np.where(D["M"] > 0, (D["M"] - np.maximum(D["S"], 0)) / np.where(D["M"] > 0, D["M"], 1), np.nan)


def frac(mask, k):
    n = int(mask.sum())
    return (D["zero"][mask, k].mean() if n else np.nan), n


def line(label, mask):
    out = f"  {label:58s}"
    for k, (u, a) in enumerate(PAIRS):
        f, n = frac(mask & (D["a1"] == a), k)
        out += f"  u{u}@a1={a}: {f:6.3f} (n={n:7d})"
    print(out)


print("\n===== 0. the lead's selection and what it leaves out (all t pooled) =====")
line("lead: M>0.3, a2>=2 (all g)", big & (D["a2"] >= 2))
line("M>0.3, a2 = 1 (runner-up arrived one step ago)", big & (D["a2"] == 1))
line("M>0.3, a2 = 0 (runner-up is the current input)", big & (D["a2"] == 0))
line("0.1 < M <= 0.3, a2>=2", (D["M"] > 0.1) & ~big & (D["a2"] >= 2))
line("0 < M <= 0.1, a2>=2", (D["M"] > 0) & (D["M"] <= 0.1) & (D["a2"] >= 2))
line("M <= 0 (all inputs so far negative), a2>=2", (D["M"] <= 0) & (D["a2"] >= 2))
line("everything with a1 = a* (no selection at all)", np.ones(n_all, bool))

print("\n===== 1. by gap g = (M - max(S,0))/M, M>0.3, a2>=2, all t pooled =====")
edges = [0, 0.02, 0.05, 0.15, 0.30, 0.45, 0.60, 0.85, 0.999999]
for lo, hi in zip(edges[:-1], edges[1:]):
    line(f"g in ({lo:.2f}, {hi:.2f}]", big & (D["a2"] >= 2) & (g > lo) & (g <= hi))
line("g = 1 exactly (S <= 0: max is the only positive input)", big & (D["a2"] >= 2) & (D["S"] <= 0))

print("\n===== 2. by t (M>0.3, a2>=2) =====")
for t in range(2, 10):
    line(f"t = {t}", big & (D["a2"] >= 2) & (D["t"] == t))

print("\n===== 3. P(unit == 0) as a function of (a1, a2): rows a1 = 1..9, columns a2 = 0..9; M>0.3, all t pooled =====")
print("       (if the zero belonged to the latest event, it would follow min(a1, a2); if to the runner-up, a column)")
for k, (u, a) in enumerate(PAIRS):
    print(f"  unit {u} (claimed zero at a1 = {a}); '  .  ' = fewer than 300 states")
    print("        a2:" + "".join(f"{c:7d}" for c in range(10)))
    for r in range(1, 10):
        row = f"    a1={r}  "
        for c in range(10):
            f, n = frac(big & (D["a1"] == r) & (D["a2"] == c), k)
            row += f"{f:7.3f}" if n >= 300 else "    .  "
        print(row)

print("\n===== 4. runner-up's age instead of the max's: P(unit == 0 | a2 = a*, a1 != a*), M>0.3, S>0.3 =====")
for k, (u, a) in enumerate(PAIRS):
    f1, n1 = frac(big & (D["S"] > 0.3) & (D["a2"] == a) & (D["a1"] > a), k)
    f2, n2 = frac(big & (D["S"] > 0.3) & (D["a2"] == a) & (D["a1"] < a) & (D["a1"] >= 1), k)
    f3, n3 = frac(big & (D["a1"] != a), k)
    print(f"  unit {u}: a2={a} and a1>{a} (runner-up is the latest of the two): {f1:.3f} (n={n1});  a2={a} and a1<{a}: {f2:.3f} (n={n2});"
          f"  any state with a1 != {a}: {f3:.3f} (n={n3})")

print("\n===== 5. unit-4 firings (the network's own 'record' signal) =====")
m_ok = big
mis = m_ok & (D["e4"] >= 0) & (D["e4"] < D["a1"])       # unit 4 fired after the max arrived -> misfire
nofire = m_ok & (D["e4"] > D["a1"]) | (m_ok & (D["e4"] < 0))   # unit 4 did not fire at the max's step
print(f"  states (M>0.3, a1>=1): {int(m_ok.sum())};  unit 4 last fired exactly at the max's step: {(m_ok & (D['e4'] == D['a1'])).mean() / m_ok.mean():.4f};"
      f"  fired later (misfire): {mis.sum() / m_ok.sum():.4f};  did not fire at the max: {nofire.sum() / m_ok.sum():.4f}")
for k, (u, a) in enumerate(PAIRS):
    f0, n0 = frac(m_ok & (D["a1"] == a) & (D["e4"] == a), k)
    out = f"  unit {u}: a1={a} & e4={a} (clean): {f0:.3f} (n={n0})"
    for c in range(0, a):
        f, n = frac(m_ok & (D["a1"] == a) & (D["e4"] == c), k)
        out += f" | a1={a} & e4={c}: {f:.3f} (n={n})"
    print(out)
    f1, n1 = frac(m_ok & (D["e4"] == a) & (D["a1"] > a), k)
    f2, n2 = frac(m_ok & (D["a1"] == a) & ((D["e4"] > a) | (D["e4"] < 0)), k)
    print(f"           misfire has age {a} but the max is older (e4={a}, a1>{a}): {f1:.3f} (n={n1});   a1={a} but unit 4 did not fire at the max: {f2:.3f} (n={n2})")

print("\n  split of the runner-up's own age: P(unit == 0 | a2 = a*, a1 > a*) when unit 4 fired at the runner-up's arrival (misfire, e4 = a2)")
print("  and when it did not (e4 = a1); M>0.3, S>0.3")
for k, (u, a) in enumerate(PAIRS):
    base = m_ok & (D["S"] > 0.3) & (D["a2"] == a) & (D["a1"] > a)
    f1, n1 = frac(base & (D["e4"] == a), k)
    f2, n2 = frac(base & (D["e4"] == D["a1"]), k)
    out = f"  unit {u}: runner-up {a} old, max older: with misfire at the runner-up {f1:.3f} (n={n1}); without {f2:.3f} (n={n2});  by a1: "
    for b in range(a + 1, 10):
        g1, k1 = frac(base & (D["a1"] == b) & (D["e4"] == a), k)
        g2, k2 = frac(base & (D["a1"] == b) & (D["e4"] == D["a1"]), k)
        out += f" a1={b}: {g1:.2f} (n={k1}) / {g2:.2f} (n={k2});"
    print(out)

print("\n===== 6. the maximum is the first positive input (no positive input before it), M>0.3 =====")
fp = big & (D["nb"] == 0)
line("first positive, any later inputs, a2 any", fp)
line("first positive AND no positive input after it (g = 1)", fp & (D["na"] == 0))
line("first positive AND >= 1 positive after it", fp & (D["na"] >= 1))
line("first positive AND max is x_0 (nothing before)", fp & (D["t"] == D["a1"]))
line("first positive AND >= 1 negative before", fp & (D["t"] > D["a1"]))
line("NOT first positive (>= 1 positive before)", big & (D["nb"] >= 1))
line("NOT first positive, a2 >= 2", big & (D["nb"] >= 1) & (D["a2"] >= 2))

print("\n===== 7. t = 9 only (the states the readout sees), M>0.3 =====")
t9 = big & (D["t"] == 9)
line("t=9, a2>=2", t9 & (D["a2"] >= 2))
line("t=9, a2 = 1", t9 & (D["a2"] == 1))
line("t=9, a2 = 0", t9 & (D["a2"] == 0))
line("t=9, all a2", t9)
line("t=9, all a2, no M selection", D["t"] == 9)
