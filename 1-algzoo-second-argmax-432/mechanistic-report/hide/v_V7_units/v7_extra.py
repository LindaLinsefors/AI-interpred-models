"""V7 extra tests.
  1. deterministic single-maximum inputs (the max is the first positive input): where are units 0, 12, 11, 10 zero?
  2. how far below zero is the pre-activation at the matched age (is the cut robust or marginal)?
  3. a new maximum has just arrived (a_1 = 0): do the zeros now follow the age of the OLD maximum (a_2)?
  4. functional test on the network's own answer: x_9 is the largest input, so the answer must be the old maximum.
     Are the comparators off (answer = top of m)?  Does forcing the silent unit on change the ANSWER?
Seeds 5041..5045, 400k each.
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *

PAIRS = [(0, 2), (12, 4), (11, 5), (10, 6)]
UNITS = [0, 12, 11, 10]

# ---------------------------------------------------------------- 1
print("===== 1. single maximum, deterministic: h_t[j] (pre-activation in brackets) by age of the max =====")
for name, x in [("x = (1, 0, 0, ..., 0)", np.r_[1.0, np.zeros(9)]),
                ("x = (1, -1, -1, ..., -1)", np.r_[1.0, -np.ones(9)]),
                ("x = (-1, -1, 1, -1, ..., -1)", np.r_[-1.0, -1.0, 1.0, -np.ones(7)]),
                ("x = (1, -0.3, -0.3, ..., -0.3)", np.r_[1.0, -0.3 * np.ones(9)]),
                ("x = (1, -3, -3, ..., -3)", np.r_[1.0, -3 * np.ones(9)])]:
    H, Z = run(x, return_pre=True)
    p = int(np.argmax(x))
    print(f"  {name}")
    for j in UNITS:
        print(f"    unit {j:2d}: " + "  ".join(f"a={t - p}: {H[0, t, j]:6.2f} ({Z[0, t, j]:6.2f})" for t in range(p, 10)))
    print(f"    zero ages: " + ", ".join(f"unit {j}: {[t - p for t in range(p + 1, 10) if H[0, t, j] == 0]}" for j in UNITS))

# ---------------------------------------------------------------- data
SEEDS = [5041, 5042, 5043, 5044, 5045]
NB = 400_000
acc = {k: [] for k in ("z", "M", "S", "a1", "a2", "t", "h9", "h8", "x")}
for sd in SEEDS:
    x = sample(NB, sd)
    H, Z = run(x, return_pre=True)
    rs = running_stats(x)
    b, t = np.nonzero(rs["M1"] > 0.3)
    acc["z"].append(Z[b, t][:, UNITS].astype(np.float32)); acc["M"].append(rs["M1"][b, t]); acc["S"].append(rs["M2"][b, t])
    acc["a1"].append(rs["a1"][b, t]); acc["a2"].append(rs["a2"][b, t]); acc["t"].append(t)
    acc["h9"].append(H[:, 9]); acc["h8"].append(H[:, 8]); acc["x"].append(x)
D = {k: np.concatenate(v) for k, v in acc.items()}
del acc
print(f"\nseeds {SEEDS}, {NB} each; states with M > 0.3: {D['M'].size}")

# ---------------------------------------------------------------- 2
print("\n===== 2. pre-activation z_j / M at the matched age (M > 0.3, a2 >= 2) =====")
for k, (j, a) in enumerate(PAIRS):
    for lab, mk in (("all t", (D["a1"] == a) & (D["a2"] >= 2)), ("t = 9", (D["a1"] == a) & (D["a2"] >= 2) & (D["t"] == 9))):
        r = D["z"][mk, k] / D["M"][mk]
        q = np.quantile(r, [.01, .1, .5, .9, .99])
        print(f"  unit {j:2d} at a1={a} ({lab}): quantiles 1/10/50/90/99%: {q[0]:7.2f} {q[1]:7.2f} {q[2]:7.2f} {q[3]:7.2f} {q[4]:7.2f};  P(z > 0) = {np.mean(r > 0):.3f};  P(z > 1 M) = {np.mean(r > 1):.3f}")

# ---------------------------------------------------------------- 3
print("\n===== 3. a new maximum has just arrived (a1 = 0): P(unit == 0) by age a2 of the OLD maximum (M > 0.3, S > 0.1) =====")
print("        a2:" + "".join(f"{c:7d}" for c in range(1, 10)))
for k, (j, a) in enumerate(PAIRS):
    row = f"  unit {j:2d}  "
    for c in range(1, 10):
        mk = (D["a1"] == 0) & (D["a2"] == c) & (D["S"] > 0.1)
        row += f"{np.mean(D['z'][mk, k] <= 0):7.3f}"
    print(row + f"    (claimed zero at age {a} of the max)")

# ---------------------------------------------------------------- 4
print("\n===== 4. x_9 is the largest input: the answer must be the position of the old maximum (= runner-up p2) =====")
x, h9, h8 = D["x"], D["h9"], D["h8"]
B = len(x)
order = np.argsort(-x, axis=1, kind="stable")
p1, p2 = order[:, 0], order[:, 1]
Sx = x[np.arange(B), p2]; Mx = x[np.arange(B), p1]
NCu = PULSE + WAVE
m = h9[:, NCu] @ U[:, NCu].T
V = h9[:, COMP] @ U[:, COMP].T
s = m + V
ans = s.argmax(1)
last = (p1 == 9) & (Sx > 0.3)
a2 = 9 - p2
ncomp_on = (h9[:, COMP] > 0).sum(1)
print(f"  sequences with the max at step 9 and old max > 0.3: {last.sum()};  P(answer = p2) = {(ans[last] == p2[last]).mean():.4f}")
print(f"  comparators on at t = 9: none {np.mean(ncomp_on[last] == 0):.3f}, one {np.mean(ncomp_on[last] == 1):.3f}, two or more {np.mean(ncomp_on[last] >= 2):.3f};"
      f"  median |V| / |m| (10-vector norms) = {np.median(np.linalg.norm(V[last], axis=1) / np.linalg.norm(m[last], axis=1)):.3f}")
gapbig = last & (Mx - Sx > 0.3)
print(f"  with a clear record (x_9 - old max > 0.3): n = {gapbig.sum()}, comparators all off in {np.mean(ncomp_on[gapbig] == 0):.3f}, P(answer = p2) = {(ans[gapbig] == p2[gapbig]).mean():.4f},"
      f" P(argmax m over 0..8 = p2) = {(m[gapbig][:, :9].argmax(1) == p2[gapbig]).mean():.4f}")
print("\n  by age of the old max at t = 9:")
print("   a2  pos      n   P(ans = p2)  P(top m[0..8] = p2)  P(all comps off)   P(unit = 0): u0     u12     u11     u10")
for a in range(1, 10):
    k = last & (a2 == a)
    print(f"   {a:2d}  {9 - a:3d} {k.sum():6d}    {(ans[k] == p2[k]).mean():.4f}        {(m[k][:, :9].argmax(1) == p2[k]).mean():.4f}            {np.mean(ncomp_on[k] == 0):.3f}"
          f"                    " + "  ".join(f"{np.mean(h9[k, j] == 0):.3f}" for j in UNITS))

print("\n  intervention h_9[k] <- h_8[k] (unit keeps its previous-step value): P(answer' = p2); rows = age of the old max, columns = unit forced")
print("   a2   none  " + "".join(f"   u{k:<3d}" for k in WAVE))
for a in range(1, 10):
    km = last & (a2 == a)
    row = f"   {a:2d}  {(ans[km] == p2[km]).mean():.3f} "
    for k in WAVE:
        row += f"  {((s[km] + (h8[km, k] - h9[km, k])[:, None] * U[None, :, k]).argmax(1) == p2[km]).mean():.3f}"
    print(row)
print("\n  the four matched cells in detail (what the network answers instead, and dose controls):")
for j, a in PAIRS:
    km = last & (a2 == a)
    d = h8[km, j] - h9[km, j]
    q = 9 - a
    s2 = s[km] + d[:, None] * U[None, :, j]
    a2n = s2.argmax(1)
    s3 = s[km].copy(); s3[:, q] += d * U[q, j]                     # only its vote on the old max's position
    col = U[:, j].copy(); col[q] = 0
    s4 = s[km] + d[:, None] * col[None, :]                          # everything except that vote
    typ = np.median(h9[last & (a2 != a), j] / Sx[last & (a2 != a)])
    s5 = s[km] + (typ * Sx[km] - h9[km, j])[:, None] * U[None, :, j]   # a typical 'on' value (median at other ages, x old max)
    print(f"   unit {j:2d}, old max at age {a} (position {q}): n = {km.sum()}, P(unit = 0) = {np.mean(h9[km, j] == 0):.3f}, forced value / old max: median {np.median(h8[km, j] / Sx[km]):.2f}")
    print(f"       P(answer = p2): {(ans[km] == p2[km]).mean():.3f} -> {(a2n == p2[km]).mean():.3f};  new answers: "
          + ", ".join(f"pos {r}: {np.mean(a2n == r):.3f}" for r in np.argsort(-np.bincount(a2n, minlength=10))[:4]))
    print(f"       only its vote on position {q}: -> {(s3.argmax(1) == p2[km]).mean():.3f};   all its votes except on {q}: -> {(s4.argmax(1) == p2[km]).mean():.3f};"
          f"   typical on-value ({typ:.1f} x old max): -> {(s5.argmax(1) == p2[km]).mean():.3f}")
