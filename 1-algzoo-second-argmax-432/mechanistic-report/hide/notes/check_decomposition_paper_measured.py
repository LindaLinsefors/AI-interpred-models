"""Spot-replication of measured numbers in the decomposition-route working paper
("How ARC's 432-parameter RNN finds the second-largest input: comparators, a recorder and a rotating tape").

Fresh N(0,1) inputs, N = 2^19, seed 0. The paper's value is printed in brackets after each of ours.
Read-only on the weights. Run:  python -B check_decomposition_paper_measured.py   (about 30 s on a laptop CPU)
"""
from pathlib import Path
import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=200, floatmode="fixed")
ROOT = Path(__file__).resolve().parents[4]                # workspace root
Z = np.load(ROOT / "reverse-engineered-networks/1-algzoo-second-argmax-432/code/m16_10_weights.npz")
w, Whh, Wout = Z["W_ih"][:, 0].astype(float), Z["W_hh"].astype(float), Z["W_out"].astype(float)
CMP = [4, 7, 6, 1, 8]                                     # the paper's five comparators, in its table order
N, T = 2 ** 19, 10
x = np.random.default_rng(0).standard_normal((N, T))
rows, steps = np.arange(N), np.arange(T)


def run(x, force_off=None):
    """Forward pass. force_off(t, gates_so_far) -> bool (N, 16): gates forced off at step t.
    Returns gates (N, T, 16), thresholds theta_j = -Whh[j].h_{t-1}/w_j for j in CMP (N, T, 5), logits (N, 10)."""
    h = np.zeros((len(x), 16)); G, TH = [], []
    for t in range(T):
        rec = h @ Whh.T
        TH.append(-rec[:, CMP] / w[CMP])
        pre = x[:, t, None] * w + rec
        g = pre > 0
        if force_off is not None:
            g &= ~force_off(t, G)
        h = np.where(g, pre, 0.0)
        G.append(g)
    return np.stack(G, 1), np.stack(TH, 1), h @ Wout.T


def trace(G, p):
    """Exact trace J_p in the final state, D_9 Whh ... D_{p+1} Whh D_p w, for one position p per sequence."""
    J = np.zeros((N, 16))
    for s in range(T):
        J = np.where((s == p)[:, None], G[:, s] * w, np.where((s > p)[:, None], G[:, s] * (J @ Whh.T), 0.0))
    return J


def eff(J, p):
    """Effect of a trace on the logit at position p (one p per sequence)."""
    return np.einsum("ni,ni->n", Wout[p], J)


def frac(a):
    return float(np.mean(a))


# ---- bookkeeping: the top three, and the maximum / runner-up / argmax of the inputs before step t ----
order = np.argsort(-x, axis=1)
pm, p2, p3 = order[:, 0], order[:, 1], order[:, 2]
xm, x2, x3 = x[rows, pm], x[rows, p2], x[rows, p3]
m = np.full((N, T), -np.inf); s2 = np.full((N, T), -np.inf); am = np.full((N, T), -1)
for t in range(1, T):
    xt = x[:, t - 1]; new = xt > m[:, t - 1]
    m[:, t] = np.where(new, xt, m[:, t - 1])
    s2[:, t] = np.where(new, m[:, t - 1], np.maximum(s2[:, t - 1], xt))
    am[:, t] = np.where(new, t - 1, am[:, t - 1])
age = steps - am                                          # age of the maximum so far at step t
t2 = steps >= 2                                           # steps at which a runner-up so far exists

G, TH, logits = run(x)
ans = logits.argmax(1)
ok = ans == p2
print(f"accuracy {frac(ok):.4f} [95.3%]; answers 3rd {frac(ans == p3):.4f} [2.97%], max {frac(ans == pm):.4f} [0.95%], "
      f"smaller {frac(~ok & (ans != p3) & (ans != pm)):.4f} [0.76%]")

# ---- Table 2: threshold position between the runner-up so far (0) and the maximum so far (1) ----
print("\nTable 2, medians by the maximum's age 1..9 (steps 2-9, max so far - runner-up so far > 0.2):")
with np.errstate(invalid="ignore"):                       # -inf - -inf at steps 0 and 1
    gap = np.where(t2, m - s2, np.nan)
pos = (TH - np.where(t2, s2, 0.0)[:, :, None]) / gap[:, :, None]
sel = gap > 0.2
for k, j in enumerate(CMP):
    print(f"  n{j}: " + "  ".join(f"{np.median(pos[:, :, k][sel & (age == a)]):+.2f}" for a in range(1, 10)))
print("  [n4 .99 .98 .95 .85 .73 .63 .62 .70 .82 | n7 1.02 1.02 1.01 .87 .68 .56 .55 .66 .84 | n6 -.04 .70 1.04 1.19 1.08 .81 .50 .32 .36]")
print("  [n1 .85 -.01 .29 .82 1.30 1.42 1.13 .59 .07 | n8 .06 .38 .07 .06 .42 .97 1.42 1.49 1.16]")

passed = np.stack([G[:, :, 4]] + [~G[:, :, j] for j in CMP[1:]], -1)      # n4 on / the other four off
for name, k in [("below the runner-up so far [<= 0.5% each]", (x < s2) & t2),
                ("between runner-up and max so far [n8 ~1/2, n1 n6 ~1/3, n4 n7 6%]", (x > s2) & (x < m) & t2),
                ("new maximum [93-95%]", (x > m) & t2)]:
    print(f"pass rates (n4 n7 n6 n1 n8), input {name}: " + " ".join(f"{frac(passed[..., i][k]):.3f}" for i in range(5)))

# ---- gates of the timers and the recorder; the resting pattern ----
n4p, n7p, n9p = G[:, :-1, 4], G[:, :-1, 7], G[:, :-1, 9]
print(f"\nn9 gate == n4 gate one step earlier: {frac(G[:, 1:, 9] == n4p):.4f} [99.0%]")
print(f"n0 gate == not (n7 and n9 on one step earlier): {frac(G[:, 1:, 0] == ~(n7p & n9p)):.4f} [97.4%]")
for j, ref in [(9, "96.0% / 0.1%"), (5, "99.8% / 21%"), (0, "99.4% / 71%")]:
    print(f"  n{j} on after n4 fired / on other steps: {frac(G[:, 1:, j][n4p]):.3f} / {frac(G[:, 1:, j][~n4p]):.3f} [{ref}]")
rest_sel = (x < s2) & (age >= 2) & t2
pat, cnt = np.unique(G[rest_sel], axis=0, return_counts=True)
top = np.argsort(-cnt)[:5]
print("commonest gate patterns on steps with input below the runner-up so far and max so far >= 2 steps old (units off: share):")
print("  " + "; ".join(f"{np.where(~pat[i])[0].tolist()}: {cnt[i] / cnt.sum():.3f}" for i in top)
      + f"; top five together {cnt[top].sum() / cnt.sum():.3f} [resting 21%, with four one-flip patterns 51%]")

# ---- record rates ----
newmax = (x > m) & (steps >= 1)
newrun = (x < m) & (x > s2) & (steps >= 1)
rec = np.zeros((N, T), bool); rec[:, :-1] = G[:, 1:, 5]   # n5 on at the step after the input's step
print("\nrecorded at the next step, arrivals at steps 1..8:")
print("  new maxima    ", np.array([frac(rec[:, t][newmax[:, t]]) for t in range(1, 9)]), "[.75 .88 .94 .97 .99 .99 1.00 1.00]")
print("  new runners-up", np.array([frac(rec[:, t][newrun[:, t]]) for t in range(1, 9)]), "[.27 .52 .70 .83 .91 .95 .97 .99]")
low = (x < s2) & t2 & (steps <= 8)
low2 = low & (s2 > 0.2)                                   # the set of Fig 3b
pred = 18 * x - 12.8 * np.where(t2, s2, 0.0) > 0
print(f"  inputs below the runner-up so far that are not recorded: {1 - frac(rec[low]):.3f} [94%]")
print(f"  Fig 3b set: recorded {frac(rec[low2]):.3f} [7.8%]; sign of 18x - 12.8s agrees with n5's gate {frac(pred[low2] == rec[low2]):.3f} [97%], "
      f"picks out {frac(pred[low2 & rec]):.3f} of the recorded [85%], {frac(rec[low2 & pred]):.3f} of its picks are recorded [80%]")

# ---- Table 3: routes of the final runner-up ----
g2 = G[rows, p2]                                          # gates at the runner-up's arrival
first, n4on = p2 < pm, g2[:, 4]
route = np.select([first & n4on, first, n4on, g2[:, [1, 6, 7, 8]].all(1), ~g2[:, 8] & g2[:, 7]],
                  ["first, n4 fires", "first, n4 silent", "later, misfire", "later, passes no threshold", "later, passes n8"],
                  "later, other")
recorded = G[rows, np.minimum(p2 + 1, 9), 5]
J2 = trace(G, p2)
lg = logits.copy(); lg[rows, pm] = lg[rows, p2] = lg[rows, p3] = -np.inf
pb = lg.argmax(1)                                         # best other position
print("\nTable 3 (means): route | sequences | recorded | own | at max | at 3rd | at best other | accuracy")
paper = {"first, n4 fires": "49.4% 100% +36 +51 -19 +16 96.3%", "later, passes n8": "29.1% 100% +36 +36 -20 +14 97.7%",
         "later, passes no threshold": "9.6% 96% +59 -4 -9 +2 90.4%", "later, misfire": "6.0% 100% -28 +125 -22 +25 90.3%",
         "later, other": "5.3% 100% +47 +52 -22 +15 94.4%", "first, n4 silent": "0.5% 24% +12 -4 -3 0 29%"}
for r, ref in paper.items():
    k = route == r
    print(f"  {r:27s} {frac(k):.3f}  {frac(recorded[k & (p2 < 9)]):.2f}  {eff(J2, p2)[k].mean():+6.1f} {eff(J2, pm)[k].mean():+6.1f} "
          f"{eff(J2, p3)[k].mean():+6.1f} {eff(J2, pb)[k].mean():+6.1f}  {frac(ok[k]):.3f}   [{ref}]")
second = np.argsort(-logits, 1)[:, 1]
print(f"second-highest logit on correct sequences: max {frac(second[ok] == pm[ok]):.2f}, third {frac(second[ok] == p3[ok]):.2f} [42%, 36%, rest 22%]")

# ---- Fig 10b, c: n4's threshold and misfires when a near-tie runner-up arrives after the maximum ----
nt = (p2 > pm) & (xm - x2 < 0.05)
a2 = p2 - pm
th4 = TH[rows, p2, 0] - xm
print("\nFig 10c, share of near-tie runners-up on which n4 fires, by the maximum's age 1..9:")
print("  ", np.array([frac(n4on[nt & (a2 == a)]) for a in range(1, 10)]), "[.27 .48 .62 .89 .96 .98 .98 .93 .78]")
print(f"   pooled {frac(n4on[nt]):.3f} [67%]; later runners-up over all gaps {frac(n4on[p2 > pm]):.3f} [12%]")
print("Fig 10b, median theta4 - xmax by age 1..9:", np.array([np.median(th4[nt & (a2 == a)]) for a in range(1, 10)]), "[-0.01 at age 1, -0.16 at ages 6-7]")

# ---- Table 4: errors by the input answered and its gap to the runner-up ----
err = ~ok
xa = x[rows, ans]
cat = np.select([(ans == p3) & (x2 - x3 < 0.05), (ans == pm) & (xm - x2 < 0.05), (ans != p3) & (ans != pm) & (x2 - xa < 0.05)],
                ["third on a near tie", "maximum on a near tie", "smaller input within 0.05"], "away from near ties")
print("\nTable 4, shares of errors: " + "; ".join(f"{c} {frac(cat[err] == c):.3f}" for c in
      ["third on a near tie", "maximum on a near tie", "smaller input within 0.05", "away from near ties"]) + " [48%, 15%, 2%, 35%]")

# ---- the final maximum's own effect by its age, zero-padded frame: Wout[9 - a] . J_p(after step p + a) ----
J = np.zeros((N, 16)); own = {a: [] for a in range(T)}
for s in range(T):
    J = np.where((s == pm)[:, None], G[:, s] * w, np.where((s > pm)[:, None], G[:, s] * (J @ Whh.T), 0.0))
    for a in range(s + 1):
        own[a].append(J[pm == s - a] @ Wout[9 - a])
own = [np.concatenate(own[a]) for a in range(T)]
print("\nthe maximum's own effect per unit by age 0..9:")
print("  medians", np.array([np.median(o) for o in own]), "\n  [paper  -51.5 -28.1 -4.1 -5.9 -6.5 -8.3 -8.0 -6.6 -5.0 -3.7]")
print("  means  ", np.array([np.mean(o) for o in own]), "\n  [paper  -50.6 -27.2 -7.2 -10.3 -13.1 -17.7 -21.1 -22.7 -23.0 -24.1]")

# ---- Fig 4b: own effect of a recorded third-largest input that arrived after the top two, by its position ----
k3 = (p3 > pm) & (p3 > p2) & (p3 < 9) & G[rows, np.minimum(p3 + 1, 9), 5]
own3 = eff(trace(G, p3), p3)
print("\nFig 4b, own effect at positions 2..8:", np.array([own3[k3 & (p3 == p)].mean() for p in range(2, 9)]), "[39 55 62 67 65 61 62]")


# ---- interventions: 'decisions kept' = same answer as the model ----
def off(unit, when):
    def f(t, Gs):
        mask = np.zeros((N, 16), bool)
        if t:
            mask[:, unit] = when(t, Gs)
        return mask
    return f


print("\ninterventions (decisions kept, accuracy):")
for j in CMP:
    a = run(x, lambda t, Gs, j=j: np.arange(16) == j)[2].argmax(1)
    print(f"  n{j} off at every step: kept {frac(a == ans):.3f} [11-19%], accuracy {frac(a == p2):.3f}")
tests = [("n5 off after every step at which n4 was off [kept 63.1%, accuracy 61.2%]", off(5, lambda t, Gs: ~Gs[-1][:, 4])),
         ("oracle: n5 off after every input below the runner-up so far [kept 99.0%, accuracy 95.9%]", off(5, lambda t, Gs: x[:, t - 1] < s2[:, t - 1])),
         ("n9 off after n4 fires [kept 67%; 87% of changed answers are the max's position]", off(9, lambda t, Gs: Gs[-1][:, 4])),
         ("n0 off after n4 fires [kept 43%; 65% of changed answers are the input just before the max]", off(0, lambda t, Gs: Gs[-1][:, 4])),
         ("n4 off unless the input is a new maximum [accuracy +0.35 points; fixes 0.44%, breaks 0.09%]", off(4, lambda t, Gs: x[:, t] <= m[:, t]))]
for name, f in tests:
    a = run(x, f)[2].argmax(1)
    ch = a != ans
    print(f"  {name}:\n      kept {frac(a == ans):.3f}, accuracy {frac(a == p2):.4f}; changed answers at the max's position {frac(a[ch] == pm[ch]):.2f}, "
          f"just before it {frac(a[ch] == pm[ch] - 1):.2f}; fixes {frac((a == p2) & ~ok):.4f}, breaks {frac((a != p2) & ok):.4f}")
