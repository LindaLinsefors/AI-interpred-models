"""H1 tables.  Event N at step t: S_old < x_t < M_old  (S_old > 0, t >= 2).
r = (x_t - S_old)/(M_old - S_old);  a = age of the max at step t (t - p1, 1..9).
Comparator j is OFF iff x_t > theta_j(t)  <=>  r > rho_j,   rho_j = (theta_j - S_old)/(M_old - S_old).
Unit 4 is ON (a misfire, since x_t < M_old) iff r > rho_4.
"""
from hcommon import *

RB = np.linspace(0, 1, 6)          # r bins
acc = {}                           # (name, a, rb) -> [sum, n]
cols = {}
for seed in (1001, 1002, 1003):
    x = sample(400_000, seed)
    e = Ev(x)
    ok = (e.tt >= 2) & (e.So > 0)
    isN = ok & (x > e.So) & (x < e.Mo)
    b, t = np.nonzero(isN)
    gap = e.Mo[b, t] - e.So[b, t]
    d = dict(a=e.a1[b, t], a2=e.a2[b, t], t=t, r=(x[b, t] - e.So[b, t]) / gap, gap=gap, M=e.Mo[b, t], S=e.So[b, t], x=x[b, t],
             on4=e.G[b, t, 4], h4=e.H[b, t, 4])
    for j in COMP + [4]:
        d[f"rho{j}"] = (e.th[j][b, t] - e.So[b, t]) / gap
        d[f"off{j}"] = ~e.G[b, t, j]
    # units 5 / 9 at t+1 (only for t <= 8)
    tn = np.minimum(t + 1, 9)
    d["on5n"] = np.where(t < 9, e.G[b, tn, 5], False); d["on9n"] = np.where(t < 9, e.G[b, tn, 9], False)
    d["on5"] = e.G[b, t, 5]; d["on9"] = e.G[b, t, 9]
    d["correct"] = (logits(e.H).argmax(1) == second_argmax(x))[b]
    d["nwave_off"] = (~e.G[b, t][:, WAVE]).sum(1)
    for k, v in d.items():
        cols.setdefault(k, []).append(v)
    del e
d = {k: np.concatenate(v) for k, v in cols.items()}
n = len(d["a"])
print(f"N-range events (S_old>0, S_old<x_t<M_old, t>=2): n = {n}  from 1.2M sequences (seeds 1001-1003)")
print("  by age of max:", {a: int((d['a'] == a).sum()) for a in range(1, 10)})
print(f"  unit 4 ON (misfire) in {d['on4'].mean():.4f} of them")
rb = np.clip(np.digitize(d["r"], RB) - 1, 0, len(RB) - 2)
hdr = "  a  | " + " ".join(f"r {RB[i]:.1f}-{RB[i+1]:.1f}" for i in range(len(RB) - 1)) + " |  all r"

def table(val, title, sel=None, fmt="{:9.3f}"):
    print(f"\n{title}\n{hdr}")
    sel = np.ones(n, bool) if sel is None else sel
    for a in list(range(1, 10)) + ["all"]:
        sa = sel & ((d["a"] == a) if a != "all" else True)
        row = ""
        for i in range(len(RB) - 1):
            s = sa & (rb == i)
            row += (fmt.format(val[s].mean()) if s.sum() >= 200 else "      .  ") + " "
        print(f"  {a!s:3s}| {row}| " + fmt.format(val[sa].mean()) + f"   n={sa.sum()}")

for j in COMP:
    table(d[f"off{j}"].astype(float), f"P(comparator {j} OFF at t | N-range event)")
table(d["on4"].astype(float), "P(unit 4 ON at t | N-range event)  = misfire probability")

print("\nmedian rho_j [25%, 75%] at the N-range events, by age (the comparator is off iff r > rho_j):")
print("  a  |" + "".join(f"        unit {j}          |" for j in COMP + [4]))
for a in range(1, 10):
    s = d["a"] == a
    print(f"  {a}  |" + "".join(" {:5.2f} [{:5.2f}, {:5.2f}] |".format(*np.percentile(d[f'rho{j}'][s], [50, 25, 75])) for j in COMP + [4]))

print("\nexcess of the clipped comparators in units of the gap, (x_t - theta_j)/(M-S) = r - rho_j: mean over events where j is off")
print("  a  |" + "".join(f"  unit {j} |" for j in COMP) + "  unit 4 (when on)")
for a in range(1, 10):
    s = d["a"] == a
    row = ""
    for j in COMP:
        ss = s & d[f"off{j}"]
        row += f"  {np.mean(d['r'][ss] - d[f'rho{j}'][ss]):6.3f} |" if ss.sum() > 200 else "     .   |"
    ss = s & d["on4"]
    row += f"  {np.mean(d['r'][ss] - d['rho4'][ss]):6.3f}" if ss.sum() > 200 else "     .  "
    print(f"  {a}  |{row}")

print("\nnumber of comparators OFF at an N-range event (unit 4 silent): distribution by age")
noff = sum(d[f"off{j}"].astype(int) for j in COMP)
for a in range(1, 10):
    s = (d["a"] == a) & ~d["on4"]
    print(f"  a={a}: " + " ".join(f"{k}:{np.mean(noff[s] == k):.3f}" for k in range(5)) + f"   n={s.sum()}")

print("\ncommonest comparator OFF-sets at N events (unit 4 silent), by age: set (share)")
code = sum(d[f"off{j}"].astype(int) << i for i, j in enumerate(COMP))
name = lambda c: "{" + ",".join(str(j) for i, j in enumerate(COMP) if (c >> i) & 1) + "}"
for a in range(1, 10):
    s = (d["a"] == a) & ~d["on4"]
    cnt = np.bincount(code[s], minlength=16) / s.sum()
    top = np.argsort(-cnt)[:5]
    print(f"  a={a}: " + "  ".join(f"{name(c)} {cnt[c]:.3f}" for c in top if cnt[c] > 0.01))
cnt = np.bincount(code[~d["on4"]], minlength=16) / (~d["on4"]).sum()
print("  all ages: " + "  ".join(f"{name(c)} {cnt[c]:.3f}" for c in np.argsort(-cnt) if cnt[c] > 0.005))

print("\ndeterministic rule check: predict 'j off' by r > median rho_j(a) (median taken per age): agreement")
for j in COMP + [4]:
    agree = 0
    for a in range(1, 10):
        s = d["a"] == a
        med = np.median(d[f"rho{j}"][s])
        agree += ((d["r"][s] > med) == (d[f"off{j}"][s] if j != 4 else d["on4"][s])).sum()
    print(f"  unit {j}: {agree / n:.4f}")

print("\nmisfire size: h_t[4] / (K (M - S)) = r - rho_4 when on; quantiles by age")
for a in range(1, 10):
    s = (d["a"] == a) & d["on4"]
    if s.sum() < 200: continue
    v = d["h4"][s] / (K * d["gap"][s])
    print(f"  a={a}: P(misfire)={np.mean(d['on4'][d['a']==a]):.4f}  median {np.median(v):.3f}  90% {np.percentile(v, 90):.3f}   h4 itself: median {np.median(d['h4'][s]):.3f}")

print("\naccuracy of the final answer for sequences containing an N-range event, by misfire: ")
print(f"  misfire: {d['correct'][d['on4']].mean():.4f} (n={d['on4'].sum()})   no misfire: {d['correct'][~d['on4']].mean():.4f}")
for a in range(1, 10):
    s = d["a"] == a
    print(f"   a={a}: misfire acc {d['correct'][s & d['on4']].mean():.4f} (n={int((s & d['on4']).sum())})  no-misfire acc {d['correct'][s & ~d['on4']].mean():.4f}")

table(d["on5n"].astype(float), "P(unit 5 ON at t+1 | N event at t<=8, unit 4 silent)", sel=~d["on4"] & (d["t"] < 9))
table(d["on9n"].astype(float), "P(unit 9 ON at t+1 | N event at t<=8, unit 4 silent)", sel=~d["on4"] & (d["t"] < 9))
table(d["nwave_off"].astype(float), "mean number of wave units OFF at t | N event, unit 4 silent", sel=~d["on4"])
