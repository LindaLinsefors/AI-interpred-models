"""Referee S, checks 2-9 (see the printed headers). Fresh seeds 92xx; N(0,1) inputs."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT, LAM, coords

w4 = w[4]
g = -w[COMP]
B = 1_000_000
x = sample(B, 9201)
H, Z = run(x, return_pre=True)
st = running_stats(x)
ar = np.arange(B)
zc = H[..., ORD] @ LEFT["clock"]                     # clock coordinate, (B, T)

print("=== 2. New maximum over a positive old maximum: share with all four comparators exactly zero ===")
tot = {}
for s in range(1, 10):
    M0, S0, p0 = st["M1"][:, s - 1], st["M2"][:, s - 1], st["p1"][:, s - 1]
    nm = (x[:, s] > M0) & (M0 > 0)
    off = (H[:, s][:, COMP] == 0).all(1)
    tot[s] = (nm, off, s - p0, M0, S0)
    print(f"  step {s}: {off[nm].mean():.3f}  (n={nm.sum()});  with old runner-up > 0: {off[nm & (S0 > 0)].mean():.3f}")
allnm = np.concatenate([tot[s][0] for s in range(1, 9)]); alloff = np.concatenate([tot[s][1] for s in range(1, 9)])
allage = np.concatenate([tot[s][2] for s in range(1, 9)]); allS0 = np.concatenate([tot[s][4] for s in range(1, 9)])
allM0 = np.concatenate([tot[s][3] for s in range(1, 9)])
print(f"  pooled steps 1..8: {alloff[allnm].mean():.3f}")
print("  by age of the old maximum at the step of the new one (1..8): " +
      ", ".join(f"{alloff[allnm & (allage == a)].mean():.3f}" for a in range(1, 9)))
sel = allnm & (allage >= 3) & (allS0 > 0.2) & (allM0 - allS0 > 0.1)
print(f"  old maximum at least 3 steps old at the new one (= age >= 2 one step before), S0 > 0.2, M0 - S0 > 0.1: {alloff[sel].mean():.3f} (n={sel.sum()})")
sel = allnm & (allage >= 3)
print(f"  old maximum at least 3 steps old, no other condition: {alloff[sel].mean():.3f}; share of all new maxima at steps 1..8: {sel.sum() / allnm.sum():.3f}")

print("\n=== 4. Launch: clock radius one step after a new maximum against unit 4's pulse (gauge: h4 writes 1) ===")
rat, ages, quiet = [], [], []
for s in range(1, 9):
    nm, off, age, M0, S0 = tot[s]
    r = np.abs(zc[:, s + 1]) / np.where(H[:, s, 4] > 0, H[:, s, 4], np.nan)
    ok = nm & (H[:, s, 4] > 0)
    rat.append(r[ok]); ages.append(age[ok]); quiet.append((x[:, s + 1] < 0.5 * np.maximum(M0, 0))[ok])
rat, ages, quiet = map(np.concatenate, (rat, ages, quiet))
q = lambda v: "median %.2f, 10%%..90%% %.2f..%.2f" % (np.median(v), np.quantile(v, .1), np.quantile(v, .9))
print("  |z(s+1)| / h4(s), all new maxima at s = 1..8:      " + q(rat))
print("  same, next input below half the old maximum:      " + q(rat[quiet]))
for a in (1, 2, 3, 5, 7):
    print(f"    old maximum {a} steps old at s: " + q(rat[quiet & (ages == a)]))
print(f"  share with |z(s+1)|/h4(s) outside 0.9..1.1 (quiet next step): {((rat[quiet] < 0.9) | (rat[quiet] > 1.1)).mean():.3f}")

print("\n=== 12. What is left of the clock coordinate at the step of the new maximum: |z(s)| / |z(s-1)| ===")
for a in range(2, 9):
    v = []
    for s in range(2, 10):
        nm, off, age, M0, S0 = tot[s]
        ok = nm & off & (age == a) & (np.abs(zc[:, s - 1]) > 0.1 * w4 * M0)
        v.append((np.abs(zc[:, s]) / np.abs(zc[:, s - 1]))[ok])
    v = np.concatenate(v)
    print(f"  old maximum {a} steps old at s (all four comparators off): median {np.median(v):.2f} (n={len(v)})")

print("\n=== 5. Margin logit[runner-up] - logit[max] at t = 9 (a1 >= 2, a2 >= 1, S > 0.2): regression ===")
M, S, p1, p2, a1, a2 = (st[k][:, 9] for k in ("M1", "M2", "p1", "p2", "a1", "a2"))
L = logits(H)
mar = L[ar, p2] - L[ar, p1]
ok = (a1 >= 2) & (a2 >= 1) & (S > 0.2)
X = np.stack([M - S, S, x[:, 9]], 1)[ok]
co = np.linalg.lstsq(X, mar[ok], rcond=None)[0]
print(f"  margin ~ {co[0]:.2f} (M - S) + {co[1]:.2f} S + {co[2]:.2f} x9   (n={ok.sum()})")
gm = ((M - S) / M)[ok]; mm = (mar / M)[ok]
for lo, hi in [(0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8)]:
    b = (gm > lo) & (gm <= hi)
    print(f"  relative gap ({lo}, {hi}]: median margin/M = {np.median(mm[b]):.2f}")
D = np.abs(p1 - p2)[ok]
print("  median margin/M by distance D = 1..8 (relative gap 0.3-0.5): " +
      ", ".join(f"{np.median(mm[(D == d) & (gm > .3) & (gm <= .5)]):.1f}" for d in range(1, 9)))
print("  median margin/M by distance D = 1..8 (all relative gaps > 0.05): " +
      ", ".join(f"{np.median(mm[(D == d) & (gm > .05)]):.1f}" for d in range(1, 9)))

print("\n=== 6. d theta_j(t+1) / d x_t with all four comparators on and unit 4 off at t (exact, through all 16 units) ===")
G = (Z > 0)
ok = G[..., COMP].all(-1) & ~G[..., 4]
ok[:, 9] = False
Gs = G[ok].astype(float)                               # (n, 16)
J = (Gs * w) @ W[COMP].T / g                           # (n, 4)
print(f"  n = {len(J)};  mean |.| per threshold: {np.abs(J).mean(0).round(4)};  99th pct: {np.quantile(np.abs(J), .99, axis=0).round(3)};  max: {np.abs(J).max(0).round(3)}")
print(f"  share of states with max_j |.| > 0.011: {(np.abs(J).max(1) > 0.011).mean():.3f};  > 0.05: {(np.abs(J).max(1) > 0.05).mean():.4f}")

print("\n=== 7. Largest entry of m_t among the inputs before x_t is the maximum: with and without S_t > 0 ===")
NC = [u for u in range(N) if u not in COMP]
for a in range(1, 9):
    hit_all = tot_all = hit_pos = tot_pos = 0
    for t in range(max(a, 2), 10):
        m = H[:, t][:, NC] @ U[:, NC].T
        m[:, :9 - t] = -np.inf; m[:, 9] = -np.inf
        top = m.argmax(1) - (9 - t)
        sel = (st["a1"][:, t] == a) & (st["M1"][:, t] > 0)
        hit_all += (top[sel] == st["p1"][:, t][sel]).sum(); tot_all += sel.sum()
        sel2 = sel & (st["M2"][:, t] > 0)
        hit_pos += (top[sel2] == st["p1"][:, t][sel2]).sum(); tot_pos += sel2.sum()
    print(f"  maximum {a} steps old: S_t > 0: {hit_pos / tot_pos:.3f};  all states with M_t > 0: {hit_all / tot_all:.3f};  share with S_t <= 0: {1 - tot_pos / tot_all:.3f}")

print("\n=== 8. Runner coordinate / w4 on (M, S+), states with a1 >= 2, t >= 2 ===")
run_c = np.real(H[..., ORD] @ LEFT["runner"]) / w4
for lab, cond in [("S > 0.05", st["M2"] > 0.05), ("all S (S+ used)", st["M1"] > 0.05)]:
    ok = (st["a1"] >= 2) & cond & (st["M1"] > 0.05)
    Mv, Sv = st["M1"][ok], np.maximum(st["M2"][ok], 0)
    X = np.stack([Mv, Sv], 1); y = run_c[ok]
    co = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ co
    yM = y / Mv; fit = (X @ co) / Mv
    print(f"  {lab}: runner/w4 ~ {co[0]:.2f} M + {co[1]:.2f} S+ ; uncentred R2 {1 - (res**2).sum() / (y**2).sum():.3f}; "
          f"centred R2 after dividing by M {1 - ((yM - fit)**2).sum() / ((yM - yM.mean())**2).sum():.3f}")

print("\n=== 9. Old maximum (>= 2 steps old one step before) overtaken at s <= 8 and true runner-up at the end: answered? ===")
truth = second_argmax(x); ans = L.argmax(1)
for lab in ("all four off", "some comparator on"):
    hit = n = 0
    for s in range(3, 9):
        nm, off, age, M0, S0 = tot[s]
        sel = nm & (age >= 3) & (truth == st["p1"][:, s - 1]) & (x.argmax(1) == s) & (off if lab == "all four off" else ~off)
        hit += (ans[sel] == truth[sel]).sum(); n += sel.sum()
    print(f"  {lab}: P(answer = old maximum) = {hit / n:.3f} (n={n})")
