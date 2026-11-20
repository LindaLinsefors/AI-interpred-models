"""V7 attack 4: is the qualitative pattern of the lead's sweep (O3) general?

Lead's sweep: x = (0.5, -1, 1.5, -1, -1, -1, v, -1, -1, -1): runner-up S = 0.5 at position 0, max M = 1.5 at 2,
probe v at 6 (ages at the probe step: max 4, runner-up 6).
Pattern to test in other configurations:
  P1  one comparator (8) has its threshold next to S, the other three next to M; they switch off in the order 8, 1, 7, 6;
  P2  unit 4 fires above a threshold a little below M;
  P3  the answer is right on both sides of S and of M;
  P4  the clock amplitude at t = 9 falls as v goes from S to M and is smallest near v = M.
Family: background b, S = 0.5 at p_S, M = 1.5 at p_M, probe v at p_v > p_S, p_M.  Deterministic.
f_j = (theta_j - S) / (M - S): 0 = threshold at the runner-up, 1 = at the maximum.
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

S0, M0 = 0.5, 1.5
VG = np.round(np.arange(0.0, 2.5001, 0.0025), 4)
UN = [1, 6, 7, 8, 4]


def config(pS, pM, pv, bg=-1.0, S=S0, M=M0):
    x = np.full(10, bg, dtype=float) if np.isscalar(bg) else np.array(bg, dtype=float)
    x[pS], x[pM] = S, M
    return x


def sweep(x, pv, S=S0, M=M0, vg=VG):
    """Thresholds at step pv and the sweep over v. Returns a dict."""
    h_prev = run(x[:pv])[0, -1] if pv > 0 else np.zeros(N)
    theta = -(W @ h_prev) / w
    X = np.tile(x, (vg.size, 1)); X[:, pv] = vg
    H = run(X)
    ans = logits(H).argmax(1)
    tgt = second_argmax(X)
    clock = np.abs(coords(H[:, 9])["clock"]) / w[4]
    wrong = ans != tgt
    # decision boundaries: last v answered like 'v below S', first v answered like 'v above M'
    pS, pM = int(np.argmin(np.abs(x - S))), int(np.argmax(x))
    lowS = vg[(ans == pS)]
    b1 = lowS.max() if lowS.size else np.nan              # answer leaves p_S here (true: S)
    hiM = vg[(ans == pM)]
    b2 = hiM.min() if hiM.size else np.nan                # answer becomes p_M here (true: M)
    other = ~np.isin(ans, [pS, pM, pv])
    k = clock.argmin()
    iS, iM = np.searchsorted(vg, S), np.searchsorted(vg, M)
    on5 = vg[H[:, pv + 1, 5] > 0] if pv < 9 else np.array([])       # unit 5 one step after the probe
    return dict(theta=theta, f=(theta - S) / (M - S), ans=ans, wrong=wrong, b1=b1, b2=b2, other=other.mean(),
                clock=clock, vmin=vg[k], cmin=clock[k], c_at_S=clock[iS], c_at_M=clock[iM], c_at_0=clock[0],
                h4=H[:, pv, 4], H=H, f5=((on5.min() - S) / (M - S) if on5.size else np.nan))


# ---------------------------------------------------------------- 0. the lead's configuration again
print("===== 0. lead's configuration (p_S, p_M, p_v) = (0, 2, 6), background -1 =====")
r = sweep(config(0, 2, 6), 6)
print("  f_j = (theta_j - S)/(M - S):  " + "  ".join(f"unit {j}: {r['f'][j]:.3f}" for j in UN))
print(f"  answer leaves p_S at v = {r['b1']:.4f} (true 0.5); becomes p_M at v = {r['b2']:.4f} (true 1.5); wrong on {r['wrong'].mean():.4f} of the v grid [0, 2.5]"
      f"; clock at t=9: {r['c_at_S']:.3f} at v=S, {r['c_at_M']:.3f} at v=M, min {r['cmin']:.3f} at v={r['vmin']:.3f}")

# ---------------------------------------------------------------- 1. second configuration, row by row like the lead's table
def rows(pS, pM, pv, vs, bg=-1.0):
    x = config(pS, pM, pv, bg)
    r = sweep(x, pv, vg=np.array(vs, dtype=float))
    print(f"\n  x = {np.array2string(np.where(np.arange(10) == pv, np.nan, x), precision=2)} (nan = v);  ages at the probe step: max {pv - pM}, runner-up {pv - pS}")
    print("  thresholds: " + "  ".join(f"theta_{j} = {r['theta'][j]:.3f} (f={r['f'][j]:.2f})" for j in UN))
    print("      v   comps on at probe step   h[4]    h_next[5]   clock(t=9)   answer  target")
    X = np.tile(x, (len(vs), 1)); X[:, pv] = vs
    tgt = second_argmax(X)
    for i, v in enumerate(vs):
        H = r["H"][i]
        on = [j for j in COMP if H[pv, j] > 0]
        nx = f"{H[pv + 1, 5]:7.3f}" if pv < 9 else "    -  "
        print(f"   {v:5.2f}   {str(on):22s} {H[pv, 4]:7.3f}   {nx}     {r['clock'][i]:7.4f}      {r['ans'][i]}      {tgt[i]}" + ("   <-- WRONG" if r['ans'][i] != tgt[i] else ""))


VS = [0.30, 0.45, 0.55, 0.85, 1.15, 1.30, 1.45, 1.55, 1.80]
print("\n===== 1. the same nine probe values in other configurations =====")
rows(0, 2, 6, VS)                    # lead's, for reference
rows(4, 1, 7, VS)                    # max first (age 6), runner-up younger (age 3)
rows(1, 4, 6, VS)                    # max age 2, runner-up age 5 (separation probe-max = 2 steps: clocks at 85 deg)
rows(3, 5, 9, VS)                    # the lead's ages (max 4, runner-up 6 at the probe step), three steps later
rows(3, 5, 8, VS)                    # max age 3, runner-up age 5
rows(0, 2, 6, VS, bg=0.0)            # lead's positions, background 0 instead of -1
rows(0, 2, 6, VS, bg=-0.2)

# ---------------------------------------------------------------- 2. all configurations
print("\n===== 2. all 240 configurations (background -1): thresholds by ages at the probe step =====")
res = {}
for pv in range(2, 10):
    for pS in range(pv):
        for pM in range(pv):
            if pS != pM:
                res[(pS, pM, pv)] = sweep(config(pS, pM, pv), pv)
keys = list(res)
aM = np.array([k[2] - k[1] for k in keys]); aS = np.array([k[2] - k[0] for k in keys]); PV = np.array([k[2] for k in keys])
F = np.array([res[k]["f"][UN] for k in keys])            # (240, 5)
print("  f_j by age of the max at the probe step (mean [min, max] over all runner-up ages and probe steps):")
print("   age    n    " + "".join(f"        unit {j}         " for j in UN))
for a in range(1, 10):
    mk = aM == a
    print(f"   {a:2d}  {mk.sum():4d}   " + "".join(f"  {F[mk, i].mean():6.2f} [{F[mk, i].min():6.2f},{F[mk, i].max():6.2f}]" for i in range(5)))
low = F[:, :4].argmin(1)
print("\n  which comparator has the LOWEST threshold, by age of the max at the probe step (counts over configurations):")
for a in range(1, 10):
    mk = aM == a
    print(f"   age {a}: " + "  ".join(f"unit {COMP[i]}: {np.sum(low[mk] == i):3d}" for i in range(4))
          + f"   | lowest f: mean {F[mk, :4].min(1).mean():6.2f};  number of comparators with f < 0.5: mean {np.mean((F[mk, :4] < 0.5).sum(1)):.2f}")
near = (np.abs(F[:, :4]) < 0.25)
print(f"\n  configurations in which exactly one comparator sits near S (|f| < 0.25) and the other three near or above M (f > 0.6): "
      f"{np.mean((near.sum(1) == 1) & ((F[:, :4] > 0.6).sum(1) == 3)):.3f}")
print(f"  configurations in which NO comparator threshold lies within |f| < 0.25 of S: {np.mean(near.sum(1) == 0):.3f}")
print(f"  configurations in which unit 8 is the one with the lowest threshold: {np.mean(low == 3):.3f}")
order = np.array(["".join(str(COMP[i]) for i in np.argsort(F[k, :4])) for k in range(len(keys))])
uo, co = np.unique(order, return_counts=True)
print("  switch-off orders (lowest threshold first): " + ", ".join(f"{o}: {c}" for o, c in sorted(zip(uo, co), key=lambda z: -z[1])))
print(f"  unit 4 threshold f_4: mean {F[:, 4].mean():.3f}, min {F[:, 4].min():.3f}, max {F[:, 4].max():.3f};  by age of the max: "
      + "  ".join(f"{a}: {F[aM == a, 4].mean():.2f}" for a in range(1, 10)))
F5 = np.array([res[k]["f5"] for k in keys])
print("  onset of unit 5 one step after the probe, f = (v_on - S)/(M - S), by age of the max (mean [min, max]): "
      + "  ".join(f"{a}: {np.nanmean(F5[aM == a]):.2f} [{np.nanmin(F5[aM == a]):.2f},{np.nanmax(F5[aM == a]):.2f}]" for a in range(1, 9)))
print(f"  theta_4 in absolute terms (M = 1.5): mean {1.5 - (1 - F[:, 4].mean()):.3f}; configurations with theta_4 < M (misfire window): {np.mean(F[:, 4] < 1):.3f}; with theta_4 > M (missed record window): {np.mean(F[:, 4] > 1):.3f}")

print("\n===== 3. answers over the fine v grid [0, 2.5] (step 0.0025) =====")
wr = np.array([res[k]["wrong"].mean() for k in keys])
b1 = np.array([res[k]["b1"] for k in keys]); b2 = np.array([res[k]["b2"] for k in keys]); oth = np.array([res[k]["other"] for k in keys])
print(f"  fraction of the v grid answered wrongly: mean {wr.mean():.4f}, median {np.median(wr):.4f}, max {wr.max():.4f} (config {keys[wr.argmax()]});  lead's config: {res[(0, 2, 6)]['wrong'].mean():.4f}")
print(f"  boundary p_S -> p_v (true 0.5): mean {np.nanmean(b1):.4f}, min {np.nanmin(b1):.4f}, max {np.nanmax(b1):.4f}; |error| > 0.05 in {np.mean(np.abs(b1 - 0.5) > 0.05):.3f} of configs")
print(f"  boundary p_v -> p_M (true 1.5): mean {np.nanmean(b2):.4f}, min {np.nanmin(b2):.4f}, max {np.nanmax(b2):.4f}; |error| > 0.05 in {np.mean(np.abs(b2 - 1.5) > 0.05):.3f} of configs; never answers p_M: {np.isnan(b2).sum()}")
print(f"  configurations in which some other position is answered for some v: {np.mean(oth > 0):.3f}")
print("  wrong fraction by probe step: " + "  ".join(f"p_v={p}: {wr[PV == p].mean():.3f}" for p in range(2, 10)))
print("  wrong fraction by |p_S - p_M|: " + "  ".join(f"{d}: {wr[np.abs(aM - aS) == d].mean():.3f}" for d in range(1, 9)))
print("  ten worst configurations (p_S, p_M, p_v): " + ", ".join(f"{keys[i]}: {wr[i]:.3f}" for i in np.argsort(-wr)[:10]))
def runs(k):
    a = res[k]["ans"]; cut = np.r_[0, np.nonzero(np.diff(a))[0] + 1]
    return "; ".join(f"v>={VG[c]:.4f}: {a[c]}" for c in cut)


for k in [(0, 2, 6)] + [keys[i] for i in np.argsort(-wr)[:4]]:
    print(f"    answers along v for (p_S, p_M, p_v) = {k} (true: {k[0]} below 0.5, {k[2]} between, {k[1]} above 1.5):  {runs(k)}")
nine = np.array([[res[k]["ans"][np.searchsorted(VG, v)] for v in VS] for k in keys])
tgt9 = np.array([[k[0] if v < S0 else (k[2] if v < M0 else k[1]) for v in VS] for k in keys])
print(f"  the lead's nine probe values: all nine answered correctly in {np.mean((nine == tgt9).all(1)):.3f} of the 240 configurations; "
      f"per value: " + "  ".join(f"{v}: {np.mean(nine[:, i] == tgt9[:, i]):.3f}" for i, v in enumerate(VS)))

print("\n===== 4. clock amplitude |z|/w4 at t = 9 as a function of v =====")
vmin = np.array([res[k]["vmin"] for k in keys]); cmin = np.array([res[k]["cmin"] for k in keys])
cS = np.array([res[k]["c_at_S"] for k in keys]); cM = np.array([res[k]["c_at_M"] for k in keys]); c0 = np.array([res[k]["c_at_0"] for k in keys])
print(f"  location of the minimum over v in [0, 2.5]: within 0.1 of M in {np.mean(np.abs(vmin - 1.5) < 0.1):.3f} of configs; within 0.25: {np.mean(np.abs(vmin - 1.5) < 0.25):.3f}; "
      f"at the edge v = 0 or 2.5: {np.mean((vmin < 0.01) | (vmin > 2.49)):.3f}")
print(f"  depth: clock(v = M)/clock(v = S): mean {np.mean(cM / cS):.3f}, median {np.median(cM / cS):.3f}, min {np.min(cM / cS):.3f}, max {np.max(cM / cS):.3f};  < 0.25 in {np.mean(cM / cS < 0.25):.3f} of configs; > 0.75 in {np.mean(cM / cS > 0.75):.3f}")
print("  by separation probe - max (steps), restricted to p_v <= 7 so that at least two steps follow the probe:")
for d in range(1, 8):
    mk = (aM == d) & (PV <= 7)
    if mk.sum():
        print(f"    probe - max = {d} ({d * 42.75:6.1f} deg): n = {mk.sum():3d}  clock(v=M)/clock(v=S): mean {np.mean(cM[mk] / cS[mk]):.3f} [{np.min(cM[mk] / cS[mk]):.3f}, {np.max(cM[mk] / cS[mk]):.3f}];"
              f"  v at minimum: mean {vmin[mk].mean():.3f} [{vmin[mk].min():.3f}, {vmin[mk].max():.3f}];  min value {cmin[mk].mean():.3f}"
              f"   | two equal phasors of that separation would give |1 + e^(i d w)|/2 = {abs(1 + np.exp(1j * np.radians(42.75 * d))) / 2:.3f} of their sum")
