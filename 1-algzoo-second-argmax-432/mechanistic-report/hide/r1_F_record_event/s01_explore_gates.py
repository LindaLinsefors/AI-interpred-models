"""s01. Exploration: gate frequencies around a record at step s (N(0,1), seed 1001).
Conditions on the following inputs: none / below M0 (new runner-up) / below max(S0,0)."""
from fcommon import *

x = sample(400_000, 1001)
H, Z = run(x, return_pre=True)
G = Z > 0
st = running_stats(x)

# what do the thresholds track?  theta_j(t) = -W[j].h_{t-1}/w_j  (j in comparators and unit 4)
print("=== thresholds theta_j(t) vs prefix order statistics of x_0..x_{t-1} (t>=3, M1>0), no intercept ===")
for j in COMP + [4]:
    th = -(H[:, :-1] @ W[j]) / w[j]           # theta_j(t) for t = 1..9  -> index t-1
    sel = np.zeros((x.shape[0], 9), bool); sel[:, 2:] = True
    M1, M2, M3 = st["M1"][:, :-1], st["M2"][:, :-1], st["M3"][:, :-1]
    sel &= (M1 > 0) & np.isfinite(M3)
    y = th[sel]
    for lab, cols in (("M1", [M1[sel]]), ("M1,M2+", [M1[sel], np.maximum(M2[sel], 0)]),
                      ("M1,M2+,M3+", [M1[sel], np.maximum(M2[sel], 0), np.maximum(M3[sel], 0)])):
        b, R = ols(y, cols)
        print(f"  theta_{j} ~ {lab:12s}: coef {np.round(b, 3)}  R2 = {R:.4f}")

for cond_name in ("none", "below M0", "below S0+"):
    print(f"\n################ following inputs: {cond_name} ################")
    for s in (2, 4, 6):
        rec, q = records(x, s, st)
        ok = rec.copy()
        for d in (1, 2, 3):
            if s + d > 9:
                continue
            if cond_name == "below M0":
                ok &= x[:, s + d] < q["M0"]
            elif cond_name == "below S0+":
                ok &= x[:, s + d] < q["S0p"]
        print(f"\n--- record at s = {s}: n = {ok.sum()} (of {rec.sum()} records with M0>0) ---")
        print("  P(unit on):  unit " + " ".join(f"{u:5d}" for u in range(16)))
        for d in (-1, 0, 1, 2, 3):
            if s + d > 9:
                continue
            f = G[ok, s + d].mean(0)
            print(f"    step s{d:+d}:      " + " ".join(f"{v:5.2f}" for v in f))
        for d in (0, 1, 2, 3):
            if s + d > 9:
                continue
            pid = pattern_id(G[ok, s + d])
            vals, cnt = np.unique(pid, return_counts=True)
            order = np.argsort(-cnt)[:6]
            print(f"    step s{d:+d} top patterns (units ON): " +
                  "; ".join(f"{pattern_str(int(vals[k]))} {cnt[k]/ok.sum():.3f}" for k in order))
