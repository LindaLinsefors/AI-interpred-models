"""B1. theta_4(t) = -W[4].h_{t-1}/w_4 against M_{t-1} = max(x_0..x_{t-1}).
Data: N(0,1), seeds 1010 and 1011, 400k sequences each (800k sequences, steps t = 1..9).
"""
from blib import *

print("=== [EXACT] coefficients of theta_4 in mode coordinates of h_{t-1} ===")
print("theta_4 = sum_real C_n m_n + sum_pairs 2 Re(C_n z_n) + sum_pulse CP_q h_q ;  below 10.157*C_n (per unit of m/10.157)")
for n in NAMES:
    c = C4[n] * W4
    if IS_COMPLEX[n]:
        print(f"  {n:7s}: 2|C| = {2*abs(c):.4f}  phase {np.degrees(np.angle(c)):8.2f} deg   (lam {abs(LAM[n]):.4f} @ {np.degrees(np.angle(LAM[n])):.2f})")
    else:
        print(f"  {n:7s}: C = {c.real:8.4f}   (lam {LAM[n].real:.4f})")
for q in PULSE:
    print(f"  unit {q}: 10.157*CP = {CP4[q]*W4:8.4f}   (= -W[4,{q}])")
print("  sum over modes of 10.157*C_n (real + 2Re for pairs) =",
      sum((2 * C4[n].real if IS_COMPLEX[n] else C4[n].real) for n in NAMES) * W4,
      "  = -W[4,ORD].W[ORD,4] =", -(W[4, ORD] @ W[ORD, 4]))

# ---------- impulse response (exact arithmetic on one input) ----------
print("\n=== [EXACT] impulse x = (1,0,...,0): theta_4(t) for the next input, t = 1..9, split into modes ===")
for label, xs in (("zeros after the impulse", np.r_[1.0, np.zeros(9)]),):
    H, Z = run(xs, return_pre=True)
    terms = theta4_terms(H[0])
    th = theta4(H[0])
    print(f"  {label}")
    print("   age a=t-1 | theta4 | " + " ".join(f"{k:>7s}" for k in terms) + " | ordinary units off")
    for t in range(1, 10):
        off = [u for u in ORD if Z[0, t - 1, u] <= 0]
        print(f"   {t-1:9d} | {th[t-1]:6.3f} | " + " ".join(f"{terms[k][t-1]:7.3f}" for k in terms) + f" | {off}")

# free-flow prediction: all 13 ordinary on from age 1: m_n(a) = lam^(a-1) * 10.157 (kick=1) + pulse-5/9 kicks
print("\n=== [EXACT] 'free flow' theta_4 after an isolated record of size 1 with zero later inputs, pulses 5, 9 as in the impulse ===")

# ---------- data ----------
acc = []
rows_all = []
SEEDS = (1010, 1011)
D = {k: [] for k in ("th", "M", "S", "a", "t", "a2", "x")}
TERMS = None
for seed in SEEDS:
    x = sample(400_000, seed)
    H = run(x)
    st = prefix_stats(x)
    th = theta4(H[:, :-1])                       # (B, 9): threshold for x_t, t = 1..9
    tm = theta4_terms(H[:, :-1])
    assert np.allclose(sum(tm.values()), th, atol=1e-9)
    D["th"].append(th); D["M"].append(st["M1"][:, :-1]); D["S"].append(st["M2"][:, :-1])
    D["a"].append(st["a1"][:, :-1]); D["a2"].append(st["a2"][:, :-1])
    D["t"].append(np.broadcast_to(np.arange(1, 10), th.shape)); D["x"].append(x[:, 1:])
    if TERMS is None:
        TERMS = {k: [v] for k, v in tm.items()}
    else:
        for k, v in tm.items():
            TERMS[k].append(v)
D = {k: np.concatenate(v) for k, v in D.items()}
TERMS = {k: np.concatenate(v) for k, v in TERMS.items()}
th, M, S, a, t, a2 = D["th"], D["M"], D["S"], D["a"], D["t"], D["a2"]
Sp = pos(np.where(np.isfinite(S), S, 0))
nseq = th.shape[0]
print(f"\n=== [MEASURED n = {nseq} sequences x 9 steps, N(0,1), seeds {SEEDS}] ===")

print("\n--- theta_4 - M_{t-1} by age of the max (states with M_{t-1} > 0) ---")
print("age | n | mean | sd | median | q05 | q25 | q75 | q95 | median theta/M | slope theta~M (no icpt) | R2c | slopes theta~(M,S+) | R2c")
for age in range(9):
    s = (a == age) & (M > 0)
    d = th[s] - M[s]
    q = np.percentile(d, [5, 25, 50, 75, 95])
    b1, _, r1 = slope0(th[s], [M[s]])
    b2, _, r2 = slope0(th[s], [M[s], Sp[s]])
    print(f"{age} | {s.sum()} | {d.mean():.4f} | {d.std():.4f} | {q[2]:.4f} | {q[0]:.4f} | {q[1]:.4f} | {q[3]:.4f} | {q[4]:.4f} | "
          f"{np.median(th[s]/M[s]):.4f} | {b1[0]:.4f} | {r1:.4f} | {b2[0]:.4f}, {b2[1]:.4f} | {r2:.4f}")
s = (M > 0)
d = th[s] - M[s]
print(f"all | {s.sum()} | {d.mean():.4f} | {d.std():.4f} | {np.median(d):.4f}")

print("\n--- same, by t (step whose input is compared), M_{t-1} > 0 ---")
for tt in range(1, 10):
    s = (t == tt) & (M > 0)
    d = th[s] - M[s]
    print(f"t={tt} | n={s.sum()} | mean {d.mean():.4f} | sd {d.std():.4f} | median {np.median(d):.4f}")

print("\n--- median of theta_4/M_{t-1} by age (rows) and relative gap g = (M - S+)/M (columns), M_{t-1} > 0 ---")
edges = [0, 0.1, 0.25, 0.5, 0.75, 0.999999, 1.0000001]
labels = ["[0,.1)", "[.1,.25)", "[.25,.5)", "[.5,.75)", "[.75,1)", "S<=0 (g=1)"]
g = (M - Sp) / np.where(M > 0, M, 1)
print("age | " + " | ".join(labels))
for age in range(9):
    row = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (a == age) & (M > 0) & (g >= lo) & (g < hi)
        row.append(f"{np.median(th[s]/M[s]):.3f} (n={s.sum()})" if s.sum() >= 200 else "-")
    print(f"{age} | " + " | ".join(row))

print("\n--- mean and sd of (theta_4 - M)/M by age and gap bin ---")
print("age | " + " | ".join(labels))
for age in range(9):
    row = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (a == age) & (M > 0.2) & (g >= lo) & (g < hi)
        r = th[s] / M[s] - 1
        row.append(f"{r.mean():+.3f} +- {r.std():.3f}" if s.sum() >= 200 else "-")
    print(f"{age} | " + " | ".join(row))

print("\n--- theta_4 ~ b_M M + b_S S+ split by whether the runner-up is older or younger than the max ---")
print("age | older runner-up (old max): b_M, b_S, R2c, n | younger runner-up: b_M, b_S, R2c, n")
for age in range(9):
    out = []
    for cond in ((a2 > a), (a2 < a) & (a2 >= 0)):
        s = (a == age) & (M > 0) & cond & (S > 0)
        if s.sum() < 500:
            out.append("-"); continue
        b, _, r = slope0(th[s], [M[s], Sp[s]])
        out.append(f"{b[0]:.3f}, {b[1]:.3f}, {r:.4f}, {s.sum()}")
    print(f"{age} | " + " | ".join(out))

print("\n--- which modes carry theta_4: least-squares slope (through 0) of each term on M_{t-1}, by age (M_{t-1} > 0). Rows sum to the theta~M slope ---")
keys = list(TERMS)
print("age | theta | " + " | ".join(keys))
for age in range(9):
    s = (a == age) & (M > 0)
    den = (M[s] ** 2).sum()
    print(f"{age} | {(th[s]*M[s]).sum()/den:.4f} | " + " | ".join(f"{(TERMS[k][s]*M[s]).sum()/den:.4f}" for k in keys))

print("\n--- same but only states with S <= 0 (isolated positive max) ---")
print("age | n | theta | " + " | ".join(keys))
for age in range(9):
    s = (a == age) & (M > 0) & (Sp == 0)
    den = (M[s] ** 2).sum()
    print(f"{age} | {s.sum()} | {(th[s]*M[s]).sum()/den:.4f} | " + " | ".join(f"{(TERMS[k][s]*M[s]).sum()/den:.4f}" for k in keys))

print("\n--- same, two regressors (M, S+): the S+ slope of each term, by age (M > 0) ---")
print("age | theta | " + " | ".join(keys))
for age in range(9):
    s = (a == age) & (M > 0)
    X = np.column_stack([M[s], Sp[s]])
    G = np.linalg.inv(X.T @ X) @ X.T
    print(f"{age} | M: {(G @ th[s])[0]:.4f} | " + " | ".join(f"{(G @ TERMS[k][s])[0]:.4f}" for k in keys))
    print(f"{age} | S+: {(G @ th[s])[1]:.4f} | " + " | ".join(f"{(G @ TERMS[k][s])[1]:.4f}" for k in keys))

print("\n--- sd of each term's residual after removing its own (M, S+) fit, by age: who makes the spread ---")
print("age | theta resid sd | " + " | ".join(keys))
for age in range(9):
    s = (a == age) & (M > 0)
    X = np.column_stack([M[s], Sp[s]])
    P = X @ np.linalg.inv(X.T @ X) @ X.T if s.sum() < 2000 else None
    G = np.linalg.inv(X.T @ X) @ X.T
    res = lambda y: (y - X @ (G @ y)).std()
    print(f"{age} | {res(th[s]):.4f} | " + " | ".join(f"{res(TERMS[k][s]):.4f}" for k in keys))

print("\n--- all earlier inputs negative (M_{t-1} < 0): theta_4 by t ---")
print("t | n | mean theta | sd | q05 | median | q95 | max | P(theta>0) | mean M | slope theta~M | slope theta~x_{t-1} | R2c(theta ~ x_{t-1})")
xprev = None
for tt in range(1, 10):
    s = (t == tt) & (M < 0)
    if s.sum() < 50:
        print(f"{tt} | n={s.sum()} (too few)"); continue
    q = np.percentile(th[s], [5, 50, 95])
    b, _, r = slope0(th[s], [M[s]])
    print(f"{tt} | {s.sum()} | {th[s].mean():.4f} | {th[s].std():.4f} | {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} | {th[s].max():.4f} | {(th[s]>0).mean():.3f} | {M[s].mean():.3f} | {b[0]:.4f} | - | {r:.3f}")
