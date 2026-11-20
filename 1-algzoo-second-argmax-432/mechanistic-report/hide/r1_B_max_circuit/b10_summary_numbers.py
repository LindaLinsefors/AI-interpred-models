"""B10. A few summary numbers for findings.md (seed 1100, 1101; 400k each).
(a) min/max of theta_4 after all-negative prefixes.
(b) sag = M - theta_4 = eps(a)*M + kappa(a)*(M - S+): table by age; pooled non-oscillating part.
(c) compact formula check: theta_4 ~ c_M M + c_S S+ + c_k (M - S+) cos(phi_a) with the clock part taken from the measured clock coordinate.
(d) how often is the false-firing input the final runner-up / what the network answers then.
"""
from blib import *

X, TH, M_, S_, A_, T_ = [], [], [], [], [], []
TERMS = None
for seed in (1100, 1101):
    x = sample(400_000, seed)
    H = run(x)
    st = prefix_stats(x)
    X.append(x[:, 1:]); TH.append(theta4(H[:, :-1])); M_.append(st["M1"][:, :-1]); S_.append(st["M2"][:, :-1]); A_.append(st["a1"][:, :-1])
    T_.append(np.broadcast_to(np.arange(1, 10), (len(x), 9)))
    tm = theta4_terms(H[:, :-1])
    TERMS = {k: [v] for k, v in tm.items()} if TERMS is None else {k: TERMS[k] + [tm[k]] for k in tm}
x, th, M, S, a, t = map(np.concatenate, (X, TH, M_, S_, A_, T_))
TERMS = {k: np.concatenate(v) for k, v in TERMS.items()}
Sp = pos(np.where(np.isfinite(S), S, 0))

neg = M < 0
print(f"(a) all earlier inputs negative: n = {neg.sum()} states; theta_4 min {th[neg].min():.5f}, max {th[neg].max():.5f}, mean {th[neg].mean():.5f}; "
      f"fraction with theta_4 > 0: {(th[neg] > 0).mean():.6f}")
print("    in units of the largest (least negative) earlier input: median theta_4/|M| =", round(float(np.median(th[neg] / -M[neg])), 4))

print("\n(b) sag = M - theta_4 = eps*M + kappa*(M - S+), least squares by age (states with M > 0)")
print("age | eps | kappa | R2c of the fit to theta_4 | slow part (level+runner terms): c_M, c_S | clock term: c_M, c_S | short term: c_M, c_S")
for age in range(9):
    s = (a == age) & (M > 0)
    Xr = np.column_stack([M[s], Sp[s]])
    G = np.linalg.inv(Xr.T @ Xr) @ Xr.T
    b = G @ th[s]
    r2 = 1 - (th[s] - Xr @ b).var() / th[s].var()
    slow = G @ (TERMS["level"][s] + TERMS["runner"][s])
    ck = G @ TERMS["clock"][s]
    sh = G @ TERMS["short"][s]
    print(f"{age} | {1-b[0]-b[1]:.4f} | {b[1]:.4f} | {r2:.4f} | {slow[0]:.3f}, {slow[1]:+.3f} | {ck[0]:+.3f}, {ck[1]:+.3f} | {sh[0]:+.3f}, {sh[1]:+.3f}")

print("\n(c) pooled over ages >= 2 (M > 0): slow part of theta_4 (level + runner terms) = c_M M + c_S S+")
s = (a >= 2) & (M > 0)
b, _, r = slope0(TERMS["level"][s] + TERMS["runner"][s], [M[s], Sp[s]])
print(f"    c_M = {b[0]:.4f}, c_S = {b[1]:.4f}, R2c = {r:.4f}")
resid = th[s] - TERMS["level"][s] - TERMS["runner"][s] - TERMS["clock"][s]
print(f"    theta_4 - (level + runner + clock terms): mean {resid.mean():+.4f}, sd {resid.std():.4f}  (all other modes and pulses)")
print(f"    theta_4 - (level + runner terms): sd {(th[s] - TERMS['level'][s] - TERMS['runner'][s]).std():.4f};  sd of theta_4 - M: {(th[s] - M[s]).std():.4f}")
cc = np.corrcoef(TERMS["clock"][s], th[s] - M[s])[0, 1]
print(f"    correlation between the clock term and theta_4 - M (ages >= 2): {cc:.4f}")
for age in range(2, 9):
    s = (a == age) & (M > 0)
    d = th[s] - M[s]
    b1 = np.polyfit(TERMS["clock"][s], d, 1)
    print(f"    age {age}: corr(clock term, theta_4 - M) = {np.corrcoef(TERMS['clock'][s], d)[0,1]:+.3f}; mean clock term / M = {(TERMS['clock'][s]*M[s]).sum()/(M[s]**2).sum():+.4f}; "
          f"mean (theta_4 - M)/M = {(d*M[s]).sum()/(M[s]**2).sum():+.4f}")
