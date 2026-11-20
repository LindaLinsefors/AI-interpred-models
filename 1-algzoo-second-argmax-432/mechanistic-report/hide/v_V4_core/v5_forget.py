"""V4 verifier, attack 1: is 'Bc forgets a common shift' special?
(a) geometry of Bc; (b) random 4x4 matrices; (c) every other 4-subset of units and every other gauge direction;
(d) on data: exact sensitivity of theta(t+1) to x_t, by gate pattern.  Seeds 5010 (random matrices), 5011 (data)."""
from itertools import combinations

from vsetup import *

out = Tee("v5_forget.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=220)
one = np.ones(4)

# ------------------------------------------------------------------ (a) geometry
out("=== (a) geometry of Bc ===")
Uu, sv, Vt = np.linalg.svd(Bc)
out("singular values:", sv)
out("right singular vectors (rows):\n", Vt)
out(f"|cos(1, smallest right singular vector)| = {abs(Vt[3] @ one) / 2:.5f};  |Bc 1| = {np.linalg.norm(Bc @ one):.5f};  "
    f"|Bc|_F = {np.linalg.norm(Bc):.4f};  ratio rho = |Bc 1| / (|Bc|_F |1|) = {np.linalg.norm(Bc @ one) / (np.linalg.norm(Bc) * 2):.5f}")
out("column sums 1^T Bc (the common level of the OUTPUT is not removed):", one @ Bc)
ev, V = np.linalg.eig(Bc)
k = int(np.argmax(np.angle(ev) * (np.abs(ev) > 0.5)))
vr = V[:, k]
out(f"right eigenvector of the rotating pair (lambda = {ev[k]:.4f}): {vr}")
out(f"   its common-level fraction |mean|^2*4 / |v|^2 = {abs(vr.mean()) ** 2 * 4 / np.linalg.norm(vr) ** 2:.4f}  "
    "(0 would mean the rotating plane consists of pure differences)")
P = np.eye(4) - np.ones((4, 4)) / 4                       # projector on differences
out("eigenvalues of P Bc P (differences -> differences):", np.linalg.eigvals(P @ Bc @ P))
out("|Bc - Bc P|_F (what is lost by first removing the common level of the input):", np.linalg.norm(Bc - Bc @ P))
out("|Bc - P Bc|_F (what is lost by removing the common level of the output):", np.linalg.norm(Bc - P @ Bc))
kern = V[:, np.abs(ev) < 0.5]
out("the two 'dying' eigenvectors (|lambda| = 0.043) of Bc:\n", kern.T)
# real basis of that invariant plane, and how much of 1 lies in it
Q, _ = np.linalg.qr(np.stack([kern[:, 0].real, kern[:, 0].imag], 1))
out(f"   fraction of 1 inside that invariant plane: {np.linalg.norm(Q.T @ one) ** 2 / 4:.5f}")
out("   Bc^2 1 =", Bc @ Bc @ one, "  Bc^3 1 =", Bc @ Bc @ Bc @ one)
# second direction of that plane, orthogonal to 1
resid = Q @ (Q.T @ one)
other = Q[:, 0] - (Q[:, 0] @ resid) / (resid @ resid) * resid
other /= np.linalg.norm(other)
out("   direction in the dying plane orthogonal to (the projection of) 1:", other, " |Bc other| =", np.linalg.norm(Bc @ other),
    " |Bc^2 other| =", np.linalg.norm(Bc @ Bc @ other))
out("   so a threshold pattern along this direction is passed on once (factor", round(float(np.linalg.norm(Bc @ other)), 3),
    ") and is gone after two steps")

# ------------------------------------------------------------------ (b) random matrices
out("\n=== (b) random 4x4 matrices: rho = |G 1| / (|G|_F |1|); observed rho(Bc) above ===")
rho_obs = np.linalg.norm(Bc @ one) / (np.linalg.norm(Bc) * 2)
rng = np.random.default_rng(5010)
n = 2_000_000
G = rng.standard_normal((n, 4, 4))
rho = np.linalg.norm(G.sum(2), axis=1) / (np.linalg.norm(G.reshape(n, 16), axis=1) * 2)
out(f"  iid Gaussian: median rho = {np.median(rho):.4f}, min over {n} = {rho.min():.5f}, P(rho <= {rho_obs:.5f}) = {(rho <= rho_obs).mean():.2e}")
out(f"     analytic small-rho law P(rho <= r) ~ (16 r^2)^2 / 8 = {((16 * rho_obs ** 2) ** 2) / 8:.2e}")
flat = Bc.ravel()
perm = np.array([rng.permutation(16) for _ in range(400_000)])
Gp = flat[perm].reshape(-1, 4, 4)
rho = np.linalg.norm(Gp.sum(2), axis=1) / (np.linalg.norm(Bc) * 2)
out(f"  entries of Bc randomly permuted (400k): median rho = {np.median(rho):.4f}, min = {rho.min():.5f}, P(rho <= obs) = {(rho <= rho_obs).mean():.2e}")
sg = rng.choice([-1.0, 1.0], size=(400_000, 4, 4))
rho = np.linalg.norm((Bc * sg).sum(2), axis=1) / (np.linalg.norm(Bc) * 2)
out(f"  entries of Bc with random signs (all 65536 patterns sampled 400k times): median rho = {np.median(rho):.4f}, min = {rho.min():.5f}, "
    f"P(rho <= obs) = {(rho <= rho_obs).mean():.2e}")
# random direction instead of 1: |Bc v| / (|Bc|_F) for random unit v  (is the null direction just easy to hit?)
v = rng.standard_normal((2_000_000, 4))
v /= np.linalg.norm(v, axis=1, keepdims=True)
rho = np.linalg.norm(v @ Bc.T, axis=1) / np.linalg.norm(Bc)
out(f"  Bc itself, random unit direction v instead of 1/2: median |Bc v|/|Bc|_F = {np.median(rho):.4f}, P(<= obs) = {(rho <= rho_obs).mean():.2e}")
# near-singularity alone: smallest singular value relative to Frobenius norm for Gaussian matrices
s = np.linalg.svd(G[:400_000], compute_uv=False)
rs_ = s[:, 3] / np.linalg.norm(s, axis=1)
out(f"  near-singularity alone: sigma_min/|.|_F of Bc = {sv[3] / np.linalg.norm(sv):.5f}; Gaussian: median {np.median(rs_):.4f}, "
    f"P(<= obs) = {(rs_ <= sv[3] / np.linalg.norm(sv)).mean():.2e}")

# ------------------------------------------------------------------ (c) other subsets, other gauges
out("\n=== (c) raw form: W[Q,Q] w[Q] for every 4-subset Q of the 16 units; rho(Q) = |W[Q,Q] w[Q]| / (|W[Q,Q]|_F |w[Q]|) ===")
res = []
for Qs in combinations(range(16), 4):
    Qs = list(Qs)
    Wq = W[np.ix_(Qs, Qs)]
    res.append((np.linalg.norm(Wq @ w[Qs]) / (np.linalg.norm(Wq) * np.linalg.norm(w[Qs])), Qs))
res.sort(key=lambda r_: r_[0])
rank = [i for i, r_ in enumerate(res) if r_[1] == COMP][0]
out(f"  COMP = {COMP}: rho = {res[rank][0]:.5f}, rank {rank + 1} of {len(res)} subsets (smallest first)")
out("  five smallest:", [(round(r_[0], 5), r_[1]) for r_ in res[:5]])
out(f"  median over subsets: {np.median([r_[0] for r_ in res]):.4f}")
out("  rows: |W[u,COMP] . w[COMP]| / (|W[u,COMP]| |w[COMP]|) is claim K2 (max 0.038 without units 5, 9).")
out("  the same for the WAVE columns: cos(W[u,WAVE], w[WAVE]) per unit u:")
cw = np.array([W[u, WAVE] @ w[WAVE] / (np.linalg.norm(W[u, WAVE]) * np.linalg.norm(w[WAVE])) for u in range(16)])
out("    ", cw)
out("  and W[u,WAVE].w[WAVE] (raw) per unit u:", np.array([W[u, WAVE] @ w[WAVE] for u in range(16)]))
out("  W[u,COMP].w[COMP] (raw) per unit u:   ", np.array([W[u, COMP] @ w[COMP] for u in range(16)]))
out("  W[u,:].w (all 16 columns) per unit u:  ", W @ w)
out("  |w[COMP]| =", np.linalg.norm(w[COMP]), " |w[WAVE]| =", np.linalg.norm(w[WAVE]), " |w[PULSE]| =", np.linalg.norm(w[PULSE]))

# the forgetting in threshold units, for all receivers: d(pre-activation of u at t+1)/d x_t when all comparators on, pulses off
out("\n  sensitivity of each unit's next pre-activation to x_t through the comparators alone, W[u,COMP].w[COMP], relative to its direct input weight w[u]:")
for u in range(16):
    out(f"    unit {u:2d}: via comparators {W[u, COMP] @ w[COMP]:+8.4f}   direct w[u] {w[u]:+8.4f}   via all 13 ORD units {W[u, ORD] @ w[ORD]:+8.4f}")

# ------------------------------------------------------------------ (d) data: exact sensitivity of theta(t+1) to x_t
out("\n=== (d) data (seed 5011, n = 400000): exact d theta_j(t+1) / d x_t = (W[j,:] diag(gate_t) w) / g_j, by gate pattern at t ===")
x = sample(400_000, 5011)
H, Zp = run(x, return_pre=True)
gate = (Zp > 0)
ncomp = gate[:, :, COMP].sum(2)
u4 = gate[:, :, 4]
sens = (gate * w) @ W[COMP].T / g                          # (B, T, 4)
sens_comp = (gate[:, :, COMP] * w[COMP]) @ W[np.ix_(COMP, COMP)].T / g
out("  columns: j = 1, 6, 7, 8.   'total' uses all 16 units, 'via comparators' only the four comparators")
for label, sel in (("all four comparators on, unit 4 off", (ncomp == 4) & ~u4),
                   ("3 on, unit 4 off", (ncomp == 3) & ~u4), ("2 on, unit 4 off", (ncomp == 2) & ~u4),
                   ("1 on, unit 4 off", (ncomp == 1) & ~u4), ("0 on, unit 4 off", (ncomp == 0) & ~u4),
                   ("unit 4 on (any comparators)", u4)):
    if sel.sum() == 0:
        continue
    out(f"  {label:38s} frac of step-states {sel.mean():.4f}: total mean {sens[sel].mean(0)}  mean|.| {np.abs(sens[sel]).mean(0)}"
        f"   via comparators mean {sens_comp[sel].mean(0)}")
out("  unit 4 on & number of comparators on:", np.bincount(ncomp[u4], minlength=5) / u4.sum())
rs = running_stats(x)
out("  P(all four comparators on) by step:", (ncomp == 4).mean(0))
out("  P(no comparator on) by step:      ", (ncomp == 0).mean(0))
out("  P(1..3 on) by step:               ", ((ncomp > 0) & (ncomp < 4)).mean(0))
# which events cause partial clipping?
newmax = np.zeros_like(x, dtype=bool); newmax[:, 0] = True
newmax[:, 1:] = x[:, 1:] > rs["M1"][:, :-1]
newsec = np.zeros_like(x, dtype=bool)
newsec[:, 1:] = (x[:, 1:] < rs["M1"][:, :-1]) & (x[:, 1:] > rs["M2"][:, :-1])
newthird = np.zeros_like(x, dtype=bool)
newthird[:, 2:] = (x[:, 2:] < rs["M2"][:, 1:-1]) & (x[:, 2:] > rs["M3"][:, 1:-1])
rest = ~(newmax | newsec | newthird)
for label, ev_ in (("x_t new max", newmax), ("x_t new runner-up", newsec), ("x_t new third", newthird), ("x_t below the top three", rest)):
    e = ev_[:, 2:]
    nc = ncomp[:, 2:][e]
    out(f"  t >= 2, {label:24s}: frac {e.mean():.4f}; P(#comparators on = 0..4) = {np.bincount(nc, minlength=5) / nc.size}")
