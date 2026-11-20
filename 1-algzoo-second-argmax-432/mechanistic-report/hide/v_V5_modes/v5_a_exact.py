"""Verifier V5, part A: claims M1-M4 (exact, from the weights only). Own code; uses only common.py / modes.py.

Run:  python -B v5_a_exact.py > v5_a_exact.txt
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *      # noqa
from modes import *       # noqa

np.set_printoptions(precision=4, suppress=True, linewidth=220)
CI = [ORD.index(j) for j in COMP]          # comparator indices inside ORD
WI = [ORD.index(j) for j in WAVE]          # wave indices inside ORD
Uo = U[:, ORD]
A = W[np.ix_(ORD, ORD)]

print("=" * 100)
print("M1  eigenvalues of A = W[ORD, ORD]  (own eig call, independent of modes.py)")
lam, V = np.linalg.eig(A)
order = np.argsort(-np.abs(lam))
for k in order:
    print(f"   lam = {lam[k].real:+.6f} {lam[k].imag:+.6f}i   |lam| = {abs(lam[k]):.4f}   angle = {np.degrees(np.angle(lam[k])):8.3f} deg")
print("   modes.py LAM:", {n: (round(abs(LAM[n]), 4), round(float(np.degrees(np.angle(LAM[n]))), 2)) for n in NAMES})
# check modes.py gauge and biorthogonality
print("   gauge check LEFT @ W[ORD,4]:", {n: np.round(LEFT[n] @ W[ORD, 4], 6) for n in NAMES})
print("   LEFT[n] @ RIGHT[n]:", {n: np.round(LEFT[n] @ RIGHT[n], 6) for n in NAMES})
print("   max |A RIGHT - lam RIGHT|:", max(np.abs(A @ RIGHT[n] - LAM[n] * RIGHT[n]).max() for n in NAMES))
print("   max |LEFT A - lam LEFT|:", max(np.abs(LEFT[n] @ A - LAM[n] * LEFT[n]).max() for n in NAMES))

k_lev = order[0]
v_lev = V[:, k_lev].real
v_lev = v_lev / v_lev[ORD.index(2)]
print("\n   level right eigenvector / entry of unit 2 (units", ORD, "):")
print("   ", np.round(v_lev, 3))
claim = np.array([1.35, 0.51, 1, 0.19, 0.63, 0.79, 0.70, 1.06, 1.14, 1.59, 0.43, 1.23, 0.81])
print("    claim:", claim)
print("    max abs diff to claim:", np.abs(v_lev - claim).max(), "  all one sign:", bool((v_lev > 0).all() or (v_lev < 0).all()))
l_lev = np.linalg.inv(V)[k_lev].real
print("   level LEFT eigenvector signs (not claimed):", np.sign(l_lev / l_lev[ORD.index(2)]).astype(int))

k_echo = order[-1]
v_echo = V[:, k_echo].real
l_echo = np.linalg.inv(V)[k_echo].real
wo = w[ORD]
cosv = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
print(f"\n   echo eigenvalue {lam[k_echo].real:.5f}: |cos(w[ORD], RIGHT echo)| = {abs(cosv(wo, v_echo)):.4f}"
      f"   |cos(w[ORD], LEFT echo)| = {abs(cosv(wo, l_echo)):.4f}")
print("   |A w_o| / |w_o| =", np.linalg.norm(A @ wo) / np.linalg.norm(wo), "  |A|_2 =", np.linalg.norm(A, 2))
print("   invariant per-unit ratio (A w_o)_i / (w_o)_i :", np.round((A @ wo) / wo, 4))
print("   invariant per-unit ratio (A^2 w_o)_i / (w_o)_i:", np.round((A @ A @ wo) / wo, 4))
print("   w[ORD] =", np.round(wo, 3))
print("   mode coordinates of w[ORD]  (LEFT[n] @ w_o; = what input x_t=1 adds to each coordinate on a resting step):")
for n in NAMES:
    c = LEFT[n] @ wo
    print(f"      {n:7s} {abs(c):9.5f} @ {np.degrees(np.angle(c)):7.1f} deg    (per unit of w_4 = {abs(c)/w[4]:.5f})")

print("=" * 100)
print("M2  fraction of each LEFT vector's squared norm on the comparators (raw units)")
claimM2 = dict(level=0.041, runner=0.506, clock=0.966, h2=0.606, alt=0.106, h3=0.347, short=0.916, trans=0.976)
for n in NAMES:
    L2 = np.abs(LEFT[n]) ** 2
    print(f"   {n:7s} comp frac = {L2[CI].sum() / L2.sum():.4f}   claim = {claimM2.get(n, float('nan'))}"
          f"   largest unit = {ORD[int(L2.argmax())]} ({L2.max() / L2.sum():.3f})   frac on unit 2 = {L2[ORD.index(2)] / L2.sum():.4f}")
print("   (same with L1 norm instead of squared norm)")
for n in NAMES:
    L1 = np.abs(LEFT[n])
    print(f"   {n:7s} comp frac (L1) = {L1[CI].sum() / L1.sum():.4f}")

print("=" * 100)
print("M3  readout of the clock mode per unit coordinate: 2|R|, R = U[p, group] RIGHT['clock'][group]")
r = RIGHT["clock"]
Rc = U[:, COMP] @ r[CI]
Rw = U[:, WAVE] @ r[WI]
Ra = Uo @ r
for nm, R in (("comparators", Rc), ("wave units", Rw), ("all 13", Ra)):
    print(f"   {nm:12s} 2|R| p=0..9: {np.round(2 * np.abs(R), 2)}")
    print(f"   {'':12s} phase deg  : {np.round(np.degrees(np.angle(R)), 0)}")
nn = np.linalg.norm
print(f"   ratio |total| / (|comp| + |wave|) over all 10 positions = {nn(Ra) / (nn(Rc) + nn(Rw)):.4f}")
print(f"   same over positions 0..7 = {nn(Ra[:8]) / (nn(Rc[:8]) + nn(Rw[:8])):.4f};  0..8 = {nn(Ra[:9]) / (nn(Rc[:9]) + nn(Rw[:9])):.4f}")
print(f"   |total|/|comp| = {nn(Ra) / nn(Rc):.4f}   |total|/|wave| = {nn(Ra) / nn(Rw):.4f}")
# mean-removed version (softmax only sees differences between positions)
c = lambda R, k=10: R[:k] - R[:k].mean()
print(f"   mean-over-positions removed (10 pos): ratio = {nn(c(Ra)) / (nn(c(Rc)) + nn(c(Rw))):.4f};  (pos 0..8): {nn(c(Ra, 9)) / (nn(c(Rc, 9)) + nn(c(Rw, 9))):.4f}")

print("\n   the same cancellation ratio for EVERY mode (10 positions; complex 10-vector norms for pairs):")
print("   mode      |tot|/(|comp|+|wave|)   mean-removed    |tot|/sum_i|U[:,i] v_i| (per-unit split)")
for n in NAMES:
    r = RIGHT[n]
    Rc, Rw, Ra = U[:, COMP] @ r[CI], U[:, WAVE] @ r[WI], Uo @ r
    per_unit = sum(nn(Uo[:, i] * r[i]) for i in range(13))
    print(f"   {n:7s}   {nn(Ra) / (nn(Rc) + nn(Rw)):8.4f}              {nn(c(Ra)) / (nn(c(Rc)) + nn(c(Rw))):8.4f}        {nn(Ra) / per_unit:8.4f}")

# null model: random vectors
rng = np.random.default_rng(5001)
def ratio(v):
    Rc, Rw, Ra = U[:, COMP] @ v[CI], U[:, WAVE] @ v[WI], Uo @ v
    return nn(Ra) / (nn(Rc) + nn(Rw))
rc = RIGHT["clock"]
null_gauss = np.array([ratio(rng.standard_normal(13) + 1j * rng.standard_normal(13)) for _ in range(20000)])
null_phase = np.array([ratio(np.abs(rc) * np.exp(2j * np.pi * rng.random(13))) for _ in range(20000)])
print(f"\n   null A (iid complex Gaussian 13-vectors, seed 5001): ratio quantiles 0.1%/1%/5%/50% = "
      f"{np.quantile(null_gauss, [0.001, 0.01, 0.05, 0.5]).round(3)};  P(ratio <= clock ratio) = {(null_gauss <= ratio(rc)).mean():.5f}")
print(f"   null B (clock |v_i| kept, phases uniform random):  ratio quantiles 0.1%/1%/5%/50% = "
      f"{np.quantile(null_phase, [0.001, 0.01, 0.05, 0.5]).round(3)};  P(ratio <= clock ratio) = {(null_phase <= ratio(rc)).mean():.5f}")

# relation to the null space of U_o (10 x 13 -> exact 3-dim kernel)
sv = np.linalg.svd(Uo, compute_uv=False)
_, s, Vt = np.linalg.svd(Uo)
print("\n   singular values of U_o:", np.round(sv, 3))
K = Vt[10:]                                   # exact kernel basis (3 x 13)
for n in NAMES:
    r = RIGHT[n]
    cols = [r.real, r.imag] if IS_COMPLEX[n] else [r.real]
    fr = [np.linalg.norm(K @ v) ** 2 / np.linalg.norm(v) ** 2 for v in cols]
    print(f"   {n:7s} fraction of |v|^2 inside ker(U_o) (raw unit scaling; NOT scale invariant): {np.round(fr, 3)}")

print("=" * 100)
print("M4  size of U_o v over positions 0..8 after removing their mean; v = Re / Im of RIGHT (modes.py gauge)")
print("   mode     L2-norm(Re)  L2-norm(Im)   rms(Re)   rms(Im)   [x2 for pairs: L2]   max/min over phase (L2, invariant to phase gauge)")
for n in NAMES:
    r = RIGHT[n]
    y = Uo[:9] @ r
    y = y - y.mean()
    yr, yi = y.real, y.imag
    if IS_COMPLEX[n]:
        s2 = np.linalg.svd(np.stack([yr, yi], 1), compute_uv=False)
        print(f"   {n:7s} {nn(yr):10.3f} {nn(yi):10.3f}   {nn(yr) / 3:8.3f} {nn(yi) / 3:8.3f}    [{2 * nn(yr):.3f} / {2 * nn(yi):.3f}]     {s2[0]:.3f} / {s2[1]:.3f}")
    else:
        print(f"   {n:7s} {nn(yr):10.3f} {'':10s}   {nn(yr) / 3:8.3f}")
print("   same over all 10 positions (mean removed):")
for n in NAMES:
    r = RIGHT[n]
    y = Uo @ r
    y = y - y.mean()
    print(f"   {n:7s} {nn(y.real):10.3f} {nn(y.imag) if IS_COMPLEX[n] else 0:10.3f}")
print("   same over positions 0..8 WITHOUT removing the mean:")
for n in NAMES:
    r = RIGHT[n]
    y = Uo[:9] @ r
    print(f"   {n:7s} {nn(y.real):10.3f} {nn(y.imag) if IS_COMPLEX[n] else 0:10.3f}")

print("=" * 100)
print("INVARIANCE CHECK: random positive per-unit rescaling h_i -> c_i h_i (W' = C W C^-1, w' = C w, U' = U C^-1), seed 5002")
rng = np.random.default_rng(5002)
cs = np.exp(rng.normal(0, 1.0, 16))
Wp = (cs[:, None] * W) / cs[None, :]
wp = cs * w
Up = U / cs[None, :]
Ap = Wp[np.ix_(ORD, ORD)]
lamp, Vp = np.linalg.eig(Ap)
Lp = np.linalg.inv(Vp)
print("   eigenvalues unchanged:", np.allclose(np.sort_complex(lamp), np.sort_complex(lam)))
def mode_of(lam_target):
    k = int(np.argmin(np.abs(lamp - lam_target)))
    cc = Lp[k] @ Wp[ORD, 4]
    return Lp[k] / cc, Vp[:, k] * cc
print("   (gauge in the rescaled net fixed the same way, LEFT' @ W'[ORD,4] = 1; then coordinates scale by c_4 =", round(cs[4], 4), ")")
for n in NAMES:
    Lr, Rr = mode_of(LAM[n])
    L2 = np.abs(Lr) ** 2
    fr_comp = L2[CI].sum() / L2.sum()
    L2o = np.abs(LEFT[n]) ** 2
    Rc, Rw, Ra = Up[:, COMP] @ Rr[CI], Up[:, WAVE] @ Rr[WI], Up[:, ORD] @ Rr
    y = Up[:9][:, ORD] @ Rr
    y = y - y.mean()
    yo = Uo[:9] @ RIGHT[n]
    yo = yo - yo.mean()
    print(f"   {n:7s} M2 comp frac: raw {L2o[CI].sum() / L2o.sum():.3f} -> rescaled {fr_comp:.3f} | "
          f"M3 ratio: raw {ratio(RIGHT[n]):.4f} -> rescaled {nn(Ra) / (nn(Rc) + nn(Rw)):.4f} | "
          f"M4 size(Re): raw {nn(yo.real):.3f} -> rescaled {nn(y.real):.3f}, x w_4': {nn(y.real) * wp[4]:.2f} (raw x w_4: {nn(yo.real) * w[4]:.2f})")
v_lev_p = Vp[:, int(np.argmin(np.abs(lamp - lam[k_lev])))].real
print("   level eigenvector still one-signed:", bool((v_lev_p > 0).all() or (v_lev_p < 0).all()),
      "  ratios to unit 2 change (not invariant):", np.round(v_lev_p / v_lev_p[ORD.index(2)], 2))
v_echo_p = Vp[:, int(np.argmin(np.abs(lamp - lam[k_echo])))].real
print(f"   |cos(w'[ORD], echo eigenvector')| = {abs(cosv(wp[ORD], v_echo_p)):.4f}  (raw 0.998: NOT invariant)")
