"""E32. The transform between mode coordinates and the scores ('tape').  Exact synthesis:
    s[i] = sum_real zeta_m p_m[i] + sum_pairs 2 Re(z_m p_m[i]) + U[i, pulses] h[pulses],   p_m = U_o r_m.
If a resting step (z -> lam z) moved every entry one index down, p_m[i] would be p_m[0] lam_m^i.
Here: p_m, the best mu in p_m[i+1] = mu p_m[i] on indices 0..7, and how well c mu^i and c lam^i fit.
Run:  python -B hide/e32_tape_transform.py > hide/outputs/e32_tape_transform.txt"""
from common import *
from modes import RIGHT, LAM, NAMES, IS_COMPLEX

Uo = U[:, ORD]
i8 = np.arange(8)
pol = lambda v: f"{abs(v):6.3f} at {np.degrees(np.angle(v)):7.1f}"
def fitq(p, mu):                                     # share of |p|^2 on indices 0..7 explained by c mu^i
    b = mu ** i8; c = (b.conj() @ p[:8]) / (b.conj() @ b)
    return 1 - np.abs(p[:8] - c * b).__pow__(2).sum() / np.abs(p[:8]).__pow__(2).sum(), c
for n in NAMES:
    p, lam = Uo @ RIGHT[n], LAM[n]
    mu = (p[:7].conj() @ p[1:8]) / (p[:7].conj() @ p[:7])
    q_mu, c_mu = fitq(p, mu); q_lam, c_lam = fitq(p, lam)
    print(f"\n{n}: lam = {pol(lam)}   best mu = {pol(mu)}   fit with mu {q_mu:.3f}, with lam {q_lam:.3f};  c (lam fit) = {pol(c_lam)}")
    if IS_COMPLEX[n]:
        print("   |p[i]|  :", " ".join(f"{abs(v):6.2f}" for v in p))
        print("   arg p[i]:", " ".join(f"{np.degrees(np.angle(v)):6.0f}" for v in p))
        print("   step in arg:", " ".join(f"{np.degrees(np.angle(p[k + 1] / p[k])):6.0f}" for k in range(9)))
    else:
        print("   p[i]    :", " ".join(f"{v.real:6.2f}" for v in p))
