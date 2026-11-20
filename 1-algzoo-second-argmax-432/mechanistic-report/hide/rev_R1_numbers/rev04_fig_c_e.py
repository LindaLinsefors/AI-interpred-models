"""Reviewer R1: the regenerated figure (fig_mechanism.py of 17:56) against the caption in report.md (17:24).
Panel (c): best-fit cosine amplitudes of the plotted dots; panel (e): what row 9 of Sigma is.
Run:  python -B rev04_fig_c_e.py > rev04_fig_c_e_out.txt   (reads only)"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=220)
d = np.load(HIDE / "fig_mechanism.npz")
om = 42.748
vd, vA, vM = d["vote_d"], d["vote_A"], d["vote_Am"]
c = np.cos(np.radians(om * vd))
for lab, y in (("V (comparators)", vA), ("m (other units)", vM)):
    a_all = (y * c).sum() / (c * c).sum()
    pos = vd >= 0; neg = vd <= 0
    a_pos = (y[pos] * c[pos]).sum() / (c[pos] ** 2).sum()
    a_neg = (y[neg] * c[neg]).sum() / (c[neg] ** 2).sum()
    X = np.column_stack([c, np.ones_like(c)])
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2_60 = 1 - ((y - (-60 if lab[0] == 'V' else 60) * c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r2_66 = 1 - ((y - (-66 if lab[0] == 'V' else 66) * c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"{lab}: least-squares amplitude of cos(42.75 d): all d {a_all:6.1f};  d >= 0 only {a_pos:6.1f};  d <= 0 only {a_neg:6.1f};  with offset: {co[0]:6.1f} cos + {co[1]:5.1f};   R2 with amplitude 60: {r2_60:.3f}, with 66: {r2_66:.3f}")
print("V + m by d (mean over ages):", {int(k): round(float(np.mean((vA + vM)[vd == k])), 1) for k in range(-7, 8)})
print("value at d = 0:  V", np.round(vA[vd == 0], 1), "  m", np.round(vM[vd == 0], 1))

print("\npanel (e) / section 3.2, row 9 of Sigma = U_o A pinv(U_o):")
Uo = U[:, ORD]; A = W[np.ix_(ORD, ORD)]
u_, s_, vt = np.linalg.svd(Uo)
print("  singular values of U_o:", np.round(s_, 3))
print("  left singular vector of the smallest one (0.29):", np.round(u_[:, 9], 3), "  cos with (1,..,1)/sqrt(10):", round(abs(u_[:, 9].sum()) / np.sqrt(10), 4))
print("  |1^T U_o| =", round(np.linalg.norm(Uo.sum(0)), 3), "  (column sums of U over positions; per unit:", np.round(U.sum(0), 2), ")")
Sig = Uo @ A @ np.linalg.pinv(Uo)
print("  row 9 of Sigma:", np.round(Sig[9], 2))
# the same row if the smallest singular direction is left out of the pseudo-inverse
pinv9 = vt[:9].T @ np.diag(1 / s_[:9]) @ u_[:, :9].T
Sig9 = Uo @ A @ pinv9
print("  row 9 with the smallest singular direction dropped from the pseudo-inverse:", np.round(Sig9[9], 2))
print("  rows 0-8 change by at most", round(np.abs(Sig9[:9] - Sig[:9]).max(), 3), " superdiagonal then:", np.round([Sig9[i, i + 1] for i in range(9)], 2))
x = sample(200_000, 77)
H = run(x)
L = H @ U.T
print("  on data: rms of sum_i s_t[i] =", round(float(np.sqrt((L.sum(2) ** 2).mean())), 2), " rms of s_t[9] =", round(float(np.sqrt((L[:, :, 9] ** 2).mean())), 2),
      " rms of a single score =", round(float(np.sqrt((L ** 2).mean())), 2))
