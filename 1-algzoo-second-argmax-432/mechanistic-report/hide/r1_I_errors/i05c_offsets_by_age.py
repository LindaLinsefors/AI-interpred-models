"""I5 (detail). Takeover offsets eps* (runner-up vs third, from i04b_RT.npz) against the ages of the max and of the
earlier candidate AT THE STEP THE LATER CANDIDATE ARRIVES ('M before both' only), and record offsets against the age of the max."""
from icommon import *

Z = np.load(HIDE / "r1_I_errors" / "i04b_RT.npz")
eps, pe, pl, pM = Z["eps"], Z["pe"], Z["pl"], Z["pM"]
fin = np.isfinite(eps) & (pM < pe)
dM, de = pl - pM, pl - pe
print("runner-up vs third, max before both: eps* by d_M = age of the max when the later candidate arrives")
print(fmt_table([[d, (fin & (dM == d)).sum(), f"{np.median(eps[fin & (dM == d)]):+.4f}", f"{np.mean(np.abs(eps[fin & (dM == d)])):.4f}",
                  f"{np.subtract(*np.quantile(eps[fin & (dM == d)], [.75, .25])):.4f}", f"{np.mean(np.abs(eps[fin & (dM == d)]) > 0.03):.3f}"] for d in range(2, 10)],
                ["d_M", "n", "median eps*", "mean abs eps*", "IQR", "P(abs eps* > 0.03)"]))
print("\nsame, by d_e = age of the earlier candidate when the later one arrives")
print(fmt_table([[d, (fin & (de == d)).sum(), f"{np.median(eps[fin & (de == d)]):+.4f}", f"{np.mean(np.abs(eps[fin & (de == d)])):.4f}",
                  f"{np.subtract(*np.quantile(eps[fin & (de == d)], [.75, .25])):.4f}", f"{np.mean(np.abs(eps[fin & (de == d)]) > 0.03):.3f}"] for d in range(1, 9)],
                ["d_e", "n", "median eps*", "mean abs eps*", "IQR", "P(abs eps* > 0.03)"]))
Z = np.load(HIDE / "r1_I_errors" / "i04b_MR.npz")
eps, pe, pl, sag = Z["eps"], Z["pe"], Z["pl"], Z["sag"]
fin = np.isfinite(eps); d_ = pl - pe
print("\nmax vs later input: record offset eps* by d = age of the max when the later input arrives; sag = theta_4/M - 1 at that step")
print(fmt_table([[d, (fin & (d_ == d)).sum(), f"{np.median(sag[fin & (d_ == d)]):+.3f}", f"{np.median(eps[fin & (d_ == d)]):+.4f}", f"{np.mean(np.abs(eps[fin & (d_ == d)])):.4f}",
                  f"{np.subtract(*np.quantile(eps[fin & (d_ == d)], [.75, .25])):.4f}", f"{np.mean(eps[fin & (d_ == d)] > sag[fin & (d_ == d)]):.3f}"] for d in range(1, 10)],
                ["d", "n", "median sag of theta_4", "median eps*", "mean abs eps*", "IQR", "P(eps* > sag)"]))
