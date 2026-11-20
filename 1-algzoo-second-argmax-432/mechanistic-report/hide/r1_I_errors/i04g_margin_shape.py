"""I4 (shape of the comparison). Is the margin proportional to the gap, or does it saturate?
Median of delta(eps)/M along the sweeps of i04b (same samples, first 20000 sequences), and the local slope
d delta / d eps (logits per unit of gap/M) in each interval."""
from icommon import *
import i04b_takeover as tk

Z = np.load(HIDE / "r1_I_errors" / "i04b_RT.npz")
n = 20000
x, pe, pl, M = Z["x"][:n], Z["pe"][:n], Z["pl"][:n], Z["M"][:n]
D, _ = tk.sweep(x, pe, pl, "RT", M)
pts = [-0.15, -0.1, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15]
idx = [int(np.argmin(np.abs(tk.EPS - p))) for p in pts]
print("runner-up vs third: delta(eps) = logit[l] - logit[e] at x_l = x_e + eps M  (n = 20000)")
rows = []
for k, i in enumerate(idx):
    v = D[:, i] / M
    sl = "" if k == 0 else f"{np.median((D[:, i] - D[:, idx[k-1]]) / M / (tk.EPS[i] - tk.EPS[idx[k-1]])):.0f}"
    rows.append([f"{tk.EPS[i]:+.3f}", f"{np.median(v):+.2f}", f"{np.quantile(v, .1):+.2f}", f"{np.quantile(v, .9):+.2f}", sl])
print(fmt_table(rows, ["eps", "median delta/M", "10%", "90%", "median slope since previous row (logits/M per unit eps)"]))

Z = np.load(HIDE / "r1_I_errors" / "i04b_MR.npz")
x, pe, pl, M = Z["x"][:n], Z["pe"][:n], Z["pl"][:n], Z["M"][:n]
D, _ = tk.sweep(x, pe, pl, "MR", M)
print("\nmax vs later input: delta(eps) = logit[e] - logit[l] at x_l = M (1 + eps)  (n = 20000)")
rows = []
for k, i in enumerate(idx):
    v = D[:, i] / M
    sl = "" if k == 0 else f"{np.median((D[:, i] - D[:, idx[k-1]]) / M / (tk.EPS[i] - tk.EPS[idx[k-1]])):.0f}"
    rows.append([f"{tk.EPS[i]:+.3f}", f"{np.median(v):+.2f}", f"{np.quantile(v, .1):+.2f}", f"{np.quantile(v, .9):+.2f}", sl])
print(fmt_table(rows, ["eps", "median delta/M", "10%", "90%", "median slope since previous row"]))
