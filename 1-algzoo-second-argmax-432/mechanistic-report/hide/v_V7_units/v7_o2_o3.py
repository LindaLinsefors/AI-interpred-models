"""V7 / claims O2 and O3 (independent code). Deterministic: no sampling."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

# ---------------------------------------------------------------- O2
print("================ O2: readout columns of the wave units ================")
print("rows = output position 0..9, columns = wave units")
print("pos   " + " ".join(f"u{j:<6d}" for j in WAVE))
for p in range(10):
    print(f"{p:3d}   " + " ".join(f"{U[p, j]:7.3f}" for j in WAVE))
print("\nper unit: most negative entry (position), second most negative (position), most positive (position)")
for j in range(N):
    col = U[:, j]
    o = np.argsort(col)
    grp = "comp" if j in COMP else "pulse" if j in PULSE else "wave"
    print(f"  unit {j:2d} [{grp:5s}]: min {col[o[0]]:7.3f} at {o[0]}   2nd {col[o[1]]:7.3f} at {o[1]}   "
          f"max {col[o[-1]]:7.3f} at {o[-1]}   column sum {col.sum():7.3f}")

# ---------------------------------------------------------------- O3
print("\n================ O3: sweep x = (0.5, -1, 1.5, -1, -1, -1, v, -1, -1, -1) ================")
base = np.array([0.5, -1, 1.5, -1, -1, -1, 0.0, -1, -1, -1])
H5 = run(base[:6])[0, -1]                       # h_5, independent of v
drive = W @ H5                                  # recurrent drive into step 6
theta = -drive / w                              # z_6[j] = w_j v + drive_j = 0  at  v = theta_j
print("w (input weights):", np.round(w, 3))
print("thresholds theta_j = -(W h_5)_j / w_j for step 6 (unit j is ON for v < theta_j if w_j < 0, for v > theta_j if w_j > 0):")
for j in [1, 6, 7, 8, 4]:
    print(f"  theta_{j} = {theta[j]:.4f}   (w_{j} = {w[j]:8.3f}, drive = {drive[j]:8.3f})")
print("all units: " + "  ".join(f"{j}:{theta[j]:.3f}" for j in range(N)))

vs = [0.30, 0.45, 0.55, 0.85, 1.15, 1.30, 1.45, 1.55, 1.80]
X = np.tile(base, (len(vs), 1))
X[:, 6] = vs
H, Z = run(X, return_pre=True)
m = coords(H)
ans = logits(H).argmax(1)
tgt = second_argmax(X)
print(f"\nw[4] = {w[4]:.4f}")
print("   v   comps on at t=6    h6[4]    h7[5]   |z_clock(t=9)|/w4   answer  target  | h6 comparators (1, 6, 7, 8)")
for i, v in enumerate(vs):
    on = [j for j in COMP if H[i, 6, j] > 0]
    print(f"{v:5.2f}  {str(on):16s} {H[i, 6, 4]:7.3f}  {H[i, 7, 5]:7.3f}   {abs(m['clock'][i, 9]) / w[4]:8.4f}        "
          f"{ans[i]:3d}    {tgt[i]:3d}    | " + " ".join(f"{H[i, 6, j]:7.3f}" for j in COMP))

print("\nfine sweep: where does the answer change, where is the clock amplitude at t = 9 smallest?")
vf = np.round(np.arange(0.0, 2.5001, 0.001), 3)
Xf = np.tile(base, (vf.size, 1))
Xf[:, 6] = vf
Hf = run(Xf)
af = logits(Hf).argmax(1)
tf = second_argmax(Xf)
ch = np.nonzero(np.diff(af))[0]
for c in ch:
    print(f"  answer {af[c]} -> {af[c + 1]} between v = {vf[c]:.3f} and {vf[c + 1]:.3f}")
wrong = af != tf
print(f"  wrong answers for v in: {[(vf[i]) for i in np.nonzero(wrong)[0]][:1]} ... n = {wrong.sum()} of {vf.size}"
      + (f" (range {vf[wrong].min():.3f}..{vf[wrong].max():.3f})" if wrong.any() else ""))
ac = np.abs(coords(Hf)["clock"][:, 9]) / w[4]
k = ac.argmin()
print(f"  clock |z|/w4 at t = 9: min {ac[k]:.4f} at v = {vf[k]:.3f};  at v = 0: {ac[0]:.4f};  at v = 2.5: {ac[-1]:.4f}")
for v in [0.0, 0.3, 0.5, 0.56, 1.0, 1.27, 1.37, 1.38, 1.5, 1.6, 1.67, 1.68, 2.0, 2.5]:
    i = np.searchsorted(vf, v)
    print(f"    v = {v:4.2f}: clock {ac[i]:.4f}   h6[4] {Hf[i, 6, 4]:.3f}   h7[5] {Hf[i, 7, 5]:.3f}   answer {af[i]}")
