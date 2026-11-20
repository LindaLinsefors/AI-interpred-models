"""Extra checks behind my own assessment of the decomposition-route paper (these are NOT in the paper).

(a) what n2 holds; (b) how often each tape unit is off; (c) what switches n0 off two steps after a new maximum;
(d) Table 1's wording for n9; (e) targeted and global misfire interventions.
Reuses the sample and helpers of check_decomposition_paper_measured.py (N = 2^19, seed 0).
Run:  python -B check_decomposition_paper_extra.py   (about 1 min)
"""
import contextlib, io, runpy
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(HERE / "check_decomposition_paper_measured.py"))
N, T, x, w, Whh = g["N"], g["T"], g["x"], g["w"], g["Whh"]
G, ans, run, off, frac, rows = g["G"], g["ans"], g["run"], g["off"], g["frac"], g["rows"]
m, pm, p2, p3, xm, x2 = g["m"], g["pm"], g["p2"], g["p3"], g["xm"], g["x2"]
steps = np.arange(T)
newmax = (x > m) & (steps >= 1)

# (a) n2 against the maximum of the inputs before the latest one
h = np.zeros((N, 16)); H = []
for t in range(T):
    h = np.maximum(x[:, t, None] * w + h @ Whh.T, 0); H.append(h.astype(np.float32))
H = np.stack(H, 1)
print("(a) h_t[2] against max(0, x_0..x_(t-1)); the paper only cites Whh[2,2] = 0.97, Whh[2,4] = 1.73 (1.73 x 10.16 = 17.6):")
for t in (2, 5, 9):
    prev = np.maximum(m[:, t], 0)
    slope, icpt = np.polyfit(prev, H[:, t, 2], 1)
    print(f"    t = {t}: correlation {np.corrcoef(H[:, t, 2], prev)[0, 1]:.3f}, slope {slope:.2f}, intercept {icpt:.2f}")

# (b) gates of the tape units
tape = [2, 12, 13, 14, 3, 10, 11, 15, 0]
n4 = G[:, 1:, 4]
print("\n(b) share of steps >= 1 at which each tape unit is off (paper: n2 n12 n13 n14 'nearly always on'; n3 n10 n11 n15 'thresholds of their own'):")
print("    all steps:          " + "  ".join(f"n{j}: {frac(~G[:, 1:, j]):.3f}" for j in tape))
print("    steps with n4 on:   " + "  ".join(f"n{j}: {frac(~G[:, 1:, j][n4]):.3f}" for j in tape))
print("    steps with n4 off:  " + "  ".join(f"n{j}: {frac(~G[:, 1:, j][~n4]):.3f}" for j in tape))


# (c) n0 two steps after a new maximum that fired n4 (no new maximum since)
def age2(Gx):
    sel = np.zeros((N, T), bool)
    sel[:, 3:] = newmax[:, 1:-2] & Gx[:, 1:-2, 4] & ~newmax[:, 2:-1] & ~newmax[:, 3:]
    return sel


def terms(sel):
    n, t = np.where(sel)
    return (H[n, t - 1] * Whh[0]).mean(0)


sel = age2(G)
rec = np.zeros((N, T), bool); rec[:, 2:] = G[:, 1:-1, 5] & ~G[:, 1:-1, 9] & ~G[:, :-2, 4]   # step after an ordinary record step
print("\n(c) n0 two steps after a new maximum (paper: 'off, because n7 and n9 were both on'):")
print(f"    model: n0 off at {frac(~G[:, :, 0][sel]):.3f} of these steps; one step after an ordinary (non-max) record step: off at {frac(~G[:, :, 0][rec]):.3f}")
for name, k in [("after a new max", sel), ("after an ordinary record", rec)]:
    c = terms(k)
    print(f"    mean terms Whh[0,j] h_(t-1)[j], {name}: " + "  ".join(f"n{j}: {c[j]:+.1f}" for j in [5, 7, 8, 9, 2, 1, 6, 10, 14]) + f"   sum {c.sum():+.1f}")
for name, units in [("n9", [9]), ("n5 (the maximum's record)", [5]), ("n5 and n9", [5, 9])]:
    def f(t, Gs, units=units):
        mask = np.zeros((N, 16), bool)
        if t:
            mask[:, units] = Gs[-1][:, 4][:, None]
        return mask
    Gf, _, lf = run(x, f)
    print(f"    {name} forced off at the step after n4 fires: n0 off two steps after a new max at {frac(~Gf[:, :, 0][age2(Gf)]):.3f}; "
          f"decisions kept {frac(lf.argmax(1) == ans):.3f}")

n, t = np.where(G[:, :-1, 4])                              # steps t + 1 that follow a step at which n4 fired
pre0 = (H[n, t] * Whh[0]).sum(1) + w[0] * x[n, t + 1]
print(f"    n0 one step after n4 fires: mean pre-activation {pre0.mean():+.1f}, of which {(H[n, t, 4] * Whh[0, 4]).mean():+.1f} from n4 [paper +6.6 and +4.2]")

# (d) Table 1's wording for n9
both_off = ~G[:, :-1, 7] & ~G[:, :-1, 8]
print(f"\n(d) n9 gate == (n7 and n8 both off one step earlier): {frac(G[:, 1:, 9] == both_off):.4f}; == n4 gate one step earlier: {frac(G[:, 1:, 9] == G[:, :-1, 4]):.4f}")

# (e) misfire interventions
mis = (p2 > pm) & G[rows, p2, 4]
k = mis & (xm - x2 < 0.05)


def targeted(t, Gs):
    mask = np.zeros((N, 16), bool); mask[:, 4] = mis & (p2 == t)
    return mask


a_t = run(x, targeted)[2].argmax(1)
a_g = run(x, off(4, lambda t, Gs: x[:, t] <= m[:, t]))[2].argmax(1)
print(f"\n(e) near-tie misfires ({frac(k):.4f} of sequences): model answers the max on {frac(ans[k] == pm[k]):.3f} [14%]; "
      f"with n4 forced off at that step {frac(a_t[k] == pm[k]):.3f} [6%]")
print(f"    global n4 intervention: errors answering the max {frac(ans == pm):.4f} -> {frac(a_g == pm):.4f} [-15%], "
      f"of which newly created {frac((a_g == pm) & (ans != pm)):.4f}; answering the third {frac(ans == p3):.4f} -> {frac(a_g == p3):.4f} [-7%]")
