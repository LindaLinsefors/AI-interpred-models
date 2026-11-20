"""Claim I2 (clock before a record), own reproduction.  Seeds 5400.. (400k sequences per batch).

Selection: argmax of the ten inputs at t_r in [3, 8]; prefix x_0..x_{t_r-1} has max at p0 (age t_r-1-p0 >= 2,
value M0) and runner-up at q0 (value S0 > 0.2, M0 - S0 > 0.1); second argmax of the whole sequence is p0 and the
third largest of the whole sequence is at q0.
Intervention on h_{t_r-1}: z = LEFT[clock] . h[ORD] -> z';  h[ORD] += 2 Re((z'-z) RIGHT[clock]).
"""
from v3lib import *

out = Tee("i2_clock.txt")
x, tr, p0, q0, M0, S0, nb = gather_I2(250_000, 5400)
n = len(x)
out(f"seeds 5400..{5400 + nb - 1}; sequences {nb * 400_000}; selected {n} ({n / (nb * 400_000):.4f})")
out("t_r counts 3..8:", np.bincount(tr, minlength=10)[3:9])
out("age of old max (t_r-1-p0) counts 2..7:", np.bincount(tr - 1 - p0, minlength=8)[2:8])
tint = tr - 1
se = lambda p: np.sqrt(p * (1 - p) / n)


def dist(a):
    """Answer categories relative to p0, plus t_r and q0."""
    d = a - p0
    s = "  ".join(f"p0{j:+d}: {np.mean(d == j):.4f}" for j in range(-3, 4))
    return s + f"  | t_r: {np.mean(a == tr):.4f}  q0: {np.mean(a == q0):.4f}  other: {np.mean((a != p0) & (a != tr) & (a != q0)):.4f}"


a0 = answer(run(x)[:, -1])
out(f"unperturbed: P(ans = p0) = {np.mean(a0 == p0):.4f}")

out("\n--- z' = f z ---   (lead: P(p0) = 0.019, 0.098, 0.342, 0.889, 0.990, 0.893, 0.746; f=0: P(t_r)=0.775, P(q0)=0.164)")
for f in (0, 0.25, 0.5, 0.75, 1, 1.5, 2):
    h9, hiv = run_iv(x, tint, mode_fn("clock", f=f), return_state=True)
    a = answer(h9)
    out(f"f = {f:<5}: P(p0) = {np.mean(a == p0):.4f} (se {se(np.mean(a == p0)):.4f})   P(t_r) = {np.mean(a == tr):.4f}   "
        f"P(q0) = {np.mean(a == q0):.4f}   frac states with a negative entry = {np.mean((hiv < -1e-9).any(1)):.4f}")
    out("            " + dist(a))

out("\n--- z' = z exp(i k omega) ---")
out("lead: k=-1: P(p0+1)=0.474, P(p0+2)=0.154, P(p0)=0.093; k=-2: P(p0+2)=0.321, P(p0+1)=0.153, P(p0)=0.002;")
out("      k=+1: P(p0)=0.458, P(p0-1)=0.381, P(p0-2)=0.077; k=+2: P(p0-1)=0.298, P(p0-2)=0.249, P(p0)=0.071")
for k in (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3):
    h9, hiv = run_iv(x, tint, mode_fn("clock", k=k), return_state=True)
    a = answer(h9)
    d = a - p0
    out(f"k = {k:+}: mean(ans - p0) = {d.mean():+.3f}  P(ans < p0) = {np.mean(d < 0):.4f}  P(ans > p0) = {np.mean(d > 0):.4f}"
        f"   frac states with a negative entry = {np.mean((hiv < -1e-9).any(1)):.4f}")
    out("            " + dist(a))
