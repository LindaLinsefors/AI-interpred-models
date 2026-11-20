"""I2: where does the promotion go?  Measured on the logits, not on the argmax.  Seeds 6020..

 bump(d)  = median over sequences of [s9(unperturbed) - s9(clock x 0)] at position p0 + d, in units of M0 - S0
 after a rotation by k:  delta(d) = s9(rotated) - s9(clock x 0)   -> is its maximum over old positions at d = -k?
"""
from v3lib import *

out = Tee("a11_bump.txt")
x, tr, p0, q0, M0, S0, nb = gather_I2(250_000, 6020)
n = len(x); r = np.arange(n); tint = tr - 1
out(f"I2 pool: seeds 6020..{6020 + nb - 1}, n = {n}")
s_base = run(x)[:, -1] @ U.T
s_zero = run_iv(x, tint, mode_fn("clock", f=0)) @ U.T
gap = (M0 - S0)[:, None]
pos = np.arange(T)[None, :]
d = pos - p0[:, None]                              # offset of every position from p0
old = pos < tr[:, None]                            # positions before the record


def profile(delta, label):
    row = []
    for dd in range(-4, 5):
        m = (d == dd) & old
        row.append(f"{np.median((delta / gap)[m]):+7.1f}" if m.sum() > 500 else "    -  ")
    out(f"  {label:34s} d = -4..+4: " + " ".join(row) + f"   | at t_r: {np.median((delta / gap)[r, tr]):+7.1f}")


out("\nmedian change of the final logits, in units of (M0 - S0), by offset d from the old max (old positions only):")
profile(s_base - s_zero, "unperturbed minus (clock x 0)")
profile(run_iv(x, tint, mode_fn("clock", f=0.5)) @ U.T - s_zero, "(clock x 0.5) minus (clock x 0)")
profile(run_iv(x, tint, mode_fn("clock", f=2.0, clip=True)) @ U.T - s_zero, "(clock x 2, clipped) minus (clock x 0)")
for k in (-2, -1, 1, 2):
    for clip in (False, True):
        s_rot = run_iv(x, tint, mode_fn("clock", k=k, clip=clip)) @ U.T
        delta = s_rot - s_zero
        profile(delta, f"(k = {k:+}{', clipped' if clip else ''}) minus (clock x 0)")
        dm = np.where(old, delta, -np.inf)
        am = dm.argmax(1) - p0
        tgt_ok = (p0 - k >= 0)                     # p0 - k always < t_r for k >= -2 because age >= 2
        out(f"      position of the largest increase among old positions, relative to p0 (target {-k:+d}): "
            + " ".join(f"{j:+d}: {np.mean(am[tgt_ok] == j):.3f}" for j in range(-3, 4))
            + f"   (target exists in {tgt_ok.mean():.3f}; hit rate given it exists {np.mean(am[tgt_ok] == -k):.3f})")
out("\nreference: position of the largest value of [unperturbed minus (clock x 0)] among old positions, relative to p0: "
    + " ".join(f"{j:+d}: {np.mean((np.where(old, s_base - s_zero, -np.inf).argmax(1) - p0) == j):.3f}" for j in range(-3, 4)))
