"""I2: 'the same intervention on another mode of similar size'.  Seeds 5960.. ; rng 5998.

Removing the 83-degree mode changes the state 3.4 times less than removing the clock, so f = 0 on it is not a
size-matched comparison.  Here every perturbation has, state by state, the SAME norm as the clock removal:
  delta = 2 Re(c e^{i phi} RIGHT[mode]),  phi uniform,  c chosen so that |delta| = |2 Re(z_clock RIGHT[clock])|.
"""
from v3lib import *

out = Tee("a8_matched_modes.txt")
rng = np.random.default_rng(5998)
x, tr, p0, q0, M0, S0, nb = gather_I2(250_000, 5960)
n = len(x); r = np.arange(n); tint = tr - 1
out(f"I2 pool: seeds 5960..{5960 + nb - 1}, n = {n}")
hpre = run(x)[r, tint]
zc = hpre[:, ORD_A] @ LEFT["clock"]
dclock = 2 * np.real(np.multiply.outer(zc, RIGHT["clock"]))
target = np.linalg.norm(dclock, axis=1)


def add_fn(delta, clip):
    def fn(h, rows):
        h = h.copy()
        h[:, ORD_A] += delta[rows]
        return np.maximum(h, 0.0) if clip else h
    return fn


def report(label, delta):
    res = []
    for clip in (False, True):
        a = answer(run_iv(x, tint, add_fn(delta, clip)))
        res.append(f"P(p0) = {np.mean(a == p0):.4f}  P(t_r) = {np.mean(a == tr):.4f}  P(q0) = {np.mean(a == q0):.4f}")
    out(f"  {label:44s} raw: {res[0]}  | clipped: {res[1]}")


report("minus the clock part (f = 0, lead)", -dclock)
for mode in ("clock", "h2", "h3", "short"):
    phi = rng.uniform(0, 2 * np.pi, n)
    v = 2 * np.real(np.multiply.outer(np.exp(1j * phi), RIGHT[mode]))
    report(f"random phase in the {mode} plane, same norm", v * (target / np.linalg.norm(v, axis=1))[:, None])
for mode in ("level", "runner", "alt", "trans"):
    sgn = rng.choice([-1.0, 1.0], n)
    v = np.real(np.multiply.outer(sgn, RIGHT[mode]))
    report(f"+- the {mode} direction, same norm", v * (target / np.linalg.norm(v, axis=1))[:, None])
g = rng.standard_normal((n, 13))
report("random Gaussian direction, same norm", g * (target / np.linalg.norm(g, axis=1))[:, None])
# the removal of each mode scaled up to the clock's norm (sign kept: this is 'over-removal' for the small modes)
for mode in ("h2", "h3", "short", "runner"):
    z = hpre[:, ORD_A] @ LEFT[mode]
    d = -(2 if IS_COMPLEX[mode] else 1) * np.real(np.multiply.outer(z, RIGHT[mode]))
    fac = target / np.linalg.norm(d, axis=1)
    report(f"removal direction of {mode}, scaled to the clock's norm (median factor {np.median(fac):.2f})", d * fac[:, None])
