"""Code review of e23_core_alone.py (claim K5).

e23 scores the six-unit simulation by the centred R2 of the ACTIVATIONS over all steps.  The comparator
activations are g_j [theta_j - x_t]_+ and unit 4 is w_4 [x_t - theta_4]_+ : most of their variance is the
current input x_t, which the simulation is given.  So how high is the same R2 for models that know
nothing about the recurrence?  And how good is the simulation on the thing it is meant to model, the
thresholds theta_j(t) = -(W h_{t-1})_j / w_j ?  (e23's docstring announces a threshold comparison; the
code has none.)

Baselines:  N0  no recurrence at all:            h_j = ReLU(w_j x_t)
            N1  oracle running max as threshold: h_j = ReLU(w_j (x_t - c_j M+_{t-1})) with the best constant c_j
                (and h_2 = c M+_{t-1})
Same sample as e23 (seed 2323).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from modes import RIGHT

core = [2, 4, 1, 6, 7, 8]
x = sample(300_000, 2323)
H, Z = run(x, return_pre=True)
st = running_stats(x)
ratio = np.zeros(16)
for u in WAVE:
    ratio[u] = RIGHT["level"][ORD.index(u)].real / RIGHT["level"][ORD.index(2)].real
fb = np.array([sum(W[j, u] * ratio[u] for u in WAVE if u != 2) for j in core])


def sim(variant):
    Wc = W[np.ix_(core, core)].copy()
    if variant == "B":
        Wc[:, core.index(2)] += fb
    if variant == "N0":
        Wc[:] = 0
    h = np.zeros((len(x), len(core)))
    Hc = np.empty((len(x), T, len(core))); Rc = np.empty_like(Hc)
    for t in range(T):
        r = h @ Wc.T                               # recurrent input = -w_j * theta_j
        h = np.maximum(x[:, t, None] * w[core] + r, 0)
        Hc[:, t], Rc[:, t] = h, r
    return Hc, Rc


def r2(a, b):
    a, b = a.ravel(), b.ravel()
    return 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()


Rreal = Z - x[:, :, None] * w                      # (W h_{t-1}) for every unit
theta_real = -Rreal[:, :, core] / w[core]          # threshold form for every core unit (for unit 2 this is just a scaled recurrent input)
Mprev = np.concatenate([np.zeros((len(x), 1)), np.maximum(st["M1"][:, :-1], 0)], 1)     # max(0, x_0..x_{t-1})

res = {}
for variant in ("A", "B", "N0"):
    Hc, Rc = sim(variant)
    res[variant] = (Hc, -Rc / w[core])

# N1: oracle running max, one constant per unit fitted by least squares on the real thresholds
HN1 = np.empty_like(res["A"][0]); TN1 = np.empty_like(HN1)
cN1 = []
for k, u in enumerate(core):
    th = theta_real[:, :, k]
    c = (th * Mprev).sum() / (Mprev ** 2).sum()
    cN1.append(c)
    TN1[:, :, k] = c * Mprev
    HN1[:, :, k] = np.maximum(w[u] * (x - c * Mprev), 0)
res["N1"] = (HN1, TN1)
print("N1 constants c_j (theta_j ~ c_j * max(0, x_0..x_{t-1})) for units", core, ":", np.round(cN1, 3))

print("\ncentred R2 of ACTIVATIONS over all steps (e23's measure); units", core)
for v, lab in (("A", "A  core only (e23)            "), ("B", "B  core + level feedback (e23)"), ("N0", "N0 no recurrence at all       "), ("N1", "N1 threshold = c_j * running max")):
    print(f"   {lab}: " + "  ".join(f"{r2(H[:, :, u], res[v][0][:, :, k]):.4f}" for k, u in enumerate(core)))

print("\ncentred R2 of activations, steps t >= 5 only:")
for v in ("A", "B", "N0", "N1"):
    print(f"   {v:2s}: " + "  ".join(f"{r2(H[:, 5:, u], res[v][0][:, 5:, k]):.4f}" for k, u in enumerate(core)))

print("\ncentred R2 of THRESHOLDS theta_j(t) = -(W h_{t-1})_j / w_j, steps t = 1..9, comparators and unit 4 (units 4, 1, 6, 7, 8):")
for v in ("A", "B", "N1"):
    print(f"   {v:2s}: " + "  ".join(f"{r2(theta_real[:, 1:, k], res[v][1][:, 1:, k]):.4f}" for k, u in enumerate(core) if u != 2))
print("   same, only states whose running max M_{t-1} > 0.3 (thresholds matter there), in units of M_{t-1}; also rms error / M:")
ok = st["M1"][:, :-1] > 0.3
Mm = st["M1"][:, :-1]
for v in ("A", "B", "N1"):
    row = []
    for k, u in enumerate(core):
        if u == 2:
            continue
        a = (theta_real[:, 1:, k] / Mm)[ok]; b = (res[v][1][:, 1:, k] / Mm)[ok]
        row.append(f"{1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum():7.4f} (rms {np.sqrt(np.mean((a - b) ** 2)):.3f}, sd of true {a.std():.3f})")
    print(f"   {v:2s}: " + "  ".join(row))

print("\nthe clock inside the simulated core: z_c = l . theta (l = left eigenvector of Bc, as e24), real vs simulated, states with M_{t-1} > 0.3, t = 2..9")
g = -w[COMP]
Bc = W[np.ix_(COMP, COMP)] * g[None, :] / g[:, None]
lam, V = np.linalg.eig(Bc)
k = int(np.argmax(lam.imag))
l = np.linalg.inv(V)[k]
l = l / (l @ (W[COMP, 4] / g * w[4]))
ci = [core.index(j) for j in COMP]
zr = theta_real[:, :, ci] @ l
ok2 = np.zeros_like(zr, bool); ok2[:, 2:] = st["M1"][:, 1:-1] > 0.3
for v in ("A", "B", "N1"):
    zs = res[v][1][:, :, ci] @ l
    a, b = zr[ok2], zs[ok2]
    print(f"   {v:2s}: complex R2 (centred) = {1 - np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(a - a.mean()) ** 2):.4f}   "
          f"uncentred = {1 - np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(a) ** 2):.4f}   median |z_sim|/|z_real| = {np.median(np.abs(b) / np.abs(a)):.3f}")
