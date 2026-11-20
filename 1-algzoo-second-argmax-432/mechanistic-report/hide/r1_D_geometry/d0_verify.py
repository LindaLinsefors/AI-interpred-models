"""D0. Re-verify the lead's facts F1-F6 that arm D builds on.  python -B d0_verify.py"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

SEED = 1000
x = sample(200_000, SEED)
H, Z = run(x, return_pre=True)
st = running_stats(x)
print("accuracy N(0,1), seed", SEED, ":", (logits(H).argmax(1) == second_argmax(x)).mean())

# F1: h_t = sum_p J[t,p] x_p exactly
xs = x[:2000]
Hs, J = traces(xs)
print("F1 max |h_t - sum_p J x_p| =", np.abs(np.einsum("btpn,bp->btn", J, xs) - Hs).max())
# positive homogeneity
print("F1 homogeneity max |run(3x) - 3 run(x)| =", np.abs(run(3 * xs) - 3 * Hs).max())

# F3: eigenvalues
print("\nF3 eigenvalues of A = W[ORD, ORD]")
for n in NAMES:
    print(f"  {n:7s} |lam| = {abs(LAM[n]):.5f}  angle = {np.degrees(np.angle(LAM[n])):8.3f}   LEFT.KICK4 = {LEFT[n] @ KICK4:.6f}")
# eigen-equation residuals
for n in NAMES:
    print(f"  {n:7s} |A R - lam R| = {np.abs(A @ RIGHT[n] - LAM[n] * RIGHT[n]).max():.2e}   |L A - lam L| = {np.abs(LEFT[n] @ A - LAM[n] * LEFT[n]).max():.2e}")

# F4: input direction ~ echo eigenvector
wo = w[ORD]
e = RIGHT["echo"].real
print("\nF4 cos(w[ORD], echo eigenvector) =", wo @ e / np.linalg.norm(wo) / np.linalg.norm(e))
print("   A w[ORD] =", A @ wo, " max |.| =", np.abs(A @ wo).max())
print("   W[PULSE, ORD] w[ORD] =", W[np.ix_(PULSE, ORD)] @ wo)
print("   mode coordinates of w[ORD]:", {n: np.round(LEFT[n] @ wo, 4) for n in NAMES})

# F5: level eigenvector positive; level coordinate vs M
print("\nF5 level eigenvector (gauge kick=1):", RIGHT["level"].real)
print("   all positive:", (RIGHT["level"].real > 0).all())
m = coords(H)
M, S, a1 = st["M1"], st["M2"], st["a1"]
for a in range(10):
    s = (a1 == a) & (M > 0)
    y, X = m["level"][s], 10.157 * M[s]
    b = (X @ y) / (X @ X)
    print(f"   age {a}: level ~ {b:.3f} * 10.157 M   R2(uncentred) = {1 - ((y - b * X) ** 2).sum() / (y ** 2).sum():.4f}"
          f"   R2(centred) = {1 - (y - b * X).var() / y.var():.4f}   n = {s.sum()}")

# F6: clock coordinate vs (M - S+)
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)
print("\nF6 clock z / (10.157 (M - S+)): complex LS coefficient by age (M > 0)")
for a in range(1, 10):
    s = (a1 == a) & (M > 0)
    g = 10.157 * (M[s] - Sp[s])
    c = (g @ m["clock"][s]) / (g @ g)
    res = m["clock"][s] - c * g
    print(f"   age {a}: rho = {abs(c):.3f}  phi = {np.degrees(np.angle(c)) % 360:6.1f} deg   R2(uncentred) = {1 - (abs(res) ** 2).sum() / (abs(m['clock'][s]) ** 2).sum():.4f}")
print("   squared-norm share of LEFT on comparators:",
      {n: round(float((abs(LEFT[n][[ORD.index(j) for j in COMP]]) ** 2).sum() / (abs(LEFT[n]) ** 2).sum()), 3) for n in NAMES})
