"""S03. How do the margins scale with the gaps between the top inputs?  [FIT]  seed 1030, n = 400k N(0,1)."""
from src import *

x = sample(400_000, 1030)
H = run(x); L = logits(H)
order = np.argsort(-x, axis=1)
p1, p2, p3, p4 = order[:, 0], order[:, 1], order[:, 2], order[:, 3]
r = np.arange(len(x))
M, S, T, Q = x[r, p1], x[r, p2], x[r, p3], x[r, p4]
m21 = L[r, p2] - L[r, p1]; m23 = L[r, p2] - L[r, p3]; m34 = L[r, p3] - L[r, p4]
print(f"means: M {M.mean():.3f}, S {S.mean():.3f}, T {T.mean():.3f}; M-S {np.mean(M-S):.3f}, S-T {np.mean(S-T):.3f}")
print(f"mean margins: p2-p1 {m21.mean():.2f}, p2-p3 {m23.mean():.2f}, p3-p4 {m34.mean():.2f}")


def fit(y, X, names, mask=None, label=""):
    if mask is not None:
        y, X = y[mask], X[mask]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  {label:38s} n={len(y):6d}  " + "  ".join(f"{n}: {b:7.2f}" for n, b in zip(names, beta)) + f"   R^2 = {r2:.3f}")
    return beta, r2


one = np.ones(len(x))
RM, MR = p2 < p1, p2 > p1
print("\nmargin p2-p3 ~ a (S-T) [+ b (M-S) + c]")
for nm, mk in (("all", one > 0), ("RM", RM), ("MR", MR), ("S>0,T>0", (T > 0)), ("p2,p3 both <= 7, p1 <= 7", (p2 <= 7) & (p3 <= 7) & (p1 <= 7))):
    fit(m23, np.stack([S - T], 1), ["S-T"], mk, nm + " (no intercept)")
    fit(m23, np.stack([S - T, M - S, one], 1), ["S-T", "M-S", "1"], mk, nm)
print("\nmargin p2-p1 ~ a (M-S) [+ b (S-T) + c]")
for nm, mk in (("all", one > 0), ("RM", RM), ("MR", MR), ("p1,p2 <= 7", (p2 <= 7) & (p1 <= 7))):
    fit(m21, np.stack([M - S], 1), ["M-S"], mk, nm + " (no intercept)")
    fit(m21, np.stack([M - S, S - T, one], 1), ["M-S", "S-T", "1"], mk, nm)
print("\nby age of the positions: margin p2-p3 ~ a (S-T), no intercept; rows a2 = 9-p2, columns a3 = 9-p3 (slope / R^2)")
print("        " + "".join(f"   a3={a}     " for a in range(10)))
for a2 in range(10):
    row = ""
    for a3 in range(10):
        mk = (9 - p2 == a2) & (9 - p3 == a3)
        if a2 == a3 or mk.sum() < 200:
            row += "      --     "; continue
        g = (S - T)[mk]; y = m23[mk]
        b = (g @ y) / (g @ g); r2 = 1 - ((y - b * g) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        row += f" {b:5.1f}/{r2:5.2f} "
    print(f"a2={a2}   {row}")
print("\nby age: margin p2-p1 ~ a (M-S), no intercept; rows a2 = 9-p2, columns a1 = 9-p1 (slope / R^2)")
print("        " + "".join(f"   a1={a}     " for a in range(10)))
for a2 in range(10):
    row = ""
    for a1 in range(10):
        mk = (9 - p2 == a2) & (9 - p1 == a1)
        if a2 == a1 or mk.sum() < 200:
            row += "      --     "; continue
        g = (M - S)[mk]; y = m21[mk]
        b = (g @ y) / (g @ g); r2 = 1 - ((y - b * g) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        row += f" {b:5.1f}/{r2:5.2f} "
    print(f"a2={a2}   {row}")
