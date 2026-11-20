"""V6: the cosine law against alternatives on three versions of the pooled c1(Delta) table (natural weights,
seeds 5000..5009): all D1 records (the lead's table), aS >= 2 only, aS = 1 only."""
from v6_data import *
from scipy.optimize import least_squares

out = open(HERE / "a12_law_tables.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


e = dict(np.load(HERE / "d1_events.npz"))
y = e["yk"][:, 0]; gap = e["M0"] - e["S0"]; M0 = e["M0"]
D = np.abs(e["pM"] - e["pS"]); aS = e["t"] - e["pS"]
for nm, fam, dmax in (("all D1 records, Delta 1..7 (lead's table)", np.ones(len(y), bool), 7), ("all D1 records, Delta 1..8", np.ones(len(y), bool), 8),
                      ("aS >= 2, Delta 1..7", aS >= 2, 7), ("aS = 1, Delta 1..8", aS == 1, 8)):
    d = np.arange(1, dmax + 1).astype(float)
    c = np.array([np.linalg.lstsq(np.column_stack([gap[m], M0[m]]), y[m], rcond=None)[0][0] for dd in d for m in [fam & (D == dd)]])
    sst = ((c - c.mean()) ** 2).sum()
    f = 1 - np.cos(OMEGA * d); K = f @ c / (f @ f)
    fr = least_squares(lambda p: p[0] * (1 - np.cos(np.radians(p[1]) * d)) - c, [50, 42.75])
    a = d @ c / (d @ d)
    X = np.column_stack([d, d ** 2]); b = np.linalg.lstsq(X, c, rcond=None)[0]
    X1 = np.column_stack([d, np.ones(len(d))]); b1 = np.linalg.lstsq(X1, c, rcond=None)[0]
    P(f"{nm}: c1 = " + ", ".join(f"{v:.1f}" for v in c))
    P("   c1/(1-cos(42.75 Delta)) = " + ", ".join(f"{v:.1f}" for v in c / f))
    P(f"   R2 over the table: cosine 42.75 (K = {K:.1f}): {1 - ((c - K * f) ** 2).sum() / sst:.3f} | cosine free w (K = {fr.x[0]:.1f}, w = {fr.x[1]:.2f}): "
      f"{1 - (fr.fun ** 2).sum() / sst:.3f} | linear through 0: {1 - ((c - a * d) ** 2).sum() / sst:.3f} | affine: {1 - ((c - X1 @ b1) ** 2).sum() / sst:.3f} | "
      f"parabola through 0: {1 - ((c - X @ b) ** 2).sum() / sst:.3f} | constant: 0.000")
out.close()
