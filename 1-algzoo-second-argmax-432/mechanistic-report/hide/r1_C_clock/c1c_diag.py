"""C1c (diagnostic). Distribution of the max's own clock trace zeta_M inside cells; which events change it."""
from cl import *
x = sample(40_000, 1710)
Hh, J = traces(x)
Zp = run(x, return_pre=True)[1]
st = running_stats(x)
zeta = J[..., ORD] @ LC / K4
del J
bi = np.arange(x.shape[0])
G = Zp > 0
for (a, b) in ((2, 3), (5, 2), (5, 7), (3, 1), (6, 0)):
    for t in range(max(a, b), 10):
        pass
    vals, flags = [], []
    for t in range(max(a, b), 10):
        q = (st["M1"][:, t] > 0) & (st["a1"][:, t] == a) & (st["a2"][:, t] == b)
        zm = zeta[bi, t, st["p1"][:, t]][q]
        # was unit 4 on at any step after the max arrived (a 'false record' on the max's watch)?
        f4 = np.zeros(q.sum(), bool); nclip = np.zeros(q.sum())
        for d in range(0, a):
            f4 |= G[q, t - d, 4]
            nclip += (~G[q, t - d][:, COMP]).sum(1)
        vals.append(zm); flags.append(np.column_stack([f4, nclip]))
    zm = np.concatenate(vals); fl = np.concatenate(flags)
    ref = np.exp(1j * np.angle(zm.mean()))
    r = (zm / ref)
    print(f"cell (a1={a}, a2={b}): n={zm.size}  |zeta_M| percentiles 1/10/25/50/75/90/99: " + " ".join(f"{v:.2f}" for v in np.percentile(abs(zm), [1, 10, 25, 50, 75, 90, 99]))
          + f"   phase-about-mean percentiles 10/50/90: " + " ".join(f"{v:+.0f}" for v in np.percentile(np.degrees(np.angle(r)), [10, 50, 90])))
    for lab, m_ in (("unit 4 fired after the max arrived", fl[:, 0] == 1), ("no unit-4 firing, no comparator clipped since", (fl[:, 0] == 0) & (fl[:, 1] == 0)),
                    ("no unit-4 firing, some comparator clipped since", (fl[:, 0] == 0) & (fl[:, 1] > 0))):
        if m_.sum() > 20:
            print(f"      {lab:48s}: share {m_.mean():.3f}  median |zeta_M| {np.median(abs(zm[m_])):.3f}  IQR [{np.percentile(abs(zm[m_]), 25):.3f}, {np.percentile(abs(zm[m_]), 75):.3f}]  median phase {np.degrees(np.angle(np.median(zm[m_].real) + 1j * np.median(zm[m_].imag))):+.1f}")
