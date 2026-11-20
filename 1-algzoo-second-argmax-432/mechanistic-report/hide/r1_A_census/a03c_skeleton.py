"""A3 extra [INTERVENTION]: are the gates at step t determined by the two largest inputs so far?

Skeleton A: the prefix x_0..x_t with every input except the current largest and second largest (if positive) set to 0.
Skeleton B: as A, but the two most recent inputs x_(t-1), x_t are also kept at their actual values
            (comparators and unit 4 compare x_t, units 5 and 9 compare x_(t-1); see A5).
Agreement of each unit's gate at step t between the real run and the skeleton run, on steps t >= 1 with M_t > 0.
Also (a1, a2) tables of P(off) for wave units 3, 10, 11, 15 (a1, a2 = ages of the largest and second largest so far; S_t > 0).
"""
from cc import *
from a02_events import eff_class

agree = {v: [] for v in "AB"}
cls = []
tab = {i: np.zeros((10, 10, 2)) for i in (3, 10, 11, 15)}
for seed, x, H, Z in batches():
    ev = events(x); e = eff_class(x, ev)
    B = len(x); rows = np.arange(B); tt = np.broadcast_to(np.arange(T), (B, T))
    M, S = ev["M"], ev["S"]
    G = Z > 0
    reg = (tt >= 1) & (M > 0)
    for v in "AB":
        Gs = np.zeros((B, T, N), bool)
        for t in range(T):
            xs = np.zeros((B, t + 1))
            if v == "B":
                xs[:, max(t - 1, 0):] = x[:, max(t - 1, 0): t + 1]
            xs[rows, ev["pM"][:, t]] = np.where(M[:, t] > 0, M[:, t], xs[rows, ev["pM"][:, t]])
            has2 = (ev["pS"][:, t] >= 0) & (S[:, t] > 0)
            xs[rows[has2], ev["pS"][has2, t]] = S[has2, t]
            Gs[:, t] = run(xs, return_pre=True)[1][:, t] > 0
        agree[v].append((Gs == G)[reg])
    cls.append(e[reg])
    ok = ev["S"] > 0
    for i in tab:
        off = ~G[:, :, i]
        np.add.at(tab[i], (ev["a1"][ok], ev["a2"][ok], off[ok].astype(int)), 1)
cls = np.concatenate(cls)
print("## top-two skeleton: share of steps where the gate of each unit equals the real one (steps t >= 1 with M_t > 0, n = %d)\n" % len(cls))
rows = []
for v, name in (("A", "A: only the top two"), ("B", "B: top two + x_(t-1), x_t")):
    a = np.concatenate(agree[v])
    for label, m in (("all", np.ones(len(cls), bool)), ("R* at t", cls == 0), ("N* at t", cls == 1), ("L* at t", cls == 2)):
        rows.append([name, label, int(m.sum())] + [f"{a[m, i].mean():.4f}" for i in range(N)] + [f"{a[m].all(1).mean():.4f}"])
print(md_table(["skeleton", "steps", "n"] + [f"u{i}" for i in range(N)] + ["all 16"], rows))

print("\n## P(unit off | a1, a2), states with S_t > 0 (two positive inputs so far); rows a1 = age of the max, columns a2 = age of the runner-up; '.' if n < 1000\n")
for i in tab:
    print(f"\nunit {i}\n")
    rows = []
    for a in range(10):
        r = [f"a1={a}"]
        for b in range(10):
            n = tab[i][a, b].sum()
            r.append(f"{tab[i][a, b, 1] / n:.2f}" if n >= 1000 else ".")
        rows.append(r)
    print(md_table([""] + [f"a2={b}" for b in range(10)], rows))
