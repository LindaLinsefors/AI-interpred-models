"""A1 + A4: P(gate on) per unit and step; resting-pattern share; top full patterns."""
from cc import *

on = np.zeros((T, N)); n = 0
pid_all, cls_all = [], []
for seed, x, H, Z in batches():
    # check my running top-2 against common.running_stats on the first batch
    ev = events(x)
    if seed == SEEDS[0]:
        rs = running_stats(x[:50000])
        assert np.array_equal(rs["M1"], ev["M"][:50000]) and np.array_equal(rs["M2"], ev["S"][:50000])
        assert np.array_equal(rs["p1"], ev["pM"][:50000]) and np.array_equal(rs["p2"], ev["pS"][:50000])
        print("top2 == running_stats: ok")
    G = Z > 0
    assert np.array_equal(G, H > 0)
    on += G.sum(0); n += len(x)
    pid_all.append(pattern_id(G).astype(np.int32)); cls_all.append(ev["cls"])
pid = np.concatenate(pid_all); cls = np.concatenate(cls_all)
print("n sequences", n)
P = on / n
print("\nA1  P(gate on), rows = unit, columns = t = 0..9, last = all steps")
rows = []
for i in range(N):
    grp = "comp" if i in COMP else "pulse" if i in PULSE else "wave"
    rows.append([i, grp] + [f"{P[t, i]:.4f}" for t in range(T)] + [f"{P[:, i].mean():.4f}"])
print(md_table(["unit", "group"] + [f"t={t}" for t in range(T)] + ["all"], rows))

print("\nA4  resting pattern share per t and overall")
rest = pid == RESTING
print(md_table([f"t={t}" for t in range(T)] + ["all", "t>=1", "t>=2"],
               [[f"{rest[:, t].mean():.4f}" for t in range(T)] +
                [f"{rest.mean():.4f}", f"{rest[:, 1:].mean():.4f}", f"{rest[:, 2:].mean():.4f}"]]))
print("distinct patterns:", len(np.unique(pid)), " distinct at t>=2:", len(np.unique(pid[:, 2:])))
u, c = np.unique(pid, return_counts=True)
o = np.argsort(-c)
print("\ntop 25 patterns overall")
rows = []
for k in o[:25]:
    m = pid == u[k]
    per_t = m.mean(0)
    cl = [f"{(m & (cls == q)).sum() / m.sum():.2f}" for q in range(4)]
    rows.append([u[k], pstr(int(u[k])), f"{c[k] / pid.size:.4f}", f"{np.cumsum(c[o])[list(o).index(k)] / pid.size:.4f}"] + cl
                + [" ".join(f"{v:.2f}" for v in per_t)])
print(md_table(["id", "pattern", "freq", "cum", "F", "R", "N", "L", "freq per t=0..9"], rows))
