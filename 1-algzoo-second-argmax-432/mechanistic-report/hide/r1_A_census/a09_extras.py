"""Extras: (i) P(wave unit off) once a positive input has been seen; (ii) false records (unit 4 on without a new maximum)
and accuracy; (iii) expected class shares; (iv) accuracy by events near the end."""
from cc import *
from a02_events import eff_class

WV = WAVE
cnt = {k: np.zeros((T, len(WV), 2)) for k in ("pos1", "pos2")}
fr_n = np.zeros(T); r_n = np.zeros(T); n4 = np.zeros(T)
acc = {k: np.zeros(2) for k in ("all", "no false record", ">= 1 false record", "false record at t >= 7", "final S > 0", "final S < 0",
                               "N* at t=9", "N* at t=8", "R* at t=9", "R* at t=8")}
for seed, x, H, Z in batches():
    ev = events(x); e = eff_class(x, ev); G = Z > 0
    ok = logits(H).argmax(1) == second_argmax(x)
    for t in range(1, T):
        for name, m in (("pos1", ev["Mp"][:, t] > 0), ("pos2", ev["Sp"][:, t] > 0)):
            off = ~G[:, t][:, WV][m]
            cnt[name][t, :, 0] += off.sum(0); cnt[name][t, :, 1] += m.sum()
    fr = G[:, :, 4] & (e != 0)                          # unit 4 on although x_t is not a new positive maximum
    fr_n += fr.sum(0); n4 += G[:, :, 4].sum(0); r_n += (e == 0).sum(0)
    anyfr = fr.any(1)
    for name, m in (("all", np.ones(len(x), bool)), ("no false record", ~anyfr), (">= 1 false record", anyfr),
                    ("false record at t >= 7", fr[:, 7:].any(1)), ("final S > 0", ev["S"][:, 9] > 0), ("final S < 0", ev["S"][:, 9] < 0),
                    ("N* at t=9", e[:, 9] == 1), ("N* at t=8", e[:, 8] == 1), ("R* at t=9", e[:, 9] == 0), ("R* at t=8", e[:, 8] == 0)):
        acc[name] += [ok[m].sum(), m.sum()]

print("## (i) P(wave unit off at t) given that a positive input occurred before t (M_(t-1) > 0)\n")
rows = [[f"t={t}", int(cnt['pos1'][t, 0, 1])] + [f"{cnt['pos1'][t, k, 0] / cnt['pos1'][t, k, 1]:.4f}" for k in range(len(WV))] for t in range(1, T)]
tot = cnt["pos1"].sum(0)
rows.append(["all t>=1", int(tot[0, 1])] + [f"{tot[k, 0] / tot[k, 1]:.4f}" for k in range(len(WV))])
print(md_table(["", "n"] + [f"unit {i}" for i in WV], rows))
print("\ngiven two positive inputs before t (S_(t-1) > 0)\n")
rows = [[f"t={t}", int(cnt['pos2'][t, 0, 1])] + [f"{cnt['pos2'][t, k, 0] / cnt['pos2'][t, k, 1]:.4f}" for k in range(len(WV))] for t in range(2, T)]
tot = cnt["pos2"].sum(0)
rows.append(["all t>=2", int(tot[0, 1])] + [f"{tot[k, 0] / tot[k, 1]:.4f}" for k in range(len(WV))])
print(md_table(["", "n"] + [f"unit {i}" for i in WV], rows))

print("\n## (ii) false records: unit 4 on at t although x_t is not a new positive maximum\n")
print(md_table(["", ] + [f"t={t}" for t in range(T)] + ["all"],
               [["unit-4 firings"] + [int(v) for v in n4] + [int(n4.sum())],
                ["false records"] + [int(v) for v in fr_n] + [int(fr_n.sum())],
                ["share of firings"] + [f"{a / b:.4f}" for a, b in zip(fr_n, n4)] + [f"{fr_n.sum() / n4.sum():.4f}"]]))
print()
print(md_table(["sequences", "n", "accuracy"], [[k, int(v[1]), f"{v[0] / v[1]:.4f}"] for k, v in acc.items()]))
print("\n## (iii) expected class shares for iid continuous inputs: P(R at t) = P(N at t) = 1/(t+1);  mean over t = 1..9 =",
      f"{np.mean([1 / (t + 1) for t in range(1, 10)]):.5f}", " P(L) =", f"{1 - 2 * np.mean([1 / (t + 1) for t in range(1, 10)]):.5f}")
