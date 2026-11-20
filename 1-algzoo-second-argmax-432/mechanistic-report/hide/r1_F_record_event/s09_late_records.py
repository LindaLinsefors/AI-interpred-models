"""s09 (task F6). Records at the last step (s = 9) and at the second-to-last step (s = 8), compared with earlier ones.
N(0,1), seeds 1070, 1071 (400k each).
'relabel' sequences: x_s is the overall max (record at s, old max M0 > 0) and the old max p0 is the true runner-up.
"""
from fcommon import *

acc = {}
def put(k, v):
    acc.setdefault(k, []).append(v)

for seed in (1070, 1071):
    x = sample(400_000, seed)
    H, Z = run(x, return_pre=True)
    st = running_stats(x)
    lg = H[:, 9] @ U.T
    ans = lg.argmax(1)
    order = np.argsort(-x, axis=1)
    put("all_pmax", order[:, 0]); put("all_ok", ans == order[:, 1]); put("all_ans_max", ans == order[:, 0])
    put("all_ans_3rd", ans == order[:, 2])
    for s in range(2, 10):
        rec, q = records(x, s, st)
        ok = rec & (order[:, 0] == s) & (order[:, 1] == q["p0"])
        i = np.flatnonzero(ok)
        n = len(i); ar = np.arange(n)
        p0 = q["p0"][i]
        put("s", np.full(n, s)); put("g", q["g"][i]); put("ok", ans[i] == p0); put("a_max", ans[i] == s)
        put("a_3rd", ans[i] == order[i, 2]); put("a_nb", np.abs(ans[i] - p0) == 1)
        put("relgap", (q["Mn"][i] - q["M0"][i]) / q["M0"][i]); put("oldgap", (q["M0"][i] - q["S0p"][i]) / q["M0"][i])
        put("h4", H[i, s, 4]); put("Mn", q["Mn"][i]); put("M0", q["M0"][i]); put("S0p", q["S0p"][i])
        put("m_p0_s", lg[i][ar, p0] - lg[i, s])
        if s == 9:
            z9 = Z[i, 9]
            put("s9_compoff", ~(z9[:, COMP] > 0).any(1))
            # exact readout pieces at the last step
            put("s9_u4", (U[p0, 4] - U[9, 4]) * H[i, 9, 4])                              # margin p0 - 9 from unit 4
            put("s9_wave", np.einsum("bu,bu->b", U[p0][:, WAVE] - U[9][WAVE], H[i, 9][:, WAVE]))
            put("s9_p59", np.einsum("bu,bu->b", U[p0][:, [5, 9]] - U[9][[5, 9]], H[i, 9][:, [5, 9]]))
            put("s9_comp", np.einsum("bu,bu->b", U[p0][:, COMP] - U[9][COMP], H[i, 9][:, COMP]))
            # interventions at the last step
            h = H[i, 9].copy(); h[:, COMP] = np.where(z9[:, COMP] > 0, h[:, COMP], z9[:, COMP])   # comparators NOT clipped
            a1 = (h @ U.T).argmax(1)
            put("s9_lin_ok", a1 == p0); put("s9_lin_max", a1 == 9); put("s9_lin_3rd", a1 == order[i, 2])
            h = H[i, 9].copy(); h[:, 4] = 0
            a2 = (h @ U.T).argmax(1)
            put("s9_no4_ok", a2 == p0); put("s9_no4_max", a2 == 9)
            put("s9_g", q["g"][i])
        if s == 8:
            h = H[i, 9].copy()
            for nm, units in (("no5", [5]), ("no9", [9]), ("no59", [5, 9])):
                hh = h.copy(); hh[:, units] = 0
                a = (hh @ U.T).argmax(1)
                put(f"s8_{nm}_ok", a == p0); put(f"s8_{nm}_max", a == 8)
            put("s8_u5", U[8, 5] * h[:, 5]); put("s8_u9", U[8, 9] * h[:, 9]); put("s8_h5", h[:, 5]); put("s8_h9", h[:, 9])
            put("s8_on", (h[:, [5, 9]] > 0))
R = {k: np.concatenate(v) for k, v in acc.items()}

print("=== all sequences: accuracy by the position of the overall max ===")
print("  pos of max:        " + " ".join(f"{p:7d}" for p in range(10)))
for lab, k in (("accuracy", "all_ok"), ("answers the max", "all_ans_max"), ("answers the 3rd", "all_ans_3rd")):
    print(f"  {lab:18s} " + " ".join(f"{R[k][R['all_pmax'] == p].mean():7.4f}" for p in range(10)))
print(f"  n per position ~ {int((R['all_pmax'] == 5).sum())}")

print("\n=== relabel sequences: P(answer = old max p0) by record step s and age g of the old max ===")
print("   s     n      all g |" + "".join(f"   g={g}  " for g in range(1, 8)) + "| errors: answers s (the max) | answers the 3rd | answers p0+-1")
for s in range(2, 10):
    m = R["s"] == s
    row = f"   {s} {m.sum():6d}   {R['ok'][m].mean():.4f} |"
    for g in range(1, 8):
        mg = m & (R["g"] == g)
        row += f"  {R['ok'][mg].mean():.3f} " if mg.sum() > 300 else "    .   "
    row += f"|   {R['a_max'][m].mean():.4f}   |   {R['a_3rd'][m].mean():.4f}   |   {R['a_nb'][m].mean():.4f}"
    print(row)

print("\n=== margin logit[p0] - logit[s] ~ a (Mn-M0) + b (M0-S0+) + c S0+ by record step s (no intercept) ===")
for s in range(2, 10):
    m = R["s"] == s
    b, r2 = ols(R["m_p0_s"][m], [R["Mn"][m] - R["M0"][m], R["M0"][m] - R["S0p"][m], R["S0p"][m]])
    print(f"   s={s}: {b[0]:6.2f} (Mn-M0) {b[1]:+6.2f} (M0-S0+) {b[2]:+6.2f} S0+   R2 = {r2:.3f}   P(margin < 0) = {(R['m_p0_s'][m] < 0).mean():.4f}")
print("  P(network answers the max s) by relative new gap (Mn-M0)/M0 quintile, for s = 9, 8 and s <= 7:")
for lab, m in (("s=9", R["s"] == 9), ("s=8", R["s"] == 8), ("s<=7", R["s"] <= 7)):
    qs = np.quantile(R["relgap"][m], [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(R["relgap"], qs)
    print(f"   {lab:5s}: " + "  ".join(f"q{k+1} (<{qs[k]:.2f}): {R['a_max'][m & (b == k)].mean():.3f}" if k < 4 else f"q5: {R['a_max'][m & (b == k)].mean():.3f}" for k in range(5)))
print("  P(network answers the 3rd) by relative old gap (M0-S0+)/M0 quintile:")
for lab, m in (("s=9", R["s"] == 9), ("s=8", R["s"] == 8), ("s<=7", R["s"] <= 7)):
    qs = np.quantile(R["oldgap"][m], [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(R["oldgap"], qs)
    print(f"   {lab:5s}: " + "  ".join(f"q{k+1} (<{qs[k]:.2f}): {R['a_3rd'][m & (b == k)].mean():.3f}" if k < 4 else f"q5: {R['a_3rd'][m & (b == k)].mean():.3f}" for k in range(5)))

print("\n=== s = 9 (record at the last step): what the readout sees ===")
n9 = len(R["s9_g"])
print(f"  n = {n9}; all four comparators clipped at step 9: {R['s9_compoff'].mean():.4f}")
print(f"  margin logit[p0] - logit[9] = unit 4 part + wave part + pulses(5,9) part + comparator part; means: "
      f"{R['s9_u4'].mean():.2f} + {R['s9_wave'].mean():.2f} + {R['s9_p59'].mean():.2f} + {R['s9_comp'].mean():.2f}")
m9 = R["s"] == 9
for lab, y in (("unit-4 part", R["s9_u4"]), ("wave part", R["s9_wave"]), ("pulses 5,9 part", R["s9_p59"])):
    b, r2 = ols(y, [R["Mn"][m9] - R["M0"][m9], R["M0"][m9] - R["S0p"][m9], R["S0p"][m9]])
    print(f"     {lab:16s} ~ {b[0]:7.2f} (Mn-M0) {b[1]:+7.2f} (M0-S0+) {b[2]:+7.2f} S0+   R2 = {r2:.3f}")
print("  (U[p,4] - U[9,4]) for p = 0..8:", np.round(U[:9, 4] - U[9, 4], 2), " times w4 =", np.round((U[:9, 4] - U[9, 4]) * w[4], 1))
print(f"  [INTERVENTION] comparators not clipped at step 9 (their negative pre-activations are read):  P(ans = p0) {R['s9_lin_ok'].mean():.4f}"
      f"  P(ans = 9) {R['s9_lin_max'].mean():.4f}  P(ans = 3rd) {R['s9_lin_3rd'].mean():.4f}     [unmodified: P(ans = p0) {R['ok'][m9].mean():.4f}]")
print(f"  [INTERVENTION] unit 4 set to 0 at step 9:  P(ans = p0) {R['s9_no4_ok'].mean():.4f}   P(ans = 9) {R['s9_no4_max'].mean():.4f}")
print("     by g: comparators not clipped -> P(ans = p0): " + "  ".join(f"g={g}: {R['s9_lin_ok'][R['s9_g'] == g].mean():.3f}" for g in range(1, 8)))

print("\n=== s = 8 (record at the second-to-last step): units 5 and 9 fire at step 9 and are read directly ===")
m8 = R["s"] == 8
print(f"  n = {m8.sum()}; unit 5 on at step 9: {R['s8_on'][:, 0].mean():.4f}; unit 9 on: {R['s8_on'][:, 1].mean():.4f};"
      f" mean h5(9) = {R['s8_h5'].mean():.2f}, mean h9(9) = {R['s8_h9'].mean():.2f}")
print(f"  direct readout at position 8: U[8,5] h5 = {R['s8_u5'].mean():.2f} (U[8,5] = {U[8,5]:.2f});  U[8,9] h9 = {R['s8_u9'].mean():.2f} (U[8,9] = {U[8,9]:.2f})")
print(f"  U[:,5] = {np.round(U[:, 5], 2)}\n  U[:,9] = {np.round(U[:, 9], 2)}")
for nm, lab in (("no5", "unit 5 removed at step 9"), ("no9", "unit 9 removed at step 9"), ("no59", "both removed")):
    print(f"  [INTERVENTION] {lab:26s}: P(ans = p0) {R[f's8_{nm}_ok'].mean():.4f}   P(ans = 8) {R[f's8_{nm}_max'].mean():.4f}     [unmodified: {R['ok'][m8].mean():.4f}, P(ans = 8) {R['a_max'][m8].mean():.4f}]")
