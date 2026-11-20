"""H1 extras: (1) does S+ = max(S_old, 0) extend the threshold laws to S_old <= 0?  (2) intervention on unit-4 misfires;
(3) check of F6: is the clock radius reduced by a runner-up that arrives late (max age >= 5)?"""
from hcommon import *
from h4_intervention import run_force

# ---------- 1. S_old <= 0: thresholds in units of M (S+ = 0) ----------
x = sample(400_000, 1025)
e = Ev(x)
print("=== 1. rho_j with S+ = max(S_old, 0): median by age of max, steps with M_old > 0.3 ===")
print("   a  | S_old > 0 (rho = (theta - S)/(M - S)), gap/M > 0.5      | S_old <= 0 or absent (rho = theta / M)")
print("      |  unit 1   unit 6   unit 7   unit 8   unit 4            |  unit 1   unit 6   unit 7   unit 8   unit 4      n")
for a in range(1, 9):
    s1 = (e.tt >= 2) & (e.a1 == a) & (e.So > 0) & ((e.Mo - e.So) / e.Mo > 0.5) & (e.Mo > 0.3)
    s0 = (e.tt >= 1) & (e.a1 == a) & ~(e.So > 0) & (e.Mo > 0.3)
    r1 = [np.median((e.th[j][s1] - e.So[s1]) / (e.Mo[s1] - e.So[s1])) for j in COMP + [4]]
    r0 = [np.median(e.th[j][s0] / e.Mo[s0]) for j in COMP + [4]] if s0.sum() > 300 else [np.nan] * 5
    print(f"   {a}  | " + "  ".join(f"{v:7.3f}" for v in r1) + "            | " + "  ".join(f"{v:7.3f}" for v in r0) + f"   {s0.sum()}")

# ---------- 2. misfire intervention ----------
print("\n=== 2. intervention: force unit 4 off at every step where x_t is NOT a new prefix maximum (t >= 1) ===")
for dist in ("normal", "uniform"):
    tot = np.zeros(6); nseq = 0
    for seed in (1026, 1027, 1028):
        x = sample(400_000, seed, dist)
        y = second_argmax(x)
        H = run(x)
        st = running_stats(x)
        Mo = np.concatenate([np.full((len(x), 1), -np.inf), st["M1"][:, :-1]], 1)
        notmax = x < Mo
        mis = notmax & (H[:, :, 4] > 0)
        Hf = run_force(x, notmax, unit=4)
        a0, af = logits(H).argmax(1), logits(Hf).argmax(1)
        tot += [(a0 == y).sum(), (af == y).sum(), (a0 == af).sum(), ((a0 == y) & (af != y)).sum(), ((a0 != y) & (af == y)).sum(), mis.any(1).sum()]
        nseq += len(x)
    print(f"  {dist}: n = {nseq}; sequences with at least one misfire: {tot[5] / nseq:.4f}")
    print(f"     accuracy {tot[0] / nseq:.4f} -> {tot[1] / nseq:.4f};  agreement {tot[2] / nseq:.4f};  broke {tot[3] / nseq:.4f}  fixed {tot[4] / nseq:.4f}")

# ---------- 3. F6 check ----------
print("\n=== 3. clock coordinate at the final step when the runner-up arrived after the max (a_ev = age of the max at that moment) ===")
print("    z/K rotated by -phase of its M-coefficient ~ bM M + bS S + bT T+ ; F6 predicts bS = -bM (radius ~ M - S)")
acc = {}
for seed in (1029, 1030, 1031):
    x = sample(400_000, seed)
    H = run(x); st = running_stats(x)
    p1, p2 = st["p1"][:, -1], st["p2"][:, -1]
    sel = (p2 > p1) & (st["M2"][:, -1] > 0)
    for k, v in dict(a_ev=(p2 - p1)[sel], a1=(9 - p1)[sel], M=st["M1"][sel, -1], S=st["M2"][sel, -1], T=np.maximum(st["M3"][sel, -1], 0), z=coords(H[sel, -1])["clock"] / K).items():
        acc.setdefault(k, []).append(v)
d = {k: np.concatenate(v) for k, v in acc.items()}
print("   a_ev | a1 (age of max at the end) | bM (amp@phase) | bS relative to bM: Re(bS/bM), Im(bS/bM) | R2 | n")
for a_ev in range(1, 8):
    for a1 in range(a_ev + 2, 10):
        s = (d["a_ev"] == a_ev) & (d["a1"] == a1)
        if s.sum() < 3000: continue
        X = np.column_stack([d["M"][s], d["S"][s], d["T"][s]]).astype(complex)
        coef, *_ = np.linalg.lstsq(X, d["z"][s], rcond=None)
        res = d["z"][s] - X @ coef
        R2 = 1 - np.sum(np.abs(res) ** 2) / np.sum(np.abs(d["z"][s]) ** 2)
        ratio = coef[1] / coef[0]
        if a1 in (a_ev + 2, 9):
            print(f"    {a_ev}   |  {a1}  | {abs(coef[0]):.3f}@{np.degrees(np.angle(coef[0])):6.1f} | {ratio.real:6.3f} {ratio.imag:6.3f} | {R2:.4f} | {s.sum()}")
