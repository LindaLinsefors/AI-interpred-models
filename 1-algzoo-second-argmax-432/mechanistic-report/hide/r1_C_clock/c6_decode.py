"""C6. Decoding the age of the max from the angle of the clock coordinate (nearest reference phase).
Reference phases phi3(a), a = 1..9, were fitted on seeds 1100-1104 (c1_cells.py); evaluation on fresh data: 1M sequences, seeds 1600-1603."""
from cl import *
phi = np.load(HERE / "c1_laws.npz")["phi3"]                       # radians, index = age
ref = np.exp(1j * phi[1:10])                                      # ages 1..9
print("reference phases (deg):", np.round(np.degrees(phi[1:10]) % 360, 1))
zs, A1, TT, RR, MM, SS = [], [], [], [], [], []
for seed in range(1600, 1604):
    x, H, Z, st = gen(250_000, seed)
    zs.append(H[..., ORD] @ LC / K4); A1.append(st["a1"]); MM.append(st["M1"]); SS.append(pos(st["M2"]))
    TT.append(np.broadcast_to(np.arange(T), x.shape))
z, a1, M, S, tt = (np.concatenate(v).ravel() for v in (zs, A1, MM, SS, TT))
sel = (M > 0) & (a1 >= 1)
z, a1, M, S, tt = z[sel], a1[sel], M[sel], S[sel], tt[sel]
ratio = (M - S) / M
u = z / abs(z)
score = np.real(u[:, None] * np.conj(ref)[None, :])               # cos of the angle to each reference
dec = score.argmax(1) + 1
score_t = np.where(np.arange(1, 10)[None, :] <= tt[:, None], score, -np.inf)   # only ages possible at step t
dec_t = score_t.argmax(1) + 1
print(f"states: {z.size} (M > 0, a1 >= 1)")
print(f"overall accuracy: nearest phase among ages 1..9: {np.mean(dec == a1):.4f};  among ages <= t (uses t): {np.mean(dec_t == a1):.4f};  majority baseline (always a=1): {np.mean(a1 == 1):.4f}")
print(f"within one step (|decoded - true| <= 1, with 9 and 1 counted as neighbours): {np.mean((abs(dec - a1) <= 1) | (abs(dec - a1) == 8)):.4f}")
rb = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0000001]
print("\naccuracy by a1 (rows) and ratio (M - S+)/M (columns); last columns: all ratios, all ratios with ages <= t")
print("   a1 |       n | " + "  ".join(f"[{lo:.2f},{min(hi, 1):.2f})" for lo, hi in zip(rb[:-1], rb[1:])) + " |   all   | ages<=t")
for a in range(1, 10):
    s = a1 == a
    row = []
    for lo, hi in zip(rb[:-1], rb[1:]):
        q = s & (ratio >= lo) & (ratio < hi)
        row.append(f"   {np.mean(dec[q] == a):.3f}   ")
    print(f"   {a}  | {int(s.sum()):7d} | " + "".join(row) + f"|  {np.mean(dec[s] == a):.4f} | {np.mean(dec_t[s] == a):.4f}")
row = []
for lo, hi in zip(rb[:-1], rb[1:]):
    q = (ratio >= lo) & (ratio < hi)
    row.append(f"   {np.mean(dec[q] == a1[q]):.3f}   ")
print(f"  all | {z.size:7d} | " + "".join(row) + f"|  {np.mean(dec == a1):.4f} | {np.mean(dec_t == a1):.4f}")
print("  share of states per ratio bin: " + "  ".join(f"{np.mean((ratio >= lo) & (ratio < hi)):.3f}" for lo, hi in zip(rb[:-1], rb[1:])))
print("\nconfusion: share of states of true age a1 (rows) decoded as age (columns 1..9), nearest phase among 1..9")
for a in range(1, 10):
    s = a1 == a
    print(f"   {a}: " + "  ".join(f"{np.mean(dec[s] == d):.3f}" for d in range(1, 10)))
print("\naccuracy by step t (ages 1..9 | ages <= t | majority baseline at that t):")
for t in range(1, 10):
    s = tt == t
    maj = np.bincount(a1[s], minlength=10).max() / s.sum()
    print(f"   t={t}: {np.mean(dec[s] == a1[s]):.4f} | {np.mean(dec_t[s] == a1[s]):.4f} | {maj:.4f}   n={int(s.sum())}")
print("\ncheck with free-rotation references instead of fitted ones (phi(a) = 36.5 + (a - 2) 42.748 for a >= 2, phi(1) = 0):")
ref2 = np.exp(1j * np.radians(np.array([0.0] + [36.5 + (a - 2) * np.degrees(OMEGA) for a in range(2, 10)])))
dec2 = np.real(u[:, None] * np.conj(ref2)[None, :]).argmax(1) + 1
print(f"   accuracy {np.mean(dec2 == a1):.4f}")
print("check that could fail: decoding with the phases reversed (phi -> -phi) gives", round(float(np.mean((np.real(u[:, None] * ref[None, :]).argmax(1) + 1) == a1)), 4))
