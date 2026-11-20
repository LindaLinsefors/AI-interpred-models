"""S06. Re-verification of the lead's facts that this arm touches (F1-F4, F7-F9)."""
from src import *

x = sample(50_000, 1130)
H, J = traces(x)
print("F1: max |h_t - sum_p J[t,p] x_p| =", np.abs(np.einsum("btpi,bp->bti", J, x) - H).max())
Hh, Z = run(x, return_pre=True)
Hprev = np.concatenate([np.zeros((len(x), 1, 16)), Hh[:, :-1]], 1)
for j in COMP:
    th = -(Hprev @ W[j]) / w[j]
    print(f"F2: comparator {j}: gate == (x_t < theta) mismatches: {int(((Z[:, :, j] > 0) != (x < th)).sum())}")
th4 = -(Hprev @ W[4]) / w[4]
print(f"F2: unit 4: gate == (x_t > theta_4) mismatches: {int(((Z[:, :, 4] > 0) != (x > th4)).sum())}")
print("F3: " + ", ".join(f"{n} {abs(LAM[n]):.4f}@{np.degrees(np.angle(LAM[n])):.2f}" for n in NAMES))
print("    gauge: LEFT . W[ORD,4] =", [round(abs(LEFT[n] @ W[ORD, 4]), 6) for n in NAMES])
cosw = abs(np.real(RIGHT["echo"]) @ w[ORD]) / np.linalg.norm(np.real(RIGHT["echo"])) / np.linalg.norm(w[ORD])
print(f"F4: cosine(w[ORD], echo eigenvector) = {cosw:.4f}; max |A w[ORD]| = {np.abs(A @ w[ORD]).max():.3f}; W[PULSE,ORD] w[ORD] = {W[np.ix_(PULSE, ORD)] @ w[ORD]}")
Rl = np.real(Uo @ RIGHT["level"])
print("F7: R_level =", Rl.round(3), "-> range over positions 0..8:", Rl[:9].min().round(3), "..", Rl[:9].max().round(3))
print("F8: unit-4 kernel at own position by age:", [round(float(KP[0][a, 9 - a]), 2) for a in range(10)])
print("    unit-5 kernel at own position (input one step before) by age 0..8:", [round(float(KP[1][a, 8 - a]), 2) for a in range(9)])
print("    unit-5 kernel, mean over the 3 positions before own, ages 0..5:", [round(float(KP[1][a, 5 - a:8 - a].mean()), 2) for a in range(6)])
print("    unit-9 kernel one position before its input, ages 0..7:", [round(float(KP[2][a, 7 - a]), 2) for a in range(8)])
# F9
ok = tot = 0; fails = []
for bg in (-1.0, -0.3):
    for Sv in (0.3, 1.0, 1.4):
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                xx = np.full(10, bg); xx[i] = 1.5; xx[j] = Sv
                a = int(logits(run(xx)).argmax())
                tot += 1; ok += a == j
                if a != j:
                    fails.append((bg, Sv, i, j, a))
print(f"F9: two-pulse inputs answered correctly: {ok}/{tot}; failures (background, S, pos of M, pos of S, answer): {fails}")
