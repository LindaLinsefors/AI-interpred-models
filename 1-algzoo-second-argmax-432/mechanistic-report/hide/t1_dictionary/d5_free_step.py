"""D5. Does one more step multiply the tape coordinates of a random quiet state by the eigenvalues?
States quiet at step t that are still quiet at t + 1 (the new input is below the runner-up), t = 5..8, iid N(0,1), 400000 sequences,
seed 2314.  e_free = |c(t+1) - lam c(t)| / |c(t+1)|;  e_lin = the same after also subtracting the writes of a step that moves no
gate of the 13 non-pulse units (direct input and the pulses of units 4, 5, 9 present at step t), i.e. e_lin = |clip terms| / |c(t+1)|.
Split by the age of the maximum and by whether the newest input x_t (age 0 at step t) fired unit 5 at step t + 1.
Run:  python -B d5_free_step.py > d5_free_step.txt"""
from dict_common import *

x = sample(400_000, 2314); H, Z = run(x, return_pre=True)
kw, kp = LEFTM @ w[ORD], {u: LEFTM @ W[ORD, u] for u in PULSE}
nrm = lambda D: np.sqrt((np.abs(D) ** 2).sum(-1))
rows = []
for t in range(5, 9):
    a, b = prefix_stats(x, t), prefix_stats(x, t + 1); q = a["quiet"] & b["quiet"]
    c0, c1 = zc(H[q, t]), zc(H[q, t + 1])
    lin = LAMV * c0 + x[q, t + 1, None] * kw + sum(H[q, t, u, None] * kp[u] for u in PULSE)
    nclip = (Z[q, t + 1][:, ORD] < 0).sum(1)
    rows.append(np.stack([nrm(c1 - LAMV * c0) / nrm(c1), nrm(c1 - lin) / nrm(c1), a["aM"][q], a["aS"][q], H[q, t, 5] > 0, nclip,
                          nrm(c1) / b["M"][q], (Z[q, t + 1][:, COMP] < 0).sum(1)], 1))
R = np.concatenate(rows)
d = lambda v: f"median {np.median(v):.3f}, 90% {np.quantile(v, .9):.3f}, share < 0.1: {np.mean(v < 0.1):.2f}"
print(f"{len(R)} pairs of consecutive quiet states.  |c(t+1)| / M median {np.median(R[:, 6]):.2f}")
print("   e_free:", d(R[:, 0])); print("   e_lin :", d(R[:, 1]))
print(f"   share of steps on which no non-pulse unit is clipped: {np.mean(R[:, 5] == 0):.3f};  a comparator is clipped: {np.mean(R[:, 7] > 0):.3f}")
print("   by the number of clipped non-pulse units at step t + 1:")
for k in range(0, 5):
    s = R[:, 5] == k if k < 4 else R[:, 5] >= 4
    print(f"     {k}{'+' if k == 4 else ' '} clipped (share {s.mean():.3f}):  e_lin {d(R[s, 1])}")
print("   by the age of the maximum at step t:")
for k in range(2, 9):
    s = R[:, 2] == k
    print(f"     a_M = {k} (n = {s.sum():6d}):  e_free {d(R[s, 0])} | e_lin {d(R[s, 1])}")
print("   by the age of the runner-up at step t:")
for k in range(2, 9):
    s = R[:, 3] == k
    print(f"     a_S = {k} (n = {s.sum():6d}):  e_lin {d(R[s, 1])}")
s = R[:, 4] > 0
print(f"   unit 5 on at step t (share {s.mean():.3f}): e_free {d(R[s, 0])} | e_lin {d(R[s, 1])}")
print(f"   unit 5 off at step t:                  e_free {d(R[~s, 0])} | e_lin {d(R[~s, 1])}")
