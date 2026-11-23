"""D4. Capacity of the tape under the dictionary  c = sum_e A_e kA lam^(a_e - 2)  (ages a_e in 2..9, amplitudes A_e > 0).
Phi = the 7 x 8 real matrix whose column for age a is the 7 tape coordinates of a unit write of that age.
(1) Phi: column norms, singular values, null direction.
(2) Formula-generated tapes with N writes (distinct random ages, amplitudes Unif(1, 5)) plus Gaussian noise of sd sigma per
    coordinate: are the ages and amplitudes read back?  Decoder "subset": least squares over all N-subsets of the 8 ages.
    Decoder "readout": the network's own tape piece of the scores, U[:, o] (tape part), top N entries among ages 2..9.
(3) Real quiet final states: least-squares slot amplitudes pinv(Phi) c against the real pulses of unit 5.
Run:  python -B d4_capacity.py > d4_capacity.txt"""
from itertools import combinations
from dict_common import *

rng = np.random.default_rng(4)
kAc = from7(np.load(HERE / "d1_patterns.npz")["kA"]); AGES = np.arange(2, 10)
Phi = np.stack([to7(kAc * LAMV ** (a - 2)) for a in AGES], 1)                       # (7, 8)
sv = np.linalg.svd(Phi)
print("1. Phi.  column norms (ages 2..9):", np.round(np.linalg.norm(Phi, axis=0), 2))
print("   singular values:", np.round(sv[1], 3), f"  condition number {sv[1][0] / sv[1][-1]:.1f}")
print("   null direction over the ages 2..9 (a change of the amplitudes that leaves the tape unchanged):", np.round(sv[2][-1], 3))
Phin = Phi / np.linalg.norm(Phi, axis=0)
print("   cosine between the columns of two ages (rows, columns = ages 2..9):")
for i in range(8): print("     " + " ".join(f"{v:6.2f}" for v in (Phin.T @ Phin)[i]))
S_read = np.stack([coords_to_tape13(Phi[:, j]) @ U[:, ORD].T for j in range(8)], 1)  # (10 logits, 8 ages): tape piece of the scores per unit write
print("   the network's readout of one unit write: tape piece of the scores at the entries of age 2..9 (columns), write of age a (rows)")
for j in range(8): print(f"     a = {AGES[j]}: " + " ".join(f"{S_read[9 - b, j]:6.2f}" for b in AGES))

print("\n2. Formula-generated tapes, 20000 per cell.  ages = share with the set of ages read back exactly; amp = rms error of the amplitudes")
print("   (subset decoder, true ages given) in units of the mean amplitude 3.")
n = 20000
for sigma in (0.0, 0.1, 0.3, 1.0):
    row = []
    for N in (1, 2, 3, 4, 5):
        ages_idx = np.argsort(rng.random((n, 8)), 1)[:, :N]; A = rng.uniform(1, 5, (n, N))
        amp = np.zeros((n, 8)); np.put_along_axis(amp, ages_idx, A, 1)
        c = amp @ Phi.T + sigma * rng.standard_normal((n, 7))
        best, bres = np.zeros((n, N), int), np.full(n, np.inf)
        for sub in combinations(range(8), N):
            P = Phi[:, sub]; res = ((c - c @ np.linalg.pinv(P).T @ P.T) ** 2).sum(1)
            better = res < bres - 1e-12; bres[better] = res[better]; best[better] = sub
        ok_sub = (np.sort(best, 1) == np.sort(ages_idx, 1)).all(1).mean()
        sc = coords_to_tape13(c) @ U[:, ORD].T                                       # (n, 10) tape piece of the scores
        topN = np.argsort(-sc[:, :8], 1)[:, :N]                                      # logit indices 0..7 = ages 9..2
        ok_read = (np.sort(7 - topN, 1) == np.sort(ages_idx, 1)).all(1).mean()       # age index j = 7 - logit index
        true_sorted = np.sort(ages_idx, 1); Ahat = np.stack([np.linalg.lstsq(Phi[:, s_], c_, rcond=None)[0] for s_, c_ in zip(true_sorted[:2000], c[:2000])])
        Atrue = np.take_along_axis(amp[:2000], true_sorted[:2000], 1)
        row.append(f"N={N}: subset {ok_sub:.2f} readout {ok_read:.2f} amp {np.sqrt(((Ahat - Atrue) ** 2).mean()) / 3:.2f}")
    print(f"   sigma = {sigma:3.1f} | " + " | ".join(row))

print("\n3. Real quiet final states (200000 sequences, seed 2313): slot amplitudes pinv(Phi) c  against unit 5's pulse for the input of that age")
x = sample(200_000, 2313); H = run(x); ps = prefix_stats(x, 9); q = ps["quiet"]
c = to7(zc(H[q, 9])) / ps["M"][q, None]; Ahat = c @ np.linalg.pinv(Phi).T                                # (nq, 8)
u = np.stack([H[q, 9 - a + 1, 5] for a in AGES], 1) / ps["M"][q, None]                                   # pulse one step after the input of age a
print("   age:                       " + " ".join(f"{a:6d}" for a in AGES))
print("   correlation(Ahat_a, u_a):  " + " ".join(f"{np.corrcoef(Ahat[:, j], u[:, j])[0, 1]:6.2f}" for j in range(8)))
print("   rms u_a / M:               " + " ".join(f"{np.sqrt((u[:, j] ** 2).mean()):6.2f}" for j in range(8)))
print("   rms Ahat_a:                " + " ".join(f"{np.sqrt((Ahat[:, j] ** 2).mean()):6.2f}" for j in range(8)))
print(f"   R2 of the tape from the real pulses, c = Phi u (uncentred): {1 - ((c - u @ Phi.T) ** 2).sum() / (c ** 2).sum():.3f}")
aS = ps["aS"][q]; aM = ps["aM"][q]
print(f"   the age with the largest Ahat is the runner-up's age in {np.mean(AGES[Ahat.argmax(1)] == aS):.3f} of the states, the maximum's in {np.mean(AGES[Ahat.argmax(1)] == aM):.3f}")
sc = tape_scores(H[q, 9])
print(f"   for comparison, the largest entry of the tape piece of the scores (ages 2..9) is the runner-up's in {np.mean(sc[:, :8].argmax(1) == ps['pS'][q]):.3f}")
