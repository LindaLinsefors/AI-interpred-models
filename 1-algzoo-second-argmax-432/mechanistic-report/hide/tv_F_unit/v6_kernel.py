"""V6. Claim 5: how a write spreads over the modes (weights only), and what the data say.
A. Resting kernel U_o A^(k-1) W[o, u] for u = 5, 4, split into tape / trans / echo / level+runner+clock.
B. Data (quiet final states, own seed): where the tape piece alone fails, by age of the runner-up, and whether
   adding the transient mode repairs exactly those failures.
C. Data check the investigator did not make: the "slow lobe on entries older than the maximum".  Prediction from
   the kernel: on entries OLDER than the maximum the tape piece is too high and level+runner+clock (LRC) is negative,
   relative to entries YOUNGER than the maximum.  Mean of each piece on plain entries (old, below the runner-up,
   more than 0.3 M below it so that the comparison itself is saturated), centred on the state's mean over such entries.
D. The seed-dependence of the variance shares of claim 3 (heavy tails from dividing by M): three seeds.
Run:  python -B v6_kernel.py > v6_kernel.txt"""
from vcommon import *

print("A. resting kernel (all 13 gates open), logits per unit pulse")
GR = {"all modes": MODES, "tape": TAPE, "trans": ["trans"], "echo": ["echo"], "level+runner+clock": ["level", "runner", "clock"]}
for u, ks in ((5, (1, 2, 3)), (4, (1, 5))):
    for k in ks:
        kick = np.linalg.matrix_power(A, k - 1) @ W[ORD, u]
        print(f"   unit {u}, {k} step(s) after the pulse (written input at index {9 - k - (u == 5)}); index 0..9:")
        for nm, names in GR.items():
            print(f"      {nm:20s} " + " ".join(f"{v_:7.2f}" for v_ in Uo @ sum(PROJ[m] for m in names) @ kick))
p = np.real(Uo @ RVEC["trans"] * (LVEC["trans"] @ W[ORD, 5])); print("   trans readout pattern x its coordinate after a unit-5 pulse, centred over indices 0-7: " + " ".join(f"{v_:6.2f}" for v_ in p[:8] - p[:8].mean()))

d = population(sample(600_000, 9601)); n = len(d["h"]); ar = np.arange(n)
pc = pieces(d["h"]); full = d["h"] @ U.T; tape = sum(pc[m] for m in TAPE)
tt = top_old(tape, d["valid"]); ttr = top_old(tape + pc["trans"], d["valid"]); tte = top_old(tape + pc["trans"] + pc["echo"], d["valid"])
print(f"\nB. {n} quiet final states (600000 sequences, seed 9601)")
for nm, q in (("runner-up of age 2", d["aS"] == 2), ("runner-up of age 3", d["aS"] == 3), ("runner-up of age >= 4", d["aS"] >= 4)):
    bad = q & (tt != d["iS"]); old9 = bad & (tt == 0)
    print(f"   {nm:22s} n {q.sum():6d}  tape fails {bad.sum() / q.sum():.3f};  of the failures the age-9 entry is on top in {old9.sum() / bad.sum():.3f};"
          f"  + trans repairs {np.mean(ttr[bad] == d['iS'][bad]):.3f} of the failures ({np.mean(ttr[old9] == d['iS'][old9]):.3f} of those with the age-9 entry on top), + trans + echo {np.mean(tte[bad] == d['iS'][bad]):.3f};"
          f"  + trans breaks {np.mean(ttr[q & ~bad] != d['iS'][q & ~bad]):.3f} of the successes")

print("\nC. plain entries (old, more than 0.3 M below the runner-up): mean of each piece / M minus the state's mean over its plain entries, by position relative to the maximum")
v = (d["xr"][:, :8] - d["S"][:, None]) / d["M"][:, None]; plain = d["valid"] & (v < -0.3)
i8 = np.broadcast_to(np.arange(8), (n, 8)); older = i8 < d["iM"][:, None]
LRC = pc["level"] + pc["runner"] + pc["clock"]; TE = pc["trans"] + pc["echo"]
def centred(Sc):
    Y = Sc[:, :8] / d["M"][:, None]; mu_ = np.where(plain, Y, 0).sum(1) / np.maximum(plain.sum(1), 1); return Y - mu_[:, None]
print("   a_M   n(states)   OLDER than the maximum: tape, LRC, trans+echo, full   |   YOUNGER: tape, LRC, trans+echo, full   |  sd of the tape value on older entries")
for aM in range(3, 9):
    q = (d["aM"] == aM) & (plain & older).any(1) & (plain & ~older).any(1)
    row = f"   {aM}   {q.sum():7d}   "
    for side in (older, ~older):
        e = plain & side & q[:, None]
        row += "  ".join(f"{centred(Sc)[e].mean():6.2f}" for Sc in (tape, LRC, TE, full)) + "   |   "
    print(row + f"{centred(tape)[plain & older & q[:, None]].std():6.2f}")

print("\nD. claim 3, entry-level variance shares of the rest piece for three seeds (offset per (a, a_M, a_S) | + S/M | + x9, x8 | + own input), and the rms of D")
for seed, B in ((201, 600_000), (9602, 500_000), (9603, 500_000)):
    d = population(sample(B, seed)); n = len(d["h"]); ar = np.arange(n); full = d["h"] @ U.T
    rest = full - d["h"][:, ORD] @ (Uo @ P_TAPE).T
    M, S = d["M"], d["S"]; x9, x8 = d["xr"][:, 9], d["xr"][:, 8]; train = ar < n // 2; test = ~train
    v = (d["xr"][:, :8] - S[:, None]) / M[:, None]; below = d["valid"] & (v < 0); cell2 = d["aM"] * 10 + d["aS"]
    D = (rest[:, :8] - rest[ar, d["iS"]][:, None]) / M[:, None]
    sel = below & test[:, None]; tot = ((D[sel] - D[sel].mean()) ** 2).sum(); sse = np.zeros(4)
    for c in np.unique(cell2):
        for i in range(8):
            rows = np.flatnonzero((cell2 == c) & below[:, i]); tr = train[rows]
            if tr.sum() < 25:
                sse += (D[rows[~tr], i] ** 2).sum(); continue
            F = np.stack([np.ones(len(rows)), S[rows] / M[rows], x9[rows] / M[rows], x8[rows] / M[rows], v[rows, i]], 1)
            for j, k in enumerate((1, 2, 4, 5)):
                b = np.linalg.lstsq(F[tr, :k], D[rows[tr], i], rcond=None)[0]; sse[j] += ((D[rows[~tr], i] - F[~tr, :k] @ b) ** 2).sum()
    Dq = D[sel]
    print(f"   seed {seed}: {n} states; rms D {np.sqrt((Dq ** 2).mean()):.2f}; shares " + "  ".join(f"{1 - s / tot:.3f}" for s in sse)
          + f";  share of sum D^2 from the 0.1% largest |D|: {np.sort(Dq ** 2)[-len(Dq) // 1000:].sum() / (Dq ** 2).sum():.3f};  largest |D| {np.abs(Dq).max():.0f}")
