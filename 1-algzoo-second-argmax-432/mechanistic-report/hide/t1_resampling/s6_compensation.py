"""S6. When the inputs outside V are resampled, the tape piece of the scores changes, and so does the rest of the
scores (the other five modes + pulse units).  Do the two changes add up or cancel?
Final quiet states (t = 9) of 60,000 sequences (seed 6), one resample each.  Entries of age >= 2, in units of M.
d_tape = change of the tape piece, d_rest = change of the rest of the score, d_tot = d_tape + d_rest = change of the score.
"relative" = the same after subtracting the change at the runner-up's own entry (only differences between entries
decide the answer).
Run:  python -B s6_compensation.py > s6_compensation.txt"""
from rs import *

P = Pop(60_000, seed=6, steps=[9])
rng = np.random.default_rng(6)
ar = np.arange(P.n)
rms = lambda a: np.sqrt(np.mean(a ** 2))
print(f"{P.n} final quiet states\n")
print(f"{'V':38s} |  rms d_tape  d_rest  d_tot   corr(d_tape, d_rest) | relative: rms d_tape  d_rest  d_tot   corr")
for name, rules in (("V0 = max and runner-up", ()), ("V1 = V0 + third largest", (r_top(3),)),
                    ("records above zero (c = 1)", (r_records(2),)), ("contenders c = 0.75", (r_runthr(0.75),)),
                    ("all positive inputs (c = 0)", (r_runthr(0.0),))):
    xk, _ = V(P, *rules)(rng)
    hk = run(xk)[:, 9]
    dt = (tape_scores(hk) - tape_scores(P.h))[:, :8] / P.M[:, None]
    dtot = ((hk - P.h) @ U.T)[:, :8] / P.M[:, None]
    dr = dtot - dt
    iS = 9 - P.aS
    rel = lambda d: (d - d[ar, iS][:, None])[COL[None, :8] != iS[:, None]]      # the other entries minus the runner-up's
    a, b, c = rel(dt), rel(dr), rel(dtot)
    print(f"{name:38s} |  {rms(dt):9.2f} {rms(dr):7.2f} {rms(dtot):6.2f}   {np.corrcoef(dt.ravel(), dr.ravel())[0, 1]:12.3f}         |"
          f"          {rms(a):9.2f} {rms(b):7.2f} {rms(c):6.2f}  {np.corrcoef(a, b)[0, 1]:6.3f}")
print("\nFor scale: sd over all final quiet states of the tape piece, of the rest and of the full score (entries of age >= 2, / M),")
g, s = tape_scores(P.h)[:, :8] / P.M[:, None], (P.h @ U.T)[:, :8] / P.M[:, None]
print(f"   tape {g.std():.2f}   rest {(s - g).std():.2f}   full score {s.std():.2f}   corr(tape, rest) = {np.corrcoef(g.ravel(), (s - g).ravel())[0, 1]:.3f}")
