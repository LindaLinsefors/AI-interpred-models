"""V0. Does my own tape (vb.py) agree with tape_common.py, and is my sampler a draw from the conditional distribution?
(a) projector, coordinates, tape scores, forward pass, quiet mask against tape_common / common.
(b) exchangeability: if the redraw is from the exact conditional given V, the redrawn states have the same
    distribution as the originals.  Compare mean and sd of the tape coordinates / M, the network's accuracy and the
    share of quiet states between originals and redraws.
Run:  python -B v0_check.py > v0_check.txt"""
from vb import *
import tape_common as tc

x = sample(20000, 101)
h = fwd(x)
print("(a) agreement with tape_common.py / common.py on 20000 sequences (seed 101)")
print("    forward pass, max |h9 - common.run|:        ", np.abs(h - tc.run(x)[:, 9]).max())
print("    projector, max |P_TAPE - tape_common|:      ", np.abs(P_TAPE - tc.P_TAPE).max(), " rank", np.linalg.matrix_rank(P_TAPE))
print("    tape coordinates, max abs difference:       ", np.abs(h[:, o] @ C_TAPE.T - tc.tape_coords(h)).max())
print("    tape scores, max abs difference:            ", np.abs(h[:, o] @ G_TAPE.T - tc.tape_scores(h)).max())
print("    tape part, max abs difference:              ", np.abs(tape_part(h) - tc.tape_part(h)).max())
P = Pop(20000, 101)
nq = sum(int(tc.prefix_stats(x, t)["quiet"].sum()) for t in range(3, 10))
print(f"    quiet states at steps 3..9: mine {P.n}, tape_common.prefix_stats {nq}")
st = tc.prefix_stats(x, 9)
q9 = P.t == 9
print("    t = 9: same M, S, ages:", np.allclose(P.M[q9], st["M"][st["quiet"]]), np.allclose(P.S[q9], st["S"][st["quiet"]]),
      (P.aM[q9] == st["aM"][st["quiet"]]).all(), (P.aS[q9] == st["aS"][st["quiet"]]).all())
print("    zero-prefix identity, max |h(xr) - h_t(x)| at t = 6:", np.abs(P.h[P.t == 6] - fwd(P.x[P.t == 6, :7])).max())

print("\n(b) exchangeability of the redraws (40000 sequences, seed 102, one redraw per state)")
P = Pop(40000, 102)
rng = np.random.default_rng(1)
acc = lambda xr: 100 * np.mean((fwd(xr[q9]) @ U.T).argmax(1) == np.argsort(xr[q9], 1)[:, -2])
q9 = P.t == 9
print(f"    {'V':28s} | mean of c/M (7)                                   | sd of c/M (7)                                     | acc at t=9 | kept")
show = lambda nm, c, xr, k: print(f"    {nm:28s} | " + " ".join(f"{v:6.2f}" for v in c.mean(0)) + " | " + " ".join(f"{v:6.2f}" for v in c.std(0)) + f" | {acc(xr):6.2f}     | {k:.2f}")
show("originals", P.f["cT"], P.xr, (P.t + 1).mean())
for nm, rules in (("V0", ()), ("top 3", (top(3),)), ("contenders 0.75", (contenders(0.75),)), ("contenders 0 (positive)", (contenders(0.0),)),
                  ("V0 + D", (segment("pre", 1),)), ("above 0.5 S", (above_final(0.5),))):
    xk, mask = V(P, *rules)(rng)
    hk = fwd(xk)
    assert np.allclose(xk.max(1), P.M)
    show(nm, hk[:, o] @ C_TAPE.T / P.M[:, None], xk, mask.sum(1).mean())
print("    (standard error of a mean is about sd / sqrt(n) = 0.006-0.013; of the accuracy about 0.12)")
