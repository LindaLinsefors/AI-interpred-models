"""D3. Test of the dictionary on random quiet states (iid N(0,1) inputs, B = 400000 sequences, seed 2311), at t = 9 and t = 6.

(a) EXACT WRITES.  The tape coordinates obey, exactly,
      z_m(s) = lam_m z_m(s-1) + (l_m.w[o]) x_s + sum_{u in 4,5,9} (l_m.W[o,u]) h_(s-1)[u] + l_m.clip_s ,   clip_s = [-pre_s[o]]_+ ,
    so z_m(t) = sum over steps s of lam_m^(t-s) (write at step s).  Terms: direct input; pulses of units 4, 9; pulse of unit 5
    split by the input that caused it (input s-2): the maximum at t, the runner-up at t, an input that was above the runner-up of its
    time when it arrived ("dethroned"), an input that was below it ("below"); clips of comparators; clips of wave units.
(b) SKELETONS.  Tape of the controlled sequence that keeps only some inputs of x_0..x_t (the rest replaced by 0).
(c) FORMULA N1 = tape of the skeleton (M, S) + sum over the later inputs p (age >= 2) of g (x_p - Theta_p)_+ kA lam^(age-2),
    Theta = 0.75 S after the later of M, S, raised by 0.44 (x_p - Theta)_+ after every input.  g = 19.1.
Scores: R2 of each tape coordinate / M_t over quiet states (centred), and the FUNCTIONAL TEST: the tape part of h_t is replaced
(tape_common.set_tape), the run continued with the real x_(t+1)..x_9, and the answer compared with the unmodified run.
Run:  python -B d3_random_states.py > d3_random_states.txt"""
from dict_common import *

B = 400_000
x = sample(B, 2311); H, Z = run(x, return_pre=True); st = running_stats(x)
ans0 = answer(H[:, 9]); ar = np.arange(B)
kw, k4, k5, k9 = (LEFTM @ w[ORD], LEFTM @ W[ORD, 4], LEFTM @ W[ORD, 5], LEFTM @ W[ORD, 9])
Lc, Lw = LEFTM[:, [ORD.index(k) for k in COMP]], LEFTM[:, [ORD.index(k) for k in WAVE]]
kAc = from7(np.load(HERE / "d1_patterns.npz")["kA"])
rng = np.random.default_rng(5)


def exact_terms(t, ps):
    names = ["direct", "pulse 4", "pulse 9", "pulse 5: maximum", "pulse 5: runner-up", "pulse 5: dethroned", "pulse 5: below",
             "pulse 5: step 0", "clip comparators", "clip wave units"]
    T = {n: np.zeros((B, 4), complex) for n in names}
    for s in range(t + 1):
        for n in names: T[n] *= LAMV
        T["direct"] += x[:, s, None] * kw
        if s >= 1:
            T["pulse 4"] += H[:, s - 1, 4, None] * k4; T["pulse 9"] += H[:, s - 1, 9, None] * k9
            u = H[:, s - 1, 5, None] * k5; j = s - 2
            if j < 0: T["pulse 5: step 0"] += u
            else:
                above = (x[:, j] > st["M2"][:, j - 1]) if j >= 1 else np.ones(B, bool)
                isM, isS = ps["pM"] == j, ps["pS"] == j
                for n, m in (("pulse 5: maximum", isM), ("pulse 5: runner-up", isS), ("pulse 5: dethroned", ~isM & ~isS & above),
                             ("pulse 5: below", ~isM & ~isS & ~above)):
                    T[n] += u * m[:, None]
        clip = np.maximum(-Z[:, s], 0)
        T["clip comparators"] += clip[:, COMP] @ Lc.T; T["clip wave units"] += clip[:, WAVE] @ Lw.T
    return T


def skeleton(t, keep):
    return zc(run(np.where(keep, x[:, : t + 1], 0.0))[:, -1])


def formula_n1(t, ps, g=19.1, th0=0.75, beta=0.44):
    pre = x[:, : t + 1]; pos = np.arange(t + 1)
    out = skeleton(t, (pos == ps["pM"][:, None]) | (pos == ps["pS"][:, None]))
    start = np.maximum(ps["pM"], ps["pS"]) + 1; Th = th0 * ps["S"]; nw = np.zeros(B)
    for p in range(t + 1):
        e = np.where(p >= start, np.maximum(pre[:, p] - Th, 0), 0.0)
        if t - p >= 2: out = out + (g * e)[:, None] * kAc * LAMV ** (t - p - 2); nw += e > 0
        Th = Th + beta * e
    return out, nw


def functional(t, q, zf):
    """Share of quiet states whose final answer changes when the tape at step t is replaced by zf (complex (B,4));
    the same with the modified state clipped at 0; share of modified states with a negative activation."""
    h = set_tape(H[q, t], to7(zf[q]))
    fin = lambda h_: answer(h_ if t == 9 else continue_from(h_, x[q, t + 1:])[:, -1])
    return np.mean(fin(h) != ans0[q]), np.mean(fin(np.maximum(h, 0)) != ans0[q]), np.mean((h < -1e-9).any(1))


def report(t):
    ps = prefix_stats(x, t); q = ps["quiet"]; Mq = ps["M"][q, None]
    zreal = zc(H[:, t]); c = to7(zreal[q]) / Mq
    r2c = lambda zf: 1 - ((to7(zf[q]) / Mq - c) ** 2).sum(0) / ((c - c.mean(0)) ** 2).sum(0)
    r2all = lambda zf: 1 - ((to7(zf[q]) / Mq - c) ** 2).sum() / ((c - c.mean(0)) ** 2).sum()
    def line(label, zf):
        f = functional(t, q, zf)
        print(f"   {label:46s} R2 all {r2all(zf):6.3f} | per coordinate " + " ".join(f"{v:6.2f}" for v in r2c(zf)) +
              f" | answer changes {f[0]:.3f}  (clipped at 0: {f[1]:.3f}; states with a negative activation {f[2]:.2f})")
    print(f"\n===== t = {t}: {q.sum()} quiet states of {B} ({q.mean():.3f}).  rms of the tape coordinates / M: {np.round(np.sqrt((c ** 2).mean(0)), 2)}, "
          f"|c| rms {np.sqrt((c ** 2).sum(1).mean()):.2f}")
    print(f"   accuracy of the unmodified network on these sequences: {np.mean(ans0[q] == second_argmax(x)[q]):.3f}")
    print("   coordinates: " + ", ".join(COORD7))
    print(" (0) reference substitutions")
    perm = rng.permutation(np.flatnonzero(q)); donor = np.zeros((B, 4), complex); donor[q] = zreal[perm] / ps["M"][perm, None] * ps["M"][q, None]
    line("zero tape", np.zeros((B, 4), complex))
    line("mean tape of quiet states (times M)", from7(c.mean(0))[None] * ps["M"][:, None])
    line("tape of another random quiet state (times M)", donor)
    print(" (a) exact writes: rms size of each term / M, and nested sums")
    T = exact_terms(t, ps); tot = sum(T.values())
    print(f"   exactness: max |sum of terms - tape| = {np.abs(tot - zreal).max():.2e}")
    for n, v in T.items():
        print(f"     {n:22s} rms |term| / M = {np.sqrt((np.abs(v[q] / Mq) ** 2).sum(1).mean()):6.2f}")
    p5 = sum(v for n, v in T.items() if n.startswith("pulse 5"))
    line("unit 5's pulses only", p5)
    line("all three pulse units", p5 + T["pulse 4"] + T["pulse 9"])
    line("... + direct input", p5 + T["pulse 4"] + T["pulse 9"] + T["direct"])
    line("... + comparator clips (all but wave clips)", tot - T["clip wave units"])
    print("   leave one term out of the exact tape:")
    for n, v in T.items():
        line("  exact minus " + n, tot - v)
    print(" (b) skeletons (inputs not kept are replaced by 0)")
    pre = x[:, : t + 1]; pos = np.arange(t + 1)
    isM, isS, isT = (pos == ps["pM"][:, None]), (pos == ps["pS"][:, None]), (pos == ps["pT"][:, None])
    rank = np.argsort(np.argsort(-pre, 1), 1)
    sk = {"keep M, S": isM | isS, "keep M, S, T (top 3)": rank < 3, "keep top 4": rank < 4, "keep top 5": rank < 5,
          "keep inputs > 0.5 S": pre > 0.5 * ps["S"][:, None], "keep inputs > 0.75 S": pre > 0.75 * ps["S"][:, None],
          "keep inputs > 0 (negatives -> 0)": pre > 0, "keep M, S and all inputs after both": isM | isS | (pos > np.maximum(ps["pM"], ps["pS"])[:, None]),
          "keep M, S and all inputs before both": isM | isS | (pos < np.minimum(ps["pM"], ps["pS"])[:, None]),
          "keep M, S and all inputs between them": isM | isS | ((pos > np.minimum(ps["pM"], ps["pS"])[:, None]) & (pos < np.maximum(ps["pM"], ps["pS"])[:, None]))}
    for n, keep in sk.items():
        line(f"{n} [{keep[q].sum(1).mean():.2f} kept]", skeleton(t, keep))
    line("keep M, S, T; the rest replaced by -M", zc(run(np.where(rank < 3, pre, -ps["M"][:, None]))[:, -1]))
    print(" (c) formula N1")
    zf, nw = formula_n1(t, ps)
    line("N1: skeleton (M, S) + explicit later writes", zf)
    print(f"      later inputs written (age >= 2) per quiet state: mean {nw[q].mean():.2f}; share of states with 0, 1, 2+: "
          f"{np.mean(nw[q] == 0):.2f}, {np.mean(nw[q] == 1):.2f}, {np.mean(nw[q] >= 2):.2f}")
    for k in (0, 1, 2):
        s = q & (nw == k if k < 2 else nw >= k)
        e = lambda z_: 1 - ((to7(z_[s]) - to7(zreal[s])) ** 2).sum() / ((to7(zreal[s]) / ps["M"][s, None] - c.mean(0)) ** 2 * ps["M"][s, None] ** 2).sum()
        print(f"      states with {k}{'+' if k == 2 else ''} written later inputs (n = {s.sum()}): R2 all, skeleton (M, S) {e(skeleton(t, isM | isS)):.3f}, N1 {e(zf):.3f}")


for t in (9, 6):
    report(t)
