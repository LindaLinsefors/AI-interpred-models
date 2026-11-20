"""R4 / r01: section 7 census (error classes, positive-input counts, shift) and section 4.3
(level + clock share of the logit differences, clock zeroed in h_9, gate-pattern census).
Seeds 7001-7020.  Output: r01_census_out.txt
"""
from rlib import *
import modes

out = open(__file__.replace(".py", "_out.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s); out.write(s + "\n"); out.flush()


# ---------------------------------------------------------------- census on 2^21 sequences
NB, BS = 8, 1 << 18
acc_b = []
cnt = {}


def add(k, v):
    cnt[k] = cnt.get(k, 0) + v


Rl = modes.RIGHT["level"].real
Rc = modes.RIGHT["clock"]
UL = U[:, ORD] @ Rl                       # logits per unit level coordinate
UC = U[:, ORD] @ Rc                       # complex: logits = 2 Re(z UC)
sq = {k: 0.0 for k in ("tot", "lev", "clk", "lc")}
pat_counts = {}
rest_t = np.zeros(10); n_t = 0
rest_age2 = [0, 0]; rest_age2_pos = [0, 0]
corr = {"a": [0.0, 0.0], "b": [0.0, 0.0]}
exact_lin = [0, 0]
REST = sum(1 << i for i in ORD)
for b in range(NB):
    x = gauss(BS, 7001 + b)
    H, Z = fwd(x)
    lg = H[:, -1] @ U.T
    ans = lg.argmax(1)
    xs = np.sort(x, 1)
    order = np.argsort(x, 1)
    pM, pS, pT = order[:, -1], order[:, -2], order[:, -3]
    Mv, Sv, Tv = xs[:, -1], xs[:, -2], xs[:, -3]
    ok = ans == pS
    acc_b.append(ok.mean())
    err = ~ok
    rank = 10 - np.argsort(order, 1)[np.arange(BS), ans]          # 1 = max, 2 = runner-up, ...
    add("n", BS); add("err", err.sum())
    add("err_rank1", (err & (rank == 1)).sum()); add("err_rank3", (err & (rank == 3)).sum())
    add("err_rank4+", (err & (rank >= 4)).sum())
    Sp, Tp = np.maximum(Sv, 0), np.maximum(Tv, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g1 = (Mv - Sp) / Mv
        g2 = (Sp - Tp) / Mv
        g2raw = (Sv - Tv) / Mv
    cls = np.full(BS, -1)
    cls[Mv <= 0] = 0
    cls[(Mv > 0) & (Sv <= 0)] = 1
    pos = Sv > 0
    cls[pos & (g2 < 0.05) & (g1 >= 0.05)] = 2
    cls[pos & (g1 < 0.05) & (g2 >= 0.05)] = 3
    cls[pos & (g1 < 0.05) & (g2 < 0.05)] = 4
    mg = np.minimum(g1, g2)
    cls[pos & (mg >= 0.05) & (mg < 0.1)] = 5
    cls[pos & (mg >= 0.1) & (mg < 0.2)] = 6
    cls[pos & (mg >= 0.2)] = 7
    assert (cls >= 0).all()
    for c in range(8):
        add(f"cls{c}_n", (cls == c).sum()); add(f"cls{c}_err", (err & (cls == c)).sum())
    # same gap classes with the raw third (no positive part)
    add("raw_g2<.05<=g1_err", (err & pos & (g2raw < 0.05) & (g1 >= 0.05)).sum())
    add("raw_any<.05_err", (err & pos & ((g2raw < 0.05) | (g1 < 0.05))).sum())
    add("posgap_T<=0_err", (err & pos & (Tv <= 0) & ((g2 < 0.05) | (g1 < 0.05))).sum())
    easy = (Tv > 0) & (g1 >= 0.2) & (g2 >= 0.2)
    add("easy_n", easy.sum()); add("easy_err", (err & easy).sum())
    npos = (x > 0).sum(1)
    for k in range(4):
        m = npos == k if k < 3 else npos >= 3
        add(f"npos{k}_n", m.sum()); add(f"npos{k}_ok", (ok & m).sum())
    for p in range(10):
        add(f"pM{p}_n", (pM == p).sum()); add(f"pM{p}_ok", (ok & (pM == p)).sum())
        add(f"pS{p}_n", (pS == p).sum()); add(f"pS{p}_ok", (ok & (pS == p)).sum())

    # ---------------- section 4.3 quantities on the same data
    m = modes.coords(H[:, -1])
    cl = np.outer(m["level"], UL)
    cc = 2 * np.real(np.multiply.outer(m["clock"], UC))
    cen = lambda a: a[:, :9] - a[:, :9].mean(1, keepdims=True)
    sq["tot"] += (cen(lg) ** 2).sum(); sq["lev"] += (cen(cl) ** 2).sum()
    sq["clk"] += (cen(cc) ** 2).sum(); sq["lc"] += (cen(cl + cc) ** 2).sum()
    lg0 = lg - cc                                   # clock coordinate of h_9 set to zero
    ans0 = lg0.argmax(1)
    add("clk0_changed", (ans0 != ans).sum()); add("clk0_ok", (ans0 == pS).sum())
    sub = (Mv > 0) & (pM != 9)                      # arm C's population: M > 0, max-age >= 1
    add("sub_n", sub.sum()); add("sub_ok", (ok & sub).sum())
    add("sub_clk0_changed", ((ans0 != ans) & sub).sum()); add("sub_clk0_ok", ((ans0 == pS) & sub).sum())
    lgL = lg - cl - cc
    add("lc0_best08_changed", (lgL[:, :9].argmax(1) != lg[:, :9].argmax(1)).sum())
    add("c0_best08_changed", (lg0[:, :9].argmax(1) != lg[:, :9].argmax(1)).sum())

    # ---------------- gate patterns
    pid = ((Z > 0) * (1 << np.arange(16))).sum(-1)           # (B, 10)
    if b < 4:                                                # 2^20 sequences for the pattern count
        u, c = np.unique(pid, return_counts=True)
        for a, n in zip(u, c):
            pat_counts[a] = pat_counts.get(a, 0) + n
    rest = pid == REST
    rest_t += rest.sum(0); n_t += BS
    st = prefix_stats(x)
    age = np.arange(10)[None, :] - st["pM"]
    a2 = age >= 2
    rest_age2[0] += (rest & a2).sum(); rest_age2[1] += a2.sum()
    a2p = a2 & (st["M"] > 0)
    rest_age2_pos[0] += (rest & a2p).sum(); rest_age2_pos[1] += a2p.sum()
    # rms ReLU correction to the resting map, max >= 3 steps old (two readings)
    a3 = (age >= 3)[:, 1:]
    Ho = H[:, 1:][..., ORD]
    lin_a = Z[:, 1:][..., ORD]                                # (a) ReLU clip only
    lin_b = x[:, 1:, None] * w[ORD] + H[:, :-1][..., ORD] @ W[np.ix_(ORD, ORD)].T   # (b) resting map (pulses dropped too)
    for k, lin in (("a", lin_a), ("b", lin_b)):
        corr[k][0] += ((Ho - lin)[a3] ** 2).sum(); corr[k][1] += (Ho[a3] ** 2).sum()
    exact_lin[0] += (np.abs(Ho - lin_b).max(-1) < 1e-9)[a3].sum(); exact_lin[1] += a3.sum()

n = cnt["n"]; E = cnt["err"]
P("=== SECTION 7 census, n = %d (seeds 7001-7008)" % n)
P("accuracy %.5f  +- %.5f (binomial)   per-seed %s" % (1 - E / n, np.sqrt((E / n) * (1 - E / n) / n), np.round(acc_b, 4)))
P("errors %d ; share answering third %.4f, max %.4f, rank>=4 %.4f" % (E, cnt["err_rank3"] / E, cnt["err_rank1"] / E, cnt["err_rank4+"] / E))
names = ["M<=0", "S<=0<M", "g2<.05<=g1 (runner-up~third)", "g1<.05<=g2 (max~runner-up)", "both<.05",
         "min gap [.05,.1)", "min gap [.1,.2)", "both>=.2"]
P("class: share of sequences / error rate / share of errors")
for c in range(8):
    P("  %-30s %.4f  %.4f  %.4f" % (names[c], cnt[f"cls{c}_n"] / n, cnt[f"cls{c}_err"] / max(cnt[f"cls{c}_n"], 1), cnt[f"cls{c}_err"] / E))
P("runner-up not positive (classes 0+1): share of errors %.4f" % ((cnt["cls0_err"] + cnt["cls1_err"]) / E))
P("relative gap < 0.05 (classes 2+3+4): share of errors %.4f" % ((cnt["cls2_err"] + cnt["cls3_err"] + cnt["cls4_err"]) / E))
P("two classes together: %.4f" % ((cnt["cls0_err"] + cnt["cls1_err"] + cnt["cls2_err"] + cnt["cls3_err"] + cnt["cls4_err"]) / E))
P("with RAW third (S-T)/M instead of (S-T+)/M: errors with S>0 and (g1<.05 or g2raw<.05): %.4f of errors" % (cnt["raw_any<.05_err"] / E))
P("   of the 'gap<0.05' errors, share whose third is not positive (gap is S/M, not S-T): %.4f of all errors" % (cnt["posgap_T<=0_err"] / E))
P("T>0 and both gaps>=0.2: share %.4f, accuracy %.5f (errors %d of %d)" % (cnt["easy_n"] / n, 1 - cnt["easy_err"] / cnt["easy_n"], cnt["easy_err"], cnt["easy_n"]))
P("accuracy by number of positive inputs (0,1,2,3+), unconditional sample:")
for k in range(4):
    a = cnt[f"npos{k}_ok"] / cnt[f"npos{k}_n"]
    P("  k=%d  n=%d  share %.5f  acc %.4f +- %.4f  share of errors %.4f" % (k, cnt[f"npos{k}_n"], cnt[f"npos{k}_n"] / n, a, np.sqrt(a * (1 - a) / cnt[f"npos{k}_n"]), (cnt[f"npos{k}_n"] - cnt[f"npos{k}_ok"]) / E))
P("accuracy by position of the max 0..9:", np.round([cnt[f"pM{p}_ok"] / cnt[f"pM{p}_n"] for p in range(10)], 4))
P("accuracy by position of the runner-up 0..9:", np.round([cnt[f"pS{p}_ok"] / cnt[f"pS{p}_n"] for p in range(10)], 4))

# conditional sampling: exactly k positive inputs
P("\naccuracy with exactly k positive inputs, conditional sampling (n = 2^20 each, seeds 7011-7013):")
for k in range(3):
    rng = np.random.default_rng(7011 + k)
    mag = np.abs(rng.standard_normal((1 << 20, 10)))
    idx = np.argsort(rng.random((1 << 20, 10)), 1)[:, :k]
    sgn = -np.ones((1 << 20, 10))
    np.put_along_axis(sgn, idx, 1.0, 1)
    xx = mag * sgn
    H, _ = fwd(xx)
    a = ((H[:, -1] @ U.T).argmax(1) == np.argsort(xx, 1)[:, -2]).mean()
    P("  k=%d  acc %.4f +- %.4f" % (k, a, np.sqrt(a * (1 - a) / (1 << 20))))

P("\naccuracy under a constant shift c (n = 2^20, seed 7020):")
x = gauss(1 << 20, 7020)
for c in (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3):
    H, _ = fwd(x + c)
    P("  c=%+.1f  acc %.4f" % (c, ((H[:, -1] @ U.T).argmax(1) == np.argsort(x, 1)[:, -2]).mean()))

P("\n=== SECTION 4.3")
P("rms over positions 0-8 (centred): total %.2f, level %.2f (%.3f), clock %.2f (%.3f), level+clock %.2f (%.3f; %.3f of sum of squares)" % (
    np.sqrt(sq["tot"] / (9 * n)), np.sqrt(sq["lev"] / (9 * n)), np.sqrt(sq["lev"] / sq["tot"]), np.sqrt(sq["clk"] / (9 * n)), np.sqrt(sq["clk"] / sq["tot"]),
    np.sqrt(sq["lc"] / (9 * n)), np.sqrt(sq["lc"] / sq["tot"]), sq["lc"] / sq["tot"]))
P("clock coordinate of h_9 set to zero, all sequences: answers changed %.4f ; accuracy %.4f -> %.4f (%.2f points)" % (
    cnt["clk0_changed"] / n, 1 - E / n, cnt["clk0_ok"] / n, 100 * (1 - E / n - cnt["clk0_ok"] / n)))
P("same, sequences with M>0 and max not last (n=%d): changed %.4f ; accuracy %.4f -> %.4f (%.2f points)" % (
    cnt["sub_n"], cnt["sub_clk0_changed"] / cnt["sub_n"], cnt["sub_ok"] / cnt["sub_n"], cnt["sub_clk0_ok"] / cnt["sub_n"],
    100 * (cnt["sub_ok"] - cnt["sub_clk0_ok"]) / cnt["sub_n"]))
P("best position among 0-8 changes: clock removed %.4f ; level+clock removed %.4f" % (cnt["c0_best08_changed"] / n, cnt["lc0_best08_changed"] / n))
P("clock readout 2|U_o R_clock| by position 0..9:", np.round(2 * np.abs(UC), 2))
P("\ngate patterns: resting pattern (13 ordinary on, pulses off)")
P("  share of all steps %.4f ; by t = 0..9 %s" % (rest_t.sum() / (10 * n_t), np.round(rest_t / n_t, 3)))
P("  share among states with running max >= 2 steps old: %.4f (also requiring M>0: %.4f)" % (rest_age2[0] / rest_age2[1], rest_age2_pos[0] / rest_age2_pos[1]))
c = np.sort(np.array(list(pat_counts.values())))[::-1]
cs = np.cumsum(c) / c.sum()
P("  distinct patterns in 2^20 sequences (seeds 7001-7004): %d ; patterns to cover 50%% of steps: %d ; 90%%: %d ; 99%%: %d" % (
    len(c), np.searchsorted(cs, 0.5) + 1, np.searchsorted(cs, 0.9) + 1, np.searchsorted(cs, 0.99) + 1))
P("  rms ReLU correction / rms state (ordinary units, max >= 3 steps old): clip only %.4f ; vs resting map incl. pulse inputs %.4f ; exactly linear in %.4f of states" % (
    np.sqrt(corr["a"][0] / corr["a"][1]), np.sqrt(corr["b"][0] / corr["b"][1]), exact_lin[0] / exact_lin[1]))
out.close()
