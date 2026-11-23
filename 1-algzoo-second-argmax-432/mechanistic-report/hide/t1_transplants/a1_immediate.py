"""a1: matched-pair tape transplants, read out at once.  Quiet states at step t (t = 9: final answer; t < 9: the
running answer, which by the zero-prefix identity is the final answer of the zero-padded sequence).

For every pair kind (tp.py) the answer of
   A intact, B intact, hybrid = A's rest + B's tape, the same clipped at zero, reverse = B's rest + A's tape (+clipped)
is compared with the positions of the top three in A and in B.  Columns can coincide (e.g. S_A = S_B in M_pos).
"""
from tp import *

N = 600_000
for t in (9, 6):
    x, st = quiet_sample(N, t, seed=11)
    hA = state_at(x, t)
    print(f"\n{'=' * 110}\nstep t = {t}: {len(x)} quiet states of {N} sequences;  intact accuracy on them "
          f"{np.mean(running_answer(hA, t) == st['pS']):.4f}")
    rng = np.random.default_rng(100 + t)
    for kind in KINDS:
        xB, ok = make_pair(kind, x, st, t, rng)
        sB = prefix_stats(xB, t)
        ok &= sB["quiet"]
        xa, xb, a, b = x[ok], xB[ok], {k: v[ok] for k, v in st.items()}, {k: v[ok] for k, v in sB.items()}
        hA_, hB_ = hA[ok], state_at(xb, t)
        names = {"S_A": a["pS"], "S_B": b["pS"], "M_A": a["pM"], "M_B": b["pM"], "T_A": a["pT"], "T_B": b["pT"]}
        fw, rv = hybrid(hA_, hB_), hybrid(hB_, hA_)
        dc = tape_coords(hB_) - tape_coords(hA_)
        rel = np.sqrt((dc ** 2).sum(1)) / np.sqrt((tape_coords(hA_) ** 2).sum(1))
        print(f"\n--- {kind}: n = {ok.sum()}   |tape_B - tape_A| / |tape_A|: median {np.median(rel):.2f} "
              f"(quartiles {np.quantile(rel, .25):.2f}, {np.quantile(rel, .75):.2f})   "
              f"hybrids with a negative entry: forward {neg_share(fw):.3f}, reverse {neg_share(rv):.3f}")
        rows = {"A intact": hA_, "B intact": hB_, "A rest + B tape": fw, "  clipped": np.maximum(fw, 0),
                "B rest + A tape": rv, "  clipped ": np.maximum(rv, 0)}
        for lab, h in rows.items():
            print(f"   {lab:17s} {fmt(where_ans(running_answer(h, t), names))}")
        if kind in ("S_pos", "unrel"):                                    # split by how close the third is
            near = a["T3"] > a["S"] - NEAR * a["M"]
            if kind == "unrel":
                near |= b["T3"] > b["S"] - NEAR * b["M"]
            for lab, m in (("third within 0.1 M of S", near), ("third further away", ~near)):
                nm = {k: v[m] for k, v in names.items()}
                print(f"   [{lab}, n = {m.sum()}]  A rest + B tape: {fmt(where_ans(running_answer(fw[m], t), nm))}")
        if kind in ("S_val", "M_val"):                                    # does the answer move with the size of the change?
            key = "S" if kind == "S_val" else "M"
            d = (b[key] - a[key]) / a["M"]
            ans = running_answer(fw, t)
            qs = np.quantile(d, [0, .2, .4, .6, .8, 1])
            print("   by change of the value, (new - old) / M_A:  " + "  ".join(
                f"[{lo:+.2f},{hi:+.2f}] S_A {np.mean(ans[(d >= lo) & (d <= hi)] == a['pS'][(d >= lo) & (d <= hi)]):.3f}"
                for lo, hi in zip(qs[:-1], qs[1:])))
