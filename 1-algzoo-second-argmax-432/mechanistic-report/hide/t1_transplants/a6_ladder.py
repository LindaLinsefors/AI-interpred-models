"""a6: how much must donor and recipient share for the transplanted tape to work?  A ladder of donors B for a quiet
recipient A (same step), from unrelated to identical; hybrid = A's rest + B's tape.

  unrelated          another quiet sequence, rescaled to A's M
  same pM, pS        another quiet sequence with the same POSITIONS of maximum and runner-up, rescaled to A's M
  top 2 kept         A with every input except M and S redrawn from N(0,1) truncated below S       (values + positions of M, S shared)
  top 3 kept         A with every input except the top three redrawn below T
  top 4 kept         A with every input except the top four redrawn below the fourth
  far-below redrawn  A with the inputs more than 0.4 M below S redrawn (the 'low_val' pair)
  identical          B = A
Reported: accuracy of the hybrid (answer = the correct position for A's prefix + the future), next to A intact,
 at t = 9 read out at once, and for a transplant at t = 6 followed by chosen futures (as in a3; rows where the truth
 for B's prefix differs from A's are dropped, which only matters for the first two donors).
Also: 2. S_pos transplants restricted to hybrids without any negative entry; 3. what a half dose answers.
"""
from tp import *

N = 600_000


def redraw_below(x, t, k, rng):
    """Keep the k largest inputs of the prefix, redraw the others from N(0,1) truncated at the k-th largest."""
    pre = x[:, : t + 1]
    kth = np.sort(pre, 1)[:, -k]
    new = norm.ppf(rng.random(pre.shape) * norm.cdf(kth)[:, None])
    xB = x.copy()
    xB[:, : t + 1] = np.where(pre >= kth[:, None], pre, new)
    return xB


def same_positions(x, st, rng):
    """For each row another row with the same (pM, pS): cyclic shift inside each group after a random shuffle."""
    key = st["pM"] * 10 + st["pS"]
    order = np.lexsort((rng.random(len(x)), key))
    ks = key[order]
    nxt = np.roll(order, -1)
    first = np.r_[True, ks[1:] != ks[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(x)), 0))
    last = np.r_[ks[1:] != ks[:-1], True]
    nxt[last] = order[start[last]]
    partner = np.empty(len(x), int); partner[order] = nxt
    return x[partner] * (st["M"] / st["M"][partner])[:, None], partner != np.arange(len(x))


def donors(x, st, t, rng):
    yield "unrelated", *make_pair("unrel", x, st, t, rng)
    yield "same pM, pS", *same_positions(x, st, rng)
    for k in (2, 3, 4):
        yield f"top {k} kept", redraw_below(x, t, k, rng), np.ones(len(x), bool)
    yield "far-below redrawn", *make_pair("low_val", x, st, t, rng)
    yield "identical", x.copy(), np.ones(len(x), bool)


print("1. Accuracy of A's rest + B's tape [clipped at zero] | A intact | share of rows used")
for t in (9, 6):
    x, st = quiet_sample(N, t, seed=61)
    hA = state_at(x, t)
    rng = np.random.default_rng(62 + t)
    n, K = len(x), 9 - t
    low = -st["M"][:, None] * rng.uniform(0.05, 0.6, (n, K))
    futs = {"read out at once": low[:, :0]} if t == 9 else {}
    if t < 9:
        nm, z = 1.3 * st["M"], (st["S"] + st["M"]) / 2
        y = (np.maximum(st["T3"], 0) + st["S"]) / 2
        put = lambda v, k: np.concatenate([low[:, :k], v[:, None], low[:, k + 1:]], 1)
        futs = {"low x3": low, "new max at 7": put(nm, 0), "new max at 9": put(nm, 2), "new runner-up at 7": put(z, 0),
                "new runner-up at 9": put(z, 2), "new third at 7": put(y, 0), "A's own inputs": x[:, t + 1:]}
    print(f"\n   transplant at t = {t}, {n} quiet states")
    print("   " + " " * 20 + "".join(f"{k:>26s}" for k in futs))
    for lab, xB, ok in donors(x, st, t, rng):
        sB = prefix_stats(xB, t)
        ok = ok & sB["quiet"]
        hyb = hybrid(hA, state_at(xB, t))
        row = f"   {lab:18s}"
        for f in futs.values():
            tA = second_argmax(np.concatenate([x[:, : t + 1], f], 1))
            tB = second_argmax(np.concatenate([xB[:, : t + 1], f], 1))
            m = ok & (tA == tB)
            acc = lambda h, clip=False: np.mean(finish(h[m], f[m], clip) == tA[m])
            row += f"  {acc(hyb):.3f} [{acc(hyb, True):.3f}] | {acc(hA):.3f} | {m.mean():.2f}"
        print(row)

# ------------------------------------------------------------------ 2, 3
x, st = quiet_sample(N, 9, seed=63)
rng = np.random.default_rng(64)
xB, ok = make_pair("S_pos", x, st, 9, rng)
sB = prefix_stats(xB, 9)
ok &= sB["quiet"]
a, b = {k: v[ok] for k, v in st.items()}, {k: v[ok] for k, v in sB.items()}
hA_, hB_ = state_at(x[ok], 9), state_at(xB[ok], 9)
fw = hybrid(hA_, hB_)
neg = (fw < -1e-9).any(1)
ans = answer(fw)
print(f"\n2. S_pos at t = 9, n = {ok.sum()}: P(answer = tape's runner-up) among hybrids without a negative entry "
      f"{np.mean(ans[~neg] == b['pS'][~neg]):.3f} (n = {(~neg).sum()}), with one {np.mean(ans[neg] == b['pS'][neg]):.3f};"
      f"  most negative entry / M among those with one: median {np.median((fw.min(1) / a['M'])[neg]):.2f}"
      f"   (rms activation / M of intact states {np.sqrt((hA_ ** 2).mean(1)).mean() / a['M'].mean():.2f})")
half = answer(hybrid(hA_, hB_, q=0.5))
print("3. S_pos at t = 9, half dose (tape = mean of A's and B's): " + fmt(where_ans(half, {"S_A": a["pS"], "S_B": b["pS"], "third": a["pT"], "maximum": a["pM"]})))
