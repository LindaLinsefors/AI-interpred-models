"""A6: do low inputs leave a trace?  Causal test by resampling one input within its rank class.

For a step t >= 2 with S = S_{t-1} > 0 (second largest of x_0..x_{t-1}) and M = M_{t-1}, the input x_t is replaced by an
independent N(0,1) draw conditioned on the same interval, everything else kept, and the network is rerun.
   L: x<0          interval (-inf, 0)
   L: 0<x<.5S      interval (0, 0.5 S)
   L: x<.5S        interval (-inf, 0.5 S)        (union of the two above)
   L: .5S<x<.7S    interval (0.5 S, 0.7 S)
   L: .7S<x<S      interval (0.7 S, S)
   N: S<x<M        interval (S, M)
Resampling seeds: 2000 + batch index.  Data: the 2^20 sequences of cc.batches().
"""
from cc import *
from scipy.special import ndtr, ndtri

CONDS = ["L: x<0", "L: 0<x<.5S", "L: x<.5S", "L: .5S<x<.7S", "L: .7S<x<S", "N: S<x<M"]


def interval(cond, S, M):
    ninf = np.full_like(S, -np.inf)
    return {"L: x<0": (ninf, np.zeros_like(S)), "L: 0<x<.5S": (np.zeros_like(S), 0.5 * S), "L: x<.5S": (ninf, 0.5 * S),
            "L: .5S<x<.7S": (0.5 * S, 0.7 * S), "L: .7S<x<S": (0.7 * S, S), "N: S<x<M": (S, M)}[cond]


def trunc_normal(lo, hi, rng):
    """N(0,1) conditioned on lo < x < hi, elementwise (upper-tail intervals are reflected for accuracy)."""
    flip = lo >= 0
    a, b = np.where(flip, -hi, lo), np.where(flip, -lo, hi)
    pa, pb = ndtr(a), ndtr(b)
    xs = np.clip(ndtri(pa + (pb - pa) * rng.random(lo.shape)), a, b)
    return np.where(flip, -xs, xs)


def softmax(lg):
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def margin(lg, label):
    own = np.take_along_axis(lg, label[:, None], 1)[:, 0]
    other = lg.copy(); np.put_along_axis(other, label[:, None], -np.inf, 1)
    return own - other.max(1)


if __name__ == "__main__":
    rec = {(c, k): [] for c in CONDS for k in range(8)}       # rows: dx, D, Dc, dmargin, ans_changed, label_changed, ok0, ok1, gateflip_later, margin0
    trace = {c: [] for c in CONDS}
    for bi, (seed, x, H, Z) in enumerate(batches()):
        rng = np.random.default_rng(2000 + bi)
        ev = events(x)
        lg0 = logits(H); lab0 = second_argmax(x); ans0 = lg0.argmax(1)
        G0 = Z > 0
        for t in range(2, T):
            S, M = ev["Sp"][:, t], ev["Mp"][:, t]
            for c in CONDS:
                lo, hi = interval(c, np.where(S > 0, S, 1.0), M)
                idx = np.flatnonzero((S > 0) & (x[:, t] > lo) & (x[:, t] < hi))
                xn = x[idx].copy()
                xn[:, t] = trunc_normal(lo[idx], hi[idx], rng)
                assert ((xn[:, t] > lo[idx]) | (xn[:, t] == lo[idx])).all() and (xn[:, t] <= hi[idx]).all()
                Hn, Zn = run(xn, return_pre=True)
                lg1 = logits(Hn); lab1 = second_argmax(xn); ans1 = lg1.argmax(1)
                if c[0] == "L":
                    assert np.array_equal(lab1, lab0[idx])
                d = lg1 - lg0[idx]
                D = np.abs(d).max(1)
                Dc = np.abs(d - d.mean(1, keepdims=True)).max(1)
                m0, m1 = margin(lg0[idx], lab0[idx]), margin(lg1, lab1)
                later = ((Zn > 0) != G0[idx])[:, t + 1:].any((1, 2)) if t < T - 1 else np.zeros(len(idx), bool)
                tv = 0.5 * np.abs(softmax(lg1) - softmax(lg0[idx])).sum(1)
                g5 = G0[idx, t + 1, 5] if t < T - 1 else np.full(len(idx), np.nan)
                rec[(c, T - 1 - t)].append(np.stack([np.abs(xn[:, t] - x[idx, t]), D, Dc, m1 - m0, ans1 != ans0[idx], lab1 != lab0[idx],
                                                     ans0[idx] == lab0[idx], ans1 == lab1, later, m0, tv, g5], 1))
                if t <= 6:                                 # trace in the hidden state 0..3 steps later
                    dh = Hn[:, t:t + 4] - H[idx, t:t + 4]
                    dx = np.abs(xn[:, t] - x[idx, t])[:, None]
                    trace[c].append(np.concatenate([np.linalg.norm(dh[:, :, ORD], axis=2) / dx, np.linalg.norm(dh[:, :, PULSE], axis=2) / dx], 1))
    rec = {k: np.concatenate(v) for k, v in rec.items()}
    pc = lambda a, ps=(50, 90, 99): " / ".join(f"{v:.3f}" for v in np.percentile(a, ps))

    def table(title, cols, fn):
        print(f"\n### {title}\n")
        rows = []
        for c in CONDS:
            rows.append([c] + [fn(rec[(c, k)]) for k in range(8)] + [fn(np.concatenate([rec[(c, k)] for k in range(8)]))])
        print(md_table(["condition"] + [f"k={k} (t={9 - k})" for k in range(8)] + ["all"], rows))

    print("## A6 resampling one input within its class; k = number of steps between the input and the end (k = 9 - t)\n")
    table("n (resampled inputs)", None, lambda r: len(r))
    table("median |change of the input|", None, lambda r: f"{np.median(r[:, 0]):.3f}")
    table("max_i |change of logit i|: 50% / 90% / 99%", None, lambda r: pc(r[:, 1]))
    table("max_i |change of logit i - mean change| (softmax-relevant part): 50% / 90% / 99%", None, lambda r: pc(r[:, 2]))
    table("softmax-relevant change per unit input change, median of Dc / |dx|", None, lambda r: f"{np.median(r[:, 2] / r[:, 0]):.3f}")
    table("|change of the margin| (margin = logit of the true answer - best other logit): 50% / 90% / 99%", None, lambda r: pc(np.abs(r[:, 3])))
    table("mean change of the margin", None, lambda r: f"{r[:, 3].mean():+.3f}")
    table("P(network answer changes)", None, lambda r: f"{r[:, 4].mean():.4f}")
    table("P(true answer changes)", None, lambda r: f"{r[:, 5].mean():.4f}")
    table("P(network answer changes | true answer unchanged)", None, lambda r: f"{r[r[:, 5] == 0, 4].mean():.4f}")
    table("accuracy before -> after", None, lambda r: f"{r[:, 6].mean():.4f} -> {r[:, 7].mean():.4f}")
    table("P(correct before and wrong after) / P(wrong before and correct after)", None,
          lambda r: f"{((r[:, 6] == 1) & (r[:, 7] == 0)).mean():.4f} / {((r[:, 6] == 0) & (r[:, 7] == 1)).mean():.4f}")
    table("P(some gate at a later step differs)", None, lambda r: f"{r[:, 8].mean():.4f}")
    table("total-variation distance between the output distributions softmax(logits) before and after: 50% / 90% / 99%", None, lambda r: pc(r[:, 10]))
    table("P(TV distance > 0.01) / P(TV distance > 0.1)", None, lambda r: f"{(r[:, 10] > 0.01).mean():.4f} / {(r[:, 10] > 0.1).mean():.4f}")
    table("P(unit 5 on at t+1) in the original run", None, lambda r: f"{np.nanmean(r[:, 11]):.4f}" if not np.isnan(r[:, 11]).all() else "-")
    table("margin before the change: 25% / 50% / 75%", None, lambda r: pc(r[:, 9], (25, 50, 75)))
    table("softmax-relevant change when no later gate differs: 50% / 90% / 99%", None, lambda r: pc(r[r[:, 8] == 0, 2]) if (r[:, 8] == 0).sum() else "-")
    table("P(network answer changes | no later gate differs)", None, lambda r: f"{r[r[:, 8] == 0, 4].mean():.4f}" if (r[:, 8] == 0).sum() else "-")

    print("\n### trace in the hidden state j steps after the resampled input (t <= 6): median of |dh_(t+j)| / |dx|\n")
    rows = []
    for c in CONDS:
        tr = np.concatenate(trace[c])
        rows.append([c] + [f"{np.median(tr[:, j]):.3f}" for j in range(4)] + [f"{np.median(tr[:, 4 + j]):.3f}" for j in range(4)]
                    + [f"{np.percentile(tr[:, 1], 90):.3f}", f"{np.percentile(tr[:, 5], 90):.3f}"])
    print(md_table(["condition"] + [f"13 ordinary units, j={j}" for j in range(4)] + [f"pulses 4,5,9, j={j}" for j in range(4)]
                   + ["ordinary j=1, 90%", "pulses j=1, 90%"], rows))
