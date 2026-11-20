"""s08 (task F4). Interventions after every record: unit 5 off / unit 9 off / unit 0 on.

Record: x_s > 0 and x_s > max(x_0..x_{s-1})  (s = 0 counts when x_0 > 0: the net treats it as a record against h = 0).
'off' = activation set to 0;  'on' = the ReLU of the unit is removed (h = pre-activation, may be negative).
N(0,1), seeds 1060, 1061 (400k each).
"""
from fcommon import *


def run_clamped(x, off=None, lin=None):
    """off[unit] / lin[unit]: boolean (B, T) masks of steps at which the unit is forced off / linear."""
    B, Tn = x.shape
    h = np.zeros((B, N))
    for t in range(Tn):
        z = x[:, t, None] * w + h @ W.T
        h = np.maximum(z, 0.0)
        for u, m in (lin or {}).items():
            h[:, u] = np.where(m[:, t], z[:, u], h[:, u])
        for u, m in (off or {}).items():
            h[:, u] = np.where(m[:, t], 0.0, h[:, u])
    return h @ U.T


def shifted(rec, d):
    out = np.zeros_like(rec)
    out[:, d:] = rec[:, :-d]
    return out


res = {}
for seed in (1060, 1061):
    x = sample(400_000, seed)
    B = x.shape[0]
    pm = np.maximum.accumulate(x, axis=1)
    rec = np.zeros_like(x, bool)
    rec[:, 0] = x[:, 0] > 0
    rec[:, 1:] = (x[:, 1:] > pm[:, :-1]) & (x[:, 1:] > 0)
    r1, r2 = shifted(rec, 1), shifted(rec, 2)
    order = np.argsort(-x, axis=1)
    pmax, p2nd, p3rd = order[:, 0], order[:, 1], order[:, 2]
    H, Z = run(x, return_pre=True)
    base = (H[:, 9] @ U.T).argmax(1)
    # old max of the last record: argmax of x[:pmax]
    xm = np.where(np.arange(10)[None, :] < pmax[:, None], x, -np.inf)
    pold = np.where(pmax > 0, xm.argmax(1), -1)
    relabel = (p2nd == pold)                       # the true answer is the old max of the final record
    variants = {
        "unit 5 off @ s+1": dict(off={5: r1}), "unit 5 off @ s+2": dict(off={5: r2}), "unit 5 off @ s+1,s+2": dict(off={5: r1 | r2}),
        "unit 9 off @ s+1": dict(off={9: r1}), "unit 9 off @ s+2": dict(off={9: r2}), "unit 9 off @ s+1,s+2": dict(off={9: r1 | r2}),
        "units 5,9 off @ s+1": dict(off={5: r1, 9: r1}),
        "unit 0 on @ s+1": dict(lin={0: r1}), "unit 0 on @ s+2": dict(lin={0: r2}), "unit 0 on @ s+1,s+2": dict(lin={0: r1 | r2}),
        "unit 0 off @ s+3 (control: off where it is normally on)": dict(off={0: shifted(rec, 3)}),
    }
    for name, kw in variants.items():
        ans = run_clamped(x, **kw).argmax(1)
        d = res.setdefault(name, {})
        for k, v in (("ans", ans), ("base", base), ("pmax", pmax), ("p2nd", p2nd), ("p3rd", p3rd), ("pold", pold), ("relabel", relabel),
                     ("nrec", rec.sum(1))):
            d.setdefault(k, []).append(v)
    # how often is the clamp a real change?
    st_ = res.setdefault("_stats", {})
    for k, v in (("u5_on_s1", (H[:, :, 5] > 0)[r1].mean()), ("u9_on_s1", (H[:, :, 9] > 0)[r1].mean()), ("u0_off_s2", (Z[:, :, 0] <= 0)[r2].mean()),
                 ("u0_off_s1", (Z[:, :, 0] <= 0)[r1].mean()), ("u5_on_s2", (H[:, :, 5] > 0)[r2].mean()), ("u9_on_s2", (H[:, :, 9] > 0)[r2].mean()),
                 ("recs_per_seq", rec.sum(1).mean())):
        st_.setdefault(k, []).append(v)

print("how often the clamp changes something (fraction of clamped steps):", {k: round(float(np.mean(v)), 4) for k, v in res["_stats"].items()})
first = True
for name, d in res.items():
    if name == "_stats":
        continue
    d = {k: np.concatenate(v) for k, v in d.items()}
    ans, base, pmax, p2nd, p3rd, pold, rl = (d[k] for k in ("ans", "base", "pmax", "p2nd", "p3rd", "pold", "relabel"))
    if first:
        print(f"n = {len(ans)}; unmodified network accuracy {np.mean(base == p2nd):.4f}; "
              f"on 'relabel' sequences (true answer = old max of the final record, {rl.mean():.3f} of all): {np.mean(base[rl] == p2nd[rl]):.4f}; others: {np.mean(base[~rl] == p2nd[~rl]):.4f}")
        first = False
    ch = ans != base
    c = ch & (base == p2nd)                      # changed and the unmodified answer was right
    print(f"\n--- {name} ---")
    print(f"  agreement with the network {1 - ch.mean():.4f};  task accuracy {np.mean(ans == p2nd):.4f} "
          f"(relabel sequences {np.mean(ans[rl] == p2nd[rl]):.4f}, others {np.mean(ans[~rl] == p2nd[~rl]):.4f})")
    if ch.sum() == 0:
        continue
    cat = {"the max": ans == pmax, "true 2nd": ans == p2nd, "3rd largest": ans == p3rd,
           "old max of final record (when not the true 2nd)": (ans == pold) & (pold != p2nd)}
    print(f"  changed answers ({ch.sum()}, {ch.mean():.4f}) go to: " + "; ".join(f"{k} {v[ch].mean():.3f}" for k, v in cat.items())
          + f"; position max+1 {np.mean(ans[ch] == pmax[ch] + 1):.3f}; max-1 {np.mean(ans[ch] == pmax[ch] - 1):.3f};"
          + f" neighbour (+-1) of the unmodified answer {np.mean(np.abs(ans[ch] - base[ch]) == 1):.3f} (+1: {np.mean(ans[ch] - base[ch] == 1):.3f}, -1: {np.mean(ans[ch] - base[ch] == -1):.3f})")
    print(f"  of the changed answers that were right before ({c.sum()}): to the max {np.mean(ans[c] == pmax[c]):.3f}; to the 3rd {np.mean(ans[c] == p3rd[c]):.3f};"
          f" to 2nd+1 {np.mean(ans[c] == p2nd[c] + 1):.3f}; to 2nd-1 {np.mean(ans[c] == p2nd[c] - 1):.3f}; share from relabel sequences {rl[c].mean():.3f}")
    dd = ans[ch] - base[ch]
    print("  shift new - old answer among changed: " + " ".join(f"{k:+d}:{np.mean(dd == k):.2f}" for k in range(-4, 5) if k != 0))
    dm = ans[ch] - pmax[ch]
    print("  new answer relative to the max position: " + " ".join(f"{k:+d}:{np.mean(dm == k):.2f}" for k in range(-4, 5)))
