"""A6 extra [INTERVENTION]: resample ALL low inputs of a sequence at once.

An input x_p (p >= 2) is 'low' if S_{p-1} > 0 and x_p < c * S_{p-1} (S_{p-1} = second largest of x_0..x_{p-1}).
Every low input is replaced by an independent N(0,1) draw conditioned on x < c * S_{p-1}; the running top two are
unchanged, so the true answer is unchanged.  Variants: c = 0 (negative inputs), c = 0.5, c = 0.7, c = 1 (every input
below the runner-up).  Resampling seeds 3000 + batch index.
"""
from cc import *
from a06_resample import trunc_normal, margin, softmax

VAR = [("x < 0", 0.0), ("x < 0.5 S", 0.5), ("x < 0.7 S", 0.7), ("x < S (all of class L)", 1.0)]
out = {name: [] for name, _ in VAR}
gate9 = {name: [] for name, _ in VAR}
for bi, (seed, x, H, Z) in enumerate(batches()):
    rng = np.random.default_rng(3000 + bi)
    ev = events(x)
    lab = second_argmax(x); lg0 = logits(H); ans0 = lg0.argmax(1); m0 = margin(lg0, lab)
    for name, c in VAR:
        Sp = ev["Sp"]
        mask = (Sp > 0) & (x < c * np.where(Sp > 0, Sp, 1))
        mask[:, :2] = False
        xn = x.copy()
        hi = c * Sp[mask]
        xn[mask] = trunc_normal(np.full(hi.shape, -np.inf), hi, rng)
        evn = events(xn)
        assert np.array_equal(second_argmax(xn), lab)
        assert all(np.array_equal(ev[k][:, 2:], evn[k][:, 2:]) for k in ("M", "S", "pM", "pS"))
        Hn, Zn = run(xn, return_pre=True)
        lg1 = logits(Hn); ans1 = lg1.argmax(1)
        tv = 0.5 * np.abs(softmax(lg1) - softmax(lg0)).sum(1)
        out[name].append(np.stack([mask.sum(1), ans1 != ans0, ans0 == lab, ans1 == lab, np.abs(margin(lg1, lab) - m0), tv,
                                   np.abs(lg1 - lg0)[:, :9].max(1), mask[:, 9]], 1))
        gate9[name].append((Zn[:, 9] > 0) == (Z[:, 9] > 0))
print("## A6.x all low inputs of a sequence resampled at once (2^20 sequences)\n")
rows = []
for name, _ in VAR:
    r = np.concatenate(out[name]); g = np.concatenate(gate9[name])
    has = r[:, 0] > 0
    q = lambda v, ps=(50, 90, 99): " / ".join(f"{p:.3f}" for p in np.percentile(v, ps))
    rows.append([name, f"{r[:, 0].mean():.2f}", f"{has.mean():.4f}", f"{r[has, 1].mean():.4f}", f"{r[has, 2].mean():.4f} -> {r[has, 3].mean():.4f}",
                 q(r[has, 4]), q(r[has, 5]), f"{(r[has, 5] > 0.1).mean():.4f}", q(r[has & (r[:, 7] == 0), 6]), f"{g[has].all(1).mean():.4f}"])
print(md_table(["low = ", "mean number resampled", "share of sequences with >= 1", "P(answer changes)", "accuracy before -> after",
                "abs. change of margin 50/90/99%", "TV distance of softmax 50/90/99%", "P(TV > 0.1)",
                "max abs. change of logits 0..8 when x_9 is not resampled, 50/90/99%", "P(all 16 gates at t=9 unchanged)"], rows))
print("\nP(answer changes) by the number of resampled inputs\n")
rows = []
for name, _ in VAR:
    r = np.concatenate(out[name])
    rows.append([name] + [f"{r[r[:, 0] == k, 1].mean():.4f} (n={int((r[:, 0] == k).sum())})" if (r[:, 0] == k).sum() > 2000 else "-" for k in range(1, 9)])
print(md_table(["low = "] + [str(k) for k in range(1, 9)], rows))
print("\nper-unit agreement of the gate at t = 9 (sequences with >= 1 resampled input)\n")
rows = []
for name, _ in VAR:
    r = np.concatenate(out[name]); g = np.concatenate(gate9[name]); has = r[:, 0] > 0
    rows.append([name] + [f"{g[has, i].mean():.4f}" for i in range(N)])
print(md_table(["low = "] + [f"u{i}" for i in range(N)], rows))
