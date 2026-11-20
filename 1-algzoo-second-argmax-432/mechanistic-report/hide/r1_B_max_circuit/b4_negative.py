"""B4. What does the network keep of negative inputs?
Seeds: 1041..1047 (conditional samples of N(0,1) with a prescribed number of positive inputs), 1048, 1049.
"""
from blib import *
from math import comb

def cond_sample(n, k, seed):
    """N(0,1)^10 conditioned on exactly k positive inputs (positions uniformly random)."""
    rng = np.random.default_rng(seed)
    mag = np.abs(rng.standard_normal((n, 10)))
    order = np.argsort(rng.random((n, 10)), axis=1)
    sign = np.where(order < k, 1.0, -1.0)
    return mag * sign

print("=== (a) accuracy by number of positive inputs k  [MEASURED, 200k sequences per k, seeds 1041+k] ===")
print("k | P(k) under N(0,1) | accuracy | error contribution P(k)*(1-acc) | P(answer = argmax) | P(answer = 3rd largest)")
tot = 0
res_k = {}
for k in range(0, 7):
    x = cond_sample(200_000, k, 1041 + k)
    H = run(x)
    ans = logits(H).argmax(1)
    srt = np.argsort(x, axis=1)
    acc = (ans == srt[:, -2]).mean()
    pk = comb(10, k) / 1024
    tot += pk * (1 - acc)
    res_k[k] = (x, H, ans, srt)
    print(f"{k} | {pk:.5f} | {acc:.4f} | {pk*(1-acc):.5f} | {(ans == srt[:, -1]).mean():.4f} | {(ans == srt[:, -3]).mean():.4f}")
print(f"  sum of error contributions for k = 0..6: {tot:.5f}  (k <= 1 only: {sum(comb(10,k)/1024*(1-(res_k[k][2]==res_k[k][3][:, -2]).mean()) for k in (0,1)):.5f})")

# ------------------------------------------------------------- (b) one positive input
x, H, ans, srt = res_k[1]
n = len(x)
p = srt[:, -1]                    # position of the positive input (the max)
q = srt[:, -2]                    # true answer: largest negative
M = x[np.arange(n), p]; S = x[np.arange(n), q]; T3 = x[np.arange(n), srt[:, -3]]
print("\n=== (b) exactly one positive input (the max at p, true answer q = largest negative)  [MEASURED n = 200k, seed 1042] ===")
print(f"  P(answer = q) = {(ans == q).mean():.4f};  P(answer = p) = {(ans == p).mean():.4f};  P(answer = p+1) = {(ans == p + 1).mean():.4f};  "
      f"P(answer = p-1) = {(ans == p - 1).mean():.4f};  P(answer = 9) = {(ans == 9).mean():.4f};  P(answer = 0) = {(ans == 0).mean():.4f}")
print("  chance level for q given p: 1/9 = 0.111")
print("  by position p of the max: accuracy | P(ans = p) | most common answer offsets (ans - p) with frequency")
for pp in range(10):
    s = p == pp
    off = ans[s] - pp
    vals, cnt = np.unique(off, return_counts=True)
    top = sorted(zip(cnt / s.sum(), vals), reverse=True)[:4]
    print(f"   p={pp}: {(ans[s] == q[s]).mean():.4f} | {(ans[s] == pp).mean():.4f} | " + ", ".join(f"{v:+d}: {c:.3f}" for c, v in top))
print("  accuracy by where q sits relative to p:")
for name, s in (("q < p (before the max)", q < p), ("q = p+1", q == p + 1), ("q = p+2", q == p + 2), ("q > p+2", q > p + 2), ("q = 9", q == 9)):
    print(f"   {name}: n = {s.sum()}, accuracy {(ans[s] == q[s]).mean():.4f}")
print("  accuracy by |S|/M (how negative the runner-up is, relative to the max):")
r = -S / M
for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, np.inf)):
    s = (r >= lo) & (r < hi)
    print(f"   |S|/M in [{lo}, {hi}): n = {s.sum()}, accuracy {(ans[s] == q[s]).mean():.4f}")
print("  accuracy by the gap between the two largest negatives (S - T3):")
for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, np.inf)):
    s = (S - T3 >= lo) & (S - T3 < hi)
    print(f"   S - 3rd in [{lo}, {hi}): n = {s.sum()}, accuracy {(ans[s] == q[s]).mean():.4f}")

# exact traces: how much of the final state / logits comes from the max, from the last input, from the other negatives
sub = np.arange(40_000)
Hs, J = traces(x[sub])
ps, qs = p[sub], q[sub]
contrib = J[:, 9] * x[sub][:, :, None]                 # (n, p, 16): term of x_p in h_9
lg = contrib @ U.T                                     # (n, p, 10): term of x_p in the logits
idx = np.arange(len(sub))
maxterm = lg[idx, ps]                                  # (n, 10)
lastterm = np.where((ps == 9)[:, None], 0, lg[:, 9])
mask = np.ones((len(sub), 10), bool); mask[idx, ps] = False; mask[:, 9] = False
negterm = (lg * mask[:, :, None]).sum(1)
assert np.allclose(maxterm + lastterm + negterm, logits(Hs), atol=1e-8)
sd = lambda a: np.median(a.std(1))
print("\n  exact split of the final logits (h_9 = sum_p J[9,p] x_p) for one-positive sequences (n = 40k): median over sequences of the sd across the 10 logits")
print(f"   term of the max: {sd(maxterm):.3f};  term of the last input x_9 (if not the max): {sd(lastterm):.3f};  all other negative inputs together: {sd(negterm):.3f};  total logits: {sd(logits(Hs)):.3f}")
hn = lambda a: np.median(np.linalg.norm(a, axis=-1))
hmax = contrib[idx, ps]; hlast = np.where((ps == 9)[:, None], 0, contrib[:, 9]); hneg = (contrib * mask[:, :, None]).sum(1)
print(f"   same split of ||h_9||: max {hn(hmax):.2f}, last input {hn(hlast):.2f}, other negatives {hn(hneg):.2f}, total {hn(Hs[:, 9]):.2f}")
tq = lg[idx, qs]                                       # term of the true runner-up
own = tq[idx, qs] - (tq.sum(1) - tq[idx, qs]) / 9
s = (qs != 9)
print(f"   term of the true (negative) runner-up x_q in its own logit minus its mean effect on the other logits, q != 9: median {np.median(own[s]):.3f}, "
      f"per unit of x_q: median {np.median((own / x[sub][idx, qs])[s]):.3f}")
print(f"   (for comparison: winning margin of the answer over the 2nd best logit, median {np.median(np.sort(logits(Hs), 1)[:, -1] - np.sort(logits(Hs), 1)[:, -2]):.3f})")

# ------------------------------------------------------------- (c) all-negative prefixes: what is kept
print("\n=== (c) all-negative sequences: what is kept  [MEASURED n = 200k, seed 1041; exact traces on the first 40k] ===")
x0, H0, ans0, srt0 = res_k[0]
Hs, J = traces(x0[:40_000])
Zs = run(x0[:40_000], return_pre=True)[1]
print("  median trace norm ||d h_9 / d x_p|| by p:", np.round(np.median(np.linalg.norm(J[:, 9], axis=-1), 0), 3), f"(current input: {np.linalg.norm(np.minimum(w, 0)):.2f} = |w restricted to w<0|)")
print("  P(unit 4 on) over all steps:", (Zs[:, :, 4] > 0).mean(), " P(unit 2 on) by t:", np.round((Zs[:, :, 2] > 0).mean(0), 3))
m = coords(H0)
print("  mode coordinates / 10.157 at t = 9 (mean, sd) vs. the same per unit of mean|x| = 0.798:")
for nme in ("level", "runner"):
    v = m[nme][:, 9] / W4
    print(f"   {nme}: mean {v.mean():+.4f}, sd {v.std():.4f}")
v = np.abs(m["clock"][:, 9]) / W4
print(f"   |clock|: mean {v.mean():.4f}, sd {v.std():.4f}")
th_terms = theta4_terms(H0)
print("  theta_4 for the next input after t all-negative inputs: mean of each term (x 1000)")
print("   t | theta4 | " + " | ".join(th_terms))
for t in (0, 2, 4, 6, 8):
    print(f"   {t+1} | {1000*theta4(H0[:, t]).mean():.2f} | " + " | ".join(f"{1000*th_terms[k][:, t].mean():.2f}" for k in th_terms))
lev = m["level"] / W4
print("  level/10.157 after t+1 all-negative inputs, regression on sum of |x| so far: ")
cs = np.cumsum(-x0, 1)
for t in (0, 2, 4, 6, 9):
    b, _, r = slope0(lev[:, t], [cs[:, t]])
    print(f"   t={t}: slope {b[0]:.5f} per unit of sum|x|, R2c {r:.3f}, mean level/10.157 = {lev[:, t].mean():.4f}")

# ------------------------------------------------------------- (d) first positive input after a negative prefix
print("\n=== (d) first positive input after an all-negative prefix: does the prefix matter?  [INTERVENTION: reset h_{s-1} := 0; n = 100k per s, seed 1048] ===")
print("s | theta_4(s) mean | h4(s)/(10.157 x_s) median | rel. state diff ||h_t - h'_t||/||h_t|| at t = s, s+1, s+2, 9 (median) | answer agreement | acc actual | acc reset | P(true answer in prefix)")
rng = np.random.default_rng(1048)
for s_ in range(1, 9):
    nn = 100_000
    xx = rng.standard_normal((nn, 10))
    xx[:, :s_] = -np.abs(xx[:, :s_]); xx[:, s_] = np.abs(xx[:, s_])
    Ha = run(xx)
    Hr = run(xx[:, s_:])                                 # prefix forgotten: start from h = 0 at step s
    Hr_full = np.concatenate([Ha[:, :s_] * 0, Hr], 1)
    d = lambda t: np.median(np.linalg.norm(Ha[:, t] - Hr_full[:, t], axis=1) / np.linalg.norm(Ha[:, t], axis=1))
    aa, ar = logits(Ha).argmax(1), (Hr[:, -1] @ U.T).argmax(1)
    sa = second_argmax(xx)
    ts = [s_, min(s_ + 1, 9), min(s_ + 2, 9), 9]
    print(f"{s_} | {theta4(Ha[:, s_ - 1]).mean():.4f} | {np.median(Ha[:, s_, 4] / (W4 * xx[:, s_])):.4f} | " + ", ".join(f"{d(t):.4f}" for t in ts)
          + f" | {(aa == ar).mean():.4f} | {(aa == sa).mean():.4f} | {(ar == sa).mean():.4f} | {(sa < s_).mean():.4f}")
    if s_ in (3, 6):
        inpre = sa < s_
        print(f"      when the true answer lies in the negative prefix (n = {inpre.sum()}): acc actual {(aa[inpre] == sa[inpre]).mean():.4f}, acc reset {(ar[inpre] == sa[inpre]).mean():.4f}; "
              f"otherwise: acc actual {(aa[~inpre] == sa[~inpre]).mean():.4f}, acc reset {(ar[~inpre] == sa[~inpre]).mean():.4f}, agreement {(aa[~inpre] == ar[~inpre]).mean():.4f}")

# ------------------------------------------------------------- (e) thresholds when the runner-up is negative
print("\n=== (e) thresholds when the runner-up is negative (states with M > 0 >= S), N(0,1), seed 1049, n = 400k ===")
x = sample(400_000, 1049)
H = run(x)
st = prefix_stats(x)
Mx, Sx, ax = st["M1"][:, :-1], st["M2"][:, :-1], st["a1"][:, :-1]
Sfin = np.where(np.isfinite(Sx), Sx, 0)
th4 = theta4(H[:, :-1])
thc = {j: -(H[:, :-1] @ W[j]) / w[j] for j in COMP}
print("regression through 0 of each threshold on (M, S) for states with M > 0 > S, by age of the max; if negative runner-ups were stored, the S coefficient would be near that of the S > 0 case")
print("age | n | theta_4: b_M, b_S | " + " | ".join(f"theta_{j}: b_M, b_S" for j in COMP) + " || same for M > S > 0: theta_4 | " + " | ".join(f"theta_{j}" for j in COMP))
for age in range(1, 9):
    out = []
    for cond in ((Mx > 0) & (Sx < 0) & np.isfinite(Sx), (Sx > 0)):
        s = (ax == age) & cond
        row = []
        for y in [th4] + [thc[j] for j in COMP]:
            b, _, r = slope0(y[s], [Mx[s], Sfin[s]])
            row.append(f"{b[0]:.3f}, {b[1]:+.3f}")
        out.append((s.sum(), row))
    print(f"{age} | {out[0][0]} | " + " | ".join(out[0][1]) + " || " + " | ".join(out[1][1]))
