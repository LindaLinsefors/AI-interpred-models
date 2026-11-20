"""B7. Follow-up checks for B1, B4 and B6.
(H) free-flow formula for theta_4 after an isolated record, against the real impulse response.
(A) closed loop: theta_4 after a record in the network vs. in the oracle-patched run.
(B) level perturbation: do answers change without any gate change? which gates change?
(C) level perturbation with clipping to >= 0 (physical states) for f < 1.
(D) reset-prefix experiment restricted to sequences with >= 2 positive inputs.
(E) accuracy with two positive inputs as a function of S/M; with one positive input as a function of S/M (continuity through 0).
Seeds: 1070..1075.
"""
from blib import *

# ------------------------------------------------------------------ (H)
print("=== (H) [EXACT] free-flow theta_4 after an isolated record: mode coordinates of the impulse at age 2, propagated with lam^k ===")
xs = np.r_[1.0, np.zeros(9)]
H = run(xs)[0]
m2 = coords(H[2])
print("  mode coordinates / 10.157 at age 1 and age 2 of the impulse (age 1 would be exactly 1 for a pure kick):")
m1 = coords(H[1])
for n in NAMES:
    a1_, a2_ = m1[n] / W4, m2[n] / W4
    if IS_COMPLEX[n]:
        print(f"   {n:7s}: age 1: {abs(a1_):.4f} @ {np.degrees(np.angle(a1_)):7.2f} deg;  age 2: {abs(a2_):.4f} @ {np.degrees(np.angle(a2_)):7.2f} deg")
    else:
        print(f"   {n:7s}: age 1: {a1_:+.4f};  age 2: {a2_:+.4f}")
print("  age | actual theta_4 | free flow from age 2 | level+runner part | clock part | actual with background -1 | free flow, clock removed")
Hb = run(np.r_[1.0, -np.ones(9)])[0]
for age in range(2, 9):
    k = age - 2
    tot, lr, ck = 0.0, 0.0, 0.0
    for n in NAMES:
        mk = m2[n] * LAM[n] ** k
        c = 2 * np.real(C4[n] * mk) if IS_COMPLEX[n] else (C4[n] * mk).real
        tot += c
        if n in ("level", "runner"):
            lr += c
        if n == "clock":
            ck += c
    print(f"   {age} | {theta4(H[age]):.4f} | {tot:.4f} | {lr:.4f} | {ck:+.4f} | {theta4(Hb[age]):.4f} | {tot-ck:.4f}")

# ------------------------------------------------------------------ (A)
print("\n=== (A) closed loop: threshold after a record, network vs oracle-patched run  [INTERVENTION, n = 400k, seed 1070] ===")
x = sample(400_000, 1070)
n = len(x)
st = prefix_stats(x)
Hn = run(x)


def run_oracle(x):
    B = len(x); h = np.zeros((B, N)); Mp = np.zeros(B); Hh = np.empty((B, 10, N))
    for t in range(10):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        h[:, 4] = W4 * pos(x[:, t] - Mp)
        Hh[:, t] = h
        Mp = np.maximum(Mp, x[:, t])
    return Hh


Ho = run_oracle(x)
M, a = st["M1"], st["a1"]
Mold = pos(np.where(np.isfinite(st["M1prev"]), st["M1prev"], 0))
aold = st["a1prev"]
thn = theta4(Hn[:, :-1]); tho = theta4(Ho[:, :-1])           # thresholds for x_1..x_9 from states 0..8
print("theta_4 for the next input, relative to the standing max: mean of (theta_4 - M)/M by age of the max, M > 0")
print("age | network | oracle-patched run")
for age in range(9):
    s = (a[:, :-1] == age) & (M[:, :-1] > 0.2)
    print(f"{age} | {np.mean((thn[s] - M[:, :-1][s]) / M[:, :-1][s]):+.4f} | {np.mean((tho[s] - M[:, :-1][s]) / M[:, :-1][s]):+.4f}")
print("\nrecords at step s over a positive old max (M_old > 0.2): threshold two steps later, theta_4(s+2) - M_new, against the sag before the record, sag = M_old - theta_4(s)")
print("(network: slope ~ 0 means the record resets the threshold whatever the sag; oracle run: slope -1 would mean the sag is carried over)")
rec = (a == 0) & (np.arange(10)[None, :] >= 1) & (np.arange(10)[None, :] <= 7) & (Mold > 0.2)
ii, ss = np.where(rec)
keep = a[ii, ss + 1] == 1
ii, ss = ii[keep], ss[keep]
for name, Hx in (("network", Hn), ("oracle run", Ho)):
    sag = Mold[ii, ss] - theta4(Hx[ii, ss - 1])
    after1 = theta4(Hx[ii, ss]) - M[ii, ss]                  # threshold for x_{s+1}
    after2 = theta4(Hx[ii, ss + 1]) - M[ii, ss]              # threshold for x_{s+2}
    for lab, y in (("theta_4(s+1) - M_new", after1), ("theta_4(s+2) - M_new", after2)):
        A_ = np.column_stack([sag, M[ii, ss], Mold[ii, ss]])
        b, *_ = np.linalg.lstsq(A_, y, rcond=None)
        r2 = 1 - ((y - A_ @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {name:10s} {lab}: = {b[0]:+.4f}*sag {b[1]:+.4f}*M_new {b[2]:+.4f}*M_old  (R2c {r2:.4f}); mean {y.mean():+.4f}; mean sag {sag.mean():.4f}")

# ------------------------------------------------------------------ (B), (C)
print("\n=== (B) level perturbation: which gates change, and do answers change without gate changes?  [INTERVENTION, n = 400k, seed 1071] ===")
x = sample(400_000, 1071)
n = len(x)
Hn, Zn = run(x, return_pre=True)
base_ans = logits(Hn).argmax(1)
sa = second_argmax(x)
L, R = LEFT["level"].real, RIGHT["level"].real


def perturb(h, f, clip=False):
    h2 = h.copy()
    h2[:, ORD] += ((f - 1) * (h[:, ORD] @ L))[:, None] * R[None, :]
    return np.maximum(h2, 0) if clip else h2


print("t0 | f | P(answer changed) | P(no gate change at all after t0) | P(answer changed | no gate change) | of changed answers: unit-4 gate changed | comparator gate changed | pulse 5/9 gate changed | wave gate changed | "
      "P(changed | only wave gates changed) | n(only wave)")
for t0 in (3, 5, 7):
    for f in (0.8, 0.9, 1.1, 1.2):
        Hs, Zs = run(x[:, t0 + 1:], return_pre=True, h0=perturb(Hn[:, t0], f))
        ans = (Hs[:, -1] @ U.T).argmax(1)
        ch = ans != base_ans
        dg = (Zs > 0) != (Zn[:, t0 + 1:] > 0)                # (n, steps, 16)
        g4 = dg[:, :, 4].any(1); gc = dg[:, :, COMP].any((1, 2)); gp = dg[:, :, [5, 9]].any((1, 2)); gw = dg[:, :, WAVE].any((1, 2))
        none = ~(g4 | gc | gp | gw)
        onlyw = gw & ~(g4 | gc | gp)
        print(f"{t0} | {f} | {ch.mean():.4f} | {none.mean():.4f} | {ch[none].mean():.4f} | {g4[ch].mean():.3f} | {gc[ch].mean():.3f} | {gp[ch].mean():.3f} | {gw[ch].mean():.3f} | "
              f"{ch[onlyw].mean():.4f} | {onlyw.sum()}")

print("\n--- which step's comparator gates change first, relative to the arrival of the final runner-up (t0 = 3, f = 1.2 and 0.8) ---")
t0 = 3
srt = np.argsort(x, axis=1)
q, pmax = srt[:, -2], srt[:, -1]
for f in (1.2, 0.8):
    Hs, Zs = run(x[:, t0 + 1:], return_pre=True, h0=perturb(Hn[:, t0], f))
    ans = (Hs[:, -1] @ U.T).argmax(1)
    ch = ans != base_ans
    print(f"  f = {f}: P(answer changed) by where the true runner-up q and the max p sit relative to t0:")
    for name, s in (("q <= t0, p <= t0 (both already seen)", (q <= t0) & (pmax <= t0)), ("q > t0, p <= t0 (runner-up arrives later)", (q > t0) & (pmax <= t0)),
                    ("q <= t0, p > t0 (max arrives later)", (q <= t0) & (pmax > t0)), ("q > t0, p > t0", (q > t0) & (pmax > t0))):
        c = ch & s
        print(f"    {name}: share of sequences {s.mean():.3f}; P(changed) {ch[s].mean():.4f}; of the changed: new answer = argmax {(ans[c] == pmax[c]).mean():.3f}, "
              f"new = 3rd largest {(ans[c] == srt[c, -3]).mean():.3f}; base acc {(base_ans[s] == sa[s]).mean():.3f} -> {(ans[s] == sa[s]).mean():.3f}")

print("\n=== (C) the same perturbation followed by clipping negative activations to 0 (physical states), compared with the unclipped one ===")
print("t0 | f | P(answer changed): unclipped | clipped | accuracy: unclipped | clipped | level ratio actually achieved after clipping (median)")
for t0 in (3, 5, 7):
    for f in (0.8, 0.9):
        out = []
        for clip in (False, True):
            hp = perturb(Hn[:, t0], f, clip)
            Hs = run(x[:, t0 + 1:], h0=hp)
            ans = (Hs[:, -1] @ U.T).argmax(1)
            out.append(((ans != base_ans).mean(), (ans == sa).mean(), np.median((hp[:, ORD] @ L) / (Hn[:, t0][:, ORD] @ L))))
        print(f"{t0} | {f} | {out[0][0]:.4f} | {out[1][0]:.4f} | {out[0][1]:.4f} | {out[1][1]:.4f} | {out[1][2]:.4f}")

# ------------------------------------------------------------------ (D)
print("\n=== (D) reset of the state before the first positive input, sequences with >= 2 positive inputs (runner-up positive)  [INTERVENTION, n = 100k per s, seed 1072] ===")
print("s | n with >= 2 positives | answer agreement | acc actual | acc reset || exactly 1 positive: n | agreement | acc actual | acc reset")
rng = np.random.default_rng(1072)
for s_ in range(1, 9):
    xx = rng.standard_normal((100_000, 10))
    xx[:, :s_] = -np.abs(xx[:, :s_]); xx[:, s_] = np.abs(xx[:, s_])
    aa = logits(run(xx)).argmax(1)
    ar = (run(xx[:, s_:])[:, -1] @ U.T).argmax(1)
    sa_ = second_argmax(xx)
    two = (xx > 0).sum(1) >= 2
    one = ~two
    print(f"{s_} | {two.sum()} | {(aa[two] == ar[two]).mean():.4f} | {(aa[two] == sa_[two]).mean():.4f} | {(ar[two] == sa_[two]).mean():.4f} || "
          f"{one.sum()} | {(aa[one] == ar[one]).mean():.4f} | {(aa[one] == sa_[one]).mean():.4f} | {(ar[one] == sa_[one]).mean():.4f}")

# ------------------------------------------------------------------ (E)
print("\n=== (E) accuracy against S/M through zero  [MEASURED; one positive: seed 1073, two positives: seed 1074, 400k each] ===")
def cond_sample(n, k, seed):
    rng = np.random.default_rng(seed)
    mag = np.abs(rng.standard_normal((n, 10)))
    order = np.argsort(rng.random((n, 10)), axis=1)
    return mag * np.where(order < k, 1.0, -1.0)
rows = []
for k, seed in ((1, 1073), (2, 1074)):
    xx = cond_sample(400_000, k, seed)
    ans = logits(run(xx)).argmax(1)
    srt = np.sort(xx, 1)
    r = srt[:, -2] / srt[:, -1]
    ok = ans == second_argmax(xx)
    edges = [-np.inf, -1, -0.5, -0.2, -0.1, -0.05, -0.02, -0.01, 0] if k == 1 else [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (r >= lo) & (r < hi)
        rows.append((k, lo, hi, s.sum(), ok[s].mean()))
print("positives | S/M bin | n | accuracy")
for k, lo, hi, cnt, acc in rows:
    print(f"{k} | [{lo}, {hi}) | {cnt} | {acc:.4f}")
