"""Attack on I2 (clock before a record).  Seeds 5600.. (400k per batch), I2 selection.

 1. off-manifold: negative entries; repeat with the state clipped at zero
 2. the same interventions on the other modes; size-matched random perturbations
 3. the q0 / existence confounds in the rotation result; direction
 4. radius -> size of the promotion (logit margins)
 5. timing of the intervention (t_r-2, t_r-1, t_r, t_r+1)
 6. wave part vs comparator part of the clock perturbation
 7. shift every mode instead of the clock only
"""
from v3lib import *

out = Tee("a4_clock_attacks.txt")
x, tr, p0, q0, M0, S0, nb = gather_I2(250_000, 5600)
n = len(x)
out(f"seeds 5600..{5600 + nb - 1}; I2-selected n = {n}")
tint = tr - 1
r = np.arange(n)
H0 = run(x)
h_pre = H0[r, tint]                                      # the state that is intervened on
hn = np.linalg.norm(h_pre[:, ORD], axis=1)


def go(fn, tint=tint):
    h9, hiv = run_iv(x, tint, fn, return_state=True)
    return answer(h9), h9, hiv


def line(a):
    d = a - p0
    return (f"P(p0) = {np.mean(a == p0):.4f}  P(t_r) = {np.mean(a == tr):.4f}  P(q0) = {np.mean(a == q0):.4f}  "
            + " ".join(f"p0{j:+d}: {np.mean(d == j):.3f}" for j in (-2, -1, 1, 2)))


# ------------------------------------------------------------------ 1. clipping
out("\n[1] negative entries and clipping.  'neg' = fraction of intervened states with a negative entry; 'depth' = median over")
out("    those states of (most negative entry) / |h[ORD]| before the intervention")
for label, kw in ([(f"f = {f}", dict(f=f)) for f in (0, 0.25, 0.5, 0.75, 1.25, 1.5, 2, 3)]
                  + [(f"k = {k:+}", dict(k=k)) for k in (-2, -1, 1, 2)]):
    a, _, hiv = go(mode_fn("clock", **kw))
    neg = (hiv < -1e-9).any(1)
    depth = np.median(hiv.min(1)[neg] / hn[neg]) if neg.any() else 0.0
    ac, _, _ = go(mode_fn("clock", clip=True, **kw))
    out(f"  {label:9s} neg = {neg.mean():.3f} depth = {depth:+.3f} | raw:     {line(a)}")
    out(f"  {'':9s} {'':33s} | clipped: {line(ac)}")
    pos = ~neg
    if pos.sum() > 200:
        out(f"  {'':9s} {'':33s} | raw, only the states WITHOUT a negative entry (n = {pos.sum()}): P(p0) = {np.mean(a[pos] == p0[pos]):.4f}"
            f"  P(t_r) = {np.mean(a[pos] == tr[pos]):.4f}  p0-1: {np.mean(a[pos] == p0[pos] - 1):.3f}  p0+1: {np.mean(a[pos] == p0[pos] + 1):.3f}")

# ------------------------------------------------------------------ 2. other modes
out("\n[2] the same intervention on each mode (z' = f z; z' = z exp(i k angle_mode) for complex modes).")
out("    size = median |change of h| / |h[ORD]| for f = 0;  |z| = median |mode coordinate| / (10.157 M0)")
c0 = coords(h_pre)
for mode in NAMES:
    R = RIGHT[mode]
    z = c0[mode]
    d = (2 if IS_COMPLEX[mode] else 1) * np.real(np.multiply.outer(z, R))
    size = np.median(np.linalg.norm(d, axis=1) / hn)
    out(f"  {mode:7s} size = {size:.3f}  |z| = {np.median(np.abs(z) / (10.157 * M0)):.3f}")
    for f in (0, 0.5, 2):
        a, _, hiv = go(mode_fn(mode, f=f))
        ac, _, _ = go(mode_fn(mode, f=f, clip=True))
        out(f"     f = {f:<4}: raw {line(a)} | clipped P(p0) = {np.mean(ac == p0):.4f} P(t_r) = {np.mean(ac == tr):.4f}  neg = {np.mean((hiv < -1e-9).any(1)):.2f}")
    if IS_COMPLEX[mode]:
        for k in (-1, 1):
            a, _, hiv = go(mode_fn(mode, k=k))
            ac, _, _ = go(mode_fn(mode, k=k, clip=True))
            out(f"     k = {k:+}  : raw {line(a)} | clipped P(p0) = {np.mean(ac == p0):.4f} p0-1: {np.mean(ac == p0 - 1):.3f} p0+1: {np.mean(ac == p0 + 1):.3f}  neg = {np.mean((hiv < -1e-9).any(1)):.2f}")

out("\n    size-matched random perturbations of h[ORD] (same norm as removing the clock mode, per state):")
rng = np.random.default_rng(5699)
zc = c0["clock"]
dclock = 2 * np.real(np.multiply.outer(zc, RIGHT["clock"]))
dn = np.linalg.norm(dclock, axis=1)


def add_fn(delta, clip=False):
    def fn(h, rows):
        h = h.copy()
        h[:, ORD_A] += delta[rows]
        return np.maximum(h, 0.0) if clip else h
    return fn


g = rng.standard_normal((n, 13)); g *= (dn / np.linalg.norm(g, axis=1))[:, None]
a, _, _ = go(add_fn(g)); out(f"     random Gaussian direction, all 13 units:        {line(a)}")
a, _, _ = go(add_fn(g, clip=True)); out(f"     same, clipped at 0:                             {line(a)}")
isC = np.isin(ORD_A, COMP)
g2 = rng.standard_normal((n, 13)) * isC; g2 *= (dn / np.linalg.norm(g2, axis=1))[:, None]
a, _, _ = go(add_fn(g2, clip=True)); out(f"     random direction on the comparators, clipped:   {line(a)}")
perm = rng.permutation(n)
g3 = -dclock[perm] * (dn / dn[perm])[:, None]
a, _, _ = go(add_fn(g3)); out(f"     minus ANOTHER sequence's clock part (rescaled):  {line(a)}")
a, _, _ = go(add_fn(-dclock)); out(f"     minus its own clock part (= f 0, for reference): {line(a)}")
# random phase: z' = z exp(i phi), phi uniform
phi = rng.uniform(0, 2 * np.pi, n)
g4 = 2 * np.real(np.multiply.outer(zc * (np.exp(1j * phi) - 1), RIGHT["clock"]))
a, _, _ = go(add_fn(g4)); out(f"     clock phase randomised (radius kept):            {line(a)}")
g5 = 2 * np.real(np.multiply.outer(zc * (np.exp(1j * np.pi) - 1), RIGHT["clock"]))
a, _, _ = go(add_fn(g5)); out(f"     clock phase + 180 deg (f = -1):                  {line(a)}")

# ------------------------------------------------------------------ 3. confounds in the rotation result
out("\n[3] rotation z' = z exp(i k omega): does the answer go to p0 - k?  Conditioned on that position existing,")
out("    and split by whether it coincides with q0 (the prefix runner-up, a natural fallback answer).")
age = tr - 1 - p0
for k in (-2, -1, 1, 2):
    a, _, _ = go(mode_fn("clock", k=k))
    ac, _, _ = go(mode_fn("clock", k=k, clip=True))
    tgt = p0 - k
    ex = (tgt >= 0)
    isq = tgt == q0
    out(f"  k = {k:+}: target p0{-k:+d} exists in {ex.mean():.3f}; P(ans = target | exists) = {np.mean(a[ex] == tgt[ex]):.4f}"
        f" (clipped {np.mean(ac[ex] == tgt[ex]):.4f});  | target = q0 (n={np.sum(ex & isq)}): {np.mean(a[ex & isq] == tgt[ex & isq]):.4f}"
        f";  | target != q0 (n={np.sum(ex & ~isq)}): {np.mean(a[ex & ~isq] == tgt[ex & ~isq]):.4f} (clipped {np.mean(ac[ex & ~isq] == tgt[ex & ~isq]):.4f})")
    m = ex & ~isq
    d = (a - p0)[m]
    out(f"         among target != q0: " + " ".join(f"p0{j:+d}: {np.mean(d == j):.3f}" for j in range(-3, 4))
        + f"  t_r: {np.mean(a[m] == tr[m]):.3f}  q0: {np.mean(a[m] == q0[m]):.3f}")
    # where is x at the target position in the ranking of the prefix? (it is an ordinary small input)
    xt = x[r[m], tgt[m]]
    out(f"         the input at the target position: median x/M0 = {np.median(xt / M0[m]):+.2f}; P(x < 0) = {np.mean(xt < 0):.2f};"
        f" P(ans = target | x_target < 0) = {np.mean(a[m][xt < 0] == tgt[m][xt < 0]):.4f}")
    out("         by age of the old max (2..7): P(ans = target | exists, != q0) = "
        + " ".join(f"{np.mean(a[m & (age == g_)] == tgt[m & (age == g_)]):.3f}" for g_ in range(2, 8)))
out("  signed displacement, answers restricted to old positions (ans < t_r) other than q0: mean(ans - p0) and P(<0), P(>0)")
for k in (-2, -1, -0.5, 0.5, 1, 2):
    a, _, _ = go(mode_fn("clock", k=k))
    m = (a < tr) & (a != q0)
    d = (a - p0)[m]
    out(f"  k = {k:+}: n = {m.sum()} ({m.mean():.3f})  mean = {d.mean():+.3f}  P(ans < p0) = {np.mean(d < 0):.4f}  P(ans > p0) = {np.mean(d > 0):.4f}")

# ------------------------------------------------------------------ 4. radius -> promotion
out("\n[4] logit margins at t = 9 as a function of f (units of M0): median of s[p0] - s[t_r], of s[p0] - s[q0], and of")
out("    s[p0] - max over the other nine positions; raw | clipped")
for f in (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3):
    row = []
    for clip in (False, True):
        a, h9, _ = go(mode_fn("clock", f=f, clip=clip))
        s = h9 @ U.T
        sp = s[r, p0]
        so = s.copy(); so[r, p0] = -np.inf
        row.append(f"{np.median((sp - s[r, tr]) / M0):+7.2f} {np.median((sp - s[r, q0]) / M0):+7.2f} {np.median((sp - so.max(1)) / M0):+7.2f}  P(p0) = {np.mean(a == p0):.3f}")
    out(f"  f = {f:<4}: " + "  |  ".join(row))

# ------------------------------------------------------------------ 5. timing
out("\n[5] the same clock interventions at other steps (lead: t_r - 1).  P(p0), P(t_r), P(q0); |z| = median |z_clock| / (10.157 M0) there")
for j in (-2, -1, 0, 1, 2):
    ti = tr - 1 + j
    ok = (ti <= 9) & (ti - p0 >= 1)
    zz = np.abs(coords(H0[r, np.minimum(ti, 9)])["clock"]) / (10.157 * M0)
    for label, kw in (("f = 0", dict(f=0)), ("f = 0.5", dict(f=0.5)), ("k = +1", dict(k=1)), ("k = -1", dict(k=-1))):
        h9 = run_iv(x[ok], ti[ok], mode_fn("clock", **kw))
        a = answer(h9)
        out(f"  step t_r{j - 1:+d} (n={ok.sum()}, |z| = {np.median(zz[ok]):.2f}) {label:8s}: P(p0) = {np.mean(a == p0[ok]):.4f}  P(t_r) = {np.mean(a == tr[ok]):.4f}"
            f"  P(q0) = {np.mean(a == q0[ok]):.4f}  p0-1: {np.mean(a == p0[ok] - 1):.3f}  p0+1: {np.mean(a == p0[ok] + 1):.3f}")

# ------------------------------------------------------------------ 6. wave part vs comparator part
out("\n[6] apply the clock change only to the wave units / only to the comparators (at t_r - 1)")
for label, kw in (("f = 0", dict(f=0)), ("f = 0.5", dict(f=0.5)), ("f = 2", dict(f=2)), ("k = +1", dict(k=1)), ("k = -1", dict(k=-1))):
    for ulabel, units in (("all 13", None), ("wave only", WAVE), ("comparators only", COMP)):
        a, _, hiv = go(mode_fn("clock", units=units, **kw))
        ac, _, _ = go(mode_fn("clock", units=units, clip=True, **kw))
        out(f"  {label:8s} {ulabel:17s}: {line(a)}  | clipped P(p0) = {np.mean(ac == p0):.4f} p0-1: {np.mean(ac == p0 - 1):.3f} p0+1: {np.mean(ac == p0 + 1):.3f}  neg = {np.mean((hiv < -1e-9).any(1)):.2f}")
out("  comparators at the record step t_r (unperturbed): fraction of sequences with all four comparators off = "
    f"{np.mean((H0[r, tr][:, COMP] == 0).all(1)):.4f}; number of comparators on: " + " ".join(f"{c}: {np.mean((H0[r, tr][:, COMP] > 0).sum(1) == c):.3f}" for c in range(5)))
out("  wipe the comparators at t_r - 1 (I1-style, just before the record): " + line(go(scale_units(COMP, 0.0))[0]))
out("  wipe the comparators at t_r (the record step):                      " + line(go(scale_units(COMP, 0.0), tint=tr)[0]))

# ------------------------------------------------------------------ 7. shift every mode
out("\n[7] rotate several modes by k of their own steps (complex: exp(i k angle_n); alt: (-1)^k; real positive modes untouched)")


def multi_fn(modes, k, clip=False, decay=False):
    def fn(h, rows):
        h0 = h
        h = h.copy()
        for mode in modes:
            z = h0[:, ORD_A] @ LEFT[mode]
            lam = LAM[mode]
            fac = (lam ** k) if decay else (lam / abs(lam)) ** k
            if IS_COMPLEX[mode]:
                h[:, ORD_A] += 2 * np.real(np.multiply.outer(z * (fac - 1), RIGHT[mode]))
            else:
                h[:, ORD_A] += np.real(np.multiply.outer(np.real(z) * (np.real(fac) - 1), RIGHT[mode]))
        return np.maximum(h, 0.0) if clip else h
    return fn


osc = ["clock", "h2", "alt", "h3", "short"]
for k in (-1, 1):
    for label, modes in (("clock only", ["clock"]), ("h2 only", ["h2"]), ("short only", ["short"]), ("h3 only", ["h3"]), ("alt only", ["alt"]),
                         ("clock + short", ["clock", "short"]), ("clock + h2", ["clock", "h2"]), ("all 5 oscillating", osc),
                         ("the 4 others, not the clock", [m for m in osc if m != "clock"])):
        a, _, _ = go(multi_fn(modes, k))
        ac, _, _ = go(multi_fn(modes, k, clip=True))
        tgt = p0 - k
        m = (tgt >= 0) & (tgt != q0)
        out(f"  k = {k:+} {label:28s}: {line(a)}  | P(ans = p0{-k:+d} | exists, != q0) = {np.mean(a[m] == tgt[m]):.4f} (clipped {np.mean(ac[m] == tgt[m]):.4f})")
out("  one extra resting step for every mode, h[ORD] -> A h[ORD] (k = +1 with decay, all nine modes):")
a, _, _ = go(multi_fn(NAMES, 1, decay=True)); tgt = p0 - 1; m = (tgt >= 0) & (tgt != q0)
out(f"     {line(a)}  | P(ans = p0-1 | exists, != q0) = {np.mean(a[m] == tgt[m]):.4f}")
a, _, _ = go(multi_fn([m_ for m_ in NAMES if m_ != 'clock'], 1, decay=True))
out(f"     all modes except the clock: {line(a)}  | P(ans = p0-1 | exists, != q0) = {np.mean(a[m] == tgt[m]):.4f}")
a, _, _ = go(multi_fn(["clock"], 1, decay=True))
out(f"     the clock only (with decay): {line(a)}  | P(ans = p0-1 | exists, != q0) = {np.mean(a[m] == tgt[m]):.4f}")
