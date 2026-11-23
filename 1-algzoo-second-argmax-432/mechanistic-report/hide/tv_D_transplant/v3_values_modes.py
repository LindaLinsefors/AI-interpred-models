"""v3: claims 4 (order of two close old inputs), 5 (values as thresholds for a later input) and 8 (modes, dose).

Claim 5 estimator: pairs A, B that differ only in the VALUE of the runner-up (or of the maximum).  Transplant at step
T0, then a probe input z = v_A + u (v_B - v_A) at a later step and low inputs elsewhere.  If the hybrid (rest of A +
tape of B) behaves as if the value were v_A + f (v_B - v_A), then P(z counted as above it) crosses 1/2 at u = f:
f = 0 "read from the rest", f = 1 "read from the tape".  ATTACK 6: f split by configuration and by the probe's delay.
"""
from v_common import *

N = 500_000
ar_ = lambda n: np.arange(n)

# ---------------------------------------------------------------- claim 4
print("[claim 4] old third (age >= 2, more than 0.4 M below S, M - S > 0.25 M) set to S + d M in the donor B")
for t in (9, 6):
    x, st = quiet(N, t, 7100 + t)
    ok = (st["T"] < st["S"] - 0.4 * st["M"]) & (st["aT"] >= 2) & (st["M"] - st["S"] > 0.25 * st["M"])
    x, st = x[ok], sub(st, ok)
    n = len(x)
    rng = np.random.default_rng(7200 + t)
    hA = state(x, t)
    fut = lows(st["M"], 9 - t, rng)
    print(f"   t = {t}, n = {n}, " + ("read out at once" if t == 9 else "then three low inputs") + ";  P(answer = that input):")
    print("       d     B intact   rest A + tape B  [clipped]   rest B + tape A")
    for d in (-0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2):
        xB = x.copy(); xB[ar_(n), st["pT"]] = st["S"] + d * st["M"]
        hB = state(xB, t)
        p = lambda h, clip=False: np.mean(final(h, fut, clip) == st["pT"])
        print(f"     {d:+.2f}    {p(hB):.3f}       {p(hyb(hA, hB)):.3f}        [{p(hyb(hA, hB), True):.3f}]       {p(hyb(hB, hA)):.3f}")

# ---------------------------------------------------------------- claim 5
US = np.round(np.arange(-0.2, 1.21, 0.05), 2)


def half_point(us, p):
    """u at which p crosses 1/2 (linear interpolation; p may rise or fall)."""
    s = np.sign(p - 0.5)
    k = np.where(s[:-1] * s[1:] <= 0)[0]
    if len(k) == 0:
        return np.nan
    k = k[0]
    return us[k] + (0.5 - p[k]) * (us[k + 1] - us[k]) / (p[k + 1] - p[k])


def sweep(key, T0, step, seed, split=True):
    """f for the value `key` ('S' or 'M'), transplant at T0, probe at `step` (> T0), low inputs elsewhere."""
    x, st = quiet(N, T0, seed)
    rng = np.random.default_rng(seed + 1)
    n = len(x)
    xB = x.copy()
    if key == "S":
        lo, hi = np.maximum(st["T"], 0) + 0.02 * st["M"], 0.98 * st["M"]
        ok = hi > lo
        xB[ar_(n), st["pS"]] = lo + rng.random(n) * (hi - lo)
    else:
        g = np.exp(rng.uniform(-np.log(3), np.log(3), n))
        ok = np.ones(n, bool)
        xB[ar_(n), st["pM"]] = st["S"] + g * (st["M"] - st["S"])
    sB = stats(xB, T0)
    ok &= sB["quiet"] & (np.abs(sB[key] - st[key]) > 0.15 * st["M"])
    a, b = sub(st, ok), sub(sB, ok)
    assert np.array_equal(a["pS"], b["pS"]) and np.array_equal(a["pM"], b["pM"])
    hA_, hB_ = state(x[ok], T0), state(xB[ok], T0)
    fw, rv = hyb(hA_, hB_), hyb(hB_, hA_)
    m_ = ok.sum()
    low = lows(a["M"], 9 - T0, rng)
    D = b[key] - a[key]
    if key == "S":
        zlo, zhi, target = np.maximum(a["T"], 0) + 0.02 * a["M"], np.minimum(a["M"], b["M"]) - 0.02 * a["M"], np.full(m_, step)
    else:
        zlo, zhi, target = a["S"] + 0.02 * a["M"], np.inf, a["pM"]
    groups = {"all": np.ones(m_, bool), f"{key}_B > {key}_A": D > 0, f"{key}_B < {key}_A": D < 0}
    if split:
        for s in range(2, T0 + 1):
            groups[f"a_S = {s}"] = a["aS"] == s
        for s in range(2, T0 + 1):
            groups[f"a_M = {s}"] = a["aM"] == s
        groups["S before M"] = a["pS"] < a["pM"]
        groups["S after M"] = a["pS"] > a["pM"]
    hits, valid = {"fw": [], "rv": [], "A": []}, []
    for u in US:                                                  # one run per u and state kind
        z = a[key] + u * D
        valid.append((z > zlo) & (z < zhi))
        f = low.copy(); f[:, step - T0 - 1] = z
        for kk, hh in (("fw", fw), ("rv", rv), ("A", hA_)):
            hits[kk].append(final(hh, f) == target)
    valid = np.array(valid)
    # the crossing is found separately for the two signs of D (p rises with u for D > 0, falls for D < 0)
    out = {}
    for g, gm in groups.items():
        fs = {}
        for kk in hits:
            H = np.array(hits[kk])
            fs[kk] = []
            for sgn in (1, -1):
                mm = valid & (gm & (np.sign(D) == sgn))[None]
                p = np.where(mm.sum(1) > 200, (H & mm).sum(1) / np.maximum(mm.sum(1), 1), np.nan)
                fs[kk].append(half_point(US, p))
        out[g] = (gm.sum(), fs)
    return m_, out


for key, T0, step, seed, split in (("S", 6, 7, 7301, True), ("S", 6, 8, 7303, False), ("S", 6, 9, 7305, False), ("S", 8, 9, 7307, False),
                                   ("S", 4, 5, 7309, False), ("M", 6, 7, 7311, True)):
    m_, out = sweep(key, T0, step, seed, split)
    print(f"\n[claim 5] value of {'the runner-up' if key == 'S' else 'the maximum'}: transplant at t = {T0}, probe at step {step}, n = {m_} pairs")
    print("   group            n       f of rest A + tape B (value_B > / < value_A)    f of rest B + tape A      f of A intact")
    for g, (ng, fs) in out.items():
        print(f"   {g:14s} {ng:7d}        {fs['fw'][0]:5.2f} / {fs['fw'][1]:5.2f}                               {fs['rv'][0]:5.2f} / {fs['rv'][1]:5.2f}            {fs['A'][0]:5.2f} / {fs['A'][1]:5.2f}")

# ---------------------------------------------------------------- claim 8
print("\n[claim 8] S_pos pairs: one mode / all but one mode / a dose q of the donor's tape;  P(answer = A's S) / P(answer = B's S) [clipped]")
for t in (9, 6):
    x, st = quiet(N, t, 7400 + t)
    rng = np.random.default_rng(7500 + t)
    j, ok = pick_low_slot(x, st, t, rng)
    xB = swapped(x, st["pS"], j)
    ok &= stats(xB, t)["quiet"]
    a, b = sub(st, ok), sub(stats(xB, t), ok)
    hA_, hB_ = state(x[ok], t), state(xB[ok], t)
    cell = lambda h: (f"{np.mean(ans_at(h, t) == a['pS']):.3f} / {np.mean(ans_at(h, t) == b['pS']):.3f}  "
                      f"[{np.mean(ans_at(h, t, True) == a['pS']):.3f} / {np.mean(ans_at(h, t, True) == b['pS']):.3f}]  neg {np.mean((h < -1e-9).any(1)):.2f}")
    print(f"   t = {t}, n = {ok.sum()}")
    for m in TAPE:
        print(f"      only {m:5s} from B:     {cell(hyb(hA_, hB_, proj([m])))}")
    for m in TAPE:
        print(f"      all but {m:5s} from B:  {cell(hyb(hA_, hB_, proj([k for k in TAPE if k != m])))}")
    for q in (0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
        h = hyb(hA_, hB_, q=q)
        extra = f"   maximum {np.mean(ans_at(h, t) == a['pM']):.3f}, third {np.mean(ans_at(h, t) == a['pT']):.3f}" if q == 0.5 else ""
        print(f"      dose q = {q:4.2f}:           {cell(h)}{extra}")
