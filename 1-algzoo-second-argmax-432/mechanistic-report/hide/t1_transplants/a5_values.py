"""a5: which VALUES does the network read from the tape?  Three sweeps.

1. Value of an old third relative to the runner-up.  A: third far below (T < S - 0.4 M), of age >= 2, M - S > 0.25 M.
   B_d: the same sequence with that input set to S + d M.  For d > 0 the two swap roles.  Reported: P(answer = that
   input's position) and P(answer = A's runner-up position), read out at once (t = 9) and after three low inputs
   (transplant at t = 6), for B intact, fw = A's rest + B's tape, rv = B's rest + A's tape.
2. Value of the runner-up as the threshold for a LATER input.  S_val pairs with |S_B - S_A| > 0.15 M, transplant at
   t = 6, then z = S_A + u (S_B - S_A) at step 7 and two low inputs.  "z counted as above the runner-up" = the answer is
   step 7.  If the effective threshold is S_A + f (S_B - S_A), then f = 0 means "read from the rest", f = 1 "from the tape".
3. The same for the maximum: M_val pairs with |M_B - M_A| > 0.15 M, z = M_A + u (M_B - M_A) at step 7; "z counted as a new
   maximum" = the answer is the old maximum's position (z below the maximum: the answer is step 7).
"""
from tp import *

N = 600_000
ar_ = lambda n: np.arange(n)

# ------------------------------------------------------------------------------- 1
print("1. Old third moved to S + d M   (columns: P(answer = that input) / P(answer = A's runner-up))")
for t in (9, 6):
    x, st = quiet_sample(N, t, seed=51)
    ok = (st["T3"] < st["S"] - FAR * st["M"]) & (st["aT"] >= 2) & (st["M"] - st["S"] > 0.25 * st["M"])
    x, st = x[ok], {k: v[ok] for k, v in st.items()}
    n = len(x)
    hA = state_at(x, t)
    rng = np.random.default_rng(52 + t)
    fut = None if t == 9 else -st["M"][:, None] * rng.uniform(0.05, 0.6, (n, 9 - t))
    print(f"   transplant at t = {t}, n = {n}, " + ("read out at once" if t == 9 else "then three low inputs"))
    print("      d      B intact       A rest + B tape   (clipped)        B rest + A tape   neg. entry fw")
    for d in (-0.3, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2):
        xB = x.copy(); xB[ar_(n), st["pT"]] = st["S"] + d * st["M"]
        hB = state_at(xB, t)
        fw, rv = hybrid(hA, hB), hybrid(hB, hA)
        cell = lambda h, clip=False: (lambda a: f"{np.mean(a == st['pT']):.3f}/{np.mean(a == st['pS']):.3f}")(finish(h, fut, clip))
        print(f"   {d:+.2f}    {cell(hB)}    {cell(fw)}       {cell(fw, True)}      {cell(rv)}       {neg_share(fw):.3f}")

# ------------------------------------------------------------------------------- 2, 3
US = (-0.4, -0.2, -0.1, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.1, 1.2, 1.4)
T0 = 6
x, st = quiet_sample(N, T0, seed=53)
hA = state_at(x, T0)
rng = np.random.default_rng(54)
for key, kind, what in (("S", "S_val", "above the runner-up (answer = step 7)"), ("M", "M_val", "a new maximum (answer = old maximum's position)")):
    xB, ok = make_pair(kind, x, st, T0, rng)
    sB = prefix_stats(xB, T0)
    ok &= sB["quiet"] & (np.abs(sB[key] - st[key]) > 0.15 * st["M"])
    a, b, hA_, hB_ = {k: v[ok] for k, v in st.items()}, {k: v[ok] for k, v in sB.items()}, hA[ok], state_at(xB[ok], T0)
    fw, rv = hybrid(hA_, hB_), hybrid(hB_, hA_)
    n = ok.sum()
    low = -a["M"][:, None] * rng.uniform(0.05, 0.6, (n, 3))
    D = b[key] - a[key]
    lo = (np.maximum(a["T3"], 0) + 0.02 * a["M"]) if key == "S" else np.maximum(a["S"], b["S"]) + 0.02 * a["M"]
    hi = (np.minimum(a["M"], b["M"]) - 0.02 * a["M"]) if key == "S" else np.inf
    target = np.full(n, T0 + 1) if key == "S" else a["pM"]
    print(f"\n{2 if key == 'S' else 3}. {kind}: n = {n} pairs, transplant at t = {T0}; z = {key}_A + u ({key}_B - {key}_A) at step 7, then two low inputs."
          f"\n   P(z counted as {what});  A's own threshold is at u = 0, B's at u = 1")
    for sgn, lab in ((1, f"{key}_B > {key}_A"), (-1, f"{key}_B < {key}_A")):
        print(f"   --- {lab}:   u     n     A intact  B intact | A rest + B tape  (clipped) | B rest + A tape  | answers neither step 7 nor the old top two (fw)")
        for u in US:
            z = a[key] + u * D
            m = (np.sign(D) == sgn) & (z > lo) & (z < hi)
            f = low[m].copy(); f[:, 0] = z[m]
            p = lambda h, clip=False: np.mean(finish(h[m], f, clip) == target[m])
            af = finish(fw[m], f)
            oth = np.mean((af != T0 + 1) & (af != a["pS"][m]) & (af != a["pM"][m]))
            print(f"            {u:+.2f} {m.sum():6d}   {p(hA_):.3f}     {p(hB_):.3f}   |    {p(fw):.3f}        {p(fw, True):.3f}    |    {p(rv):.3f}        |  {oth:.3f}")
