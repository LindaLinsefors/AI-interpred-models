"""a4: where do the answers go?  Three break-downs.

1. The two score pieces on their own (quiet final states, entries of age >= 2 only): which input is on top of
   s_tape = U[:, o] tape part,  s_rest = scores - s_tape,  and of the full scores.
2. Unrelated donor (B rescaled to A's M), t = 9, hybrid = A's rest + B's tape, split by how the two top pairs collide:
   does A's rest veto its own maximum's position when B's tape names it?
3. Position of the maximum (M_pos pairs at t = 6) followed by a new maximum: the full break-down of the answers.
"""
from tp import *

N = 600_000
top8 = lambda s: s[:, :8].argmax(1)

# ---------------------------------------------------------------- 1
x, st = quiet_sample(N, 9, seed=41)
h = state_at(x, 9)
s_tape = tape_scores(h); s_all = h @ U.T; s_rest = s_all - s_tape
print(f"1. Top entry among ages >= 2, {len(x)} quiet final states")
for lab, s in (("full scores", s_all), ("tape piece", s_tape), ("rest piece", s_rest)):
    a = top8(s)
    print(f"   {lab:12s} runner-up {np.mean(a == st['pS']):.3f}  maximum {np.mean(a == st['pM']):.3f}  "
          f"third {np.mean(a == st['pT']):.3f}  other {np.mean((a != st['pS']) & (a != st['pM']) & (a != st['pT'])):.3f}")
ar = np.arange(len(x))
for lab, s in (("tape piece", s_tape), ("rest piece", s_rest), ("full scores", s_all)):
    d_SM = (s[ar, st["pS"]] - s[ar, st["pM"]]) / st["M"]
    ok = st["aT"] >= 2
    d_ST = ((s[ar, st["pS"]] - s[ar, st["pT"]]) / st["M"])[ok]
    q = lambda v: "  ".join(f"{u:7.1f}" for u in np.quantile(v, [.1, .25, .5, .75, .9]))
    print(f"   {lab:12s} entry(runner-up) - entry(maximum), in M: 10/25/50/75/90% = {q(d_SM)}   share > 0: {np.mean(d_SM > 0):.3f}")
    print(f"   {'':12s} entry(runner-up) - entry(third, age >= 2):        = {q(d_ST)}   share > 0: {np.mean(d_ST > 0):.3f}")

# ---------------------------------------------------------------- 2
rng = np.random.default_rng(42)
xB, _ = make_pair("unrel", x, st, 9, rng)
sB = prefix_stats(xB, 9)
fw = hybrid(h, state_at(xB, 9))
ans = answer(fw)
print(f"\n2. Unrelated donor, t = 9, A's rest + B's tape, n = {len(x)}")
cases = {"no collision of {M,S}_A with {M,S}_B": (st["pS"] != sB["pS"]) & (st["pM"] != sB["pM"]) & (st["pM"] != sB["pS"]) & (st["pS"] != sB["pM"]),
         "S_B = M_A (tape's runner-up sits on the rest's maximum)": sB["pS"] == st["pM"],
         "M_B = S_A (tape's maximum sits on the rest's runner-up)": sB["pM"] == st["pS"],
         "M_B = M_A, S_B != S_A": (sB["pM"] == st["pM"]) & (sB["pS"] != st["pS"]),
         "S_B = S_A": sB["pS"] == st["pS"]}
for lab, m in cases.items():
    nm = {"S_A": st["pS"][m], "S_B": sB["pS"][m], "M_A": st["pM"][m], "M_B": sB["pM"][m], "T_A": st["pT"][m], "T_B": sB["pT"][m]}
    print(f"   {lab:58s} n = {m.sum():6d}  {fmt(where_ans(ans[m], nm))}")

# ---------------------------------------------------------------- 3
T0 = 6
x, st = quiet_sample(N, T0, seed=43)
rng = np.random.default_rng(44)
print(f"\n3. Position of the maximum differs (M_pos) or donor unrelated; transplant at t = {T0}, then a new maximum")
for kind in ("M_pos", "unrel"):
    xB, ok = make_pair(kind, x, st, T0, rng)
    sB = prefix_stats(xB, T0)
    ok &= sB["quiet"] & (sB["pM"] != st["pM"])
    a, b = {k: v[ok] for k, v in st.items()}, {k: v[ok] for k, v in sB.items()}
    hA_, hB_ = state_at(x[ok], T0), state_at(xB[ok], T0)
    n = ok.sum()
    low = -a["M"][:, None] * rng.uniform(0.05, 0.6, (n, 3))
    nm = 1.3 * np.maximum(a["M"], b["M"])
    print(f"   --- {kind}, n = {n} (rows with M_B = M_A dropped)")
    for lab, k in (("new maximum at step 7, then two low inputs", 0), ("two low inputs, new maximum at step 9", 2)):
        fut = low.copy(); fut[:, k] = nm
        names = {"old max of A (rest)": a["pM"], "old max of B (tape)": b["pM"], "S_A": a["pS"], "S_B": b["pS"],
                 "the new maximum": np.full(n, T0 + 1 + k)}
        print(f"   {lab}")
        for l2, hh in (("A intact", hA_), ("A rest + B tape", hybrid(hA_, hB_)), ("  clipped", np.maximum(hybrid(hA_, hB_), 0)),
                       ("B rest + A tape", hybrid(hB_, hA_))):
            nn = names if "B rest" not in l2 else {"old max of B (rest)": b["pM"], "old max of A (tape)": a["pM"], "S_B": b["pS"],
                                                   "S_A": a["pS"], "the new maximum": names["the new maximum"]}
            print(f"      {l2:16s} {fmt(where_ans(finish(hh, fut), nn))}")
