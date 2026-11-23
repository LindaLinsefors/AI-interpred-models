"""a10: the configurations in which the answer does NOT follow a transplanted tape (found in a9).

1. Intact quiet final states: P(the runner-up is the top entry among ages >= 2) for the tape piece alone and for the
   rest piece alone, by (a_M, a_S).
2. S_pos pairs, t = 9: break-down of the hybrid's answer in the donor configurations a_S = 2 with a_M = 8, 9, and
   for all donors with a_S = 2, a_S = 9, and 3 <= a_S <= 8.
"""
from tp import *

N = 800_000
x, st = quiet_sample(N, 9, seed=101)
h = state_at(x, 9)
s_tape = tape_scores(h); s_rest = h @ U.T - s_tape
top8 = lambda s: s[:, :8].argmax(1)
for lab, s in (("tape piece alone", s_tape), ("rest piece alone", s_rest)):
    hit = top8(s) == st["pS"]
    print(f"\n1. {lab}: P(top entry among ages >= 2 is the runner-up), {len(x)} quiet final states\n   a_M \\ a_S " + "".join(f"{a:7d}" for a in range(2, 10)))
    for m in range(2, 10):
        print(f"   {m:9d} " + "".join(f"{hit[(st['aM'] == m) & (st['aS'] == s)].mean():7.3f}" if m != s else "      ." for s in range(2, 10)))

rng = np.random.default_rng(102)
xB, ok = make_pair("S_pos", x, st, 9, rng)
sB = prefix_stats(xB, 9)
ok &= sB["quiet"]
a, b = {k: v[ok] for k, v in st.items()}, {k: v[ok] for k, v in sB.items()}
fw = hybrid(h[ok], state_at(xB[ok], 9))
ans = answer(fw)
print(f"\n2. S_pos, t = 9, n = {ok.sum()}: answer of A's rest + B's tape, by the donor's configuration")
sel = {"donor a_S = 2, a_M = 9": (b["aS"] == 2) & (b["aM"] == 9), "donor a_S = 2, a_M = 8": (b["aS"] == 2) & (b["aM"] == 8),
       "donor a_S = 2, a_M <= 7": (b["aS"] == 2) & (b["aM"] <= 7), "donor a_S = 9": b["aS"] == 9,
       "donor 3 <= a_S <= 8": (b["aS"] >= 3) & (b["aS"] <= 8)}
for lab, m in sel.items():
    nm = {"S_A (rest)": a["pS"][m], "S_B (tape)": b["pS"][m], "maximum": a["pM"][m], "third": a["pT"][m]}
    age = 9 - ans[m]
    print(f"   {lab:24s} n = {m.sum():6d}  {fmt(where_ans(ans[m], nm))}   answer has age 0 or 1: {np.mean(age <= 1):.3f}")
