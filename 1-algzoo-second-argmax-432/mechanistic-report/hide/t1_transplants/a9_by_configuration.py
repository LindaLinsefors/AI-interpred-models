"""a9: the main transplant rates split by configuration (quiet final states, t = 9, read out at once).

1. S_pos pairs (runner-up moved to a low slot): P(answer = the tape donor's runner-up position) by the donor's
   runner-up age a_S (columns) and maximum age a_M (rows).  Cells with fewer than 300 pairs are left blank.
2. Two-age tape (donor = zeros except M and 0.6 M at the two positions): accuracy by (a_M, a_S), next to A intact.
3. The same two tables reduced to one line each: runner-up arrived after / before the maximum.
"""
from tp import *

N = 800_000
x, st = quiet_sample(N, 9, seed=91)
hA = state_at(x, 9)
rng = np.random.default_rng(92)
ar = np.arange(len(x))


def grid(hit, aM, aS, title):
    print(f"\n{title}\n   a_M \\ a_S " + "".join(f"{a:7d}" for a in range(2, 10)))
    for m in range(2, 10):
        cells = []
        for s in range(2, 10):
            k = (aM == m) & (aS == s)
            cells.append(f"{hit[k].mean():7.3f}" if k.sum() >= 300 else "      .")
        print(f"   {m:9d} " + "".join(cells))
    after = aS < aM
    print(f"   runner-up arrived AFTER the maximum: {hit[after].mean():.3f} (n = {after.sum()});  BEFORE: {hit[~after].mean():.3f} (n = {(~after).sum()})")
    c = np.array([hit[(aM == m) & (aS == s)].mean() for m in range(2, 10) for s in range(2, 10) if ((aM == m) & (aS == s)).sum() >= 300])
    print(f"   range over the {len(c)} cells: {c.min():.3f} to {c.max():.3f}, median {np.median(c):.3f}")


xB, ok = make_pair("S_pos", x, st, 9, rng)
sB = prefix_stats(xB, 9)
ok &= sB["quiet"]
fw = hybrid(hA[ok], state_at(xB[ok], 9))
grid(answer(fw) == sB["pS"][ok], sB["aM"][ok], sB["aS"][ok],
     f"1. S_pos, n = {ok.sum()}: P(answer = tape donor's runner-up) by the DONOR's ages")
grid(answer(fw) == sB["pS"][ok], st["aM"][ok], st["aS"][ok], "1b. the same by the RECIPIENT's ages (a_S = age of the runner-up that the rest belongs to)")

xD = x.copy(); xD[:] = 0
xD[ar, st["pM"]], xD[ar, st["pS"]] = st["M"], 0.6 * st["M"]
hyb = hybrid(hA, state_at(xD, 9))
grid(answer(hyb) == st["pS"], st["aM"], st["aS"], f"2. Two-age tape (M and 0.6 M only), n = {len(x)}: accuracy of A's rest + that tape")
grid(answer(hA) == st["pS"], st["aM"], st["aS"], "2b. A intact")
