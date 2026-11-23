"""a8: follow-ups to a7.

1. Tape deleted (tape coordinates set to 0) in quiet final states: what does the network answer?
2. A tape that knows only the two AGES: donor = zeros except M at the maximum's position and r M at the runner-up's
   position, for fixed r.  hybrid = A's rest + that tape.  Accuracy against r, at t = 9 (read out at once; also split
   by A's true S/M) and for a transplant at t = 6 followed by chosen futures.  "true" = r is A's own S/M.
"""
from tp import *

N = 600_000
RS = (0.1, 0.2, 0.4, 0.6, 0.8, 0.95, "true")


def two_input_donor(x, st, t, r):
    xB = x.copy(); xB[:, : t + 1] = 0
    ar = np.arange(len(x))
    xB[ar, st["pM"]] = st["M"]
    xB[ar, st["pS"]] = st["S"] if r == "true" else r * st["M"]
    return xB


x, st = quiet_sample(N, 9, seed=81)
hA = state_at(x, 9)
ans = answer(set_tape(hA, np.zeros((len(x), 7))))
age = 9 - ans
print(f"1. Tape deleted, {len(x)} quiet final states: answer = runner-up {np.mean(ans == st['pS']):.3f}, maximum {np.mean(ans == st['pM']):.3f}, "
      f"third {np.mean(ans == st['pT']):.3f};  by age of the named input: " + "  ".join(f"{a}: {np.mean(age == a):.3f}" for a in range(10)))
print("   for comparison, ages of the true runner-up:                    " + " " * 44 + "  ".join(f"{a}: {np.mean(st['aS'] == a):.3f}" for a in range(10)))

print("\n2. Donor with only M and r M at the two positions")
ratio = st["S"] / st["M"]
edges = np.quantile(ratio, [0, 1 / 3, 2 / 3, 1])
print(f"   t = 9, read out at once (A intact: {np.mean(answer(hA) == st['pS']):.3f});  terciles of A's S/M: "
      + ", ".join(f"[{a:.2f}, {b:.2f}]" for a, b in zip(edges[:-1], edges[1:])))
print("      r      all     [clipped]   low S/M   mid S/M   high S/M   neg. entry")
for r in RS:
    hyb = hybrid(hA, state_at(two_input_donor(x, st, 9, r), 9))
    ok = answer(hyb) == st["pS"]
    terc = "     ".join(f"{ok[(ratio >= a) & (ratio <= b)].mean():.3f}" for a, b in zip(edges[:-1], edges[1:]))
    print(f"   {str(r):>5s}   {ok.mean():.3f}   [{np.mean(answer(np.maximum(hyb, 0)) == st['pS']):.3f}]     {terc}      {neg_share(hyb):.3f}")

t = 6
x, st = quiet_sample(N, t, seed=82)
hA = state_at(x, t)
rng = np.random.default_rng(83)
n, pre = len(x), x[:, : t + 1]
low = -st["M"][:, None] * rng.uniform(0.05, 0.6, (n, 3))
put = lambda v, k: np.concatenate([low[:, :k], v[:, None], low[:, k + 1:]], 1)
futs = {"low x3": low, "new max at 7": put(1.3 * st["M"], 0), "new max at 9": put(1.3 * st["M"], 2),
        "new runner-up at 7": put((st["S"] + st["M"]) / 2, 0), "new third at 7": put((np.maximum(st["T3"], 0) + st["S"]) / 2, 0),
        "A's own inputs": x[:, t + 1:]}
print(f"\n   transplant at t = {t}, {n} quiet states, accuracy after the future [clipped]")
print("      r   " + "".join(f"{k:>22s}" for k in futs))
rows = [("intact", hA)] + [(str(r), hybrid(hA, state_at(two_input_donor(x, st, t, r), t))) for r in RS]
for lab, h in rows:
    row = f"   {lab:>6s} "
    for f in futs.values():
        tA = second_argmax(np.concatenate([pre, f], 1))
        row += f"     {np.mean(finish(h, f) == tA):.3f} [{np.mean(finish(h, f, True) == tA):.3f}]  "
    print(row)
