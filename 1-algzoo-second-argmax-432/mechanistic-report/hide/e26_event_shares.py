"""E26. Shares of steps by event class, and how often unit 5 fires after each (N(0,1), 400k sequences)."""
from common import *
x = sample(400_000, 2626)
H, Z = run(x, return_pre=True)
st = running_stats(x)
rows = {"new max": [], "new runner-up": [], "below runner-up": []}
tot = 0
cnt = {k: 0 for k in rows}; u5 = {k: 0 for k in rows}; u4 = {k: 0 for k in rows}; n5 = {k: 0 for k in rows}
allon = 0; rest = 0; nstates = 0
for t in range(1, 10):
    Mp, Sp = st["M1"][:, t - 1], st["M2"][:, t - 1]
    xt = x[:, t]
    cls = {"new max": xt > Mp, "new runner-up": (xt < Mp) & (xt > Sp), "below runner-up": xt < Sp}
    for k, m in cls.items():
        cnt[k] += m.sum(); u4[k] += (Z[m, t, 4] > 0).sum()
        if t <= 8:
            u5[k] += (Z[m, t + 1, 5] > 0).sum(); n5[k] += m.sum()
    tot += len(x)
    g = Z[:, t] > 0
    nstates += len(x)
    allon += g[:, ORD].all(1).sum()
    rest += (g[:, ORD].all(1) & ~g[:, PULSE].any(1)).sum()
for k in rows:
    print(f"{k:16s}: share of steps 1..9 = {cnt[k]/tot:.4f}   unit 4 on at that step = {u4[k]/cnt[k]:.4f}   unit 5 on at the next step = {u5[k]/n5[k]:.4f}")
print(f"all 13 ordinary units on: {allon/nstates:.4f} of states at t=1..9;  resting pattern (and pulses off): {rest/nstates:.4f}")
# the same restricted to positive runner-up before the step
