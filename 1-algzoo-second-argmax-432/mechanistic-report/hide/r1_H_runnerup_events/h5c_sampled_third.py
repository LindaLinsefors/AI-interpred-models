"""H5 (in distribution): final margin between the positions of the runner-up S and of the third-largest T on N(0,1) inputs,
by order of arrival of (M, S, T)."""
from hcommon import *

acc = {}
for seed in (1032, 1033, 1034):
    x = sample(400_000, seed)
    L = logits(run(x))
    o = np.argsort(-x, axis=1)
    p1, p2, p3 = o[:, 0], o[:, 1], o[:, 2]
    ar = np.arange(len(x))
    rec = dict(p1=p1, p2=p2, p3=p3, M=x[ar, p1], S=x[ar, p2], T=x[ar, p3], mg=L[ar, p2] - L[ar, p3], mg1=L[ar, p2] - L[ar, p1], ans=L.argmax(1))
    for k, v in rec.items():
        acc.setdefault(k, []).append(v)
d = {k: np.concatenate(v) for k, v in acc.items()}
n = len(d["M"])
order = np.select([(d["p3"] < d["p1"]) & (d["p1"] < d["p2"]), (d["p1"] < d["p3"]) & (d["p3"] < d["p2"]), (d["p1"] < d["p2"]) & (d["p2"] < d["p3"]),
                   (d["p3"] < d["p2"]) & (d["p2"] < d["p1"]), (d["p2"] < d["p3"]) & (d["p3"] < d["p1"]), (d["p2"] < d["p1"]) & (d["p1"] < d["p3"])],
                  ["T,M,S", "M,T,S", "M,S,T", "T,S,M", "S,T,M", "S,M,T"], default="?")
pos = (d["T"] > 0)
ratio = d["T"] / np.where(d["S"] > 0, d["S"], np.nan)
RB = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
print(f"n = {n} sequences (seeds 1032-1034); T > 0 in {pos.mean():.3f}")
print("\nP(answer = p2) / P(answer = p3) by order of arrival and T/S (T > 0):")
print("  order  | " + " | ".join(f"T/S {RB[i]:.2f}-{RB[i+1]:.2f}" for i in range(len(RB) - 1)) + " |  share of all")
for o_ in ["T,M,S", "M,T,S", "M,S,T", "T,S,M", "S,T,M", "S,M,T"]:
    row = ""
    for i in range(len(RB) - 1):
        s = pos & (order == o_) & (ratio >= RB[i]) & (ratio < RB[i + 1])
        row += f"  {np.mean(d['ans'][s] == d['p2'][s]):.3f} / {np.mean(d['ans'][s] == d['p3'][s]):.3f} |"
    print(f"  {o_}  | {row}  {(order == o_).mean():.3f}")
print("\nregression  margin(p2 over p3) ~ c1 (S - T) + c2 S + c3 M   (T/S >= 0.7, T > 0):")
for o_ in ["T,M,S", "M,T,S", "M,S,T", "T,S,M", "S,T,M", "S,M,T"]:
    s = pos & (order == o_) & (ratio >= 0.7)
    coef, R2 = lsq(d["mg"][s], [d["S"][s] - d["T"][s], d["S"][s], d["M"][s]])
    print(f"   {o_}: {coef[0]:7.1f} (S-T) + {coef[1]:6.1f} S + {coef[2]:6.1f} M   R2 = {R2:.3f}   n = {s.sum()}   P(margin > 0) = {np.mean(d['mg'][s] > 0):.3f}")
print("\noverall error budget: P(answer = p3) =", round(np.mean(d["ans"] == d["p3"]), 4), " P(answer = p1) =", round(np.mean(d["ans"] == d["p1"]), 4),
      " P(other wrong) =", round(np.mean((d["ans"] != d["p1"]) & (d["ans"] != d["p2"]) & (d["ans"] != d["p3"])), 4), " accuracy =", round(np.mean(d["ans"] == d["p2"]), 4))
print("P(answer = p3) by order:", {o_: round(float(np.mean((d['ans'] == d['p3'])[order == o_])), 4) for o_ in ["T,M,S", "M,T,S", "M,S,T", "T,S,M", "S,T,M", "S,M,T"]})
print("P(answer = p1) by order:", {o_: round(float(np.mean((d['ans'] == d['p1'])[order == o_])), 4) for o_ in ["T,M,S", "M,T,S", "M,S,T", "T,S,M", "S,T,M", "S,M,T"]})
