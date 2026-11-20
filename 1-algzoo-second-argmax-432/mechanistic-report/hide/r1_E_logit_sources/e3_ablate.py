"""E3 (ablations). (i) Remove single (composite source, role) terms from the exact sum and re-score the task by scenario.
(ii) [INTERVENTION] in the running network: at the step where the final max arrives, do not clip the four comparators
(let them go negative for that one step), then run on normally.   seeds 1120, 1121 (2 x 131072 N(0,1))."""
from src import *

x = np.concatenate([sample(131072, sd) for sd in (1120, 1121)])
n = len(x); r = np.arange(n)
out, L = class_role_pos(x)
order = np.argsort(-x, axis=1); p1, p2, p3 = order[:, 0], order[:, 1], order[:, 2]
RM, MR = p2 < p1, p2 > p1
COMPO = {"CLIP": [0, 4, 5, 6, 7], "MARK": [1, 8], "TAPE": [2], "P9": [3], "P4": [1], "CW": [8], "C": [4, 5, 6, 7], "X": [0]}
print(f"n = {n}; accuracy all {np.mean(L.argmax(1) == p2):.4f}, RM {np.mean((L.argmax(1) == p2)[RM]):.4f}, MR {np.mean((L.argmax(1) == p2)[MR]):.4f}")
print("\n(i) remove terms from the exact sum; task accuracy (and what is answered instead)")
print("removed term                         | acc RM   ans=max  ans=third | acc MR   ans=max  ans=third")
for cn, rl in (("CLIP", [0]), ("CLIP", [1]), ("CLIP", [0, 1, 2, 3]), ("C", [0]), ("MARK", [0, 1, 2, 3]), ("P4", [0]), ("P4", [1]), ("P4", [0, 1, 2, 3]), ("CW", [0, 1, 2, 3]),
               ("TAPE", [0]), ("TAPE", [1]), ("TAPE", [0, 1, 2, 3]), ("P9", [0, 1, 2, 3])):
    Lr = L - out[:, COMPO[cn]][:, :, rl].sum((1, 2))
    a = Lr.argmax(1)
    lab = f"{cn} caused by " + ("all inputs" if len(rl) == 4 else "+".join(ROLES[k] for k in rl))
    print(f"{lab:36s} | {np.mean((a == p2)[RM]):.4f}   {np.mean((a == p1)[RM]):.4f}   {np.mean((a == p3)[RM]):.4f}   | {np.mean((a == p2)[MR]):.4f}   {np.mean((a == p1)[MR]):.4f}   {np.mean((a == p3)[MR]):.4f}")

print("\n(ii) [INTERVENTION] un-clip units at the step of the max (that step only)")


def run_unclip(x, step, units):
    """Run the RNN, but at step[b] the given units are NOT rectified."""
    B = len(x); h = np.zeros((B, 16))
    for t in range(10):
        zt = x[:, t, None] * w + h @ W.T
        hn = np.maximum(zt, 0)
        m = step == t
        if m.any():
            hn[np.ix_(m, units)] = zt[np.ix_(m, units)]
        h = hn
    return h @ U.T


for nm, units in (("comparators 1,6,7,8", COMP), ("comparator 7 only", [7]), ("comparators 1,6,8", [1, 6, 8])):
    for sc_name, sc in (("RM", RM), ("MR", MR)):
        mk = sc & (p1 >= 1) & (p1 <= 8)
        Li = run_unclip(x[mk], p1[mk], units)
        a = Li.argmax(1); a0 = L[mk].argmax(1)
        print(f"  {nm:22s} {sc_name} (max at steps 1..8, n = {mk.sum()}): accuracy {np.mean(a0 == p2[mk]):.4f} -> {np.mean(a == p2[mk]):.4f};  answers the max: {np.mean(a0 == p1[mk]):.4f} -> {np.mean(a == p1[mk]):.4f};"
              f"  mean L[p2]-L[p1]: {np.mean(L[mk][np.arange(mk.sum()), p2[mk]] - L[mk][np.arange(mk.sum()), p1[mk]]):.2f} -> {np.mean(Li[np.arange(mk.sum()), p2[mk]] - Li[np.arange(mk.sum()), p1[mk]]):.2f}")
