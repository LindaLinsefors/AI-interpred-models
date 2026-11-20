"""E29 (intervention). What do the wave units feed back into the comparator thresholds?

The six core units are not a closed system: the eight wave units other than unit 2 (SAT) add a term to each
comparator's pre-activation,  eps_j(t+1) = sum_{u in SAT} W[j, u] h_u(t),  and without it accuracy is about 55%.
Here h[ORD] is split exactly into its nine mode pieces (modes.py), so eps = sum_k eps^(k), and the part that
comes from one mode is removed from the comparators' input at every step.  Accuracy tells which modes' parts
are needed.  In threshold units the part of mode k is eps^(k)_j / g_j.
"""
from common import *
from modes import NAMES, LEFT, RIGHT, IS_COMPLEX

SAT = [0, 3, 10, 11, 12, 13, 14, 15]
sat = [ORD.index(u) for u in SAT]
g = -w[COMP]

P = {n: (2 * np.outer(RIGHT[n], LEFT[n]).real if IS_COMPLEX[n] else np.outer(RIGHT[n], LEFT[n]).real) for n in NAMES}
assert np.abs(sum(P.values()) - np.eye(13)).max() < 1e-9          # the pieces add up to the state
G = {n: W[np.ix_(COMP, SAT)] @ P[n][sat] for n in NAMES}          # eps^(k) = G[k] @ h[ORD]
Gall = sum(G.values())

x = sample(400_000, 2929)
y = second_argmax(x)
st = running_stats(x)


def acc(D):
    """Accuracy when D @ h[ORD] is subtracted from the comparators' pre-activation at every step."""
    h = np.zeros((len(x), N))
    for t in range(T):
        z = x[:, t, None] * w + h @ W.T
        z[:, COMP] -= h[:, ORD] @ D.T
        h = np.maximum(z, 0)
    return np.mean((h @ U.T).argmax(1) == y)


print(f"intact {acc(0 * Gall):.4f}   all of eps removed {acc(Gall):.4f}")
print("\nmode      remove its part   keep only its part")
for n in NAMES:
    print(f"{n:8s}  {acc(G[n]):.4f}            {acc(Gall - G[n]):.4f}")
print("\nkeep two modes' parts:")
for a, b in [("runner", "clock"), ("runner", "level"), ("clock", "level"), ("runner", "short"), ("clock", "short")]:
    print(f"  {a:6s} + {b:6s} {acc(Gall - G[a] - G[b]):.4f}")
print(f"  level + runner + clock {acc(Gall - G['level'] - G['runner'] - G['clock']):.4f}")

# size of each mode's part of the thresholds on data, in units of M: common level and differences
H = run(x)
M, S, a1 = st["M1"], st["M2"], st["a1"]
ok = (M[:, :9] > 0.3) & (a1[:, :9] >= 1)
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)[:, :9][ok]
Mo = M[:, :9][ok]
Ho = H[:, :9][ok][:, ORD]
print("\npart of the thresholds from each mode, in units of M (states with M > 0.3, max at least 1 step old):")
print("mode      mean of the common level   rms of the common level   rms of the differences   level ~ a M + b S+ (a, b, R2)")
for n in NAMES + ["all"]:
    e = (Ho @ (Gall if n == "all" else G[n]).T) / g / Mo[:, None]          # contribution to theta_j / M
    com, dif = e.mean(1), e - e.mean(1, keepdims=True)
    X = np.column_stack([np.ones(len(Mo)), Sp / Mo])                        # com ~ a + b S+/M
    co, *_ = np.linalg.lstsq(X, com, rcond=None)
    r2 = 1 - ((com - X @ co) ** 2).sum() / ((com - com.mean()) ** 2).sum()
    print(f"{n:8s}  {com.mean():+.3f}                     {np.sqrt((com ** 2).mean()):.3f}"
          f"                     {np.sqrt((dif ** 2).mean()):.3f}                    {co[0]:+.3f}, {co[1]:+.3f}, {r2:.2f}")
