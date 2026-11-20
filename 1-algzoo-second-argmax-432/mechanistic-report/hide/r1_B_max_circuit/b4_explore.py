"""B4 exploration: states and gates after all-negative prefixes (seed 1040)."""
from blib import *

rng = np.random.default_rng(1040)
n = 200_000
x = -np.abs(rng.standard_normal((n, 10)))          # N(0,1) conditioned on all inputs negative
H, J = traces(x)
Z = run(x, return_pre=True)[1]
G = Z > 0
print("all-negative sequences, n =", n)
print("P(unit on) by step t (rows) and unit (columns)")
print("  t  " + " ".join(f"{u:5d}" for u in range(N)))
for t in range(10):
    print(f"  {t}  " + " ".join(f"{G[:, t, u].mean():5.3f}" for u in range(N)))

print("\nmean |h| by step and unit, divided by mean|x| = 0.798")
for t in range(10):
    print(f"  {t}  " + " ".join(f"{H[:, t, u].mean():5.2f}" for u in range(N)))

print("\ntrace norms ||J[t, p]|| (d h_t / d x_p), median over sequences; rows t, columns lag t - p")
for t in range(10):
    print(f"  {t}  " + " ".join(f"{np.median(np.linalg.norm(J[:, t, t - lag], axis=-1)):8.4f}" for lag in range(t + 1)))
print("  |w| =", np.linalg.norm(w), " |w on units with w<0| =", np.linalg.norm(np.minimum(w, 0)))

print("\nmost common gate patterns at t = 5 (all-negative prefix)")
pid = pattern_id(G[:, 5])
vals, cnt = np.unique(pid, return_counts=True)
for i in np.argsort(-cnt)[:8]:
    print(f"   {cnt[i]/n:.4f}  ON = {pattern_str(vals[i])}")

lg = logits(H)
sa = second_argmax(x)
print("\naccuracy on all-negative sequences:", (lg.argmax(1) == sa).mean(), " answer histogram:", np.bincount(lg.argmax(1), minlength=10) / n)
print("true histogram:", np.bincount(sa, minlength=10) / n)
print("P(answer = argmax):", (lg.argmax(1) == x.argmax(1)).mean())

# logit sensitivity: d logit / d x_p at the final step
dl = J[:, 9] @ U.T          # (n, p, 10): d logits_k / d x_p
print("\nmedian |d logit_k / d x_p| over k, by p (final step):", np.round(np.median(np.abs(dl).max(-1), 0), 3))
print("median spread across positions k of d logit_k/d x_p (max - min), by p:", np.round(np.median(dl.max(-1) - dl.min(-1), 0), 3))
