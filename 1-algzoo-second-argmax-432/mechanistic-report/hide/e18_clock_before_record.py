"""E18 (intervention). Is the clock the 'pending deposit' for the old max?
Take sequences in which a record happens at step tr, the old max (position p0, age g >= 2 just before)
ends up as the final runner-up, the old runner-up (q0) as the final third, and nothing else changes the
top three afterwards.  One step before the record (state h_{tr-1}) we change only the clock coordinate z:
   z -> f * z            (radius scaled)
   z -> z * exp(i k w)   (phase advanced by k free steps, w = 42.75 deg)
and continue with the real inputs.  Predictions: f = 0 removes the old max's lead; a rotation by k moves the
developed bump k positions earlier, to position p0 - k.
(With f = 0 the answer does not fall back to the old runner-up q0, as first guessed here: it goes to the
new max tr in about 78% and to q0 in about 16%.)
The edit acts on a post-ReLU state and can leave negative activations, which the network cannot produce
(nearly always for f > 1), so the radius scan is repeated with the edited state clipped at zero."""
from common import *
from modes import LEFT, RIGHT, LAM

x = sample(1_500_000, 1818)
st = running_stats(x)
truth = second_argmax(x)
B = len(x)
om = np.angle(LAM["clock"])

# choose the record step: the step at which the FINAL max arrived
tr = x.argmax(1)
ar = np.arange(B)
ok = (tr >= 3)
p0 = np.where(ok, st["p1"][ar, np.maximum(tr - 1, 0)], -1)       # old max
q0 = np.where(ok, st["p2"][ar, np.maximum(tr - 1, 0)], -1)       # old runner-up
M0 = st["M1"][ar, np.maximum(tr - 1, 0)]; S0 = st["M2"][ar, np.maximum(tr - 1, 0)]
g = tr - 1 - p0
ok &= (g >= 2) & (S0 > 0.2) & (M0 - S0 > 0.1) & (truth == p0) & (st["p3"][:, 9] == q0) & (tr <= 8)
xs, trs, p0s, q0s, gs = x[ok], tr[ok], p0[ok], q0[ok], g[ok]
print("selected sequences:", ok.sum())

def run_mod(x, tr, fn, clip=False):
    """Returns the answers and the share of edited states with a negative entry (counted before any clipping)."""
    Bn = len(x); h = np.zeros((Bn, N)); neg = np.zeros(Bn, bool)
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        sel = np.where(tr - 1 == t)[0]
        if len(sel) and fn is not None:
            z = h[sel][:, ORD] @ LEFT["clock"]
            dz = fn(z) - z
            h[np.ix_(sel, ORD)] += 2 * np.real(np.multiply.outer(dz, RIGHT["clock"]))
            neg[sel] = (h[sel] < 0).any(1)
            if clip:
                h[sel] = np.maximum(h[sel], 0)
    return (h @ U.T).argmax(1), neg.mean()

def outcome(a):                                        # p0, q0 and tr are three different positions, so these four shares are exclusive
    return f"{np.mean(a == p0s):.3f}   {np.mean(a == q0s):.3f}   {np.mean(a == trs):.3f}      {np.mean((a != p0s) & (a != q0s) & (a != trs)):.3f}"

base, _ = run_mod(xs, trs, None)
print(f"unmodified: P(answer = old max p0) = {np.mean(base == p0s):.3f}, P(answer = old runner-up q0) = {np.mean(base == q0s):.3f}")
print("\nradius scaled by f:   f | P(p0)   P(q0)   P(new max)  P(other) | edited state clipped at 0: P(p0)   P(q0)   P(new max)  P(other) | share of edited states with a negative entry")
for f in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    a, neg = run_mod(xs, trs, lambda z, f=f: f * z)
    c, _ = run_mod(xs, trs, lambda z, f=f: f * z, clip=True)
    print(f"                   {f:4.2f} | {outcome(a)}      |                            {outcome(c)}      | {neg:.3f}")
    if f == 0:
        a0 = a
print(f"where the answer goes for f = 0: the new max tr {np.mean(a0 == trs):.3f}, the old runner-up q0 {np.mean(a0 == q0s):.3f}, "
      f"still the old max p0 {np.mean(a0 == p0s):.3f}, another position {np.mean((a0 != p0s) & (a0 != q0s) & (a0 != trs)):.3f}")

print("\nphase advanced by k steps.  The seven plain columns are over all selected sequences and are NOT exclusive: p0 +- j can be q0, and p0 - k need not exist.")
print("The target p0 - k is 'clean' if it is an input that arrived before the record (0 <= p0 - k < tr) and is not q0; the last probability is taken over those sequences only.")
print("   k | P(p0)  P(p0-1)  P(p0-2)  P(p0+1)  P(p0+2)  P(q0)  P(new max) | share with a clean target   P(answer = p0-k | clean target) | share of edited states with a negative entry")
for k in (-3, -2, -1, 0, 1, 2, 3):
    a, neg = run_mod(xs, trs, lambda z, k=k: z * np.exp(1j * om * k))
    tgt = p0s - k
    clean = (tgt >= 0) & (tgt < trs) & (tgt != q0s)
    print(f"  {k:+d} | {np.mean(a == p0s):.3f}  {np.mean(a == p0s - 1):.3f}    {np.mean(a == p0s - 2):.3f}    {np.mean(a == p0s + 1):.3f}    {np.mean(a == p0s + 2):.3f}   {np.mean(a == q0s):.3f}   {np.mean(a == trs):.3f}"
          f"      |        {clean.mean():.3f}                       {np.mean(a[clean] == tgt[clean]):.3f}                 | {neg:.3f}")
