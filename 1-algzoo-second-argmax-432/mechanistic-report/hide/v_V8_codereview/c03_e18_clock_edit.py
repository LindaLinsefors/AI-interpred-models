"""Code review of e18_clock_before_record.py (claim I2).

Checks:
 (1) the edit h[ORD] += 2 Re(dz RIGHT) is applied to a post-ReLU state; it can make activations
     negative, which the network can never produce.  How often, how large, and do the numbers
     change if the edited state is clipped at zero (then the clock coordinate is no longer exactly z')?
 (2) 'answer = p0 + 1' can coincide with the old runner-up q0 (and p0 - 1 likewise).  Split.
 (3) specificity: the same edits applied to the other complex modes (and f = 0 for the real ones).
 (4) the realised clock coordinate after clipping.
Same sample and selection as e18 (seed 1818, 1.5M sequences).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from modes import LEFT, RIGHT, LAM, NAMES, IS_COMPLEX

x = sample(1_500_000, 1818)
st = running_stats(x)
truth = second_argmax(x)
B = len(x); ar = np.arange(B)
om = np.angle(LAM["clock"])
tr = x.argmax(1)
ok = tr >= 3
p0 = np.where(ok, st["p1"][ar, np.maximum(tr - 1, 0)], -1)
q0 = np.where(ok, st["p2"][ar, np.maximum(tr - 1, 0)], -1)
M0 = st["M1"][ar, np.maximum(tr - 1, 0)]; S0 = st["M2"][ar, np.maximum(tr - 1, 0)]
g = tr - 1 - p0
ok &= (g >= 2) & (S0 > 0.2) & (M0 - S0 > 0.1) & (truth == p0) & (st["p3"][:, 9] == q0) & (tr <= 8)
xs, trs, p0s, q0s, gs, M0s = x[ok], tr[ok], p0[ok], q0[ok], g[ok], M0[ok]
print("selected sequences:", ok.sum())
del st


def run_mod(x, tr, fn, mode="clock", clip=False, info=None):
    Bn = len(x); h = np.zeros((Bn, N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        sel = np.where(tr - 1 == t)[0]
        if len(sel) and fn is not None:
            z = h[sel][:, ORD] @ LEFT[mode]
            dz = fn(z) - z
            add = np.multiply.outer(dz, RIGHT[mode])
            add = 2 * np.real(add) if IS_COMPLEX[mode] else np.real(add)
            hn = h[sel][:, ORD] + add
            if info is not None:
                info["neg_any"].append((hn < 0).any(1)); info["neg_min"].append(hn.min(1) / M0s[sel])
                info["neg_n"].append((hn < 0).sum(1))
                info["negmass"].append(-np.minimum(hn, 0).sum(1) / np.abs(hn).sum(1))
            if clip:
                hn = np.maximum(hn, 0)
                if info is not None:
                    info["z_target"].append(fn(z)); info["z_real"].append(hn @ LEFT[mode])
            h[np.ix_(sel, ORD)] = hn
    return (h @ U.T).argmax(1)


def new_info():
    return {k: [] for k in ("neg_any", "neg_min", "neg_n", "negmass", "z_target", "z_real")}


def cat(info, k):
    return np.concatenate(info[k])


base = run_mod(xs, trs, None)
print(f"unmodified: P(answer = p0) = {np.mean(base == p0s):.3f}")

print("\n(1) radius scaled by f: e18's edit (may leave negative activations) versus the same edit followed by clipping at 0")
print("    f   | e18: P(p0) P(q0) P(new max) | clipped: P(p0) P(q0) P(new max) | states with a negative activation after the edit; median number of negative units; "
      "median of min(h)/M0 among those; share of |h| that is negative | clipped: median |z_real/z_target| (f>0) or |z_real|/|z_orig|")
for f in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    i1 = new_info(); i2 = new_info()
    a = run_mod(xs, trs, lambda z, f=f: f * z, info=i1)
    c = run_mod(xs, trs, lambda z, f=f: f * z, clip=True, info=i2)
    na = cat(i1, "neg_any")
    zt, zr = cat(i2, "z_target"), cat(i2, "z_real")
    ratio = np.median(np.abs(zr) / np.abs(zt)) if f > 0 else np.nan
    print(f"   {f:4.2f} |      {np.mean(a == p0s):.3f} {np.mean(a == q0s):.3f} {np.mean(a == trs):.3f}     |"
          f"          {np.mean(c == p0s):.3f} {np.mean(c == q0s):.3f} {np.mean(c == trs):.3f}     |"
          f" {na.mean():.3f};  {np.median(cat(i1, 'neg_n')[na]) if na.any() else 0:.0f};  {np.median(cat(i1, 'neg_min')[na]) if na.any() else 0:7.2f};  "
          f"{np.mean(cat(i1, 'negmass')):.4f} | {ratio:.3f}")
i2 = new_info(); run_mod(xs, trs, lambda z: 0 * z, clip=True, info=i2)
i0 = new_info(); run_mod(xs, trs, lambda z: z, clip=True, info=i0)
print("    f = 0 with clipping: median |z_real| / |z_original| =", round(float(np.median(np.abs(cat(i2, 'z_real')) / np.abs(cat(i0, 'z_target')))), 3))

print("\n(2) phase advanced by k steps: e18's edit versus clipped; and the overlap of 'answer = p0 + j' with the old runner-up q0")
print("    k  | e18: P(p0) P(p0-1) P(p0-2) P(p0+1) P(p0+2) | clipped: P(p0) P(p0-1) P(p0-2) P(p0+1) P(p0+2) | neg. states | "
      "P(answer = p0-k) of which q0 = p0-k | P(answer = p0-k | q0 != p0-k) | P(answer = p0-k | q0 = p0-k)")
for k in (-3, -2, -1, 0, 1, 2, 3):
    i1 = new_info()
    fn = lambda z, k=k: z * np.exp(1j * om * k)
    a = run_mod(xs, trs, fn, info=i1)
    c = run_mod(xs, trs, fn, clip=True)
    tgt = p0s - k
    same = q0s == tgt
    hit = a == tgt
    print(f"   {k:+d}  |      {np.mean(a == p0s):.3f} {np.mean(a == p0s - 1):.3f}  {np.mean(a == p0s - 2):.3f}  {np.mean(a == p0s + 1):.3f}  {np.mean(a == p0s + 2):.3f}  |"
          f"          {np.mean(c == p0s):.3f} {np.mean(c == p0s - 1):.3f}  {np.mean(c == p0s - 2):.3f}  {np.mean(c == p0s + 1):.3f}  {np.mean(c == p0s + 2):.3f}  |"
          f"   {cat(i1, 'neg_any').mean():.3f}     |   {hit.mean():.3f}   {np.mean(hit & same):.3f}   |   {hit[~same].mean():.3f}   |   "
          + (f"{hit[same].mean():.3f} (n={same.sum()})" if same.sum() > 100 else ".") +
          f"   | target exists (0 <= p0-k <= tr-2): {np.mean((tgt >= 0) & (tgt <= trs - 2)):.3f}")

print("\n(3) specificity: the same two edits applied to other modes of the resting map (e18's unclipped edit):")
print("    mode     f=0: P(p0)  P(new max)  |  rotate by -1 step of that mode's own angle: P(p0) P(p0+1)  |  rotate by +1: P(p0) P(p0-1)")
for mode in NAMES:
    a0 = run_mod(xs, trs, lambda z: 0 * z, mode=mode)
    if IS_COMPLEX[mode]:
        omm = np.angle(LAM[mode])
        am = run_mod(xs, trs, lambda z: z * np.exp(-1j * omm), mode=mode)
        ap = run_mod(xs, trs, lambda z: z * np.exp(+1j * omm), mode=mode)
        extra = f"{np.mean(am == p0s):.3f} {np.mean(am == p0s + 1):.3f}                           |  {np.mean(ap == p0s):.3f} {np.mean(ap == p0s - 1):.3f}"
    else:
        extra = "(real mode)"
    print(f"    {mode:7s}      {np.mean(a0 == p0s):.3f}     {np.mean(a0 == trs):.3f}     |  {extra}")

print("\n(4) by age g of the old max just before the record (e18 pools g >= 2): clock f = 0 and k = -1, +1 (unclipped)")
a0 = run_mod(xs, trs, lambda z: 0 * z)
am = run_mod(xs, trs, lambda z: z * np.exp(-1j * om))
ap = run_mod(xs, trs, lambda z: z * np.exp(+1j * om))
for gg in range(2, 8):
    s = gs == gg
    if s.sum() < 500:
        continue
    print(f"    g = {gg}: n = {s.sum():6d}   f=0: P(p0) = {np.mean(a0[s] == p0s[s]):.3f}   k=-1: P(p0+1) = {np.mean(am[s] == p0s[s] + 1):.3f}, P(p0) = {np.mean(am[s] == p0s[s]):.3f}"
          f"   k=+1: P(p0-1) = {np.mean(ap[s] == p0s[s] - 1):.3f}, P(p0) = {np.mean(ap[s] == p0s[s]):.3f}, P(p0-1 exists) = {np.mean(p0s[s] >= 1):.3f}")
