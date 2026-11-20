"""B6 probe breakdown: two-pulse inputs (M = 1.5 at p, S at q, background b), level scaled by f at t0 = 4.
Which cases flip: runner-up older (q < p) or younger (q > p) than the max? And natural data: flips by order of q and p."""
from blib import *
L, R = LEFT["level"].real, RIGHT["level"].real

def perturb(h, f):
    h2 = h.copy(); h2[:, ORD] += ((f - 1) * (h[:, ORD] @ L))[:, None] * R[None, :]; return h2

print("two-pulse inputs, t0 = 4, positions p, q in 0..4, S in {0.3, 1.0, 1.4}, b in {-1, -0.3}")
for f in (0.8, 0.9, 1.1, 1.2):
    out = {"q<p": [0, 0, 0, 0], "q>p": [0, 0, 0, 0]}
    for b in (-1.0, -0.3):
        for Sv in (0.3, 1.0, 1.4):
            for p in range(5):
                for q in range(5):
                    if p == q: continue
                    x = np.full((1, 10), b); x[0, p] = 1.5; x[0, q] = Sv
                    H = run(x)
                    a0 = logits(H).argmax(1)[0]
                    Hs = run(x[:, 5:], h0=perturb(H[:, 4], f))
                    a1 = (Hs[0, -1] @ U.T).argmax()
                    k = "q<p" if q < p else "q>p"
                    out[k][0] += 1; out[k][1] += (a0 == q); out[k][2] += (a1 == p); out[k][3] += (a1 != p) and (a1 != a0)
    print(f"  f = {f}: " + "; ".join(f"{k}: {v[0]} cases, baseline correct {v[1]}, perturbed answer = max position {v[2]}, = some other new position {v[3]}" for k, v in out.items()))

print("\nnatural data (N(0,1), seed 1076, 400k), t0 = 5: P(answer changed) when both the max p and the runner-up q are at positions <= t0")
x = sample(400_000, 1076)
H = run(x); base = logits(H).argmax(1)
srt = np.argsort(x, axis=1); q, p, third = srt[:, -2], srt[:, -1], srt[:, -3]
for f in (0.8, 0.9, 1.1, 1.2):
    Hs = run(x[:, 6:], h0=perturb(H[:, 5], f))
    ans = (Hs[:, -1] @ U.T).argmax(1)
    ch = ans != base
    row = []
    for name, s in (("q<p<=t0", (q < p) & (p <= 5)), ("p<q<=t0", (p < q) & (q <= 5))):
        c = ch & s
        row.append(f"{name}: share {s.mean():.3f}, P(changed) {ch[s].mean():.3f}, new = argmax {(ans[c] == p[c]).mean():.3f}, new = 3rd {(ans[c] == third[c]).mean():.3f}")
    print(f"  f = {f}: " + "; ".join(row))
