"""E08. Controlled inputs: two large values M (at pM) and S (at pS) on a constant low background b.
Does the network answer pS?  How do the logits decompose into mode contributions?"""
from common import *
from modes import coords, RIGHT, NAMES, IS_COMPLEX

def two_pulse(pM, pS, M=1.5, S=1.0, b=-1.0):
    x = np.full(10, b); x[pM] = M; x[pS] = S
    return x

for b in (-1.0, -0.3, 0.0):
    for (M, S) in ((1.5, 1.0), (1.5, 1.4), (1.5, 0.3)):
        ans = np.full((10, 10), -1)
        marg = np.full((10, 10), np.nan)
        for pM in range(10):
            for pS in range(10):
                if pM == pS: continue
                lg = logits(run(two_pulse(pM, pS, M, S, b)))[0]
                ans[pM, pS] = lg.argmax()
                srt = np.sort(lg)
                marg[pM, pS] = lg[pS] - np.max(np.delete(lg, pS))
        correct = sum(ans[pM, pS] == pS for pM in range(10) for pS in range(10) if pM != pS)
        print(f"\nbackground {b}, M={M}, S={S}: correct {correct}/90.  margin of the true position over the best other (rows pM, cols pS):")
        for pM in range(10):
            print(f"  pM={pM}: " + " ".join("   .  " if pM == pS else f"{marg[pM,pS]:6.1f}" for pS in range(10)))

# Decompose one example by mode and pulse
def decompose(x, label):
    H = run(x)[0]
    m = coords(H[-1])
    lg = U @ H[-1]
    print(f"\n{label}: x = {x}")
    print("  final logits:", np.round(lg, 1), " answer", lg.argmax())
    Uo = U[:, ORD]
    parts = {}
    for n in NAMES:
        v = m[n] * RIGHT[n]
        parts[n] = Uo @ (2 * np.real(v) if IS_COMPLEX[n] else np.real(v))
    for pu in PULSE:
        parts[f"unit{pu}"] = U[:, pu] * H[-1, pu]
    for k, v in parts.items():
        print(f"  {k:7s}: " + " ".join(f"{q:7.1f}" for q in v))
    tot = sum(parts.values())
    assert np.allclose(tot, lg)
    for n in NAMES:
        c = m[n]
        print(f"    coord {n:7s} = " + (f"{abs(c):7.2f} @ {np.degrees(np.angle(c)):6.1f} deg" if IS_COMPLEX[n] else f"{c.real:7.2f}"))

decompose(two_pulse(6, 2, 1.5, 1.0, -1.0), "S at 2 (old max), M at 6")
decompose(two_pulse(2, 6, 1.5, 1.0, -1.0), "M at 2, S at 6 (younger runner-up)")
