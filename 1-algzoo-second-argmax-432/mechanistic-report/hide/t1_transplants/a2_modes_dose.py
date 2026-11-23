"""a2: which tape modes carry the runner-up's position, and the dose-response.

Pairs in which A's and B's runner-up positions differ: S_pos, T_up_hi, unrel (rows with pS_A = pS_B dropped).
Quiet states at t = 9 (final answer) and t = 6 (running answer).  hybrid = A's rest + A's tape with
 (a) ONE mode's coordinates taken from B,   (b) all modes EXCEPT one taken from B,   (c) (1-q) A + q B on all four.
Reported: P(answer = A's runner-up), P(answer = B's runner-up), P(other); clipped-at-zero versions in brackets.
"""
from tp import *

N = 400_000
ALL = list(MODE_SEL)


def line(lab, h, t, a, b):
    out = []
    for hh in (h, np.maximum(h, 0)):
        ans = running_answer(hh, t)
        out.append((np.mean(ans == a), np.mean(ans == b)))
    (pa, pb), (ca, cb) = out
    print(f"   {lab:22s} S_A {pa:.3f}  S_B {pb:.3f}  other {1 - pa - pb:.3f}   [clipped: S_A {ca:.3f}  S_B {cb:.3f}]"
          f"   negative entry {neg_share(h):.3f}")


for t in (9, 6):
    x, st = quiet_sample(N, t, seed=21)
    hA = state_at(x, t)
    rng = np.random.default_rng(200 + t)
    print(f"\n{'=' * 100}\nstep t = {t}: {len(x)} quiet states")
    for kind in ("S_pos", "T_up_hi", "unrel"):
        xB, ok = make_pair(kind, x, st, t, rng)
        sB = prefix_stats(xB, t)
        ok &= sB["quiet"] & (sB["pS"] != st["pS"])
        a, b, hA_, hB_ = st["pS"][ok], sB["pS"][ok], hA[ok], state_at(xB[ok], t)
        print(f"\n--- {kind}: n = {ok.sum()}")
        line("A intact (q = 0)", hA_, t, a, b)
        for m in ALL:
            line(f"only {m} from B", hybrid(hA_, hB_, modes=[m]), t, a, b)
        for m in ALL:
            line(f"all but {m} from B", hybrid(hA_, hB_, modes=[k for k in ALL if k != m]), t, a, b)
        for pair in (["alt", "h2"], ["short", "h3"], ["alt", "short"], ["h2", "h3"], ["alt", "h3"], ["short", "h2"]):
            line(f"{'+'.join(pair)} from B", hybrid(hA_, hB_, modes=pair), t, a, b)
        for q in (0.2, 0.35, 0.45, 0.5, 0.55, 0.65, 0.8, 1.0):
            line(f"dose q = {q}", hybrid(hA_, hB_, q=q), t, a, b)
