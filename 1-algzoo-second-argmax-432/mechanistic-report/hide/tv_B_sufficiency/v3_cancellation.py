"""V3. Claim 5: when the inputs outside V are redrawn, do the changes of the tape piece and of the rest of the scores cancel?
Score entries of age >= 2 (indices 0..7), units of M, one redraw per state.  Three ways to remove what cannot matter
for the answer: raw; minus the change at the runner-up's own entry (the investigator's choice); minus the mean change
over the eight entries.
Controls that could explain the cancellation away:
 (i)  a random perturbation of the 13 non-pulse units (isotropic), and the difference between two unrelated quiet states:
      does ANY state difference split into anticorrelated tape and rest pieces in this (non-orthogonal) mode basis?
 (ii) which redrawn inputs produce it: only the non-contenders of age 0-1, or only those of age >= 2;
 (iii) which part of the rest cancels: slow modes (level, runner, clock), fast modes (trans, echo), pulse units.
Populations: final quiet states (t = 9) of 60,000 sequences (seed 23); and all steps 3..9 of 30,000 sequences (seed 24).
Run:  python -B v3_cancellation.py > v3_cancellation.txt"""
from vb import *

G_SLOW, G_FAST = U[:, o] @ projector(["level", "runner", "clock"]), U[:, o] @ projector(["trans", "echo"])
assert np.abs(G_TAPE + G_SLOW + G_FAST - U[:, o]).max() < 1e-9
rms = lambda a: np.sqrt(np.mean(a ** 2))
cor = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]


def pieces(dh, M):
    """Change of the scores at ages >= 2 split into tape, slow, fast, pulse pieces (units of M)."""
    d = dh / M[:, None]
    return [(d[:, o] @ G.T)[:, :8] for G in (G_TAPE, G_SLOW, G_FAST)] + [(d[:, [4, 5, 9]] @ U[:, [4, 5, 9]].T)[:, :8]]


def views(P, a):
    ar = np.arange(P.n)
    iS = 9 - P.aS
    rel = (a - a[ar, iS][:, None])[COL[None, :8] != iS[:, None]]
    return a, rel, a - a.mean(1, keepdims=True)


def report(P, name, dh):
    t, s, f, p = pieces(dh, P.M)
    r = s + f + p
    out = f"{name:44s} |"
    for (a, b) in zip(views(P, t), views(P, r)):
        out += f" {rms(a):6.2f} {rms(b):6.2f} {rms(a + b):6.2f} {cor(a, b):6.2f} |"
    print(out, flush=True)
    return t, s, f, p


for label, P in (("FINAL QUIET STATES (t = 9), 60000 sequences, seed 23", Pop(60_000, seed=23, steps=[9])),
                 ("ALL STEPS 3..9, 30000 sequences, seed 24", Pop(30_000, seed=24))):
    rng = np.random.default_rng(5)
    print(f"{label}: {P.n} states")
    print(f"{'what is redrawn / perturbed':44s} | raw: tape   rest    sum   corr | minus runner-up's entry        | minus mean of the 8 entries   |")
    keep = {}
    for name, rules in (("V0 fixed", ()), ("top 3 fixed", (top(3),)), ("contenders c = 1 fixed", (contenders(1.0),)),
                        ("contenders c = 0.75 fixed", (contenders(0.75),)), ("all positive inputs fixed", (contenders(0.0),)),
                        ("contenders 0.75 + ages 0,1 fixed", (contenders(0.75), ages([0, 1]))),
                        ("contenders 0.75 + ages 2..9 fixed", (contenders(0.75), ages(range(2, 10))))):
        xk, _ = V(P, *rules)(rng)
        keep[name] = report(P, name, fwd(xk) - P.h)
    perm = rng.permutation(P.n)
    report(P, "control: unrelated quiet state (/M each)", P.h[perm] * (P.M / P.M[perm])[:, None] - P.h)
    dh = np.zeros((P.n, 16)); dh[:, o] = rng.standard_normal((P.n, 13)) * P.M[:, None]
    report(P, "control: isotropic noise on the 13 units", dh)
    dh = np.zeros((P.n, 16)); dh[:, [1, 6, 7, 8]] = rng.standard_normal((P.n, 4)) * P.M[:, None]
    report(P, "control: noise on comparators 1,6,7,8", dh)
    dh = np.zeros((P.n, 16)); dh[:, o] = np.outer(P.M * rng.standard_normal(P.n), w[o])
    report(P, "control: one more input write, w[o] dx", dh)

    t, s, f, p = keep["contenders c = 0.75 fixed"]
    print("  contenders 0.75 fixed, parts of the rest (minus mean of the 8 entries): rms of the part, its correlation with the tape change, rms(tape + part)")
    c8 = lambda a: a - a.mean(1, keepdims=True)
    for nm, part in (("slow modes (level, runner, clock)", s), ("fast modes (trans, echo)", f), ("pulse units 4, 5, 9", p), ("fast + pulse", f + p), ("whole rest", s + f + p)):
        print(f"    {nm:34s} rms {rms(c8(part)):6.2f}   corr with tape {cor(c8(t), c8(part)):6.2f}   rms(tape + part) {rms(c8(t + part)):6.2f}   (tape alone {rms(c8(t)):.2f})")
    full = (P.h @ U.T)[:, :8] / P.M[:, None]; tp = (P.h[:, o] @ G_TAPE.T)[:, :8] / P.M[:, None]
    print(f"  for scale, sd over states (minus mean of the 8 entries): tape piece {rms(c8(tp) - c8(tp).mean(0)):.2f}, rest {rms(c8(full - tp) - c8(full - tp).mean(0)):.2f}, "
          f"full score {rms(c8(full) - c8(full).mean(0)):.2f}\n")
