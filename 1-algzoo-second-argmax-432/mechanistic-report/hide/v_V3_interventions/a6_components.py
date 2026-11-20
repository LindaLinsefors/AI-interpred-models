"""More follow-ups.  Seeds 5800.. (I1 pool), 5820.. (I2 pool).

 1. I1: which mode components of the comparators hide the max? (remove the comparators' share of sets of modes)
 2. I1: is the wipe's effect carried by the pulse units' reaction at t0 + 1, or by the comparators themselves? (patching)
 3. I2: scale several modes together (is the clock 'the' pending correction or one of several?)
"""
from v3lib import *

out = Tee("a6_components.txt")
isC = np.isin(ORD_A, COMP)
mult = {n_: (2 if IS_COMPLEX[n_] else 1) for n_ in NAMES}


def comp_part(h, modes):
    """The comparators' share of the given modes' components of the states h (n, 16) -> (n, 4)."""
    d = 0
    for n_ in modes:
        z = h[:, ORD_A] @ LEFT[n_]
        d = d + mult[n_] * np.real(np.multiply.outer(z, RIGHT[n_]))[:, isC]
    return d


def remove_comp_part(modes, clip=True):
    def fn(h, rows):
        h = h.copy()
        h[:, COMP] -= comp_part(h, modes)
        return np.maximum(h, 0.0) if clip else h
    return fn


x, pM, pS, M, S, nb = gather_I1(500_000, 5800)
out(f"I1 pool: seeds 5800..{5800 + nb - 1}, n = {len(x)}")
H = run(x)
h9 = H[:, -1]
out("check: sum over all nine modes of the comparators' share = h[COMP]: max abs err "
    f"{np.abs(comp_part(h9, NAMES) - h9[:, COMP]).max():.2e}")
osc = ["clock", "h2", "alt", "h3", "short"]
SUBSETS = [("clock", ["clock"]), ("short", ["short"]), ("h2", ["h2"]), ("runner", ["runner"]), ("level", ["level"]), ("echo", ["echo"]), ("trans", ["trans"]),
           ("clock+short", ["clock", "short"]), ("clock+h2", ["clock", "h2"]), ("clock+short+h2", ["clock", "short", "h2"]),
           ("all 5 oscillating", osc), ("the 4 oscillating without clock", [m for m in osc if m != "clock"]),
           ("all but level,runner,echo", [m for m in NAMES if m not in ("level", "runner", "echo")]),
           ("all but level", [m for m in NAMES if m != "level"]), ("all nine (= wipe)", NAMES)]

out("\n[1a] readout at the last step: P(argmax = pM) after removing from the logits the comparators' share of the listed modes;")
out("     by age k = 1..9 of the max.  Second block: the same with position 9 (the newest input) excluded from the argmax.")
full = h9 @ U.T
for label, modes in SUBSETS:
    s = full - comp_part(h9, modes) @ U[:, COMP].T
    s2 = s.copy(); s2[:, 9] = -np.inf
    out(f"  {label:34s} " + " ".join(f"{np.mean(s[pM == 9 - k].argmax(1) == 9 - k):.3f}" for k in range(1, 10))
        + "   | excl. 9: " + " ".join(f"{np.mean(s2[pM == 9 - k].argmax(1) == 9 - k):.3f}" for k in range(1, 10)))

out("\n[1b] dynamic version: remove the comparators' share of the listed modes at t0 = pM + k <= 7 (clipped at 0), continue.")
out("     P(ans = pM) for k = 1..7   |   P(ans = t0)   |   P(runner-up)")
for label, modes in SUBSETS:
    r1, r2, r3 = [], [], []
    for k in range(1, 8):
        v = pM + k <= 7
        t0 = (pM + k)[v]
        a = answer(run_iv(x[v], t0, remove_comp_part(modes)))
        r1.append(f"{np.mean(a == pM[v]):.3f}"); r2.append(f"{np.mean(a == t0):.3f}"); r3.append(f"{np.mean(a == pS[v]):.3f}")
    out(f"  {label:34s} " + " ".join(r1) + "   | " + " ".join(r2) + "   | " + " ".join(r3))

# ---------------------------------------------------------------- 2. patching
out("\n[2] patching, wipe at t0 = pM + k <= 7; P(ans = pM) | P(runner-up) | P(ans = t0); k = 1..6")


def run_two(xv, t0, fn0, fn1):
    """fn0 at step t0, fn1 at step t0 + 1."""
    B = len(xv)
    h = np.zeros((B, N)); Hh = np.empty((B, T, N))
    for t in range(T):
        h = np.maximum(xv[:, t, None] * w + h @ W.T, 0.0)
        rows = np.flatnonzero(t0 == t)
        if rows.size and fn0 is not None:
            h[rows] = fn0(h[rows], rows)
        rows = np.flatnonzero(t0 + 1 == t)
        if rows.size and fn1 is not None:
            h[rows] = fn1(h[rows], rows)
        Hh[:, t] = h
    return Hh


def wipe(h, rows):
    h = h.copy(); h[:, COMP] = 0.0
    return h


variants = ["wipe only", "wipe, pulses 4,5,9 at t0+1 set to their unperturbed values", "no wipe, pulses at t0+1 taken from the wiped run",
            "wipe, comparators at t0+1 set to their unperturbed values", "wipe, wave units at t0+1 set to their unperturbed values",
            "no wipe, comparators at t0+1 taken from the wiped run", "no wipe, wave units at t0+1 taken from the wiped run",
            "no wipe, comparators at t0+1 set to 0 (wipe one step later)"]
rows_out = {v_: [] for v_ in variants}
pulse_stats = []
for k in range(1, 7):
    v = pM + k <= 7
    xv, t0, pMv, pSv = x[v], (pM + k)[v], pM[v], pS[v]
    Hu = H[v]
    Hw = run_two(xv, t0, wipe, None)
    rr = np.arange(len(xv))
    u1, w1 = Hu[rr, t0 + 1], Hw[rr, t0 + 1]
    pulse_stats.append((np.median(u1[:, 5] / M[v]), np.median(w1[:, 5] / M[v]), np.median(u1[:, 9] / M[v]), np.median(w1[:, 9] / M[v]),
                        np.mean(w1[:, 4] > 0), np.median(np.linalg.norm(w1[:, WAVE] - u1[:, WAVE], axis=1) / np.linalg.norm(u1[:, WAVE], axis=1)),
                        np.median(np.linalg.norm(w1[:, COMP] - u1[:, COMP], axis=1) / np.maximum(np.linalg.norm(u1[:, COMP], axis=1), 1e-9))))

    def setter(src, units):
        def fn(h, rows):
            h = h.copy(); h[:, units] = src[rows][:, units]
            return h
        return fn

    def zero_comp(h, rows):
        h = h.copy(); h[:, COMP] = 0
        return h
    runs = [Hw, run_two(xv, t0, wipe, setter(u1, PULSE)), run_two(xv, t0, None, setter(w1, PULSE)),
            run_two(xv, t0, wipe, setter(u1, COMP)), run_two(xv, t0, wipe, setter(u1, WAVE)),
            run_two(xv, t0, None, setter(w1, COMP)), run_two(xv, t0, None, setter(w1, WAVE)), run_two(xv, t0, None, zero_comp)]
    for name, Hh in zip(variants, runs):
        a = answer(Hh[:, -1])
        rows_out[name].append(f"{np.mean(a == pMv):.3f}|{np.mean(a == pSv):.3f}|{np.mean(a == t0):.3f}")
for name in variants:
    out(f"  {name:62s} " + "  ".join(rows_out[name]))
out("  at t0 + 1 (k = 1..6): median h5/M unperturbed, wiped; median h9/M unperturbed, wiped; P(unit 4 fires | wiped); median relative change of the")
out("  wave units and of the comparators at t0 + 1 caused by the wipe:")
for k, ps in zip(range(1, 7), pulse_stats):
    out(f"    k={k}: h5 {ps[0]:.2f} -> {ps[1]:.2f};  h9 {ps[2]:.2f} -> {ps[3]:.2f};  P(h4 > 0) = {ps[4]:.3f};  wave {ps[5]:.3f};  comparators {ps[6]:.3f}")

# ---------------------------------------------------------------- 3. I2, several modes together
x2, tr, p0, q0, M0, S0, nb2 = gather_I2(250_000, 5820)
out(f"\n[3] I2 pool: seeds 5820..{5820 + nb2 - 1}, n = {len(x2)}.  z' = f z for several modes at once (at t_r - 1); raw | clipped")


def multi_scale(modes, f, clip=False, units=None):
    def fn(h, rows):
        h0 = h
        h = h.copy()
        for n_ in modes:
            z = h0[:, ORD_A] @ LEFT[n_]
            d = mult[n_] * np.real(np.multiply.outer(z * (f - 1.0), RIGHT[n_]))
            if units is not None:
                d = d * np.isin(ORD_A, units)
            h[:, ORD_A] += d
        return np.maximum(h, 0.0) if clip else h
    return fn


for f in (0, 0.5):
    for label, modes in (("clock", ["clock"]), ("h2", ["h2"]), ("short", ["short"]), ("h2+h3+short+alt (no clock)", [m for m in osc if m != "clock"]),
                         ("all 5 oscillating", osc), ("runner", ["runner"]), ("clock+runner", ["clock", "runner"])):
        res = []
        for clip in (False, True):
            a = answer(run_iv(x2, tr - 1, multi_scale(modes, f, clip)))
            res.append(f"P(p0) = {np.mean(a == p0):.4f}  P(t_r) = {np.mean(a == tr):.4f}  P(q0) = {np.mean(a == q0):.4f}")
        out(f"  f = {f:<4} {label:28s}: " + "  |  ".join(res))
out("  wave-unit part only (the part that survives a record):")
for f in (0, 0.5):
    for label, modes in (("clock", ["clock"]), ("h2+h3+short+alt (no clock)", [m for m in osc if m != "clock"]), ("all 5 oscillating", osc), ("runner", ["runner"]), ("level", ["level"])):
        a = answer(run_iv(x2, tr - 1, multi_scale(modes, f, True, units=WAVE)))
        out(f"  f = {f:<4} {label:28s}: P(p0) = {np.mean(a == p0):.4f}  P(t_r) = {np.mean(a == tr):.4f}  P(q0) = {np.mean(a == q0):.4f}")
