"""Referee S, check 4: section 3.5 says that removing the level or the runner mode one step before a maximum is
overtaken destroys the answer 'since these hold the thresholds'.  Test: remove the mode at t_r - 1 (selection of
e18 / claim I2), then put single groups of units back to their unperturbed values at later steps.
If the loss goes through the thresholds, restoring the units that the thresholds gate should restore the answer."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT, IS_COMPLEX

B = 1_500_000
x = sample(B, 9401)
st = running_stats(x)
truth = second_argmax(x)
ar = np.arange(B)
tr = x.argmax(1)
ok = (tr >= 3) & (tr <= 8)
tm = np.maximum(tr - 1, 0)
p0, q0, M0, S0 = st["p1"][ar, tm], st["p2"][ar, tm], st["M1"][ar, tm], st["M2"][ar, tm]
ok &= (tr - 1 - p0 >= 2) & (S0 > 0.2) & (M0 - S0 > 0.1) & (truth == p0) & (st["p3"][:, 9] == q0)
x, tr, p0, q0 = x[ok], tr[ok], p0[ok], q0[ok]
n = len(x)
Hb = run(x)
print(f"selected {n}; unperturbed P(answer = old maximum) = {(logits(Hb).argmax(1) == p0).mean():.3f}")
OI = np.array(ORD)


def remove(h, mode):
    c = h[:, OI] @ LEFT[mode]
    d = 2 * np.real(np.outer(c, RIGHT[mode])) if IS_COMPLEX[mode] else np.real(np.outer(c, RIGHT[mode]))
    h = h.copy(); h[:, OI] -= d
    return h


def go(mode, patches=()):
    """patches: tuples (offset from t_r, list of units) put back to the unperturbed run's values."""
    h = np.zeros((n, N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        sel = np.where(tr - 1 == t)[0]
        if len(sel):
            h[sel] = remove(h[sel], mode)
        for off, units in patches:
            sel = np.where(tr + off == t)[0]
            if len(sel):
                h[np.ix_(sel, units)] = Hb[sel, t][:, units]
    a = (h @ U.T).argmax(1)
    return (a == p0).mean(), (a == tr).mean(), (a == q0).mean()


VAR = [("nothing put back", ()),
       ("comparators + unit 4 at t_r", ((0, COMP + [4]),)),
       ("unit 5 at t_r, t_r+1", ((0, [5]), (1, [5]))),
       ("unit 9 at t_r, t_r+1", ((0, [9]), (1, [9]))),
       ("units 5, 9 at t_r, t_r+1", ((0, [5, 9]), (1, [5, 9]))),
       ("comp + 4 at t_r and 5, 9 at t_r, t_r+1", ((0, COMP + [4, 5, 9]), (1, [5, 9]))),
       ("comp + 4, 5, 9 at t_r and t_r+1", ((0, COMP + [4, 5, 9]), (1, COMP + [4, 5, 9]))),
       ("all pulses + comparators at t_r..t_r+2", ((0, COMP + PULSE), (1, COMP + PULSE), (2, COMP + PULSE))),
       ("wave units at t_r (control)", ((0, WAVE),))]
for mode in ("runner", "level", "clock"):
    print(f"\nmode removed at t_r - 1: {mode}      P(old maximum) / P(new maximum) / P(old runner-up)")
    for lab, pt in VAR:
        r = go(mode, pt)
        print(f"  {lab:45s} {r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}")
