"""Reviewer R2, test 1.  Is "the comparators' vote V hides the maximum" the same thing as "the clock hides it"?

Setting of the lead's e17: the final max is at pM, intervene once at t0 = pM + k (no new max afterwards),
final S > 0.2 and final gap > 0.2, real inputs afterwards.  Interventions on h_t0:
  wipe      : h[1,6,7,8] = 0                                  (the lead's e17)
  clock0    : remove the clock mode entirely (state moved along RIGHT_clock, both halves)
  clockC0   : remove only the comparators' share of the clock (h[COMP] -= 2Re(z R_clock)[COMP])
  clockW0   : remove only the wave units' share of the clock
  wipe_noP  : wipe, and at t0+1 hold units 5 and 9 at their values in the unmodified run (no 'phantom record' pulses)
Reported: P(answer = max), P(answer = runner-up), by k, separately for t0 <= 8 and t0 = 9.
Also: after 'wipe', do units 5 and 9 fire at t0+1 and is unit 0 cut at t0+2, as after a real new maximum?
python -B t1_wipe_vs_clock.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT

x = sample(600_000, 9101)
st = running_stats(x)
truth = second_argmax(x)
pM = x.argmax(1)
Href, Zref = run(x, return_pre=True)
oi = np.array(ORD)
ci = [ORD.index(u) for u in COMP]
wi = [ORD.index(u) for u in WAVE]
Lc, Rc = LEFT["clock"], RIGHT["clock"]


def run_mod(xs, t0, kind, Hr):
    B = len(xs); h = np.zeros((B, N)); rec = {}
    idx = np.arange(B)
    for t in range(T):
        z = xs[:, t, None] * w + h @ W.T
        h = np.maximum(z, 0)
        if kind == "wipe_noP":
            sel = np.where(t0 + 1 == t)[0]
            h[sel, 5] = Hr[sel, t, 5]; h[sel, 9] = Hr[sel, t, 9]
        sel = np.where(t0 == t)[0]
        if len(sel):
            if kind in ("wipe", "wipe_noP"):
                h[np.ix_(sel, COMP)] = 0.0
            elif kind in ("clock0", "clockC0", "clockW0"):
                zc = h[sel][:, oi] @ Lc
                d = 2 * np.real(np.multiply.outer(zc, Rc))            # (n, 13)
                if kind == "clockC0":
                    keep = np.zeros(13); keep[ci] = 1; d = d * keep
                if kind == "clockW0":
                    keep = np.zeros(13); keep[wi] = 1; d = d * keep
                h[np.ix_(sel, oi)] -= d
        # record pulses one step after t0 and unit 0 two steps after
        s1 = np.where(t0 + 1 == t)[0]; s2 = np.where(t0 + 2 == t)[0]
        if len(s1): rec.setdefault("h5", np.full(B, np.nan))[s1] = h[s1, 5]; rec.setdefault("h9", np.full(B, np.nan))[s1] = h[s1, 9]
        if len(s2): rec.setdefault("z0", np.full(B, np.nan))[s2] = z[s2, 0]
    return (h @ U.T).argmax(1), rec


kinds = ["wipe", "clock0", "clockC0", "clockW0", "wipe_noP"]
print("P(answer = max) / P(answer = runner-up); n in brackets.  Final S > 0.2, final gap > 0.2, intervention at t0 = pM + k")
for late in (False, True):
    print(f"\n=== t0 {'= 9' if late else '<= 8'} ===")
    print("  k |      none        " + "".join(f"{k_:>18s}" for k_ in kinds))
    for k in range(1, 8):
        t0 = pM + k
        ok = (st["M2"][:, 9] > 0.2) & (st["M1"][:, 9] - st["M2"][:, 9] > 0.2) & ((t0 == 9) if late else (t0 <= 8))
        if ok.sum() < 500: continue
        xs, t0s, tm, tr, Hr = x[ok], t0[ok], pM[ok], truth[ok], Href[ok]
        base = (Hr[:, -1] @ U.T).argmax(1)
        row = f"  {k} | {np.mean(base == tm):.3f}/{np.mean(base == tr):.3f} [{ok.sum():6d}]"
        for kind in kinds:
            a, rec = run_mod(xs, t0s, kind, Hr)
            row += f"     {np.mean(a == tm):.3f}/{np.mean(a == tr):.3f} "
        print(row)

print("\n=== is the wipe a 'phantom record'?  (k = 2..5, t0 <= 6 so that t0+2 exists) ===")
for k in (2, 3, 4, 5):
    t0 = pM + k
    ok = (st["M2"][:, 9] > 0.2) & (st["M1"][:, 9] - st["M2"][:, 9] > 0.2) & (t0 <= 6)
    xs, t0s, Hr, Zr = x[ok], t0[ok], Href[ok], Zref[ok]
    ar = np.arange(ok.sum())
    a, rec = run_mod(xs, t0s, "wipe", Hr)
    Mv = st["M1"][ok, 9]
    print(f"  k={k} (n={ok.sum()}): unit 5 at t0+1: on {np.mean(rec['h5'] > 0):.3f} (unmodified {np.mean(Hr[ar, t0s + 1, 5] > 0):.3f}), "
          f"median h5/M {np.median(rec['h5'] / Mv):.2f} (unmodified {np.median(Hr[ar, t0s + 1, 5] / Mv):.2f}); "
          f"unit 9 on {np.mean(rec['h9'] > 0):.3f} (unmodified {np.mean(Hr[ar, t0s + 1, 9] > 0):.3f}), median h9/M {np.median(rec['h9'] / Mv):.2f}; "
          f"unit 0 OFF at t0+2: {np.mean(rec['z0'] <= 0):.3f} (unmodified {np.mean(Zr[ar, t0s + 2, 0] <= 0):.3f})")

# ---------------------------------------------------------------------------------------------
print("\n=== static check (no dynamics): which part of the state makes m rank the max first? ===")
print("m = readout without comparator columns; older inputs only; S_t > 0 and max not the current input (as in e15b).")
print("m_noclock = m minus the wave units' share of the clock;  s_noclock = s minus the whole clock mode.")
H = Href
Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
zc = H[..., oi] @ Lc                                        # (B, T)
dclock = 2 * np.real(zc[..., None] * Rc)                    # (B, T, 13) clock part of the state
Uo = U[:, oi]
keepW = np.zeros(13); keepW[wi] = 1
S_all = H @ U.T
M_all = S_all - H @ Uc.T
M_nc = M_all - (dclock * keepW) @ Uo.T
S_nc = S_all - dclock @ Uo.T
print("   t | P(argmax m = max)  P(argmax m_noclock = max)  P(argmax m_noclock = runner-up) | P(argmax s = runner-up)  P(argmax s_noclock = runner-up)  P(argmax s_noclock = max)")
for t in range(2, 10):
    valid = (np.arange(10) >= 9 - t) & (np.arange(10) <= 8)
    ok = (st["M2"][:, t] > 0) & (st["p1"][:, t] < t) & (st["p2"][:, t] < t)
    am = lambda Lg: np.where(valid, Lg[:, t], -np.inf).argmax(1) + t - 9
    p1, p2 = st["p1"][ok, t], st["p2"][ok, t]
    print(f"   {t} |      {np.mean(am(M_all)[ok] == p1):.3f}               {np.mean(am(M_nc)[ok] == p1):.3f}                     {np.mean(am(M_nc)[ok] == p2):.3f}"
          f"              |        {np.mean(am(S_all)[ok] == p2):.3f}                  {np.mean(am(S_nc)[ok] == p2):.3f}                        {np.mean(am(S_nc)[ok] == p1):.3f}")
print("\n  the same at t = 9 by relative gap (M - S)/M:")
t = 9
valid = (np.arange(10) <= 8)
am = lambda Lg: np.where(valid, Lg[:, t], -np.inf).argmax(1)
for lo, hi in ((0, .05), (.05, .15), (.15, .3), (.3, .5), (.5, 1.0)):
    g = (st["M1"][:, t] - st["M2"][:, t]) / st["M1"][:, t]
    ok = (st["M2"][:, t] > 0) & (st["p1"][:, t] < t) & (st["p2"][:, t] < t) & (g > lo) & (g <= hi)
    p1, p2 = st["p1"][ok, t], st["p2"][ok, t]
    print(f"   gap ({lo:.2f},{hi:.2f}]: m->max {np.mean(am(M_all)[ok] == p1):.3f}   m_noclock->max {np.mean(am(M_nc)[ok] == p1):.3f}   m_noclock->runner-up {np.mean(am(M_nc)[ok] == p2):.3f}"
          f"   | s->runner-up {np.mean(am(S_all)[ok] == p2):.3f}   s_noclock->runner-up {np.mean(am(S_nc)[ok] == p2):.3f}   n={ok.sum()}")
