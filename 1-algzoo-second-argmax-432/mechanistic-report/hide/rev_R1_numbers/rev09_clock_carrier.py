"""Reviewer R1: section 3.5 item 2 says 'only the clock coordinate of the state is changed'.  Moving the state along the
clock's right vector changes the comparators AND the wave units.  Which part does the work?  (e18's sample and selection.)
  full : h[ORD] -= 2 Re(z R_clock)                      (the report's edit, radius x0)
  wave : only the wave units' entries of that edit       (clock coordinate barely changes: it is read from the comparators)
  comp : only the comparators' entries of that edit
  z0   : smallest change of the four comparators alone that makes the clock coordinate exactly 0 (wave units untouched)
Run:  python -B rev09_clock_carrier.py > rev09_clock_carrier_out.txt"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT, LAM
x = sample(1_500_000, 1818)
st = running_stats(x); truth = second_argmax(x); B = len(x); ar = np.arange(B)
tr = x.argmax(1); tm1 = np.maximum(tr - 1, 0)
p0 = st["p1"][ar, tm1]; q0 = st["p2"][ar, tm1]; M0 = st["M1"][ar, tm1]; S0 = st["M2"][ar, tm1]
g = tr - 1 - p0
ok = (tr >= 3) & (g >= 2) & (S0 > 0.2) & (M0 - S0 > 0.1) & (truth == p0) & (st["p3"][:, 9] == q0) & (tr <= 8)
xs, trs, p0s, q0s = x[ok], tr[ok], p0[ok], q0[ok]
ic = np.array([ORD.index(u) for u in COMP]); iw = np.array([ORD.index(u) for u in WAVE])
Lc = LEFT["clock"]; Rc = RIGHT["clock"]
# minimal-norm comparator change d (4 reals) with Lc[ic] . d = -z  (two real equations)
Amat = np.vstack([Lc[ic].real, Lc[ic].imag])            # 2 x 4
Pinv = np.linalg.pinv(Amat)                              # 4 x 2


def run_mod(mode):
    h = np.zeros((len(xs), N)); zafter = None; neg = None
    for t in range(T):
        h = np.maximum(xs[:, t, None] * w + h @ W.T, 0)
        s = np.where(trs - 1 == t)[0]
        if len(s) and mode != "none":
            ho = h[s][:, ORD]
            z = ho @ Lc
            d = -2 * np.real(np.multiply.outer(z, Rc))  # full edit on the 13 ordinary units
            if mode == "wave":
                d[:, ic] = 0
            elif mode == "comp":
                d[:, iw] = 0
            elif mode == "z0":
                d = np.zeros_like(d)
                d[:, ic] = -(np.column_stack([z.real, z.imag]) @ Pinv.T)
            ho2 = ho + d
            h[np.ix_(s, ORD)] = ho2
    return (h @ U.T).argmax(1)


def clock_left(mode):
    """|z after| / |z before| and share of edited states with a negative entry (first matching step only, for reporting)."""
    h = np.zeros((len(xs), N)); ratio = np.full(len(xs), np.nan); neg = np.zeros(len(xs), bool)
    for t in range(T):
        h = np.maximum(xs[:, t, None] * w + h @ W.T, 0)
        s = np.where(trs - 1 == t)[0]
        if len(s):
            ho = h[s][:, ORD]; z = ho @ Lc
            d = -2 * np.real(np.multiply.outer(z, Rc))
            if mode == "wave": d[:, ic] = 0
            elif mode == "comp": d[:, iw] = 0
            elif mode == "z0":
                d = np.zeros_like(d); d[:, ic] = -(np.column_stack([z.real, z.imag]) @ Pinv.T)
            ho2 = ho + d
            ratio[s] = np.abs(ho2 @ Lc) / np.abs(z); neg[s] = (ho2 < 0).any(1)
    return np.nanmedian(ratio), neg.mean()


print(f"n = {ok.sum()} (e18's selection)")
print("edit            | P(answer = old max) | P(old runner-up) | P(new max) | median |z after|/|z before| | share of edited states with a negative unit")
for mode, lab in (("none", "none           "), ("full", "full (report)  "), ("wave", "wave units only"), ("comp", "comparators only"), ("z0", "comparators, z=0")):
    a = run_mod(mode)
    rr = clock_left(mode) if mode != "none" else (1.0, 0.0)
    print(f"{lab} |        {np.mean(a == p0s):.3f}        |      {np.mean(a == q0s):.3f}       |   {np.mean(a == trs):.3f}    |        {rr[0]:.3f}        |   {rr[1]:.3f}")
