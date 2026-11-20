"""E27. Alignment of the comparators' vote with the maximum's position, from the weights plus the
measured phase law of the clock.  Vote of the comparators for a clock coordinate z:  V[p] = 2 Re(z R_c[p]),
R_c[p] = U[p, COMP] . RIGHT_clock[COMP].  With arg z = phi(a) and the max at position p_M = 9 - a,
the phase of the vote at the max's own position is phi(a) + arg R_c[9 - a]."""
from common import *
from modes import RIGHT, coords

ci = [ORD.index(u) for u in COMP]
Rc = U[:, COMP] @ RIGHT["clock"][ci]
# measured phase law at t = 9
x = sample(600_000, 2727)
H = run(x)
st = running_stats(x)
z = coords(H[:, 9])["clock"] / 10.157
M, S, a1 = st["M1"][:, 9], st["M2"][:, 9], st["a1"][:, 9]
print(" age a | position of max | measured phi(a) | 2|R_c| at that position | phase of the vote there | cos")
for a in range(1, 10):
    s = (a1 == a) & (S > 0.05)
    X = np.column_stack([(M - S)[s], S[s]]).astype(complex)
    b, *_ = np.linalg.lstsq(X, z[s], rcond=None)
    phi = np.degrees(np.angle(b[0]))
    p = 9 - a
    ph = (phi + np.degrees(np.angle(Rc[p]))) % 360
    print(f"   {a}   |       {p}         |    {phi % 360:6.1f}      |        {2*abs(Rc[p]):5.2f}            |        {ph:6.1f}          | {np.cos(np.radians(ph)):+.3f}")
