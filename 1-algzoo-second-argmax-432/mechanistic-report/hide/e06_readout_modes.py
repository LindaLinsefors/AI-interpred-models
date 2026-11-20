"""E06. How the readout U reads each resting mode, and the logit kernels of the three pulses.

R_k[p] = U[p, ORD] . RIGHT_k : logit at position p per unit mode coordinate (complex for pairs:
logit = 2 Re(z R_k[p])).  If the readout decodes 'age a -> position 9 - a' from a rotating mode
whose phase is  omega_k * a  (+const), then arg R_k[p] should be  +omega_k * p + const.
"""
from common import *
from modes import LEFT, RIGHT, NAMES, LAM, IS_COMPLEX

print("R_k[p] per mode (complex: 2|R|@phase)")
print("pos   " + "".join(f"{n:>15s}" for n in NAMES))
R = {n: U[:, ORD] @ RIGHT[n] for n in NAMES}
for p in range(10):
    row = ""
    for n in NAMES:
        v = R[n][p]
        row += (f"{2*abs(v):8.2f}@{np.degrees(np.angle(v)):5.0f}d" if IS_COMPLEX[n] else f"{v.real:15.2f}")
    print(f"{p:3d}   {row}")

print("\nphase slope of R_k over positions 0..7 (deg per position), vs the mode's own rotation angle:")
for n in NAMES:
    if not IS_COMPLEX[n]:
        continue
    ph = np.unwrap(np.angle(R[n][:8]))
    slope = np.polyfit(np.arange(8), ph, 1)[0]
    resid = ph - np.polyval(np.polyfit(np.arange(8), ph, 1), np.arange(8))
    print(f"  {n:6s}: slope = {np.degrees(slope):7.2f} deg/pos   rotation = {np.degrees(np.angle(LAM[n])):7.2f} deg/step"
          f"   rms phase residual = {np.degrees(np.sqrt((resid**2).mean())):5.1f} deg   amplitudes 2|R| = {np.round(2*np.abs(R[n][:8]),1)}")

A = W[np.ix_(ORD, ORD)]
Uo = U[:, ORD]
print("\nKernel of a unit-4 pulse (per unit of h4), free resting flow: logits at each position when the pulse is 'a' steps old")
print("(position 9-a is the pulse's own position; age 0 is read directly: U[:,4])")
print("age  own |" + "".join(f"   p{p}  " for p in range(10)))
for a in range(10):
    k = U[:, 4] if a == 0 else Uo @ np.linalg.matrix_power(A, a - 1) @ W[ORD, 4]
    own = 9 - a
    print(f" {a}   p{own}  |" + "".join(f"{v:7.2f}" for v in k))
for pu, name in ((5, "unit-5"), (9, "unit-9")):
    print(f"\nKernel of a {name} pulse (per unit activation), free resting flow; age = steps since the pulse fired;")
    print("the pulse fires one step after its input, so the input's own position is 9 - age - 1")
    print("age  own |" + "".join(f"   p{p}  " for p in range(10)))
    for a in range(9):
        k = U[:, pu] if a == 0 else Uo @ np.linalg.matrix_power(A, a - 1) @ W[ORD, pu]
        own = 9 - a - 1
        print(f" {a}   p{own}  |" + "".join(f"{v:7.2f}" for v in k))
