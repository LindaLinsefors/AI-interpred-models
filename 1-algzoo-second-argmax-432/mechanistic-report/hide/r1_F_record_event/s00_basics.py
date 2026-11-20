"""s00. Basic look at the weights that matter for a record: w, rows of W for pulses and unit 0,
columns of W for the pulses, mode coordinates of the pulse kicks. Re-verifies F2-F4 basics."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

np.set_printoptions(precision=3, suppress=True, linewidth=220)
print("w =", w)
print("\nW rows (incoming weights) for units 4, 5, 9, 0  [columns = source unit 0..15]")
for u in (4, 5, 9, 0, 1, 6, 7, 8):
    print(f"  W[{u:2d}] =", W[u])
print("\nW columns (outgoing) for pulses 4, 5, 9 [rows = target unit 0..15]")
for u in PULSE:
    print(f"  W[:, {u}] =", W[:, u])
print("\nU columns for pulses 4, 5, 9 (direct readout)")
for u in PULSE:
    print(f"  U[:, {u}] =", U[:, u])
print("\nU (rows positions, columns units)")
print(U)

print("\nmodes:")
for n in NAMES:
    print(f"  {n:7s} lam = {abs(LAM[n]):.4f} @ {np.degrees(np.angle(LAM[n])):7.2f} deg")
print("\nLEFT[mode] . W[ORD, pulse]  (complex as |.|@deg)")
for n in NAMES:
    row = []
    for pu in PULSE:
        c = LEFT[n] @ W[ORD, pu]
        row.append(f"{abs(c):7.3f}@{np.degrees(np.angle(c)):7.1f}" if IS_COMPLEX[n] else f"{c.real:8.3f}        ")
    cw = LEFT[n] @ w[ORD]
    row.append(f"w[ORD]: {abs(cw):7.3f}@{np.degrees(np.angle(cw)):7.1f}" if IS_COMPLEX[n] else f"w[ORD]: {cw.real:8.3f}")
    print(f"  {n:7s}: " + "   ".join(row))

# F4 check
A = W[np.ix_(ORD, ORD)]
print("\nA w[ORD] =", A @ w[ORD], " max |.| =", np.abs(A @ w[ORD]).max())
print("W[PULSE, ORD] w[ORD] =", W[np.ix_(PULSE, ORD)] @ w[ORD])
print("cos(w[ORD], echo RIGHT) =", abs(w[ORD] @ RIGHT["echo"].real) / np.linalg.norm(w[ORD]) / np.linalg.norm(RIGHT["echo"].real))
print("\npulse-pulse block W[PULSE, PULSE] (rows 4,5,9; cols 4,5,9):")
print(W[np.ix_(PULSE, PULSE)])
print("\naccuracy N(0,1) n=400k seed 1000:", accuracy(sample(400_000, 1000)))
