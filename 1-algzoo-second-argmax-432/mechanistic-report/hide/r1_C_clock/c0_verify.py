"""C0. Re-verify the lead's facts that arm C builds on (F1-F4, the gauge, F7 in brief)."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

deg = lambda c: np.degrees(np.angle(c))
ci = [ORD.index(u) for u in COMP]
wi = [ORD.index(u) for u in WAVE]

print("input weights w:", np.round(w, 3))
print("\neigenvalues of A = W[ORD, ORD]:")
for n in NAMES:
    print(f"  {n:7s} |lam| = {abs(LAM[n]):.5f}  angle = {deg(LAM[n]):8.3f}  LEFT.KICK4 = {LEFT[n] @ W[ORD, 4]:.6f}")

# biorthogonality and eigen-equations
for n in NAMES:
    assert np.allclose(LEFT[n] @ A, LAM[n] * LEFT[n], atol=1e-9)
    assert np.allclose(A @ RIGHT[n], LAM[n] * RIGHT[n], atol=1e-9)
    assert np.isclose(LEFT[n] @ RIGHT[n], 1, atol=1e-9)
print("eigen-equations and LEFT.RIGHT = 1 hold for all modes (1e-9)")

# F1: h_t = sum_p J[t,p] x_p and accuracy
x = sample(200_000, 1000)
H, J = traces(x[:2000])
print("\nF1 max |h - sum_p J x_p| =", np.abs(H - np.einsum("btpn,bp->btn", J, x[:2000])).max())
Hf = run(x)
print("accuracy N(0,1), n=200k, seed 1000:", (logits(Hf).argmax(1) == second_argmax(x)).mean())
m = coords(Hf)
print("reconstruction error:", np.abs(reconstruct(m) - Hf[..., ORD]).max())

# F4: input direction vs echo eigenvector
wo = w[ORD]
re = RIGHT["echo"].real
print("\nF4 cos(w[ORD], echo right vector) =", wo @ re / np.linalg.norm(wo) / np.linalg.norm(re))
print("   max |A w[ORD]| =", np.abs(A @ wo).max(), "  W[PULSE, ORD] w[ORD] =", W[np.ix_(PULSE, ORD)] @ wo)
print("   mode coordinates of w[ORD] (per unit input), |.|@deg:")
for n in NAMES:
    c = LEFT[n] @ wo
    print(f"     {n:7s} {abs(c):8.4f} @ {deg(c):7.1f}")

# clock left vector by unit
print("\nLEFT['clock'] by unit (|.|@deg), and share of squared norm")
L = LEFT["clock"]
for i, u in enumerate(ORD):
    print(f"  unit {u:2d} {'comp' if u in COMP else 'wave'}  {abs(L[i]):7.4f} @ {deg(L[i]):7.1f}   share {abs(L[i])**2 / np.sum(abs(L)**2):.4f}")
print("  comparator share of |LEFT clock|^2:", np.sum(abs(L[ci]) ** 2) / np.sum(abs(L) ** 2))

# coordinates of what each pulse unit sends to the ordinary units
print("\nmode coordinates of W[ORD, pulse] (per unit pulse activation), |.|@deg")
for pu in PULSE:
    print(f"  unit {pu}: " + "  ".join(f"{n} {abs(LEFT[n] @ W[ORD, pu]):6.3f}@{deg(LEFT[n] @ W[ORD, pu]):6.1f}" for n in NAMES))

# F7: readout of the clock
Uo = U[:, ORD]
Rc, Rw = Uo[:, ci] @ RIGHT["clock"][ci], Uo[:, wi] @ RIGHT["clock"][wi]
print("\nF7 clock readout |total| / (|comp| + |wave|) =", np.linalg.norm(Rc + Rw) / (np.linalg.norm(Rc) + np.linalg.norm(Rw)))
print("   2|U RIGHT_clock| per position:", np.round(2 * abs(Rc + Rw), 3))
print("   2|via comp| per position:     ", np.round(2 * abs(Rc), 3))
print("   U[:,ORD] RIGHT_level:", np.round((Uo @ RIGHT["level"]).real, 3))

# F9: two-pulse inputs
print()
for b in (-1.0, -0.3):
    for Sv in (0.3, 1.0, 1.4):
        ok_ = 0
        for pM in range(10):
            for pS in range(10):
                if pM == pS: continue
                xx = np.full(10, b); xx[pM] = 1.5; xx[pS] = Sv
                ok_ += int(logits(run(xx))[0].argmax() == pS)
        print(f"F9 two-pulse inputs, background {b}, M = 1.5, S = {Sv}: correct {ok_}/90")

# F5: level law by age
st = running_stats(x)
Mx, a1x = st["M1"], st["a1"]
lev = m["level"] / w[4]
print("F5 level/(K4 M) by age (M > 0.3): " + "  ".join(f"a={a}: {np.median(lev[(a1x == a) & (Mx > 0.3)] / Mx[(a1x == a) & (Mx > 0.3)]):.3f}" for a in range(10)))
