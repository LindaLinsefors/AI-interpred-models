"""E04. Left (dual) vectors of the resting modes: which units does each mode coordinate read?
And the comparator-zeroing projector in mode coordinates: where does each mode's content go
when the four comparators are clipped to zero (as at a new maximum)?
"""
from common import *
from modes import LEFT, RIGHT, NAMES, LAM, IS_COMPLEX

print("LEFT vectors (mode coordinate = LEFT . h[ORD]); complex entries as |.|@phase")
print("unit   " + "".join(f"{n:>16s}" for n in NAMES))
for i, u in enumerate(ORD):
    row = ""
    for n in NAMES:
        v = LEFT[n][i]
        row += (f"{abs(v):8.3f}@{np.degrees(np.angle(v)):6.0f}d" if IS_COMPLEX[n] else f"{v.real:16.3f}")
    tag = " <- comparator" if u in COMP else ""
    print(f"{u:4d}   {row}{tag}")

print("\nfraction of each LEFT vector's squared norm that sits on the comparators 1,6,7,8:")
ci = [ORD.index(u) for u in COMP]
for n in NAMES:
    v = LEFT[n]
    print(f"  {n:7s}: {np.sum(abs(v[ci])**2)/np.sum(abs(v)**2):.3f}")

# Real 13x13 bases: stack real modes and (Re, Im) of complex ones.
def real_basis():
    Lr, Rr, labels = [], [], []
    for n in NAMES:
        if IS_COMPLEX[n]:
            # h = 2 Re(z v);  z = L.h  ->  coordinates (Re z, Im z), basis (2 Re v, -2 Im v)
            Lr += [LEFT[n].real, LEFT[n].imag]
            Rr += [2 * RIGHT[n].real, -2 * RIGHT[n].imag]
            labels += [n + ".re", n + ".im"]
        else:
            Lr.append(LEFT[n].real); Rr.append(RIGHT[n].real); labels.append(n)
    return np.array(Lr), np.array(Rr).T, labels

Lr, Rr, labels = real_basis()
assert np.allclose(Lr @ Rr, np.eye(13), atol=1e-9)
E = np.eye(13)
E[ci, ci] = 0                      # zero the comparators
P = Lr @ E @ Rr                    # in mode coordinates
print("\nP = comparator-zeroing map in mode coordinates (row = mode coordinate after, column = mode coordinate before)")
print(" " * 10 + "".join(f"{l:>9s}" for l in labels))
for i, l in enumerate(labels):
    print(f"{l:>10s}" + "".join(f"{P[i, j]:9.3f}" for j in range(13)))
np.savez(HIDE / "e04_real_basis.npz", Lr=Lr, Rr=Rr, labels=np.array(labels), P=P)
