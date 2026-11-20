"""Mode coordinates of the resting map, in a fixed, documented gauge.

Resting map: A = W[ORD, ORD] (pulses 4, 5, 9 off, all 13 ordinary units on).
A = V diag(lam) V^-1.  Mode coordinates of a state h: m = L h[ORD] with L = V^-1.

Gauge (fixes the arbitrary scale/phase of each eigenvector):
  each mode coordinate is scaled so that the 'record kick' has coordinate exactly 1, where
  the record kick is the vector that a unit activation of unit 4 sends into the ordinary units
  at the next step, W[ORD, 4].  So after unit 4 fires with activation h4, the next resting state
  has mode coordinates  h4 * 1  (+ whatever else is there).  For a complex pair we keep the
  member with positive angle; its coordinate is a complex number z whose phase advances by
  +angle per resting step.

NAMES / indices (sorted by |lam|):
  level (1.0099), runner (0.9735), clock (0.951 @ 42.75), h2 (0.937 @ 83.28), alt (-0.928),
  h3 (0.837 @ 124.78), short (0.735 @ 42.99), trans (0.231), echo (0.001)
"""
from common import *

A = W[np.ix_(ORD, ORD)]
_lam, _V = np.linalg.eig(A)
_L = np.linalg.inv(_V)

# keep one member of each complex pair (positive angle), order by |lam| descending
_keep = [k for k in range(13) if _lam[k].imag >= -1e-12]
_keep.sort(key=lambda k: -abs(_lam[k]))
NAMES = ["level", "runner", "clock", "h2", "alt", "h3", "short", "trans", "echo"]
assert len(_keep) == len(NAMES)
# the names are assigned by rank of |lam| alone, so refuse any weights whose spectrum is not that of M_16,10
_ABS_LAM = [1.0099, 0.9735, 0.9509, 0.9370, 0.9283, 0.8373, 0.7350, 0.2310, 0.0010]
assert np.abs(np.abs(_lam[_keep]) - _ABS_LAM).max() < 1e-3, \
    f"modes.py: |lam| = {np.round(np.abs(_lam[_keep]), 4)} differ from M_16,10's {_ABS_LAM}; the mode NAMES would be mislabelled"

KICK4 = W[ORD, 4]                       # what unit 4 sends to the ordinary units
LAM = {}
LEFT = {}                               # mode name -> left (dual) vector on ORD (complex for pairs)
RIGHT = {}                              # mode name -> right eigenvector on ORD
for name, k in zip(NAMES, _keep):
    c = _L[k] @ KICK4                   # coordinate of the record kick in the raw gauge
    LAM[name] = _lam[k]
    LEFT[name] = _L[k] / c              # now LEFT @ KICK4 == 1
    RIGHT[name] = _V[:, k] * c
IS_COMPLEX = {n: abs(LAM[n].imag) > 1e-9 for n in NAMES}


def coords(H):
    """Mode coordinates of states H (..., 16) -> dict name -> array (...,) (complex for pairs)."""
    Ho = H[..., ORD]
    return {n: (Ho @ LEFT[n] if IS_COMPLEX[n] else np.real(Ho @ LEFT[n])) for n in NAMES}


def reconstruct(m):
    """Inverse of coords: ordinary-unit state from mode coordinates."""
    out = 0
    for n in NAMES:
        if IS_COMPLEX[n]:
            out = out + 2 * np.real(np.multiply.outer(m[n], RIGHT[n]))
        else:
            out = out + np.real(np.multiply.outer(m[n], RIGHT[n]))
    return out


if __name__ == "__main__":
    for n in NAMES:
        l = LAM[n]
        print(f"{n:7s} |lam| = {abs(l):.4f}  angle = {np.degrees(np.angle(l)):7.2f}")
    # sanity: reconstruct random states
    H = run(sample(1000, 0))
    m = coords(H)
    err = np.abs(reconstruct(m) - H[..., ORD]).max()
    print("max reconstruction error:", err)
    print("\nright eigenvectors in this gauge (state of the ordinary units per unit coordinate):")
    print("unit   " + "".join(f"{n:>16s}" for n in NAMES))
    for i, u in enumerate(ORD):
        row = ""
        for n in NAMES:
            v = RIGHT[n][i]
            row += (f"{abs(2*v):8.2f}@{np.degrees(np.angle(v)):6.0f}d" if IS_COMPLEX[n] else f"{v.real:16.3f}")
        print(f"{u:4d}   {row}")
