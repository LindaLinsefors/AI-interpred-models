"""E07. Is the main clock 'hidden' from the readout?  Split the readout of each mode into the part
read through the comparators (1,6,7,8) and the part read through the wave units."""
from common import *
from modes import LEFT, RIGHT, NAMES, LAM, IS_COMPLEX

ci = [ORD.index(u) for u in COMP]
wi = [ORD.index(u) for u in WAVE]
Uo = U[:, ORD]
for n in NAMES:
    Rc = Uo[:, ci] @ RIGHT[n][ci]
    Rw = Uo[:, wi] @ RIGHT[n][wi]
    Rt = Rc + Rw
    f = 2 if IS_COMPLEX[n] else 1
    print(f"\nmode {n}: readout per unit coordinate ({'2|R|@phase' if IS_COMPLEX[n] else 'R'}), positions 0..9")
    for label, R in (("via comparators", Rc), ("via wave units ", Rw), ("total          ", Rt)):
        if IS_COMPLEX[n]:
            print(f"  {label}: " + " ".join(f"{f*abs(v):5.2f}@{np.degrees(np.angle(v)):4.0f}" for v in R))
        else:
            print(f"  {label}: " + " ".join(f"{v.real:10.2f}" for v in R))
    print(f"  norm ratio |total| / (|comp| + |wave|) = {np.linalg.norm(Rt) / (np.linalg.norm(Rc) + np.linalg.norm(Rw)):.3f}")
