"""N2: the 715 splits of the 13 non-pulse units into 4 + 9 for the clock's readout cancellation (section 4.3)."""
import sys
from itertools import combinations
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from modes import RIGHT

Uo = U[:, ORD]; v = RIGHT["clock"]
def ratio(a, pos=slice(None)):
    b = [i for i in range(13) if i not in a]
    Ra, Rb = Uo[pos][:, list(a)] @ v[list(a)], Uo[pos][:, b] @ v[b]
    return np.linalg.norm(Ra + Rb) / (np.linalg.norm(Ra) + np.linalg.norm(Rb))
res = sorted((ratio(a), [ORD[i] for i in a]) for a in combinations(range(13), 4))
print("five smallest:", [(round(r, 4), u) for r, u in res[:5]])
print("median %.3f, max %.3f, n below 0.07: %d, below 0.08: %d" % (np.median([r for r, _ in res]), res[-1][0], sum(r < 0.07 for r, _ in res), sum(r < 0.08 for r, _ in res)))
print("|Uo v| / sum_i |Uo[:,i] v_i| =", np.linalg.norm(Uo @ v) / sum(np.linalg.norm(Uo[:, i] * v[i]) for i in range(13)))
