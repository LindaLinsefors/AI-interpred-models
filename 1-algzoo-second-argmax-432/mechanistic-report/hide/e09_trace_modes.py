"""E09. Two-pulse inputs: exact split of the final logits at the two marked positions into
(contribution of M's trace) + (contribution of S's trace) + (background), per mode."""
from common import *
from modes import coords, RIGHT, LEFT, NAMES, IS_COMPLEX

Uo = U[:, ORD]

def mode_logits(vec16):
    """vec16: a state vector (16,). Returns dict mode -> logits (10,) plus pulses."""
    out = {}
    ho = vec16[ORD]
    for n in NAMES:
        c = ho @ LEFT[n]
        v = c * RIGHT[n]
        out[n] = Uo @ (2 * np.real(v) if IS_COMPLEX[n] else np.real(v))
    out["pulses"] = U[:, PULSE] @ vec16[PULSE]
    return out

def show(pM, pS, M=1.5, S=1.0, b=-1.0):
    x = np.full(10, b); x[pM] = M; x[pS] = S
    H, J = traces(x)
    J9 = J[0, 9]                                   # (p, 16)
    partM, partS = J9[pM] * M, J9[pS] * S
    partB = H[0, 9] - partM - partS
    lg = U @ H[0, 9]
    print(f"\n=== M={M} at {pM}, S={S} at {pS}, background {b}:  logit[pS]={lg[pS]:.1f}, logit[pM]={lg[pM]:.1f}, "
          f"best other={np.max(np.delete(lg, [pM, pS])):.1f} at {int(np.argmax(np.where(np.isin(np.arange(10), [pM, pS]), -1e9, lg)))}")
    print(f"{'':10s}|  logit at pS={pS} from (M trace, S trace, bg)  |  logit at pM={pM} from (M trace, S trace, bg)  | margin pS-pM")
    tot = np.zeros(7)
    dM, dS, dB = mode_logits(partM), mode_logits(partS), mode_logits(partB)
    for k in list(NAMES) + ["pulses"]:
        row = [dM[k][pS], dS[k][pS], dB[k][pS], dM[k][pM], dS[k][pM], dB[k][pM]]
        mg = sum(row[:3]) - sum(row[3:])
        tot += np.array(row + [mg])
        print(f"{k:10s}|  {row[0]:8.1f} {row[1]:8.1f} {row[2]:8.1f}              |  {row[3]:8.1f} {row[4]:8.1f} {row[5]:8.1f}              | {mg:7.1f}")
    print(f"{'total':10s}|  {tot[0]:8.1f} {tot[1]:8.1f} {tot[2]:8.1f}              |  {tot[3]:8.1f} {tot[4]:8.1f} {tot[5]:8.1f}              | {tot[6]:7.1f}")

for pM, pS in ((6, 2), (2, 6), (3, 1), (1, 3), (5, 4), (4, 5), (7, 0), (0, 7)):
    show(pM, pS)
