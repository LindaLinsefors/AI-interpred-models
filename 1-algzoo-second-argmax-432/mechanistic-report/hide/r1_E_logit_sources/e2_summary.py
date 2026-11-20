"""E2 compact summaries for findings.md (markdown tables).  [EXACT]"""
from src import *


def summ(K, sh, ages=range(10)):
    rows = []
    for a in ages:
        own = 9 - a - sh
        if not 0 <= own <= 9:
            continue
        k = K[a]
        rows.append((a, own, k[own], int(k.argmax()) - own, k.max(), int(k.argmin()) - own, k.min(), np.linalg.norm(k - k.mean())))
    return rows


def md(name, K, sh):
    print(f"\n**{name}**\n")
    print("| age | own pos | K at own | max: offset, value | min: offset, value | norm of centred K |")
    print("|---|---|---|---|---|---|")
    for a, own, v, dmax, vmax, dmin, vmin, nrm in summ(K, sh):
        print(f"| {a} | {own} | {v:.2f} | {dmax:+d}, {vmax:.2f} | {dmin:+d}, {vmin:.2f} | {nrm:.2f} |")


md("raw input (per unit $x_s$)", KX, 0)
md("pulse 4 (per unit $h_s[4]$)", KP[0], 0)
md("pulse 5 (per unit $h_s[5]$; own = $s-1$)", KP[1], 1)
md("pulse 9 (per unit $h_s[9]$; own = $s-1$)", KP[2], 1)
for j in COMP:
    md(f"comparator {j} clip (per unit excess $x_s-\theta_{j}$)", abs(w[j]) * KC[ORD.index(j)], 0)
KS = sum(abs(w[j]) * KC[ORD.index(j)] for j in COMP)
md("sum of the four comparator kernels (per unit of a common excess)", KS, 0)
for i in WAVE:
    md(f"wave unit {i} clip (per unit clip amount)", KC[ORD.index(i)], 0)

print("\n\ncheck: max |KS + KX - Uo A^a w_wave| =", max(np.abs(KS[a] + KX[a] - Uo @ APOW[a] @ np.where(np.isin(ORD, WAVE), w[ORD], 0)).max() for a in range(10)))
print("max |KS + KX| for a >= 1:", np.abs((KS + KX)[1:]).max(), "; max |KS| for a >= 1:", np.abs(KS[1:]).max(), "; max |KX| for a >= 1:", np.abs(KX[1:]).max())
print("single comparator kernels, max |K| over positions by age:")
for j in COMP:
    print(f"  comp {j}: " + " ".join(f"{np.abs(abs(w[j]) * KC[ORD.index(j)][a]).max():6.1f}" for a in range(10)))
# wavelength of the comparator kernels: dominant spatial frequency (FFT over positions 0..8 would be crude); instead, phase of the clock readout
print("\nfraction of each comparator kernel's centred squared norm carried by each mode, at ages 0, 2, 5, 8:")
for j in COMP:
    e = np.eye(13)[ORD.index(j)] * abs(w[j])
    for a in (0, 2, 5, 8):
        parts = {}
        for nme in NAMES:
            cfs = LEFT[nme] @ e * LAM[nme] ** a
            v = 2 * np.real(cfs * (Uo @ RIGHT[nme])) if IS_COMPLEX[nme] else np.real(cfs * (Uo @ RIGHT[nme]))
            parts[nme] = v
        tot = sum(parts.values()); tot_c = tot - tot.mean()
        print(f"  comp {j} age {a}: " + " ".join(f"{nme} {np.linalg.norm(v - v.mean()):6.1f}" for nme, v in parts.items()) + f" | total {np.linalg.norm(tot_c):6.1f}")
