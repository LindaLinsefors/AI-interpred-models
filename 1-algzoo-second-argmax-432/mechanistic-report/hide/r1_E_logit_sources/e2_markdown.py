"""E2: markdown tables for findings.md.  [EXACT]"""
from src import *


def rel_table(K, sh, title, offs=range(-4, 4)):
    print(f"\n**{title}**\n")
    print("| age | own pos | " + " | ".join(f"d={d:+d}" if d else "**own**" for d in offs) + " | most extreme other entry (pos) | centred norm |")
    print("|---|---|" + "---|" * (len(list(offs)) + 2))
    for a in range(10):
        own = 9 - a - sh
        if not 0 <= own <= 9:
            continue
        cells = []
        for d in offs:
            p = own + d
            cells.append((f"**{K[a, p]:.2f}**" if d == 0 else f"{K[a, p]:.2f}") if 0 <= p <= 9 else "")
        oth = np.delete(K[a], own); op = np.delete(np.arange(10), own); k = np.abs(oth).argmax()
        print(f"| {a} | {own} | " + " | ".join(cells) + f" | {oth[k]:.2f} (p{op[k]}) | {np.linalg.norm(K[a] - K[a].mean()):.2f} |")


rel_table(KX, 0, "Raw input, logits per unit $x_s$ (own = position $s$)")
rel_table(KP[0], 0, "Pulse 4, logits per unit $h_s[4]$ (own = position $s$)")
rel_table(KP[1], 1, "Pulse 5, logits per unit $h_s[5]$ (own = position $s-1$, the input that caused it)")
rel_table(KP[2], 1, "Pulse 9, logits per unit $h_s[9]$ (own = position $s-1$)")
KS = sum(abs(w[j]) * KC[ORD.index(j)] for j in COMP)
rel_table(KS, 0, "Sum of the four comparator kernels, logits per unit of a common excess")

print("\n**Single comparator clips, logits per unit of excess $x_s-\theta_j(s)$**: value at own position, largest and smallest entry with their offset from own\n")
print("| age | " + " | ".join(f"comp {j}: own / max (offset) / min (offset)" for j in COMP) + " |")
print("|---|---|---|---|---|")
for a in range(10):
    own = 9 - a
    cells = []
    for j in COMP:
        K = abs(w[j]) * KC[ORD.index(j)][a]
        cells.append(f"{K[own]:.0f} / {K.max():.0f} ({int(K.argmax()) - own:+d}) / {K.min():.0f} ({int(K.argmin()) - own:+d})")
    print(f"| {a} | " + " | ".join(cells) + " |")

print("\n**Wave-unit clips, logits per unit clip amount**: the dominant feature of each kernel\n")
print("| unit | value at own position, ages 0..9 | largest |entry| at ages 0, 2, 4, 6: value (offset from own) | centred norm at ages 0, 3, 6, 9 |")
print("|---|---|---|---|")
for i in WAVE:
    K = KC[ORD.index(i)]
    own = [f"{K[a, 9 - a]:.1f}" for a in range(10)]
    ext = []
    for a in (0, 2, 4, 6):
        k = np.abs(K[a]).argmax(); ext.append(f"{K[a, k]:.1f} ({k - (9 - a):+d})")
    print(f"| {i} | " + ", ".join(own) + " | " + ", ".join(ext) + " | " + ", ".join(f"{np.linalg.norm(K[a] - K[a].mean()):.1f}" for a in (0, 3, 6, 9)) + " |")
