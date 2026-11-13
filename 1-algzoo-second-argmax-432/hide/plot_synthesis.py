"""Heatmap of SYNTHESIS (modes -> hidden units) from full_account.py."""
import sys, os
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'reverse-engineered-networks', '1-algzoo-second-argmax-432', 'code'))
import full_account as fa
from plot_weights import ORDER, UNIT_COLOR  # same unit order/colours as weights_grouped.png

# rows of SYNTHESIS are fa.ORDINARY; reorder them to the grouped order
rows = [fa.ORDINARY.index(u) for u in ORDER if u in fa.ORDINARY]
units = [fa.ORDINARY[r] for r in rows]

S = fa.SYNTHESIS[rows]
vmax = np.abs(S).max()
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(S, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
for r in range(S.shape[0]):
    for c in range(S.shape[1]):
        ax.text(c, r, f'{S[r, c]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if abs(S[r, c]) > 0.6 * vmax else 'black')

# columns: modes; pole written once per group (pairs rotate with pole a +/- bi)
ax.set_xticks(range(13), [f'm{i}' for i in range(13)], fontsize=8)
k = 0
for g in fa.GROUPS:
    a = fa.POLES[k]; k += 1
    if len(g) == 1:
        label = f'{a:.2f}'
    else:
        b = fa.POLES[k]; k += 1
        label = f'{a:.2f}±{abs(b):.2f}i\n|{abs(complex(a, b)):.2f}|'
    ax.text(np.mean(g), 13.0, label, ha='center', va='top', fontsize=7.5)
for g in fa.GROUPS[:-1]:
    ax.axvline(g[-1] + 0.5, color='k', lw=1.5)
ax.set_yticks(range(13), [str(u) for u in units])
for t, u in zip(ax.get_yticklabels(), units):
    t.set_color(UNIT_COLOR[u]); t.set_fontweight('bold')
n_comp = sum(UNIT_COLOR[u] == UNIT_COLOR[ORDER[0]] for u in units)
ax.axhline(n_comp - 0.5, color='k', lw=1.5)
ax.set_xlabel('mode (memory component); pole below (pairs: complex pole and |modulus|)', labelpad=30)
ax.set_ylabel('neuron id  (blue: comparators, grey: wave units)')
ax.set_title('SYNTHESIS: units = ReLU(SYNTHESIS @ flow + INPUT·x)')
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, '..', 'plots', 'synthesis_matrix.png'), dpi=150)
