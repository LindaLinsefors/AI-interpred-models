"""Checks behind digest_explainer.md (Section 5): the explainer's worked example on the real weights.

1. Unit values of the explainer's step table (unit 4 and comparators 1, 6, 7, 8).
2. The thresholds the network used at each step. These are exact identities, not fits:
     h_t[4] = w_4 ReLU(x_t - th4_t),    th4_t = -W_hh[4] . h_{t-1} / w_4,   w_4 = W_ih[4] > 0
     h_t[j] = g_j ReLU(th_jt - x_t),    th_jt =  W_hh[j] . h_{t-1} / g_j,   g_j = -W_ih[j] > 0,  j = 1, 6, 7, 8
3. The final vote, split into comparators and the other 12 units.
4. The clock of code/exact_rewrite.py (memory[8:10]) and what changes it at each step:
     new clock = ROTATE @ old clock + PULSE_DRIVE[8:10] @ old pulses + write-back (input drive and ReLU clipping)

Run:  python -B check_explainer_example.py > check_explainer_example_output.txt     (about 1 s, NumPy only)
The colleague's folder is only read. No bytecode is written there.
"""
import sys
sys.dont_write_bytecode = True
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
np.set_printoptions(precision=3, suppress=True, linewidth=200)
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[3] / "reverse-engineered-networks" / "1-algzoo-second-argmax-432" / "code"
COMP = [1, 6, 7, 8]

w = np.load(CODE / "m16_10_weights.npz")
W_ih, W_hh, W_out = w["W_ih"][:, 0], w["W_hh"], w["W_out"]
x = np.array([0.3, 1.4, -0.5, 0.2, 0.9, -1.0, 1.8, 0.1, -0.3, 0.4])      # the explainer's example

# ---- 1 + 2: hidden states and thresholds
h = np.zeros(16); H, REC = [], []
for t in range(10):
    rec = W_hh @ h                                   # recurrent drive, before x_t is added
    h = np.maximum(W_ih * x[t] + rec, 0); H.append(h); REC.append(rec)
H, REC = np.array(H), np.array(REC)
th4 = -REC[:, 4] / W_ih[4]
thc = REC[:, COMP] / -W_ih[COMP]
print("W_ih[4] =", W_ih[4].round(3), "  -W_ih[1, 6, 7, 8] =", (-W_ih[COMP]).round(3))
print("\n t    x   Mprev Sprev |   u4    th4 | u1, u6, u7, u8              | th1, th6, th7, th8          | 10*max(x - Mprev, 0)")
M = S = 0.0
for t in range(10):
    print(f"{t:2d} {x[t]:+4.1f}  {M:5.1f} {S:5.1f} | {H[t, 4]:6.3f} {th4[t]:6.3f} | {H[t, COMP].round(2)} | {thc[t].round(3)} | {10 * max(x[t] - M, 0):.2f}")
    if x[t] > M: S, M = M, x[t]
    elif x[t] > S: S = x[t]
print("\nunit 5:", H[:, 5].round(3), "\nunit 9:", H[:, 9].round(3), "\nunit 2:", H[:, 2].round(2))
print("largest activation:", H.max().round(3), "at (step, unit)", tuple(int(i) for i in np.unravel_index(H.argmax(), H.shape)))

# ---- 3: final vote
logits = W_out @ H[-1]; comp = W_out[:, COMP] @ H[-1, COMP]
print("\nlogits     ", logits.round(2), " answer", int(logits.argmax()), " true", int(np.argsort(x)[-2]))
print("comparators", comp.round(2))
print("other 12   ", (logits - comp).round(2))

# ---- 4: the exact rewrite's clock
spec = importlib.util.spec_from_file_location("exact_rewrite", CODE / "exact_rewrite.py")
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
pol = lambda v: "%.3f @ %6.1f" % (np.hypot(*v), np.degrees(np.arctan2(v[1], v[0])))
kick = ex.PULSE_DRIVE[8:10]                          # rows clock_c, clock_s; columns record, contender, coincidence
print("\nkick into the clock per unit of pulse (radius @ degrees): record", pol(kick[:, 0]), "| contender", pol(kick[:, 1]), "| coincidence", pol(kick[:, 2]))
print("free rotation:", pol(ex.ROTATE[:, 0]))
print("level column of SYNTHESIS (units", ex.ORDINARY, "):", ex.SYNTHESIS[:, 12].round(2))
print("runner column:", ex.SYNTHESIS[:, 11].round(2))
print("clock amplitude per unit:", np.hypot(ex.SYNTHESIS[:, 8], ex.SYNTHESIS[:, 9]).round(2))
print("\n t    x | old pulses (rec, cont, coin) | rotated old clock | pulse kick      | write-back      | new clock       | level  runner | max |state error|")
carry = None
for t in range(10):
    mem0, pul0 = (np.zeros(13), np.zeros(3)) if carry is None else carry
    rot, k = ex.ROTATE @ mem0[8:10], kick @ pul0
    carry, out = ex.step(carry, SimpleNamespace(values=x, position=t), {})
    mem = carry[0]; err = np.abs(np.array(out[ex.PREDICTS[0]]) - H[t]).max()
    print(f"{t:2d} {x[t]:+4.1f} | {pul0.round(3)!s:28s} | {pol(rot)} | {pol(k)} | {pol(mem[8:10] - rot - k)} | {pol(mem[8:10])} | {mem[12]:.3f} {mem[11]:6.3f} | {err:.4f}")
