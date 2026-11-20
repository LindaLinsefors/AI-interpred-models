"""D3 (note, EXACT). The centre is also what the thresholds read: contribution of each mode to the thresholds of F2.

theta_j(t+1) = -W[j] . h_t / w_j.   For a state with mode coordinates m_k (pulses off):
   theta_j = sum_k m_k * tau_jk,   tau_jk = -W[j, ORD] . RIGHT_k / w_j     (for a rotating pair: amplitude 2|tau| and a phase)
Printed per unit of (coordinate / 10.157), i.e. in units of the input value.     python -B d3c_thresholds.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

print("threshold contribution per unit of (mode coordinate / 10.157):  real modes: value;  rotating pairs: amplitude @ phase offset (deg)")
print("unit   " + "".join(f"{n:>15s}" for n in NAMES))
for j in [4] + COMP:
    row = ""
    for n in NAMES:
        tau = -(W[j, ORD] @ RIGHT[n]) / w[j] * 10.157
        row += f"{2 * abs(tau):8.3f}@{np.degrees(np.angle(tau)) % 360:5.0f} " if IS_COMPLEX[n] else f"{tau.real:14.3f} "
    print(f"{j:4d}   {row}")
print("\nwith the measured coordinates (S/M = 0: level 0.94 M, runner 1.9 M, clock rho M; S/M -> 1: level 0.92 M, runner 7 M, clock ~0.25 M):")
for j in [4] + COMP:
    tl = -(W[j, ORD] @ RIGHT["level"].real) / w[j] * 10.157
    tr = -(W[j, ORD] @ RIGHT["runner"].real) / w[j] * 10.157
    tc = 2 * abs(-(W[j, ORD] @ RIGHT["clock"]) / w[j] * 10.157)
    print(f"   unit {j}: centre of theta = ({0.94 * tl + 1.9 * tr:.3f}) M + ({5.1 * tr:.3f}) S;   swing = +-{tc:.3f} * |z|/10.157"
          f"   -> at S = 0, a = 2 (|z| = 0.96 M): theta in [{0.94 * tl + 1.9 * tr - 0.96 * tc:.2f}, {0.94 * tl + 1.9 * tr + 0.96 * tc:.2f}] M")
