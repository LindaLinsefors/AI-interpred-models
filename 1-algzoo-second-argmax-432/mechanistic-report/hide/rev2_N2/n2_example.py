"""N2: recompute the table of section 5.5 (own code)."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from modes import coords, LEFT

w4 = w[4]
x = np.array([0.8, -0.5, 0.3, 1.2, -1.0, 0.9, -0.7, 1.6, -0.2, 0.4])
H, Z = run(x, return_pre=True); H, Z = H[0], Z[0]
hp = np.zeros(16)
idx = np.arange(10)
for t in range(10):
    th = {j: (W[j] @ hp) / (-w[j]) for j in COMP}; th4 = -(W[4] @ hp) / w4
    cut = ",".join(str(j) for j in COMP if Z[t, j] <= 0) or "none"
    z = (H[t, ORD] @ LEFT["clock"]) / w4
    s = U @ H[t]; m = s - U[:, COMP] @ H[t, COMP]
    tops = int(np.where(idx >= 9 - t, s, -np.inf).argmax()) + t - 9
    topm = int(np.where((idx >= 9 - t) & (idx <= 8), m, -np.inf).argmax()) + t - 9 if t else -1
    topm_all = int(np.where(idx >= 9 - t, m, -np.inf).argmax()) + t - 9
    pre = x[:t + 1]; order = np.argsort(-pre)
    print(f"{t} {x[t]:5.1f} | th4 {th4:5.2f} th1 {th[1]:5.2f} th6 {th[6]:5.2f} th7 {th[7]:5.2f} th8 {th[8]:5.2f} | cut {cut:8s} | h4 {H[t,4]:5.2f} h5 {H[t,5]:5.2f} h9 {H[t,9]:5.2f} | clock {abs(z):.2f} @ {np.degrees(np.angle(z)):5.0f} | top s {tops} top m(older) {topm} top m(all) {topm_all} | true max {order[0]} runner-up {order[1] if t else -1} | h0 {H[t,0]:.2f}")
    hp = H[t]
print("answer", (U @ H[9]).argmax(), " excess over th4 at 3, 7:", H[3, 4] / w4, H[7, 4] / w4)
