"""Referee S, check 3: the worked example of section 5.5, and two counterfactuals for its clock phases."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT

w4 = w[4]; g = -w[COMP]
NC = [u for u in range(N) if u not in COMP]


def table(x, label):
    x = np.asarray(x, float)
    H, Z = run(x[None], return_pre=True); H, Z = H[0], Z[0]
    print(f"\n{label}: x = {x.tolist()}")
    print(" t    x   th4   th1   th6   th7   th8  cut        h4     h5    h9   |z|/w4  phase  top_s top_m")
    for t in range(10):
        hp = H[t - 1] if t else np.zeros(N)
        th = (W[COMP] @ hp) / g; th4 = -(W[4] @ hp) / w4
        cut = [u for u in COMP if H[t, u] == 0]
        z = H[t, ORD] @ LEFT["clock"]
        s = U @ H[t]; m = U[:, NC] @ H[t, NC]
        s[:9 - t] = -np.inf; m[:9 - t] = -np.inf; m[9] = -np.inf
        tm = (m.argmax() - (9 - t)) if t else -1
        print(f"{t:2d} {x[t]:5.1f} {th4:5.2f} {th[0]:5.2f} {th[1]:5.2f} {th[2]:5.2f} {th[3]:5.2f}  {str(cut):10s} "
              f"{H[t,4]:5.2f} {H[t,5]:6.2f} {H[t,9]:5.2f}  {abs(z)/w4:5.2f}  {np.degrees(np.angle(z)):6.1f}  {s.argmax() - (9 - t):4d} {tm:5d}")
    return H


x0 = [0.8, -0.5, 0.3, 1.2, -1.0, 0.9, -0.7, 1.6, -0.2, 0.4]
table(x0, "section 5.5")
x1 = list(x0); x1[2] = -0.5
table(x1, "counterfactual A: no new runner-up at step 2 (x_2 = -0.5); first maximum has no earlier clock")
x2 = [-1.0, -1.0, -1.0, 1.2, -1.0, 0.9, -0.7, 1.6, -0.2, 0.4]
table(x2, "counterfactual B: nothing positive before the maximum at 3 (no earlier clock), runner-up 0.9 at step 5 kept")
x3 = [-1.0, -1.0, -1.0, 1.2, -1.0, -1.0, -0.7, 1.6, -0.2, 0.4]
table(x3, "counterfactual C: as B without the runner-up at step 5")
