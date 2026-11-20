"""N2: where do the phase deviations of the 5.5 example come from? Replace single inputs and look at the clock."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from modes import coords

w4 = w[4]
def clock(x):
    z = coords(run(np.array(x, float))[0])["clock"] / w4
    return [f"{abs(v):.2f}@{np.degrees(np.angle(v)):.0f}" for v in z]
base = [0.8, -0.5, 0.3, 1.2, -1.0, 0.9, -0.7, 1.6, -0.2, 0.4]
print("example            :", clock(base))
a = list(base); a[2] = -0.5
print("x_2 = -0.5 (no new runner-up at 2):", clock(a)[:3])
b = [0.8, -0.5, 0.3, 1.2, -1.0, -1.0, -0.7, -1, -1, -1]
print("x_5 = -1.0 (no new runner-up at 5):", clock(b)[3:7])
c = [-1, -1, -1, 1.2, -1.0, 0.9, -0.7, -1, -1, -1]
print("max 1.2 at 3 with nothing positive before, x_5 = 0.9:", clock(c)[3:7])
d = [-1, -1, -1, 1.2, -1.0, -1.0, -0.7, -1, -1, -1]
print("max 1.2 at 3 with nothing positive before, no runner-up:", clock(d)[3:7])
