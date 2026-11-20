"""exploration: impulse response gates and preactivations; background -1 variant."""
from cc import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
for bg in (0.0, -0.3, -1.0):
    x = np.full((1, 10), bg); x[0, 0] = 1.0
    H, Z = run(x, return_pre=True)
    print(f"\nimpulse 1 at t=0, background {bg}: pre-activations Z[t, unit]")
    print("  t  " + "".join(f"{i:8d}" for i in range(N)))
    for t in range(10):
        print(f"  {t}  " + "".join(f"{Z[0, t, i]:8.2f}" for i in range(N)))
    print("  off units per t:", [[i for i in range(N) if Z[0, t, i] <= 0] for t in range(10)])
