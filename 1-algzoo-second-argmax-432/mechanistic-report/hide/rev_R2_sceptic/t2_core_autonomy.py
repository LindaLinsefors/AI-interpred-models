"""Reviewer R2, test 2.  "The core runs almost by itself" (report 4.1, lead's e23) is supported by R2 of the
core activations in a six-unit simulation.  Functional test in the FULL network: change only the weights from
the ten satellites into the six core units and measure the accuracy of the network's answer.
  A: W[core, satellites] = 0
  B: W[core, satellites] = 0 and the lead's variant-B compensation (fixed multiple of unit 2 from the level eigenvector)
  C: only the wave-unit -> core weights removed (pulses 5, 9 -> core kept), with the variant-B compensation
The satellites still receive the (now slightly different) core and still feed the readout.
python -B t2_core_autonomy.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import RIGHT

core = [2, 4, 1, 6, 7, 8]
sat = [u for u in range(16) if u not in core]
x = sample(1_000_000, 9202)
truth = second_argmax(x)
st = running_stats(x)
H0, Z0 = run(x, return_pre=True)
base = logits(H0).argmax(1)

ratio = np.zeros(16)
for u in WAVE:
    ratio[u] = RIGHT["level"][ORD.index(u)].real / RIGHT["level"][ORD.index(2)].real
fb = np.array([sum(W[j, u] * ratio[u] for u in WAVE if u != 2) for j in core])
print("variant-B extra weight from unit 2 into core units", dict(zip(core, np.round(fb, 3))))

def variant(kind):
    W2 = W.copy()
    if kind in ("A", "B"):
        W2[np.ix_(core, sat)] = 0.0
    if kind == "C":
        W2[np.ix_(core, [u for u in WAVE if u != 2])] = 0.0
    if kind in ("B", "C"):
        W2[core, 2] += fb
    return W2

print(f"\nbaseline accuracy {np.mean(base == truth):.4f}   (n = {len(x)})")
print("kind | accuracy  agreement with the network | R2 of core activations (units 2,4,1,6,7,8) | unit 4: P(fires | new positive max)  P(fires | not) | answers that become the max")
newmax = (x[:, 1:] > np.maximum(st["M1"][:, :-1], 0))
for kind in ("A", "B", "C"):
    H, Z = run(x, W=variant(kind), return_pre=True)
    ans = logits(H).argmax(1)
    r2 = [1 - ((H0[:, :, u] - H[:, :, u]) ** 2).sum() / ((H0[:, :, u] - H0[:, :, u].mean()) ** 2).sum() for u in core]
    f = Z[:, 1:, 4] > 0
    ch = ans != base
    print(f"  {kind}  |  {np.mean(ans == truth):.4f}    {np.mean(ans == base):.4f}   | " + " ".join(f"{v:.3f}" for v in r2)
          + f" |  {f[newmax].mean():.4f}  {f[~newmax].mean():.4f} | {np.mean(ans[ch] == x[ch].argmax(1)):.3f} of changed")
f0 = Z0[:, 1:, 4] > 0
print(f"  (real network: unit 4 P(fires | new positive max) {f0[newmax].mean():.4f}, P(fires | not) {f0[~newmax].mean():.4f})")

print("\nsize of what is removed: rms over steps 1..9 of the satellites' summed input to each core unit, relative to the rms of the unit's pre-activation")
for j in core:
    s = H0[:, :-1][:, :, sat] @ W[j, sat]
    print(f"   unit {j}: {np.sqrt((s ** 2).mean()) / np.sqrt((Z0[:, 1:, j] ** 2).mean()):.3f}")
