"""E19. Anatomy of a new runner-up, controlled: x = (0.5, b, 1.5, b, b, b, v, b, b, b) with background b,
old runner-up 0.5 at position 0, max 1.5 at position 2, test input v at position 6 (age of max = 4)."""
from common import *
from modes import coords

b = -1.0
print("thresholds at step 6 (before v arrives) for v-independent state:")
x0 = np.array([0.5, b, 1.5, b, b, b, b, b, b, b])
H0, Z0 = run(x0, return_pre=True)
h5 = H0[0, 5]
th = {j: -(W[j] @ h5) / w[j] for j in COMP + [4]}
print("  " + "  ".join(f"theta_{j} = {th[j]:.3f}" for j in COMP + [4]), "   (S = 0.5, M = 1.5)")
z5 = coords(H0[0, 5])["clock"] / 10.157
print(f"  clock before: |z|/10.157 = {abs(z5):.3f}, phase = {np.degrees(np.angle(z5)):.1f} deg;  M - 0.87 S = {1.5 - 0.87*0.5:.3f}")

print("\n    v    comparators on at t=6   unit4   h5(t=7)  h9(t=7) | clock |z|/10.157 and phase at t=6, 7, 8, 9          | (M - 0.87 max(v,S)) | answer  margin  s[6]-s[0] at t=9")
for v in (-1.0, 0.0, 0.3, 0.45, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.4, 1.45, 1.49, 1.55, 1.8):
    x = x0.copy(); x[6] = v
    H, Z = run(x, return_pre=True)
    on = "".join(str(j) if Z[0, 6, j] > 0 else "-" for j in COMP)
    zs = [coords(H[0, t])["clock"] / 10.157 for t in (6, 7, 8, 9)]
    lg = U @ H[0, 9]
    ans = lg.argmax()
    srt = np.sort(lg)
    newS = max(v, 0.5) if v < 1.5 else 1.5
    newM = max(v, 1.5)
    print(f"  {v:5.2f}      {on:8s}        {H[0,6,4]:6.2f}  {H[0,7,5]:6.2f}  {H[0,7,9]:6.2f}   | " +
          "  ".join(f"{abs(z):5.3f}@{np.degrees(np.angle(z)):6.1f}" for z in zs) +
          f" |      {newM - 0.87*newS:5.3f}          |   {ans}    {srt[-1]-srt[-2]:6.1f}   {lg[6]-lg[0]:7.1f}")
