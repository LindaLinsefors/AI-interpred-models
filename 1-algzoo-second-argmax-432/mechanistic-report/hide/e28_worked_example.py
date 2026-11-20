"""E28. One sequence, step by step: thresholds, pulses, clock and running answer."""
from common import *
from modes import coords

x = np.array([0.8, -0.5, 0.3, 1.2, -1.0, 0.9, -0.7, 1.6, -0.2, 0.4])
H, Z = run(x, return_pre=True)
H, Z = H[0], Z[0]
st = running_stats(x[None, :])
print("x =", x, "  true answer:", int(second_argmax(x[None, :])[0]), "  network:", int((U @ H[9]).argmax()))
print(" t    x_t  | th4    th1    th6    th7    th8  | cut comps | h4     h5     h9   | level/10.157  clock |z|/10.157 @ deg | runner-up so far (pos) | argmax s_t (pos) | argmax m_t older (pos)")
hprev = np.zeros(16)
for t in range(10):
    th = {j: (W[j] @ hprev) / (-w[j]) for j in COMP}
    th4 = -(W[4] @ hprev) / w[4]
    cut = "".join(str(j) for j in COMP if Z[t, j] <= 0) or "-"
    m = coords(H[t])
    z = m["clock"] / 10.157
    s = U @ H[t]
    valid = np.arange(10) >= 9 - t
    pos_s = int(np.where(valid, s, -np.inf).argmax()) + t - 9
    Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
    mm = s - Uc @ H[t]
    validm = valid & (np.arange(10) <= 8)
    pos_m = int(np.where(validm, mm, -np.inf).argmax()) + t - 9 if t >= 1 else -1
    ru = st["p2"][0, t]
    print(f" {t}  {x[t]:5.2f}  | {th4:5.2f}  {th[1]:5.2f}  {th[6]:5.2f}  {th[7]:5.2f}  {th[8]:5.2f} |   {cut:5s}   | {H[t,4]:5.2f}  {H[t,5]:5.2f}  {H[t,9]:5.2f} |   {m['level']/10.157:6.3f}        {abs(z):5.3f} @ {np.degrees(np.angle(z)):6.1f}     |        {ru:2d}              |       {pos_s:2d}         |      {pos_m:2d}")
    hprev = H[t]
print("\nfinal logits:", np.round(U @ H[9], 1))
