"""Reviewer N1: the summary's '98.6% answer the maximum after a wipe two or more steps before the end'.
How much does it depend on the selection (final runner-up > 0.2 and final gap > 0.2) that the summary does not state?
Wipe: h_t0[1,6,7,8] = 0 at one step t0 = p_M + k after the final maximum, then continue with the real inputs.
Run: python -B n1_wipe.py"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *

x = sample(400_000, 8301)
H = run(x)
srt = np.sort(x, axis=1)
Mx, Sx = srt[:, -1], srt[:, -2]
pM = x.argmax(1)
pS = second_argmax(x)
base = logits(H).argmax(1)

sels = {
    "report's selection (S > 0.2, M - S > 0.2)": (Sx > 0.2) & (Mx - Sx > 0.2),
    "S > 0.2, M - S <= 0.2": (Sx > 0.2) & (Mx - Sx <= 0.2),
    "S > 0, any gap": Sx > 0,
    "all sequences with M > 0": Mx > 0,
}
print("share of sequences:", {k: round(float(v.mean()), 3) for k, v in sels.items()})
print("unperturbed P(answer = runner-up):", {k: round(float((base == pS)[v].mean()), 4) for k, v in sels.items()})

cnt = {name: {"t0<=7": np.zeros(3), "t0=8": np.zeros(3), "t0=9": np.zeros(3)} for name in sels}
bylag = {name: np.zeros((10, 3)) for name in sels}          # t0 <= 7 only
for t0 in range(1, 10):
    h = H[:, t0].copy()
    h[:, COMP] = 0.0
    hf = run(x[:, t0 + 1:], h0=h)[:, -1] if t0 < 9 else h
    ans = (hf @ U.T).argmax(1)
    ok = pM < t0                                             # the final maximum is already there, k = t0 - pM >= 1
    grp = "t0<=7" if t0 <= 7 else f"t0={t0}"
    for name, s in sels.items():
        m_ = ok & s
        cnt[name][grp] += [np.sum(ans[m_] == pM[m_]), np.sum(ans[m_] == pS[m_]), m_.sum()]
        if t0 <= 7:
            for k in range(1, 8):
                mk = m_ & (t0 - pM == k)
                bylag[name][k] += [np.sum(ans[mk] == pM[mk]), np.sum(ans[mk] == pS[mk]), mk.sum()]

for name in sels:
    print("\n" + name)
    for grp, c in cnt[name].items():
        print(f"   {grp}: P(answer = max) = {c[0]/c[2]:.4f}   P(answer = runner-up) = {c[1]/c[2]:.4f}   n = {int(c[2])}")
    b = bylag[name]
    print("   t0 <= 7 by lag k = 1..7: P(max) =", np.round(b[1:8, 0] / b[1:8, 2], 3))
