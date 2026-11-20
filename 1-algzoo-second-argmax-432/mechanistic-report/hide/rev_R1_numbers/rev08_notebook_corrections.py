"""Reviewer R1: the lab notebook's 'Verification round' section (appended after report.md was written) withdraws several
readings that report.md still contains.  Independent recomputation of the ones that are numbers.
Run:  python -B rev08_notebook_corrections.py > rev08_notebook_corrections_out.txt"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT, LAM
np.set_printoptions(precision=3, suppress=True, linewidth=220)
CORE = [2, 4, 1, 6, 7, 8]
SAT = [u for u in range(16) if u not in CORE]


def acc_with(Wm, x, wm=w):
    h = np.zeros((len(x), N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * wm + h @ Wm.T, 0)
    return np.mean((h @ U.T).argmax(1) == second_argmax(x))


x = sample(400_000, 8080)
print("(1) section 4.1 'The core runs almost by itself' / 'unit 2 moves all four thresholds by the same amount'")
print(f"    unmodified accuracy                                             {acc_with(W, x):.4f}")
W1 = W.copy(); W1[np.ix_(CORE, SAT)] = 0
print(f"    the 60 weights satellites -> core set to 0                      {acc_with(W1, x):.4f}")
W2 = W.copy(); g = -w[COMP]; W2[COMP, 2] = np.mean(W[COMP, 2] / g) * g
print(f"    b_2 made exactly uniform (W[j,2] = mean(b_2) g_j)               {acc_with(W2, x):.4f}    b_2 = {np.round(W[COMP, 2] / g, 4)}")
W3 = W.copy(); W3[COMP, 2] = 0
print(f"    unit 2 -> comparators removed                                   {acc_with(W3, x):.4f}")

print("\n(2) section 3.5 item 1: what the wipe does.  P(answer = max) for a wipe at t0 = pM + k, t0 <= 7, final S > 0.2 and gap > 0.2")
x = sample(300_000, 1717)
st = running_stats(x); truth = second_argmax(x); pM = x.argmax(1)
sel0 = (st["M2"][:, 9] > 0.2) & (st["M1"][:, 9] - st["M2"][:, 9] > 0.2)


def run_var(x, t0, scale, kill9=False, kill5=False):
    h = np.zeros((len(x), N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        s = np.where(t0 == t)[0]
        if len(s):
            h[np.ix_(s, COMP)] *= scale
        s1 = np.where(t0 + 1 == t)[0]
        if len(s1) and kill9:
            h[s1, 9] = 0.0
        if len(s1) and kill5:
            h[s1, 5] = 0.0
    return (h @ U.T).argmax(1)


print("    k | comparators x0 | x0.02 | x0.1 | x0.5 | x0 and unit 9 forced off at t0+1 | x0 and unit 5 forced off at t0+1 |  P(unit 9 on at t0+1): x0 / x0.02 / unmodified")
for k in range(1, 7):
    t0 = pM + k
    ok = sel0 & (t0 <= 7)
    xs, ts, tm = x[ok], t0[ok], pM[ok]
    row = f"    {k} |"
    for sc in (0.0, 0.02, 0.1, 0.5):
        row += f"  {np.mean(run_var(xs, ts, sc) == tm):.3f} |"
    row += f"  {np.mean(run_var(xs, ts, 0.0, kill9=True) == tm):.3f} |  {np.mean(run_var(xs, ts, 0.0, kill5=True) == tm):.3f} |"
    # unit 9 at t0+1
    def u9(scale):
        h = np.zeros((len(xs), N)); out = np.zeros(len(xs), bool)
        for t in range(T):
            h = np.maximum(xs[:, t, None] * w + h @ W.T, 0)
            s = np.where(ts == t)[0]
            if len(s): h[np.ix_(s, COMP)] *= scale
            s1 = np.where(ts + 1 == t)[0]
            if len(s1): out[s1] = h[s1, 9] > 0
        return out.mean()
    row += f"  {u9(0.0):.3f} / {u9(0.02):.3f} / {u9(1.0):.3f}   n={ok.sum()}"
    print(row)

print("\n(3) section 3.5 item 2: rotation of the clock one step before the overtaking, e18's sample; outcome categories made exclusive")
x = sample(1_500_000, 1818)
st = running_stats(x); truth = second_argmax(x); B = len(x); ar = np.arange(B)
om = np.angle(LAM["clock"])
tr = x.argmax(1); tm1 = np.maximum(tr - 1, 0)
p0 = st["p1"][ar, tm1]; q0 = st["p2"][ar, tm1]; M0 = st["M1"][ar, tm1]; S0 = st["M2"][ar, tm1]
g = tr - 1 - p0
ok = (tr >= 3) & (g >= 2) & (S0 > 0.2) & (M0 - S0 > 0.1) & (truth == p0) & (st["p3"][:, 9] == q0) & (tr <= 8)
xs, trs, p0s, q0s = x[ok], tr[ok], p0[ok], q0[ok]


def run_rot(k):
    h = np.zeros((len(xs), N))
    for t in range(T):
        h = np.maximum(xs[:, t, None] * w + h @ W.T, 0)
        s = np.where(trs - 1 == t)[0]
        if len(s):
            z = h[s][:, ORD] @ LEFT["clock"]
            h[np.ix_(s, ORD)] += 2 * np.real(np.multiply.outer(z * np.exp(1j * om * k) - z, RIGHT["clock"]))
    return (h @ U.T).argmax(1)


print("    k | P(answer = p0 - k), all (as in the report) | share of sequences where p0 - k is the old runner-up or the new max or does not exist | P(answer = p0 - k) given it exists (0 <= p0-k < record step) and is not the old runner-up")
for k in (-2, -1, 1, 2):
    a = run_rot(k)
    tgt = p0s - k
    exists = (tgt >= 0) & (tgt < trs)
    clean = exists & (tgt != q0s)
    print(f"   {k:+d} |   {np.mean(a == tgt):.3f}   |   {1 - clean.mean():.3f} (is old runner-up: {np.mean(tgt == q0s):.3f}; is new max: {np.mean(tgt == trs):.3f}; outside: {np.mean((tgt < 0) | (tgt > trs)):.3f})   |   {np.mean(a[clean] == tgt[clean]):.3f}   (n = {clean.sum()})")

print("\n(4) section 7 'The deposit is a single harmonic': P(answer = third) at separation D = |p_S - p_T| of 7 and 8, split by which of the two comes first")
print("    selection as e22: the max arrives after both, T > 0.2; relative gap (S - T)/S in (0.03, 0.08] and (0.08, 0.2]")
x = sample(4_000_000, 2222)
srt = np.argsort(-x, axis=1)
pMx, pSx, pTx = srt[:, 0], srt[:, 1], srt[:, 2]
ar = np.arange(len(x))
Sx, Tx = x[ar, pSx], x[ar, pTx]
h = np.zeros((len(x), N))
for t in range(T):
    h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
ans = (h @ U.T).argmax(1)
base = (pMx > np.maximum(pSx, pTx)) & (Tx > 0.2)
rg = (Sx - Tx) / Sx
for D in (1, 4, 6, 7, 8):
    for lo, hi in ((0.03, 0.08), (0.08, 0.2)):
        q = base & (np.abs(pSx - pTx) == D) & (rg > lo) & (rg <= hi)
        q1 = q & (pSx < pTx); q2 = q & (pSx > pTx)
        ml = q & (pMx == 9)
        print(f"    D={D} gap ({lo},{hi}]: all {np.mean(ans[q] == pTx[q]):.3f} (n={q.sum()}) | runner-up earlier than third {np.mean(ans[q1] == pTx[q1]):.3f} (n={q1.sum()}) | third earlier than runner-up {np.mean(ans[q2] == pTx[q2]):.3f} (n={q2.sum()}) | share with the max at the last step {ml.sum() / max(q.sum(), 1):.2f}")
