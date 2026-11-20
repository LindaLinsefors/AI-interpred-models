"""Shared pieces for arm A (census of gates and events).

Data: 2^20 sequences of iid N(0,1) inputs = 4 batches of 2^18 with seeds 1000..1003.
Event class of step t >= 1, from the prefix x_0..x_{t-1} (M = its max, S = its second largest,
S = -inf at t = 1):   R: x_t > M,   N: S < x_t < M,   L: x_t < S.   Step 0 is class F (first).
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *      # noqa
from modes import *       # noqa

SEEDS = (1000, 1001, 1002, 1003)
NB = 2 ** 18
F_, R_, N_, L_ = 0, 1, 2, 3
CLS = "FRNL"
RESTING = sum(1 << i for i in ORD)          # pattern id: 4, 5, 9 off, the other 13 on


def batches(seeds=SEEDS, nb=NB):
    for s in seeds:
        x = sample(nb, s)
        H, Z = run(x, return_pre=True)
        yield s, x, H, Z


def top2(x):
    """Running max M, second largest S and their positions pM, pS for the prefix x_0..x_t (inclusive)."""
    B, Tn = x.shape
    M, S = np.empty((B, Tn)), np.empty((B, Tn))
    pM, pS = np.empty((B, Tn), int), np.empty((B, Tn), int)
    m, s = np.full(B, -np.inf), np.full(B, -np.inf)
    pm, ps = np.full(B, -1), np.full(B, -1)
    for t in range(Tn):
        xt = x[:, t]
        r = xt > m
        n = ~r & (xt > s)
        s, ps = np.where(r, m, np.where(n, xt, s)), np.where(r, pm, np.where(n, t, ps))
        m, pm = np.where(r, xt, m), np.where(r, t, pm)
        M[:, t], S[:, t], pM[:, t], pS[:, t] = m, s, pm, ps
    return M, S, pM, pS


def events(x):
    """Dict with, per (b, t):
    M, S, pM, pS : running top-2 including x_t;   Mp, Sp, pMp, pSp : the same for the prefix before t
    cls : 0 F (t = 0), 1 R, 2 N, 3 L
    sR, sN : steps since the last R / N event up to and including t (t = 0 counts as R; -1 if none)
    """
    B, Tn = x.shape
    M, S, pM, pS = top2(x)
    shift = lambda a, fill: np.concatenate([np.full((B, 1), fill, dtype=a.dtype), a[:, :-1]], 1)
    Mp, Sp, pMp, pSp = shift(M, -np.inf), shift(S, -np.inf), shift(pM, -1), shift(pS, -1)
    cls = np.where(x > Mp, R_, np.where(x > Sp, N_, L_)).astype(np.int8)
    cls[:, 0] = F_
    tt = np.arange(Tn)[None, :]
    lastR = np.maximum.accumulate(np.where((cls == R_) | (cls == F_), tt, -1), 1)
    lastN = np.maximum.accumulate(np.where(cls == N_, tt, -1), 1)
    return dict(M=M, S=S, pM=pM, pS=pS, Mp=Mp, Sp=Sp, pMp=pMp, pSp=pSp, cls=cls,
                sR=tt - lastR, sN=np.where(lastN >= 0, tt - lastN, -1), a1=tt - pM,
                a2=np.where(pS >= 0, tt - pS, -1))


def pstr(pid):
    """Pattern as text: which pulses are on, which comparators are off, which wave units are off."""
    on = [(pid >> i) & 1 for i in range(N)]
    p = "".join(str(i) + " " for i in PULSE if on[i]).strip() or "-"
    c = "".join(str(i) + " " for i in COMP if not on[i]).strip() or "-"
    v = "".join(str(i) + " " for i in WAVE if not on[i]).strip() or "-"
    return f"pulses on [{p}] comp off [{c}] wave off [{v}]"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
