"""Verifier's own code for the 'writes' angle (tv_C_writes). Nothing is imported from t1_writes.

The tape projector is built here directly from the eigen-decomposition of A = W[o, o], and everything is done in the
13-dimensional space of the non-pulse units (no eigenvector gauge needed):
    tape part of h_t[o] = P h_t[o] = sum_{s <= t} A^(t-s) P q_s ,   q_s = sum_ch Amp[s, ch] D[ch]
Channels: 0 input x_s | 1, 2, 3 pulses h_(s-1)[4], [5], [9] | 4..16 clips [-a_s[u]]_+ of u = ORD[ch - 4].
"""
import sys
ROOT = "c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report"
sys.path.insert(0, ROOT + "/hide")
sys.dont_write_bytecode = True
import numpy as np
from common import w, W, U, run, sample, ORD, COMP, PULSE, WAVE, second_argmax

A = W[np.ix_(ORD, ORD)]
Uo = U[:, ORD]
lam, V = np.linalg.eig(A)
Linv = np.linalg.inv(V)
SPEC = {"alt": (0.928, 180.0), "short": (0.735, 42.99), "h2": (0.937, 83.28), "h3": (0.837, 124.78)}
ang = np.abs(np.degrees(np.angle(lam)))
IDX = {n: np.where((np.abs(np.abs(lam) - m) < 2e-3) & (np.abs(ang - a) < 0.5))[0] for n, (m, a) in SPEC.items()}
assert [len(IDX[n]) for n in SPEC] == [1, 2, 2, 2], IDX
tape_idx = np.concatenate(list(IDX.values()))
P = np.real(V[:, tape_idx] @ Linv[tape_idx])              # 13 x 13 projector on the four tape modes
PR = np.eye(13) - P                                       # the other five modes

NCH = 17
D = np.vstack([w[ORD], W[ORD, 4], W[ORD, 5], W[ORD, 9], np.eye(13)])     # (17, 13)
CCH = [4 + ORD.index(u) for u in COMP]
WCH = [4 + ORD.index(u) for u in WAVE]
KINDS = {"input": [0], "pulse 4": [1], "pulse 5": [2], "pulse 9": [3], "comparators off": CCH, "wave clips": WCH}


def kernel(Pm=P, n=10):
    """G[age, ch, 13]: the part of h[o] in the modes of Pm, 'age' steps after a unit write in channel ch."""
    out, M_ = [], Pm.copy()
    for _ in range(n):
        out.append(D @ M_.T)
        M_ = A @ M_
    return np.stack(out)


G = kernel()


def amplitudes(x):
    H, Z = run(x, return_pre=True)
    Amp = np.zeros((len(x), x.shape[1], NCH))
    Amp[:, :, 0] = x
    Amp[:, 1:, 1:4] = H[:, :-1][:, :, PULSE]
    Amp[:, :, 4:] = np.maximum(-Z[:, :, ORD], 0)
    return H, Z, Amp


def rebuild(Amp, t=9, mask=None, G=G):
    """(B, 13) mode part of h_t[o] from the write amplitudes of steps 0..t; mask broadcastable to (B, t+1, 17)."""
    A_ = Amp[:, : t + 1]
    if mask is not None:
        A_ = A_ * mask
    return np.einsum("bsc,sck->bk", A_, G[t - np.arange(t + 1)])


def with_tape(h, tape13):
    """State h (B, 16) with the tape part of its non-pulse units replaced by tape13 (B, 13)."""
    out = h.copy()
    out[:, ORD] = h[:, ORD] @ PR.T + tape13
    return out


def answer(h):
    return (h @ U.T).argmax(1)


def top3(x, t):
    """Values, positions of the three largest of x_0..x_t, and the quiet mask (a_M >= 2, a_S >= 2, S > 0)."""
    pre = x[:, : t + 1]
    o = np.argsort(-pre, 1, kind="stable")
    ar = np.arange(len(x))
    d = {"pM": o[:, 0], "pS": o[:, 1], "M": pre[ar, o[:, 0]], "S": pre[ar, o[:, 1]]}
    if t >= 2:
        d["pT"], d["T"] = o[:, 2], pre[ar, o[:, 2]]
    d["quiet"] = (t - d["pM"] >= 2) & (t - d["pS"] >= 2) & (d["S"] > 0)
    return d


def prefix_before(x):
    """Mp, Sp, Tp (B, 10): the three largest of x_0..x_(j-1) clipped at 0 (0 if absent), and their ages at step j-1
    (9 if the value is absent or not positive)."""
    B, T = x.shape
    vals = np.zeros((3, B, T)); ages = np.full((3, B, T), 9)
    for j in range(1, T):
        pre = x[:, :j]
        o = np.argsort(-pre, 1, kind="stable")
        for k in range(min(3, j)):
            v = np.take_along_axis(pre, o[:, k: k + 1], 1)[:, 0]
            vals[k, :, j] = np.maximum(v, 0)
            ages[k, :, j] = np.where(v > 0, j - 1 - o[:, k], 9)
    return vals, ages


def arrival_kind(x, vals):
    """0 not positive, 1 new maximum, 2 new runner-up, 3 near (0.7 Sp < x <= Sp), 4 far below."""
    Mp, Sp = vals[0], vals[1]
    return np.where(x <= 0, 0, np.where(x > Mp, 1, np.where(x > Sp, 2, np.where(x > 0.7 * Sp, 3, 4))))


def cov_share(part, whole):
    """Covariance share of 'part' in 'whole' (both (n, k)), summed over columns; and the std ratio."""
    p = part - part.mean(0); s = whole - whole.mean(0)
    return (p * s).sum() / (s ** 2).sum(), np.sqrt((p ** 2).sum() / (s ** 2).sum())


def r2(y, yh):
    return 1 - ((y - yh) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
