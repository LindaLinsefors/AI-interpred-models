"""Shared code for 'follow the writes' (t1_writes).

Exact identity used everywhere (h = ReLU(a) = a + [-a]_+, a_s = w x_s + W h_(s-1)):
    h_s[o] = A h_(s-1)[o] + q_s,     q_s = w[o] x_s + W[o,P] h_(s-1)[P] + [-a_s[o]]_+
so every tape coordinate is   c(t) = sum_{s<=t} sum_ch  Amp[s, ch] * K[t - s, ch]
with 17 write channels ch and the fixed kernel K[age, ch] = (lam_m^age l_m . D[ch]) in the 7 real tape coordinates.

Channels (index: name: amplitude at step s):
    0      x     the input x_s
    1,2,3  p4, p5, p9   the pulses h_(s-1)[4], h_(s-1)[5], h_(s-1)[9] of the previous step
    4..16  k<u>  the clip [-a_s[u]]_+ of non-pulse unit u = ORD[ch - 4]; for a comparator u in {1,6,7,8}
                 this is g_u [x_s - theta_u(s)]_+ ("comparator switched off by the input").
"""
import sys
ROOT = "c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report"
sys.path.insert(0, ROOT + "/hide")
sys.dont_write_bytecode = True
from tape_common import *

NCH = 17
CH = ["x", "p4", "p5", "p9"] + [f"k{u}" for u in ORD]
D = np.vstack([w[ORD], W[ORD, 4], W[ORD, 5], W[ORD, 9], np.eye(13)])          # (17, 13) write direction per channel
COMP_CH = [4 + ORD.index(u) for u in COMP]
WAVE_CH = [4 + ORD.index(u) for u in WAVE]
GROUPS = {"input": [0], "pulse 4": [1], "pulse 5": [2], "pulse 9": [3], "comparators off": COMP_CH, "wave clips": WAVE_CH}
LT = np.stack([LEFT[n] for n in TAPE])                                        # (4, 13) complex dual vectors
LAMT = np.array([LAM[n] for n in TAPE])
CNAMES = ["alt", "Re short", "Im short", "Re h2", "Im h2", "Re h3", "Im h3"]
R_SCORE = U[:, ORD] @ coords_to_tape13(np.eye(7)).T                           # (10, 7): tape scores = R_SCORE @ c


def to_real(zc):
    """(..., 4) complex mode coordinates (alt, short, h2, h3) -> (..., 7) real tape coordinates."""
    return np.stack([zc[..., 0].real, zc[..., 1].real, zc[..., 1].imag, zc[..., 2].real, zc[..., 2].imag,
                     zc[..., 3].real, zc[..., 3].imag], -1)


def kernel(nage=10):
    """K[age, ch, 7]: tape coordinates 'age' steps after a unit write in channel ch (exact)."""
    kc = (LAMT[None, :, None] ** np.arange(nage)[:, None, None]) * (LT @ D.T)[None]     # (age, 4, 17)
    return to_real(np.moveaxis(kc, 1, -1))


K = kernel()


def amplitudes(x):
    """Returns H, Z (B, 10, 16) and the write amplitudes Amp (B, 10, 17)."""
    H, Z = run(x, return_pre=True)
    Amp = np.zeros((len(x), x.shape[1], NCH))
    Amp[:, :, 0] = x
    Amp[:, 1:, 1:4] = H[:, :-1][:, :, PULSE]
    Amp[:, :, 4:] = np.maximum(-Z[:, :, ORD], 0)
    return H, Z, Amp


def rebuild(Amp, t=9, mask=None):
    """Tape coordinates at step t from write amplitudes; mask (B, 10, 17) or broadcastable selects writes to keep."""
    A_ = Amp[:, : t + 1]
    if mask is not None:
        mask = np.asarray(mask)
        A_ = A_ * (mask[..., : t + 1, :] if mask.ndim >= 2 else mask)
    return np.einsum("bsc,sck->bk", A_, K[t - np.arange(t + 1)])


def r2(y, yh):
    return 1 - ((y - yh) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


CLS = ["M", "S>M", "S<M", "exmax", "exrun", "near", "far", "neg", "spont"]


def write_classes(x, t=9):
    """Class (index into CLS) of the input that caused each write, seen from the state at step t: (B, t+1, 17).
    Same rules as w2_by_cause.py (see its docstring)."""
    x = x[:, : t + 1]
    kind = roles(x)[0]; st = prefix_stats(x, t); pM, pS = st["pM"][:, None], st["pS"][:, None]
    jj = np.arange(t + 1)[None, :]
    role = np.select([jj == pM, (jj == pS) & (pS > pM), (jj == pS) & (pS < pM), kind == 1, kind == 2, kind == 3, kind == 4],
                     [0, 1, 2, 3, 4, 5, 6], 7)
    def shift(a, n, fill):
        out = np.full_like(a, fill); out[:, n:] = a[:, : a.shape[1] - n]
        return out
    k1, k2, r1, r2_ = shift(kind, 1, 0), shift(kind, 2, 0), shift(role, 1, 7), shift(role, 2, 7)
    ctx = np.select([kind == 1, kind == 2, k1 == 1, k2 == 1, (k2 == 2) | (k2 == 3), (k1 == 2) | (k1 == 3)], [0, 1, 2, 3, 4, 5], 6)
    wave_role = np.select([ctx <= 1, ctx == 2, ctx == 3, ctx == 4, ctx == 5], [role, r1, r2_, r2_, r1], 8)
    wcls = np.empty((len(x), t + 1, NCH), dtype=np.int8)
    wcls[:, :, 0] = role; wcls[:, :, COMP_CH] = role[:, :, None]
    wcls[:, :, 1] = r1; wcls[:, :, 2] = r2_; wcls[:, :, 3] = r2_
    wcls[:, :, WAVE_CH] = wave_role[:, :, None]
    return wcls


def roles(x):
    """Event class of every input j AT ITS ARRIVAL (B, 10) and the prefix statistics before it.
    Mp, Sp = maximum and runner-up of x_0..x_(j-1), clipped at 0 (0 if absent or negative).
    kind: 0 = not positive, 1 = new maximum (x_j > Mp), 2 = new runner-up (Sp < x_j <= Mp),
          3 = near the runner-up (0.7 Sp < x_j <= Sp), 4 = far below (0 < x_j <= 0.7 Sp)."""
    st = running_stats(x)
    Mp = np.zeros_like(x); Sp = np.zeros_like(x)
    Mp[:, 1:] = np.maximum(st["M1"][:, :-1], 0); Sp[:, 1:] = np.maximum(st["M2"][:, :-1], 0)
    kind = np.where(x <= 0, 0, np.where(x > Mp, 1, np.where(x > Sp, 2, np.where(x > 0.7 * Sp, 3, 4))))
    return kind, Mp, Sp, st
