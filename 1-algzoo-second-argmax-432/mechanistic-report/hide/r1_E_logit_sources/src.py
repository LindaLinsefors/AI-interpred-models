"""Exact source decomposition of the final logits (arm E).

    h_9[ORD] = sum_{s=0..9} A^{9-s} [ w[ORD] x_s + W[ORD, PULSE] p_{s-1} + c_s ],
    c_s = h_s[ORD] - preact_s[ORD] >= 0,   p_s = h_s[PULSE],   p_{-1} = 0,
    logits   = U[:, ORD] h_9[ORD] + U[:, PULSE] p_9.

Sources and kernels (age a = 9 - s = steps between the event and the readout):
    raw input x_s            KX[a]      = Uo A^a w[ORD]
    pulse unit u, h_s[u]     KP[k, a]   = U[:, u]                 (a = 0)
                                        = Uo A^(a-1) W[ORD, u]    (a >= 1)        k indexes PULSE = [4, 5, 9]
    clip of ordinary unit i  KC[i, a]   = Uo A^a e_i                              i indexes ORD
All kernels have shape (..., 10 ages, 10 positions).
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

Uo = U[:, ORD]
APOW = np.stack([np.linalg.matrix_power(A, a) for a in range(10)])       # (10, 13, 13)
KX = np.stack([Uo @ APOW[a] @ w[ORD] for a in range(10)])                 # (10, 10)
KP = np.stack([np.stack([U[:, u] if a == 0 else Uo @ APOW[a - 1] @ W[ORD, u] for a in range(10)])
               for u in PULSE])                                           # (3, 10, 10)
KC = np.stack([np.stack([Uo @ APOW[a][:, i] for a in range(10)]) for i in range(13)])   # (13, 10, 10)
AGE = 9 - np.arange(10)                                                   # age of an event at step s
CI = [ORD.index(u) for u in COMP]                                         # comparators inside ORD
WI = [ORD.index(u) for u in WAVE]                                         # wave units inside ORD


def amplitudes(x):
    """Event amplitudes of a batch x (B, 10): returns H, Z, p (B,10,3), c (B,10,13)."""
    H, Z = run(x, return_pre=True)
    return H, Z, H[:, :, PULSE], H[:, :, ORD] - Z[:, :, ORD]


def contributions(x, p, c, pos=None):
    """Contribution of every single source to the logits.

    pos = None: all 10 positions.  pos = (B, m) integer array: only those positions of each sequence.
    Returns CX (B, 10 steps, m), CP (B, 10 steps, 3 pulses, m), CC (B, 10 steps, 13 units, m)."""
    kx, kp, kc = KX[AGE], KP[:, AGE], KC[:, AGE]            # indexed by step s: (10,10), (3,10,10), (13,10,10)
    if pos is None:
        CX = x[:, :, None] * kx[None]
        CP = p[:, :, :, None] * kp.transpose(1, 0, 2)[None]
        CC = c[:, :, :, None] * kc.transpose(1, 0, 2)[None]
        return CX, CP, CC
    B, m = pos.shape
    s = np.arange(10)
    CX = x[:, :, None] * kx[s[None, :, None], pos[:, None, :]]                       # (B, 10, m)
    CP = p[:, :, :, None] * kp.transpose(1, 0, 2)[s[None, :, None, None], np.arange(3)[None, None, :, None], pos[:, None, None, :]]
    CC = c[:, :, :, None] * kc.transpose(1, 0, 2)[s[None, :, None, None], np.arange(13)[None, None, :, None], pos[:, None, None, :]]
    return CX, CP, CC


CLASSES = ["X", "P4", "P5", "P9", "C1", "C6", "C7", "C8", "CW"]
ROLES = ["max", "runner", "third", "other"]


def class_role_pos(x):
    """Exact contributions to all 10 logits, summed into 9 classes x 4 roles: returns (out (B, 9, 4, 10), L (B, 10)).
    Role = which input caused the event (final max / runner-up / third / other); the causing input of an event at
    step s is x_s for X, P4 and clips and x_{s-1} for P5, P9 (step-0 firings of units 5, 9 go to 'other')."""
    Bn = len(x)
    H, Z, p, c = amplitudes(x)
    L = logits(H)
    CX, CP, CC = contributions(x, p, c)
    order = np.argsort(-x, axis=1)
    s = np.arange(10)[None, :]
    role = np.full((Bn, 10), 3)
    role[s == order[:, 2:3]] = 2; role[s == order[:, 1:2]] = 1; role[s == order[:, 0:1]] = 0
    R = np.eye(4)[role]
    Rlag = np.concatenate([np.eye(4)[np.full((Bn, 1), 3)], R[:, :-1]], 1)
    out = np.empty((Bn, 9, 4, 10))
    out[:, 0] = np.einsum("bsr,bst->brt", R, CX)
    out[:, 1] = np.einsum("bsr,bst->brt", R, CP[:, :, 0])
    out[:, 2] = np.einsum("bsr,bst->brt", Rlag, CP[:, :, 1])
    out[:, 3] = np.einsum("bsr,bst->brt", Rlag, CP[:, :, 2])
    for k, j in enumerate(COMP):
        out[:, 4 + k] = np.einsum("bsr,bst->brt", R, CC[:, :, ORD.index(j)])
    out[:, 8] = np.einsum("bsr,bst->brt", R, CC[:, :, WI].sum(2))
    return out, L
