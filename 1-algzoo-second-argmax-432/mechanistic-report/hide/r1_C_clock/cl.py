"""Shared helpers for arm C (the clock). Read-only use of common.py / modes.py."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

HERE = HIDE / "r1_C_clock"
K4 = w[4]                                  # 10.157..., input weight of unit 4 (the normalisation of F5/F6)
LC = LEFT["clock"]                         # clock left vector on ORD (complex)
RC = RIGHT["clock"]
LAMC = LAM["clock"]
OMEGA = np.angle(LAMC)                     # 42.748 deg in rad
ci = [ORD.index(u) for u in COMP]          # comparator indices inside ORD
wi = [ORD.index(u) for u in WAVE]
CP = {p: LC @ W[ORD, p] for p in PULSE}    # clock coordinate of what one unit of pulse p sends to ORD
ECHO = LC @ w[ORD]                         # clock coordinate written directly by one unit of input
deg = lambda c: np.degrees(np.angle(c))
pos = lambda a: np.maximum(np.where(np.isfinite(a), a, 0.0), 0.0)


def wrap(d):
    """Wrap degrees to (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def gen(n, seed, dist="normal"):
    """Sample, run, and return everything arm C needs for one batch."""
    x = sample(n, seed, dist)
    H, Z = run(x, return_pre=True)
    st = running_stats(x)
    return x, H, Z, st


def cfit(y, cols):
    """Complex least squares y ~ sum_k c_k cols_k (no intercept; regressors may be real or complex).
    Returns coefficients and the explained fraction 1 - sum|res|^2 / sum|y|^2."""
    X = np.column_stack(cols).astype(complex)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ c
    return c, 1 - (abs(res) ** 2).sum() / (abs(y) ** 2).sum()


def thresholds(H):
    """theta_j for the NEXT step from states H (..., 16), columns in the order (1, 6, 7, 8, 4):
    unit j is at its switching point when x = theta_j = -W[j].h / w_j.
    Comparators are ON iff x < theta_j, unit 4 is ON iff x > theta_4."""
    units = COMP + [4]
    return -(H @ W[units].T) / w[units]


def clock_terms(x, H, Z):
    """Exact split of the clock coordinate for t = 1..T-1 (arrays (B, T-1), index t-1):
        z(t) = LAMC z(t-1) + sum_p CP[p] h_p(t-1) + ECHO x_t + sum_u LC[u] clip_u(t),
    clip_u(t) = max(-pre_u(t), 0) = what the ReLU adds on ordinary unit u.  Returns dict; 'check' = max abs error."""
    zc = H[..., ORD] @ LC
    out = {"z": zc, "carried": LAMC * zc[:, :-1], "echo": ECHO * x[:, 1:]}
    for p in PULSE:
        out[f"p{p}"] = CP[p] * H[:, :-1, p]
    clip = np.maximum(-Z[:, 1:][..., ORD], 0.0)
    for j in COMP:
        out[f"clip{j}"] = LC[ORD.index(j)] * clip[..., ORD.index(j)]
    out["clipwave"] = clip[..., wi] @ LC[wi]
    tot = out["carried"] + out["echo"] + out["clipwave"] + sum(out[f"p{p}"] for p in PULSE) + sum(out[f"clip{j}"] for j in COMP)
    out["check"] = np.abs(tot - zc[:, 1:]).max()
    return out


def fmt(c, nd=3):
    return f"{abs(c):.{nd}f}@{deg(c):+7.1f}"
