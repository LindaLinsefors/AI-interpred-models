"""Shared helpers of arm I (error analysis). Imports the lead's read-only helpers."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *          # noqa: F401,F403
from modes import *           # noqa: F401,F403

W4 = w[4]                     # +10.157, input weight of unit 4
SEEDS = list(range(1000, 1008))   # 8 batches
BATCH = 2 ** 18                   # 8 * 2^18 = 2^21 sequences


def batches(dist="normal", seeds=SEEDS, n=BATCH):
    for s in seeds:
        yield s, sample(n, s, dist)


def top3(x):
    """Final order statistics: positions and values of the three largest inputs."""
    order = np.argsort(-x, axis=1, kind="stable")
    pM, pS, pT = order[:, 0], order[:, 1], order[:, 2]
    r = np.arange(len(x))
    return pM, pS, pT, x[r, pM], x[r, pS], x[r, pT]


def describe(x, lg=None):
    """Per-sequence quantities for the census. lg: logits (computed if None)."""
    if lg is None:
        lg = logits(run(x))
    r = np.arange(len(x))
    pM, pS, pT, M, S, Tv = top3(x)
    ans = lg.argmax(1)
    rank = (x > x[r, ans][:, None]).sum(1) + 1          # rank of the answered input (1 = max)
    other = lg.copy(); other[r, pS] = -np.inf
    margin = lg[r, pS] - other.max(1)                    # > 0 iff correct
    return dict(pM=pM, pS=pS, pT=pT, M=M, S=S, T=Tv, ans=ans, rank=rank, margin=margin,
                ok=ans == pS, npos=(x > 0).sum(1), lg=lg)


def fmt_table(rows, header):
    """Markdown table from a list of rows (lists of str)."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def run_gates(x):
    """Forward pass; returns final state h_9 (B, 16) and all gates G (B, T, 16) as float."""
    H, Z = run(x, return_pre=True)
    return H, (Z > 0).astype(np.float64)


def coefs(v, G):
    """Exact input coefficients of the linear functional v . h_9 with the gates G of this input held fixed:
    v . h_9 = sum_p c[:, p] x_p.   v: (B, 16), G: (B, T, 16).  Returns c (B, T).   (backward pass of traces())"""
    B, Tn, _ = G.shape
    c = np.empty((B, Tn))
    g = v * G[:, -1]
    for t in range(Tn - 1, -1, -1):
        c[:, t] = g @ w
        if t > 0:
            g = (g @ W) * G[:, t - 1]
    return c
