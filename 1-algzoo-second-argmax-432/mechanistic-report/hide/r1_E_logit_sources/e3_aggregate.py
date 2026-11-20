"""E3. Aggregate every source class's contribution to the logits at the true runner-up / max / third positions
and to the three margins, on 2^19 N(0,1) sequences (seeds 1010..1017, 65536 each).

Classes: X raw inputs | P4, P5, P9 pulse activations | C1, C6, C7, C8 comparator clips | CW clips of the 9 wave units.
Roles (which input caused the event): the final max / runner-up / third / any other input.  The causing input of an
event at step s is x_s for X, P4 and clips, and x_{s-1} for P5 and P9 (they fire one step later; step-0 firings of
units 5, 9 are put in 'other').
Targets: L[p2] (runner-up), L[p1] (max), L[p3] (third), L[q] with q = the best position other than p2.

Writes e3_cache.npz (per-sequence class x role x target contributions, float32) beside this script and prints tables.
"""
from src import *
import time

HERE = Path(__file__).resolve().parent
CLASSES = ["X", "P4", "P5", "P9", "C1", "C6", "C7", "C8", "CW"]
ROLES = ["max", "runner", "third", "other"]
TARGETS = ["L[p2]", "L[p1]", "L[p3]", "L[q]"]
SEEDS = list(range(1010, 1018))
NB = 65536


def batch(seed):
    x = sample(NB, seed)
    H, Z, p, c = amplitudes(x)
    L = logits(H)
    order = np.argsort(-x, axis=1)
    p1, p2, p3 = order[:, 0], order[:, 1], order[:, 2]
    Lm = L.copy(); Lm[np.arange(NB), p2] = -np.inf
    q = Lm.argmax(1)
    pos = np.stack([p2, p1, p3, q], 1)
    CX, CP, CC = contributions(x, p, c, pos)                 # (B,10,4), (B,10,3,4), (B,10,13,4)
    # additivity check in float64
    tot = CX.sum(1) + CP.sum((1, 2)) + CC.sum((1, 2))
    err = np.abs(tot - np.take_along_axis(L, pos, 1)).max()
    # roles of each step: 0 max, 1 runner, 2 third, 3 other
    s = np.arange(10)[None, :]
    role = np.full((NB, 10), 3)
    role[s == p3[:, None]] = 2; role[s == p2[:, None]] = 1; role[s == p1[:, None]] = 0
    R = np.eye(4)[role]                                       # (B, 10, 4) one-hot, role of the input at step s
    Rlag = np.concatenate([np.eye(4)[np.full((NB, 1), 3)], R[:, :-1]], 1)     # role of the input at step s-1
    out = np.empty((NB, 9, 4, 4), dtype=np.float64)
    out[:, 0] = np.einsum("bsr,bst->brt", R, CX)
    out[:, 1] = np.einsum("bsr,bst->brt", R, CP[:, :, 0])
    out[:, 2] = np.einsum("bsr,bst->brt", Rlag, CP[:, :, 1])
    out[:, 3] = np.einsum("bsr,bst->brt", Rlag, CP[:, :, 2])
    for k, j in enumerate(COMP):
        out[:, 4 + k] = np.einsum("bsr,bst->brt", R, CC[:, :, ORD.index(j)])
    out[:, 8] = np.einsum("bsr,bst->brt", R, CC[:, :, WI].sum(2))
    err2 = np.abs(out.sum((1, 2)) - np.take_along_axis(L, pos, 1)).max()
    wave_units = CC[:, :, WI].sum(1)                          # (B, 9 wave units, 4 targets)
    # age-0 part (events at step 9, read directly by U) of each class, for E4
    return dict(out=out.astype(np.float32), wave=wave_units.astype(np.float32), x=x.astype(np.float32), L=L.astype(np.float32),
                pos=pos.astype(np.int8), err=max(err, err2))


if __name__ == "__main__":
    t0 = time.time()
    res = [batch(sd) for sd in SEEDS]
    print(f"computed {len(SEEDS) * NB} sequences in {time.time() - t0:.0f} s; max additivity error (classes vs exact logits) = {max(r['err'] for r in res):.2e}")
    np.savez(HERE / "e3_cache.npz", **{k: np.concatenate([r[k] for r in res]) for k in ("out", "wave", "x", "L", "pos")})
