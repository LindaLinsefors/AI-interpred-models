"""D1 (step 1). Sufficient statistics for the regressions  h_t[ORD] ~ (M-S) u + S v (+ T, x_t)  per cell.

Cell = (t, a1, a2).  Columns of Z:  [g = M - S, S, T, x_t, h_t[ORD] (13), 1, h_t[PULSE] (3), echo-free h (13),
                                     pre-activation z_t[ORD] (13)]
Two inclusion rules:
  strict : S > 0 (so M > S > 0);  T replaced by max(T, 0)
  pos    : M > 0;  S, T replaced by their positive parts
Gram matrices G[rule][t, a1, a2 + 1] = Z^T Z are saved to d1_gram.npz  (a2 = -1 -> index 0).
Seeds 1001..1008, 200k sequences each (N(0,1)).   python -B d1_gram.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = range(1001, 1009)
NB = 200_000
K = 4 + 13 + 1 + 3 + 13 + 13
G = {r: np.zeros((10, 10, 11, K, K)) for r in ("strict", "pos")}
ECHO_L, ECHO_R = LEFT["echo"].real, RIGHT["echo"].real

for seed in SEEDS:
    x = sample(NB, seed)
    H, Zpre = run(x, return_pre=True)
    st = running_stats(x)
    M, S, T3 = st["M1"], st["M2"], st["M3"]
    Sf = np.where(np.isfinite(S), S, 0.0)
    Tf = np.where(np.isfinite(T3), T3, 0.0)
    Ho = H[..., ORD]
    Hfree = Ho - (Ho @ ECHO_L)[..., None] * ECHO_R            # echo coordinate removed
    tt = np.broadcast_to(np.arange(10), M.shape)
    cid = (tt * 10 + st["a1"]) * 11 + (st["a2"] + 1)
    for rule in ("strict", "pos"):
        if rule == "strict":
            keep = Sf > 0
            Su, Tu = Sf, np.maximum(Tf, 0)
        else:
            keep = M > 0
            Su, Tu = np.maximum(Sf, 0), np.maximum(Tf, 0)
        Z = np.concatenate([np.stack([M - Su, Su, Tu, x], -1), Ho, np.ones(M.shape + (1,)), H[..., PULSE], Hfree, Zpre[..., ORD]], -1)
        Zk, ck = Z[keep], cid[keep]
        order = np.argsort(ck, kind="stable")
        Zk, ck = Zk[order], ck[order]
        ids, lo = np.unique(ck, return_index=True)
        hi = np.append(lo[1:], len(ck))
        for c, a, b in zip(ids, lo, hi):
            t, a1, a2p = c // 110, (c // 11) % 10, c % 11
            G[rule][t, a1, a2p] += Zk[a:b].T @ Zk[a:b]
    print("seed", seed, "done", flush=True)

np.savez_compressed(HERE / "d1_gram.npz", strict=G["strict"], pos=G["pos"], seeds=np.array(list(SEEDS)), nb=NB)
print("saved; states per rule:", {r: int(G[r][..., 17, 17].sum()) for r in G})
