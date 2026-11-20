"""H3. What an N event writes (counterfactual: the same sequence with x_t replaced by 0).

For every N event (S_old < x_t < M_old, S_old > 0, unit 4 silent, 2 <= t <= 8) we run the sequence twice,
as it is and with x_t := 0, and store
  * the difference of the mode coordinates at t+2 (t <= 7),
  * the difference of the final logits, split EXACTLY by source:
      in    : direct input term             Kin(9-t) x_t
      comp  : clip corrections of the comparators at t
      wave  : clip corrections of the wave units at t
      u5,u9,u4 : pulse units at t+1
      clip1 : clip corrections of all ordinary units at t+1
      later : every source at steps >= t+2 (knock-on)
Results are cached in h3_event.npz (this folder).
"""
from hcommon import *
from h3_kernels_exact import Kin, Kp, Kc

REL = list(range(-4, 4))
GROUPS = ["in", "comp", "wave", "u5", "u9", "u4", "clip1", "later", "total"]
CI, WI = [OI[j] for j in COMP], [OI[j] for j in WAVE]
relu = lambda v: np.maximum(v, 0)


def at(L, pos):
    """L (n, 10) logits, pos (n,) positions (may be out of range -> nan)."""
    okp = (pos >= 0) & (pos <= 9)
    out = np.full(len(pos), np.nan)
    out[okp] = L[np.nonzero(okp)[0], pos[okp]]
    return out


def collect(seeds=(1011, 1012), n=200_000):
    cols = {}
    for seed in seeds:
        x = sample(n, seed)
        e = Ev(x)
        for t in range(2, 9):
            sel = (e.So[:, t] > 0) & (x[:, t] > e.So[:, t]) & (x[:, t] < e.Mo[:, t]) & ~e.G[:, t, 4]
            xs = x[sel]; nb = len(xs)
            x0 = xs.copy(); x0[:, t] = 0.0
            H1, Z1 = e.H[sel], e.Z[sel]
            H0, Z0 = run(x0, return_pre=True)
            L1, L0 = logits(H1), logits(H0)
            dC = relu(-Z1[:, :, ORD]) - relu(-Z0[:, :, ORD])
            dP = H1 - H0
            d = 9 - t
            g = {"in": np.outer(xs[:, t], Kin[d]),
                 "comp": dC[:, t][:, CI] @ Kc[d][CI],
                 "wave": dC[:, t][:, WI] @ Kc[d][WI],
                 "u5": np.outer(dP[:, t + 1, 5], Kp[5][d - 1]),
                 "u9": np.outer(dP[:, t + 1, 9], Kp[9][d - 1]),
                 "u4": np.outer(dP[:, t + 1, 4], Kp[4][d - 1]),
                 "clip1": dC[:, t + 1] @ Kc[d - 1],
                 "total": L1 - L0}
            g["later"] = g["total"] - sum(g[k] for k in GROUPS[:-2])
            p1, p2 = e.p1o[sel, t], e.p2o[sel, t]
            rec = dict(seed=np.full(nb, seed), t=np.full(nb, t), a=e.a1[sel, t], a2=e.a2[sel, t], x=xs[:, t], S=e.So[sel, t], M=e.Mo[sel, t],
                       T=e.To[sel, t], qmax=xs[:, t + 1:].max(1), q2=(xs[:, t + 1:t + 3].max(1) if t <= 7 else np.full(nb, np.nan)),
                       ncomp=(~e.G[sel, t][:, COMP]).sum(1), h5=H1[:, t + 1, 5], h5cf=H0[:, t + 1, 5], h9=H1[:, t + 1, 9],
                       L1_t=L1[:, t], L1_p2=at(L1, p2), L1_p1=at(L1, p1), L0_t=L0[:, t], L0_p2=at(L0, p2), L0_p1=at(L0, p1),
                       ans1=L1.argmax(1), ans0=L0.argmax(1), p1=p1, p2=p2,
                       cf_quiet=(~(Z0[:, t][:, COMP] <= 0).any(1)) & (H0[:, t + 1, 5] == 0))
            for k in GROUPS:
                for r in REL:
                    rec[f"{k}_r{r}"] = at(g[k], np.full(nb, t + r))
                rec[f"{k}_p1"] = at(g[k], p1); rec[f"{k}_p2"] = at(g[k], p2)
            if t <= 7:
                m1, m0 = coords(H1[:, t + 2]), coords(H0[:, t + 2])
                for nme in NAMES:
                    rec[f"dm_{nme}"] = m1[nme] - m0[nme]
            else:
                for nme in NAMES:
                    rec[f"dm_{nme}"] = np.full(nb, np.nan, dtype=complex if IS_COMPLEX[nme] else float)
            for k, v in rec.items():
                cols.setdefault(k, []).append(v)
        del e
    return {k: np.concatenate(v) for k, v in cols.items()}


CACHE = HIDE / "r1_H_runnerup_events" / "h3_event.npz"


def load_events():
    """Cached events (about 240 MB, regenerated in about a minute if the file is missing)."""
    if not CACHE.exists():
        np.savez(CACHE, **collect())
    return dict(np.load(CACHE))


if __name__ == "__main__":
    d = load_events()
    print("events:", len(d["t"]), " by age:", {a: int((d["a"] == a).sum()) for a in range(1, 9)})
    print("counterfactual x_t=0 is itself quiet (no comparator clipped at t, unit 5 off at t+1):", d["cf_quiet"].mean())
