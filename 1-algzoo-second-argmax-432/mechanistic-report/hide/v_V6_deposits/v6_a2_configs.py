"""V6 attack: controlled sampling, one cell per configuration (pM, pS, pT) of the top three.

For iid continuous inputs the order statistics are independent of the rank permutation, so placing
the sorted values of an iid sample at prescribed positions (largest at pM, second at pS, third at
pT, the other seven in random order) samples exactly from the conditional law given the configuration.
The SAME base values (and the same shuffle of the other seven) are used for every configuration, so
configurations differ only in where the top three sit.

Per configuration (all 720) and per relative-gap bin of (S - T)/S (D4 bins, T > 0.2):
    counts of answer == pS, pT, pM, p4 (4th largest), other.
Per max-last configuration also: the D1 regression at the record step t = pM
    y = s_t[pS] - s_t[pT] ~ c1 (S - T) + c2 S   (D1 conditions S > 0.3, T > 0.1; here old max = S,
    old runner-up = T), P(y > 0), and how the sign of y at the record relates to the final answer.
Output: a2_configs.npz
"""
from v6_data import *

SEED, B = 5100, 100_000
rng = np.random.default_rng(SEED)
v = -np.sort(-rng.standard_normal((B, 10)), axis=1)          # descending: M, S, T, 4th, ...
rest = rng.permuted(v[:, 3:], axis=1)                        # the other seven, shuffled per sequence
i4 = rest.argmax(1)                                          # which of the 7 free slots holds the 4th largest
Mv, Sv, Tv = v[:, 0], v[:, 1], v[:, 2]
rg = (Sv - Tv) / Sv
edges = [0, .03, .08, .2, .5]
d4sel = Tv > 0.2
bins = np.stack([d4sel & (rg > edges[b]) & (rg <= edges[b + 1]) for b in range(4)])      # (4, B)
d1sel = (Sv > 0.3) & (Tv > 0.1)                              # D1 conditions on the old top two
ar = np.arange(B)

cnt = np.zeros((10, 10, 10, 4, 5), dtype=np.int64)           # [pM, pS, pT, bin, outcome]
nbin = bins.sum(1)
reg = np.full((10, 10, 10, 6), np.nan)                       # c1, c2, R2cen, P(y>0), n, resid sd of y/S
rec = np.zeros((10, 10, 10, 4, 4), dtype=np.int64)           # [.., bin, (y_rec<0, y_rec<0 & ansT, y_rec>0 & ansT, y_fin<0)]
ymed = np.full((10, 10, 10, 4, 2), np.nan)                   # median of y/S at the record and at t = 9, per bin
for pM in range(10):
    for pS in range(10):
        for pT in range(10):
            if len({pM, pS, pT}) < 3:
                continue
            free = [p for p in range(10) if p not in (pM, pS, pT)]
            x = np.empty((B, 10))
            x[:, pM], x[:, pS], x[:, pT] = Mv, Sv, Tv
            x[:, free] = rest
            p4 = np.asarray(free)[i4]
            H = run(x)
            lg = H[:, -1] @ U.T
            ans = lg.argmax(1)
            o = np.stack([ans == pS, ans == pT, ans == pM, ans == p4])
            o = np.concatenate([o, ~o.any(0, keepdims=True)])                      # (5, B)
            cnt[pM, pS, pT] = (bins[:, None, :] & o[None]).sum(-1)
            if pM > pS and pM > pT:
                t = pM
                s = H[:, t] @ U.T
                y = s[ar, 9 - (t - pS)] - s[ar, 9 - (t - pT)]
                yf = lg[ar, pS] - lg[ar, pT]
                b, r2c, _ = ols([(Sv - Tv)[d1sel], Sv[d1sel]], y[d1sel])
                res = (y[d1sel] - b[0] * (Sv - Tv)[d1sel] - b[1] * Sv[d1sel]) / Sv[d1sel]
                reg[pM, pS, pT] = [b[0], b[1], r2c, np.mean(y[d1sel] > 0), d1sel.sum(), res.std()]
                rec[pM, pS, pT] = np.stack([(bins & (y < 0)).sum(1), (bins & (y < 0) & o[1]).sum(1),
                                            (bins & (y > 0) & o[1]).sum(1), (bins & (yf < 0)).sum(1)], 1)
                for bb in range(4):
                    ymed[pM, pS, pT, bb] = [np.median(y[bins[bb]] / Sv[bins[bb]]), np.median(yf[bins[bb]] / Sv[bins[bb]])]
    print("pM", pM, "done", flush=True)
np.savez_compressed(HERE / "a2_configs.npz", cnt=cnt, nbin=nbin, reg=reg, rec=rec, ymed=ymed, seed=SEED, B=B,
                    n_d1=d1sel.sum())
print("bin sizes", nbin, " D1-selected:", d1sel.sum(), " P(T > 0.2) =", d4sel.mean())
