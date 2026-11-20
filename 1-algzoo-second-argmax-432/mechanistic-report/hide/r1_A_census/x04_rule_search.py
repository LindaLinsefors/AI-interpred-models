"""exploration: best 1- and 2-feature Boolean rules for each unit's gate at t >= 2 (one batch, subsampled)."""
from cc import *
import itertools

seed, x, H, Z = next(batches())
ev = events(x); G = Z > 0; cls = ev["cls"]
B = len(x)
Mp0, Sp0 = np.maximum(ev["Mp"], 0), np.maximum(ev["Sp"], 0)
Rp = x > Mp0                                  # positive new max (t = 0 with x_0 > 0 included)
Np = (x > Sp0) & (x < ev["Mp"])               # positive new runner-up

def lag(a, k):                                # a[:, t-k], False where t-k < 0
    out = np.zeros_like(a)
    if k == 0: return a.copy()
    out[:, k:] = a[:, :-k]
    return out

feats, names = [], []
for k in range(0, 9):
    feats.append(lag(Rp, k)); names.append(f"R+(t-{k})")
    feats.append(lag(Np, k)); names.append(f"N+(t-{k})")
for k in range(0, 3):
    feats.append(lag(x > 0, k)); names.append(f"x(t-{k})>0")
for k in range(1, 7):
    for j in range(N):
        feats.append(lag(G[:, :, j], k)); names.append(f"g{j}(t-{k})")
ng_same = len(feats)
for j in range(N):
    feats.append(G[:, :, j]); names.append(f"g{j}(t)")
F = np.stack(feats, -1)[:, 2:].reshape(-1, len(feats))          # steps t >= 2
Y = G[:, 2:].reshape(-1, N)
rng = np.random.default_rng(5000)
idx = rng.choice(len(F), 400_000, replace=False)
F = F[idx].astype(np.float32); Y = Y[idx].astype(np.float32)
n = len(F); nf = F.shape[1]
print("n", n, "features", nf)
Fc = 1 - F
for i in range(N):
    y = Y[:, i]
    base = max(y.mean(), 1 - y.mean())
    # single features
    c1 = F.T @ y; c0 = F.T @ (1 - y); d1 = Fc.T @ y; d0 = Fc.T @ (1 - y)
    acc1 = (np.maximum(c1, c0) + np.maximum(d1, d0)) / n
    same = np.array([nm == f"g{i}(t)" for nm in names])
    acc1[same] = 0
    o = np.argsort(-acc1)[:4]
    s = f"unit {i:2d} base {base:.4f} | 1-feat: " + "; ".join(f"{names[k]} {acc1[k]:.4f}" for k in o)
    print(s)
    # pairs: cells (f1, f2) in {11, 10, 01, 00}
    Fy = F * y[:, None]; Fny = F * (1 - y[:, None])
    n11y = Fy.T @ F; n11n = Fny.T @ F
    f_y = Fy.sum(0); f_n = Fny.sum(0); ny = y.sum(); nn = n - ny
    n10y = f_y[:, None] - n11y; n10n = f_n[:, None] - n11n
    n01y = f_y[None, :] - n11y; n01n = f_n[None, :] - n11n
    n00y = ny - f_y[:, None] - f_y[None, :] + n11y; n00n = nn - f_n[:, None] - f_n[None, :] + n11n
    acc2 = (np.maximum(n11y, n11n) + np.maximum(n10y, n10n) + np.maximum(n01y, n01n) + np.maximum(n00y, n00n)) / n
    acc2[same, :] = 0; acc2[:, same] = 0
    for label, mask in (("events only", np.arange(nf) < 21), ("lagged gates+events", np.arange(nf) < ng_same), ("all incl same-step gates", np.ones(nf, bool))):
        a = acc2.copy(); a[~mask, :] = 0; a[:, ~mask] = 0
        a = np.triu(a, 1)
        best = np.dstack(np.unravel_index(np.argsort(-a, axis=None)[:3], a.shape))[0]
        print(f"      2-feat [{label}]: " + "; ".join(f"{names[p]} & {names[q]} {a[p, q]:.4f}" for p, q in best))
