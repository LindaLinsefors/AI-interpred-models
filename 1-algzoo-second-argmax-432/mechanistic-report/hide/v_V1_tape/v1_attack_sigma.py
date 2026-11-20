"""Verifier V1, attack E1: is Sigma a shift for trivial reasons?  Is the T2 R^2 informative?

Null models (300 draws each, rng seed 5301):
  randA   : A replaced by a Gaussian matrix with the same Frobenius norm (U real)
  rotA    : A replaced by Q A Q^T, Q random orthogonal (same eigenvalues, alignment with U destroyed)
  randU   : U_o replaced by a Gaussian matrix with the same Frobenius norm (A real)
  permU   : columns of U_o permuted at random (which unit each readout weight belongs to)
  rowrot  : U_o replaced by R U_o, R random orthogonal 10x10 (same row space, position basis destroyed)
Reported: relative Frobenius residual, mean / min superdiagonal (i=0..6), max |other| in rows 0-6,
          mean cos(u_i A, u_{i+1}) i=0..6, and on REAL states (seed 5302, 100k sequences, t=3,5,7):
          T2 R^2 (Sigma of the null model; positions 0-8; per-state mean removed) and the pure-shift R^2
          (l1[i] vs l0[i+1], i=0..7).
Run: python -B v1_attack_sigma.py -> v1_attack_sigma.txt
"""
from vlib import *

P = Tee("v1_attack_sigma.txt")
rng = np.random.default_rng(5301)
Uo = U[:, ORD]
A = W[np.ix_(ORD, ORD)]
ND = 300

x = sample(100_000, 5302)
H = run(x)
HO = {t: H[:, t][:, ORD] for t in (3, 5, 7)}


def r2c(pred, y):
    pred = pred - pred.mean(1, keepdims=True); y = y - y.mean(1, keepdims=True)
    return 1 - ((y - pred) ** 2).sum() / (y ** 2).sum()


def stats(Uo_, A_, data=True):
    Sig = Uo_ @ A_ @ np.linalg.pinv(Uo_)
    UA = Uo_ @ A_
    rel = np.linalg.norm(UA - Sig @ Uo_) / np.linalg.norm(UA)
    rel08 = np.linalg.norm((UA - Sig @ Uo_)[:9]) / np.linalg.norm(UA[:9])
    sup = np.array([Sig[i, i + 1] for i in range(7)])
    oth = np.abs(Sig[:7].copy())
    for i in range(7):
        oth[i, i + 1] = 0
    cs = np.mean([UA[i] @ Uo_[i + 1] / np.linalg.norm(UA[i]) / np.linalg.norm(Uo_[i + 1]) for i in range(7)])
    out = [rel, rel08, sup.mean(), sup.min(), oth.max(), cs]
    if data:
        for t in (3, 5, 7):
            l0 = HO[t] @ Uo_.T; l1 = HO[t] @ UA.T
            out.append(r2c((l0 @ Sig.T)[:, :9], l1[:, :9]))
        for t in (3, 5, 7):
            l0 = HO[t] @ Uo_.T; l1 = HO[t] @ UA.T
            out.append(r2c(l0[:, 1:9], l1[:, 0:8]))
    return np.array(out)


names = ["rel.resid", "rel.resid rows0-8", "mean superdiag", "min superdiag", "max|other| rows0-6", "mean cos(u_iA,u_i+1)",
         "T2 R2 t=3", "t=5", "t=7", "pure-shift R2 t=3", "t=5", "t=7"]
real = stats(Uo, A)
P("REAL network:")
for n, v in zip(names, real):
    P(f"   {n:24s} {v:8.4f}")


def qr_orth(n):
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    return Q * np.sign(np.diag(R))


nulls = {
    "randA": lambda: (Uo, rng.standard_normal((13, 13)) * np.linalg.norm(A) / np.sqrt(169)),
    "rotA": lambda: (Uo, (lambda Q: Q @ A @ Q.T)(qr_orth(13))),
    "randU": lambda: (rng.standard_normal((10, 13)) * np.linalg.norm(Uo) / np.sqrt(130), A),
    "permU": lambda: (Uo[:, rng.permutation(13)], A),
    "rowrot": lambda: (qr_orth(10) @ Uo, A),
}
for name, gen in nulls.items():
    use_data = name in ("randU", "permU", "rowrot")      # states of the real network are only meaningful with the real A
    R = np.array([stats(*gen(), data=use_data) for _ in range(ND)])
    P(f"\nNULL {name}  ({ND} draws): mean, sd, [min, max], and how many draws reach the real value")
    for k in range(R.shape[1]):
        col = R[:, k]
        better = (col <= real[k]).sum() if k in (0, 1, 4) else (col >= real[k]).sum()
        P(f"   {names[k]:24s} {col.mean():8.4f} {col.std():8.4f}  [{col.min():8.4f}, {col.max():8.4f}]   draws at least as extreme as real: {better}")

# How much of the state lies in the null space of U_o?  (the T2 residual is exactly U_o A N N^T h)
N_ = np.linalg.svd(Uo)[2][10:].T                      # (13, 3) null-space basis
P("\nShare of |h[ORD]|^2 in the 3-dim null space of U_o, real states t=3,5,7:",
  [round(float(((HO[t] @ N_) ** 2).sum() / (HO[t] ** 2).sum()), 4) for t in (3, 5, 7)])
P("  (a random 3-dim subspace would hold a share of about 3/13 = 0.231 of an isotropic state)")
P("  PCA of h[ORD] at t=5: cumulative variance share of the top PCs (uncentred second moment):",
  np.round(np.cumsum(np.linalg.svd(HO[5], compute_uv=False) ** 2) / (HO[5] ** 2).sum(), 4))
