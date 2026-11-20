"""Verifier V1 (the tape): claims T1, T2, T5.  Own code, written from the claims file only.

T1: Sigma = U_o A pinv(U_o);  T2: R^2 of Sigma l_0 for l_1 on real states;  T5: exact products.
Run:  python -B v1_t1_t2_t5.py   (writes v1_t1_t2_t5.txt beside itself)
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import *          # noqa: E402
from modes import *           # noqa: E402

out = open(HERE / "v1_t1_t2_t5.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")


np.set_printoptions(precision=3, suppress=True, linewidth=220)
Uo = U[:, ORD]                       # (10, 13)
A = W[np.ix_(ORD, ORD)]              # (13, 13)

# ---------------------------------------------------------------- T1
P("=== T1 ===")
P("rank(U_o) =", np.linalg.matrix_rank(Uo), " singular values:", np.linalg.svd(Uo, compute_uv=False))
Sig = Uo @ A @ np.linalg.pinv(Uo)
rel = np.linalg.norm(Uo @ A - Sig @ Uo) / np.linalg.norm(Uo @ A)
P(f"Frobenius ||U_o A - Sigma U_o|| / ||U_o A|| = {rel:.4f}")
P("Sigma =")
P(np.array2string(Sig, precision=2, suppress_small=True))
sup = np.array([Sig[i, i + 1] for i in range(9)])
P("superdiagonal Sigma[i,i+1], i=0..8:", np.round(sup, 3))
mask = np.ones((10, 10), bool)
for i in range(9):
    mask[i, i + 1] = False
oth = np.abs(np.where(mask, Sig, 0.0))
P("max |other entry| per row (superdiagonal excluded), rows 0..9:", np.round(oth.max(1), 3))
P("max |other entry| in rows 0-6:", round(oth[:7].max(), 4), "at", np.unravel_index(oth[:7].argmax(), (7, 10)))
P("rows 0-6: number of other entries with |.|>0.3:", int((oth[:7] > 0.3).sum()), " >0.2:", int((oth[:7] > 0.2).sum()))
P("row 9 entries: min %.3f max %.3f" % (Sig[9].min(), Sig[9].max()), np.round(Sig[9], 3))
P("row 7:", np.round(Sig[7], 3))
P("row 8:", np.round(Sig[8], 3))
P("diag:", np.round(np.diag(Sig), 3))
P("eigenvalues of Sigma (|.|, angle deg):")
ev = np.linalg.eigvals(Sig)
for e in sorted(ev, key=lambda z: -abs(z)):
    P(f"   {abs(e):.4f} @ {np.degrees(np.angle(e)):8.2f}")
P("eigenvalues of A (|.|, angle deg):")
for e in sorted(np.linalg.eigvals(A), key=lambda z: -abs(z)):
    P(f"   {abs(e):.4f} @ {np.degrees(np.angle(e)):8.2f}")

# Per-row relative error of the Sigma model and of a PURE one-position shift (u_i A vs u_{i+1})
P("\nPer row i: ||u_i A - (Sigma U_o)_i|| / ||u_i A||   and pure shift  ||u_i A - u_{i+1}|| / ||u_i A||,  cos(u_i A, u_{i+1})")
UA = Uo @ A
for i in range(10):
    r1 = np.linalg.norm(UA[i] - (Sig @ Uo)[i]) / np.linalg.norm(UA[i])
    if i < 9:
        r2 = np.linalg.norm(UA[i] - Uo[i + 1]) / np.linalg.norm(UA[i])
        c = UA[i] @ Uo[i + 1] / np.linalg.norm(UA[i]) / np.linalg.norm(Uo[i + 1])
        r3 = np.linalg.norm(UA[i] - Uo[i]) / np.linalg.norm(UA[i])
        c3 = UA[i] @ Uo[i] / np.linalg.norm(UA[i]) / np.linalg.norm(Uo[i])
        P(f"  i={i}: Sigma-model {r1:.3f} | pure shift rel.err {r2:.3f} cos {c:.3f} | no shift (u_i) rel.err {r3:.3f} cos {c3:.3f}"
          f" | |u_i|={np.linalg.norm(Uo[i]):.2f} |u_i A|={np.linalg.norm(UA[i]):.2f}")
    else:
        P(f"  i={i}: Sigma-model {r1:.3f}")
Sh = np.zeros((10, 10))
for i in range(8):
    Sh[i, i + 1] = 1.0
P("pure shift (ones on Sigma[i,i+1], i=0..7), rows 0-7: ||U_o A - Sh U_o|| / ||U_o A|| =",
  round(np.linalg.norm((UA - Sh @ Uo)[:8]) / np.linalg.norm(UA[:8]), 4))
P("Sigma model, rows 0-7 only:", round(np.linalg.norm((UA - Sig @ Uo)[:8]) / np.linalg.norm(UA[:8]), 4))

# ---------------------------------------------------------------- T5
P("\n=== T5 ===")
P("U[9,COMP]*w[COMP] =", np.round(U[9, COMP] * w[COMP], 4), " sum =", round(float((U[9, COMP] * w[COMP]).sum()), 4))
P("U[9,4]*w[4] =", round(float(U[9, 4] * w[4]), 4), "  (U[9,4] =", round(float(U[9, 4]), 4), ", w4 =", round(float(w[4]), 4), ")")
P("U[8,5] =", round(float(U[8, 5]), 4))
P("W[5,ORD].w[ORD] =", round(float(W[5, ORD] @ w[ORD]), 4))
P("W[9,ORD].w[ORD] =", round(float(W[9, ORD] @ w[ORD]), 4))
allv = np.array([W[u, ORD] @ w[ORD] for u in range(16)])
P("W[u,ORD].w[ORD] for all u:", np.round(allv, 4))
others = [u for u in range(16) if u not in (5, 9)]
P("max |.| over u not in {5,9}:", round(float(np.abs(allv[others]).max()), 4), "at unit", others[int(np.abs(allv[others]).argmax())])
P("w =", np.round(w, 3))

# ---------------------------------------------------------------- T2
P("\n=== T2 ===  seeds 5001..5003, 400k sequences each")


def r2_pooled(pred, tgt, pos):
    """per-state mean over `pos` removed from both; 1 - SSres/SStot pooled over states and positions."""
    p = pred[:, pos] - pred[:, pos].mean(1, keepdims=True)
    y = tgt[:, pos] - tgt[:, pos].mean(1, keepdims=True)
    return 1 - ((y - p) ** 2).sum() / (y ** 2).sum()


def r2_variants(l0, l1, M_, pos):
    pred = l0 @ M_.T
    a = r2_pooled(pred, l1, pos)
    # variant b: no per-state centring, SS_tot about 0
    b = 1 - ((l1[:, pos] - pred[:, pos]) ** 2).sum() / (l1[:, pos] ** 2).sum()
    # variant c: SS_tot about the per-position grand mean (no per-state centring)
    y = l1[:, pos]
    c = 1 - ((y - pred[:, pos]) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()
    return a, b, c


pos9 = list(range(9))
pos8 = list(range(8))
res = {t: [] for t in range(10)}
for seed in (5001, 5002, 5003):
    x = sample(400_000, seed)
    H = run(x)
    for t in range(10):
        ho = H[:, t][:, ORD]
        l0 = ho @ Uo.T
        l1 = ho @ (Uo @ A).T
        a, b, c = r2_variants(l0, l1, Sig, pos9)
        # pure shift: l1[i] predicted by l0[i+1]
        sh = r2_pooled(l0 @ Sh.T, l1, pos8)
        ns = r2_pooled(l0, l1, pos8)                 # identity ("no shift") control
        sg8 = r2_pooled(l0 @ Sig.T, l1, pos8)
        res[t].append((a, b, c, sh, ns, sg8))
P(" t | R2 Sigma (claim def; pos 0-8, per-state mean removed) per seed | uncentred | about per-pos mean | pos 0-7: Sigma, pure shift, identity")
for t in range(10):
    r = np.array(res[t])
    P(f" {t} | {r[:,0].round(4)} mean {r[:,0].mean():.4f} | {r[:,1].mean():.4f} | {r[:,2].mean():.4f} | {r[:,5].mean():.4f} {r[:,3].mean():.4f} {r[:,4].mean():.4f}")
P("claim: t=3,5,7 -> 0.992, 0.991, 0.985")
out.close()
