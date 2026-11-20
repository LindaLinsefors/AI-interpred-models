"""C5b. Why does a phase shift move the answer?  (1) exact attribution of the clock component present at step t to the
final logits, through the actual gates;  (2) shifting only the clock vs shifting every rotating mode by k of its own steps.
200k N(0,1) sequences, seed 1501."""
from cl import *

NS, SEED = 200_000, 1501
x, H, Z, st = gen(NS, SEED)
M, S, a1, p1 = st["M1"], pos(st["M2"]), st["a1"], st["p1"]
G = (Z > 0).astype(float)
true = second_argmax(x)
lg0 = logits(H); ans0 = lg0.argmax(1)
rank = (-x).argsort(1).argsort(1) + 1
ar = np.arange(NS)

# ---------------------------------------------------------------- (1) attribution
print("=== (1) [EXACT per sample] contribution of the clock component of h_t to the final logits: U J_{9<-t} (2 Re(z_t RIGHT_clock)), gates of the real run ===")
print("   rows: final rank of the input that is the max at step t (1 = still the max at the end, 2 = it ends as the runner-up, i.e. the correct answer, 3+ = overtaken twice)")
print("   columns: mean contribution at that input's own position | mean over the other nine positions | mean total logit there minus mean of the others | share of runs where the contribution at the own position is the largest of the ten")
tot = {r: [] for r in (1, 2, 3)}
for t in range(1, 9):
    sel = (M[:, t] > 0) & (a1[:, t] >= 1)
    idx = np.nonzero(sel)[0]
    zc = H[idx, t][:, ORD] @ LC
    v = np.zeros((idx.size, 16)); v[:, ORD] = 2 * np.real(np.multiply.outer(zc, RC))
    for tp in range(t + 1, 10):
        v = (v @ W.T) * G[idx, tp]
    c = v @ U.T                                              # (n, 10) contribution to the final logits
    q = p1[idx, t]
    rk = np.minimum(rank[idx, q], 3)
    own = c[np.arange(idx.size), q]
    oth = (c.sum(1) - own) / 9
    lown = lg0[idx, q]; loth = (lg0[idx].sum(1) - lown) / 9
    for r in (1, 2, 3):
        m_ = rk == r
        tot[r].append(np.column_stack([own[m_], oth[m_], (lown - loth)[m_], (c[m_].argmax(1) == q[m_]).astype(float), np.full(m_.sum(), t),
                                       (M[idx, t] - S[idx, t])[m_], a1[idx, t][m_]]))
for r in (1, 2, 3):
    A = np.concatenate(tot[r])
    print(f"   final rank {r if r < 3 else '3+'}: n = {A.shape[0]:7d} | {A[:, 0].mean():+7.2f} | {A[:, 1].mean():+7.2f} | {A[:, 2].mean():+7.2f} | {A[:, 3].mean():.3f}"
          f"   [own-position contribution per unit of gap (M - S+)_t, LS slope: {np.sum(A[:, 0] * A[:, 5]) / np.sum(A[:, 5] ** 2):+.2f}]")
A = np.concatenate(tot[2])
print("   final rank 2, by the age of the max at step t: mean own-position contribution / mean over other positions")
print("      " + "  ".join(f"a={a}: {A[A[:, 6] == a, 0].mean():+.2f}/{A[A[:, 6] == a, 1].mean():+.2f}" for a in range(1, 9)))
A1 = np.concatenate(tot[1])
print("   final rank 1 (no later record), by the age of the max at step t: mean own-position contribution / mean over other positions")
print("      " + "  ".join(f"a={a}: {A1[A1[:, 6] == a, 0].mean():+.2f}/{A1[A1[:, 6] == a, 1].mean():+.2f}" for a in range(1, 9)))

# ---------------------------------------------------------------- (2) clock only vs all rotating modes
print("\n=== (2) [INTERVENTION] phase shift by k steps at step t: clock only, versus every rotating mode by k of its own steps (h2, h3, short: exp(i k angle_n); alt: (-1)^k) ===")
rot = [n for n in NAMES if IS_COMPLEX[n]]
def shift(h, k, which):
    h2 = h.copy()
    d = 0
    for n in which:
        if IS_COMPLEX[n]:
            c = h[:, ORD] @ LEFT[n]
            d = d + 2 * np.real(np.multiply.outer((np.exp(1j * k * np.angle(LAM[n])) - 1) * c, RIGHT[n]))
        else:
            c = np.real(h[:, ORD] @ LEFT[n])
            d = d + np.real(np.multiply.outer(((-1.0) ** k - 1) * c, RIGHT[n]))
    h2[:, ORD] += d
    return h2

sets = (("clock only", ["clock"]), ("clock + h2 + h3", ["clock", "h2", "h3"]), ("all rotating modes + alt", rot + ["alt"]), ("all but the clock (h2,h3,short,alt)", ["h2", "h3", "short", "alt"]))
print("   class A: nothing above S+_t arrives after t;  C1: exactly one later record and nothing else above the old max's level M_t... (the then-max ends as the runner-up)")
print("   set                                  k | class A: changed, new-old=-k (of all runs) | class 'max at t ends as runner-up': changed, new-old=-k (of all runs), still correct")
for label, which in sets:
    for k in (-2, -1, 1, 2):
        out = {"A": [0, 0, 0], "R": [0, 0, 0, 0]}
        for t in range(1, 9):
            sel = (M[:, t] > 0) & (a1[:, t] >= 1)
            idx = np.nonzero(sel)[0]
            h2 = shift(H[idx, t], k, which)
            ans = (run(x[idx, t + 1:], h0=h2)[:, -1] @ U.T).argmax(1)
            xm = x[idx, t + 1:].max(1)
            clsA = xm < S[idx, t]
            clsR = rank[idx, p1[idx, t]] == 2
            d = ans - ans0[idx]
            out["A"][0] += clsA.sum(); out["A"][1] += (d != 0)[clsA].sum(); out["A"][2] += (d == -k)[clsA].sum()
            out["R"][0] += clsR.sum(); out["R"][1] += (d != 0)[clsR].sum(); out["R"][2] += (d == -k)[clsR].sum(); out["R"][3] += (ans == true[idx])[clsR].sum()
        print(f"   {label:36s} {k:+d} | {out['A'][1] / out['A'][0]:.3f}  {out['A'][2] / out['A'][0]:.3f}  (n={out['A'][0]}) | {out['R'][1] / out['R'][0]:.3f}  {out['R'][2] / out['R'][0]:.3f}  {out['R'][3] / out['R'][0]:.3f}  (n={out['R'][0]})")
