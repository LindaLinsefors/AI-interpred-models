"""Reviewer R2, test 7.  Substitution test of the report's closed forms (the report gives R2 values only).
At the final step, on sequences with all four comparators on and S > 0.2 (the report's fitting set), replace the
comparators' vote V by a description of it and read the answer from m + V_description.
  (i)   closed form  V[p] = -60 (M - 0.87 S) cos(42.75 deg (p - pM))  for p = 0..7  (V[8], V[9] kept)
  (ii)  the same for p = 0..8                                               (V[9] kept)
  (iii) the report's regression  V[idx] = A(a1, idx)(M - S) + B(a1, idx) S + C(a1, idx) x_9, fitted per age of the max
        (in-sample; this is the fit with R2 0.88-0.97)
  (iv)  V rebuilt from the comparators' share of selected eigenmodes of the final state (exact mode coordinates)
python -B t7_substitution.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT, NAMES, IS_COMPLEX

x = sample(800_000, 9901)
H, Z = run(x, return_pre=True)
st = running_stats(x)
truth = second_argmax(x)
t = 9; B = len(x); ar = np.arange(B)
h9 = H[:, t]
Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
s = h9 @ U.T; V = h9 @ Uc.T; m = s - V
M, S, a1, pM = st["M1"][:, t], st["M2"][:, t], st["a1"][:, t], st["p1"][:, t]
ok = (Z[:, t][:, COMP] > 0).all(1) & (S > 0.2) & (a1 >= 1)
base = s.argmax(1)
print(f"fitting set: n = {ok.sum()} ({ok.mean():.3f} of sequences); network accuracy on it {np.mean(base[ok] == truth[ok]):.4f}")

def score(label, Vd):
    a = (m + Vd)[ok].argmax(1)
    r2 = 1 - ((V - Vd)[ok][:, :8] ** 2).sum() / ((V[ok][:, :8] - V[ok][:, :8].mean(0)) ** 2).sum()
    print(f"  {label:78s}: agreement with the network {np.mean(a == base[ok]):.4f}   accuracy {np.mean(a == truth[ok]):.4f}   (R2 of V over indices 0..7: {r2:.3f})")

om = np.radians(42.75); P = np.arange(10)[None, :]
cf = -60 * (M - 0.87 * S)[:, None] * np.cos(om * (P - pM[:, None]))
# the closed form describes V up to a position-independent term; fit that term per sequence on the replaced indices
for label, idx in (("(i)  closed form on indices 0..7", np.arange(8)), ("(ii) closed form on indices 0..8", np.arange(9))):
    Vd = V.copy()
    off = (V[:, idx] - cf[:, idx]).mean(1, keepdims=True)          # oracle offset: the most favourable choice
    Vd[:, idx] = cf[:, idx] + off
    score(label + ", per-sequence offset taken from the true V", Vd)

Vd = V.copy()
for a in range(1, 10):
    q = ok & (a1 == a)
    X = np.column_stack([(M - S)[q], S[q], x[q, t]])
    coef, *_ = np.linalg.lstsq(X, V[q], rcond=None)
    Vd[q] = X @ coef
score("(iii) regression on (M - S, S, x_9) per age of the max, all ten indices", Vd)
Vd2 = V.copy()
T3 = st["M3"][:, t]
a2 = st["a2"][:, t]
for a in range(1, 10):
    for b in range(0, 10):
        q = ok & (a1 == a) & (a2 == b)
        if q.sum() < 200: continue
        X = np.column_stack([(M - S)[q], S[q], np.maximum(T3[q], 0), x[q, t]])
        coef, *_ = np.linalg.lstsq(X, V[q], rcond=None)
        Vd2[q] = X @ coef
score("(iii') regression on (M - S, S, T+, x_9) per (age of max, age of runner-up) cell", Vd2)

oi = np.array(ORD); ci = [ORD.index(u) for u in COMP]
def comp_share(names):
    out = np.zeros((B, 13))
    for n in names:
        c = h9[:, oi] @ LEFT[n]
        out += (2 * np.real(np.multiply.outer(c, RIGHT[n])) if IS_COMPLEX[n] else np.real(np.multiply.outer(c, RIGHT[n])))
    keep = np.zeros(13); keep[ci] = 1
    return (out * keep) @ U[:, oi].T
score("(iv) comparators' share of all nine modes (sanity: must be exact)", comp_share(NAMES))
score("(iv) comparators' share of level + runner + clock + echo", comp_share(["level", "runner", "clock", "echo"]))
score("(iv) comparators' share of level + runner + clock + echo + short", comp_share(["level", "runner", "clock", "echo", "short"]))
score("(iv) comparators' share of level + runner + clock + echo + short + trans", comp_share(["level", "runner", "clock", "echo", "short", "trans"]))
