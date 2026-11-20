"""N2 check of sections 4 and 5: numbers that follow from the weights alone, plus the two deterministic tables."""
import sys
from itertools import combinations
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from modes import A, LAM, LEFT, RIGHT, NAMES, IS_COMPLEX, coords

w4 = w[4]
g = -w[COMP]
ang = lambda z: np.degrees(np.angle(z))
print("w[COMP]", w[COMP], " w4", w4, " w2", w[2], " w14", w[14], " max|w| other wave", np.abs(w[[0, 2, 3, 10, 11, 12, 13, 15]]).max())

print("\n== 4.1 core ==")
Bc = W[np.ix_(COMP, COMP)] * g[None, :] / g[:, None]
print("Bc\n", np.round(Bc, 4))
print("b2", np.round(W[COMP, 2] / g, 4), " w4*b4", np.round(w4 * W[COMP, 4] / g, 4))
print("Bc 1", np.round(Bc.sum(1), 4), " column sums", np.round(Bc.sum(0), 3))
print("ratio |Bc1|/(|Bc|_F |1|)", np.linalg.norm(Bc.sum(1)) / (np.linalg.norm(Bc) * 2))
ev = np.linalg.eigvals(Bc)
print("eig Bc", [f"{abs(e):.4f}@{ang(e):.2f}" for e in ev])
print("W22 %.4f W24 %.4f  -W42/w4 %.4f  -W44/w4 %.4f  W24*w4 %.3f" % (W[2, 2], W[2, 4], -W[4, 2] / w4, -W[4, 4] / w4, W[2, 4] * w4))
print("W[j,2]/w_j", np.round(W[COMP, 2] / w[COMP], 4))
cosu = [W[u, COMP] @ w[COMP] / np.linalg.norm(W[u, COMP]) / np.linalg.norm(w[COMP]) for u in range(16)]
print("cos(W[u,COMP], w[COMP])", np.round(cosu, 3))
Aw = W[:, ORD] @ w[ORD]
print("W[:,ORD] w[ORD]:", np.round(Aw, 3), " max |.| excl 5, 9:", np.abs(np.delete(Aw, [5, 9])).max())

print("\n== 4.2 / 5 weights ==")
for (p, u) in [(7, 0), (4, 11), (3, 10)]:
    print(f"U[{p},{u}] = {U[p, u]:.3f}; argmin of column {u}: {U[:, u].argmin()}")
print("W[0,5] %.3f W[15,5] %.3f W[3,5] %.3f W[14,5] %.3f W[0,9] %.3f" % (W[0, 5], W[15, 5], W[3, 5], W[14, 5], W[0, 9]))
print("U[9,4] %.3f  U[9,4]*w4 %.3f  U[8,5] %.4f  U[8,9] %.3f  W[5,ORD]w[ORD] %.4f  x U[8,5] = %.3f" % (U[9, 4], U[9, 4] * w4, U[8, 5], U[8, 9], W[5, ORD] @ w[ORD], U[8, 5] * (W[5, ORD] @ w[ORD])))
print("W[5,4]*w4 %.3f  W[9,4]*w4 %.3f  w5 %.4f" % (W[5, 4] * w4, W[9, 4] * w4, w[5]))
print("W[5,8]*g8 %.3f (slope with 8 cut = %.3f)  W[5,7]*g7 %.3f" % (-W[5, 8] * w[8], W[5, ORD] @ w[ORD] + W[5, 8] * w[8] * -1 * -1 if False else W[5, ORD] @ w[ORD] - W[5, 8] * w[8], -W[5, 7] * w[7]))
print("(U[9]-U[i]).w[ORD], i=0..8:", np.round((U[9, ORD] - U[:9, ORD]) @ w[ORD], 2))

print("\n== 4.3 modes ==")
print("negative entries of A:", (A < 0).sum(), "of", A.size)
lv = RIGHT["level"].real
print("level vector / unit 2:", np.round(lv / lv[ORD.index(2)], 3))
for n in NAMES:
    c = LEFT[n] @ w[ORD]
    print(f"  input -> {n:6s}: per unit x: {c.real:+.4f}{c.imag:+.4f}i   |.|/w4 = {abs(c) / w4:.4f}")
# clock readout split
ci = [ORD.index(u) for u in COMP]; wi = [i for i in range(13) if i not in ci]
Uo = U[:, ORD]
def split_ratio(v, a, b, pos=slice(None)):
    Ra, Rb = Uo[pos][:, a] @ v[a], Uo[pos][:, b] @ v[b]
    return np.linalg.norm(Ra + Rb) / (np.linalg.norm(Ra) + np.linalg.norm(Rb))
v = RIGHT["clock"]
Rc, Rw = Uo[:, ci] @ v[ci], Uo[:, wi] @ v[wi]
print("2|R| comp", np.round(2 * abs(Rc), 2), "\n2|R| wave", np.round(2 * abs(Rw), 2), "\n2|R| sum ", np.round(2 * abs(Rc + Rw), 2))
print("phase difference comp-wave (deg), p=0..9:", np.round((ang(Rc) - ang(Rw)) % 360, 0))
print("ratio all 10 positions %.4f ; positions 0-7 %.4f" % (split_ratio(v, ci, wi), split_ratio(v, ci, wi, slice(0, 8))))
allr = [split_ratio(v, list(a), [i for i in range(13) if i not in a]) for a in combinations(range(13), 4)]
print("715 splits: min %.4f, second %.4f, comparators rank %d" % (min(allr), sorted(allr)[1], 1 + sum(r < split_ratio(v, ci, wi) for r in allr)))
rng = np.random.default_rng(7)
r1 = np.array([split_ratio(rng.standard_normal(13) + 1j * rng.standard_normal(13), ci, wi) for _ in range(20000)])
r2 = np.array([split_ratio(abs(v) * np.exp(2j * np.pi * rng.random(13)), ci, wi) for _ in range(20000)])
for nm, r in (("random complex", r1), ("clock magnitudes, random phases", r2)):
    print(f"  {nm}: min {r.min():.3f}, 0.1% quantile {np.quantile(r, 0.001):.3f}, median {np.median(r):.3f}, n below 0.29: {(r < 0.29).sum()}, n below 0.0575: {(r <= 0.0575).sum()}")

print("\n== 5.4 pulse writes ==")
for n in NAMES:
    c5, c9 = LEFT[n] @ W[ORD, 5], LEFT[n] @ W[ORD, 9]
    d = (ang(c5) - ang(LAM[n])) % 360
    print(f"  {n:6s}: unit 5 {abs(c5):.3f}@{ang(c5):7.1f}  unit 9 {abs(c9):.3f}@{ang(c9):7.1f}   arg c5 - arg lam = {d:6.1f}")
lam = LAM["clock"]; c5 = LEFT["clock"] @ W[ORD, 5]; c9 = LEFT["clock"] @ W[ORD, 9]
f = lam + c5 * W[5, 4] + c9 * W[9, 4]
print("clock first-step factor (pulses' h4-proportional part only): %.3f@%.1f" % (abs(f), ang(f)))
print("level: lam %.4f, -c5 W54 %.4f, -c9 W94 %.4f" % (LAM["level"].real, -(LEFT["level"] @ W[ORD, 5]).real * W[5, 4], -(LEFT["level"] @ W[ORD, 9]).real * W[9, 4]))
# Theta5 coefficients: z5(t+1) = c5x (x_t - Theta5) + ...; Theta5 = -W[5,ORD] A h_{t-1}[ORD] / c5x + pulses
c5x = W[5, ORD] @ w[ORD]
for n in ("level", "runner", "clock"):
    co = -(W[5, ORD] @ A @ RIGHT[n]) / c5x
    print(f"  Theta5 coefficient on {n}: {abs(co) * (2 if IS_COMPLEX[n] else 1) * np.sign(co.real if not IS_COMPLEX[n] else 1):+.5f}")

print("\n== 5.5 worked example ==")
x = np.array([0.8, -0.5, 0.3, 1.2, -1.0, 0.9, -0.7, 1.6, -0.2, 0.4])
H, Z = run(x, return_pre=True); H, Z = H[0], Z[0]
hp = np.zeros(16)
for t in range(10):
    th = {j: (W[j] @ hp) / (-w[j]) for j in COMP}; th4 = -(W[4] @ hp) / w4
    cut = ",".join(str(j) for j in COMP if Z[t, j] <= 0) or "none"
    z = coords(H[t])["clock"] / w4
    s = U @ H[t]; m = s - U[:, COMP] @ H[t, COMP]
    idx = np.arange(10)
    tops = int(np.where(idx >= 9 - t, s, -np.inf).argmax()) + t - 9
    topm = int(np.where((idx >= 9 - t) & (idx <= 8), m, -np.inf).argmax()) + t - 9 if t else -1
    print(f"{t} {x[t]:5.1f} | th4 {th4:5.2f} th1 {th[1]:5.2f} th6 {th[6]:5.2f} th7 {th[7]:5.2f} th8 {th[8]:5.2f} | cut {cut:8s} | h4 {H[t,4]:5.2f} h5 {H[t,5]:5.2f} h9 {H[t,9]:5.2f} | clock {abs(z):.2f} @ {ang(z):5.0f} | top s {tops} top m {topm}")
print("answer", (U @ H[9]).argmax())

print("\n== 5.3 sweep ==")
def seq(v): return np.array([0.5, -1, 1.5, -1, -1, -1, v, -1, -1, -1.0])
H0 = run(seq(0.0))[0]
print("thresholds before v: th8 %.3f th1 %.3f th4 %.3f th7 %.3f th6 %.3f" % tuple([(W[j] @ H0[5]) / (-w[j]) for j in (8, 1)] + [-(W[4] @ H0[5]) / w4] + [(W[j] @ H0[5]) / (-w[j]) for j in (7, 6)]))
for v in (-1, -0.5, 0, 0.2, 0.29, 0.3, 0.45, 0.55, 0.85, 1.15, 1.30, 1.45, 1.55, 1.80):
    Hh, Zz = run(seq(v), return_pre=True); Hh, Zz = Hh[0], Zz[0]
    lg = U @ Hh[9]; ans = lg.argmax(); xs = seq(v); true = np.argsort(xs)[-2]
    cut = ",".join(str(j) for j in COMP if Zz[6, j] <= 0) or "none"
    print(f"v {v:5.2f} | cut {cut:8s} | h4 {Hh[6,4]:5.2f} | h5(t+1) {Hh[7,5]:6.3f} | clock(9) {abs(coords(Hh[9])['clock'])/w4:.3f} | answer {ans} (true {true}) margin of true over best other {lg[true] - np.delete(lg, true).max():6.2f}")
vs = np.arange(0, 2.5001, 0.001)
ans = np.array([(U @ run(seq(v))[0, 9]).argmax() for v in vs])
true = np.array([np.argsort(seq(v))[-2] for v in vs])
bad = vs[ans != true]
print("wrong v:", [(round(a, 3), round(b, 3)) for a, b in zip(bad[np.r_[True, np.diff(bad) > 0.0015]], bad[np.r_[np.diff(bad) > 0.0015, True]])], " share of [0,2.5]: %.4f" % (ans != true).mean())
print("unit 5 switches on at v =", vs[np.argmax([run(seq(v))[0, 7, 5] > 0 for v in vs])])
