"""More attacks on the 'two score vectors' reading.

 a. The lead's S2 regression applied to m and to s = m + V (same selection: t = 9, all four comparators on,
    S > 0.1, per a1 = 2..9, aligned on d = i - p1, i <= 7).  Does m carry the OPPOSITE cosine?  What is left in s?
 b. Longitudinal rotation: for the same sequence, complex amplitude Z_t of V_t over the indices of inputs that are
    >= 2 steps old at t (so the index window slides with t), on steps without an event. If the vote shifts with the
    tape, Z_{t+1} = Z_t e^{+i omega} in index space.
 c. Does the amplitude depend on the third largest value T?  Re K ~ kM M + kS S + kT max(T, 0) + kx x9.

Usage: python -B x6_m_wave_rotation.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *
from s2_s3 import by_d, fit, cosfit

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 5600
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 10

pos = np.arange(8.0)
G = np.stack([np.cos(OMEGA * pos), np.sin(OMEGA * pos), np.ones(8)], 1)
PG = np.linalg.pinv(G)


def zamp(v8):
    """Complex amplitude Z of an 8-vector over indices 0..7: v ~ Re(Z e^{i omega i}) + c."""
    co = v8 @ PG.T
    return co[:, 0] - 1j * co[:, 1]


Vs, ms, Xs, a1s, Ks, Ts = [], [], [], [], [], []
ratio = {t: [] for t in range(3, 9)}; ratio_ev = {t: [] for t in range(3, 9)}
for x, H, M, S, p1, p2 in batches(nb, seed0):
    h = H[:, 9]
    sel = (h[:, COMP] > 0).all(1) & (S[:, 9] > 0.1)
    s, V, m = scores(h[sel])
    Vs.append(V); ms.append(m)
    Xs.append(np.stack([(M[:, 9] - S[:, 9])[sel], S[sel, 9], x[sel, 9]], 1)); a1s.append(9 - p1[sel, 9])
    Ks.append(-zamp(V[:, :8]) * np.exp(1j * OMEGA * p1[sel, 9]))
    T3 = np.sort(x, 1)[:, -3]
    Ts.append(T3[sel])
    # b. longitudinal: indices 0..7 at t = 9 only exist for all inputs; at general t use the complex amplitude of the
    # comparators directly: V[i] = sum_j U[i, j] h_j  ->  Z = zamp(U[:8, COMP] h_c), defined for any t (index space)
    for t in range(3, 9):
        Za = zamp(H[:, t][:, COMP] @ UC[:8].T); Zb = zamp(H[:, t + 1][:, COMP] @ UC[:8].T)
        on = (H[:, t][:, COMP] > 0).all(1) & (H[:, t + 1][:, COMP] > 0).all(1) & (S[:, t] > 0.1) & (p1[:, t] <= t - 1)
        quiet = on & (x[:, t + 1] < S[:, t])                              # no new max / runner-up at t + 1
        ratio[t].append((Zb / Za)[quiet][:40000])
        ev = on & (x[:, t + 1] > S[:, t]) & (x[:, t + 1] < M[:, t])       # new runner-up at t + 1, all four still on
        ratio_ev[t].append((Zb / Za)[ev][:40000])
V = np.concatenate(Vs); m = np.concatenate(ms); X = np.concatenate(Xs); a1 = np.concatenate(a1s)
K = np.concatenate(Ks); T3 = np.concatenate(Ts)
s = m + V
print(f"seeds {seed0}..{seed0 + nb - 1}; selected rows {len(V)}")

print("\n=== a. the lead's regression for V, m and s = m + V, aligned on d = i - p1 (i <= 7), averaged over a1 = 2..9 ===")
tabs = {}
for nm, Y in (("V", V), ("m", m), ("s", s)):
    A, B, C = {}, {}, {}
    for a in range(2, 10):
        A[a], B[a], C[a] = fit(X[a1 == a], Y[a1 == a])[0]
    tabs[nm] = (by_d(A), by_d(B), by_d(C), A, B)
print("d:        " + " ".join(f"{d:+7d}" for d in range(-7, 8)))
for nm in ("V", "m", "s"):
    print(f"A^{nm}(d):  " + " ".join(f"{tabs[nm][0][d][0]:7.1f}" for d in range(-7, 8)))
for nm in ("V", "m", "s"):
    print(f"B^{nm}(d):  " + " ".join(f"{tabs[nm][1][d][0]:7.1f}" for d in range(-7, 8)))
for nm in ("V", "m", "s"):
    print(f"C^{nm}(d):  " + " ".join(f"{tabs[nm][2][d][0]:7.1f}" for d in range(-7, 8)))
ds_ = np.arange(-7, 8.0)
for nm in ("V", "m", "s"):
    y = np.array([tabs[nm][0][d][0] for d in range(-7, 8)])
    amp, ph, c, r2 = cosfit(y, ds_)
    print(f"cosine fit of A^{nm}(d): amplitude {amp:5.1f}, phase {np.degrees(ph):7.1f} deg (180 = trough at the max, 0 = crest at the max), offset {c:+.1f}, R^2 {r2:.3f}")
print("A^s by a1 (rows) and index i = 0..9 (the net coefficient of M - S in the final score; the max is at i = 9 - a1):")
for a in range(2, 10):
    print(f"a1={a} p1={9 - a}  " + " ".join(f"{v:7.1f}" for v in tabs["s"][3][a]))
print("B^s by a1 and index (coefficient of S in the final score):")
for a in range(2, 10):
    print(f"a1={a} p1={9 - a}  " + " ".join(f"{v:7.1f}" for v in tabs["s"][4][a]))
print("per a1: cosine fit of A^m_i over i = 0..7: amplitude, position of the CREST, R^2  (V's trough amplitude for comparison)")
for a in range(2, 10):
    am, ph, c, r2 = cosfit(tabs["m"][3][a][:8], pos)
    av = cosfit(tabs["V"][3][a][:8], pos)[0]
    per = 360 / np.degrees(OMEGA)
    print(f"a1={a} p1={9 - a}: m amp {am:5.1f}, crest at i = {(np.degrees(ph) / np.degrees(OMEGA)) % per:5.2f}, R^2 {r2:.3f};   V amp {av:5.1f};  ratio m/V {am / av:.2f}")
# share of per-state variance over i<=7 that is an omega-sinusoid, for m and s
for nm, Y in (("V", V), ("m", m), ("s", s)):
    Y8 = Y[:, :8]; fitv = (Y8 @ PG.T) @ G.T
    print(f"share of the per-state variance of {nm} over i = 0..7 that is an omega-sinusoid: {1 - ((Y8 - fitv) ** 2).sum() / ((Y8 - Y8.mean(1, keepdims=True)) ** 2).sum():.3f};"
          f"  rms over states and positions of the centred vector: {np.sqrt(((Y8 - Y8.mean(1, keepdims=True)) ** 2).mean()):.1f}")
Zm = zamp(m[:, :8]); Zv = zamp(V[:, :8])
k = a1 >= 2
print(f"per sequence (a1 >= 2): Z_m / Z_V: median real part {np.median((Zm / Zv)[k].real):.3f}, median imaginary part {np.median((Zm / Zv)[k].imag):.3f};"
      f"  |Z_m + Z_V| / |Z_V| median {np.median(np.abs(Zm + Zv)[k] / np.abs(Zv)[k]):.3f}")

print("\n=== b. longitudinal rotation of the comparators' complex amplitude between t and t + 1 (index space) ===")
print("quiet steps (all four on at both steps, S_t > 0.1, max not the current input, x_{t+1} < S_t): arg(Z_{t+1}/Z_t) in deg: median [IQR];  |Z_{t+1}/Z_t| median")
allr = []
for t in range(3, 9):
    r = np.concatenate(ratio[t]); allr.append(r)
    ang = np.degrees(np.angle(r))
    print(f"t={t}->{t + 1}: n = {len(r):6d}  angle median {np.median(ang):6.2f} [{np.percentile(ang, 25):6.2f}, {np.percentile(ang, 75):6.2f}]   modulus median {np.median(np.abs(r)):.3f}")
r = np.concatenate(allr); ang = np.degrees(np.angle(r))
print(f"pooled: angle median {np.median(ang):.2f}, mean of unit phasors -> {np.degrees(np.angle(np.mean(r / np.abs(r)))):.2f}; P(20 < angle < 65) = {np.mean((ang > 20) & (ang < 65)):.3f}")
r = np.concatenate([np.concatenate(ratio_ev[t]) for t in range(3, 9)]); ang = np.degrees(np.angle(r))
print(f"steps with a new runner-up at t + 1 and all four still on: n = {len(r)}, angle median {np.median(ang):.2f} [{np.percentile(ang, 25):.2f}, {np.percentile(ang, 75):.2f}], modulus median {np.median(np.abs(r)):.3f}")

print("\n=== c. amplitude Re K against M, S, max(T, 0), x9 (a1 >= 2; t = 9; T = third largest of the sequence) ===")
k = a1 >= 2
Mt = X[:, 0] + X[:, 1]; St = X[:, 1]; x9 = X[:, 2]
for nm, cols in (("M, S, x9", [Mt, St, x9]), ("M, S, T+, x9", [Mt, St, np.maximum(T3, 0), x9]), ("M, x9", [Mt, x9]), ("M - S, x9", [Mt - St, x9])):
    Xk = np.stack([c[k] for c in cols], 1)
    cf, res = np.linalg.lstsq(Xk, K.real[k], rcond=None)[:2]
    print(f"Re K ~ [{nm}]: coefficients " + " ".join(f"{v:7.2f}" for v in cf) + f"   uncentred R^2 = {1 - res[0] / (K.real[k] ** 2).sum():.4f}"
          f"   centred R^2 = {1 - res[0] / ((K.real[k] - K.real[k].mean()) ** 2).sum():.4f}")
