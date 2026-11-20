"""s03. Re-verify the lead's facts that arm F builds on (F5, F6, F7, F8) with a fresh seed (1010).
Saves phi(a), rho(a) of the clock for later scripts."""
from fcommon import *

x = sample(400_000, 1010)
H = run(x)
st = running_stats(x)
m = coords(H)
M, S, a1 = st["M1"], st["M2"], st["a1"]
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)

print("F5: level coordinate ~ c * 10.157 * M within age cells (M>0), no intercept")
for a in range(0, 10):
    s = (a1 == a) & (M > 0)
    b, r2 = ols(m["level"][s] / w[4], [M[s]])
    b2, r22 = ols(m["level"][s] / w[4], [M[s], Sp[s]])
    print(f"   age {a}: c = {b[0]:.3f}  R2 = {r2:.4f}   | with S+: {np.round(b2, 3)}  R2 = {r22:.4f}   n = {s.sum()}")

print("\nF6: clock z/10.157 ~ alpha(a) (M - S+) + beta(a) S+   (complex LSQ, M>0)")
phi, rho, beta = np.zeros(10), np.zeros(10), np.zeros(10, complex)
for a in range(1, 10):
    s = (a1 == a) & (M > 0)
    b, r2 = cols_ols(m["clock"][s] / w[4], [M[s] - Sp[s], Sp[s]])
    phi[a], rho[a], beta[a] = np.degrees(np.angle(b[0])) % 360, abs(b[0]), b[1]
    print(f"   age {a}: alpha = {fmtc(b[0])} (phi = {phi[a]:6.1f})   beta = {fmtc(b[1])}   R2 = {r2:.4f}")
phi[1] = phi[1] if phi[1] < 180 else phi[1] - 360
print("   phi(a) for a=1..9:", np.round(phi[1:], 1), "  lead: 0, 37, 80, 123, 166, 209, 252, 294, 338")
print("   rho(a) for a=1..9:", np.round(rho[1:], 3))
np.savez(HIDE / "r1_F_record_event" / "s03_clock.npz", phi=phi, rho=rho, beta=beta)

print("\nother modes: m_k/10.157 ~ alpha_k(a) (M - S+) + beta_k(a) S+ by age of the max (M>0): alpha | beta | R2")
for n in NAMES:
    if n in ("level", "clock"):
        continue
    print(f"  mode {n}")
    for a in range(1, 10):
        s = (a1 == a) & (M > 0)
        y = m[n][s] / w[4]
        b, r2 = cols_ols(y, [M[s] - Sp[s], Sp[s]])
        if IS_COMPLEX[n]:
            print(f"     age {a}: {fmtc(b[0])} | {fmtc(b[1])} | {r2:.3f}")
        else:
            print(f"     age {a}: {b[0].real:7.3f} | {b[1].real:7.3f} | {r2:.3f}")

print("\nF7: readout of level and clock while all 13 ordinary units are on")
print("   U[:,ORD].RIGHT[level] =", np.round((Uo @ RIGHT["level"]).real, 2))
Rc, Rw = Uo[:, CI] @ RIGHT["clock"][CI], Uo[:, WI] @ RIGHT["clock"][WI]
print("   clock via comparators (2|R|@deg):", " ".join(f"{2*abs(v):5.2f}@{np.degrees(np.angle(v)):4.0f}" for v in Rc))
print("   clock via wave units  (2|R|@deg):", " ".join(f"{2*abs(v):5.2f}@{np.degrees(np.angle(v)):4.0f}" for v in Rw))
print("   clock total           (2|R|@deg):", " ".join(f"{2*abs(v):5.2f}@{np.degrees(np.angle(v)):4.0f}" for v in Rc + Rw))
print(f"   |total| / (|comp| + |wave|) = {np.linalg.norm(Rc + Rw) / (np.linalg.norm(Rc) + np.linalg.norm(Rw)):.3f};"
      f"  |total|/|wave| = {np.linalg.norm(Rc + Rw) / np.linalg.norm(Rw):.3f}")
for n in NAMES:
    Rc, Rw = Uo[:, CI] @ RIGHT[n][CI], Uo[:, WI] @ RIGHT[n][WI]
    print(f"   mode {n:6s}: |comp part| = {np.linalg.norm(Rc):6.2f}  |wave part| = {np.linalg.norm(Rw):6.2f}  |total| = {np.linalg.norm(Rc + Rw):6.2f}"
          f"   fraction of LEFT norm^2 on comparators = {np.sum(abs(LEFT[n][CI])**2) / np.sum(abs(LEFT[n])**2):.3f}")

print("\nF8: free-flow logit kernels of the pulses at the pulse's own reference position")
for a in range(0, 9):
    k4 = U[:, 4] if a == 0 else Uo @ np.linalg.matrix_power(A, a - 1) @ W[ORD, 4]
    k5 = U[:, 5] if a == 0 else Uo @ np.linalg.matrix_power(A, a - 1) @ W[ORD, 5]
    k9 = U[:, 9] if a == 0 else Uo @ np.linalg.matrix_power(A, a - 1) @ W[ORD, 9]
    print(f"   age {a}: unit4 at own pos {9-a}: {k4[9-a]:6.2f} (others max {np.delete(k4, 9-a).max():5.2f} min {np.delete(k4, 9-a).min():5.2f}) |"
          f" unit5 at pos {8-a}: {k5[8-a]:6.2f} | unit9 at pos {8-a}: {k9[8-a]:6.2f}, at pos {7-a}: {k9[7-a] if a < 8 else np.nan:6.2f}")
