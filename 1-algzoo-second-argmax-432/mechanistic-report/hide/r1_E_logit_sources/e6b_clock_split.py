"""E6b. The clock contribution to the logits, split into the part read through the wave units and the part read through
the comparators.  [EXACT readout vectors + MEASURED on seed 1110, n = 400k N(0,1)]
C_wave[p] = 2 Re(z R_w[p]),  C_comp[p] = 2 Re(z R_c[p]),  R_w = U[:, WAVE] RIGHT_clock[WAVE], R_c = U[:, COMP] RIGHT_clock[COMP]."""
from src import *

Rw = Uo[:, WI] @ RIGHT["clock"][WI]; Rc = Uo[:, CI] @ RIGHT["clock"][CI]
print("[EXACT] clock readout per unit |z| (2|R| @ phase deg), positions 0..9")
for nm, R_ in (("via wave units ", Rw), ("via comparators", Rc), ("total          ", Rw + Rc)):
    print(f"  {nm}: " + " ".join(f"{2 * abs(v):5.2f}@{np.degrees(np.angle(v)) % 360:4.0f}" for v in R_))
ph = np.unwrap(np.angle(Rw[:8]))
print(f"  phase slope of the wave-unit part over positions 0..7: {np.degrees(np.polyfit(np.arange(8), ph, 1)[0]):.1f} deg/position (clock rotation {np.degrees(np.angle(LAM['clock'])):.2f} deg/step)")
print(f"  |R_w + R_c| / (|R_w| + |R_c|) = {np.linalg.norm(Rw + Rc) / (np.linalg.norm(Rw) + np.linalg.norm(Rc)):.4f}")

x = sample(400_000, 1110)
H = run(x); n = len(x); r = np.arange(n)
z = coords(H[:, 9])["clock"]
order = np.argsort(-x, axis=1); p1, p2 = order[:, 0], order[:, 1]
M, S = x[r, p1], np.maximum(x[r, p2], 0)
a1 = 9 - p1
Cw = 2 * np.real(np.multiply.outer(z, Rw)); Cc = 2 * np.real(np.multiply.outer(z, Rc))
print("\n[MEASURED] by age of the max a (position of the max = 9 - a): clock phase, and where the two parts of the clock contribution peak")
print("  (complex least squares per age:  z ~ 10.157 [ c1 (M - S+) + c2 S+ ];  rho = |c1|, phi = arg c1)")
print("  a   n      rho = |c1|   phi = arg c1 (deg)   |c2|   R^2   P(argmax_p C_wave = max's position)   P(argmin_p C_comp = max's position)   mean C_wave at max pos   mean C_comp at max pos   mean total clock at max pos")
for a in range(0, 10):
    mk = a1 == a
    Xr = 10.157 * np.stack([(M - S)[mk], S[mk]], 1).astype(complex)
    c, *_ = np.linalg.lstsq(Xr, z[mk], rcond=None)
    r2 = 1 - (np.abs(z[mk] - Xr @ c) ** 2).sum() / (np.abs(z[mk] - z[mk].mean()) ** 2).sum()
    print(f"  {a}  {mk.sum():6d}   {abs(c[0]):8.3f}   {np.degrees(np.angle(c[0])) % 360:12.1f}   {abs(c[1]):6.3f}  {r2:.3f}   {np.mean(Cw[mk].argmax(1) == p1[mk]):28.3f}   {np.mean(Cc[mk].argmin(1) == p1[mk]):28.3f}"
          f"   {Cw[r, p1][mk].mean():18.2f}   {Cc[r, p1][mk].mean():18.2f}   {(Cw + Cc)[r, p1][mk].mean():18.2f}")
mk = (a1 >= 2)
print(f"\npooled over a >= 2: P(argmax C_wave = max's position) = {np.mean(Cw[mk].argmax(1) == p1[mk]):.3f}; P(argmin C_comp = max's position) = {np.mean(Cc[mk].argmin(1) == p1[mk]):.3f}; "
      f"rms of C_wave centred over positions 0..8 = {np.sqrt(((Cw[mk][:, :9] - Cw[mk][:, :9].mean(1, keepdims=True)) ** 2).mean()):.2f}, of C_comp = {np.sqrt(((Cc[mk][:, :9] - Cc[mk][:, :9].mean(1, keepdims=True)) ** 2).mean()):.2f}, "
      f"of their sum = {np.sqrt((((Cw + Cc)[mk][:, :9] - (Cw + Cc)[mk][:, :9].mean(1, keepdims=True)) ** 2).mean()):.2f}")
