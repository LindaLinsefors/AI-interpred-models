"""Verifier V5, part K: off-by-one check of ages / indices in M3 + M5, and per-position cancellation.

At t = 9 the input of age a sits at logit index 9 - a.  With z/w4 = alpha(a1) * gap and phi(a1) = arg alpha (my pooled fit, v5_c),
the clock's contribution to logit p is  2 |alpha| gap w4 |R[p]| cos(phi(a1) + arg R[p]).   Where does each part peak?
Run:  python -B v5_k_index_check.py > v5_k_index_check.txt
"""
from v5_lib import *

r = RIGHT["clock"]
Rc, Rw, Ra = U[:, COMP] @ r[CI], U[:, WAVE] @ r[WI], Uo @ r
phi = np.radians([0.5, 36.4, 79.3, 123.2, 166.0, 208.7, 251.0, 294.0, 337.8])       # a1 = 1..9 (v5_c_clock_attacks.txt, C3)
mod = [1.014, 1.027, 0.993, 0.951, 0.921, 0.890, 0.861, 0.827, 0.803]               # pooled |alpha| (v5_b_m5m6.txt)
print("per-position cancellation |all 13| / |comparator part| at p = 0..9:", np.round(np.abs(Ra) / np.abs(Rc), 3))
print("phase(comparator part) - phase(wave part) at p = 0..9 (deg):", np.round(np.degrees(wrap(np.angle(Rc) - np.angle(Rw))), 1))
print("|wave part| / |comparator part| at p = 0..9:", np.round(np.abs(Rw) / np.abs(Rc), 3))
print("\nlogits per unit gap contributed by the clock mode at t = 9, for each age a1 of the max (index of the max = 9 - a1):")
print(" a1 | idx of max | comparator part: most negative index, value there | wave part: most positive index, value | all 13: value at idx of max, largest |value| over p=0..8 (index)")
for k, a1 in enumerate(range(1, 10)):
    pat = lambda R: 2 * mod[k] * W4 * np.abs(R) * np.cos(phi[k] + np.angle(R))
    pc, pw, pa = pat(Rc), pat(Rw), pat(Ra)
    im = 9 - a1
    print(f"  {a1} |     {im}      | idx {int(pc[:9].argmin())}, {pc[:9].min():7.1f} (at max: {pc[im]:7.1f}) | idx {int(pw[:9].argmax())}, {pw[:9].max():7.1f} (at max: {pw[im]:7.1f}) |"
          f" {pa[im]:6.2f}, {np.abs(pa[:9]).max():5.2f} (idx {int(np.abs(pa[:9]).argmax())});  all 13 at index 9: {pa[9]:6.2f}")
print("\nphase of the comparator part minus the 'tape' prediction 180 - 42.75 (8 - p) deg, p = 0..8:",
      np.round(np.degrees(wrap(np.angle(Rc[:9]) - np.radians(180 - np.degrees(OMEGA) * (8 - np.arange(9))))), 1))
