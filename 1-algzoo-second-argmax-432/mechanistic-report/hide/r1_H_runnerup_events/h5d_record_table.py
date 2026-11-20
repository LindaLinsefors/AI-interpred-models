"""H5: the S record (run with S minus run without S) in every mode at the final step, as a function of a2 = 9 - p2,
for fixed age of the max at the event (a_ev = p2 - p1).  Units: K * S.  Also the logit bump at p2 per unit S."""
from hcommon import *
from h5_controlled import seq, Mv, mode_logits

for b, sr in ((-0.3, 0.5), (-1.0, 0.5), (-0.3, 0.2), (-0.3, 0.8)):
    S = sr * Mv
    for a_ev in (1, 3, 5):
        print(f"\nbackground {b}, S/M = {sr}, age of the max when S arrives a_ev = {a_ev}:  record / (K S) by a2")
        print("  a2 | " + " | ".join(f"{n:^13s}" for n in NAMES) + " | dlogit[p2]/S | margin of p2 over best other / S")
        for a2 in range(0, 9 - a_ev + 1):
            p2 = 9 - a2; p1 = p2 - a_ev
            h1 = run(seq(p1, p2, S, b))[0, -1]; h0 = run(seq(p1, p2, b, b))[0, -1]
            m1, m0 = coords(h1), coords(h0)
            row = ""
            for n in NAMES:
                v = (m1[n] - m0[n]) / (K * S)
                row += (f" {abs(v):5.2f}@{np.degrees(np.angle(v)):6.1f} |" if IS_COMPLEX[n] else f" {np.real(v):12.2f} |")
            l1, l0 = U @ h1, U @ h0
            print(f"  {a2}  |{row} {(l1[p2] - l0[p2]) / S:8.1f}     | {(l1[p2] - np.max(np.delete(l1, p2))) / S:8.1f}")
        if a_ev == 1:
            print("     free-decay prediction |lam|^(a2-2) relative to a2 = 2: " + "  ".join(f"{n}: " + ",".join(f"{abs(LAM[n]) ** k:.2f}" for k in (2, 4, 6)) for n in ("runner", "clock", "h2", "alt", "h3", "short")) + "   (at a2 = 4, 6, 8)")
