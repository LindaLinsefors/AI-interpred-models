"""Sanity checks of the helpers and of my intervention code (seed 5000)."""
from v3lib import *

out = Tee("i0_sanity.txt")
x = sample(200_000, 5000)
H = run(x)
acc = (logits(H).argmax(1) == second_argmax(x)).mean()
out(f"accuracy N(0,1), seed 5000, n=200000: {acc:.4f}")

# run_iv with identity = run
h9 = run_iv(x, np.full(len(x), 4), lambda h, rows: h)
out("run_iv identity max |diff| vs run:", np.abs(h9 - H[:, -1]).max())

# modes: eigen-relations and biorthogonality
out("OMEGA deg:", np.degrees(OMEGA), " |lam_clock|:", abs(LAM["clock"]))
for n in NAMES:
    out(f"{n:7s} lam = {abs(LAM[n]):.4f} @ {np.degrees(np.angle(LAM[n])):7.2f}   "
        f"L.R = {LEFT[n] @ RIGHT[n]:.6f}   L.conj(R) = {LEFT[n] @ np.conj(RIGHT[n]):.2e}   "
        f"L.kick4 = {LEFT[n] @ W[ORD, 4]:.6f}   |A R - lam R| = {np.abs(A @ RIGHT[n] - LAM[n] * RIGHT[n]).max():.2e}"
        f"   |L A - lam L| = {np.abs(LEFT[n] @ A - LAM[n] * LEFT[n]).max():.2e}")

# the clock intervention sets the clock coordinate to z' and leaves the other coordinates alone
h = H[:5000, 6]
for f, k in [(0, 0), (0.5, 0), (1, 1), (1, -2), (2, 0)]:
    h2 = mode_fn("clock", f=f, k=k)(h, None)
    c0, c1 = coords(h), coords(h2)
    err_clock = np.abs(c1["clock"] - f * np.exp(1j * k * OMEGA) * c0["clock"]).max()
    err_other = max(np.abs(c1[n] - c0[n]).max() for n in NAMES if n != "clock")
    err_pulse = np.abs(h2[:, PULSE] - h[:, PULSE]).max()
    out(f"f={f} k={k}: |z' - target| = {err_clock:.2e}, other coords changed by {err_other:.2e}, pulse units by {err_pulse:.2e}")
out("fraction of squared norm of LEFT[clock] on comparators:",
    (np.abs(LEFT["clock"][np.isin(ORD_A, COMP)]) ** 2).sum() / (np.abs(LEFT["clock"]) ** 2).sum())
