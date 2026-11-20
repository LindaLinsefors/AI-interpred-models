"""Verifier V5, part E: which statements of M1-M4 survive a change of per-unit scale?

The network is unchanged by h_i -> c_i h_i (w_i -> c_i w_i, W_ij -> c_i W_ij / c_j, U_pi -> U_pi / c_i).
Right eigenvectors scale as v_i -> c_i v_i, left vectors as L_i -> L_i / c_i, readout columns as U[:, i] -> U[:, i] / c_i.
Second natural normalisations used here:
   RMS : every unit divided by its root-mean-square activation over all states t = 0..9 of N(0,1) sequences
   STD : every unit divided by its standard deviation over the same states
   READ: every unit multiplied by the norm of its readout column (all readout columns get unit norm)
Also: scale-invariant replacements for M2 measured on data (share of each mode coordinate that is carried by the comparators).
Seed 5030 (200k sequences).   Run:  python -B v5_e_normalisation.py > v5_e_normalisation.txt
"""
from v5_lib import *

x, H, Z, st, m = batch(5030)
Hf = H.reshape(-1, N)
sig_rms = np.sqrt((Hf ** 2).mean(0))
sig_std = Hf.std(0)
unorm = np.linalg.norm(U, axis=0)
print("seed 5030, 200k sequences, all states t = 0..9")
print("unit            :", "  ".join(f"{u:6d}" for u in range(N)))
print("rms activation  :", "  ".join(f"{v:6.2f}" for v in sig_rms))
print("std activation  :", "  ".join(f"{v:6.2f}" for v in sig_std))
print("|U[:, i]|       :", "  ".join(f"{v:6.2f}" for v in unorm))
print("P(h_i > 0)      :", "  ".join(f"{v:6.2f}" for v in (Hf > 0).mean(0)))
# c_i such that h' = c h
SC = {"RAW": np.ones(N), "RMS": 1 / sig_rms, "STD": 1 / sig_std, "READ": unorm}
cosv = lambda a, b: float(abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b))

print("=" * 120)
print("M1 under rescaling")
i2 = ORD.index(2)
for nm, c in SC.items():
    co = c[ORD]
    v = RIGHT["level"].real * co
    print(f"   {nm:4s}: level eigenvector / unit 2: {np.round(v / v[i2], 2)}   one sign: {bool((v > 0).all() or (v < 0).all())}")
for nm, c in SC.items():
    co = c[ORD]
    print(f"   {nm:4s}: |cos(w[ORD], echo right eigenvector)| = {cosv(w[ORD] * co, RIGHT['echo'].real * co):.4f}")
print("   scale-invariant content of 'the input lands on the echo mode': coordinates that x_t = 1 adds on an exact-linear step, in units of w4 (one unit = the kick of a record increment 1):")
print("      " + ", ".join(f"{n} {abs(LEFT[n] @ w[ORD]) / W4:.4f}" for n in NAMES))

print("=" * 120)
print("M2 under rescaling: fraction of each left vector's squared norm on the comparators (and for the level on unit 2)")
print("   mode      " + "   ".join(f"{nm:>6s}" for nm in SC))
for n in NAMES:
    out = f"   {n:7s}  "
    for nm, c in SC.items():
        L2 = np.abs(LEFT[n] / c[ORD]) ** 2
        out += f"   {L2[CI].sum() / L2.sum():6.3f}"
    print(out)
out = "   level, unit 2:"
for nm, c in SC.items():
    L2 = np.abs(LEFT["level"] / c[ORD]) ** 2
    out += f" {nm} {L2[i2] / L2.sum():.3f}"
print(out)
print("   level left vector in RMS units (entry x rms activation), by unit:", dict(zip(ORD, np.round(LEFT["level"].real * sig_rms[ORD], 3))))
print("   clock left vector |entry| x rms activation, by unit:", dict(zip(ORD, np.round(np.abs(LEFT["clock"]) * sig_rms[ORD], 3))))

print("\n   scale-invariant replacement, measured on data: coordinate m = sum_i L_i h_i; m_G = sum over group G only.")
print("   'share' = 1 - sum|m - m_G|^2 / sum|m|^2   (1 = the group alone gives the coordinate; <= 0 = it does not)")
print("   mode       comparators only   wave units only   unit 2 only    | same, states with a1 >= 2 and S > 0.05")
Ho = H[..., ORD]
sel = (st["a1"] >= 2) & (st["S"] > 0.05)
def share(mfull, mpart, s=None):
    if s is not None:
        mfull, mpart = mfull[s], mpart[s]
    return 1 - (np.abs(mfull - mpart) ** 2).sum() / (np.abs(mfull) ** 2).sum()
for n in NAMES:
    L = LEFT[n]
    full = Ho @ L
    mc = Ho[..., CI] @ L[CI]
    mw = Ho[..., WI] @ L[WI]
    m2 = Ho[..., i2] * L[i2]
    print(f"   {n:7s}   {share(full, mc):10.3f}        {share(full, mw):10.3f}       {share(full, m2):10.3f}     |  {share(full, mc, sel):8.3f}  {share(full, mw, sel):8.3f}  {share(full, m2, sel):8.3f}")
print("\n   what happens to the clock coordinate when the comparators are zeroed (the wipe a record produces)? |z_wave| / |z| on data:")
L = LEFT["clock"]
z = Ho @ L
zw = Ho[..., WI] @ L[WI]
for A1 in range(1, 10):
    s = (st["a1"] == A1) & (st["S"] > 0.05) & (st["M"] - st["S"] > 0.2)
    rr = np.abs(zw[s]) / np.abs(z[s])
    print(f"      a1={A1}: median |z_wave|/|z| = {np.median(rr):.3f}  (90th pct {np.percentile(rr, 90):.3f});  rms |z_wave|/w4 = {np.sqrt(np.mean(np.abs(zw[s]) ** 2)) / W4:.3f}  vs rms |z|/w4 = {np.sqrt(np.mean(np.abs(z[s]) ** 2)) / W4:.3f}")
s = st["a1"] == 0
print(f"      a1=0 (the record step itself; comparators are off): rms |z|/w4 = {np.sqrt(np.mean(np.abs(z[s]) ** 2)) / W4:.3f},"
      f" P(all four comparators exactly 0) = {np.mean((H[s][:, COMP] == 0).all(1)):.4f}")

print("=" * 120)
print("M3 / M4 under rescaling: products U[:, i] v_i are invariant, so every entry of M3 and the cancellation ratio are invariant (checked numerically in part A).")
print("M4 'per unit coordinate' depends on the gauge only through the scale of unit 4 (LEFT @ W[ORD, 4] = 1). Invariant form: size x w4 (logits per unit of record increment):")
for n in NAMES:
    y = Uo[:9] @ RIGHT[n]
    y = y - y.mean()
    if IS_COMPLEX[n]:
        sv = np.linalg.svd(np.stack([y.real, y.imag], 1), compute_uv=False)
        print(f"   {n:7s} Re {np.linalg.norm(y.real) * W4:7.2f}  Im {np.linalg.norm(y.imag) * W4:7.2f}   as the state is 2 Re(z v): amplitude between {2 * sv[1] * W4:7.2f} and {2 * sv[0] * W4:7.2f} per unit |z|/w4")
    else:
        print(f"   {n:7s}    {np.linalg.norm(y.real) * W4:7.2f}")
