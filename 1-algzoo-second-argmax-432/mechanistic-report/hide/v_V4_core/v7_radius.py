"""V4 verifier, attack 3: is the radius of the threshold rotation proportional to M - S, or to M - c S with c < 1,
and how does that depend on when the runner-up arrived?  Also: the one-step map of z_c on real data.
z_c(t) = ell . theta(t+1).  Seeds 5012, 5013 (2 x 400k).  Selection as the lead: t <= 8, M_t > 0.3."""
from vsetup import *

out = Tee("v7_radius.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=220)

ev, vecs = np.linalg.eig(Bc.T)
k = int(np.argmax(np.angle(ev) * (np.abs(ev) > 0.5)))
lam = ev[k]
ell = vecs[:, k] / (vecs[:, k] @ (w[4] * b4))

Z, Ms, Ss, Rs, A1, A2, TT, NON, U4 = [], [], [], [], [], [], [], [], []
for seed in (5012, 5013):
    x = sample(400_000, seed)
    H, Zp = run(x, return_pre=True)
    th, _ = thresholds(H)
    rs = running_stats(x)
    Z.append(th[:, 1:] @ ell)
    Ms.append(rs["M1"]); Ss.append(rs["M2"]); A1.append(rs["a1"]); A2.append(rs["a2"])
    # R = max of the inputs BEFORE the current max arrived (what the record had to beat); -inf if the max is x_0
    Mshift = np.concatenate([np.full((x.shape[0], 1), -np.inf), rs["M1"][:, :-1]], axis=1)   # index p -> max(x_0..x_{p-1})
    Rs.append(np.take_along_axis(Mshift, rs["p1"], axis=1))
    TT.append(np.broadcast_to(np.arange(10), x.shape))
    NON.append((Zp[:, :, COMP] > 0).sum(2)); U4.append(Zp[:, :, 4] > 0)
Z, Ms, Ss, Rs, A1, A2, TT, NON, U4 = map(np.concatenate, (Z, Ms, Ss, Rs, A1, A2, TT, NON, U4))
Sp, Rp = np.maximum(Ss, 0.0), np.maximum(Rs, 0.0)
base = (TT <= 8) & (Ms > 0.3)


def cmed(z):
    return np.median(z.real) + 1j * np.median(z.imag)


def pol(c):
    return f"{abs(c):.3f}@{np.degrees(np.angle(c)) % 360:5.1f}"


def cfit(sel, cols):
    """complex least squares z ~ sum_k coef_k * col_k (real regressors). Returns coefs, uncentred complex R2,
    centred R2 of the M-normalised problem."""
    X = np.stack([c[sel] for c in cols], 1)
    zz = Z[sel]
    coef, *_ = np.linalg.lstsq(X.astype(complex), zz, rcond=None)
    res = zz - X @ coef
    r2u = 1 - (np.abs(res) ** 2).sum() / (np.abs(zz) ** 2).sum()
    zn = zz / Ms[sel]
    r2n = 1 - (np.abs(res / Ms[sel]) ** 2).sum() / (np.abs(zn - zn.mean()) ** 2).sum()
    return coef, r2u, r2n


def lead_fit(sel, phase):
    y_ = np.real(Z[sel] * np.exp(-1j * phase))
    X = np.stack([Ms[sel], Sp[sel]], 1)
    b, *_ = np.linalg.lstsq(X, y_, rcond=None)
    return b, 1 - ((y_ - X @ b) ** 2).sum() / (y_ ** 2).sum()


# ------------------------------------------------------------------ (A) complex regression per age
out("=== (A) per age a1: complex least squares z_c ~ alpha (M - S+) + beta S+   (beta = clock coordinate per unit M at a tie) ===")
out("  a1 :       n    alpha (mod@deg)   beta (mod@deg)   |beta|/|alpha|   Re(beta/alpha)   uncentred R2   centred R2 of z/M   "
    "lead's c = -b_S/b_M   1 - Re(beta e^-iphi)/Re(alpha e^-iphi)")
PH = {}
for a in range(0, 9):
    sel = base & (A1 == a) & (TT >= 1)
    PH[a] = np.angle(cmed(Z[sel] / Ms[sel]))
    (al, be), r2u, r2n = cfit(sel, [Ms - Sp, Sp])
    b, _ = lead_fit(sel, PH[a])
    rot = np.exp(-1j * PH[a])
    out(f"  {a}  : {int(sel.sum()):7d}   {pol(al)}      {pol(be)}      {abs(be) / abs(al):.3f}           {(be / al).real:+.3f}          "
        f"{r2u:.4f}         {r2n:.4f}              {-b[1] / b[0]:.3f}                 {1 - (be * rot).real / (al * rot).real:.3f}")

# ------------------------------------------------------------------ (B) model-free: near ties and far from ties
out("\n=== (B) model-free check: componentwise median of z_c/M and median |z_c|/M in bins of s = S+/M ===")
out("  if the radius were (M - S) the modulus would go to 0 for s -> 1")
bins = [(-0.01, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)]
out("  a1 :  " + "   ".join(f"s in ({lo:.2f},{hi:.2f}]" for lo, hi in bins))
sratio = Sp / Ms
for a in range(0, 9):
    row = []
    for lo, hi in bins:
        sel = base & (A1 == a) & (TT >= 1) & (sratio > lo) & (sratio <= hi)
        q = Z[sel] / Ms[sel]
        row.append(f"{pol(cmed(q))} |{np.median(np.abs(q)):.3f}|")
    out(f"  {a}  :  " + "  ".join(row))

# ------------------------------------------------------------------ (C) runner-up older / younger than the max
out("\n=== (C) lead's regression Re(z_c e^{-i phase_a}) ~ b_M M + b_S S+, split by when the runner-up arrived ===")
out("  older  : a2 > a1 (the runner-up is the value the record had to beat, R = S)")
out("  younger: a2 < a1 (the runner-up arrived after the max; the record beat R <= S)")
out("  a1 :  all: b_M  c=-b_S/b_M  R2u |  older: n  b_M  c  R2u |  younger: n  b_M  c  R2u | complex fit older alpha, beta | younger alpha, beta")
for a in range(0, 9):
    row = f"  {a}  :"
    comp_txt = ""
    for name, extra in (("all", np.ones_like(base)), ("older", A2 > A1), ("younger", (A2 < A1) & (A2 >= 0))):
        sel = base & (A1 == a) & (TT >= 1) & extra
        if sel.sum() < 2000:
            row += "        -- too few --       |"
            continue
        b, r2 = lead_fit(sel, PH[a])
        row += f" {int(sel.sum()):7d} {b[0]:.3f} {-b[1] / b[0]:.3f} {r2:.4f} |"
        if name != "all":
            (al, be), r2u, _ = cfit(sel, [Ms - Sp, Sp])
            comp_txt += f"  {name}: {pol(al)}, {pol(be)} (R2u {r2u:.3f})"
    out(row + comp_txt)

# ------------------------------------------------------------------ (D) younger runner-up: gap or record increment?
out("\n=== (D) younger runner-up (a2 < a1): z_c ~ alpha (M - S+) + beta S+ + gamma (S+ - R+);  R = what the record had to beat ===")
out("  'radius = current gap M - S'      predicts gamma = 0;   'radius = increment of the record M - R' predicts gamma = alpha")
out("  only states with S+ - R+ > 0.1 M are informative; reported for all younger states and for that subset")
out("  a1 :      n   alpha   beta   gamma   Re(gamma/alpha)  |gamma/alpha|   R2u(full)  R2u(gamma=0)  R2u(gamma=alpha)   [subset S+-R+ > 0.1M: n, Re(gamma/alpha)]")
for a in range(1, 9):
    sel = base & (A1 == a) & (A2 < A1) & (A2 >= 0)
    (al, be, ga), r2f, _ = cfit(sel, [Ms - Sp, Sp, Sp - Rp])
    _, r20, _ = cfit(sel, [Ms - Sp, Sp])
    _, r21, _ = cfit(sel, [Ms - Rp, Sp])
    sub = sel & (Sp - Rp > 0.1 * Ms)
    (al2, be2, ga2), _, _ = cfit(sub, [Ms - Sp, Sp, Sp - Rp])
    out(f"  {a}  : {int(sel.sum()):7d}  {pol(al)}  {pol(be)}  {pol(ga)}   {(ga / al).real:+.3f}          {abs(ga / al):.3f}        "
        f"{r2f:.4f}     {r20:.4f}        {r21:.4f}          [{int(sub.sum())}, {(ga2 / al2).real:+.3f}]")

out("\n  by (a1, a2), younger runner-up: Re(gamma/alpha) (0 = gap, 1 = record increment); rows a1 = 1..8, columns a2 = 0..7")
for a in range(1, 9):
    row = []
    for a2 in range(0, a):
        sel = base & (A1 == a) & (A2 == a2)
        if sel.sum() < 3000:
            row.append("   --  ")
            continue
        (al, be, ga), _, _ = cfit(sel, [Ms - Sp, Sp, Sp - Rp])
        row.append(f"{(ga / al).real:+.3f}")
    out(f"    a1 = {a}: " + "  ".join(row))

# ------------------------------------------------------------------ (E) c by (a1, a2)
out("\n=== (E) lead's c = -b_S/b_M per cell (a1, a2), phase = that cell's own median phase; rows a1 = 0..8, columns a2 = 0..9 ('--' < 3000 states) ===")
for a in range(0, 9):
    row = []
    for a2 in range(0, 10):
        sel = base & (A1 == a) & (A2 == a2) & (TT >= 1)
        if sel.sum() < 3000:
            row.append("  -- ")
            continue
        ph = np.angle(cmed(Z[sel] / Ms[sel]))
        b, _ = lead_fit(sel, ph)
        row.append(f"{-b[1] / b[0]:.2f}")
    out(f"    a1 = {a}: " + "  ".join(row))
out("  b_M per cell (same layout):")
for a in range(0, 9):
    row = []
    for a2 in range(0, 10):
        sel = base & (A1 == a) & (A2 == a2) & (TT >= 1)
        if sel.sum() < 3000:
            row.append("  -- ")
            continue
        ph = np.angle(cmed(Z[sel] / Ms[sel]))
        b, _ = lead_fit(sel, ph)
        row.append(f"{b[0]:.2f}")
    out(f"    a1 = {a}: " + "  ".join(row))
out("  cell's own median phase in degrees (same layout):")
for a in range(0, 9):
    row = []
    for a2 in range(0, 10):
        sel = base & (A1 == a) & (A2 == a2) & (TT >= 1)
        if sel.sum() < 3000:
            row.append("  -- ")
            continue
        row.append(f"{np.degrees(np.angle(cmed(Z[sel] / Ms[sel]))) % 360:5.1f}")
    out(f"    a1 = {a}: " + "  ".join(row))

# ------------------------------------------------------------------ (F) the one-step map on data
out("\n=== (F) one step of the real network: z_c(t+1) against z_c(t), for steps t+1 <= 8 without a new max at t+1, M_t > 0.3 ===")
z0, z1 = Z[:, :-1], Z[:, 1:]
ok = (A1[:, 1:] >= 1) & (Ms[:, :-1] > 0.3) & (TT[:, 1:] <= 8)
non1, u41 = NON[:, 1:], U4[:, 1:]
M0, S1 = Ms[:, :-1], Sp[:, 1:]
out(f"  Bc alone predicts z_c(t+1) = lambda z_c(t) + ..., lambda = {pol(lam)};  unit 2 adds (ell.b2) h2 ~ {pol((ell @ b2) * 17.3)} M per step")
for label, cond in (("all four comparators on at t+1", non1 == 4), ("3 on", non1 == 3), ("2 on", non1 == 2), ("1 on", non1 == 1)):
    sel = ok & cond & ~u41
    if sel.sum() < 2000:
        continue
    X = np.stack([z0[sel], M0[sel].astype(complex)], 1)
    coef, *_ = np.linalg.lstsq(X, z1[sel], rcond=None)
    res = z1[sel] - X @ coef
    r2 = 1 - (np.abs(res) ** 2).sum() / (np.abs(z1[sel]) ** 2).sum()
    big = sel & (np.abs(z0) > 0.2 * M0)
    q = z1[big] / z0[big]
    out(f"  {label:32s} n = {int(sel.sum()):8d}: fit z1 ~ L z0 + K M: L = {pol(coef[0])}, K = {pol(coef[1])}, uncentred R2 = {r2:.4f};  "
        f"raw ratio z1/z0 where |z0| > 0.2M: median modulus {np.median(np.abs(q)):.3f}, median angle {np.degrees(np.median(np.angle(q))):.1f} deg, "
        f"IQR of angle {np.degrees(np.percentile(np.angle(q), [25, 75]))}")
    if cond is not None and label.startswith("all four"):
        # fixed point (centre of rotation) implied by the fit, and the radius about it
        centre = coef[1] / (1 - coef[0])
        out(f"      implied centre of the rotation z* = K M / (1 - L) = {pol(centre)} M")
        # does the fit improve with S as an extra regressor?
        X3 = np.stack([z0[sel], M0[sel].astype(complex), S1[sel].astype(complex)], 1)
        c3, *_ = np.linalg.lstsq(X3, z1[sel], rcond=None)
        r23 = 1 - (np.abs(z1[sel] - X3 @ c3) ** 2).sum() / (np.abs(z1[sel]) ** 2).sum()
        out(f"      with S+ as extra regressor: L = {pol(c3[0])}, K_M = {pol(c3[1])}, K_S = {pol(c3[2])}, R2 = {r23:.4f}")
        # by age of the max at t (how regular is the rotation?)
        a_prev = A1[:, :-1]
        for a in range(0, 8):
            sa = sel & (a_prev == a)
            Xa = np.stack([z0[sa], M0[sa].astype(complex)], 1)
            ca, *_ = np.linalg.lstsq(Xa, z1[sa], rcond=None)
            out(f"      age of the max at t = {a}: n = {int(sa.sum()):7d}  L = {pol(ca[0])}  K = {pol(ca[1])}")

# ------------------------------------------------------------------ (G) radius about the fitted centre
out("\n=== (G) lead's regression repeated on z_c - z* M (z* = centre from (F), all-four-on fit): b_M, c = -b_S/b_M, uncentred R2 ===")
sel = ok & (non1 == 4) & ~u41
X = np.stack([z0[sel], M0[sel].astype(complex)], 1)
coef, *_ = np.linalg.lstsq(X, z1[sel], rcond=None)
centre = coef[1] / (1 - coef[0])
for a in range(0, 9):
    s_ = base & (A1 == a) & (TT >= 1)
    zz = Z[s_] - centre * Ms[s_]
    ph = np.angle(cmed(zz / Ms[s_]))
    y_ = np.real(zz * np.exp(-1j * ph))
    Xr = np.stack([Ms[s_], Sp[s_]], 1)
    b, *_ = np.linalg.lstsq(Xr, y_, rcond=None)
    r2 = 1 - ((y_ - Xr @ b) ** 2).sum() / (y_ ** 2).sum()
    out(f"  a1 = {a}: phase {np.degrees(ph) % 360:6.1f}   b_M {b[0]:.3f}   c {-b[1] / b[0]:.3f}   R2u {r2:.4f}   median |z_c - z* M| / (M - S+) = "
        f"{np.median(np.abs(zz) / np.maximum(Ms[s_] - Sp[s_], 1e-9)):.3f}   median |z_c| / (M - S+) = {np.median(np.abs(Z[s_]) / np.maximum(Ms[s_] - Sp[s_], 1e-9)):.3f}")
