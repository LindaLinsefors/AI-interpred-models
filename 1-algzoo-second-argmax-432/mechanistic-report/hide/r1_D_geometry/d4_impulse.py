"""D4. Impulse and two-pulse inputs: plane geometry (PCA of h_2..h_9 or of the steps after the last pulse) and mode decomposition.
python -B d4_impulse.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

np.set_printoptions(precision=2, suppress=True, linewidth=220)


def plane_geometry(Ho, label):
    """Ho: (n, 13) states. Fit a 2-plane through them (PCA), report distance from origin, radius, off-plane rms, rotation."""
    mu = Ho.mean(0)
    Uc, s, Vt = np.linalg.svd(Ho - mu, full_matrices=False)
    V = Vt[:2]                                         # plane directions
    n_perp = mu - V.T @ (V @ mu)                       # perpendicular from the origin to the plane
    P = (Ho - mu) @ V.T                                # in-plane coordinates about the mean
    off = (Ho - mu) - P @ V
    # circle fit (algebraic, Kasa) in the plane for the centre
    Amat = np.column_stack([2 * P, np.ones(len(P))])
    sol, *_ = np.linalg.lstsq(Amat, (P ** 2).sum(1), rcond=None)
    c2 = sol[:2]
    rad = np.sqrt(sol[2] + c2 @ c2)
    ang = np.degrees(np.unwrap(np.arctan2(P[:, 1] - c2[1], P[:, 0] - c2[0])))
    r_each = np.linalg.norm(P - c2, axis=1)
    centre13 = mu + c2 @ V
    print(f"[{label}] var share PC0, PC1, PC2 = {(s ** 2 / (s ** 2).sum())[:3].round(4)}")
    print(f"   |n| (origin to plane) = {np.linalg.norm(n_perp):.2f};  rms in-plane radius about the mean = {np.sqrt((P ** 2).sum(1).mean()):.2f};"
          f"  off-plane rms = {np.sqrt((off ** 2).sum(1).mean()):.2f}")
    print(f"   circle fit: radius = {rad:.2f}, |fitted centre - mean| = {np.linalg.norm(c2):.2f}, radii per step = {r_each.round(1)}")
    print(f"   rotation per step (deg) = {np.abs(np.diff(ang)).round(1)}  mean = {np.abs(np.diff(ang)).mean():.2f}")
    return dict(mu=mu, V=V, n=n_perp, centre=centre13, rad=rad)


def mode_table(H, steps):
    m = coords(H)
    print("   mode coordinates / 10.157  (|z| and phase in deg for the rotating pairs)")
    print("     t  " + "".join(f"{n:>16s}" for n in NAMES))
    for t in steps:
        row = ""
        for n in NAMES:
            v = m[n][t] / 10.157
            row += f"{abs(v):9.3f}@{np.degrees(np.angle(v)):6.1f}" if IS_COMPLEX[n] else f"{v.real:16.3f}"
        print(f"    {t:2d}  {row}")
    return m


def mode_split(H, steps, geo):
    """How much of the centre / of the rotating part does each mode carry?"""
    Ho = H[steps][:, ORD]
    m = coords(H)
    parts = {}
    for n in NAMES:
        mm = {k: (m[k][steps] if k == n else np.zeros_like(m[k][steps])) for k in NAMES}
        parts[n] = reconstruct(mm)                     # (len(steps), 13) contribution of mode n
    mu = Ho.mean(0)
    print("   share of the time-mean state (centre) and of the variance about it, by mode:")
    print("     mode      mean.mu/|mu|^2    |mean|     var share (sum_t |dev|^2)   covariance share")
    dev_tot = Ho - mu
    for n in NAMES:
        pm = parts[n].mean(0)
        dev = parts[n] - pm
        print(f"     {n:7s}   {pm @ mu / (mu @ mu):8.4f}      {np.linalg.norm(pm):7.2f}     {(dev ** 2).sum() / (dev_tot ** 2).sum():8.4f}"
              f"                  {(dev * dev_tot).sum() / (dev_tot ** 2).sum():8.4f}")
    # the perpendicular n: which modes make it up (n is linear in the mode means)
    nvec = geo["n"]
    print("     perpendicular n: contribution of each mode's mean, n.pm_perp/|n|^2 :",
          {k: round(float((parts[k].mean(0) - geo["V"].T @ (geo["V"] @ parts[k].mean(0))) @ nvec / (nvec @ nvec)), 3) for k in NAMES})


def show(xv, label, steps, show_states=True):
    xv = np.asarray(xv, float)
    H, Z = run(xv, return_pre=True)
    H, Z = H[0], Z[0]
    print("\n" + "=" * 110 + f"\n{label}: x = {xv}")
    if show_states:
        print("   h_t (columns: units 0..15)")
        for t in range(10):
            print(f"    t={t} " + " ".join(f"{v:6.2f}" for v in H[t]))
        print("   OFF units per step:", {t: [u for u in range(16) if Z[t, u] <= 0] for t in range(10)})
    geo = plane_geometry(H[steps][:, ORD], label + f", steps {steps[0]}..{steps[-1]}")
    lev = RIGHT["level"].real
    ck = np.stack([RIGHT["clock"].real, RIGHT["clock"].imag])
    Q, _ = np.linalg.qr(ck.T)
    sv = np.linalg.svd(Q.T @ geo["V"].T, compute_uv=False)
    print(f"   principal angles between the PCA plane and the clock plane (deg) = {np.degrees(np.arccos(np.clip(sv, -1, 1))).round(1)}")
    print(f"   cos(mean state, level eigvec) = {geo['mu'] @ lev / np.linalg.norm(geo['mu']) / np.linalg.norm(lev):.4f};"
          f"  cos(n, level eigvec) = {geo['n'] @ lev / np.linalg.norm(geo['n']) / np.linalg.norm(lev):.4f};"
          f"  |mean| = {np.linalg.norm(geo['mu']):.2f}")
    mode_table(H, range(10))
    mode_split(H, steps, geo)
    return H, Z, geo


if __name__ == "__main__":
    e = lambda **kw: [kw.get(f"p{i}", kw.get("bg", 0.0)) for i in range(10)]
    S29 = list(range(2, 10))
    show(e(p0=1), "impulse", S29)
    show(e(p0=1, bg=-0.5), "impulse, background -0.5", S29)
    show(e(p0=1, p4=2), "second larger input at 4 (bg 0)", list(range(6, 10)))
    show(e(p0=1, p4=2, bg=-0.5), "second larger input at 4 (bg -0.5)", list(range(6, 10)))
    show(e(p0=2, p4=1), "second smaller input at 4 (bg 0)", list(range(6, 10)))
    show(e(p0=2, p4=1, bg=-0.5), "second smaller input at 4 (bg -0.5)", list(range(6, 10)))
