"""H6. Is the runner-up's age held by freely rotating records in the visible modes?

Part 1 (controlled, exact).  M at p1, S at p2 = p1 + a_ev, background b.  S record = run with S minus run with x[p2] = b.
dm_k(a2) = record in mode k at the final step, a2 = 9 - p2 (so p1 = 9 - a2 - a_ev; a_ev = age of the max when S arrives).
Exact split of dm_k(a2) by the step at which a source enters the ordinary units (offset j = step - p2):
   j = 0: input x_p2 and clip corrections at p2;   j = 1: pulses fired at p2, clips at p2+1;
   j = 2: pulses fired at p2+1 (unit 5), clips at p2+2;   j >= 3: everything later.
Fits over a2 >= a2min:  (A) c * lam_k^a2 with the mode's own eigenvalue;  (B) c * mu^a2 with free complex mu.

Part 2 (N(0,1) samples, final step).  Sequences whose runner-up arrived after the max.  For each cell (a_ev, a2):
   m_k(9)/K ~ alpha M + beta S + gamma T+   (free per-age vectors),   versus the constraint  beta = c_k(a_ev) * lam_k^a2.
"""
from hcommon import *
from h5_controlled import seq, Mv
from scipy.optimize import least_squares

VIS = ["runner", "clock", "h2", "alt", "h3", "short"]
relu = lambda v: np.maximum(v, 0)
WOP = W[np.ix_(ORD, PULSE)]


def record(p1, p2, S, b):
    """dm (mode -> complex) at the final step, and its exact split by entry offset j = 0, 1, 2, 3+ ."""
    (H1, Z1), (H0, Z0) = run(seq(p1, p2, S, b), return_pre=True), run(seq(p1, p2, b, b), return_pre=True)
    H1, Z1, H0, Z0 = H1[0], Z1[0], H0[0], Z0[0]
    m1, m0 = coords(H1[-1]), coords(H0[-1])
    dC = relu(-Z1[:, ORD]) - relu(-Z0[:, ORD])
    dP = (H1 - H0)[:, PULSE]
    dm, parts = {}, {}
    for n in NAMES:
        L, lam = LEFT[n], LAM[n]
        dm[n] = complex(m1[n] - m0[n])
        pj = np.zeros(4, complex)
        for s in range(p2, 10):
            src = dC[s] + (w[ORD] * (S - b) if s == p2 else 0) + (WOP @ dP[s - 1] if s > p2 else 0)
            pj[min(s - p2, 3)] += lam ** (9 - s) * (L @ src)
        parts[n] = pj
    return dm, parts


def fits(a2a, dms, lam, cplx):
    basis = lam ** a2a
    c = (np.conj(basis) @ dms) / (np.conj(basis) @ basis)
    if not cplx: c = c.real
    R2A = 1 - np.sum(np.abs(dms - c * basis) ** 2) / np.sum(np.abs(dms) ** 2)
    def resid(p):
        r = dms - (p[2] + 1j * p[3]) * (p[0] + 1j * p[1]) ** a2a
        return np.concatenate([r.real, r.imag])
    sol = least_squares(resid, [lam.real, lam.imag, np.real(c), np.imag(c)])
    mu = sol.x[0] + 1j * sol.x[1]
    return c, R2A, mu, 1 - np.sum(sol.fun ** 2) / np.sum(np.abs(dms) ** 2)


if __name__ == "__main__":
    print("=== Part 1: controlled inputs ===")
    summary = {n: [] for n in VIS}
    for b in (-1.0, -0.3):
        for sr in (0.2, 0.5, 0.8):
            S = sr * Mv
            print(f"\n--- background {b}, S/M = {sr} (M = {Mv}); record per unit of K*S ---")
            for n in VIS:
                cplx, lam = IS_COMPLEX[n], LAM[n]
                print(f"  mode {n}: lam = {abs(lam):.3f} @ {np.degrees(np.angle(lam)):.1f} deg")
                print("    a_ev | fit A (own lam) R2, a2>=2 | a2>=3 | fit B: |mu| @ angle (R2), a2>=2 | share of sum|dm|^2 over a2>=3 not explained by sources entering at offsets <=1 / <=2 | record amplitude at a2=2 and a2=3")
                for a_ev in range(1, 6):
                    a2s = np.arange(2, 9 - a_ev + 1)
                    dms, pj = [], []
                    for a2 in a2s:
                        dm, parts = record(9 - a2 - a_ev, 9 - a2, S, b)
                        dms.append(dm[n] / (K * S)); pj.append(parts[n] / (K * S))
                    dms, pj = np.array(dms), np.array(pj)
                    c, R2A, mu, R2B = fits(a2s, dms, lam, cplx)
                    k3 = a2s >= 3
                    c3, R2A3, _, _ = fits(a2s[k3], dms[k3], lam, cplx) if k3.sum() >= 2 else (np.nan,) * 4
                    un1 = np.sum(np.abs(dms[k3] - pj[k3][:, :2].sum(1)) ** 2) / np.sum(np.abs(dms[k3]) ** 2)
                    un2 = np.sum(np.abs(dms[k3] - pj[k3][:, :3].sum(1)) ** 2) / np.sum(np.abs(dms[k3]) ** 2)
                    summary[n].append((R2A, R2A3, abs(mu), np.degrees(np.angle(mu)), un2, np.mean(np.abs(dms))))
                    print(f"     {a_ev}   |  {R2A:6.3f} | {R2A3:6.3f} | {abs(mu):.3f} @ {np.degrees(np.angle(mu)):6.1f} ({R2B:.3f}) | {un1:7.3f} / {un2:6.4f} | {abs(dms[0]):.2f}, {abs(dms[1]):.2f}")
    print("\n--- summary over the 30 conditions (2 backgrounds x 3 S/M x 5 a_ev) per mode ---")
    print("  mode   | lam: |.| @ angle | fit A R2 (a2>=2): median, min | fit A R2 (a2>=3): median, min | fitted |mu|: median [min, max] | fitted angle: median [min, max] | later-source share (offset>=3): median, max")
    for n in VIS:
        v = np.array(summary[n]); lam = LAM[n]
        big = v[:, 5] > 0.15                         # conditions where the record is not tiny
        print(f"  {n:6s} | {abs(lam):.3f} @ {np.degrees(np.angle(lam)):6.1f} | {np.median(v[:, 0]):.3f}, {v[:, 0].min():.3f} | {np.nanmedian(v[:, 1]):.3f}, {np.nanmin(v[:, 1]):.3f} | "
              f"{np.median(v[:, 2]):.3f} [{v[:, 2].min():.3f}, {v[:, 2].max():.3f}] | {np.median(v[:, 3]):7.1f} [{v[:, 3].min():7.1f}, {v[:, 3].max():7.1f}] | {np.median(v[:, 4]):.4f}, {v[:, 4].max():.4f}"
              f" | restricted to records with mean amplitude > 0.15 K S ({big.sum()} conditions): fit A R2 min {v[big, 0].min():.3f}, |mu| in [{v[big, 2].min():.3f}, {v[big, 2].max():.3f}], angle in [{v[big, 3].min():.1f}, {v[big, 3].max():.1f}]")

    # ---------- Part 2: sampled data ----------
    print("\n=== Part 2: N(0,1) samples, final step, runner-up younger than the max ===")
    acc = {}
    for seed in (1019, 1020, 1021, 1022):
        x = sample(400_000, seed)
        H = run(x)
        st = running_stats(x)
        p1, p2 = st["p1"][:, -1], st["p2"][:, -1]
        Mx, Sx, Tx = st["M1"][:, -1], st["M2"][:, -1], np.maximum(st["M3"][:, -1], 0)
        m = coords(H[:, -1])
        sel = (p2 > p1) & (Sx > 0)
        for k, v in dict(a_ev=(p2 - p1)[sel], a2=(9 - p2)[sel], M=Mx[sel], S=Sx[sel], T=Tx[sel], **{f"m_{n}": m[n][sel] / K for n in VIS}).items():
            acc.setdefault(k, []).append(v)
    d = {k: np.concatenate(v) for k, v in acc.items()}
    print("  sequences used:", len(d["a2"]), "(from 1.6M, seeds 1019-1022)")
    for n in VIS:
        cplx, lam = IS_COMPLEX[n], LAM[n]
        print(f"\n  mode {n} (lam = {abs(lam):.3f} @ {np.degrees(np.angle(lam)):.1f} deg): beta(a_ev, a2) = coefficient of S in m_k(9)/K ~ alpha M + beta S + gamma T+")
        print("    a_ev | beta at a2 = 2, 3, ... (amplitude@phase) | fit c*lam^a2 over a2>=2: R2 on the betas | fitted free mu: |mu| @ angle | R2 of the data: free per-age vectors / rotating-record constraint")
        for a_ev in range(1, 6):
            a2s = np.arange(2, 9 - a_ev + 1)
            betas, sse_free, sse_con, sst, cells = [], 0.0, 0.0, 0.0, []
            for a2 in a2s:
                s = (d["a_ev"] == a_ev) & (d["a2"] == a2)
                X = np.column_stack([d["M"][s], d["S"][s], d["T"][s]])
                y = d[f"m_{n}"][s]
                coef, *_ = np.linalg.lstsq(X.astype(complex), y.astype(complex), rcond=None)
                betas.append(coef[1]); cells.append((X, y))
                sse_free += np.sum(np.abs(y - X @ coef) ** 2); sst += np.sum(np.abs(y - y.mean()) ** 2)
            betas = np.array(betas)
            c, R2A, mu, R2B = fits(a2s, betas, lam, True)
            for a2, (X, y) in zip(a2s, cells):           # constrained: beta fixed to c*lam^a2, alpha and gamma refitted
                yy = y - c * lam ** a2 * X[:, 1]
                coef, *_ = np.linalg.lstsq(X[:, [0, 2]].astype(complex), yy.astype(complex), rcond=None)
                sse_con += np.sum(np.abs(yy - X[:, [0, 2]] @ coef) ** 2)
            f = (lambda v: f"{abs(v):5.2f}@{np.degrees(np.angle(v)):6.1f}") if cplx else (lambda v: f"{v.real:6.2f}")
            print(f"     {a_ev}   | " + " ".join(f(v) for v in betas) + f" | {R2A:.3f} | {abs(mu):.3f} @ {np.degrees(np.angle(mu)):6.1f} | {1 - sse_free / sst:.4f} / {1 - sse_con / sst:.4f}")
