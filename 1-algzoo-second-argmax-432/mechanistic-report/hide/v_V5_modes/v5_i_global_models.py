"""Verifier V5, part I: how much of the clock coordinate does each VERSION of the clock law explain over all states?
Also: conditioning of the mode basis (non-normality) and the angle between the clock and short eigenplanes.

States: a1 >= 1 and S > 0.05 (the claim's set), pooled over all cells; then all states with a1 >= 1 and M > 0 using S+ = max(S, 0).
Seeds 5060, 5061 (2 x 200k sequences).   Run:  python -B v5_i_global_models.py > v5_i_global_models.txt
"""
from v5_lib import *

acc = {}
for seed in [5060, 5061]:
    x, H, Z, st, m = batch(seed)
    for k, a in (("M", st["M"]), ("S", st["S"]), ("a1", st["a1"]), ("a2", st["a2"]), ("z", m["clock"] / W4)):
        acc.setdefault(k, []).append(a.reshape(-1))
D = {k: np.concatenate(v) for k, v in acc.items()}
M, S, a1, a2, z = (D[k] for k in ("M", "S", "a1", "a2", "z"))
lam = LAM["clock"]
print("seeds [5060, 5061], 200k sequences each")

def score(sel, zhat, label, npar):
    res = z[sel] - zhat
    r2u = 1 - (np.abs(res) ** 2).sum() / (np.abs(z[sel]) ** 2).sum()
    print(f"   {label:86s} complex parameters {npar:4d}   uncentred R2 = {r2u:.4f}   rms error / rms |z| = {np.sqrt(1 - r2u):.3f}")

for title, sel, Suse in (("states with a1 >= 1 and S > 0.05 (the claim's set)", (a1 >= 1) & (S > 0.05), S),
                         ("ALL states with a1 >= 1 and M > 0, with S+ = max(S, 0) in place of S", (a1 >= 1) & (M > 0), np.maximum(np.where(np.isfinite(S), S, 0), 0))):
    print("=" * 120)
    print(title, f"  n = {int(sel.sum())}")
    g = (M - Suse)[sel]; s_ = Suse[sel]; A1 = a1[sel]; A2 = np.where(a2[sel] < 0, 10, a2[sel]); zz = z[sel]
    # (0) literal reading of the interpretation, no free parameter
    score(sel, np.exp(1j * (A1 - 1) * OMEGA) * g, "(0) z/w4 = e^{i (a1-1) 42.75deg} (M - S)                [literal, no decay, nothing fitted]", 0)
    score(sel, lam ** (A1 - 1) * g, "(1) z/w4 = lam^(a1-1) (M - S)                             [eigenvalue incl. decay, nothing fitted]", 0)
    # (2) one complex prefactor and best real c
    best = None
    for c in np.arange(0.5, 1.21, 0.01):
        b = lam ** (A1 - 1) * (M[sel] - c * Suse[sel])
        k = (np.conj(b) * zz).sum() / (np.abs(b) ** 2).sum()
        r = 1 - (np.abs(zz - k * b) ** 2).sum() / (np.abs(zz) ** 2).sum()
        if best is None or r > best[0]:
            best = (r, c, k)
    score(sel, best[2] * lam ** (A1 - 1) * (M[sel] - best[1] * Suse[sel]), f"(2) z/w4 = k lam^(a1-1) (M - c S), fitted k = {pol(best[2])}, c = {best[1]:.2f}", 2)
    # (3) alpha(a1) (M - S)
    zhat = np.zeros_like(zz)
    for a in range(1, 10):
        q = A1 == a
        zhat[q] = (g[q] * zz[q]).sum() / (g[q] ** 2).sum() * g[q]
    score(sel, zhat, "(3) z/w4 = alpha(a1) (M - S)", 9)
    # (4) alpha(a1) (M - S) + beta(a1) S
    zhat = np.zeros_like(zz)
    for a in range(1, 10):
        q = A1 == a
        Xq = np.stack([g[q], s_[q]], 1)
        zhat[q] = Xq @ np.linalg.solve(Xq.T @ Xq, Xq.T @ zz[q])
    score(sel, zhat, "(4) z/w4 = alpha(a1) (M - S) + beta(a1) S", 18)
    # (5) additive: alpha(a1) (M - S) + beta(a2) S     (one sparse least-squares problem)
    n = len(zz)
    Xd = np.zeros((n, 9 + 11))
    Xd[np.arange(n), A1 - 1] = g
    Xd[np.arange(n), 9 + A2] = s_
    keep = Xd.any(0)
    coef = np.linalg.lstsq(Xd[:, keep].T @ Xd[:, keep], Xd[:, keep].T @ zz, rcond=None)[0]
    score(sel, Xd[:, keep] @ coef, "(5) z/w4 = alpha(a1) (M - S) + beta(a2) S                 [a second record, rotating with the runner-up's age]", int(keep.sum()))
    full = np.zeros(20, complex); full[keep] = coef
    print("       (5) alpha(a1), a1 = 1..9: " + "  ".join(pol(c) for c in full[:9]))
    print("       (5) beta(a2),  a2 = 0..9: " + "  ".join(pol(c) for c in full[9:19]))
    print("       (5) phase steps of beta over a2 (deg): ", np.round(np.degrees(wrap(np.diff(np.angle(full[9:19])))), 1))
    # (6) per cell
    zhat = np.zeros_like(zz); ncell = 0
    for a in range(1, 10):
        for b in range(0, 11):
            q = (A1 == a) & (A2 == b)
            if q.sum() < 50:
                zhat[q] = 0
                continue
            Xq = np.stack([g[q], s_[q]], 1)
            if np.linalg.matrix_rank(Xq.T @ Xq) < 2:
                zhat[q] = (g[q] * zz[q]).sum() / (g[q] ** 2).sum() * g[q]; ncell += 1
                continue
            zhat[q] = Xq @ np.linalg.solve(Xq.T @ Xq, Xq.T @ zz[q]); ncell += 2
    score(sel, zhat, "(6) z/w4 = alpha(a1, a2) (M - S) + beta(a1, a2) S        [the per-cell model of claim M5]", ncell)

print("=" * 120)
print("CONDITIONING of the mode basis (not invariant under per-unit rescaling; shown for raw units and for unit-rms units)")
x, H, Z, st, m = batch(5060)
sig = np.sqrt((H.reshape(-1, N) ** 2).mean(0))[ORD]
nn = np.linalg.norm
for nm, c in (("raw", np.ones(13)), ("unit rms", 1 / sig)):
    print(f"   [{nm}] eigenvalue condition numbers |L_k| |R_k| / |L_k . R_k| (1 = normal matrix): "
          + ", ".join(f"{n} {nn(LEFT[n] / c) * nn(RIGHT[n] * c) / abs(LEFT[n] @ RIGHT[n]):.1f}" for n in NAMES))
    def plane(n):
        return np.linalg.qr(np.stack([(RIGHT[n] * c).real, (RIGHT[n] * c).imag], 1))[0]
    sv = np.linalg.svd(plane("clock").T @ plane("short"), compute_uv=False)
    print(f"   [{nm}] cosines of the principal angles between the clock plane and the short plane: {np.round(sv, 4)}"
          f"   (angles {np.round(np.degrees(np.arccos(np.clip(sv, 0, 1))), 1)} deg)")
    lv = RIGHT["level"].real * c
    print(f"   [{nm}] cos(level eigenvector, runner eigenvector) = {abs(lv @ (RIGHT['runner'].real * c)) / nn(lv) / nn(RIGHT['runner'].real * c):.4f};"
          f"  |cos| between level eigenvector and the clock plane = {nn(plane('clock').T @ lv) / nn(lv):.4f}")
