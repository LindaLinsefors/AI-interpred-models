"""V4 verifier: runner-ups that arrived after the max and switched NO comparator off ('unregistered').
Does the clock coordinate of the thresholds still carry the OLD gap (M - S_old), or does it come to know the new runner-up
(which a closed core could not: the only route is through the other ten units)?
z_c(t) = ell . theta(t+1) ~ alpha (M - S) + beta S + gamma (S - S_old+),  S_old = runner-up just before the present one arrived.
gamma/alpha = 1: radius still the old gap;  0: radius is the present gap.   Seeds 5024, 5025 (2 x 400k)."""
from vsetup import *

out = Tee("v12_unregistered.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=220)

ev, vecs = np.linalg.eig(Bc.T)
k = int(np.argmax(np.angle(ev) * (np.abs(ev) > 0.5)))
ell = vecs[:, k] / (vecs[:, k] @ (w[4] * b4))

Zc, M, S, A1, A2, NON, SOLD, TT, ZCOMP, ZREST = [], [], [], [], [], [], [], [], [], []
for seed in (5024, 5025):
    x = sample(400_000, seed)
    H, Zp = run(x, return_pre=True)
    th, _ = thresholds(H)
    rs = running_stats(x)
    non = (Zp[:, :, COMP] > 0).sum(2)
    p2 = np.maximum(rs["p2"], 0)
    Sshift = np.concatenate([np.full((x.shape[0], 1), -np.inf), rs["M2"][:, :-1]], axis=1)    # index p -> runner-up of x_0..x_{p-1}
    SOLD.append(np.take_along_axis(Sshift, p2, axis=1))
    NON.append(np.take_along_axis(non, p2, axis=1))                                          # comparators on at the arrival step
    Zc.append(th[:, 1:] @ ell); M.append(rs["M1"]); S.append(rs["M2"]); A1.append(rs["a1"]); A2.append(rs["a2"])
    TT.append(np.broadcast_to(np.arange(10), x.shape))
    ZCOMP.append(((H[:, :, COMP] @ W[np.ix_(COMP, COMP)].T) / g) @ ell)                      # part of z_c written by the comparators
    ZREST.append(((H[:, :, OTHER10] @ W[np.ix_(COMP, OTHER10)].T) / g) @ ell)                # part written by the other ten units
Zc, M, S, A1, A2, NON, SOLD, TT, ZCOMP, ZREST = map(np.concatenate, (Zc, M, S, A1, A2, NON, SOLD, TT, ZCOMP, ZREST))
SOLDp = np.maximum(SOLD, 0.0)
base = (TT <= 8) & (M > 0.3) & (A2 >= 0) & (A2 < A1) & (S > 0) & ((S - SOLDp) > 0.05 * M)


def fit(sel, target):
    X = np.stack([(M - S)[sel], S[sel], (S - SOLDp)[sel]], 1).astype(complex)
    c, *_ = np.linalg.lstsq(X, target[sel], rcond=None)
    return c


out("states: runner-up younger than the max (a2 < a1), S > 0, M > 0.3, t <= 8, and S - S_old+ > 0.05 M")
out(f"  n = {int(base.sum())};  unregistered (all four comparators stayed on when the runner-up arrived): {(NON[base] == 4).mean():.4f}")
out("\nRe(gamma/alpha) by steps since the runner-up arrived (a2); 1 = the clock still carries the old gap, 0 = it carries the present gap")
out("  a2 :  registered: n, |alpha|, Re(gamma/alpha)   |  unregistered: n, |alpha|, Re(gamma/alpha)   |  unregistered, split of z_c: comparator part, part from the other ten units")
for a2 in range(0, 7):
    row = f"  {a2}  :"
    for name, cond in (("reg", NON < 4), ("unreg", NON == 4)):
        sel = base & (A2 == a2) & cond
        if sel.sum() < 3000:
            row += "        -- too few --         |"
            continue
        c = fit(sel, Zc)
        row += f"  {int(sel.sum()):7d}  {abs(c[0]):.3f}  {(c[2] / c[0]).real:+.3f}            |"
        if name == "unreg":
            cc, cr = fit(sel, ZCOMP), fit(sel, ZREST)
            row += f"  comparators: gamma/alpha_tot = {(cc[2] / c[0]).real:+.3f}; other ten: {(cr[2] / c[0]).real:+.3f}"
    out(row)

out("\nthe same restricted to max ages a1 = 5..8 (where most unregistered arrivals happen):")
for a2 in range(0, 6):
    row = f"  {a2}  :"
    for name, cond in (("reg", NON < 4), ("unreg", NON == 4)):
        sel = base & (A2 == a2) & cond & (A1 >= 5)
        if sel.sum() < 3000:
            row += "        -- too few --         |"
            continue
        c = fit(sel, Zc)
        row += f"  {int(sel.sum()):7d}  {abs(c[0]):.3f}  {(c[2] / c[0]).real:+.3f}            |"
    out(row)

out("\nmodel-free: unregistered arrivals with a large step, S - S_old+ > 0.2 M; median |z_c| / (M - S) and median |z_c| / (M - S_old+) by a2")
for a2 in range(0, 6):
    sel = base & (A2 == a2) & (NON == 4) & ((S - SOLDp) > 0.2 * M)
    if sel.sum() < 1000:
        continue
    out(f"  a2 = {a2}: n = {int(sel.sum()):6d}   |z_c|/(M - S): {np.median(np.abs(Zc[sel]) / (M - S)[sel]):.3f}   |z_c|/(M - S_old+): {np.median(np.abs(Zc[sel]) / (M - SOLDp)[sel]):.3f}"
        f"   median (M - S)/(M - S_old+) = {np.median((M - S)[sel] / (M - SOLDp)[sel]):.3f}")
