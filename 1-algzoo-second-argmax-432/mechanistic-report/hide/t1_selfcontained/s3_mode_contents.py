"""S3. What does each mode hold?  Quiet final states (t = 9).  Every mode coordinate is divided by M.
For each of the nine modes: the share of the variance of its PIECE of the scores (indices 0-7, centred over the
index, divided by M; variance around the mean over states) explained by nested descriptions, least squares within
cells, fitted on the first half of the states and scored on the second half:
   D1  f(a_M) M                              one pattern per age of the maximum
   D2  per a_M:  M, S                        = alpha(a_M) (M - S) + beta(a_M) S   ("locked to the clock" form)
   D3  per (a_M, a_S):  M, S                 the top two and their ages
   D4  per (a_M, a_S, a_T):  M, S, T         the top three and their ages
   D5  D4 + x_9, x_8                         + the two newest inputs
and a FUNCTIONAL test of each description: that mode's piece is replaced by the fitted one, every other piece is
true; P(runner-up on top among ages >= 2) and agreement of the top entry with the full score's.
Then the short clock: how much of it is locked to the main clock.  Then the S-coefficient of each tape mode as a
function of the age of the runner-up: does it follow free evolution lam^a_S?
Run:  python -B s3_mode_contents.py > s3_mode_contents.txt"""
from t1_common import *

d = population(600_000, 301); n = len(d["h"]); ar = np.arange(n)
pc = pieces(d["h"]); full = d["h"] @ U.T; c = coords(d["h"])
top_full = top_entry(full, d["valid"])
train = ar < n // 2; test = ~train
M, S, T3 = d["M"], d["S"], d["T3"]; x9, x8 = d["xr"][:, 9], d["xr"][:, 8]
cM, cMS, cMST = d["aM"], d["aM"] * 10 + d["aS"], d["aM"] * 100 + d["aS"] * 10 + d["aT"]
print(f"{n} quiet final states (600000 sequences, seed 301); fits on the first half, numbers from the second half")


def cell_fit(Y, F, cell, want_coef=False):
    """Least squares of Y (n, k) on F (n, f) within each cell (fitted on train); predictions for all rows [and coefficients]."""
    out = np.zeros_like(Y); coef = {}
    order = np.argsort(cell, kind="stable"); bounds = np.flatnonzero(np.diff(cell[order])) + 1
    for idx in np.split(order, bounds):
        tr = idx[train[idx]]
        if len(tr) > 3 * F.shape[1] + 5:
            b = np.linalg.lstsq(F[tr], Y[tr], rcond=None)[0]; out[idx] = F[idx] @ b; coef[cell[idx[0]]] = b
    return (out, coef) if want_coef else out


DESCR = [("D1 f(aM) M", np.stack([M], 1), cM), ("D2 aM: M,S", np.stack([M, S], 1), cM), ("D3 (aM,aS): M,S", np.stack([M, S], 1), cMS),
         ("D4 (aM,aS,aT): M,S,T", np.stack([M, S, T3], 1), cMST), ("D5 D4 + x9,x8", np.stack([M, S, T3, x9, x8], 1), cMST)]
cen = lambda A: (A[:, :8] - A[:, :8].mean(1, keepdims=True)) / M[:, None]
pS = lambda A: np.mean((top_entry(A, d["valid"]) == d["iS"])[test])
ag = lambda A: np.mean((top_entry(A, d["valid"]) == top_full)[test])

print("\nA. share of the variance of each mode's piece (indices 0-7, centred over the index, / M) explained by each description")
print("   mode     rms of piece | " + " | ".join(f"{nm:>20s}" for nm, _, _ in DESCR))
fits = {}
for m in NAMES:
    Y = cen(pc[m]); tot = ((Y[test] - Y[test].mean(0)) ** 2).sum()
    row = f"   {m:7s}   {np.sqrt((Y[test] ** 2).mean()):6.2f}     | "
    for nm, F, cell in DESCR:
        pred = cell_fit(pc[m] / M[:, None], F / M[:, None], cell) * M[:, None]; fits[m, nm] = pred
        row += f"{1 - ((Y - cen(pred))[test] ** 2).sum() / tot:20.3f} | "
    print(row)
print(f"\nB. functional test: one mode's piece replaced by its description, all other pieces true.  P(S top) / agreement with the full score's top entry   (full: {pS(full):.3f} / 1.000)")
print("   mode      deleted        | " + " | ".join(f"{nm:>20s}" for nm, _, _ in DESCR))
for m in NAMES:
    row = f"   {m:7s}  {pS(full - pc[m]):.3f} / {ag(full - pc[m]):.3f} | "
    for nm, _, _ in DESCR:
        A = full - pc[m] + fits[m, nm]
        row += f"       {pS(A):.3f} / {ag(A):.3f} | "
    print(row)
print("   all four tape modes replaced together (rest true):")
tape = sum(pc[m] for m in TAPE)
row = f"   TAPE     {pS(full - tape):.3f} / {ag(full - tape):.3f} | "
for nm, _, _ in DESCR:
    A = full - tape + sum(fits[m, nm] for m in TAPE)
    row += f"       {pS(A):.3f} / {ag(A):.3f} | "
print(row)
ok = (top_full == d["iS"]) & test
print("   ... and among the states where the network itself has the runner-up on top, P(the substituted scores also do):")
print("            " + " | ".join(f"{nm}: {np.mean((top_entry(full - tape + sum(fits[m, nm] for m in TAPE), d['valid']) == d['iS'])[ok]):.3f}" for nm, _, _ in DESCR))

# ---------------------------------------------------------------- C. the short clock and the main clock
w4 = w[4]
zs, zc = c["short"] / M, c["clock"] / M
print("\nC. THE SHORT CLOCK.  z = short coordinate / M (complex).  Shares of sum |z - mean|^2 on the test half:")
tot = (np.abs(zs[test] - zs[test].mean()) ** 2).sum()
def share(pred):
    return 1 - (np.abs(zs - pred)[test] ** 2).sum() / tot
def cplx_fit(F, cell):
    Y = np.stack([zs.real, zs.imag], 1); F = F / M[:, None]
    p = cell_fit(Y, F, cell); return p[:, 0] + 1j * p[:, 1]
k = (np.conj(zc[train]) * zs[train]).sum() / (np.abs(zc[train]) ** 2).sum()
print(f"   kappa z_clock, one complex kappa = {abs(k):.3f} at {np.degrees(np.angle(k)):.0f} deg:                      {share(k * zc):.3f}")
pred = np.zeros(n, complex)
for a in range(2, 10):
    q = d["aM"] == a; kk = (np.conj(zc[q & train]) * zs[q & train]).sum() / (np.abs(zc[q & train]) ** 2).sum(); pred[q] = kk * zc[q]
print(f"   kappa(a_M) z_clock, one complex kappa per age of the maximum:                {share(pred):.3f}")
gap = M - S
print(f"   alpha(a_M) (M - S), a term in the gap with the phase of the maximum's age:   {share(cplx_fit(np.stack([gap], 1), cM)):.3f}")
print(f"   alpha(a_M) (M - S) + beta(a_M) S  (D2):                                      {share(cplx_fit(np.stack([gap, S], 1), cM)):.3f}")
print(f"   per age of the RUNNER-UP: M, S:                                              {share(cplx_fit(np.stack([M, S], 1), d['aS'])):.3f}")
print(f"   per (a_M, a_S): M, S  (D3):                                                  {share(cplx_fit(np.stack([M, S], 1), cMS)):.3f}")
print(f"   per (a_M, a_S, a_T): M, S, T  (D4):                                          {share(cplx_fit(np.stack([M, S, T3], 1), cMST)):.3f}")
print(f"   D4 + x_9, x_8  (D5):                                                         {share(cplx_fit(np.stack([M, S, T3, x9, x8], 1), cMST)):.3f}")
Y = np.stack([zs.real, zs.imag], 1)
_, co = cell_fit(Y, np.stack([gap, S], 1) / M[:, None], cM, want_coef=True)
print("   D2 coefficients in units of w_4 (modulus at phase):   age of the maximum:  alpha (per unit gap)   beta (per unit S)")
for a in range(2, 10):
    al, be = (co[a][0, 0] + 1j * co[a][0, 1]) / w4, (co[a][1, 0] + 1j * co[a][1, 1]) / w4
    print(f"      a_M = {a}:   alpha {abs(al):.3f} at {np.degrees(np.angle(al)):5.0f} deg    beta {abs(be):.3f} at {np.degrees(np.angle(be)):5.0f} deg")
print(f"   rms |z_short| / w_4 = {np.sqrt((np.abs(zs) ** 2).mean()) / w4:.3f} (units of M);  rms of (M - S)/M = {np.sqrt(((gap / M) ** 2).mean()):.3f}, of S/M = {np.sqrt(((S / M) ** 2).mean()):.3f}")

# ---------------------------------------------------------------- D. the coefficient of S against the runner-up's age
print("\nD. Coefficient of S in each tape coordinate (fit D3, per cell (a_M, a_S); units of w_4), by age of the runner-up.")
print("   Free evolution since the runner-up was written would give  coefficient(a_S + 1) / coefficient(a_S) = lam.")
for m in TAPE:
    cplx = IS_COMPLEX[m]
    Y = (np.stack([c[m].real, c[m].imag], 1) if cplx else np.real(c[m])[:, None]) / M[:, None]
    _, co = cell_fit(Y, np.stack([M, S], 1) / M[:, None], cMS, want_coef=True)
    get = lambda aM, aS, j: ((co[aM * 10 + aS][j, 0] + 1j * co[aM * 10 + aS][j, 1]) if cplx else co[aM * 10 + aS][j, 0]) / w4
    lam = LAM[m]
    print(f"   {m}: lam = {abs(lam):.3f} at {np.degrees(np.angle(lam)):.1f} deg")
    for nm, sel in (("runner-up arrived AFTER the maximum, a_M = 9", [(9, a) for a in range(2, 9)]), ("runner-up arrived BEFORE the maximum, a_M = 2", [(2, a) for a in range(3, 10)])):
        vals = [get(aM, aS, 1) for aM, aS in sel]
        print(f"      {nm}:  a_S = {sel[0][1]}..{sel[-1][1]}")
        print("         coefficient of S:  " + "  ".join(f"{abs(v_):5.2f}@{np.degrees(np.angle(v_)):5.0f}" for v_ in vals))
        rat = [vals[j + 1] / vals[j] for j in range(len(vals) - 1)]
        print("         ratio to previous:        " + "  ".join(f"{abs(r_):5.2f}@{np.degrees(np.angle(r_)):5.0f}" for r_ in rat))
