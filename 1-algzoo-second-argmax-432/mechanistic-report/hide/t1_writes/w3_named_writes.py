"""W3. What do the SIZES of the writes mean?  Every write amplitude is a clipped comparison of the current input with a
threshold that the previous state supplies:   a_s[u] = w_u x_s + drive_u(s),   drive_u(s) = (W h_(s-1))[u],
    clip of non-pulse unit u = [-a_s[u]]_+ ,   pulse of unit u in {4,5,9} = [a_s[u]]_+ .
Reduced description tested here: replace drive_u(s) by a linear function of NAMED quantities of the earlier inputs x_0..x_(s-1):
    M+, S+, T+ = their three largest values clipped at 0, and x_(s-1) (the latest input),
with coefficients that depend on the ages (at step s-1) of those values:
    L1: coefficients depend on the age of the maximum only; regressors M+, S+, x_(s-1)
    L2: on the ages of maximum and runner-up; regressors M+, S+, T+, x_(s-1)
    L3: on the ages of maximum, runner-up and third; regressors M+, S+, T+, x_(s-1)
(an age counts as 'absent' if the value is absent or not positive).  Fitted on 300k sequences, everything below is measured
on 200k other sequences.  The tape is then rebuilt from the modelled amplitudes with the EXACT kernel lam^age.
Run:  python -B w3_named_writes.py > w3_named_writes.txt"""
from wcommon import *

def features(x):
    st = running_stats(x)
    vals = [np.maximum(st[f"M{k}"][:, :-1], 0) for k in (1, 2, 3)]                       # prefix x_0..x_(s-1), s = 1..9
    ages = [np.where(vals[k] > 0, st[f"a{k + 1}"][:, :-1], 9) for k in range(3)]
    F = np.stack(vals + [x[:, :-1]], -1)                                                  # (B, 9, 4): M+, S+, T+, x_(s-1)
    return F.reshape(-1, 4), [a.reshape(-1) for a in ages]

LEVELS = {"L1": (lambda a: a[0], [0, 1, 3]), "L2": (lambda a: a[0] * 10 + a[1], [0, 1, 2, 3]),
          "L3": (lambda a: a[0] * 100 + a[1] * 10 + a[2], [0, 1, 2, 3])}

def fit_predict(cfg_tr, F_tr, Y_tr, cfg_te, F_te):
    """Per configuration, least squares Y = F b (no intercept: the network is homogeneous). Returns the test prediction."""
    out = np.zeros((len(cfg_te), Y_tr.shape[1])); coef = {}
    o_tr = np.argsort(cfg_tr, kind="stable"); u_tr, i_tr = np.unique(cfg_tr[o_tr], return_index=True); i_tr = list(i_tr) + [len(o_tr)]
    o_te = np.argsort(cfg_te, kind="stable"); cs = cfg_te[o_te]
    for k, c in enumerate(u_tr):
        rows = o_tr[i_tr[k]: i_tr[k + 1]]
        b = np.linalg.lstsq(F_tr[rows], Y_tr[rows], rcond=None)[0]
        lo, hi = np.searchsorted(cs, c), np.searchsorted(cs, c, "right")
        out[o_te[lo:hi]] = F_te[o_te[lo:hi]] @ b; coef[int(c)] = b
    return out, coef, len(u_tr)

xtr, xte = sample(300_000, 303), sample(200_000, 304)
Htr, Ztr, _ = amplitudes(xtr); Hte, Zte, Amp = amplitudes(xte)
drive = lambda x, Z: (Z[:, 1:] - x[:, 1:, None] * w).reshape(-1, 16)                       # (B*9, 16), steps s = 1..9
Ytr, Yte = drive(xtr, Ztr), drive(xte, Zte)
Ftr, atr = features(xtr); Fte, ate = features(xte)
Bte = len(xte)
st = prefix_stats(xte, 9); q = st["quiet"]; M = st["M"]
h9 = Hte[:, 9]; c9 = tape_coords(h9); ans0 = answer(h9); corr = second_argmax(xte)
cq = c9[q] / M[q, None]; sq = cq @ R_SCORE[:8].T
print(f"train 300000 sequences, test {Bte}; quiet final states in the test set n = {q.sum()}; baseline accuracy there {np.mean(ans0[q] == corr[q]):.4f}")

def amp_from_drive(Yhat):
    """Modelled write amplitudes (B, 10, 17) from a modelled drive (B*9, 16); step 0 is exact (drive = 0)."""
    Zh = Zte.copy(); Zh[:, 1:] = xte[:, 1:, None] * w + Yhat.reshape(Bte, 9, 16)
    A = np.zeros_like(Amp); A[:, :, 0] = xte
    A[:, 1:, 1:4] = np.maximum(Zh[:, :-1][:, :, PULSE], 0)
    A[:, :, 4:] = np.maximum(-Zh[:, :, ORD], 0)
    return A

def report(A_model, label):
    """Rebuild the tape with all / one group of channels modelled; R2 per coordinate and the functional test."""
    for g, chs in [("ALL channels", list(range(1, NCH)))] + list(GROUPS.items())[1:]:
        A = Amp.copy(); A[:, :, chs] = A_model[:, :, chs]
        c = rebuild(A, 9); ch = c[q] / M[q, None]
        a = answer(set_tape(h9, c))
        r2a = 1 - (A[:, :, chs] - Amp[:, :, chs]).var() / Amp[:, :, chs].var()
        print(f"    {label} modelled: {g:16s} R2 coords " + " ".join(f"{v:6.3f}" for v in r2(cq, ch)) +
              f"  scores {1 - (((ch - cq) @ R_SCORE[:8].T) ** 2).sum() / ((sq - sq.mean(0)) ** 2).sum():6.3f} | amplitudes R2 {r2a:5.3f}"
              f" | answers changed {np.mean(a[q] != ans0[q]):.4f}  acc {np.mean(a[q] == corr[q]):.4f}")

Mrun = np.maximum(Fte[:, 0], 1e-12); pos = Fte[:, 0] > 0.3
TH = {4: -1 / w[4], 1: -1 / w[1], 6: -1 / w[6], 7: -1 / w[7], 8: -1 / w[8]}               # threshold = drive * TH[u]
preds = {}
for lev, (cf, cols) in LEVELS.items():
    Yhat, coef, ncfg = fit_predict(cf(atr), Ftr[:, cols], Ytr, cf(ate), Fte[:, cols])
    preds[lev] = (Yhat, coef)
    print(f"\n=== {lev}: {ncfg} age configurations, {len(cols)} regressors each ===")
    print("    thresholds theta_u = -drive_u / w_u of units 4, 1, 6, 7, 8 (steps with running max > 0.3): rms error / running max, and R2 of the drive of every unit")
    print("      " + "  ".join(f"theta_{u}: {np.sqrt(np.mean((((Yhat - Yte)[:, u] * TH[u]) / Mrun)[pos] ** 2)):.3f} M" for u in TH))
    print("      R2 of drive, units 0..15: " + " ".join(f"{v:.3f}" for v in r2(Yte, Yhat)))
    report(amp_from_drive(Yhat), lev)

print("\n=== L1 coefficients: threshold of unit u when the maximum of the earlier inputs is a_M steps old (at step s-1) ===")
print("    theta_u = alpha M+ + beta S+ + gamma x_(s-1);  for units 5 and 9 the drive itself (pre-activation minus w_u x_s)")
coef = preds["L1"][1]
for u in (4, 7, 1, 6, 8, 5, 9):
    sc = TH.get(u, 1.0)
    print(f"    unit {u}:  a_M      " + " ".join(f"{a:7d}" for a in range(9)))
    for r, nm in enumerate(("M+", "S+", "x_(s-1)")):
        print(f"      {nm:8s}       " + " ".join(f"{coef[a][r, u] * sc:7.2f}" for a in range(9)))

print("\n=== How much of each drive comes from the tape part of h_(s-1)? (test set, steps 1..9; covariance share of the tape part) ===")
Hprev = Hte[:, :-1].reshape(-1, 16)
dt = tape_part(Hprev) @ W.T
print("    unit:   " + " ".join(f"{u:6d}" for u in range(16)))
print("    share:  " + " ".join(f"{v:6.3f}" for v in ((dt - dt.mean(0)) * (Yte - Yte.mean(0))).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)))
print("    std of the tape part of the drive / std of the drive: " + " ".join(f"{v:5.2f}" for v in dt.std(0) / Yte.std(0)))
np.savez_compressed("w3_cache.npz", L3=preds["L3"][0].astype(np.float32), L2=preds["L2"][0].astype(np.float32))
