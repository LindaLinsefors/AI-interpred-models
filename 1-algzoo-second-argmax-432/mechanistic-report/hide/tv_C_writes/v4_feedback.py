"""V4. Two follow-ups of v3.
(1) Is 'the thresholds do not come from the tape' true functionally?  The tape's part of the drive (W h_(s-1))[u] is
    (a) set to zero, (b) replaced by its mean given the age of the earlier maximum (in units of the running maximum),
    (c) replaced by that of a random other sequence at the same step with the same age of the maximum (rescaled).
    Only the named channels are recomputed; the tape is rebuilt with the exact kernel and put into h_9.
(2) Unif(-1,1) inputs with the reduced description L3 fitted ON uniform inputs (v3 used N(0,1) coefficients).
Run:  python -B v4_feedback.py > v4_feedback.txt"""
from vcommon import *

x = sample(200_000, 7304); B = len(x)
H, Z, Amp = amplitudes(x)
vals, ages = prefix_before(x)
Mrun = vals[0][:, 1:]; aM = ages[0][:, 1:]                                  # (B, 9): running maximum (clipped at 0) and its age at step s-1
Y = Z[:, 1:] - x[:, 1:, None] * w                                           # (B, 9, 16) drive at steps 1..9
dt = np.einsum("bsk,uk->bsu", H[:, :-1][:, :, ORD] @ P.T, W[:, ORD])        # the tape's part of the drive
st = top3(x, 9); q = st["quiet"]; h9 = H[q, 9]; a0 = answer(h9)
print(f"200000 N(0,1) test sequences (seed 7304), quiet final states n = {q.sum()}")

rel = dt / np.maximum(Mrun, 1e-12)[:, :, None] * (Mrun > 0)[:, :, None]     # in units of the running maximum
mean_rep = np.zeros_like(dt); perm_rep = np.zeros_like(dt); rng = np.random.default_rng(11)
for s in range(9):
    for a in np.unique(aM[:, s]):
        r = np.where(aM[:, s] == a)[0]
        mean_rep[r, s] = rel[r, s].mean(0) * Mrun[r, s, None]
        perm_rep[r, s] = rel[rng.permutation(r), s] * Mrun[r, s, None]


def amp_from_drive(Yh):
    Zh = Z.copy(); Zh[:, 1:] = x[:, 1:, None] * w + Yh
    A_ = np.zeros_like(Amp); A_[:, :, 0] = x
    A_[:, 1:, 1:4] = np.maximum(Zh[:, :-1][:, :, PULSE], 0); A_[:, :, 4:] = np.maximum(-Zh[:, :, ORD], 0)
    return A_


ch = lambda *us: [4 + ORD.index(u) for u in us]
tests = {"pulse 4": [1], "pulse 5": [2], "pulse 9": [3], "comparators 6, 7": ch(6, 7), "comparators 1, 8": ch(1, 8), "wave clips of 3, 15, 13, 14": ch(3, 15, 13, 14),
         "all other wave clips": ch(0, 2, 10, 11, 12)}
print("\n(1) answers changed on quiet final states when the tape's part of the drive of the named units is replaced")
print(f"    {'channels recomputed':30s}  set to zero   mean given the age of the maximum   another sequence's (same step and age)   | rms change of the amplitude / rms amplitude (zero, mean)")
models = {"zero": amp_from_drive(Y - dt), "mean": amp_from_drive(Y - dt + mean_rep), "perm": amp_from_drive(Y - dt + perm_rep)}
for nm, chs in tests.items():
    row = []
    for k in ("zero", "mean", "perm"):
        A_ = Amp[q].copy(); A_[:, :, chs] = models[k][q][:, :, chs]
        row.append(np.mean(answer(with_tape(h9, rebuild(A_, 9))) != a0))
    rc = [np.sqrt(((models[k][:, :, chs] - Amp[:, :, chs]) ** 2).mean() / (Amp[:, :, chs] ** 2).mean()) for k in ("zero", "mean")]
    print(f"    {nm:30s}  {row[0]:.4f}        {row[1]:.4f}                              {row[2]:.4f}                                   | {rc[0]:.3f}, {rc[1]:.3f}")
print("    for scale: std of the tape's part of the drive / std of the drive, unit 5: "
      f"{dt[:, :, 5].std() / Y[:, :, 5].std():.3f};  the same over steps where unit 5 fires: {dt[:, :, 5][Z[:, 1:, 5] > 0].std() / Y[:, :, 5][Z[:, 1:, 5] > 0].std():.3f};"
      f"  std of the tape's part / std of unit 5's pulse where it fires: {dt[:, :, 5][Z[:, 1:, 5] > 0].std() / Z[:, 1:, 5][Z[:, 1:, 5] > 0].std():.3f}")

print("\n(2) Unif(-1,1): L3 (thresholds from M+, S+, T+, latest input; weights by the three ages) fitted on 300000 uniform sequences (seed 7306), tested on 200000 (seed 7305)")
def feats(x):
    vals, ages = prefix_before(x)
    F = np.stack([vals[0][:, 1:], vals[1][:, 1:], vals[2][:, 1:], x[:, :-1]], -1).reshape(-1, 4)
    return F, (ages[0][:, 1:] * 100 + ages[1][:, 1:] * 10 + ages[2][:, 1:]).reshape(-1)
def fit_predict(xtr, xte):
    _, Ztr = run(xtr, return_pre=True); Ytr = (Ztr[:, 1:] - xtr[:, 1:, None] * w).reshape(-1, 16)
    Ftr, ctr = feats(xtr); Fte, cte = feats(xte)
    out = np.zeros((len(cte), 16)); o1 = np.argsort(ctr, kind="stable"); c1 = ctr[o1]; o2 = np.argsort(cte, kind="stable"); c2 = cte[o2]
    for c in np.unique(c1):
        r = o1[np.searchsorted(c1, c): np.searchsorted(c1, c, "right")]; e = o2[np.searchsorted(c2, c): np.searchsorted(c2, c, "right")]
        out[e] = Fte[e] @ np.linalg.lstsq(Ftr[r], Ytr[r], rcond=None)[0]
    return out.reshape(len(xte), 9, 16)
for dist, s1, s2 in (("uniform", 7306, 7305), ("normal", 7303, 7304)):
    x = sample(200_000, s2, dist); B = len(x); H, Z, Amp = amplitudes(x)
    A_ = amp_from_drive(fit_predict(sample(300_000, s1, dist), x)); A_[:, :, 0] = Amp[:, :, 0]
    st = top3(x, 9); q = st["quiet"]; a0 = answer(H[q, 9]); chg = answer(with_tape(H[q, 9], rebuild(A_[q], 9))) != a0
    gap = ((st["S"] - st["T"]) / st["M"])[q]
    print(f"    {dist:8s} fitted and tested on the same distribution: quiet final states n = {q.sum()}, answers changed {chg.mean():.4f};  "
          f"share of quiet states with (S - T)/M < 0.05: {np.mean(gap < 0.05):.3f}, changed there {chg[gap < 0.05].mean():.3f}, elsewhere {chg[gap >= 0.05].mean():.3f}")
