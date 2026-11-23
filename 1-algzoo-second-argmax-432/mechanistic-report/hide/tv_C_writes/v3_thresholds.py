"""V3. Checks of claims 5, 6, 8 (thresholds as mixes of the earlier maximum and runner-up; the reduced description as a
whole; where the thresholds come from), with attacks:
 (A) control models: threshold from the maximum alone, and one mix for all ages (does the runner-up / the age matter?);
 (B) functional test of 'the thresholds do not come from the tape' (the investigator gave a variance ratio only);
 (C) the same reduced description applied to the NON-tape modes (is the failure specific to the tape?);
 (D) other populations: states at t = 5, 7 read out directly, and Unif(-1,1) inputs with the N(0,1) coefficients.
Fit on 300k sequences (seed 7303), test on 200k others (seed 7304).   Run:  python -B v3_thresholds.py > v3_thresholds.txt"""
from vcommon import *


def feats(x):
    """Per step s = 1..9: F = (M+, S+, T+ of x_0..x_(s-1), x_(s-1)) and the three ages at step s-1. Shapes (B*9, .)."""
    vals, ages = prefix_before(x)
    F = np.stack([vals[0][:, 1:], vals[1][:, 1:], vals[2][:, 1:], x[:, :-1]], -1).reshape(-1, 4)
    return F, [a[:, 1:].reshape(-1) for a in ages]


LEVELS = {"L1": (lambda a: a[0], [0, 1, 3]), "L2": (lambda a: a[0] * 10 + a[1], [0, 1, 2, 3]), "L3": (lambda a: a[0] * 100 + a[1] * 10 + a[2], [0, 1, 2, 3]),
          "M only, by age": (lambda a: a[0], [0, 3]), "M and S, one mix for all ages": (lambda a: (a[0] == 0).astype(int), [0, 1, 3])}


def fit(cfg, F, Y):
    o = np.argsort(cfg, kind="stable"); cs = cfg[o]
    return {int(c): np.linalg.lstsq(F[r], Y[r], rcond=None)[0] for c in np.unique(cs)
            for r in [o[np.searchsorted(cs, c): np.searchsorted(cs, c, "right")]]}


def predict(coef, cfg, F, nout=16):
    out = np.zeros((len(cfg), nout)); o = np.argsort(cfg, kind="stable"); cs = cfg[o]
    for c, b in coef.items():
        r = o[np.searchsorted(cs, c): np.searchsorted(cs, c, "right")]
        out[r] = F[r] @ b
    return out


class Data:
    def __init__(self, x):
        self.x, self.B = x, len(x)
        self.H, self.Z, self.Amp = amplitudes(x)
        self.Y = (self.Z[:, 1:] - x[:, 1:, None] * w).reshape(-1, 16)          # drive (W h_(s-1)) at steps 1..9
        self.F, self.ages = feats(x)

    def amp_from_drive(self, Yhat):
        Zh = self.Z.copy(); Zh[:, 1:] = self.x[:, 1:, None] * w + Yhat.reshape(self.B, 9, 16)
        A_ = np.zeros_like(self.Amp); A_[:, :, 0] = self.x
        A_[:, 1:, 1:4] = np.maximum(Zh[:, :-1][:, :, PULSE], 0); A_[:, :, 4:] = np.maximum(-Zh[:, :, ORD], 0)
        return A_

    def changed(self, A_model, chs, t=9, G=G, Pm=None):
        """Share of answers changed on quiet states at step t (read out with U) when the channels chs are modelled."""
        st = top3(self.x, t); q = st["quiet"]; h = self.H[q, t]
        A_ = self.Amp[q].copy(); A_[:, :, chs] = A_model[q][:, :, chs]
        new = rebuild(A_, t, G=G)
        if Pm is None:
            hm = with_tape(h, new)
        else:                                               # replace the part of h[o] in the modes of Pm instead of the tape
            hm = h.copy(); hm[:, ORD] = h[:, ORD] @ (np.eye(13) - Pm).T + new
        return np.mean(answer(hm) != answer(h)), q.sum()


tr, te = Data(sample(300_000, 7303)), Data(sample(200_000, 7304))
del tr.H, tr.Amp                                            # not needed for fitting
st = top3(te.x, 9); q = st["quiet"]; Mq = st["M"][q, None]; h9 = te.H[:, 9]
whole = rebuild(te.Amp[q], 9) / Mq @ Uo[:8].T
print(f"train 300000 (seed 7303), test 200000 (seed 7304); quiet final states in the test set n = {q.sum()}; accuracy there {np.mean(answer(h9)[q] == second_argmax(te.x)[q]):.4f}")
TH = {4: -1 / w[4], 1: -1 / w[1], 6: -1 / w[6], 7: -1 / w[7], 8: -1 / w[8]}
Mrun = np.maximum(te.F[:, 0], 1e-12); pos = te.F[:, 0] > 0.3
ALL = list(range(1, NCH)); GROUPS = {"ALL": ALL, "pulse 4": [1], "pulse 5": [2], "pulse 9": [3], "comparators": CCH, "wave clips": WCH}
print("    spread to be explained: std of theta_u / running max over steps with running max > 0.3: " +
      "  ".join(f"theta_{u} {((te.Y[:, u] * TH[u]) / Mrun)[pos].std():.3f}" for u in TH))

coefs, models = {}, {}
for lev, (cf, cols) in LEVELS.items():
    coefs[lev] = fit(cf(tr.ages), tr.F[:, cols], tr.Y)
    Yhat = predict(coefs[lev], cf(te.ages), te.F[:, cols]); models[lev] = te.amp_from_drive(Yhat)
    print(f"\n=== {lev}: {len(coefs[lev])} configurations ===")
    print("    rms threshold error / running max:  " + "  ".join(f"theta_{u} {np.sqrt(np.mean((((Yhat - te.Y)[:, u] * TH[u]) / Mrun)[pos] ** 2)):.3f}" for u in TH))
    row = "    answers changed (quiet final states), by modelled channels:  "
    for g, chs in GROUPS.items():
        row += f"{g} {te.changed(models[lev], chs)[0]:.4f}  "
    A_ = te.amp_from_drive(Yhat)[q]; s = rebuild(A_, 9) / Mq @ Uo[:8].T
    print(row + f"| R2 of the scores of ages >= 2 (all modelled) {1 - ((s - whole) ** 2).sum() / ((whole - whole.mean(0)) ** 2).sum():.3f}")

print("\n=== L1 threshold weights by the age a of the earlier maximum (a = 1..8): alpha (on M+), beta (on S+), alpha + beta ===")
for u in (4, 7, 1, 6, 8):
    al = [coefs["L1"][a][0, u] * TH[u] for a in range(1, 9)]; be = [coefs["L1"][a][1, u] * TH[u] for a in range(1, 9)]
    print(f"    unit {u}: alpha " + " ".join(f"{v:5.2f}" for v in al) + "   beta " + " ".join(f"{v:5.2f}" for v in be) + f"   alpha+beta {min(np.add(al, be)):.2f} .. {max(np.add(al, be)):.2f}")

print("\n=== L3 by the runner-up's lead over the third, (S - T)/M, quiet final states: answers changed ===")
A_ = te.Amp[q].copy(); A_[:, :, ALL] = models["L3"][q][:, :, ALL]
chg = answer(with_tape(h9[q], rebuild(A_, 9))) != answer(h9[q]); gap = ((st["S"] - st["T"]) / st["M"])[q]; aft = (st["pS"] > st["pM"])[q]
print("    " + "  ".join(f"[{lo:.2f},{hi:.2f}) {chg[(gap >= lo) & (gap < hi)].mean():.3f} (n = {((gap >= lo) & (gap < hi)).sum()})" for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 9))))
print(f"    runner-up after the maximum {chg[aft].mean():.4f}, before {chg[~aft].mean():.4f};  share of all changed answers that have lead < 0.05: {np.mean(gap[chg] < 0.05):.3f}")

print("\n=== (B) WHERE THE THRESHOLDS COME FROM: tape part of h_(s-1) in the drive of each unit (test set, steps 1..9) ===")
Hprev = te.H[:, :-1].reshape(-1, 16)
dt = (Hprev[:, ORD] @ P.T) @ W[:, ORD].T
print("    unit:                 " + " ".join(f"{u:6d}" for u in range(16)))
print("    std ratio:            " + " ".join(f"{v:6.2f}" for v in dt.std(0) / te.Y.std(0)))
print("    covariance share:     " + " ".join(f"{v:6.3f}" for v in ((dt - dt.mean(0)) * (te.Y - te.Y.mean(0))).sum(0) / ((te.Y - te.Y.mean(0)) ** 2).sum(0)))
print("    FUNCTIONAL: amplitudes recomputed with the tape part of h_(s-1) left out of the drive (everything else real), tape rebuilt; answers changed")
A_no = te.amp_from_drive(te.Y - dt)
tests = {"pulse 4": [1], "pulse 5": [2], "pulse 9": [3], "comparators 6, 7": [4 + ORD.index(6), 4 + ORD.index(7)], "comparators 1, 8": [4 + ORD.index(1), 4 + ORD.index(8)],
         "pulses + all comparators": [1, 2, 3] + CCH, "wave clips": WCH, "wave clips of 3, 15, 13, 14": [4 + ORD.index(u) for u in (3, 15, 13, 14)]}
print("      " + "   ".join(f"{k}: {te.changed(A_no, chs)[0]:.4f}" for k, chs in tests.items()))
p5 = te.Amp[:, :, 2]; p5n = A_no[:, :, 2]
print(f"      pulse of unit 5 without the tape's part of its drive: rms change / rms of the pulse = {np.sqrt(((p5n - p5) ** 2).mean() / (p5 ** 2).mean()):.3f}; "
      f"pulses switched on or off: {np.mean((p5n > 0) != (p5 > 0)):.4f} of steps")

print("\n=== (C) CONTROL: the L3 description applied to the NON-tape modes of the 13 non-pulse units (real tape and real pulse units kept) ===")
GR = kernel(PR)
for lev in ("L1", "L2", "L3"):
    c_t = te.changed(models[lev], ALL)[0]; c_r = te.changed(models[lev], ALL, G=GR, Pm=PR)[0]; c_b = te.changed(models[lev], ALL, G=kernel(np.eye(13)), Pm=np.eye(13))[0]
    print(f"    {lev}: answers changed with the modelled TAPE {c_t:.4f};  with the modelled REST modes {c_r:.4f};  with both (all 13 non-pulse units modelled) {c_b:.4f}")
A_ = te.Amp[q].copy(); A_[:, :, ALL] = models["L3"][q][:, :, ALL]
for nm, Gm, Pm in (("tape", G, P), ("rest", GR, PR)):
    real = h9[q][:, ORD] @ Pm.T / Mq @ Uo[:8].T; mod = rebuild(A_, 9, G=Gm) / Mq @ Uo[:8].T
    print(f"    L3, {nm} piece of the scores of ages >= 2: R2 {1 - ((mod - real) ** 2).sum() / ((real - real.mean(0)) ** 2).sum():.3f};  rms error {np.sqrt(((mod - real) ** 2).mean()):.2f} M;  std of the piece {real.std():.2f} M")

print("\n=== (D) OTHER POPULATIONS, L3 with all channels modelled: answers changed ===")
for t in (5, 7):
    c, n_ = te.changed(models["L3"], ALL, t=t)
    print(f"    N(0,1), quiet states at t = {t} read out with U (n = {n_}): {c:.4f}")
un = Data(sample(200_000, 7305, "uniform"))
for lev in ("L2", "L3"):
    cf, cols = LEVELS[lev]
    c, n_ = un.changed(un.amp_from_drive(predict(coefs[lev], cf(un.ages), un.F[:, cols])), ALL)
    print(f"    Unif(-1,1) inputs, {lev} coefficients fitted on N(0,1), quiet final states (n = {n_}): {c:.4f}")

print("\n=== UNIT 5 EVENT FORMULA: pre-activation one step after x_j ~ (x_j, Mp, Sp, Tp, x_(j+1)), per kind of x_j and age of Mp ===")
def u5(d):
    vals, ages = prefix_before(d.x); kind = np.minimum(arrival_kind(d.x, vals), 3)
    F = np.stack([d.x[:, :9], vals[0][:, :9], vals[1][:, :9], vals[2][:, :9], d.x[:, 1:]], -1).reshape(-1, 5)
    return F, d.Z[:, 1:, 5].reshape(-1, 1), (kind[:, :9] * 10 + ages[0][:, :9]).reshape(-1), kind[:, :9].reshape(-1)
Ftr, ytr, ctr, _ = u5(tr); Fte, yte, cte, kte = u5(te)
c5 = fit(ctr, Ftr, ytr); pred = predict(c5, cte, Fte, 1)[:, 0]; real = np.maximum(yte[:, 0], 0); mod = np.maximum(pred, 0)
for k, nm in ((3, "positive, below the runner-up"), (2, "a new runner-up"), (1, "a new maximum")):
    ks = [c5[k * 10 + a][0, 0] for a in range(8) if k * 10 + a in c5]
    print(f"    x_j {nm:30s} slope in x_j by age of Mp (0..7): " + " ".join(f"{v:5.1f}" for v in ks) + f"   fires {np.mean(real[kte == k] > 0):.3f}")
thr = [(-c5[30 + a][1:4, 0] / c5[30 + a][0, 0]) for a in range(8)]
print("    below the runner-up, implied threshold weights on (Mp, Sp, Tp) by age 0..7: " + "  ".join("(" + " ".join(f"{v:.2f}" for v in t_) + ")" for t_ in thr))
Mfin = np.repeat(st["M"], 9); qq = np.repeat(q, 9)
print(f"    rms error of the modelled pulse / final M_t on quiet sequences {np.sqrt(np.mean(((mod - real) / Mfin)[qq] ** 2)):.3f};  rms of the real pulse {np.sqrt(np.mean((real / Mfin)[qq] ** 2)):.3f};  R2 {1 - ((mod - real) ** 2).sum() / ((real - real.mean()) ** 2).sum():.4f}")
def test5(p5_):
    A_ = te.Amp.copy(); A_[:, 2:, 2] = p5_.reshape(te.B, 9)[:, :8]          # h_(j+1)[5] is written into the tape at step j+2
    return te.changed(A_, [2])[0]
assert np.abs(te.Amp[:, 2:, 2] - real.reshape(te.B, 9)[:, :8]).max() < 1e-12
print(f"    FUNCTIONAL (only unit 5's pulses replaced): formula for every pulse {test5(mod):.4f};  only after new maxima {test5(np.where(kte == 1, mod, real)):.4f};  "
      f"only after new runners-up {test5(np.where(kte == 2, mod, real)):.4f};  only after inputs below the runner-up {test5(np.where(kte == 3, mod, real)):.4f}")
print(f"    pulses removed: after new maxima {test5(np.where(kte == 1, 0, real)):.4f};  after new runners-up {test5(np.where(kte == 2, 0, real)):.4f};  after positive inputs below the runner-up {test5(np.where(kte == 3, 0, real)):.4f}")
