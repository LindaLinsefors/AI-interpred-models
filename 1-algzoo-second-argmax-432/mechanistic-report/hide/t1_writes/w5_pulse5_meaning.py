"""W5. What unit 5's pulse (the largest and most sensitive write) means, as an event formula.
The pulse that answers input x_j is h_(j+1)[5] = [a_(j+1)[5]]_+ .  Fit its pre-activation, separately for each kind of x_j at
arrival and each age a of the previous maximum Mp (age at step j-1), as
    a_(j+1)[5] = k_x x_j + k_M Mp + k_S Sp + k_T Tp + k_n x_(j+1)
Mp, Sp, Tp = the three largest of x_0..x_(j-1), clipped at 0.  Then put [fit]_+ in place of the real pulse (all other writes
real), rebuild the tape with the exact kernel and count changed answers on quiet final states.
Run:  python -B w5_pulse5_meaning.py > w5_pulse5_meaning.txt"""
from wcommon import *

def data(x):
    H, Z, Amp = amplitudes(x)
    kind, Mp, Sp, st = roles(x)
    Tp = np.zeros_like(x); Tp[:, 1:] = np.maximum(st["M3"][:, :-1], 0)
    aM = np.full(x.shape, 9); aM[:, 1:] = np.where(Mp[:, 1:] > 0, st["a1"][:, :-1], 9)       # age of Mp at step j-1 (9 = none)
    j = slice(0, 9)                                                                           # inputs x_0..x_8, pulses at steps 1..9
    F = np.stack([x[:, j], Mp[:, j], Sp[:, j], Tp[:, j], x[:, 1:]], -1).reshape(-1, 5)
    return H, Amp, F, Z[:, 1:, 5].reshape(-1), (np.minimum(kind[:, j], 3) * 10 + aM[:, j]).reshape(-1), kind[:, j].reshape(-1)

xtr, xte = sample(200_000, 505), sample(200_000, 304)
_, _, Ftr, ytr, ctr, _ = data(xtr)
H, Amp, Fte, yte, cte, kte = data(xte)
KIND = {0: "x_j not positive", 1: "x_j a new maximum", 2: "x_j a new runner-up", 3: "x_j positive, below the runner-up"}
pred = np.zeros_like(yte); coef = {}
for c in np.unique(ctr):
    b = np.linalg.lstsq(Ftr[ctr == c], ytr[ctr == c], rcond=None)[0]; coef[int(c)] = b
    pred[cte == c] = Fte[cte == c] @ b
print("Coefficients of the pre-activation of unit 5 one step after x_j, by kind of x_j and by the age a of the previous maximum (9 = no positive earlier input)")
print("R2 and rms error are for the pulse [.]_+ on the test set; rms in units of max(x_j, Mp); 'fires' = share with a pulse")
scale = np.maximum(np.maximum(Fte[:, 0], Fte[:, 1]), 1e-9)
for k, nm in KIND.items():
    print(f"\n  {nm}")
    print("     a      n     fires     k_x     k_M     k_S     k_T     k_n   | R2 pulse  rms/scale | implied threshold for x_j: (k_M Mp + k_S Sp + k_T Tp) / -k_x")
    for a in list(range(9)) + [9]:
        c = k * 10 + a; sel = cte == c
        if c not in coef or sel.sum() < 200: continue
        b = coef[c]; yt, yp = np.maximum(yte[sel], 0), np.maximum(pred[sel], 0)
        ok = sel & (scale > 0.3)
        thr = f"{-b[1] / b[0]:5.2f} Mp {-b[2] / b[0]:+5.2f} Sp {-b[3] / b[0]:+5.2f} Tp" if k else ""
        print(f"    {a:2d} {sel.sum():7d}  {np.mean(yt > 0):6.3f}  " + " ".join(f"{v:7.2f}" for v in b) +
              f"   | {(1 - ((yt - yp) ** 2).sum() / max(((yt - yt.mean()) ** 2).sum(), 1e-12)):7.3f}   {np.sqrt(np.mean(((np.maximum(yte, 0) - np.maximum(pred, 0)) / scale)[ok] ** 2)) if ok.any() else 0:6.3f}   | {thr}")

st = prefix_stats(xte, 9); q = st["quiet"]; h9 = H[:, 9]; ans0 = answer(h9); corr = second_argmax(xte)
def test(label, p5):
    A = Amp.copy(); A[:, 2:, 2] = p5.reshape(len(xte), 9)[:, :8]          # the pulse at step j+1 is written at step j+2
    a = answer(set_tape(h9, rebuild(A, 9)))
    print(f"    {label:62s} changed {np.mean(a[q] != ans0[q]):.4f}   acc {np.mean(a[q] == corr[q]):.4f}")
print(f"\nFUNCTIONAL TEST, quiet final states n = {q.sum()} (baseline accuracy {np.mean(ans0[q] == corr[q]):.4f}): only unit 5's pulses replaced")
real = np.maximum(yte, 0)
test("event formula for every pulse", np.maximum(pred, 0))
for k, nm in KIND.items():
    test(f"event formula only after: {nm}", np.where(kte == k, np.maximum(pred, 0), real) if k < 3 else np.where(kte >= 3, np.maximum(pred, 0), real))
test("pulses after inputs below the runner-up removed", np.where(kte >= 3, 0, real))
test("pulses after new runners-up removed", np.where(kte == 2, 0, real))
test("pulses after new maxima removed", np.where(kte == 1, 0, real))
err = (np.maximum(pred, 0) - real) / np.repeat(st["M"], 9)
print(f"    rms error of the modelled pulse in units of the final M_t (quiet sequences): {np.sqrt(np.mean(err.reshape(-1, 9)[q] ** 2)):.3f};  rms of the real pulse: {np.sqrt(np.mean((real / np.repeat(st['M'], 9)).reshape(-1, 9)[q] ** 2)):.3f}")
assert np.abs(Amp[:, 2:, 2] - real.reshape(len(xte), 9)[:, :8]).max() < 1e-12
print("    tape score at the index of input x_j per unit of the pulse that answered it, when x_j is 2..9 steps old: " +
      " ".join(f"{(R_SCORE @ K[n, 2])[7 - n]:.2f}" for n in range(0, 8)))
print("    the same at the index of the input that arrived one step BEFORE x_j (3..9 steps old): " +
      " ".join(f"{(R_SCORE @ K[n, 2])[6 - n]:.2f}" for n in range(0, 7)))
