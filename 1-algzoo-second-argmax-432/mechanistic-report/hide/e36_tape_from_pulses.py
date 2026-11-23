"""E36. Is the tape piece (eq. 4a of tape.md) a fixed linear blur of what the pulse units wrote?
Model, one set of coefficients for all states:
    s_tape[9 - a] = sum_b K5[a, b] u_b + K4[a, b] d_b + K9[a, b] v_b
u_b = pulse of unit 5 one step after the input of age b, d_b = pulse of unit 4 at that input, v_b = pulse of unit 9 after it.
Then: what are the pulses as functions of the inputs?
Run:  python -B hide/e36_tape_from_pulses.py > hide/outputs/e36_tape_from_pulses.txt"""
from common import *
from modes import RIGHT, coords

B = 1_000_000
x = sample(B, 3636)
H = run(x); h = H[:, 9]; st = running_stats(x)
c = coords(h); Uo = U[:, ORD]
G = np.real(np.multiply.outer(c["alt"], Uo @ RIGHT["alt"]))
for n in ("short", "h2", "h3"):
    G = G + 2 * np.real(np.multiply.outer(c[n], Uo @ RIGHT[n]))
order = np.argsort(-x, 1); pM, pS = order[:, 0], order[:, 1]; ar = np.arange(B)
M, S = x[ar, pM], x[ar, pS]
quiet = (pM <= 7) & (pS <= 7) & (M > 0.3) & (S > 0.1)
r2 = lambda y, yh: 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()

# regressors by age b of the input: u_b = H[:, 9-b+1, 5] (b >= 1), d_b = H[:, 9-b, 4] (b >= 0), v_b = H[:, 9-b+1, 9] (b >= 1)
u = {b: H[:, 10 - b, 5] for b in range(1, 10)}; dd = {b: H[:, 9 - b, 4] for b in range(0, 10)}; v = {b: H[:, 10 - b, 9] for b in range(1, 10)}
def design(use):
    cols, names = [], []
    for nm, dct in (("u", u), ("d", dd), ("v", v)):
        if nm in use:
            for b, col in dct.items():
                cols.append(col); names.append((nm, b))
    return np.stack(cols, 1), names
print("1. R2 of the tape entry of age a from the pulses (one coefficient per pulse and age, no intercept), ages a = 2..9")
for use in ("u", "ud", "udv"):
    X, names = design(use)
    XtX = X.T @ X
    row_all, row_q = [], []
    for a in range(2, 10):
        y = G[:, 9 - a]; b = np.linalg.solve(XtX, X.T @ y); yh = X @ b
        row_all.append(r2(y, yh)); row_q.append(r2(y[quiet], yh[quiet]))
    print(f"   regressors {use:4s}: all states " + " ".join(f"{q:.3f}" for q in row_all) + "   | quiet states " + " ".join(f"{q:.3f}" for q in row_q))
X, names = design("udv"); XtX = X.T @ X
K = np.array([np.linalg.solve(XtX, X.T @ G[:, 9 - a]) for a in range(2, 10)])            # (8 ages, regressors)
for nm, title in (("u", "K5: logits in the entry of age a per unit of unit 5's pulse written for the input of age b"),
                  ("d", "K4: per unit of unit 4's pulse at the input of age b"), ("v", "K9: per unit of unit 9's pulse after the input of age b")):
    js = [k for k, (m, b) in enumerate(names) if m == nm]
    print(f"\n2. {title}\n   rows a = 2..9, columns b = " + " ".join(f"{names[k][1]:6d}" for k in js))
    for a in range(2, 10):
        print(f"   a = {a}:      " + " ".join(f"{K[a - 2, k]:6.2f}" for k in js))
print("\n   K5 along the diagonals (b - a = n): mean and range over a")
js = {names[k][1]: k for k in range(len(names)) if names[k][0] == "u"}
for n_ in range(-4, 5):
    vals = [K[a - 2, js[a + n_]] for a in range(2, 10) if (a + n_) in js]
    print(f"     n = {n_:+d}: mean {np.mean(vals):6.2f}, range {min(vals):6.2f} to {max(vals):6.2f}  ({len(vals)} entries)")

print("\n3. What unit 5's pulse is: u = h_(j+1)[5] after input x_j (j = 1..8), by what x_j was. Mp, Sp = max and runner-up of the earlier inputs, clipped at 0")
xj = x[:, 1:9]; uj = H[:, 2:10, 5]
Mp, Sp = np.maximum(st["M1"][:, 0:8], 0), np.maximum(st["M2"][:, 0:8], 0)
pos = Mp > 0.3
for name, q in (("new maximum (x > Mp)", pos & (xj > Mp)), ("new runner-up (Sp < x < Mp)", pos & (xj > Sp) & (xj < Mp)),
                ("below the runner-up, pulse fired", pos & (xj < Sp) & (uj > 0))):
    Xf = np.stack([xj[q], Mp[q], Sp[q]], 1); b = np.linalg.lstsq(Xf, uj[q], rcond=None)[0]
    print(f"   {name:34s} n = {q.sum():8d}  fired {np.mean(uj[q] > 0):.3f}   u = {b[0]:6.2f} x {b[1]:+6.2f} Mp {b[2]:+6.2f} Sp   R2 = {r2(uj[q], Xf @ b):.3f}")
q = pos & (xj < Sp) & (Sp > 0.1)
print("   share of inputs below the runner-up that fire unit 5, by x / Sp:")
for lo, hi in ((-9, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)):
    s_ = q & (xj / Sp > lo) & (xj / Sp <= hi)
    print(f"     x/Sp in ({lo:4.1f}, {hi:3.1f}]: fired {np.mean(uj[s_] > 0):.3f},  median u / Sp among fired {np.median((uj / Sp)[s_ & (uj > 0)]) if (s_ & (uj > 0)).sum() else float('nan'):6.2f}")
