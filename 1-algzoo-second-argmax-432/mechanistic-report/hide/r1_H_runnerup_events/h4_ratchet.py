"""H4 illustration: Theta5 (the level an input must exceed to fire unit 5 one step later) along controlled sequences.
Theta5(t) is computed from the state h_{t-1} only (units 5, 9 at step t evaluated for x_t = 0, unit 4 excluded)."""
from hcommon import *

c5 = W[5, ORD] @ w[ORD]


def theta5(x):
    H = run(x)[0]
    Hprev = np.vstack([np.zeros(N), H[:-1]])
    pre0 = Hprev @ W.T
    pstate = -(Hprev[:, PULSE] @ W[np.ix_(ORD, PULSE)].T) @ W[5, ORD] - W[5, 5] * np.maximum(pre0[:, 5], 0) - W[5, 9] * np.maximum(pre0[:, 9], 0)
    th = (-(Hprev[:, ORD] @ A.T) @ W[5, ORD] + pstate) / c5
    return th, H, coords(Hprev)


def show(x, label):
    th, H, m = theta5(x)
    print(f"\n{label}\n   x        : " + " ".join(f"{v:6.2f}" for v in x))
    print("   Theta5(t): " + " ".join(f"{v:6.2f}" for v in th))
    print("   h_t[5]   : " + " ".join(f"{v:6.2f}" for v in H[:, 5]))
    print("   0.01265*m_runner(t-1): " + " ".join(f"{v:6.2f}" for v in 0.01265 * m["runner"].real) + "\n   -0.02872*m_level(t-1): " + " ".join(f"{v:6.2f}" for v in -0.02872 * m["level"].real))


for b in (-0.3, -1.0):
    x = np.full(10, b); x[0] = 1.5
    show(x, f"max 1.5 at 0 only, background {b}")
    x = np.full(10, b); x[0] = 1.5; x[2] = 0.6
    show(x, f"max 1.5 at 0, S = 0.6 at 2, background {b}   (0.75 S = 0.45)")
    x = np.full(10, b); x[0] = 1.5; x[2] = 0.6; x[5] = 0.54
    show(x, f"max 1.5 at 0, S = 0.6 at 2, T = 0.54 at 5 (class L), background {b}")
    x = np.full(10, b); x[0] = 1.5; x[6] = 0.6
    show(x, f"max 1.5 at 0, S = 0.6 at 6 (max age 6: no comparator clipped), background {b}")

# measured update of Theta5 three steps after an event, on N(0,1) data: Theta5(t+3) - Theta5(t) ~ g (x_t - Theta5(t))_+ for quiet followers
x = sample(400_000, 1035)
e = Ev(x)
pre0 = e.Hprev @ W.T
pstate = -(e.Hprev[:, :, PULSE] @ W[np.ix_(ORD, PULSE)].T) @ W[5, ORD] - W[5, 5] * np.maximum(pre0[:, :, 5], 0) - W[5, 9] * np.maximum(pre0[:, :, 9], 0)
Th5 = (-(e.Hprev[:, :, ORD] @ A.T) @ W[5, ORD] + pstate) / c5
print("\nmeasured ratchet on N(0,1) (seed 1035): steps t in 2..6 with S_old > 0, x_t < M_old, unit 4 silent, and x_{t+1}, x_{t+2} < 0")
print("  regression  Theta5(t+3) - Theta5(t) ~ g * (x_t - Theta5(t))_+   by class of x_t and age of the max")
for lab, cond in (("N event (S < x_t < M)", lambda t: (x[:, t] > e.So[:, t])), ("L event (Theta5 < x_t < S)", lambda t: (x[:, t] < e.So[:, t]) & (x[:, t] > Th5[:, t]))):
    print(f"  {lab}")
    for a in range(1, 7):
        ys, xs_ = [], []
        for t in range(2, 7):
            s = (e.So[:, t] > 0) & (x[:, t] < e.Mo[:, t]) & ~e.G[:, t, 4] & (x[:, t + 1] < 0) & (x[:, t + 2] < 0) & (e.a1[:, t] == a) & cond(t)
            ys.append(Th5[s, t + 3] - Th5[s, t]); xs_.append(np.maximum(x[s, t] - Th5[s, t], 0))
        ys, xs_ = np.concatenate(ys), np.concatenate(xs_)
        if len(ys) < 300: continue
        g = (xs_ @ ys) / (xs_ @ xs_)
        print(f"     a={a}: g = {g:.3f}   R2 = {1 - np.sum((ys - g * xs_) ** 2) / np.sum((ys - ys.mean()) ** 2):.3f}   n = {len(ys)}")
print("  [EXACT] pure unit-5 record: g = c5 * (g_runner/c5) * LEFT_runner.W[ORD,5] * lam_runner^0 ... runner part only =", round(18.0085 * 0.01265 * 1.723, 3))
