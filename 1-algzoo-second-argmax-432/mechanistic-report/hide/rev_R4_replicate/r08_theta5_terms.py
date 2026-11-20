"""R4 / r08: what the '+ small' in  Theta_5 = 0.01265 m_runner - 0.02872 m_level + small  contains,
and the arithmetic of the clock's first step after a new maximum (section 5.4).
Seed 7091.  Output: r08_theta5_terms_out.txt
"""
from rlib import *
import modes

out = open(__file__.replace(".py", "_out.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s); out.write(s + "\n"); out.flush()


A = W[np.ix_(ORD, ORD)]
c5 = W[5, ORD] @ w[ORD]
x = gauss(400_000, 7091)
H, Z = fwd(x)
st = prefix_stats(x)
Mp, Sp, pMp = prev(st["M"], -np.inf), prev(st["S"], -np.inf), prev(st["pM"], -1)
terms = {k: [] for k in modes.NAMES + ["pulses(t-1)", "pulses(t)", "theta", "M", "S", "age"]}
for t in range(2, 9):
    ok = (Sp[:, t] > 0) & (Z[:, t, 7] > 0) & (Z[:, t, 8] > 0)       # the formula's condition, runner-up positive
    hm = H[ok, t - 1]
    m = modes.coords(hm)
    tot = 0
    for n in modes.NAMES:
        R = modes.RIGHT[n]
        coef = -(W[5, ORD] @ (A @ R)) / c5
        v = 2 * np.real(m[n] * coef) if modes.IS_COMPLEX[n] else m[n] * coef.real
        terms[n].append(v); tot = tot + v
    p1 = -(hm[:, PULSE] @ (W[5, ORD] @ W[np.ix_(ORD, PULSE)])) / c5      # pulses of step t-1 through the ordinary units
    p2 = -(H[ok, t][:, PULSE] @ W[5, PULSE]) / c5                         # pulses of step t read directly
    th = -((hm @ W.T)[:, ORD] @ W[5, ORD] + H[ok, t][:, PULSE] @ W[5, PULSE]) / c5
    assert np.abs(tot + p1 + p2 - th).max() < 1e-10
    terms["pulses(t-1)"].append(p1); terms["pulses(t)"].append(p2); terms["theta"].append(th)
    terms["M"].append(Mp[ok, t]); terms["S"].append(Sp[ok, t]); terms["age"].append((t - pMp[:, t])[ok])
T = {k: np.concatenate(v) for k, v in terms.items()}
P("Theta_5 = sum of mode terms + pulse terms (exact, checked to 1e-10); steps t = 2..8, S > 0, comparators 7, 8 on; n = %d" % len(T["theta"]))
P("term            mean/M    median |term|/M   90% |term|/M")
for k in modes.NAMES + ["pulses(t-1)", "pulses(t)"]:
    r = T[k] / T["M"]
    P("  %-12s  %+.3f     %.3f            %.3f" % (k, r.mean(), np.median(np.abs(r)), np.percentile(np.abs(r), 90)))
rest = T["theta"] - T["level"] - T["runner"]
P("level + runner: mean/M %+.3f ; Theta_5 mean/M %+.3f ; remainder mean/M %+.3f, mean/S %+.3f, median |.|/M %.3f, 90%% %.3f" % (
    ((T["level"] + T["runner"]) / T["M"]).mean(), (T["theta"] / T["M"]).mean(), (rest / T["M"]).mean(), np.median(rest / T["S"]),
    np.median(np.abs(rest) / T["M"]), np.percentile(np.abs(rest) / T["M"], 90)))
P("share of var(Theta_5) explained by level + runner alone: %.3f ; by age of the max 1..8: %s" % (
    1 - rest.var() / T["theta"].var() if False else 1 - ((rest - 0) ** 2).sum() / ((T["theta"] - T["theta"].mean()) ** 2).sum(),
    np.round([1 - (rest[T["age"] == a] ** 2).sum() / ((T["theta"][T["age"] == a] - T["theta"][T["age"] == a].mean()) ** 2).sum() for a in range(1, 9)], 2)))
P("median remainder / S by age of the max 1..8: %s" % np.round([np.median((rest / T["S"])[T["age"] == a]) for a in range(1, 9)], 3))
P("median Theta_5 / S by age of the max 1..8:   %s" % np.round([np.median((T["theta"] / T["S"])[T["age"] == a]) for a in range(1, 9)], 3))

P("\nclock's first step after a new maximum (from the weights)")
Lc = modes.LEFT["clock"]; lam = modes.LAM["clock"]
c5c, c9c = Lc @ W[ORD, 5], Lc @ W[ORD, 9]
f = lambda z: "%.3f@%.1f" % (abs(z), np.degrees(np.angle(z)))
P("lambda_clock = %s ; write of unit 5 per unit h5: %s ; of unit 9 per unit h9: %s ; W[5,4] = %.4f, W[9,4] = %.4f" % (f(lam), f(c5c), f(c9c), W[5, 4], W[9, 4]))
tot = lam + c5c * W[5, 4] + c9c * W[9, 4]
P("lambda + c5 W[5,4] + c9 W[9,4] = %s + %s + %s = %s" % (f(lam), f(c5c * W[5, 4]), f(c9c * W[9, 4]), f(tot)))
P("level coordinate: lambda_level = %.4f ; unit 5's write %.4f per unit h5, unit 9's %.4f per unit h9 ; x W[5,4], W[9,4]: %.4f, %.4f ; sum %.4f" % (
    modes.LAM["level"].real, (modes.LEFT["level"] @ W[ORD, 5]).real, (modes.LEFT["level"] @ W[ORD, 9]).real,
    (modes.LEFT["level"] @ W[ORD, 5]).real * W[5, 4], (modes.LEFT["level"] @ W[ORD, 9]).real * W[9, 4],
    modes.LAM["level"].real + (modes.LEFT["level"] @ W[ORD, 5]).real * W[5, 4] + (modes.LEFT["level"] @ W[ORD, 9]).real * W[9, 4]))
out.close()
